from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import httpx
from websockets.asyncio.client import connect

from harness.cli.channels.media import media_descriptors, prepare_media, send_media
from harness.core.gateway_channels import ChannelConfig, ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment


class ChannelError(RuntimeError):
    """Safe operator-facing error; never include credential-bearing request URLs."""


class RateLimited(ChannelError):
    def __init__(self, delay: float):
        self.delay = max(1, min(float(delay), 86400))
        super().__init__("Channel rate limited; delivery remains queued")


class DeliveryDeferred(ChannelError):
    """One delivery awaits a platform prerequisite; other sends may proceed."""

    def __init__(self, message: str, delay: float = 15):
        self.delay = max(1, min(float(delay), 86400))
        super().__init__(message)


class DeliveryRejected(ChannelError):
    """A recipient declined, or a platform prerequisite expired definitively."""


class SecretFilter(logging.Filter):
    def __init__(self, secrets: list[str]):
        super().__init__()
        self.secrets = secrets

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for secret in self.secrets:
            if secret:
                message = message.replace(secret, "[REDACTED]")
        record.msg, record.args = message, ()
        return True


class Transport:
    name = ""
    limit = 2000

    def __init__(
        self,
        *,
        config: ChannelConfig,
        token: str,
        app_token: str = "",
        client: httpx.AsyncClient | None = None,
        socket_connect: Any = None,
    ):
        self.config, self.token, self.app_token = config, token, app_token
        self.client = client or httpx.AsyncClient(
            timeout=40, follow_redirects=False, trust_env=False
        )
        self.owns_client = client is None
        self.socket_connect = socket_connect or connect
        self.identity = ""
        self.bot_id = ""
        self.secrets = [token, app_token]
        self.log_filter = SecretFilter(self.secrets)
        self.socket_logger = logging.getLogger("harness.channels.websocket")
        self.socket_logger.addFilter(self.log_filter)
        logging.getLogger("httpx").addFilter(self.log_filter)

    async def close(self) -> None:
        if self.owns_client:
            await self.client.aclose()
        self.socket_logger.removeFilter(self.log_filter)
        logging.getLogger("httpx").removeFilter(self.log_filter)

    async def api(
        self,
        method: str,
        url: str,
        *,
        data: dict[str, Any] | None = None,
        auth: str = "",
        files: Any = None,
    ) -> Any:
        body: dict[str, Any] = {"json": data} if files is None else {"data": data, "files": files}
        try:
            response = await self.client.request(
                method,
                url,
                **body,
                headers={"Authorization": auth} if auth else {},
            )
        except httpx.HTTPError:
            raise ChannelError(
                "Channel connection interrupted; delivery outcome may be uncertain"
            ) from None
        try:
            payload = response.json()
        except ValueError:
            raise ChannelError(
                f"Channel returned invalid JSON (HTTP {response.status_code})"
            ) from None
        if response.status_code == 429:
            delay = response.headers.get("Retry-After") or (
                payload.get(
                    "retry_after",
                    payload.get("parameters", {}).get(
                        "retry_after", float(payload.get("retry_after_ms", 1000)) / 1000
                    ),
                )
                if isinstance(payload, dict)
                else 1
            )
            raise RateLimited(float(delay))
        if not response.is_success:
            raise ChannelError(f"Channel API rejected request (HTTP {response.status_code})")
        return payload

    def accept(self, message: ChannelMessage | None, store: ChannelStore) -> None:
        if message:
            from harness.cli.channels.question_interactions import admit_text_answer

            message = admit_text_answer(store, message)
        if message:
            store.set("connection", "connected")
            store.set("connection_error", "")
        if message and self.config.permits(
            user_id=message.user_id,
            channel_id=message.channel_id,
            group=message.group,
            mentioned=message.mentioned,
        ):
            store.ingest(message)

    def socket(self, url: str, *, domains: tuple[str, ...]):
        parsed = urlsplit(url)
        if (
            parsed.scheme != "wss"
            or not parsed.hostname
            or not any(
                parsed.hostname == domain or parsed.hostname.endswith("." + domain)
                for domain in domains
            )
        ):
            raise ChannelError("Channel returned an untrusted WebSocket address")
        self.secrets.append(url)
        self.secrets.extend(value for _, value in parse_qsl(parsed.query) if value)
        return self.socket_connect(
            url, max_size=2**20, open_timeout=30, logger=self.socket_logger, proxy=None
        )

    async def authenticate(self) -> None:
        raise NotImplementedError

    def channel_id(self, thread_id: str) -> str:
        if self.name == "telegram":
            return thread_id.partition(":")[0]
        if self.name == "slack":
            return ":".join(thread_id.split(":")[:2])
        return thread_id

    def can_send(self) -> bool:
        return True

    async def prepare_media(self, message: ChannelMessage) -> list[MediaAttachment]:
        return await prepare_media(self, message)

    async def send_media(
        self, thread_id: str, attachment: MediaAttachment, delivery_id: str
    ) -> None:
        await send_media(self, thread_id, attachment, delivery_id)

    async def receive(self, store: ChannelStore) -> None:
        raise NotImplementedError

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        raise NotImplementedError

    async def send_approval(
        self, thread_id: str, text: str, delivery_id: str, buttons: dict[str, str]
    ) -> str:
        raise DeliveryRejected("This transport does not support native approval controls")

    async def send_question(
        self, thread_id: str, text: str, delivery_id: str, buttons: list[dict[str, str]]
    ) -> str:
        raise DeliveryRejected("This transport does not support native question controls")


class TelegramTransport(Transport):
    name = "telegram"
    limit = 4096

    async def call(self, method: str, data: dict[str, Any] | None = None) -> Any:
        payload = await self.api(
            "POST", f"https://api.telegram.org/bot{self.token}/{method}", data=data or {}
        )
        if not payload.get("ok"):
            if payload.get("error_code") == 429:
                raise RateLimited(payload.get("parameters", {}).get("retry_after", 1))
            raise ChannelError("Telegram rejected the request")
        return payload.get("result")

    async def authenticate(self) -> None:
        user = await self.call("getMe")
        self.bot_id = self.identity = str(user["id"])
        self.username = str(user.get("username", ""))
        webhook = await self.call("getWebhookInfo")
        if webhook.get("url"):
            raise ChannelError(
                "Telegram has a webhook configured; remove it before starting polling"
            )

    def parse(self, update: dict[str, Any]) -> ChannelMessage | None:
        message = update.get("message", {})
        sender, chat = message.get("from", {}), message.get("chat", {})
        if not sender.get("id") or sender.get("is_bot") or not chat.get("id"):
            return None
        text = str(message.get("text") or message.get("caption") or "")
        attachments = media_descriptors(self.name, message)
        if not text and not attachments:
            return None
        mention = f"@{self.username}" if self.username else ""
        reply_bot = (
            str(message.get("reply_to_message", {}).get("from", {}).get("id", "")) == self.bot_id
        )
        mentioned = bool(mention and mention.lower() in text.lower()) or reply_bot
        if mention:
            import re

            text = re.sub(re.escape(mention), "", text, flags=re.IGNORECASE).strip()
        chat_id = str(chat["id"])
        topic = str(message.get("message_thread_id") or "")
        return ChannelMessage(
            id=str(update["update_id"]),
            user_id=str(sender["id"]),
            thread_id=f"{chat_id}:{topic}" if topic else chat_id,
            text=text or "Please inspect the attached media.",
            channel_id=chat_id,
            group=chat.get("type") != "private",
            mentioned=mentioned,
            attachments=attachments,
        )

    async def poll_once(self, store: ChannelStore) -> None:
        updates = await self.call(
            "getUpdates",
            {
                "offset": store.get("offset", 0),
                "timeout": 30,
                "allowed_updates": ["message", "callback_query"],
            },
        )
        for update in updates:
            if update.get("callback_query"):
                await self.approval_callback(update["callback_query"], store)
            else:
                self.accept(self.parse(update), store)
            # Persist accepted messages before advancing the upstream acknowledgment.
            store.set("offset", int(update["update_id"]) + 1)
        store.set("connection", "connected")

    async def receive(self, store: ChannelStore) -> None:
        while True:
            await self.poll_once(store)

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        chat, _, topic = thread_id.partition(":")
        data: dict[str, Any] = {
            "chat_id": chat,
            "text": text,
            "link_preview_options": {"is_disabled": True},
        }
        if topic:
            data["message_thread_id"] = int(topic)
        await self.call("sendMessage", data)

    async def send_approval(
        self, thread_id: str, text: str, delivery_id: str, buttons: dict[str, str]
    ) -> str:
        chat, _, topic = thread_id.partition(":")
        result = await self.call(
            "sendMessage",
            {
                "chat_id": chat,
                "text": text,
                **({"message_thread_id": int(topic)} if topic else {}),
                "reply_markup": {
                    "inline_keyboard": [
                        [
                            {"text": "Approve", "callback_data": buttons["approve"]},
                            {"text": "Deny", "callback_data": buttons["deny"]},
                        ]
                    ]
                },
            },
        )
        return str(result.get("message_id") or "")

    async def send_question(
        self, thread_id: str, text: str, delivery_id: str, buttons: list[dict[str, str]]
    ) -> str:
        from harness.cli.channels.question_interactions import display_text

        chat, _, topic = thread_id.partition(":")
        result = await self.call(
            "sendMessage",
            {
                "chat_id": chat,
                "text": display_text(text, 4096),
                **({"message_thread_id": int(topic)} if topic else {}),
                "reply_markup": {
                    "inline_keyboard": [
                        [{"text": display_text(item["label"], 60), "callback_data": item["token"]}]
                        for item in buttons
                    ]
                },
            },
        )
        return str(result.get("message_id") or "")

    async def approval_callback(self, event: dict[str, Any], store: ChannelStore) -> None:
        from harness.cli.channels.question_interactions import display_text, interaction_result

        message, sender = event.get("message", {}), event.get("from", {})
        chat = str(message.get("chat", {}).get("id") or "")
        topic = str(message.get("message_thread_id") or "")
        result_text = "Unavailable, expired, or already used."
        if str(message.get("from", {}).get("id")) == self.bot_id and not sender.get("is_bot"):
            result_text = await interaction_result(
                self,
                store,
                token=event.get("data", ""),
                user_id=str(sender.get("id") or ""),
                channel_id=chat,
                thread_id=f"{chat}:{topic}" if topic else chat,
                native_message_id=str(message.get("message_id") or ""),
                event_id=str(event.get("id") or ""),
            )
        if event.get("id"):
            await self.call(
                "answerCallbackQuery",
                {
                    "callback_query_id": event["id"],
                    "text": display_text(result_text, 190),
                    "show_alert": result_text.startswith("Unavailable"),
                },
            )


class DiscordTransport(Transport):
    name = "discord"
    limit = 2000

    async def call(self, method: str, path: str, data: dict[str, Any] | None = None) -> Any:
        return await self.api(
            method, f"https://discord.com/api/v10/{path}", data=data, auth=f"Bot {self.token}"
        )

    async def authenticate(self) -> None:
        user = await self.call("GET", "users/@me")
        self.bot_id = self.identity = str(user["id"])

    def parse(self, event: dict[str, Any]) -> ChannelMessage | None:
        sender = event.get("author", {})
        if (
            sender.get("bot")
            or not sender.get("id")
            or not event.get("channel_id")
            or event.get("webhook_id")
        ):
            return None
        text = str(event.get("content") or "")
        attachments = media_descriptors(self.name, event)
        if not text and not attachments:
            return None
        mentioned = any(str(user.get("id")) == self.bot_id for user in event.get("mentions", []))
        text = text.replace(f"<@{self.bot_id}>", "").replace(f"<@!{self.bot_id}>", "").strip()
        return ChannelMessage(
            id=str(event["id"]),
            user_id=str(sender["id"]),
            thread_id=str(event["channel_id"]),
            channel_id=str(event["channel_id"]),
            text=text or "Please inspect the attached media.",
            group=bool(event.get("guild_id")),
            mentioned=mentioned,
            attachments=attachments,
        )

    async def receive(self, store: ChannelStore) -> None:
        resume = store.get("resume", {})
        url = str(resume.get("url") or (await self.call("GET", "gateway/bot"))["url"])
        async with self.socket(
            url.rstrip("/") + "/?v=10&encoding=json", domains=("discord.gg", "discord.com")
        ) as socket:
            hello = json.loads(await socket.recv())
            if hello.get("op") != 10:
                raise ChannelError("Discord did not send Gateway Hello")
            interval = float(hello["d"]["heartbeat_interval"]) / 1000
            sequence = resume.get("sequence")
            acked = True

            async def heartbeat() -> None:
                nonlocal acked
                await asyncio.sleep(interval * random.random())
                while True:
                    if not acked:
                        await socket.close()
                        return
                    acked = False
                    await socket.send(json.dumps({"op": 1, "d": sequence}))
                    await asyncio.sleep(interval)

            heartbeat_task = asyncio.create_task(heartbeat())
            try:
                if resume.get("session_id") and sequence is not None:
                    await socket.send(
                        json.dumps(
                            {
                                "op": 6,
                                "d": {
                                    "token": self.token,
                                    "session_id": resume["session_id"],
                                    "seq": sequence,
                                },
                            }
                        )
                    )
                else:
                    # Message content outside DMs/mentions needs the privileged intent.
                    intents = (1 << 0) | (1 << 9) | (1 << 12)
                    if self.config.allow_groups and not self.config.require_mention:
                        intents |= 1 << 15
                    await socket.send(
                        json.dumps(
                            {
                                "op": 2,
                                "d": {
                                    "token": self.token,
                                    "intents": intents,
                                    "properties": {
                                        "os": "linux",
                                        "browser": "harness",
                                        "device": "harness",
                                    },
                                },
                            }
                        )
                    )
                async for raw in socket:
                    event = json.loads(raw)
                    opcode = event.get("op")
                    if opcode == 11:
                        acked = True
                    elif opcode == 1:
                        await socket.send(json.dumps({"op": 1, "d": sequence}))
                    elif opcode == 7:
                        return
                    elif opcode == 9:
                        if not event.get("d"):
                            store.set("resume", {})
                        await asyncio.sleep(5)
                        return
                    elif opcode == 0:
                        if event.get("t") == "READY":
                            store.set("connection", "connected")
                            resume = {
                                "session_id": event["d"]["session_id"],
                                "url": event["d"]["resume_gateway_url"],
                            }
                        elif event.get("t") == "MESSAGE_CREATE":
                            self.accept(self.parse(event["d"]), store)
                        elif event.get("t") == "INTERACTION_CREATE":
                            await self.approval_callback(event["d"], store)
                        sequence = event.get("s", sequence)
                        if resume:
                            store.set("resume", {**resume, "sequence": sequence})
            finally:
                heartbeat_task.cancel()
                await asyncio.gather(heartbeat_task, return_exceptions=True)

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        if not thread_id.isdigit():
            raise ChannelError("Invalid Discord conversation ID")
        await self.call(
            "POST",
            f"channels/{thread_id}/messages",
            {
                "content": text,
                "allowed_mentions": {"parse": []},
                "nonce": delivery_id[:24],
                "enforce_nonce": True,
            },
        )

    async def send_approval(
        self, thread_id: str, text: str, delivery_id: str, buttons: dict[str, str]
    ) -> str:
        if not thread_id.isdigit():
            raise ChannelError("Invalid Discord conversation ID")
        response = await self.call(
            "POST",
            f"channels/{thread_id}/messages",
            {
                "content": text,
                "allowed_mentions": {"parse": []},
                "nonce": delivery_id[:24],
                "enforce_nonce": True,
                "components": [
                    {
                        "type": 1,
                        "components": [
                            {
                                "type": 2,
                                "style": 3,
                                "label": "Approve",
                                "custom_id": buttons["approve"],
                            },
                            {"type": 2, "style": 4, "label": "Deny", "custom_id": buttons["deny"]},
                        ],
                    }
                ],
            },
        )
        return str(response.get("id") or "")

    async def send_question(
        self, thread_id: str, text: str, delivery_id: str, buttons: list[dict[str, str]]
    ) -> str:
        from harness.cli.channels.question_interactions import display_text

        if not thread_id.isdigit():
            raise ChannelError("Invalid Discord conversation ID")
        result = await self.call(
            "POST",
            f"channels/{thread_id}/messages",
            {
                "content": display_text(text, 2000),
                "allowed_mentions": {"parse": []},
                "nonce": delivery_id[:24],
                "enforce_nonce": True,
                "components": [
                    {
                        "type": 1,
                        "components": [
                            {
                                "type": 2,
                                "style": 1,
                                "label": display_text(item["label"], 80),
                                "custom_id": item["token"],
                            }
                            for item in buttons[start : start + 5]
                        ],
                    }
                    for start in range(0, len(buttons), 5)
                ],
            },
        )
        return str(result.get("id") or "")

    async def approval_callback(self, event: dict[str, Any], store: ChannelStore) -> None:
        from harness.cli.channels.question_interactions import display_text, interaction_result

        if event.get("type") != 3 or str(event.get("application_id")) != self.bot_id:
            return
        actor = event.get("member", {}).get("user") or event.get("user", {})
        message = event.get("message", {})
        channel = str(event.get("channel_id") or "")
        result_text = "Unavailable, expired, or already used."
        if str(message.get("author", {}).get("id")) == self.bot_id and not actor.get("bot"):
            result_text = await interaction_result(
                self,
                store,
                token=event.get("data", {}).get("custom_id", ""),
                user_id=str(actor.get("id") or ""),
                channel_id=channel,
                thread_id=channel,
                native_message_id=str(message.get("id") or ""),
                event_id=str(event.get("id") or ""),
            )
        identifier, token = str(event.get("id") or ""), str(event.get("token") or "")
        if (
            not identifier.isdigit()
            or not token
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
                for c in token
            )
        ):
            raise ChannelError("Invalid Discord interaction receipt")
        self.secrets.append(token)
        response = await self.client.post(
            f"https://discord.com/api/v10/interactions/{identifier}/{token}/callback",
            json={
                "type": 4,
                "data": {
                    "flags": 64,
                    "content": display_text(result_text, 1900),
                    "allowed_mentions": {"parse": []},
                },
            },
        )
        if not response.is_success:
            raise ChannelError("Discord interaction acknowledgement failed")


class SlackTransport(Transport):
    name = "slack"
    limit = 4000

    async def call(
        self, method: str, data: dict[str, Any] | None = None, *, app: bool = False
    ) -> dict[str, Any]:
        payload = await self.api(
            "POST",
            f"https://slack.com/api/{method}",
            data=data or {},
            auth=f"Bearer {self.app_token if app else self.token}",
        )
        if not payload.get("ok"):
            if payload.get("error") == "ratelimited":
                raise RateLimited(payload.get("retry_after", 1))
            raise ChannelError(
                "Slack rejected the request; check token scopes and channel membership"
            )
        return payload

    async def authenticate(self) -> None:
        if not self.app_token:
            raise ChannelError("Slack Socket Mode requires an app-level token environment variable")
        auth = await self.call("auth.test")
        self.team_id = str(auth["team_id"])
        self.bot_id = str(auth["user_id"])
        self.identity = f"{self.team_id}:{self.bot_id}"

    def parse(self, payload: dict[str, Any]) -> ChannelMessage | None:
        if payload.get("team_id") != self.team_id:
            return None
        event = payload.get("event", {})
        if (
            event.get("type") not in {"message", "app_mention"}
            or event.get("subtype") not in {None, "file_share"}
            or event.get("bot_id")
            or not event.get("user")
            or event.get("user") == self.bot_id
        ):
            return None
        channel = str(event.get("channel") or "")
        text = str(event.get("text") or "")
        attachments = media_descriptors(self.name, event)
        if not channel or (not text and not attachments) or not event.get("ts"):
            return None
        group = event.get("channel_type") != "im" and not channel.startswith("D")
        mentioned = f"<@{self.bot_id}>" in text
        text = text.replace(f"<@{self.bot_id}>", "").strip()
        thread = str(event.get("thread_ts") or event["ts"]) if group else ""
        return ChannelMessage(
            id=f"{self.team_id}:{channel}:{event['ts']}",
            user_id=f"{self.team_id}:{event['user']}",
            thread_id=f"{self.team_id}:{channel}:{thread}",
            channel_id=f"{self.team_id}:{channel}",
            text=text or "Please inspect the attached media.",
            group=group,
            mentioned=mentioned,
            attachments=attachments,
        )

    async def receive(self, store: ChannelStore) -> None:
        connection = await self.call("apps.connections.open", app=True)
        async with self.socket(str(connection["url"]), domains=("slack.com",)) as socket:
            async for raw in socket:
                envelope = json.loads(raw)
                if envelope.get("type") == "hello":
                    store.set("connection", "connected")
                if envelope.get("type") == "disconnect":
                    return
                envelope_id = envelope.get("envelope_id")
                if not envelope_id:
                    continue
                response_text = ""
                if envelope.get("type") == "events_api":
                    self.accept(self.parse(envelope.get("payload", {})), store)
                elif envelope.get("type") == "interactive":
                    response_text = await self.approval_callback(envelope.get("payload", {}), store)
                # ACK after durable ingestion, before any potentially slow model work.
                await socket.send(json.dumps({"envelope_id": envelope_id}))
                if response_text:
                    from contextlib import suppress

                    from harness.cli.channels.question_interactions import display_text

                    payload = envelope.get("payload", {})
                    if payload.get("team", {}).get("id") == self.team_id:
                        with suppress(ChannelError):
                            await self.call(
                                "chat.postEphemeral",
                                {
                                    "channel": payload.get("channel", {}).get("id"),
                                    "user": payload.get("user", {}).get("id"),
                                    "text": display_text(response_text, 3000),
                                },
                            )

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        parts = thread_id.split(":", 2)
        if len(parts) != 3 or parts[0] != self.team_id:
            raise ChannelError("Slack conversation does not belong to the authenticated workspace")
        data = {
            "channel": parts[1],
            "text": text,
            "unfurl_links": False,
            "unfurl_media": False,
            "mrkdwn": False,
        }
        if parts[2]:
            data["thread_ts"] = parts[2]
        await self.call("chat.postMessage", data)

    async def send_approval(
        self, thread_id: str, text: str, delivery_id: str, buttons: dict[str, str]
    ) -> str:
        team, channel, root = thread_id.split(":", 2)
        if team != self.team_id:
            raise ChannelError("Slack conversation belongs to another workspace")
        result = await self.call(
            "chat.postMessage",
            {
                "channel": channel,
                "text": text,
                "mrkdwn": False,
                "unfurl_links": False,
                "unfurl_media": False,
                **({"thread_ts": root} if root else {}),
                "blocks": [
                    {"type": "section", "text": {"type": "plain_text", "text": text}},
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "style": "primary",
                                "text": {"type": "plain_text", "text": "Approve"},
                                "action_id": "harness_approve",
                                "value": buttons["approve"],
                            },
                            {
                                "type": "button",
                                "style": "danger",
                                "text": {"type": "plain_text", "text": "Deny"},
                                "action_id": "harness_deny",
                                "value": buttons["deny"],
                            },
                        ],
                    },
                ],
            },
        )
        return str(result.get("ts") or "")

    async def send_question(
        self, thread_id: str, text: str, delivery_id: str, buttons: list[dict[str, str]]
    ) -> str:
        from harness.cli.channels.question_interactions import display_text

        team, channel, root = thread_id.split(":", 2)
        if team != self.team_id:
            raise ChannelError("Slack conversation belongs to another workspace")
        elements = [
            {
                "type": "button",
                "text": {"type": "plain_text", "text": display_text(item["label"], 75)},
                "action_id": f"harness_question_{index}",
                "value": item["token"],
            }
            for index, item in enumerate(buttons)
        ]
        result = await self.call(
            "chat.postMessage",
            {
                "channel": channel,
                "text": display_text(text, 3000),
                "mrkdwn": False,
                **({"thread_ts": root} if root else {}),
                "blocks": [
                    {
                        "type": "section",
                        "text": {"type": "plain_text", "text": display_text(text, 3000)},
                    },
                    *[
                        {"type": "actions", "elements": elements[start : start + 5]}
                        for start in range(0, len(elements), 5)
                    ],
                ],
            },
        )
        return str(result.get("ts") or "")

    async def approval_callback(self, payload: dict[str, Any], store: ChannelStore) -> str:
        from harness.cli.channels.question_interactions import interaction_result

        if (
            payload.get("type") != "block_actions"
            or payload.get("team", {}).get("id") != self.team_id
        ):
            return "Unavailable, expired, or already used."
        message = payload.get("message", {})
        if message.get("user") != self.bot_id:
            return "Unavailable, expired, or already used."
        actions = payload.get("actions", [])
        if len(actions) != 1 or (
            actions[0].get("action_id")
            not in {
                "harness_approve",
                "harness_deny",
            }
            and actions[0].get("action_id") not in {f"harness_question_{i}" for i in range(6)}
        ):
            return "Unavailable, expired, or already used."
        channel = str(payload.get("channel", {}).get("id") or "")
        root = str(message.get("thread_ts") or "") if not channel.startswith("D") else ""
        return await interaction_result(
            self,
            store,
            token=actions[0].get("value", ""),
            user_id=self.team_id + ":" + str(payload.get("user", {}).get("id") or ""),
            channel_id=self.team_id + ":" + channel,
            thread_id=f"{self.team_id}:{channel}:{root}",
            native_message_id=str(payload.get("container", {}).get("message_ts") or ""),
            event_id=str(actions[0].get("action_ts") or ""),
        )


TRANSPORTS = {"telegram": TelegramTransport, "discord": DiscordTransport, "slack": SlackTransport}
