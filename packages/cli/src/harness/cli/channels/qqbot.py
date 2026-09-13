"""QQ Bot native Gateway and Ed25519 webhook transports, sharing durable routes."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote, urlsplit

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from harness.cli.channels.media import validate_media_url
from harness.cli.channels.transports import ChannelError
from harness.cli.channels.webhooks import AuthenticationError, WebhookTransport
from harness.core.gateway_channels import ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment

API = "https://api.sgroup.qq.com"
TOKEN_URL = "https://bots.qq.com/app/getAppAccessToken"


class QQBotTransport(WebhookTransport):
    name = "qqbot"
    limit = 4000

    async def authenticate(self) -> None:
        if not re.fullmatch(r"[0-9]+", self.config.app_id) or not self.token:
            raise ChannelError("QQ Bot requires app_id and client secret")
        if self.config.receive_mode not in {"websocket", "webhook"}:
            raise ChannelError("QQ Bot receive_mode must be websocket or webhook")
        self.identity = self.config.app_id
        self.token_lock = asyncio.Lock()
        self.access_token, self.token_expires = "", 0.0
        self.store: ChannelStore | None = None
        raw = self.token.encode()
        seed = (raw * ((32 + len(raw) - 1) // len(raw)))[:32]
        self.signing_key = Ed25519PrivateKey.from_private_bytes(seed)
        await self.access()

    async def access(self) -> str:
        async with self.token_lock:
            if time.time() < self.token_expires - 60:
                return self.access_token
            payload = await self.api(
                "POST", TOKEN_URL, data={"appId": self.config.app_id, "clientSecret": self.token}
            )
            if not payload.get("access_token"):
                raise ChannelError("QQ Bot credential exchange failed")
            self.access_token = str(payload["access_token"])
            self.secrets.append(self.access_token)
            self.token_expires = time.time() + min(
                7200, max(1, int(payload.get("expires_in", 7200)))
            )
            return self.access_token

    async def call(
        self, method: str, path: str, data: dict[str, Any] | None = None, *, files: Any = None
    ) -> Any:
        return await self.api(
            method, API + path, data=data, files=files, auth="QQBot " + await self.access()
        )

    def _thread(self, thread_id: str) -> tuple[str, str]:
        app, kind, target = json.loads(thread_id)
        if (
            app != self.config.app_id
            or kind not in {"c2c", "group", "guild", "dm"}
            or not isinstance(target, str)
            or not target
            or target in {".", ".."}
        ):
            raise AuthenticationError("QQ Bot conversation is outside the authenticated app")
        return kind, target

    def channel_id(self, thread_id: str) -> str:
        kind, target = self._thread(thread_id)
        return f"{self.config.app_id}:{kind}:{target}"

    async def ingest(self, event: dict[str, Any], store: ChannelStore) -> None:
        data, event_type = event.get("d", {}), event.get("t")
        author = data.get("author", {})
        if author.get("bot"):
            return
        if event_type == "C2C_MESSAGE_CREATE":
            kind, sender = "c2c", author.get("user_openid")
            target, group, mentioned = sender, False, False
        elif event_type == "GROUP_AT_MESSAGE_CREATE":
            kind, sender, target = "group", author.get("member_openid"), data.get("group_openid")
            group, mentioned = True, True
        elif event_type in {
            "AT_MESSAGE_CREATE",
            "GUILD_AT_MESSAGE_CREATE",
            "MESSAGE_CREATE",
            "GUILD_MESSAGE_CREATE",
        }:
            kind, sender, target = "guild", author.get("id"), data.get("channel_id")
            group, mentioned = True, event_type in {"AT_MESSAGE_CREATE", "GUILD_AT_MESSAGE_CREATE"}
        elif event_type == "DIRECT_MESSAGE_CREATE":
            kind, sender, target = "dm", author.get("id"), data.get("guild_id")
            group, mentioned = False, False
        else:
            return
        if not isinstance(sender, str) or not sender or not isinstance(target, str) or not target:
            raise ValueError("QQ Bot message omitted its sender or destination")
        user_kind = "guild" if kind == "dm" else kind
        user = f"{self.config.app_id}:{user_kind}:{sender}"
        channel = f"{self.config.app_id}:{kind}:{target}"
        if not self.config.permits(
            user_id=user, channel_id=channel, group=group, mentioned=mentioned
        ):
            return
        text = str(data.get("content", ""))
        if self.bot_id:
            text = text.replace(f"<@!{self.bot_id}>", "").replace(f"<@{self.bot_id}>", "").strip()
        attachments = [
            {
                "url": item["url"],
                "mime_type": item.get("content_type", "application/octet-stream"),
                "name": item.get("filename", "attachment"),
            }
            for item in data.get("attachments", [])[:16]
            if item.get("url")
        ]
        if not text and not attachments:
            return
        mid = data["id"]
        if not isinstance(mid, str) or not mid or len(mid) > 1024:
            raise ValueError("QQ Bot requires a message ID")
        timestamp = datetime.fromisoformat(
            str(data["timestamp"]).replace("Z", "+00:00")
        ).timestamp()
        thread = json.dumps([self.config.app_id, kind, target], separators=(",", ":"))
        previous = store.get("reply:" + thread, {})
        if timestamp >= previous.get("timestamp", 0):
            store.set(
                "reply:" + thread,
                {
                    "message_id": mid,
                    "timestamp": timestamp,
                    "expires": min(timestamp + 300, time.time() + 300),
                },
            )
        store.ingest(
            ChannelMessage(
                id=json.dumps([self.config.app_id, kind, target, mid]),
                user_id=user,
                channel_id=channel,
                thread_id=thread,
                text=text or "Please inspect the attached media.",
                group=group,
                mentioned=mentioned,
                attachments=attachments,
            )
        )

    async def handle_event(
        self, headers: Mapping[str, str], body: bytes, store: ChannelStore
    ) -> dict[str, Any]:
        payload = json.loads(body)
        if payload.get("op") == 13:
            plain, timestamp = payload["d"]["plain_token"], payload["d"]["event_ts"]
            if (
                not isinstance(plain, str)
                or not isinstance(timestamp, str)
                or len(plain) > 4096
                or len(timestamp) > 64
            ):
                raise ValueError("QQ Bot validation challenge is invalid")
            return {
                "plain_token": plain,
                "signature": self.signing_key.sign((timestamp + plain).encode()).hex(),
            }
        normalized = {key.lower(): value for key, value in headers.items()}
        try:
            timestamp = normalized["x-signature-timestamp"]
            if abs(time.time() - int(timestamp)) > 300:
                raise ValueError("stale callback")
            signature = bytes.fromhex(normalized["x-signature-ed25519"])
            self.signing_key.public_key().verify(signature, timestamp.encode() + body)
        except (InvalidSignature, ValueError, KeyError, TypeError):
            raise AuthenticationError("QQ Bot webhook signature is invalid or expired") from None
        if payload.get("op") == 0:
            await self.ingest(payload, store)
        return {"op": 12}

    def can_send(self) -> bool:
        return getattr(self, "store", None) is not None

    async def receive(self, store: ChannelStore) -> None:
        self.store = store
        try:
            if self.config.receive_mode == "webhook":
                await super().receive(store)
            else:
                await self.gateway(store)
        finally:
            self.store = None

    async def gateway(self, store: ChannelStore) -> None:
        url = (await self.call("GET", "/gateway"))["url"]
        resume = store.get("resume", {})
        sequence = resume.get("sequence")
        async with self.socket(url, domains=("sgroup.qq.com",)) as socket:
            hello = json.loads(await asyncio.wait_for(socket.recv(), 30))
            if hello.get("op") != 10:
                raise ChannelError("QQ Bot gateway omitted Hello")
            interval = float(hello["d"]["heartbeat_interval"]) / 1000
            if not 0.05 <= interval <= 120:
                raise ChannelError("QQ Bot gateway heartbeat interval is invalid")
            token = "QQBot " + await self.access()
            if resume.get("session_id") and sequence is not None:
                await socket.send(
                    json.dumps(
                        {
                            "op": 6,
                            "d": {
                                "token": token,
                                "session_id": resume["session_id"],
                                "seq": sequence,
                            },
                        }
                    )
                )
            else:
                await socket.send(
                    json.dumps(
                        {
                            "op": 2,
                            "d": {
                                "token": token,
                                "intents": (1 << 25) | (1 << 30) | (1 << 12),
                                "shard": [0, 1],
                                "properties": {"$browser": "harness"},
                            },
                        }
                    )
                )
            acked = True

            async def heartbeat():
                nonlocal acked
                while True:
                    await asyncio.sleep(interval * 0.8)
                    if not acked:
                        await socket.close()
                        return
                    acked = False
                    await socket.send(json.dumps({"op": 1, "d": sequence}))

            task = asyncio.create_task(heartbeat())
            try:
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
                        return
                    elif opcode == 0:
                        if event.get("t") == "READY":
                            resume = {"session_id": event["d"]["session_id"]}
                            self.bot_id = str(event["d"].get("user", {}).get("id", ""))
                            store.set("bot_id", self.bot_id)
                        else:
                            self.bot_id = str(store.get("bot_id", ""))
                            await self.ingest(event, store)
                        sequence = event.get("s", sequence)
                        if resume:
                            store.set("resume", {**resume, "sequence": sequence})
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def reply(self, thread: str, delivery: str) -> dict[str, Any]:
        self._thread(thread)
        if self.store is None:
            raise ChannelError("QQ Bot reply store is not open")
        previous = self.store.get("delivery:" + delivery)
        if previous:
            if previous["thread"] != thread:
                raise ChannelError("QQ Bot delivery belongs to another conversation")
            route = previous
        else:
            route = self.store.get("reply:" + thread)
            if not route:
                raise ChannelError("QQ Bot requires a prior incoming message for replies")
            count_key = "sequence:" + thread + ":" + route["message_id"]
            sequence = self.store.get(count_key, 0) + 1
            self.store.set(count_key, sequence)
            route = {**route, "thread": thread, "msg_seq": sequence}
            self.store.set("delivery:" + delivery, route)
        if time.time() >= route["expires"]:
            raise ChannelError("QQ Bot reply window expired; handle a new inbound message")
        return {"msg_id": route["message_id"], "msg_seq": route["msg_seq"]}

    def path(self, thread: str, suffix: str = "messages") -> str:
        kind, target = self._thread(thread)
        escaped = quote(target, safe="")
        return (
            f"/v2/{'users' if kind == 'c2c' else 'groups'}/{escaped}/{suffix}"
            if kind in {"c2c", "group"}
            else f"/{'channels' if kind == 'guild' else 'dms'}/{escaped}/{suffix}"
        )

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        kind, _ = self._thread(thread_id)
        data = {"content": text, **self.reply(thread_id, delivery_id)}
        if kind in {"c2c", "group"}:
            data["msg_type"] = 0
        else:
            data.pop("msg_seq")
        result = await self.call("POST", self.path(thread_id), data)
        if not result.get("id"):
            raise ChannelError("QQ Bot did not acknowledge the sent message")

    async def prepare_media(self, message: ChannelMessage) -> list[MediaAttachment]:
        result, total = [], 0
        for item in message.attachments[:16]:
            url = str(item["url"])
            if url.startswith("//"):
                url = "https:" + url
            validate_media_url(url, ("qpic.cn", "multimedia.nt.qq.com", "api.sgroup.qq.com"))
            self.secrets.append(url)
            raw = bytearray()
            headers = (
                {"Authorization": "QQBot " + await self.access()}
                if urlsplit(url).hostname == "api.sgroup.qq.com"
                else {}
            )
            async with self.client.stream(
                "GET", url, headers=headers, follow_redirects=False
            ) as response:
                if not response.is_success:
                    raise ChannelError("QQ Bot media download rejected")
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > 20 * 1024 * 1024:
                        raise ChannelError("QQ Bot media exceeds 20 MiB")
                    raw.extend(chunk)
            mime = str(item["mime_type"])
            if mime == "file":
                mime = mimetypes.guess_type(str(item["name"]))[0] or "application/octet-stream"
            elif mime == "voice":
                mime = "audio/silk"
            result.append(
                MediaAttachment(
                    kind="image"
                    if mime.startswith("image/")
                    else "audio"
                    if mime.startswith("audio/")
                    else "file",
                    mime_type=mime,
                    data=base64.b64encode(raw).decode(),
                    name=PurePosixPath(str(item["name"])).name[:255],
                )
            )
        return result

    async def send_media(
        self, thread_id: str, attachment: MediaAttachment, delivery_id: str
    ) -> None:
        kind, _ = self._thread(thread_id)
        reply = self.reply(thread_id, delivery_id)
        if attachment.data is None:
            raise ChannelError("QQ Bot uploads require inline bytes")
        raw = base64.b64decode(attachment.data, validate=True)
        if len(raw) > 20 * 1024 * 1024:
            raise ChannelError("QQ Bot upload exceeds 20 MiB")
        name = PurePosixPath(attachment.name or "attachment").name
        if kind in {"c2c", "group"}:
            file_type = (
                1
                if attachment.kind == "image"
                else 3
                if attachment.kind == "audio"
                else 2
                if attachment.mime_type.startswith("video/")
                else 4
            )
            media = await self.call(
                "POST",
                self.path(thread_id, "files"),
                {
                    "file_type": file_type,
                    "file_data": attachment.data,
                    "file_name": name,
                    "srv_send_msg": False,
                },
            )
            if not media.get("file_info"):
                raise ChannelError("QQ Bot media upload was not acknowledged")
            result = await self.call(
                "POST",
                self.path(thread_id),
                {"msg_type": 7, "media": {"file_info": media["file_info"]}, **reply},
            )
        else:
            if attachment.kind != "image":
                raise ChannelError("QQ guild/DM native uploads currently support images only")
            result = await self.call(
                "POST",
                self.path(thread_id),
                {"msg_id": reply["msg_id"]},
                files={"file_image": (name, raw, attachment.mime_type)},
            )
        if not result.get("id"):
            raise ChannelError("QQ Bot did not acknowledge the media message")
