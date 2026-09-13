"""Authenticated BlueBubbles webhook/REST bridge with owned registrations."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import ipaddress
import json
import re
from collections.abc import Mapping
from contextlib import suppress
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

from aiohttp import web

from harness.cli.channels.transports import ChannelError
from harness.cli.channels.webhooks import AuthenticationError, WebhookTransport
from harness.core.gateway_channels import ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment


def trusted_endpoint(value: str) -> str:
    parsed = urlsplit(value)
    local = parsed.hostname == "localhost"
    with suppress(ValueError):
        local = local or ipaddress.ip_address(parsed.hostname or "").is_loopback
    if (
        not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or (parsed.scheme != "https" and not (parsed.scheme == "http" and local))
    ):
        raise ChannelError("BlueBubbles requires HTTPS or an HTTP loopback endpoint")
    return value.rstrip("/")


def chat_guid(value: Any) -> str:
    if not isinstance(value, str) or len(value) > 1024 or any(ord(c) < 32 for c in value):
        raise ValueError("BlueBubbles requires an exact chat GUID")
    parts = value.split(";", 2)
    if len(parts) != 3 or not parts[0] or parts[1] not in {"+", "-"} or not parts[2]:
        raise ValueError("BlueBubbles requires an exact chat GUID")
    return value


class BlueBubblesTransport(WebhookTransport):
    name = "bluebubbles"
    limit = 4000

    @property
    def base(self) -> str:
        return trusted_endpoint(self.config.homeserver)

    def url(self, path: str) -> str:
        return (
            self.base + path + ("&" if "?" in path else "?") + urlencode({"password": self.token})
        )

    async def call(
        self, method: str, path: str, data: dict[str, Any] | None = None, *, files: Any = None
    ) -> Any:
        result = await self.api(method, self.url(path), data=data, files=files)
        if not isinstance(result, dict) or not 200 <= result.get("status", 0) < 300:
            raise ChannelError("BlueBubbles rejected the request")
        return result.get("data")

    async def authenticate(self) -> None:
        if not self.token or not self.app_token:
            raise ChannelError(
                "BlueBubbles requires server password and separate webhook app_token_env"
            )
        if self.config.allow_groups and self.config.require_mention and not self.config.username:
            raise ChannelError("BlueBubbles group mentions require an explicit username alias")
        # Query credentials are encoded on the wire; redact both representations.
        self.secrets.extend(
            [
                quote(self.token, safe=""),
                quote(self.app_token, safe=""),
                urlencode({"password": self.token}).partition("=")[2],
            ]
        )
        info = await self.call("GET", "/api/v1/server/info")
        if (
            not isinstance(info, dict)
            or not info.get("computer_id")
            or not info.get("detected_imessage")
        ):
            raise ChannelError("BlueBubbles must report its computer and active iMessage account")
        self.identity = json.dumps(
            [self.base, info["computer_id"], info["detected_imessage"]], sort_keys=True
        )
        self.bot_id = str(info["detected_imessage"])
        if self.config.webhook_url:
            self.callback_url()  # Fail configuration before starting the receiver.

    def callback_url(self) -> str:
        callback = trusted_endpoint(self.config.webhook_url)
        if urlsplit(callback).path != self.config.webhook_path:
            raise ChannelError("BlueBubbles webhook_url path must match webhook_path")
        url = callback + "?" + urlencode({"token": self.app_token})
        self.secrets.append(url)
        return url

    async def handle_request(
        self, request: web.Request, body: bytes, store: ChannelStore
    ) -> dict[str, Any]:
        tokens = request.query.getall("token", [])
        if len(tokens) != 1:
            raise AuthenticationError("BlueBubbles requires one webhook token")
        return await self.handle_event({"X-BlueBubbles-Token": tokens[0]}, body, store)

    async def handle_event(
        self, headers: Mapping[str, str], body: bytes, store: ChannelStore
    ) -> dict[str, Any]:
        if not self.app_token or not hmac.compare_digest(
            headers.get("X-BlueBubbles-Token", "").encode(), self.app_token.encode()
        ):
            raise AuthenticationError("BlueBubbles webhook token does not match")
        payload = json.loads(body)
        # Updates include delivery/read receipts; never execute them as new user work.
        if payload.get("type") != "new-message":
            return {}
        record = payload["data"]
        associated = record.get("associatedMessageType")
        if record.get("isFromMe") or (
            isinstance(associated, int)
            and (2000 <= associated <= 2005 or 3000 <= associated <= 3005)
        ):
            return {}
        sender = record.get("handle", {}).get("address")
        chats = record.get("chats", [])
        if not isinstance(sender, str) or not sender or len(chats) != 1:
            raise ValueError("BlueBubbles message must identify one sender and one chat")
        conversation = chat_guid(chats[0]["guid"])
        group = conversation.split(";", 2)[1] == "+"
        if not group and conversation.split(";", 2)[2].casefold() != sender.casefold():
            raise ValueError("BlueBubbles direct conversation does not match its sender")
        text = str(record.get("text") or "")
        alias = self.config.username
        mentioned = bool(alias and re.search(r"(?<!\w)@" + re.escape(alias) + r"(?!\w)", text))
        attachments = [
            {
                "guid": item["guid"],
                "mime_type": item.get("mimeType", "application/octet-stream"),
                "name": item.get("transferName", "attachment"),
            }
            for item in record.get("attachments", [])[:16]
            if item.get("guid")
        ]
        if not text and not attachments:
            return {}
        message_id = record["guid"]
        if not isinstance(message_id, str) or not message_id or len(message_id) > 512:
            raise ValueError("BlueBubbles requires a message GUID")
        self.accept(
            ChannelMessage(
                id=message_id,
                user_id=sender,
                channel_id=conversation,
                thread_id=conversation,
                text=text or "Please inspect the attached media.",
                group=group,
                mentioned=mentioned,
                attachments=attachments,
            ),
            store,
        )
        return {}

    async def register(self, store: ChannelStore) -> str | None:
        url = self.callback_url()
        fingerprint = hashlib.sha256(url.encode()).hexdigest()
        registered = await self.call("GET", "/api/v1/webhook")
        previous = store.get("webhook_registration", {})
        for item in registered:
            if item.get("url") == url:
                if "new-message" not in item.get("events", []):
                    raise ChannelError(
                        "Existing BlueBubbles webhook does not subscribe to new-message"
                    )
                return (
                    str(item["id"])
                    if previous == {"id": str(item["id"]), "url_hash": fingerprint}
                    else None
                )
        created = await self.call(
            "POST", "/api/v1/webhook", {"url": url, "events": ["new-message"]}
        )
        key = str(created["id"])
        try:
            store.set("webhook_registration", {"id": key, "url_hash": fingerprint})
        except BaseException:
            await self.call("DELETE", "/api/v1/webhook/" + quote(key, safe=""))
            raise
        return key

    async def receive(self, store: ChannelStore) -> None:
        runner = web.AppRunner(self.application(store), access_log=None, shutdown_timeout=5)
        await runner.setup()
        owned = None
        try:
            await web.TCPSite(runner, self.config.listen_host, self.config.listen_port).start()
            if self.config.webhook_url:
                registration = asyncio.create_task(self.register(store))
                try:
                    owned = await asyncio.shield(registration)
                except asyncio.CancelledError:
                    # A late POST response can contain a newly owned webhook.
                    # Recover its ID before closing the server and deleting it.
                    with suppress(Exception):
                        owned = await registration
                    raise
            store.set("connection", "listening")
            await asyncio.Event().wait()
        finally:
            try:
                if owned is not None:
                    await self.call("DELETE", "/api/v1/webhook/" + quote(owned, safe=""))
                    store.set("webhook_registration", {})
            finally:
                await runner.cleanup()

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        result = await self.call(
            "POST",
            "/api/v1/message/text",
            {
                "chatGuid": chat_guid(thread_id),
                "tempGuid": "harness-" + delivery_id,
                "message": text,
            },
        )
        if not isinstance(result, dict) or not result.get("guid"):
            raise ChannelError("BlueBubbles did not acknowledge the sent message")

    async def prepare_media(self, message: ChannelMessage) -> list[MediaAttachment]:
        result = []
        total = 0
        for item in message.attachments[:16]:
            guid = str(item["guid"])
            if not re.fullmatch(r"[A-Za-z0-9._-]{1,512}", guid) or guid in {".", ".."}:
                raise ValueError("BlueBubbles attachment requires a GUID")
            raw = bytearray()
            async with self.client.stream(
                "GET",
                self.url("/api/v1/attachment/" + quote(guid, safe="") + "/download"),
                follow_redirects=False,
            ) as response:
                if not response.is_success:
                    raise ChannelError("BlueBubbles media download rejected")
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > 20 * 1024 * 1024:
                        raise ChannelError("BlueBubbles attachments exceed 20 MiB")
                    raw.extend(chunk)
            mime = str(item["mime_type"])
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
        conversation = chat_guid(thread_id)
        if attachment.data is None:
            raise ChannelError("BlueBubbles uploads require inline bytes")
        raw = base64.b64decode(attachment.data, validate=True)
        if len(raw) > 20 * 1024 * 1024:
            raise ChannelError("BlueBubbles upload exceeds 20 MiB")
        name = PurePosixPath(attachment.name or "attachment").name
        result = await self.call(
            "POST",
            "/api/v1/message/attachment",
            {
                "chatGuid": conversation,
                "tempGuid": "harness-" + delivery_id,
                "name": name,
            },
            files={"attachment": (name, raw, attachment.mime_type)},
        )
        if not isinstance(result, dict) or not result.get("guid"):
            raise ChannelError("BlueBubbles did not acknowledge the uploaded attachment")
