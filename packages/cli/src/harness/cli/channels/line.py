"""LINE Messaging API callbacks and idempotent push replies."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from collections.abc import Mapping
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from aiohttp import web

from harness.cli.channels.attachment_io import (
    MediaPublisher,
    attachment_bytes,
    download,
    from_bytes,
)
from harness.cli.channels.transports import ChannelError, RateLimited
from harness.cli.channels.webhooks import AuthenticationError, WebhookTransport
from harness.core.gateway_channels import ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment


class LINETransport(WebhookTransport):
    name = "line"
    limit = 5000

    async def authenticate(self) -> None:
        if not self.app_token:
            raise ChannelError("LINE requires its channel secret in app_token_env")
        account = await self.api(
            "GET", "https://api.line.me/v2/bot/info", auth="Bearer " + self.token
        )
        self.bot_id = self.identity = str(account["userId"])

    def channel_id(self, thread_id: str) -> str:
        kind, destination = json.loads(thread_id)
        if kind not in {"user", "group", "room"}:
            raise ChannelError("Invalid LINE conversation kind")
        return destination

    def application(self, store: ChannelStore) -> web.Application:
        app = super().application(store)
        if self.config.webhook_url:
            self.publisher = MediaPublisher(
                store, public_url=self.config.webhook_url, callback_path=self.config.webhook_path
            )
            self.publisher.install(app)
        return app

    async def handle_event(
        self, headers: Mapping[str, str], body: bytes, store: ChannelStore
    ) -> dict[str, Any]:
        expected = base64.b64encode(
            hmac.new(self.app_token.encode(), body, hashlib.sha256).digest()
        ).decode()
        if not hmac.compare_digest(expected, headers.get("X-Line-Signature", "")):
            raise AuthenticationError("LINE signature verification failed")
        payload = json.loads(body)
        if payload.get("destination") != self.bot_id:
            raise AuthenticationError("LINE callback destination does not match this bot")
        for event in payload.get("events", []):
            if event.get("type") != "message" or event.get("message", {}).get("type") not in {
                "text",
                "image",
                "audio",
                "video",
                "file",
            }:
                continue
            source = event.get("source", {})
            kind, user = source.get("type"), source.get("userId")
            destination = source.get(
                {"user": "userId", "group": "groupId", "room": "roomId"}.get(kind, "")
            )
            if not user or not destination or user == self.bot_id:
                continue
            message = event["message"]
            text = str(message.get("text", ""))
            attachments = []
            if message["type"] != "text":
                if message.get("contentProvider", {}).get("type", "line") != "line":
                    continue
                attachments = [
                    {
                        "message_id": str(message["id"]),
                        "type": message["type"],
                        "name": str(message.get("fileName") or "attachment"),
                    }
                ]
            mention_ranges = []
            for mention in message.get("mention", {}).get("mentionees", []):
                if mention.get("type") == "user" and mention.get("userId") == self.bot_id:
                    mention_ranges.append((int(mention["index"]), int(mention["length"])))
            # LINE offsets use UTF-16 code units, including emoji before mentions.
            encoded = text.encode("utf-16-le")
            for start, length in sorted(mention_ranges, reverse=True):
                if start >= 0 and length >= 0 and 2 * (start + length) <= len(encoded):
                    encoded = encoded[: 2 * start] + encoded[2 * (start + length) :]
            text = encoded.decode("utf-16-le").strip()
            self.accept(
                ChannelMessage(
                    id=str(event.get("webhookEventId") or message["id"]),
                    user_id=user,
                    channel_id=destination,
                    thread_id=json.dumps([kind, destination], separators=(",", ":")),
                    text=text,
                    group=kind != "user",
                    mentioned=bool(mention_ranges),
                    attachments=attachments,
                ),
                store,
            )
        return {}

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        await self.send_native(thread_id, {"type": "text", "text": text}, delivery_id)

    async def send_native(self, thread_id: str, message: dict[str, Any], delivery_id: str) -> None:
        destination = self.channel_id(thread_id)
        response = await self.client.post(
            "https://api.line.me/v2/bot/message/push",
            headers={
                "Authorization": "Bearer " + self.token,
                "X-Line-Retry-Key": str(uuid5(NAMESPACE_URL, delivery_id)),
            },
            json={"to": destination, "messages": [message]},
        )
        if response.status_code == 409 and response.headers.get("x-line-accepted-request-id"):
            return
        if response.status_code == 429:
            raise RateLimited(float(response.headers.get("Retry-After", "60")))
        if not response.is_success:
            raise ChannelError(
                "LINE rejected the push reply; check messaging quota and recipient access"
            )

    async def prepare_media(self, message: ChannelMessage) -> list[MediaAttachment]:
        results = []
        for item in message.attachments[:16]:
            identifier = str(item["message_id"])
            if not identifier.isdigit() or len(identifier) > 100:
                raise ChannelError("LINE media message ID is invalid")
            hint = {"image": "image/jpeg", "audio": "audio/mp4", "video": "video/mp4"}.get(
                item["type"], "application/octet-stream"
            )
            results.append(
                await download(
                    self.client,
                    f"https://api-data.line.me/v2/bot/message/{identifier}/content",
                    domains=("api-data.line.me",),
                    headers={"Authorization": "Bearer " + self.token},
                    mime=hint,
                    name=item["name"],
                )
            )
        return results

    async def send_media(
        self, thread_id: str, attachment: MediaAttachment, delivery_id: str
    ) -> None:
        self.channel_id(thread_id)
        publisher = getattr(self, "publisher", None)
        if publisher is None:
            raise ChannelError(
                "LINE inline media requires its running listener and public HTTPS webhook_url"
            )
        mime = attachment.mime_type
        kind = (
            "image"
            if mime in {"image/jpeg", "image/png"}
            else "audio"
            if mime in {"audio/mp4", "audio/m4a", "audio/mpeg"}
            else "video"
            if mime == "video/mp4"
            else ""
        )
        if not kind:
            raise ChannelError(
                "LINE supports JPEG/PNG images, MPEG/MP4 audio and MP4 video; arbitrary documents cannot be sent"
            )
        attachment_bytes(
            attachment, max_bytes=10 * 1024 * 1024 if kind == "image" else 20 * 1024 * 1024
        )
        duration = getattr(attachment, "duration_ms", None)
        if kind == "audio" and not duration:
            raise ChannelError("LINE audio requires explicit duration_ms metadata")
        url = publisher.publish(delivery_id, thread_id, attachment)
        self.secrets.append(url)
        native = {"type": kind, "originalContentUrl": url}
        if kind == "image":
            native["previewImageUrl"] = url
        elif kind == "video":
            # The API requires a preview even for generated video without one.
            preview = from_bytes(
                base64.b64decode(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a5WQAAAAASUVORK5CYII="
                ),
                "image/png",
                "preview.png",
            )
            native["previewImageUrl"] = publisher.publish(
                delivery_id + ":preview", thread_id, preview
            )
        else:
            native["duration"] = duration
        await self.send_native(thread_id, native, delivery_id)
