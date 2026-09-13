"""Meta WhatsApp Cloud API with signed, account-bound webhook ingestion."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from aiohttp import web

from harness.cli.channels.media import validate_media_url
from harness.cli.channels.transports import ChannelError
from harness.cli.channels.webhooks import AuthenticationError, WebhookTransport
from harness.core.gateway_channels import ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment


def numeric_id(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{1,64}", value):
        raise ValueError("WhatsApp requires an ASCII numeric resource or recipient ID")
    return value


class WhatsAppCloudTransport(WebhookTransport):
    name = "whatsapp_cloud"
    limit = 4096

    @property
    def base(self) -> str:
        if not re.fullmatch(r"v[0-9]+\.[0-9]+", self.config.api_version):
            raise ChannelError("WhatsApp api_version must be an explicit Graph API version")
        return "https://graph.facebook.com/" + self.config.api_version

    async def call(
        self, method: str, path: str, data: dict[str, Any] | None = None, *, files: Any = None
    ) -> Any:
        payload = await self.api(
            method, self.base + path, data=data, files=files, auth=f"Bearer {self.token}"
        )
        if not isinstance(payload, dict) or payload.get("error"):
            raise ChannelError("WhatsApp rejected the request")
        return payload

    async def authenticate(self) -> None:
        self.phone = numeric_id(self.config.phone_number_id)
        self.waba = numeric_id(self.config.waba_id)
        self.secret = os.environ.get(self.config.signing_secret_env, "")
        if not self.token or not self.secret or not self.app_token:
            raise ChannelError(
                "WhatsApp requires access token, app_token_env verification token, "
                "and signing_secret_env app secret"
            )
        self.secrets.append(self.secret)
        result = await self.call("GET", f"/{self.phone}?fields=id,display_phone_number")
        if result.get("id") != self.phone:
            raise ChannelError("WhatsApp authenticated a different phone number")
        self.bot_id = self.phone
        self.identity = json.dumps([self.waba, self.phone], separators=(",", ":"))

    def application(self, store: ChannelStore) -> web.Application:
        app = super().application(store)

        async def verify(request: web.Request) -> web.Response:
            query = request.query
            if (
                any(len(query.getall(key)) != 1 for key in query)
                or query.get("hub.mode") != "subscribe"
                or not hmac.compare_digest(
                    query.get("hub.verify_token", "").encode(), self.app_token.encode()
                )
            ):
                return web.Response(status=403)
            challenge = query.get("hub.challenge", "")
            if not challenge or len(challenge) > 4096:
                return web.Response(status=400)
            return web.Response(text=challenge, content_type="text/plain")

        app.router.add_get(self.config.webhook_path, verify)
        return app

    async def handle_event(
        self, headers: Mapping[str, str], body: bytes, store: ChannelStore
    ) -> dict[str, Any]:
        expected = "sha256=" + hmac.new(self.secret.encode(), body, hashlib.sha256).hexdigest()
        signature = next(
            (value for key, value in headers.items() if key.lower() == "x-hub-signature-256"), ""
        )
        if not hmac.compare_digest(signature.encode(), expected.encode()):
            raise AuthenticationError("WhatsApp webhook signature does not match")
        payload = json.loads(body)
        if payload.get("object") != "whatsapp_business_account":
            raise ValueError("WhatsApp requires a business account event")
        pending: list[ChannelMessage] = []
        for entry in payload["entry"]:
            if entry.get("id") != self.waba:
                raise AuthenticationError("WhatsApp event belongs to a different business account")
            for change in entry.get("changes", []):
                if change.get("field") != "messages":
                    continue
                value = change["value"]
                if (
                    value.get("messaging_product") != "whatsapp"
                    or value.get("metadata", {}).get("phone_number_id") != self.phone
                ):
                    raise AuthenticationError("WhatsApp event belongs to a different phone number")
                for item in value.get("messages", []):
                    message = self.parse(item)
                    if message:
                        pending.append(message)
        # Validate the entire signed batch before persisting any accepted work.
        for message in pending:
            self.accept(message, store)
        return {"received": True}

    def parse(self, item: dict[str, Any]) -> ChannelMessage | None:
        kind = item.get("type")
        text = ""
        attachments = []
        if kind == "text":
            text = item["text"]["body"]
        elif kind == "button":
            text = item["button"].get("text", "")
        elif kind == "interactive":
            interactive = item["interactive"]
            reply = interactive.get("button_reply") or interactive.get("list_reply") or {}
            text = reply.get("title") or reply.get("id", "")
        elif kind in {"image", "audio", "document", "video", "sticker"}:
            media = item[kind]
            text = str(media.get("caption") or "Please inspect the attached media.")
            attachments = [
                {
                    "id": numeric_id(media["id"]),
                    "mime_type": media.get("mime_type", "application/octet-stream"),
                    "name": media.get("filename", kind),
                }
            ]
        if not isinstance(text, str) or not text:
            return None
        sender = numeric_id(item["from"])
        message_id = item["id"]
        if not isinstance(message_id, str) or not message_id or len(message_id) > 512:
            raise ValueError("WhatsApp requires a message ID")
        return ChannelMessage(
            id=json.dumps([self.phone, message_id], separators=(",", ":")),
            user_id=sender,
            channel_id=sender,
            thread_id=sender,
            text=text,
            attachments=attachments,
        )

    async def _send(self, recipient: str, content: dict[str, Any], delivery_id: str) -> None:
        result = await self.call(
            "POST",
            f"/{self.phone}/messages",
            {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": numeric_id(recipient),
                "biz_opaque_callback_data": delivery_id,
                **content,
            },
        )
        if not result.get("messages") or not result["messages"][0].get("id"):
            raise ChannelError("WhatsApp did not acknowledge the sent message")

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        await self._send(
            thread_id, {"type": "text", "text": {"body": text, "preview_url": False}}, delivery_id
        )

    async def prepare_media(self, message: ChannelMessage) -> list[MediaAttachment]:
        result = []
        total = 0
        for item in message.attachments[:16]:
            metadata = await self.call("GET", "/" + numeric_id(item["id"]))
            url = metadata["url"]
            validate_media_url(url, ("lookaside.fbsbx.com", "graph.facebook.com"))
            self.secrets.append(url)
            raw = bytearray()
            async with self.client.stream(
                "GET",
                url,
                headers={"Authorization": f"Bearer {self.token}"},
                follow_redirects=False,
            ) as response:
                if not response.is_success:
                    raise ChannelError("WhatsApp media download rejected")
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > 20 * 1024 * 1024:
                        raise ChannelError("WhatsApp attachments exceed 20 MiB")
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
        numeric_id(thread_id)
        if attachment.data is None:
            raise ChannelError("WhatsApp uploads require inline bytes")
        raw = base64.b64decode(attachment.data, validate=True)
        kind = (
            "image"
            if attachment.kind == "image"
            else "audio"
            if attachment.kind == "audio"
            else "video"
            if attachment.mime_type.startswith("video/")
            else "document"
        )
        cap = {"image": 5, "audio": 16, "video": 16, "document": 20}[kind] * 1024 * 1024
        if len(raw) > cap:
            raise ChannelError("WhatsApp upload exceeds the attachment byte limit")
        filename = PurePosixPath(attachment.name or "attachment").name
        upload = await self.call(
            "POST",
            f"/{self.phone}/media",
            {"messaging_product": "whatsapp", "type": attachment.mime_type},
            files={"file": (filename, raw, attachment.mime_type)},
        )
        media = {"id": numeric_id(upload["id"])}
        if kind == "document":
            media["filename"] = filename
        await self._send(thread_id, {"type": kind, kind: media}, delivery_id)
