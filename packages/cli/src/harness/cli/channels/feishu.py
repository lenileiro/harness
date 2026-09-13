"""Feishu/Lark signed, encrypted event callbacks and tenant-token replies."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from harness.cli.channels.native_media import (
    ATTACHMENT_LIMIT,
    MEDIA_LIMIT,
    attachment_bytes,
    download_bytes,
    media_attachment,
    media_name,
    media_type,
    resource_segment,
)
from harness.cli.channels.transports import ChannelError, RateLimited
from harness.cli.channels.webhooks import AuthenticationError, WebhookTransport
from harness.core.gateway_channels import ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment


class FeishuTransport(WebhookTransport):
    name = "feishu"
    limit = 4000

    @property
    def base(self) -> str:
        return (
            "https://open.larksuite.com/open-apis"
            if self.name == "lark"
            else "https://open.feishu.cn/open-apis"
        )

    async def authenticate(self) -> None:
        self.encrypt_key = os.environ.get(self.config.signing_secret_env, "")
        if not self.encrypt_key or not self.app_token or not self.config.app_id:
            raise ChannelError(
                "Feishu/Lark requires app_id, app secret token_env, verification-token app_token_env, and Encrypt Key signing_secret_env"
            )
        self.secrets.append(self.encrypt_key)
        self.identity = f"{self.name}:{self.config.app_id}"
        self.access_token = ""
        self.token_expires = 0.0
        token = await self._access()
        info = await self.api("GET", self.base + "/bot/v3/info", auth=f"Bearer {token}")
        if info.get("code", 0) != 0 or not info.get("bot", {}).get("open_id"):
            raise ChannelError("Feishu/Lark bot identity could not be verified")
        self.bot_id = str(info["bot"]["open_id"])

    async def _access(self) -> str:
        if time.time() < self.token_expires - 60:
            return self.access_token
        payload = await self.api(
            "POST",
            self.base + "/auth/v3/tenant_access_token/internal",
            data={"app_id": self.config.app_id, "app_secret": self.token},
        )
        if payload.get("code", 0) != 0 or not payload.get("tenant_access_token"):
            raise ChannelError("Feishu/Lark tenant token exchange failed")
        self.access_token = str(payload["tenant_access_token"])
        self.token_expires = time.time() + min(7200, float(payload.get("expire", 7200)))
        self.secrets.append(self.access_token)
        return self.access_token

    def channel_id(self, thread_id: str) -> str:
        chat, _ = json.loads(thread_id)
        return chat

    def _decode(self, body: bytes) -> tuple[dict[str, Any], bool]:
        raw = json.loads(body)
        encrypted = raw.get("encrypt")
        if encrypted is None:
            return raw, False
        data = base64.b64decode(encrypted, validate=True)
        if len(data) < 32 or len(data) % 16:
            raise AuthenticationError("Invalid encrypted callback")
        decryptor = Cipher(
            algorithms.AES(hashlib.sha256(self.encrypt_key.encode()).digest()), modes.CBC(data[:16])
        ).decryptor()
        padded = decryptor.update(data[16:]) + decryptor.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        return json.loads(unpadder.update(padded) + unpadder.finalize()), True

    async def handle_event(
        self, headers: Mapping[str, str], body: bytes, store: ChannelStore
    ) -> dict[str, Any]:
        try:
            event, encrypted = self._decode(body)
            header = event.get("header", {})
            verification = header.get("token", event.get("token", ""))
            if not isinstance(verification, str) or not hmac.compare_digest(
                verification, self.app_token
            ):
                raise ValueError
            # Feishu's initial encrypted URL challenge has no signature headers.
            # It is the only accepted unsigned callback, and cannot dispatch work.
            if event.get("type") == "url_verification" and encrypted:
                challenge = event["challenge"]
                if not isinstance(challenge, str) or len(challenge) > 1024:
                    raise ValueError
                return {"challenge": challenge}
            timestamp = headers.get("X-Lark-Request-Timestamp", "")
            nonce = headers.get("X-Lark-Request-Nonce", "")
            signature = headers.get("X-Lark-Signature", "")
            if not timestamp or not nonce or abs(time.time() - int(timestamp)) > 300:
                raise ValueError
            expected = hashlib.sha256(
                (timestamp + nonce + self.encrypt_key).encode() + body
            ).hexdigest()
            if (
                not hmac.compare_digest(expected, signature)
                or header.get("app_id") != self.config.app_id
            ):
                raise ValueError
        except (ValueError, KeyError, TypeError):
            raise AuthenticationError("Feishu/Lark callback authentication failed") from None
        if header.get("event_type") == "card.action.trigger":
            from harness.cli.channels.approval_interactions import accept_choice

            action_event = event.get("event", {})
            actor, context = action_event.get("operator", {}), action_event.get("context", {})
            tenant = str(actor.get("tenant_key") or "")
            if not tenant or (header.get("tenant_key") and header["tenant_key"] != tenant):
                raise AuthenticationError("Feishu/Lark card operator tenant does not match")
            value = action_event.get("action", {}).get("value", {})
            accepted = await accept_choice(
                self,
                store,
                token=value.get("harness_approval", "") if isinstance(value, dict) else "",
                user_id=tenant + ":" + str(actor.get("open_id") or ""),
                channel_id=str(context.get("open_chat_id") or ""),
                native_message_id=str(context.get("open_message_id") or ""),
                event_id=str(header.get("event_id") or ""),
            )
            return {
                "toast": {
                    "type": "success" if accepted else "error",
                    "content": "Decision queued."
                    if accepted
                    else "Unavailable, expired, or already used.",
                }
            }
        if header.get("event_type") != "im.message.receive_v1":
            return {"code": 0}
        payload = event["event"]
        sender, message = payload["sender"], payload["message"]
        if sender.get("sender_type") != "user":
            return {"code": 0}
        tenant = str(header["tenant_key"])
        user_id = tenant + ":" + str(sender["sender_id"]["open_id"])
        group = message.get("chat_type") != "p2p"
        root = str(message.get("root_id") or message["message_id"]) if group else ""
        mentioned = any(
            item.get("id", {}).get("open_id") == self.bot_id for item in message.get("mentions", [])
        )
        content = json.loads(message["content"])
        text = str(content.get("text", ""))
        attachments: list[dict[str, Any]] = []
        kind = message.get("message_type")
        if kind in {"image", "file", "audio", "media"}:
            key = "image_key" if kind == "image" else "file_key"
            attachments.append(
                {
                    "message_id": message["message_id"],
                    "key": content[key],
                    "type": "image" if kind == "image" else "file",
                    "name": content.get("file_name", "attachment"),
                    "mime": "image/jpeg"
                    if kind == "image"
                    else "audio/ogg"
                    if kind == "audio"
                    else "video/mp4"
                    if kind == "media"
                    else "",
                    "duration_ms": content.get("duration") if kind == "audio" else None,
                }
            )
        elif kind == "post":
            # Received post content is unlocalized; tolerate the localized send
            # schema as well, taking one language instead of duplicating assets.
            post = content if "content" in content else next(iter(content.values()), {})
            lines = [str(post.get("title", ""))]
            for row in post.get("content", []):
                parts = []
                for node in row:
                    if node.get("tag") == "img":
                        attachments.append(
                            {
                                "message_id": message["message_id"],
                                "key": node["image_key"],
                                "type": "image",
                                "name": "image",
                                "mime": "image/jpeg",
                            }
                        )
                    elif node.get("tag") in {"text", "a"}:
                        parts.append(str(node.get("text", "")))
                lines.append("".join(parts))
            text = "\n".join(lines).strip()
        elif kind != "text":
            return {"code": 0}
        if len(attachments) > ATTACHMENT_LIMIT:
            raise ChannelError("Too many native attachments (maximum 16)")
        for mention in message.get("mentions", []):
            if mention.get("id", {}).get("open_id") == self.bot_id:
                text = text.replace(str(mention.get("key", "@_user_1")), "").strip()
        self.accept(
            ChannelMessage(
                id=str(header["event_id"]),
                user_id=user_id,
                channel_id=str(message["chat_id"]),
                thread_id=json.dumps([message["chat_id"], root], separators=(",", ":")),
                text=text,
                group=group,
                mentioned=mentioned,
                attachments=attachments,
            ),
            store,
        )
        return {"code": 0}

    async def prepare_media(self, message: ChannelMessage) -> list[MediaAttachment]:
        if len(message.attachments) > ATTACHMENT_LIMIT:
            raise ChannelError("Too many native attachments (maximum 16)")
        resources = []
        for item in message.attachments:
            message_id, key = (
                resource_segment(item.get("message_id")),
                resource_segment(item.get("key")),
            )
            kind = item.get("type")
            if kind not in {"image", "file"}:
                raise ChannelError("Invalid Feishu/Lark attachment resource type")
            resources.append((item, message_id, key, kind))
        remaining = MEDIA_LIMIT
        result = []
        for item, message_id, key, kind in resources:
            data, mime = await download_bytes(
                self,
                f"{self.base}/im/v1/messages/{quote(message_id, safe='')}/resources/{quote(key, safe='')}?type={kind}",
                remaining=remaining,
                auth=f"Bearer {await self._access()}",
            )
            remaining -= len(data)
            attachment = media_attachment(
                data,
                item.get("name"),
                mime if mime and mime != "application/octet-stream" else item.get("mime"),
            )
            duration = item.get("duration_ms")
            if type(duration) is int and 0 < duration <= 86_400_000:
                attachment.duration_ms = duration
            result.append(attachment)
        return result

    async def _send_content(
        self, thread_id: str, kind: str, content: dict[str, Any], delivery_id: str
    ) -> str:
        chat, root = json.loads(thread_id)
        resource_segment(chat)
        if root:
            resource_segment(root)
        data: dict[str, Any] = {
            "msg_type": kind,
            "content": json.dumps(content),
            "uuid": delivery_id[:50],
        }
        if root:
            path = f"/im/v1/messages/{quote(root, safe='')}/reply"
            data["reply_in_thread"] = True
        else:
            path = "/im/v1/messages?receive_id_type=chat_id"
            data["receive_id"] = chat
        payload = await self.api(
            "POST", self.base + path, data=data, auth=f"Bearer {await self._access()}"
        )
        if payload.get("code") == 99991400:
            raise RateLimited(1)
        if payload.get("code", 0) != 0:
            raise ChannelError("Feishu/Lark rejected the message")
        return str(payload.get("data", {}).get("message_id") or "")

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        await self._send_content(thread_id, "text", {"text": text}, delivery_id)

    async def send_approval(
        self, thread_id: str, text: str, delivery_id: str, buttons: dict[str, str]
    ) -> str:
        return await self._send_content(
            thread_id,
            "interactive",
            {
                "config": {"wide_screen_mode": True, "update_multi": True},
                "elements": [
                    {"tag": "div", "text": {"tag": "plain_text", "content": text}},
                    {
                        "tag": "action",
                        "actions": [
                            {
                                "tag": "button",
                                "type": "primary",
                                "text": {"tag": "plain_text", "content": "Approve"},
                                "value": {"harness_approval": buttons["approve"]},
                            },
                            {
                                "tag": "button",
                                "type": "danger",
                                "text": {"tag": "plain_text", "content": "Deny"},
                                "value": {"harness_approval": buttons["deny"]},
                            },
                        ],
                    },
                ],
            },
            delivery_id,
        )

    async def send_media(
        self, thread_id: str, attachment: MediaAttachment, delivery_id: str
    ) -> None:
        chat, root = json.loads(thread_id)
        resource_segment(chat)
        if root:
            resource_segment(root)
        raw = attachment_bytes(attachment)
        name = media_name(attachment.name)
        mime = media_type(attachment.mime_type, name)
        image = attachment.kind == "image"
        if image and len(raw) > 10 * 1024 * 1024:
            raise ChannelError("Feishu/Lark image uploads are limited to 10 MiB")
        # Only actual Opus audio uses the native audio message type. Other
        # codecs and video are delivered as downloadable files; no transcoding.
        audio = attachment.kind == "audio" and (
            mime == "audio/opus" or (mime == "audio/ogg" and b"OpusHead" in raw[:128])
        )
        kind = "image" if image else "audio" if audio else "file"
        path = "/im/v1/images" if image else "/im/v1/files"
        fields = (
            {"image_type": "message"}
            if image
            else {"file_type": "opus" if audio else "stream", "file_name": name}
        )
        if audio and attachment.duration_ms is not None:
            fields["duration"] = str(attachment.duration_ms)
        payload = await self.api(
            "POST",
            self.base + path,
            data=fields,
            files={"image" if image else "file": (name, raw, mime)},
            auth=f"Bearer {await self._access()}",
        )
        if payload.get("code") == 99991400:
            raise RateLimited(1)
        key = "image_key" if image else "file_key"
        if payload.get("code", 0) != 0 or not payload.get("data", {}).get(key):
            raise ChannelError("Feishu/Lark attachment upload failed")
        await self._send_content(thread_id, kind, {key: payload["data"][key]}, delivery_id)


class LarkTransport(FeishuTransport):
    name = "lark"
