"""Tencent iLink transport with durable account/peer-bound reply context."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import mimetypes
import secrets
import uuid
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from harness.cli.channels.transports import ChannelError, RateLimited, Transport
from harness.core.gateway_channels import ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment

ILINK = "https://ilinkai.weixin.qq.com"
CDN = "https://novac2c.cdn.weixin.qq.com/c2c"
MEDIA_CAP = 20 * 1024 * 1024


def ilink_base(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "ilinkai.weixin.qq.com"
        or parsed.port not in {None, 443}
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ChannelError("Weixin API must use the official iLink HTTPS origin")
    return value.rstrip("/")


def media_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "novac2c.cdn.weixin.qq.com"
        or parsed.port not in {None, 443}
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        raise ChannelError("Weixin media must use its official HTTPS CDN")
    return value


def aes_key(value: str) -> bytes:
    raw = base64.b64decode(value, validate=True)
    if len(raw) == 32:
        raw = bytes.fromhex(raw.decode("ascii"))
    if len(raw) != 16:
        raise ValueError("Weixin AES key must contain 16 bytes")
    return raw


def crypt_media(raw: bytes, key: bytes, *, encrypt: bool) -> bytes:
    if len(key) != 16:
        raise ValueError("Weixin media requires AES-128")
    cipher = Cipher(algorithms.AES(key), modes.ECB())
    if encrypt:
        padder = padding.PKCS7(128).padder()
        raw = padder.update(raw) + padder.finalize()
        operation = cipher.encryptor()
        return operation.update(raw) + operation.finalize()
    operation = cipher.decryptor()
    padded = operation.update(raw) + operation.finalize()
    unpadder = padding.PKCS7(128).unpadder()
    return unpadder.update(padded) + unpadder.finalize()


class WeixinTransport(Transport):
    name = "weixin"
    limit = 2000

    async def authenticate(self) -> None:
        account = json.loads(self.token) if self.token.startswith("{") else None
        if account is not None:
            if not isinstance(account, dict) or not all(
                isinstance(account.get(key), str) and account[key]
                for key in ("ilink_bot_id", "bot_token")
            ):
                raise ChannelError(
                    "Weixin credential file requires bot_token and ilink_bot_id strings"
                )
            if self.config.app_id and self.config.app_id != account.get("ilink_bot_id"):
                raise ChannelError("Weixin credentials belong to another configured bot")
            self.bot_id = str(account["ilink_bot_id"])
            self.token = str(account["bot_token"])
            self.base = ilink_base(self.config.homeserver or account.get("base_url") or ILINK)
            self.secrets.append(self.token)
        else:
            self.bot_id = self.config.app_id
            self.base = ilink_base(self.config.homeserver or ILINK)
        if not self.bot_id or not self.token or not self.config.allowed_users:
            raise ChannelError("Weixin requires paired bot ID/token and explicit allowed_users")
        if self.config.allow_groups:
            raise ChannelError("Weixin iLink group routing is not supported by this adapter")
        self.identity = json.dumps([self.base, self.bot_id])
        self.store: ChannelStore | None = None
        # Read-only credential probe. Do not consume getupdates before loading its cursor.
        await self.call("getconfig", {"ilink_user_id": self.config.allowed_users[0]})

    async def call(
        self, method: str, data: dict[str, Any], *, poll: bool = False
    ) -> dict[str, Any]:
        try:
            response = await self.client.post(
                self.base + "/ilink/bot/" + method,
                json={
                    **data,
                    "base_info": {"channel_version": "2.4.8", "bot_agent": "Harness/0.1"},
                },
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "AuthorizationType": "ilink_bot_token",
                    "X-WECHAT-UIN": base64.b64encode(str(secrets.randbits(32)).encode()).decode(),
                    "iLink-App-Id": "bot",
                    "iLink-App-ClientVersion": str((2 << 16) | (4 << 8) | 8),
                },
                timeout=40 if poll else 15,
                follow_redirects=False,
            )
        except httpx.TimeoutException:
            if poll:
                return {"ret": 0, "msgs": [], "get_updates_buf": data["get_updates_buf"]}
            raise ChannelError("Weixin send timed out; delivery may be uncertain") from None
        except httpx.HTTPError:
            raise ChannelError("Weixin connection interrupted") from None
        if response.status_code == 429:
            raise RateLimited(float(response.headers.get("Retry-After", "30")))
        if not response.is_success:
            raise ChannelError("Weixin API rejected the request")
        result = response.json()
        if not isinstance(result, dict):
            raise ChannelError("Weixin returned invalid response data")
        codes = {result.get("ret", 0), result.get("errcode", 0)}
        if -14 in codes:
            raise ChannelError("Weixin session expired; pair the account again")
        if -2 in codes:
            raise RateLimited(30)
        if codes - {0, None}:
            raise ChannelError("Weixin API reported an error")
        return result

    async def receive(self, store: ChannelStore) -> None:
        self.store = store
        try:
            while True:
                await self.poll_once(store)
                await asyncio.sleep(0.1)
        finally:
            self.store = None

    def can_send(self) -> bool:
        return getattr(self, "store", None) is not None

    async def poll_once(self, store: ChannelStore) -> None:
        response = await self.call(
            "getupdates", {"get_updates_buf": store.get("cursor", "")}, poll=True
        )
        cursor = response.get("get_updates_buf")
        if not isinstance(cursor, str):
            raise ChannelError("Weixin omitted its restart cursor")
        records = response.get("msgs", [])
        for record in records:
            if (
                record.get("message_type") == 1
                and not record.get("room_id")
                and not record.get("chat_room_id")
                and record.get("to_user_id") != self.bot_id
            ):
                raise ChannelError("Weixin event belongs to another bot account")
        for record in records:
            if record.get("message_type") != 1 or record.get("message_state", 2) != 2:
                continue
            if record.get("room_id") or record.get("chat_room_id"):
                continue
            if record.get("to_user_id") != self.bot_id:
                raise ChannelError("Weixin event belongs to another bot account")
            sender = record.get("from_user_id")
            if (
                not isinstance(sender, str)
                or sender == self.bot_id
                or not self.config.permits(
                    user_id=sender, channel_id=sender, group=False, mentioned=False
                )
            ):
                continue
            context = record.get("context_token")
            if not isinstance(context, str) or not context:
                raise ChannelError("Weixin message omitted its reply context token")
            text, attachments = [], []
            for item in record.get("item_list", []):
                if item.get("type") == 1:
                    text.append(str(item.get("text_item", {}).get("text", "")))
                elif item.get("type") in {2, 3, 4, 5}:
                    kind = {2: "image", 3: "voice", 4: "file", 5: "video"}[item["type"]]
                    value = item.get(kind + "_item", {})
                    if kind == "voice" and value.get("text"):
                        text.append(str(value["text"]))
                    if value.get("media"):
                        attachments.append({"kind": kind, "item": value})
            if not any(text) and not attachments:
                continue
            mid = record.get("message_id")
            if type(mid) not in {int, str} or str(mid) == "":
                raise ChannelError("Weixin message requires a durable message ID")
            message = ChannelMessage(
                id=json.dumps([self.bot_id, str(mid)]),
                user_id=sender,
                thread_id=sender,
                channel_id=sender,
                text="\n".join(text) or "Please inspect the attached media.",
                attachments=attachments[:16],
            )
            # Refresh authenticated context even on a duplicate delivery. Keep it
            # in private state, never in the model transcript or conversation ID.
            key = "context:" + sender
            previous = store.get(key, {})
            sequence = record.get("seq", 0)
            if not isinstance(sequence, int):
                raise ChannelError("Weixin returned an invalid message sequence")
            if sequence >= previous.get("sequence", -1):
                store.set(key, {"token": context, "sequence": sequence, "account": self.bot_id})
            store.ingest(message)
        store.set("cursor", cursor)

    def context(self, target: str) -> str:
        if self.store is None:
            raise ChannelError("Weixin reply context store is not open")
        value = self.store.get("context:" + target, {})
        if not value.get("token") or value.get("account") != self.bot_id:
            raise ChannelError("Weixin requires a prior authenticated message from this peer")
        return value["token"]

    async def send_items(self, target: str, items: list[dict[str, Any]], delivery_id: str) -> None:
        await self.call(
            "sendmessage",
            {
                "msg": {
                    "from_user_id": "",
                    "to_user_id": target,
                    "client_id": "harness-" + delivery_id,
                    "message_type": 2,
                    "message_state": 2,
                    "item_list": items,
                    "context_token": self.context(target),
                }
            },
        )

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        await self.send_items(thread_id, [{"type": 1, "text_item": {"text": text}}], delivery_id)

    async def prepare_media(self, message: ChannelMessage) -> list[MediaAttachment]:
        result, total, plaintext_total = [], 0, 0
        for descriptor in message.attachments[:16]:
            item, kind = descriptor["item"], descriptor["kind"]
            media = item["media"]
            url = media.get("full_url") or CDN + "/download?" + urlencode(
                {"encrypted_query_param": media["encrypt_query_param"]}
            )
            media_url(url)
            self.secrets.append(url)
            raw = bytearray()
            async with self.client.stream(
                "GET", url, follow_redirects=False, timeout=60
            ) as response:
                if not response.is_success:
                    raise ChannelError("Weixin CDN download rejected")
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > MEDIA_CAP + 16 * len(message.attachments):
                        raise ChannelError("Weixin media exceeds 20 MiB")
                    raw.extend(chunk)
            key = bytes.fromhex(item["aeskey"]) if item.get("aeskey") else aes_key(media["aes_key"])
            plain = crypt_media(bytes(raw), key, encrypt=False)
            plaintext_total += len(plain)
            if plaintext_total > MEDIA_CAP:
                raise ChannelError("Weixin media exceeds 20 MiB")
            name = PurePosixPath(
                str(
                    item.get("file_name")
                    or {
                        "image": "image.jpg",
                        "voice": "voice.silk",
                        "video": "video.mp4",
                        "file": "attachment",
                    }[kind]
                )
            ).name
            if (
                item.get("md5")
                and hashlib.md5(plain, usedforsecurity=False).hexdigest() != item["md5"]
            ):
                raise ChannelError("Weixin file checksum does not match")
            mime = (
                {"image": "image/jpeg", "voice": "audio/silk", "video": "video/mp4"}.get(kind)
                or mimetypes.guess_type(name)[0]
                or "application/octet-stream"
            )
            if kind == "image":
                if plain.startswith(b"\x89PNG\r\n\x1a\n"):
                    mime, name = "image/png", "image.png"
                elif plain.startswith((b"GIF87a", b"GIF89a")):
                    mime, name = "image/gif", "image.gif"
                elif plain[:4] == b"RIFF" and plain[8:12] == b"WEBP":
                    mime, name = "image/webp", "image.webp"
                elif not plain.startswith(b"\xff\xd8\xff"):
                    kind, mime = "file", "application/octet-stream"
            result.append(
                MediaAttachment(
                    kind="image" if kind == "image" else "audio" if kind == "voice" else "file",
                    mime_type=mime,
                    name=name[:255],
                    data=base64.b64encode(plain).decode(),
                )
            )
        return result

    async def send_media(
        self, thread_id: str, attachment: MediaAttachment, delivery_id: str
    ) -> None:
        self.context(thread_id)  # Refuse uploads when there is no scoped reply route.
        if attachment.data is None:
            raise ChannelError("Weixin uploads require inline bytes")
        raw = base64.b64decode(attachment.data, validate=True)
        if len(raw) > MEDIA_CAP:
            raise ChannelError("Weixin media exceeds 20 MiB")
        # Arbitrary audio is sent as a file; native voice requires SILK encoding.
        kind = (
            "image"
            if attachment.kind == "image"
            else "video"
            if attachment.mime_type.startswith("video/")
            else "file"
        )
        key, filekey = secrets.token_bytes(16), uuid.uuid4().hex
        encrypted = crypt_media(raw, key, encrypt=True)
        slot = await self.call(
            "getuploadurl",
            {
                "filekey": filekey,
                "media_type": {"image": 1, "video": 2, "file": 3}[kind],
                "to_user_id": thread_id,
                "rawsize": len(raw),
                "rawfilemd5": hashlib.md5(raw, usedforsecurity=False).hexdigest(),
                "filesize": len(encrypted),
                "no_need_thumb": True,
                "aeskey": key.hex(),
            },
        )
        url = slot.get("upload_full_url") or CDN + "/upload?" + urlencode(
            {"encrypted_query_param": slot["upload_param"], "filekey": filekey}
        )
        media_url(url)
        self.secrets.append(url)
        response = await self.client.post(
            url,
            content=encrypted,
            headers={"Content-Type": "application/octet-stream"},
            follow_redirects=False,
            timeout=60,
        )
        param = response.headers.get("x-encrypted-param")
        if not response.is_success or not param:
            raise ChannelError("Weixin CDN upload was not acknowledged")
        item: dict[str, Any] = {
            "media": {
                "encrypt_query_param": param,
                "aes_key": base64.b64encode(key.hex().encode()).decode(),
                "encrypt_type": 1,
            }
        }
        if kind == "file":
            item.update(
                file_name=PurePosixPath(attachment.name or "attachment").name,
                len=str(len(raw)),
                md5=hashlib.md5(raw, usedforsecurity=False).hexdigest(),
            )
        elif kind == "image":
            item["mid_size"] = len(encrypted)
        else:
            item["video_size"] = len(encrypted)
        await self.send_items(
            thread_id,
            [{"type": {"image": 2, "file": 4, "video": 5}[kind], kind + "_item": item}],
            delivery_id,
        )
