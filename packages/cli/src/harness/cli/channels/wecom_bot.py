"""Native WeCom smart-bot WebSocket protocol and encrypted attachment transfer."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import mimetypes
import time
from pathlib import PurePosixPath
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from harness.cli.channels.attachment_io import MAX_BYTES, attachment_bytes, download, from_bytes
from harness.cli.channels.transports import ChannelError, DeliveryRejected, Transport
from harness.core.gateway_channels import ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment

WS_URL = "wss://openws.work.weixin.qq.com"
CALLBACKS = {"aibot_msg_callback", "aibot_callback"}


class WeComBotTransport(Transport):
    name = "wecom"
    limit = 1000

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.ws: Any = None
        self.connection: Any = None
        self.store: ChannelStore | None = None
        self.pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self.revoked = False

    async def authenticate(self) -> None:
        if not self.config.app_id or not self.token:
            raise ChannelError("WeCom bot mode requires app_id (bot ID) and token_env (bot secret)")
        if self.config.allow_groups and self.config.require_mention and not self.config.username:
            raise ChannelError(
                "WeCom group mention policy requires username set to the bot's display name, or explicit require_mention=false"
            )
        self.bot_id = self.config.app_id
        self.identity = json.dumps(["wecom_bot", self.bot_id])
        await self.connect()

    async def connect(self) -> None:
        if self.revoked:
            raise ChannelError(
                "WeCom revoked this connection because another client connected; inspect competing clients before restarting"
            )
        self.connection = self.socket(WS_URL, domains=("openws.work.weixin.qq.com",))
        self.ws = await self.connection.__aenter__()
        try:
            identifier = "subscribe_" + uuid4().hex
            await self.ws.send(
                json.dumps(
                    {
                        "cmd": "aibot_subscribe",
                        "headers": {"req_id": identifier},
                        "body": {"bot_id": self.bot_id, "secret": self.token},
                    }
                )
            )
            async with asyncio.timeout(15):
                response = json.loads(await self.ws.recv())
            if (
                response.get("headers", {}).get("req_id") != identifier
                or response.get("errcode") != 0
                or response.get("cmd") in CALLBACKS
            ):
                raise ChannelError("WeCom smart-bot subscription authentication rejected")
        except BaseException:
            await self.disconnect()
            raise

    async def disconnect(self) -> None:
        connection, self.connection, self.ws = self.connection, None, None
        if connection is not None:
            await connection.__aexit__(None, None, None)
        for future in self.pending.values():
            if not future.done():
                future.set_exception(ChannelError("WeCom request interrupted; outcome uncertain"))

    async def close(self) -> None:
        await self.disconnect()
        await super().close()

    def can_send(self) -> bool:
        return self.ws is not None and self.store is not None and not self.revoked

    def channel_id(self, thread_id: str) -> str:
        bot, kind, target, owner = json.loads(thread_id)
        if (
            bot != self.bot_id
            or kind not in {"dm", "group"}
            or not target
            or not owner
            or (kind == "dm" and target != owner)
        ):
            raise ChannelError("WeCom bot destination has an invalid owner or bot identity")
        return kind + ":" + target

    def ingest(self, frame: dict[str, Any], store: ChannelStore) -> None:
        body = frame.get("body", {})
        identifier = frame.get("headers", {}).get("req_id")
        if (
            body.get("aibotid") != self.bot_id
            or body.get("chattype") not in {"single", "group"}
            or not body.get("msgid")
            or not identifier
        ):
            return
        owner = body.get("from", {}).get("userid")
        if not isinstance(owner, str) or not owner or owner == self.bot_id:
            return
        group = body["chattype"] == "group"
        target = body.get("chatid") if group else owner
        if not isinstance(target, str) or not target:
            return
        thread = json.dumps(
            [self.bot_id, "group" if group else "dm", target, owner], separators=(",", ":")
        )
        text, attachments = [], []
        items = (
            body.get("mixed", {}).get("msg_item", []) if body.get("msgtype") == "mixed" else [body]
        )
        quote = body.get("quote")
        if isinstance(quote, dict):
            items = [*items, quote]
        for item in items[:16]:
            kind = item.get("msgtype")
            if kind in {"text", "voice"}:
                text.append(str(item.get(kind, {}).get("content", "")))
            elif kind in {"image", "file", "video"}:
                media = item.get(kind, {})
                if isinstance(media, dict) and (media.get("url") or media.get("base64")):
                    attachments.append(
                        {
                            "type": kind,
                            **{
                                key: str(media[key])
                                for key in ("url", "aeskey", "base64", "filename", "name")
                                if media.get(key)
                            },
                        }
                    )
        content = "\n".join(text)
        prefix = "@" + self.config.username if self.config.username else ""
        mentioned = bool(
            prefix
            and content.startswith(prefix)
            and (len(content) == len(prefix) or content[len(prefix)].isspace())
        )
        if mentioned:
            content = content[len(prefix) :].lstrip()
        if not content and not attachments:
            return
        if not self.config.permits(
            user_id=owner, channel_id=self.channel_id(thread), group=group, mentioned=mentioned
        ):
            return
        if store.ingest(
            ChannelMessage(
                id=str(body["msgid"]),
                user_id=owner,
                channel_id=self.channel_id(thread),
                thread_id=thread,
                text=content,
                group=group,
                mentioned=mentioned,
                attachments=attachments,
            )
        ):
            store.set(
                "wecom-bot-route:" + thread,
                {"request_id": identifier, "expires": time.time() + 300},
            )
            store.set("connection_error", "")

    async def request(
        self, command: str, body: dict[str, Any], *, identifier: str = ""
    ) -> dict[str, Any]:
        if self.ws is None:
            raise ChannelError("WeCom bot is disconnected")
        identifier = identifier or command + "_" + uuid4().hex
        if identifier in self.pending:
            raise ChannelError("WeCom callback reply is already in flight")
        future = asyncio.get_running_loop().create_future()
        self.pending[identifier] = future
        try:
            await self.ws.send(
                json.dumps({"cmd": command, "headers": {"req_id": identifier}, "body": body})
            )
            result = await asyncio.wait_for(future, 15)
            if result.get("errcode") != 0:
                raise ChannelError("WeCom bot rejected the native operation")
            return result.get("body", {})
        finally:
            self.pending.pop(identifier, None)

    async def heartbeat(self) -> None:
        while True:
            await self.request("ping", {})
            await asyncio.sleep(30)

    async def receive(self, store: ChannelStore) -> None:
        if self.ws is None:
            await self.connect()
        socket = self.ws
        assert socket is not None
        self.store = store
        try:
            async with asyncio.TaskGroup() as tasks:
                ping = tasks.create_task(self.heartbeat())
                try:
                    async for raw in socket:
                        frame = json.loads(raw)
                        command = frame.get("cmd")
                        if command in CALLBACKS:
                            self.ingest(frame, store)
                        elif command == "aibot_event_callback":
                            body = frame.get("body", {})
                            if (
                                body.get("event", {}).get("eventtype") == "disconnected_event"
                                or body.get("event_type") == "disconnected_event"
                            ):
                                self.revoked = True
                                raise ChannelError(
                                    "WeCom revoked this connection because another client connected"
                                )
                        elif (
                            future := self.pending.get(frame.get("headers", {}).get("req_id"))
                        ) and not future.done():
                            future.set_result(frame)
                finally:
                    ping.cancel()
        finally:
            await self.disconnect()

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        self.channel_id(thread_id)
        _, kind, target, _ = json.loads(thread_id)
        if kind == "dm":
            await self.request(
                "aibot_send_msg",
                {"chatid": target, "msgtype": "markdown", "markdown": {"content": text}},
            )
            return
        route = self.store.get("wecom-bot-route:" + thread_id) if self.store else None
        if not route or route["expires"] <= time.time():
            raise DeliveryRejected(
                "WeCom group reply expired; a new message from its owner is required"
            )
        # One callback stream can be updated across split outbox parts. Finish
        # only after its last text part; resumed requests always use their owner.
        parts = (
            self.store.db.execute(
                "SELECT text,part,status FROM outbox WHERE source=(SELECT source FROM outbox WHERE id=?) AND attachment='' ORDER BY part",
                (delivery_id,),
            ).fetchall()
            if self.store
            else []
        )
        current = (
            self.store.db.execute(
                "SELECT part,source FROM outbox WHERE id=?", (delivery_id,)
            ).fetchone()
            if self.store
            else None
        )
        content = (
            "".join(part["text"] for part in parts if current and part["part"] <= current["part"])
            if current
            else text
        )
        if len(content.encode()) > 20480:
            raise DeliveryRejected("WeCom callback text exceeds its 20 KiB stream limit")
        source = current["source"] if current else delivery_id
        await self.request(
            "aibot_respond_msg",
            {
                "msgtype": "stream",
                "stream": {
                    "id": uuid5(NAMESPACE_URL, source).hex,
                    "content": content,
                    "finish": not current or current["part"] == parts[-1]["part"],
                },
            },
            identifier=route["request_id"],
        )

    async def prepare_media(self, message: ChannelMessage) -> list[MediaAttachment]:
        results, remaining = [], MAX_BYTES
        for item in message.attachments[:16]:
            name = str(item.get("filename") or item.get("name") or "attachment")
            mime = mimetypes.guess_type(name)[0] or {
                "image": "image/jpeg",
                "video": "video/mp4",
            }.get(item["type"], "application/octet-stream")
            if item.get("base64"):
                encoded = str(item["base64"])
                if len(encoded) > (remaining + 2) // 3 * 4:
                    raise ChannelError("WeCom media exceeds its byte limit")
                raw = base64.b64decode(encoded, validate=True)
            else:
                url = str(item["url"])
                self.secrets.append(url)
                attachment = await download(
                    self.client,
                    url,
                    domains=("weixin.qq.com", "weixin.qq.com.cn", "myqcloud.com", "qpic.cn"),
                    max_bytes=remaining + 32,
                    mime=mime,
                    name=name,
                )
                raw = base64.b64decode(attachment.data or "")
                key = item.get("aeskey")
                if key:
                    decoded = base64.b64decode(
                        str(key) + "=" * ((-len(str(key))) % 4), validate=True
                    )
                    if len(decoded) != 32:
                        raise ChannelError("WeCom attachment AES key is invalid")
                    cipher = Cipher(algorithms.AES(decoded), modes.CBC(decoded[:16])).decryptor()
                    raw = cipher.update(raw) + cipher.finalize()
                    padding = raw[-1] if raw else 0
                    if not 1 <= padding <= 32 or raw[-padding:] != bytes([padding]) * padding:
                        raise ChannelError("WeCom attachment padding is invalid")
                    raw = raw[:-padding]
            remaining -= len(raw)
            if remaining < 0:
                raise ChannelError("WeCom media exceeds its byte limit")
            results.append(from_bytes(raw, mime, name))
        return results

    async def send_media(
        self, thread_id: str, attachment: MediaAttachment, delivery_id: str
    ) -> None:
        self.channel_id(thread_id)
        _, kind, target, _ = json.loads(thread_id)
        route = self.store.get("wecom-bot-route:" + thread_id) if self.store else None
        if kind == "group" and (not route or route["expires"] <= time.time()):
            raise DeliveryRejected(
                "WeCom group media reply expired; a new message from its owner is required"
            )
        raw = attachment_bytes(attachment)
        media_type = (
            "image"
            if attachment.kind == "image"
            else "video"
            if attachment.mime_type == "video/mp4"
            else "voice"
            if attachment.mime_type == "audio/amr"
            else "file"
        )
        cap = {"image": 10, "video": 10, "voice": 2, "file": 20}[media_type] * 1024 * 1024
        if len(raw) > cap:
            media_type = "file"
        chunk_size = 512 * 1024
        async with asyncio.timeout(180):
            slot = await self.request(
                "aibot_upload_media_init",
                {
                    "type": media_type,
                    "filename": PurePosixPath(attachment.name or "attachment").name,
                    "total_size": len(raw),
                    "total_chunks": (len(raw) + chunk_size - 1) // chunk_size,
                    "md5": hashlib.md5(raw).hexdigest(),
                },
            )
            upload_id = slot.get("upload_id")
            if not upload_id:
                raise ChannelError("WeCom did not allocate an upload")
            for index, start in enumerate(range(0, len(raw), chunk_size)):
                await self.request(
                    "aibot_upload_media_chunk",
                    {
                        "upload_id": upload_id,
                        "chunk_index": index,
                        "base64_data": base64.b64encode(raw[start : start + chunk_size]).decode(),
                    },
                )
            finish = await self.request("aibot_upload_media_finish", {"upload_id": upload_id})
            if not finish.get("media_id"):
                raise ChannelError("WeCom did not finalize the uploaded media")
            body = {"msgtype": media_type, media_type: {"media_id": finish["media_id"]}}
            if kind == "group" and route and route["expires"] > time.time():
                await self.request("aibot_respond_msg", body, identifier=route["request_id"])
            elif kind == "group":
                raise DeliveryRejected(
                    "WeCom group media reply expired; a new message from its owner is required"
                )
            else:
                await self.request("aibot_send_msg", {"chatid": target, **body})
