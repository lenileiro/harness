"""Yuanbao Bot HTTPS sign-token and authenticated binary WebSocket text transport."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import sys
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import NAMESPACE_URL, uuid4, uuid5

from harness.cli.channels.transports import ChannelError, RateLimited, Transport
from harness.cli.channels.yuanbao_proto import Frame, binary, field, number, parse, string
from harness.core.gateway_channels import ChannelMessage, ChannelStore

SERVICE = "yuanbao_openclaw_proxy"
WS_URL = "wss://bot-wss.yuanbao.tencent.com/wss/connection"
API_URL = "https://bot.yuanbao.tencent.com/api/v5/robotLogic/sign-token"


class YuanbaoTransport(Transport):
    name = "yuanbao"
    limit = 4000

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.ws: Any = None
        self.signed: dict[str, Any] = {}
        self.sequence = 0
        self.pending: dict[str, tuple[str, str, asyncio.Future[Frame]]] = {}
        self.heartbeat_interval = 20.0

    async def sign(self) -> None:
        if not self.config.app_id or not self.token:
            raise ChannelError("Yuanbao requires app_id (app key) and token_env (app secret)")
        for attempt in range(4):
            nonce = secrets.token_hex(16)
            stamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
            signature = hmac.new(
                self.token.encode(),
                (nonce + stamp + self.config.app_id + self.token).encode(),
                hashlib.sha256,
            ).hexdigest()
            response = await self.client.post(
                API_URL,
                json={
                    "app_key": self.config.app_id,
                    "nonce": nonce,
                    "timestamp": stamp,
                    "signature": signature,
                },
                headers={
                    "X-AppVersion": "harness-1",
                    "X-OperationSystem": sys.platform,
                    "X-Instance-Id": "16",
                    "X-Bot-Version": "harness-1",
                },
            )
            if response.status_code == 429:
                raise RateLimited(float(response.headers.get("Retry-After", "30")))
            if not response.is_success or len(response.content) > 16 * 1024:
                raise ChannelError("Yuanbao token exchange rejected")
            payload = response.json()
            if payload.get("code") == 10099 and attempt < 3:
                await asyncio.sleep(1)
                continue
            signed = payload.get("data", {})
            bot_id = signed.get("bot_id")
            if (
                payload.get("code") != 0
                or not isinstance(bot_id, str)
                or not bot_id
                or not signed.get("token")
            ):
                raise ChannelError("Yuanbao token exchange did not authenticate a bot")
            if self.bot_id and bot_id != self.bot_id:
                raise ChannelError("Yuanbao bot identity changed; use a separate profile")
            self.bot_id = bot_id
            self.identity = json.dumps([self.config.app_id, bot_id])
            self.signed = signed
            self.secrets.append(signed["token"])
            return

    async def authenticate(self) -> None:
        await self.sign()

    def next_sequence(self) -> int:
        self.sequence = self.sequence % (2**32 - 1) + 1
        return self.sequence

    def can_send(self) -> bool:
        return self.ws is not None

    def channel_id(self, thread_id: str) -> str:
        bot, kind, target, origin = json.loads(thread_id)
        if (
            bot != self.bot_id
            or kind not in {"dm", "group"}
            or not isinstance(target, str)
            or not target
            or not isinstance(origin, str)
        ):
            raise ChannelError("Yuanbao destination belongs to another bot or is invalid")
        return kind + ":" + target

    async def bind(self, socket) -> None:
        identifier = uuid4().hex
        auth = (
            field(1, self.bot_id)
            + field(2, self.signed.get("source") or "bot")
            + field(3, self.signed["token"])
        )
        device = field(1, "harness-1") + field(2, sys.platform) + field(10, "16")
        # Explicit conflict rejection: never take over another running client.
        body = field(1, "ybBot") + field(2, auth) + field(3, device) + field(6, 1)
        await socket.send(
            Frame(0, "auth-bind", identifier, "conn_access", body, self.next_sequence()).encode()
        )
        async with asyncio.timeout(15):
            raw = await socket.recv()
        if not isinstance(raw, bytes):
            raise ChannelError("Yuanbao authentication requires a binary acknowledgment")
        reply = Frame.decode(raw)
        result = parse(reply.data)
        if (
            reply.kind != 1
            or reply.command != "auth-bind"
            or reply.module != "conn_access"
            or reply.identifier != identifier
            or reply.status
            or number(result, 1)
            or not string(result, 3)
        ):
            raise ChannelError("Yuanbao WebSocket authentication rejected")

    def ingest(self, data: bytes, store: ChannelStore) -> None:
        body = parse(data)
        callback = string(body, 1)
        if callback not in {"C2C.CallbackAfterSendMsg", "Group.CallbackAfterSendMsg"}:
            # The server also publishes PushMsg wrappers {cmd,module,msgId,data}.
            wrapped = binary(body, 4)
            if not wrapped:
                return
            body = parse(wrapped)
            callback = string(body, 1)
        if callback not in {"C2C.CallbackAfterSendMsg", "Group.CallbackAfterSendMsg"}:
            return
        sender, recipient = string(body, 2), string(body, 3)
        message_id = string(body, 12) or string(body, 11)
        if not sender or sender == self.bot_id or not message_id:
            return
        kind_code = number(body, 18)
        group = kind_code == 1 or (kind_code == 0 and callback == "Group.CallbackAfterSendMsg")
        target = string(body, 6) if group else sender
        origin = "" if group else string(body, 19)
        if not target or (not group and recipient != self.bot_id):
            return
        text, mentioned = [], False
        for raw in body.get(13, []):
            if not isinstance(raw, bytes):
                raise ValueError("Invalid Yuanbao message body")
            element = parse(raw)
            content = parse(binary(element, 2))
            if string(element, 1) == "TIMTextElem":
                text.append(string(content, 1))
            elif string(element, 1) == "TIMCustomElem":
                try:
                    custom = json.loads(string(content, 4))
                except ValueError:
                    continue
                if (
                    isinstance(custom, dict)
                    and custom.get("elem_type") == 1002
                    and custom.get("user_id") == self.bot_id
                ):
                    mentioned = True
        if not any(text):
            return
        thread = json.dumps(
            [self.bot_id, "group" if group else "dm", target, origin], separators=(",", ":")
        )
        self.accept(
            ChannelMessage(
                id=json.dumps([self.bot_id, target, message_id]),
                user_id=sender,
                channel_id=self.channel_id(thread),
                thread_id=thread,
                text="\n".join(text),
                group=group,
                mentioned=mentioned,
            ),
            store,
        )

    async def request(
        self, command: str, module: str, data: bytes = b"", *, identifier: str = ""
    ) -> Frame:
        if self.ws is None:
            raise ChannelError("Yuanbao is disconnected")
        identifier = identifier or uuid4().hex
        future = asyncio.get_running_loop().create_future()
        if identifier in self.pending:
            raise ChannelError("Yuanbao request is already in flight")
        self.pending[identifier] = (command, module, future)
        try:
            await self.ws.send(
                Frame(0, command, identifier, module, data, self.next_sequence()).encode()
            )
            reply = await asyncio.wait_for(future, 15)
            if reply.status == 50503:
                raise RateLimited(30)
            if reply.status:
                raise ChannelError("Yuanbao rejected the request")
            return reply
        finally:
            self.pending.pop(identifier, None)

    async def heartbeat(self) -> None:
        while True:
            reply = await self.request("ping", "conn_access")
            maximum = number(parse(reply.data), 1)
            if maximum:
                self.heartbeat_interval = max(1, min(20, maximum / 2))
            await asyncio.sleep(self.heartbeat_interval)

    async def receive(self, store: ChannelStore) -> None:
        # Every reconnect renews authentication, retaining the same bot identity.
        await self.sign()
        async with self.socket(WS_URL, domains=("yuanbao.tencent.com",)) as socket:
            await self.bind(socket)
            self.ws = socket
            try:
                async with asyncio.TaskGroup() as tasks:
                    heartbeat = tasks.create_task(self.heartbeat())
                    try:
                        async for raw in socket:
                            if not isinstance(raw, bytes):
                                raise ChannelError("Yuanbao sent a non-binary frame")
                            frame = Frame.decode(raw)
                            if frame.kind == 1:
                                pending = self.pending.get(frame.identifier)
                                if (
                                    pending
                                    and (frame.command, frame.module) == pending[:2]
                                    and not pending[2].done()
                                ):
                                    pending[2].set_result(frame)
                            elif frame.kind == 2:
                                if frame.command == "kickout":
                                    raise ChannelError("Yuanbao connection was revoked")
                                if frame.module == SERVICE:
                                    self.ingest(frame.data, store)
                                # ACK only after durable ingestion (or intentional policy rejection).
                                if frame.need_ack:
                                    await socket.send(
                                        Frame(
                                            3,
                                            frame.command,
                                            frame.identifier,
                                            frame.module,
                                            sequence=self.next_sequence(),
                                        ).encode()
                                    )
                    finally:
                        heartbeat.cancel()
            finally:
                self.ws = None
                for _, _, future in self.pending.values():
                    if not future.done():
                        future.set_exception(
                            ChannelError("Yuanbao request interrupted; outcome uncertain")
                        )

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        self.channel_id(thread_id)
        _, kind, target, origin = json.loads(thread_id)
        identifier = uuid5(NAMESPACE_URL, "harness:yuanbao:" + self.bot_id + ":" + delivery_id).hex
        message = field(1, "TIMTextElem") + field(2, field(1, text))
        body = field(1, identifier) + field(2, target) + field(3, self.bot_id)
        if kind == "group":
            body += field(5, str(int(identifier[:8], 16))) + field(6, message)
        else:
            body += field(4, int(identifier[:8], 16)) + field(5, message) + field(6, origin)
        reply = await self.request(
            "send_group_message" if kind == "group" else "send_c2c_message",
            SERVICE,
            body,
            identifier=identifier,
        )
        if number(parse(reply.data), 1):
            raise ChannelError("Yuanbao did not accept the outgoing message")
