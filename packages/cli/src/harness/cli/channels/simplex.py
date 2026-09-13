"""SimpleX Chat's local daemon WebSocket API, with stable numeric destinations."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

from harness.cli.channels.transports import ChannelError, Transport
from harness.core.gateway_channels import ChannelMessage, ChannelStore


class SimpleXTransport(Transport):
    name = "simplex"
    limit = 4000

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.ws: Any = None
        self.pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self.initial_events: list[dict[str, Any]] = []

    def connection(self):
        parsed = urlsplit(self.config.homeserver)
        if (
            parsed.scheme != "ws"
            or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ChannelError("SimpleX daemon must use a local loopback ws:// URL")
        return self.socket_connect(
            self.config.homeserver,
            max_size=2**20,
            open_timeout=10,
            logger=self.socket_logger,
            proxy=None,
        )

    async def _identify(self, socket) -> list[dict[str, Any]]:
        request_id = uuid4().hex
        await socket.send(json.dumps({"corrId": request_id, "cmd": "/user"}))
        buffered = []
        async with asyncio.timeout(15):
            for _ in range(1000):
                envelope = json.loads(await socket.recv())
                if envelope.get("corrId") == request_id:
                    result = envelope.get("resp", {})
                    user_id = str(result.get("user", {}).get("userId", ""))
                    if result.get("type") != "activeUser" or user_id != self.config.username:
                        raise ChannelError(
                            "SimpleX active user differs from the configured dedicated daemon profile"
                        )
                    self.bot_id = user_id
                    self.identity = json.dumps([self.config.homeserver, user_id])
                    return buffered
                buffered.append(envelope)
        raise ChannelError("SimpleX identity acknowledgment was not received")

    async def authenticate(self) -> None:
        if not self.config.username.isdigit():
            raise ChannelError(
                "SimpleX username must name the dedicated daemon's numeric active user ID"
            )
        async with self.connection() as socket:
            self.initial_events.extend(await self._identify(socket))

    def can_send(self) -> bool:
        return self.ws is not None

    def channel_id(self, thread_id: str) -> str:
        profile, kind, identifier = json.loads(thread_id)
        if (
            profile != self.bot_id
            or kind not in {"direct", "group"}
            or not str(identifier).isdigit()
        ):
            raise ChannelError(
                "SimpleX destination does not belong to the dedicated daemon profile"
            )
        return ("group:" if kind == "group" else "contact:") + str(identifier)

    def ingest(self, envelope: dict[str, Any], store: ChannelStore) -> None:
        response = envelope.get("resp", envelope)
        if str(response.get("user", {}).get("userId", self.bot_id)) != self.bot_id:
            return
        if response.get("type") == "newChatItems":
            items = response.get("chatItems", [])
        elif response.get("type") == "newChatItem":
            items = [response]
        else:
            return
        for item in items:
            info, message = item.get("chatInfo", {}), item.get("chatItem", {})
            content = message.get("content", {})
            if (
                content.get("type") != "rcvMsgContent"
                or content.get("msgContent", {}).get("type") != "text"
            ):
                continue
            direction = message.get("chatDir", {})
            kind = info.get("type")
            if kind == "direct" and direction.get("type") == "directRcv":
                identifier = str(info.get("contact", {}).get("contactId", ""))
                sender = "contact:" + identifier
            elif kind == "group" and direction.get("type") == "groupRcv":
                identifier = str(info.get("groupInfo", {}).get("groupId", ""))
                member = str(direction.get("groupMember", {}).get("memberId", ""))
                if not member:
                    continue
                sender = "group:" + identifier + ":member:" + member
            else:
                continue
            item_id = message.get("meta", {}).get("itemId")
            if not identifier.isdigit() or item_id is None:
                continue
            thread = json.dumps([self.bot_id, kind, identifier], separators=(",", ":"))
            # SimpleX's internal member IDs are scoped to a group; never confuse
            # one with a direct-contact identity from a different relationship.
            self.accept(
                ChannelMessage(
                    id=json.dumps([self.bot_id, kind, identifier, item_id]),
                    user_id=sender,
                    channel_id=self.channel_id(thread),
                    thread_id=thread,
                    text=str(content["msgContent"].get("text", "")),
                    group=kind == "group",
                    mentioned=bool(message.get("meta", {}).get("userMention")),
                ),
                store,
            )

    async def receive(self, store: ChannelStore) -> None:
        async with self.connection() as socket:
            buffered = await self._identify(socket)
            self.ws = socket
            try:
                for envelope in self.initial_events + buffered:
                    self.ingest(envelope, store)
                self.initial_events.clear()
                async for raw in socket:
                    envelope = json.loads(raw)
                    request_id = envelope.get("corrId")
                    future = self.pending.get(request_id)
                    if future is not None and not future.done():
                        future.set_result(envelope.get("resp", {}))
                    else:
                        self.ingest(envelope, store)
            finally:
                self.ws = None
                for future in self.pending.values():
                    if not future.done():
                        future.set_exception(
                            ChannelError("SimpleX command interrupted; outcome uncertain")
                        )

    async def command(self, text: str) -> dict[str, Any]:
        if self.ws is None:
            raise ChannelError("SimpleX daemon is disconnected")
        identifier = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[identifier] = future
        try:
            await self.ws.send(json.dumps({"corrId": identifier, "cmd": text}))
            return await asyncio.wait_for(future, timeout=20)
        finally:
            self.pending.pop(identifier, None)

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        self.channel_id(thread_id)
        user = await self.command("/user")
        if (
            user.get("type") != "activeUser"
            or str(user.get("user", {}).get("userId")) != self.bot_id
        ):
            raise ChannelError("SimpleX daemon profile changed; outgoing message refused")
        _, kind, identifier = json.loads(thread_id)
        target = ("#" if kind == "group" else "@") + identifier
        response = await self.command(
            "/_send "
            + target
            + " json "
            + json.dumps([{"msgContent": {"type": "text", "text": text}}])
        )
        if response.get("type") not in {"newChatItems", "newChatItem"}:
            raise ChannelError("SimpleX daemon did not acknowledge a sent chat item")
