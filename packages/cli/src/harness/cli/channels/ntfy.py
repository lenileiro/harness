"""Authenticated ntfy polling with a signed single-owner command envelope."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from typing import Any
from urllib.parse import quote, urlsplit
from uuid import uuid4

from harness.cli.channels.transports import ChannelError, RateLimited, Transport
from harness.core.gateway_channels import ChannelMessage, ChannelStore


def sign_ntfy_message(
    *,
    text: str,
    topic: str,
    owner: str,
    secret: str,
    message_id: str | None = None,
    timestamp: int | None = None,
) -> str:
    """Serialize an owner-authenticated command for a standard ntfy publisher."""
    payload = {
        "version": 1,
        "id": message_id or uuid4().hex,
        "timestamp": timestamp if timestamp is not None else int(time.time()),
        "topic": topic,
        "owner": owner,
        "text": text,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    signature = hmac.new(secret.encode(), encoded.encode(), hashlib.sha256).hexdigest()
    envelope = json.dumps(
        {"payload": payload, "signature": signature}, separators=(",", ":"), ensure_ascii=False
    )
    if len(envelope.encode()) > 4096:
        raise ChannelError("Signed ntfy command exceeds the platform's 4096-byte message limit")
    return envelope


class NtfyTransport(Transport):
    name = "ntfy"
    limit = 1000

    @property
    def base(self) -> str:
        value = self.config.homeserver.rstrip("/")
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ChannelError("ntfy requires an operator-configured HTTPS homeserver")
        return value

    async def authenticate(self) -> None:
        topics = (self.config.topic, self.config.reply_topic)
        if (
            not self.config.username
            or self.config.allowed_users != [self.config.username]
            or len(self.app_token.encode()) < 32
        ):
            raise ChannelError(
                "ntfy requires one configured owner and an app_token_env signing secret of at least 32 bytes"
            )
        if topics[0] == topics[1] or any(
            not topic
            or len(topic) > 64
            or any(
                char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for char in topic
            )
            for topic in topics
        ):
            raise ChannelError("ntfy requires distinct valid topic and reply_topic names")
        self.identity = json.dumps([self.base, *topics, self.config.username])
        self.bot_id = self.config.username
        # The service authenticates the subscription token during the first poll.
        # Publisher identity is independently verified by the application signature.

    def channel_id(self, thread_id: str) -> str:
        if thread_id != self.config.topic:
            raise ChannelError("ntfy destination is outside the configured owner topic")
        return thread_id

    def ingest(self, event: dict[str, Any], store: ChannelStore) -> None:
        if event.get("event") != "message" or event.get("topic") != self.config.topic:
            return
        try:
            envelope = json.loads(event["message"])
            payload = envelope["payload"]
            canonical = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )
            expected = hmac.new(
                self.app_token.encode(), canonical.encode(), hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(expected, envelope["signature"]):
                return
            if (
                payload["version"] != 1
                or payload["topic"] != self.config.topic
                or payload["owner"] != self.config.username
                or type(payload["timestamp"]) is not int
                or not -300 <= time.time() - payload["timestamp"] <= 86400
                or not isinstance(payload["id"], str)
                or not 1 <= len(payload["id"]) <= 128
                or not isinstance(payload["text"], str)
            ):
                return
            self.accept(
                ChannelMessage(
                    id=payload["id"],
                    user_id=self.config.username,
                    channel_id=self.config.topic,
                    thread_id=self.config.topic,
                    text=payload["text"],
                ),
                store,
            )
        except (ValueError, TypeError, KeyError):
            return

    async def poll_once(self, store: ChannelStore) -> None:
        cursor = store.get("since")
        baseline = str(int(time.time()))
        params = {"poll": "1", "since": cursor or "latest"}
        async with self.client.stream(
            "GET",
            self.base + "/" + quote(self.config.topic, safe="") + "/json",
            params=params,
            headers={"Authorization": "Bearer " + self.token},
        ) as response:
            if response.status_code == 429:
                raise RateLimited(float(response.headers.get("Retry-After", "60")))
            if not response.is_success:
                raise ChannelError("ntfy subscription authentication or polling failed")
            if response.headers.get("X-Messages-Truncated") == "1":
                store.set("connection", "history_gap")
                raise ChannelError(
                    "ntfy cache replay was truncated; the saved cursor was preserved"
                )
            buffer = b""
            total = 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > 10 * 1024 * 1024:
                    raise ChannelError("ntfy replay exceeds the bounded response size")
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if len(line) > 65536:
                        raise ChannelError("ntfy event exceeds the bounded message size")
                    event = json.loads(line)
                    if cursor:
                        self.ingest(event, store)
                    if event.get("event") == "message":
                        baseline = str(event["id"])
                        if cursor:
                            store.set("since", baseline)
                if len(buffer) > 65536:
                    raise ChannelError("ntfy event exceeds the bounded message size")
            if buffer.strip():
                raise ChannelError("ntfy response ended with an incomplete JSON line")
        if not cursor:
            store.set("since", baseline)

    async def receive(self, store: ChannelStore) -> None:
        while True:
            await self.poll_once(store)
            await asyncio.sleep(5)

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        self.channel_id(thread_id)
        await self.api(
            "POST",
            self.base + "/",
            data={"topic": self.config.reply_topic, "message": text},
            auth="Bearer " + self.token,
        )
