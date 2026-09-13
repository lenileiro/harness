"""Buzz's official authenticated CLI as a bounded durable polling transport."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlsplit

from harness.cli.channels.process import run_client
from harness.cli.channels.transports import ChannelError, Transport
from harness.core.gateway_channels import ChannelMessage, ChannelStore


class BuzzTransport(Transport):
    name = "buzz"
    limit = 4000

    def __init__(self, *, command_runner: Any = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.command_runner = command_runner or run_client

    async def command(self, args: list[str], *, text: str | None = None) -> Any:
        env = {"BUZZ_RELAY_URL": self.config.homeserver, "BUZZ_PRIVATE_KEY": self.token}
        if self.app_token:
            env["BUZZ_AUTH_TAG"] = self.app_token
        code, output = await self.command_runner(
            [self.config.command or "buzz", *args], env=env, input_text=text
        )
        if code:
            raise ChannelError(
                "Buzz client rejected the operation; inspect credentials or relay availability"
            )
        return json.loads(output)

    async def authenticate(self) -> None:
        parsed = urlsplit(self.config.homeserver)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ChannelError("Buzz requires an explicit HTTPS community relay")
        if not self.config.allowed_channels:
            raise ChannelError("Buzz requires explicit channel IDs for bounded polling")
        profiles = await self.command(["users", "get"])
        if not isinstance(profiles, list) or not profiles or not profiles[0].get("pubkey"):
            raise ChannelError("Buzz client did not verify a relay member identity")
        self.bot_id = str(profiles[0]["pubkey"]).lower()
        self.username = str(profiles[0].get("display_name", ""))
        self.identity = json.dumps([self.config.homeserver, self.bot_id])
        channels = await self.command(["channels", "list"])
        self.channels = {str(channel.get("id", "")): channel for channel in channels}
        if any(channel not in self.channels for channel in self.config.allowed_channels):
            raise ChannelError("Buzz configured channels are not accessible to this identity")

    def channel_id(self, thread_id: str) -> str:
        channel, _ = json.loads(thread_id)
        return channel

    def ingest(self, channel: str, event: dict[str, Any], store: ChannelStore) -> None:
        sender = str(event.get("pubkey", "")).lower()
        if (
            event.get("kind") not in {9, 45001, 45003}
            or not sender
            or sender == self.bot_id
            or not isinstance(event.get("content"), str)
        ):
            return
        tags = event.get("tags", [])
        mentioned = any(len(tag) >= 2 and tag[0] == "p" and tag[1] == self.bot_id for tag in tags)
        parents = [tag[1] for tag in tags if len(tag) >= 4 and tag[0] == "e" and tag[3] == "root"]
        root = parents[0] if parents else str(event["id"])
        text = event["content"]
        if mentioned and self.username:
            prefix = "@" + self.username
            if text.startswith(prefix) and (
                len(text) == len(prefix) or text[len(prefix)].isspace()
            ):
                text = text[len(prefix) :].strip()
        # Public-key identity is verified by the official CLI/relay. Channel type
        # defaults to group rather than inferring DM status from content or tags.
        group = self.channels[channel].get("type") not in {"dm", "direct"}
        if not group:
            root = ""
        self.accept(
            ChannelMessage(
                id=str(event["id"]),
                user_id=sender,
                channel_id=channel,
                thread_id=json.dumps([channel, root], separators=(",", ":")),
                text=text,
                group=group,
                mentioned=mentioned,
            ),
            store,
        )

    async def poll_once(self, store: ChannelStore) -> None:
        for channel in self.config.allowed_channels:
            key = "since:" + channel
            since = store.get(key)
            args = ["messages", "get", "--channel", channel, "--limit", "200"]
            if since is not None:
                args += ["--since", str(since)]
            events = await self.command(args)
            if not isinstance(events, list):
                raise ChannelError("Buzz client returned malformed message history")
            if len(events) >= 200 and since is not None:
                store.set("connection", "history_gap")
                raise ChannelError(
                    "Buzz history reached its 200-event recovery bound; cursor preserved"
                )
            for event in sorted(
                events, key=lambda item: (item.get("created_at", 0), item.get("id", ""))
            ):
                if since is not None:
                    self.ingest(channel, event, store)
            store.set(
                key, max([since or 0, *(int(event.get("created_at", 0)) for event in events)])
            )

    async def receive(self, store: ChannelStore) -> None:
        while True:
            await self.poll_once(store)
            await asyncio.sleep(4)

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        channel, root = json.loads(thread_id)
        if channel not in self.config.allowed_channels:
            raise ChannelError("Buzz reply is outside configured channels")
        args = ["messages", "send", "--channel", channel, "--content", "-"]
        if root:
            args += ["--reply-to", root]
        response = await self.command(args, text=text)
        if (
            not isinstance(response, dict)
            or response.get("accepted") is not True
            or not response.get("event_id")
        ):
            raise ChannelError("Buzz client did not return an accepted event receipt")
