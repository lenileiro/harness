"""IRCv3 TLS/SASL bot with server-authenticated account ownership.

Only explicitly configured channels are destinations. Nick-addressed private
replies are disabled because a nickname can change owners while work is queued.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import re
import ssl
from contextlib import suppress
from typing import Any
from urllib.parse import urlsplit

from harness.cli.channels.transports import ChannelError, Transport
from harness.core.gateway_channels import ChannelMessage, ChannelStore

CAPABILITIES = {"sasl", "account-tag", "server-time", "message-tags"}


def parse_line(line: str) -> tuple[dict[str, str], str, str, list[str]]:
    tags: dict[str, str] = {}
    if line.startswith("@"):
        raw, _, line = line.partition(" ")
        escapes = {":": ";", "s": " ", "\\": "\\", "r": "\r", "n": "\n"}
        for tag in raw[1:].split(";"):
            key, _, value = tag.partition("=")
            tags[key] = re.sub(r"\\(.)", lambda match: escapes.get(match[1], match[1]), value)
    prefix = ""
    if line.startswith(":"):
        prefix, _, line = line[1:].partition(" ")
    head, separator, tail = line.partition(" :")
    parts = head.split()
    if not parts:
        raise ChannelError("IRC server sent a malformed line")
    return tags, prefix, parts[0].upper(), parts[1:] + ([tail] if separator else [])


def irc_casefold(value: str) -> str:
    return value.lower().translate(str.maketrans("[]\\^", "{}|~"))


class IRCTransport(Transport):
    name = "irc"
    limit = 100

    def __init__(self, *, stream_connect: Any = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.stream_connect = stream_connect or asyncio.open_connection
        self.writer: asyncio.StreamWriter | None = None
        self.reader: asyncio.StreamReader | None = None
        self.ready = False
        self.buffered: list[str] = []
        self.write_lock = asyncio.Lock()

    async def _write(self, line: str) -> None:
        if "\r" in line or "\n" in line or "\0" in line or len(line.encode()) > 510:
            raise ChannelError("IRC outgoing command exceeds its safe framing")
        async with self.write_lock:
            if self.writer is None or self.writer.is_closing():
                raise ChannelError("IRC connection is unavailable")
            self.writer.write((line + "\r\n").encode())
            await self.writer.drain()

    async def _read(self) -> str:
        assert self.reader is not None
        raw = await self.reader.readline()
        if not raw:
            raise ChannelError("IRC connection closed")
        if len(raw) > 8704 or not raw.endswith(b"\r\n"):
            raise ChannelError("IRC incoming frame exceeds its bound or framing")
        return raw[:-2].decode("utf-8", errors="replace")

    async def authenticate(self) -> None:
        parsed = urlsplit(self.config.homeserver)
        if (
            parsed.scheme != "ircs"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ChannelError("IRC homeserver must be an ircs://host:port TLS address")
        if (
            not self.config.username
            or not self.config.app_id
            or any(
                char.isspace() or char in ":\0\r\n"
                for char in self.config.username + self.config.app_id
            )
            or len(self.config.app_id) > 30
            or "\0" in self.token
        ):
            raise ChannelError("IRC requires a valid SASL username and app_id nickname")
        if not self.config.allowed_channels or any(
            not channel.startswith(("#", "&"))
            or len(channel.encode()) > 50
            or any(char.isspace() or char in ",:\0\r\n" for char in channel)
            for channel in self.config.allowed_channels
        ):
            raise ChannelError(
                "IRC requires explicit safe channel destinations; nickname DMs are unsupported"
            )
        self.server = f"{parsed.hostname}:{parsed.port or 6697}"
        self.bot_id = self.config.app_id
        self.identity = json.dumps([self.server, self.config.username])
        self.ready = False
        self.reader, self.writer = await self.stream_connect(
            parsed.hostname,
            parsed.port or 6697,
            ssl=ssl.create_default_context(),
            server_hostname=parsed.hostname,
            limit=8704,
        )
        try:
            async with asyncio.timeout(60):
                await self._write("CAP LS 302")
                await self._write("NICK " + self.bot_id)
                await self._write("USER " + self.bot_id + " 0 * :Harness")
                offered: set[str] = set()
                acked: set[str] = set()
                sasl_started = authenticated = False
                while True:
                    _, _, command, params = parse_line(await self._read())
                    if command == "PING":
                        await self._write("PONG :" + params[-1])
                    elif command == "CAP" and "LS" in params:
                        offered.update(cap.split("=", 1)[0] for cap in params[-1].split())
                        if len(params) >= 3 and params[-2] == "*":
                            continue
                        if not CAPABILITIES.issubset(offered):
                            raise ChannelError(
                                "IRC server lacks required SASL/account-tag/time/message-tag capabilities"
                            )
                        await self._write("CAP REQ :" + " ".join(sorted(CAPABILITIES)))
                    elif command == "CAP" and "ACK" in params:
                        acked.update(params[-1].split())
                        if CAPABILITIES.issubset(acked) and not sasl_started:
                            sasl_started = True
                            await self._write("AUTHENTICATE PLAIN")
                    elif command == "AUTHENTICATE" and params == ["+"] and sasl_started:
                        encoded = base64.b64encode(
                            (
                                self.config.username
                                + "\0"
                                + self.config.username
                                + "\0"
                                + self.token
                            ).encode()
                        ).decode()
                        self.secrets.append(encoded)
                        for index in range(0, len(encoded), 400):
                            await self._write("AUTHENTICATE " + encoded[index : index + 400])
                        if len(encoded) % 400 == 0:
                            await self._write("AUTHENTICATE +")
                    elif command == "903":
                        authenticated = True
                        await self._write("CAP END")
                    elif command == "001":
                        if not authenticated:
                            raise ChannelError(
                                "IRC registered without successful SASL authentication"
                            )
                        self.bot_id = params[0]
                        break
                    elif command in {"904", "905", "906", "907", "433", "464", "ERROR"} or (
                        command == "CAP" and "NAK" in params
                    ):
                        raise ChannelError("IRC authentication or capability negotiation failed")
                for channel in self.config.allowed_channels:
                    await self._write("JOIN " + channel)
                joined: set[str] = set()
                expected_channels = {
                    irc_casefold(channel) for channel in self.config.allowed_channels
                }
                self.buffered = []
                while joined != expected_channels:
                    line = await self._read()
                    _, prefix, command, params = parse_line(line)
                    if command == "PING":
                        await self._write("PONG :" + params[-1])
                    elif command == "JOIN" and irc_casefold(
                        prefix.split("!", 1)[0]
                    ) == irc_casefold(self.bot_id):
                        joined.add(irc_casefold(params[0]))
                    elif command in {
                        "403",
                        "405",
                        "471",
                        "473",
                        "474",
                        "475",
                        "476",
                        "477",
                        "489",
                        "ERROR",
                    }:
                        raise ChannelError("IRC could not join the configured channel")
                    elif command == "PRIVMSG":
                        if len(self.buffered) >= 1000:
                            raise ChannelError("IRC registration message buffer exceeded its bound")
                        self.buffered.append(line)
                self.ready = True
        except BaseException:
            await self._disconnect()
            raise

    async def _disconnect(self) -> None:
        self.ready = False
        if self.writer is not None:
            self.writer.close()
            with suppress(OSError, ssl.SSLError):
                await self.writer.wait_closed()
        self.writer = None

    async def close(self) -> None:
        await self._disconnect()
        await super().close()

    def can_send(self) -> bool:
        return self.ready

    def ingest(self, line: str, store: ChannelStore) -> None:
        tags, prefix, command, params = parse_line(line)
        if command != "PRIVMSG" or len(params) != 2 or not prefix or "!" not in prefix:
            return
        target, text = params
        channels = {irc_casefold(channel): channel for channel in self.config.allowed_channels}
        channel = channels.get(irc_casefold(target))
        account = tags.get("account", "")
        if (
            not channel
            or not account
            or account == "*"
            or text.startswith("\x01")
            or irc_casefold(prefix.split("!", 1)[0]) == irc_casefold(self.bot_id)
        ):
            return
        identifier = tags.get("msgid")
        if not identifier:
            # A bouncer may replay older messages without msgid; a server timestamp
            # and complete original frame form a stable conservative dedup key.
            if not tags.get("time"):
                return
            identifier = hashlib.sha256(line.encode()).hexdigest()
        pattern = r"(?<![\w\[\]{}|\\-])" + re.escape(self.bot_id) + r"(?![\w\[\]{}|\\-])"
        mentioned = re.search(pattern, text, flags=re.IGNORECASE) is not None
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).lstrip(":, ")
        self.accept(
            ChannelMessage(
                id=identifier,
                user_id=self.server + ":" + irc_casefold(account),
                channel_id=channel,
                thread_id=channel,
                text=text,
                group=True,
                mentioned=mentioned,
            ),
            store,
        )

    async def receive(self, store: ChannelStore) -> None:
        if not self.ready:
            await self.authenticate()
        try:
            for line in self.buffered:
                self.ingest(line, store)
            self.buffered = []
            while True:
                line = await self._read()
                _, _, command, params = parse_line(line)
                if command == "PING":
                    await self._write("PONG :" + params[-1])
                elif command == "ERROR":
                    raise ChannelError("IRC server ended the connection")
                elif command == "NICK":
                    _, prefix, _, _ = parse_line(line)
                    if irc_casefold(prefix.split("!", 1)[0]) == irc_casefold(self.bot_id):
                        self.bot_id = params[0]
                else:
                    self.ingest(line, store)
        finally:
            await self._disconnect()

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        if thread_id not in self.config.allowed_channels:
            raise ChannelError("IRC reply is outside configured channel destinations")
        # No arbitrary IRC commands or CTCP frames may be introduced by model text.
        text = "".join(
            char if ord(char) >= 32 else ("↵" if char in "\r\n" else " ") for char in text
        )
        await self._write("PRIVMSG " + thread_id + " :" + text)
