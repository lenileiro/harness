"""Supervised Photon Spectrum SDK bridge over private inherited stdio pipes."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from harness.cli.channels.process import client_environment, stop_client
from harness.cli.channels.transports import ChannelError, Transport
from harness.core.gateway_channels import ChannelMessage, ChannelStore

BRIDGE = Path(__file__).with_name("photon_bridge") / "main.mjs"


class PhotonTransport(Transport):
    name = "photon"
    limit = 4000

    def __init__(self, *, process_factory: Any = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.process_factory = process_factory or asyncio.create_subprocess_exec
        self.process: asyncio.subprocess.Process | None = None
        self.pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self.write_lock = asyncio.Lock()

    async def authenticate(self) -> None:
        if not self.config.app_id:
            raise ChannelError(
                "Photon requires its project ID in app_id and project secret in token_env"
            )
        self.bot_id = self.config.app_id
        self.identity = "photon:" + self.bot_id
        await self.start()

    async def start(self) -> None:
        self.process = await self.process_factory(
            self.config.command or "node",
            str(BRIDGE),
            env=client_environment(
                {"PHOTON_PROJECT_ID": self.bot_id, "PHOTON_PROJECT_SECRET": self.token}
            ),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=2**20,
        )
        try:
            assert self.process and self.process.stdout
            ready = json.loads(await asyncio.wait_for(self.process.stdout.readline(), timeout=45))
            if ready.get("event") != "ready" or ready.get("project") != self.bot_id:
                raise ChannelError("Photon bridge did not authenticate the configured project")
        except BaseException:
            await self._stop()
            raise

    async def _stop(self) -> None:
        if self.process is not None:
            await stop_client(self.process)
        self.process = None

    async def close(self) -> None:
        await self._stop()
        await super().close()

    def can_send(self) -> bool:
        return self.process is not None and self.process.returncode is None

    def channel_id(self, thread_id: str) -> str:
        project, space = json.loads(thread_id)
        if project != self.bot_id:
            raise ChannelError("Photon destination belongs to a different project")
        return space

    async def receive(self, store: ChannelStore) -> None:
        if not self.can_send():
            await self.start()
        assert self.process and self.process.stdout
        try:
            while line := await self.process.stdout.readline():
                if len(line) > 2**20:
                    raise ChannelError("Photon bridge event exceeds its bound")
                frame = json.loads(line)
                future = self.pending.get(frame.get("id"))
                if future is not None and not future.done():
                    if frame.get("error"):
                        future.set_exception(
                            ChannelError("Photon send failed; outcome may be uncertain")
                        )
                    else:
                        future.set_result(frame.get("result", {}))
                elif frame.get("event") == "message":
                    self.accept(
                        ChannelMessage(
                            id=str(frame["id"]),
                            user_id=str(frame["sender"]),
                            channel_id=str(frame["space"]),
                            thread_id=json.dumps(
                                [self.bot_id, frame["space"]], separators=(",", ":")
                            ),
                            text=str(frame["text"]),
                            group=frame.get("group") is not False,
                            mentioned=False,
                        ),
                        store,
                    )
            raise ChannelError("Photon bridge stopped; the channel runner will reconnect")
        finally:
            for future in self.pending.values():
                if not future.done():
                    future.set_exception(
                        ChannelError("Photon bridge interrupted; outcome uncertain")
                    )
            await self._stop()

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        space = self.channel_id(thread_id)
        if not self.process or not self.process.stdin:
            raise ChannelError("Photon bridge is unavailable")
        request_id = uuid4().hex
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            async with self.write_lock:
                self.process.stdin.write(
                    (
                        json.dumps(
                            {"id": request_id, "method": "send", "space": space, "text": text}
                        )
                        + "\n"
                    ).encode()
                )
                await self.process.stdin.drain()
            result = await asyncio.wait_for(future, timeout=40)
            if not result.get("message_id"):
                raise ChannelError("Photon bridge omitted its send receipt")
        finally:
            self.pending.pop(request_id, None)
