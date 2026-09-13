"""Process handles backed by bounded state inside an SDK-owned sandbox."""

from __future__ import annotations

import asyncio
import base64
import contextlib
from typing import Any

from harness.tools.execution.process import ManagedProcess


class CloudManagedProcess(ManagedProcess):
    async def _rpc(self, operation: str, **arguments: Any) -> dict[str, Any]:
        assert self.backend.cloud is not None
        return await self.backend.cloud.rpc(
            {"operation": operation, "process_id": self.id, **arguments}
        )

    def _update(self, result: dict[str, Any]) -> None:
        self.pid = result["pid"]
        self.cwd = result["cwd"]
        self.output = bytearray(result["output"].encode())
        self.truncated = result["truncated"]
        self.reason = result["status"]
        self.exit_code = result["exit_code"]
        if self.pid is not None or not result["running"]:
            self.started.set()
        if not result["running"]:
            self.done.set()
        else:
            self.done.clear()

    async def start(
        self, *, command: str, cwd: str, timeout_seconds: float, pty: bool, shell: str | None = None
    ) -> None:
        if pty:
            raise ValueError("PTY is currently available only for the local backend")
        request = {
            "operation": "shell",
            "root": self.backend.root,
            "cwd": cwd,
            "command": command,
            "timeout": timeout_seconds,
            "pty": False,
            "shell": shell or self.backend.config.shell,
            "max_output_bytes": self.backend.config.max_output_bytes,
        }
        try:
            self._update(await self._rpc("process_start", request=request))
            self._reader = asyncio.create_task(self._monitor(timeout_seconds + 5))
            await asyncio.wait_for(self.started.wait(), 15)
            if self.pid is None:
                raise RuntimeError(
                    self.output.decode(errors="replace") or "cloud process failed to start"
                )
        except BaseException:
            await asyncio.shield(self.terminate())
            raise

    async def _monitor(self, timeout_seconds: float) -> None:
        try:
            async with asyncio.timeout(timeout_seconds):
                while not self.done.is_set():
                    self._update(await self._rpc("process_poll"))
                    if not self.done.is_set():
                        await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.reason = "transport_error"
            self._append(f"\ncloud process status unavailable: {exc}".encode())
            self.done.set()
            self.started.set()

    async def input(self, data: str) -> None:
        if self.done.is_set():
            raise ValueError("process has exited")
        raw = data.encode()
        if len(raw) > 65536:
            raise ValueError("input exceeds 65536-byte limit")
        self._update(await self._rpc("process_input", data=base64.b64encode(raw).decode()))

    async def terminate(self) -> None:
        async with self._terminate_lock:
            if self.done.is_set() and self.reason != "transport_error":
                return
            try:
                self._update(await self._rpc("process_terminate"))
                async with asyncio.timeout(5):
                    while not self.done.is_set():
                        self._update(await self._rpc("process_poll"))
                        if not self.done.is_set():
                            await asyncio.sleep(0.1)
            finally:
                if self._reader is not None and self._reader is not asyncio.current_task():
                    self._reader.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await self._reader


__all__ = ["CloudManagedProcess"]
