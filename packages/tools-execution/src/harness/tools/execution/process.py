"""Bounded output and interactive handles for backend-owned processes."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import uuid
from typing import Any

from harness.tools.execution.backend import ExecutionBackend, encode_request, stop_transport
from harness.tools.execution.transport import is_windows, spawn_transport


class ManagedProcess:
    def __init__(self, backend: ExecutionBackend) -> None:
        self.id = uuid.uuid4().hex
        self.backend = backend
        self.process: asyncio.subprocess.Process | None = None
        self.pid: int | None = None
        self.cwd: str | None = None
        self.output = bytearray()
        self.transport_error = bytearray()
        self.truncated = False
        self.exit_code: int | None = None
        self.reason = "running"
        self.started = asyncio.Event()
        self.done = asyncio.Event()
        self._reader: asyncio.Task[None] | None = None
        self._stderr: asyncio.Task[None] | None = None
        self._watchdog: asyncio.Task[None] | None = None
        self._terminate_lock = asyncio.Lock()

    def _append(self, data: bytes) -> None:
        cap = self.backend.config.max_output_bytes
        available = max(0, cap - len(self.output))
        self.output.extend(data[:available])
        self.truncated |= len(data) > available

    async def start(
        self, *, command: str, cwd: str, timeout_seconds: float, pty: bool, shell: str | None = None
    ) -> None:
        if pty and self.backend.config.backend != "local":
            raise ValueError("PTY is currently available only for the local backend")
        if pty and is_windows():
            raise ValueError("PTY is unavailable on native Windows; use process input without PTY")
        self.process = await spawn_transport(self.backend.worker_argv())
        self._stderr = asyncio.create_task(self._read_stderr())
        self._reader = asyncio.create_task(self._read())
        request = {
            "operation": "shell",
            "root": self.backend.root,
            "cwd": cwd,
            "command": command,
            "timeout": timeout_seconds,
            "pty": pty,
            "max_output_bytes": self.backend.config.max_output_bytes,
            "shell": (
                self.backend.config.windows_shell
                if is_windows() and self.backend.config.backend == "local"
                else shell or self.backend.config.shell
            ),
        }
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(encode_request(request))
            await self.process.stdin.drain()
            await asyncio.wait_for(self.started.wait(), 15)
            if self.pid is None:
                raise RuntimeError(
                    self.output.decode(errors="replace") or "process failed to start"
                )
            self._watchdog = asyncio.create_task(self._deadline(timeout_seconds + 3))
        except BaseException:
            await asyncio.shield(self.terminate())
            raise

    async def _read_stderr(self) -> None:
        assert self.process is not None and self.process.stderr is not None
        while chunk := await self.process.stderr.read(16384):
            self.transport_error.extend(chunk[: max(0, 4096 - len(self.transport_error))])

    async def _read(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        try:
            while line := await self.process.stdout.readline():
                event = json.loads(line)
                if event.get("event") == "started":
                    self.pid, self.cwd = event["pid"], event["cwd"]
                    self.started.set()
                elif event.get("event") == "output":
                    self._append(base64.b64decode(event["data"]))
                elif event.get("event") == "truncated":
                    self.truncated = True
                elif event.get("event") == "exit":
                    self.exit_code, self.reason = event["exit_code"], event["reason"]
                elif event.get("event") == "error":
                    self._append(event["message"].encode())
            await self.process.wait()
            if self._stderr is not None:
                await self._stderr
            if self.reason == "running":
                self.reason = "transport_error"
                self.exit_code = self.process.returncode
                self._append(bytes(self.transport_error))
        except Exception as exc:
            self.reason = "transport_error"
            self._append(str(exc).encode())
        finally:
            await stop_transport(self.process)
            self.started.set()
            self.done.set()

    async def _deadline(self, timeout_seconds: float) -> None:
        try:
            await asyncio.wait_for(self.done.wait(), timeout_seconds)
        except TimeoutError:
            self.reason = "timed_out"
            await self.terminate()

    async def input(self, data: str) -> None:
        if self.done.is_set() or self.process is None or self.process.stdin is None:
            raise ValueError("process has exited")
        raw = data.encode("utf-8")
        if len(raw) > 65536:
            raise ValueError("input exceeds 65536-byte limit")
        self.process.stdin.write(
            json.dumps(
                {
                    "action": "input",
                    "data": base64.b64encode(raw).decode("ascii"),
                }
            ).encode()
            + b"\n"
        )
        await asyncio.wait_for(self.process.stdin.drain(), 5)

    async def poll(self, wait_seconds: float = 0) -> dict[str, Any]:
        if wait_seconds and not self.done.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self.done.wait(), wait_seconds)
        return {
            "process_id": self.id,
            "backend": self.backend.config.backend,
            "pid": self.pid,
            "cwd": self.cwd,
            "status": self.reason,
            "running": not self.done.is_set(),
            "exit_code": self.exit_code,
            "output": self.output.decode("utf-8", errors="replace"),
            "truncated": self.truncated,
        }

    async def terminate(self) -> None:
        async with self._terminate_lock:
            if self.process is None or self.done.is_set():
                return
            if self.process.stdin is not None:
                with contextlib.suppress(BrokenPipeError, ConnectionResetError, TimeoutError):
                    self.process.stdin.write(b'{"action":"terminate"}\n')
                    await asyncio.wait_for(self.process.stdin.drain(), 1)
                self.process.stdin.close()
            try:
                await asyncio.wait_for(self.done.wait(), 3)
            except TimeoutError:
                await stop_transport(self.process)
                if self._reader is not None:
                    await self._reader
            if self._watchdog is not None and self._watchdog is not asyncio.current_task():
                self._watchdog.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._watchdog


__all__ = ["ManagedProcess"]
