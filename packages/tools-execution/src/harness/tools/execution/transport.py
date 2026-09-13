"""Owned local transport processes on POSIX and native Windows."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import signal
import sys
from typing import Any

from harness.tools.execution.windows_job import WindowsJob

_JOBS: dict[int, WindowsJob] = {}

# The gate reads one byte at a time, so it cannot prefetch the actual worker's
# request. No selected executable runs until its ancestor is assigned to a job.
_WINDOWS_GATE = """
import base64,json,os,subprocess,sys
line=bytearray()
while True:
    byte=os.read(0,1)
    if not byte: raise SystemExit(1)
    if byte==b'\\n': break
    line.extend(byte)
    if len(line)>16777216: raise SystemExit(1)
argv=json.loads(base64.b64decode(line))
raise SystemExit(subprocess.Popen(argv,stdin=sys.stdin,stdout=sys.stdout,stderr=sys.stderr).wait())
"""


def is_windows() -> bool:
    return os.name == "nt"


async def stop_transport(process: asyncio.subprocess.Process) -> None:
    """Reap the owned tree, including descendants after the root exits."""
    if is_windows():
        job = _JOBS.pop(process.pid, None)
        if job is not None:
            job.close()
        elif process.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                process.kill()
        await process.wait()
        return
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGTERM)
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(process.wait(), 0.3)
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(process.pid, signal.SIGKILL)
    await process.wait()


async def spawn_transport(argv: list[str], **kwargs: Any) -> asyncio.subprocess.Process:
    windows = is_windows()
    job = WindowsJob() if windows else None
    command = [sys.executable, "-u", "-c", _WINDOWS_GATE] if windows else argv
    creation = asyncio.create_task(
        asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=not windows,
            **kwargs,
        )
    )
    process: asyncio.subprocess.Process | None = None
    try:
        process = await asyncio.shield(creation)
        assert process is not None
        if job is not None:
            job.assign(process.pid)
            _JOBS[process.pid] = job
            assert process.stdin is not None
            process.stdin.write(base64.b64encode(json.dumps(argv).encode()) + b"\n")
            await process.stdin.drain()
        return process
    except BaseException:
        if process is None:
            with contextlib.suppress(Exception):
                process = await creation
        if process is not None:
            if windows and process.pid not in _JOBS and process.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    process.kill()  # Gate has not received any executable or payload.
            await asyncio.shield(stop_transport(process))
        if job is not None:
            job.close()
        raise


__all__ = ["is_windows", "spawn_transport", "stop_transport"]
