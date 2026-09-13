"""Bounded argv-only execution of an explicitly configured channel client."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Sequence
from contextlib import suppress

from harness.cli.channels.transports import ChannelError


def client_environment(values: dict[str, str]) -> dict[str, str]:
    # Client credentials are explicit; model/API/channel secrets from the parent
    # process are not handed to unrelated channel subprocesses.
    inherited = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        in {
            "PATH",
            "HOME",
            "USERPROFILE",
            "APPDATA",
            "LOCALAPPDATA",
            "SYSTEMROOT",
            "WINDIR",
            "TEMP",
            "TMP",
            "LANG",
            "LC_ALL",
        }
    }
    return {**inherited, **values}


async def stop_client(process: asyncio.subprocess.Process) -> None:
    if process.returncode is None:
        with suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=5)
        except TimeoutError:
            with suppress(ProcessLookupError):
                process.kill()
            await process.wait()


async def run_client(
    argv: Sequence[str],
    *,
    env: dict[str, str],
    input_text: str | None = None,
    deadline_seconds: float = 30,
    output_limit: int = 2**20,
) -> tuple[int, str]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if input_text is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=client_environment(env),
    )

    async def read(stream: asyncio.StreamReader | None) -> bytes:
        assert stream is not None
        output = bytearray()
        while chunk := await stream.read(65536):
            output.extend(chunk)
            if len(output) > output_limit:
                raise ChannelError("Channel client output exceeded its configured bound")
        return bytes(output)

    async def write() -> None:
        if process.stdin is not None:
            process.stdin.write((input_text or "").encode())
            await process.stdin.drain()
            process.stdin.close()

    try:
        async with asyncio.timeout(deadline_seconds):
            async with asyncio.TaskGroup() as group:
                stdout = group.create_task(read(process.stdout))
                group.create_task(
                    read(process.stderr)
                )  # Drain but never expose credential-bearing errors.
                group.create_task(write())
                group.create_task(process.wait())
        return process.returncode or 0, stdout.result().decode("utf-8", errors="replace")
    finally:
        await stop_client(process)
