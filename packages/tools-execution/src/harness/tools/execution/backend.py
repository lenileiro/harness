"""Owned backend lifecycle and argument-safe command transports."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shlex
import sys
import uuid
from pathlib import Path
from typing import Any

from harness.tools.execution.cloud import CLOUD_BACKENDS, CloudSandbox
from harness.tools.execution.config import ExecutionConfig
from harness.tools.execution.transport import is_windows, spawn_transport, stop_transport

_WORKER = Path(__file__).with_name("worker.py").read_text(encoding="utf-8")


def encode_request(request: dict[str, Any]) -> bytes:
    return base64.b64encode(json.dumps(request).encode("utf-8")) + b"\n"


async def transport_command(
    argv: list[str], *, payload: bytes | None = None, timeout_seconds: float = 30
) -> tuple[int, bytes, bytes]:
    process = await spawn_transport(argv)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(payload), timeout_seconds)
        return process.returncode or 0, stdout, stderr
    finally:
        await asyncio.shield(stop_transport(process))


class ExecutionBackend:
    def __init__(self, config: ExecutionConfig, cwd: Path) -> None:
        self.config = config
        self.host_cwd = cwd.resolve()
        self.root = str(self.host_cwd) if config.backend == "local" else "/workspace"
        if config.backend == "ssh":
            assert config.ssh_cwd is not None
            self.root = config.ssh_cwd
        self.container_name: str | None = None
        self.active = False
        self.cloud = CloudSandbox(config) if config.backend in CLOUD_BACKENDS else None
        if self.cloud is not None:
            self.root = self.cloud.root

    def worker_argv(self) -> list[str]:
        if not self.active:
            raise RuntimeError("execution backend is not open")
        if self.cloud is not None:
            raise RuntimeError("cloud backend requires its SDK transport")
        if self.config.backend == "local":
            return [sys.executable, "-u", "-c", _WORKER]
        remote = [self.config.python_binary, "-u", "-c", _WORKER]
        if self.config.backend == "docker":
            assert self.container_name is not None
            return [self.config.docker_binary, "exec", "-i", self.container_name, *remote]
        if self.config.backend == "singularity":
            return [
                self.config.singularity_binary,
                "exec",
                "--containall",
                "--cleanenv",
                "--bind",
                f"{self.host_cwd}:/workspace",
                "--pwd",
                "/workspace",
                self.config.singularity_image or "",
                *remote,
            ]
        assert self.config.ssh_host is not None
        destination = self.config.ssh_host
        if self.config.ssh_user:
            destination = f"{self.config.ssh_user}@{destination}"
        return [
            self.config.ssh_binary,
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
            "-p",
            str(self.config.ssh_port),
            "--",
            destination,
            shlex.join(remote),
        ]

    async def open(self) -> None:
        if self.active:
            raise RuntimeError("execution backend already open")
        if os.name not in {"posix", "nt"} and self.cloud is None:
            raise RuntimeError("managed execution requires a POSIX or Windows host")
        if is_windows() and self.config.backend == "local":
            shell = self.config.windows_shell
            if not shell or not await asyncio.to_thread(Path(shell).is_file):
                raise ValueError(
                    "native Windows execution requires an explicit windows_shell Bash executable path"
                )
        try:
            if self.cloud is not None:
                await self.cloud.open()
            if self.config.backend == "singularity" and (
                not self.host_cwd.is_dir() or any(char in str(self.host_cwd) for char in ":,")
            ):
                raise ValueError(
                    "Singularity workspace must be an existing path without ':' or ','"
                )
            if self.config.backend == "docker":
                if not self.host_cwd.is_dir():
                    raise ValueError("Docker bind-mount workspace must be an existing directory")
                if "," in str(self.host_cwd):
                    raise ValueError("Docker bind-mount workspace cannot contain ','")
                self.container_name = f"harness-execution-{uuid.uuid4().hex}"
                code, _, error = await transport_command(
                    [
                        self.config.docker_binary,
                        "run",
                        "--detach",
                        "--init",
                        "--name",
                        self.container_name,
                        "--label",
                        "harness.execution=owned",
                        "--network",
                        self.config.docker_network,
                        "--mount",
                        f"type=bind,src={self.host_cwd},dst=/workspace",
                        "--workdir",
                        "/workspace",
                        "--entrypoint",
                        self.config.python_binary,
                        self.config.docker_image or "",
                        "-u",
                        "-c",
                        "import time; time.sleep(2147483647)",
                    ],
                    timeout_seconds=120,
                )
                if code:
                    raise RuntimeError(
                        f"Docker startup failed: {error.decode(errors='replace')[:4096]}"
                    )
            self.active = True
            result = await self.files("probe", {})
            self.root = result["metadata"]["cwd"]
        except BaseException:
            await asyncio.shield(self.close())
            raise

    async def files(self, operation: str, arguments: dict[str, Any]) -> dict[str, Any]:
        request = {
            "operation": operation,
            "root": self.root,
            "arguments": arguments,
            "max_file_bytes": self.config.max_file_bytes,
        }
        if self.cloud is not None:
            if not self.active:
                raise RuntimeError("execution backend is not open")
            return await self.cloud.rpc(request)
        code, output, error = await transport_command(
            self.worker_argv(),
            payload=encode_request(request),
            timeout_seconds=min(30, self.config.timeout_seconds),
        )
        try:
            result = json.loads(output)
        except (ValueError, UnicodeDecodeError):
            raise RuntimeError(
                f"{self.config.backend} transport failed ({code}): {error.decode(errors='replace')[:4096]}"
            ) from None
        if code or result.get("event") != "result":
            raise RuntimeError(result.get("message", f"{self.config.backend} operation failed"))
        return result

    async def close(self) -> None:
        self.active = False
        if self.cloud is not None:
            await self.cloud.close()
        if self.container_name is not None:
            name, self.container_name = self.container_name, None
            code, _, error = await transport_command(
                [self.config.docker_binary, "rm", "--force", name],
                timeout_seconds=30,
            )
            if code and not any(
                message in error.lower() for message in (b"no such container", b"no such object")
            ):
                raise RuntimeError(
                    f"Docker cleanup failed for owned container {name}: {error.decode(errors='replace')[:4096]}"
                )


__all__ = ["ExecutionBackend", "encode_request", "stop_transport"]
