"""Optional provider SDK transports; only their owned sandbox is ever mutated."""

from __future__ import annotations

import asyncio
import importlib
import json
import math
import shlex
import uuid
from pathlib import Path
from typing import Any

from harness.tools.execution.config import ExecutionConfig

CLOUD_BACKENDS = frozenset({"modal", "daytona", "vercel_sandbox"})


class CloudSandbox:
    def __init__(self, config: ExecutionConfig) -> None:
        self.config = config
        self.box: Any = None
        self.client: Any = None
        self.name = f"harness-execution-{uuid.uuid4().hex}"
        self.control_root = f"/tmp/{self.name}"
        self.root = config.cloud_cwd

    async def _create(self) -> None:
        name = {"modal": "modal", "daytona": "daytona", "vercel_sandbox": "vercel.sandbox"}[
            self.config.backend
        ]
        try:
            sdk = importlib.import_module(name)
        except ImportError as exc:
            extra = "vercel" if self.config.backend == "vercel_sandbox" else self.config.backend
            raise RuntimeError(
                f"{self.config.backend} backend needs the tools-execution[{extra}] optional dependencies"
            ) from exc
        lifetime = self.config.cloud_lifetime_seconds
        if self.config.backend == "modal":
            app = await sdk.App.lookup.aio(self.config.modal_app, create_if_missing=True)
            image = sdk.Image.from_registry(self.config.modal_image)
            self.box = await sdk.Sandbox.create.aio(
                self.config.python_binary,
                "-c",
                f"import time; time.sleep({lifetime})",
                app=app,
                image=image,
                timeout=lifetime,
                name=self.name,
                tags={"harness.execution": "owned"},
            )
        elif self.config.backend == "daytona":
            self.client = sdk.AsyncDaytona()
            params = sdk.CreateSandboxFromSnapshotParams(
                name=self.name,
                language="python",
                ephemeral=True,
                snapshot=self.config.daytona_snapshot,
                labels={"harness.execution": "owned", "harness.execution.id": self.name},
                auto_stop_interval=max(1, math.ceil(lifetime / 60)),
                auto_delete_interval=0,
                ttl_minutes=max(1, math.ceil(lifetime / 60)),
            )
            self.box = await self.client.create(params)
        else:
            self.box = await sdk.create_sandbox(
                name=self.name,
                image=self.config.vercel_image,
                execution_time_limit=lifetime,
                persistent=False,
                tags={"harness.execution": "owned"},
            )

    async def open(self) -> None:
        # Preserve the creation result on cancellation so a sandbox created by
        # an in-flight SDK request can still be stopped, never silently orphaned.
        creating = asyncio.create_task(self._create())
        try:
            await asyncio.shield(creating)
        except BaseException:
            try:
                await creating
            finally:
                await self.close()
            raise
        try:
            code, output = await self.execute(
                [
                    self.config.python_binary,
                    "-c",
                    "import os,sys; [os.makedirs(p,mode=0o700,exist_ok=True) for p in sys.argv[1:]]",
                    self.control_root,
                    self.root,
                ]
            )
            if code:
                raise RuntimeError(f"cloud workspace initialization failed: {output[:4096]}")
            for name in ("worker.py", "cloud_worker.py"):
                await self.write_file(
                    f"{self.control_root}/{name}", Path(__file__).with_name(name).read_bytes()
                )
        except BaseException:
            await self.close()
            raise

    async def execute(self, argv: list[str]) -> tuple[int, str]:
        if self.box is None:
            raise RuntimeError("cloud sandbox is closed")
        if self.config.backend == "modal":
            process = await self.box.exec.aio(*argv, timeout=30)
            stdout, stderr, code = await asyncio.gather(
                process.stdout.read.aio(),
                process.stderr.read.aio(),
                process.wait.aio(),
            )
            return int(code), str(stdout) + str(stderr)
        if self.config.backend == "daytona":
            result = await self.box.process.exec(shlex.join(argv), timeout=30)
            return int(result.exit_code), result.result
        result = await self.box.run_process(argv[0], argv[1:], capture_output=True, kill_after=30)
        return int(result.returncode), (result.stdout or "") + (result.stderr or "")

    async def write_file(self, path: str, data: bytes) -> None:
        if self.box is None:
            raise RuntimeError("cloud sandbox is closed")
        if self.config.backend == "modal":
            await self.box.filesystem.write_bytes.aio(data, path)
        elif self.config.backend == "daytona":
            await self.box.fs.upload_file(data, path)
        else:
            await self.box.fs.write_bytes(path, data)

    async def rpc(self, request: dict[str, Any]) -> dict[str, Any]:
        path = f"{self.control_root}/request-{uuid.uuid4().hex}.json"
        await self.write_file(path, json.dumps(request).encode())
        code, output = await self.execute(
            [
                self.config.python_binary,
                "-u",
                f"{self.control_root}/cloud_worker.py",
                path,
            ]
        )
        try:
            result = json.loads(output)
        except ValueError:
            raise RuntimeError(
                f"{self.config.backend} worker returned invalid data: {output[:4096]}"
            ) from None
        if code or "error" in result:
            raise RuntimeError(result.get("error", f"cloud worker failed with exit {code}"))
        return result

    async def close(self) -> None:
        box, self.box = self.box, None
        client, self.client = self.client, None
        try:
            if box is not None:
                if self.config.backend == "modal":
                    await box.terminate.aio()
                elif self.config.backend == "daytona":
                    await client.delete(box, wait=True)
                else:
                    await box.destroy()
        finally:
            if client is not None:
                await client.close()


__all__ = ["CLOUD_BACKENDS", "CloudSandbox"]
