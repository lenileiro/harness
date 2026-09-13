from __future__ import annotations

import asyncio
import contextlib
import importlib
import inspect
import json
import os
import shlex
import shutil
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from harness.core import ToolCall
from harness.tools.execution import ExecutionConfig, ExecutionToolset
from harness.tools.execution.cloud import CloudSandbox


async def invoke(toolset, name, **arguments):
    tool = next(tool for tool in toolset.tools if tool.name == name)
    return await tool(ToolCall(id="cloud", name=name, arguments=arguments))


def metadata(result):
    assert result.metadata is not None
    return result.metadata


@pytest.fixture
def cloud_sdk(tmp_path, monkeypatch):
    """Fake SDKs execute real uploaded workers inside temporary fixture paths.

    Every public SDK invocation binds against the installed real signature,
    so mocks cannot silently accept invented APIs. No provider client is made.
    """
    calls = []
    controls = tmp_path / "controls"
    remote = tmp_path / "remote"
    original_init = CloudSandbox.__init__

    def initialize(self, config):
        original_init(self, config)
        self.control_root = str(controls)

    monkeypatch.setattr(CloudSandbox, "__init__", initialize)

    async def execute(argv):
        calls.append(("execute", argv))
        # Uploaded workers always use selected cloud paths, not CLI cwd.
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        output, error = await process.communicate()
        return process.returncode or 0, output.decode(), error.decode()

    async def write(path, data):
        calls.append(("write", path))
        assert Path(path).is_relative_to(controls)
        await asyncio.to_thread(Path(path).write_bytes, data)

    async def cleanup():
        calls.append(("cleanup",))
        if controls.exists():
            for path in controls.glob("*/supervisor.pid"):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(int(path.read_text()), signal.SIGTERM)
        if remote.exists():
            await asyncio.to_thread(shutil.rmtree, remote)

    original_import = importlib.import_module

    def sdk(name, package=None):
        if name not in {"modal", "daytona", "vercel.sandbox"}:
            return original_import(name, package)
        real = pytest.importorskip(name)
        if name == "modal":

            async def app_lookup(*args, **kwargs):
                inspect.signature(real.App.lookup).bind(*args, **kwargs)
                return "fake-app"

            async def modal_run(*args, **kwargs):
                inspect.signature(real.Sandbox.exec).bind(None, *args, **kwargs)
                code, output, error = await execute(list(args))
                return SimpleNamespace(
                    stdout=SimpleNamespace(
                        read=SimpleNamespace(aio=AsyncMock(return_value=output))
                    ),
                    stderr=SimpleNamespace(read=SimpleNamespace(aio=AsyncMock(return_value=error))),
                    wait=SimpleNamespace(aio=AsyncMock(return_value=code)),
                )

            async def modal_write(data, remote_path):
                from modal.sandbox_fs import _SandboxFilesystem

                inspect.signature(_SandboxFilesystem.write_bytes).bind(None, data, remote_path)
                await write(remote_path, data)

            box = SimpleNamespace(
                exec=SimpleNamespace(aio=modal_run),
                filesystem=SimpleNamespace(write_bytes=SimpleNamespace(aio=modal_write)),
                terminate=SimpleNamespace(aio=cleanup),
            )

            async def modal_create(*args, **kwargs):
                inspect.signature(real.Sandbox.create).bind(*args, **kwargs)
                calls.append(("create", kwargs))
                return box

            return SimpleNamespace(
                App=SimpleNamespace(lookup=SimpleNamespace(aio=app_lookup)),
                Image=SimpleNamespace(from_registry=lambda image: image),
                Sandbox=SimpleNamespace(create=SimpleNamespace(aio=modal_create)),
            )
        if name == "daytona":
            from daytona._async.filesystem import AsyncFileSystem
            from daytona._async.process import AsyncProcess

            async def daytona_run(command, **kwargs):
                inspect.signature(AsyncProcess.exec).bind(None, command, **kwargs)
                code, output, error = await execute(shlex.split(command))
                return SimpleNamespace(exit_code=code, result=output + error)

            async def upload(src, dst):
                inspect.signature(AsyncFileSystem.upload_file).bind(None, src, dst)
                await write(dst, src)

            box = SimpleNamespace(
                process=SimpleNamespace(exec=daytona_run), fs=SimpleNamespace(upload_file=upload)
            )

            async def daytona_create(params):
                inspect.signature(real.AsyncDaytona.create).bind(None, params)
                calls.append(("create", params.model_dump()))
                return box

            async def delete(sandbox, **kwargs):
                inspect.signature(real.AsyncDaytona.delete).bind(None, sandbox, **kwargs)
                assert sandbox is box and kwargs["wait"]
                await cleanup()

            client = SimpleNamespace(create=daytona_create, delete=delete, close=AsyncMock())
            return SimpleNamespace(
                AsyncDaytona=lambda: client,
                CreateSandboxFromSnapshotParams=real.CreateSandboxFromSnapshotParams,
            )

        async def run(command, args, **kwargs):
            inspect.signature(real.Sandbox.run_process).bind(None, command, args, **kwargs)
            code, output, error = await execute([command, *args])
            return SimpleNamespace(returncode=code, stdout=output, stderr=error)

        async def write_bytes(path, data):
            inspect.signature(real.SandboxFilesystem.write_bytes).bind(None, path, data)
            await write(path, data)

        box = SimpleNamespace(
            run_process=run, fs=SimpleNamespace(write_bytes=write_bytes), destroy=cleanup
        )

        async def create(**kwargs):
            inspect.signature(real.create_sandbox).bind(**kwargs)
            calls.append(("create", kwargs))
            return box

        return SimpleNamespace(create_sandbox=create)

    monkeypatch.setattr("harness.tools.execution.cloud.importlib.import_module", sdk)
    return remote, controls, calls


@pytest.mark.parametrize("backend", ["modal", "daytona", "vercel_sandbox"])
async def test_cloud_sdks_files_shell_verification_processes_and_cleanup(
    tmp_path, cloud_sdk, backend
):
    remote, controls, calls = cloud_sdk
    host = tmp_path / "host"
    host.mkdir()
    config = ExecutionConfig(backend=backend, cloud_cwd=str(remote), python_binary=sys.executable)
    async with ExecutionToolset(config, cwd=host) as tools:
        assert not (await invoke(tools, "write_file", path="cloud.txt", content="remote")).is_error
        assert not (host / "cloud.txt").exists() and (remote / "cloud.txt").read_text() == "remote"
        assert not (
            await invoke(tools, "edit_file", path="cloud.txt", old="remote", new="cloud")
        ).is_error
        assert (await invoke(tools, "read_file", path="cloud.txt")).content == "cloud"
        assert "cloud.txt" in (await invoke(tools, "list_dir")).content
        assert "cloud.txt" in (await invoke(tools, "glob", pattern="*.txt")).content
        verified = await invoke(tools, "verify_work", command='test "$(cat cloud.txt)" = cloud')
        assert not verified.is_error, verified.content
        shell = await invoke(tools, "shell", command="cat cloud.txt")
        assert not shell.is_error and "cloud" in shell.content
        started = await invoke(
            tools, "process", action="start", command='read value; printf %s "$value"'
        )
        assert not started.is_error, started.content
        handle = json.loads(started.content)["process_id"]
        result = await invoke(
            tools,
            "process",
            action="input",
            process_id=handle,
            input="interactive\n",
            wait_seconds=3,
        )
        assert not result.is_error, result.content
        assert json.loads(result.content)["output"] == "interactive"
        result = await invoke(tools, "shell", command="sleep 30", timeout=0.2)
        assert result.is_error and metadata(result)["status"] == "timed_out"
        stopped = await tools.start(command="sleep 30")
        await stopped.terminate()
        assert stopped.done.is_set() and stopped.reason == "terminated"
        assert (await invoke(tools, "read_file", path="../host/cloud.txt")).is_error
    assert calls[-1] == ("cleanup",)
    assert not list(controls.glob("request-*.json"))


@pytest.mark.parametrize("backend", ["modal", "daytona", "vercel_sandbox"])
async def test_missing_cloud_extra_fails_without_host_execution(tmp_path, monkeypatch, backend):
    def missing(_name):
        raise ImportError("SDK absent")

    monkeypatch.setattr("harness.tools.execution.cloud.importlib.import_module", missing)
    with pytest.raises(RuntimeError, match="optional dependencies"):
        async with ExecutionToolset(ExecutionConfig(backend=backend), cwd=tmp_path):
            pytest.fail("missing SDK must fail closed")
    assert not list(tmp_path.iterdir())


async def test_cloud_creation_cancellation_waits_for_owned_cleanup(tmp_path, monkeypatch):
    creating = asyncio.Event()
    release = asyncio.Event()
    cleaned = asyncio.Event()

    async def create(self):
        creating.set()
        await release.wait()
        self.box = "owned"

    async def close(self):
        if self.box is not None:
            cleaned.set()
        self.box = None

    monkeypatch.setattr(CloudSandbox, "_create", create)
    monkeypatch.setattr(CloudSandbox, "close", close)
    cloud = CloudSandbox(ExecutionConfig(backend="modal"))
    task = asyncio.create_task(cloud.open())
    await creating.wait()
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleaned.is_set()


async def test_cloud_output_is_bounded_at_remote_source(tmp_path, cloud_sdk):
    remote, controls, _ = cloud_sdk
    config = ExecutionConfig(
        backend="modal", cloud_cwd=str(remote), python_binary=sys.executable, max_output_bytes=1024
    )
    async with ExecutionToolset(config, cwd=tmp_path) as tools:
        result = await invoke(
            tools, "shell", command=shlex.join([sys.executable, "-c", "print('x'*200000)"])
        )
        assert not result.is_error and metadata(result)["truncated"]
        assert len(metadata(result)["output"]) == 1024
        assert all(path.stat().st_size < 10000 for path in controls.glob("*/events"))


@pytest.mark.parametrize("backend", ["modal", "daytona", "vercel_sandbox"])
async def test_explicit_binary_import_export_survives_sandbox_destruction(
    tmp_path, cloud_sdk, backend
):
    remote, _, calls = cloud_sdk
    host = tmp_path / "host"
    host.mkdir()
    original = b"\x00binary\xffartifact"
    (host / "input.bin").write_bytes(original)
    config = ExecutionConfig(
        backend=backend,
        cloud_cwd=str(remote),
        python_binary=sys.executable,
        import_paths=("input.bin",),
        export_paths=("results/output.bin",),
    )
    async with ExecutionToolset(config, cwd=host) as tools:
        assert (remote / "input.bin").read_bytes() == original
        result = await invoke(
            tools, "shell", command="mkdir -p results; cp input.bin results/output.bin"
        )
        assert not result.is_error
    assert not remote.exists() and calls[-1] == ("cleanup",)
    assert len(tools.exported_files) == 1
    exported = Path(tools.exported_files[0])
    assert exported.is_relative_to(host / "artifacts/execution")
    assert await asyncio.to_thread(exported.read_bytes) == original
    await tools.close()
    assert len(tools.exported_files) == 1


@pytest.mark.parametrize("path", [".harness/secret", "alias/secret", "../outside"])
async def test_import_scope_fails_before_any_sdk_resource_or_upload(tmp_path, cloud_sdk, path):
    remote, _, calls = cloud_sdk
    host = tmp_path / "host"
    (host / ".harness").mkdir(parents=True)
    (host / ".harness/secret").write_text("private")
    (host / "alias").symlink_to(host / ".harness", target_is_directory=True)
    (host / "allowed").write_text("allowed")
    config = ExecutionConfig(backend="modal", cloud_cwd=str(remote), import_paths=("allowed", path))
    with pytest.raises(ValueError):
        async with ExecutionToolset(config, cwd=host):
            pytest.fail("invalid import")
    assert not calls


async def test_export_symlink_failure_is_visible_and_owned_sandbox_still_deleted(
    tmp_path, cloud_sdk
):
    remote, _, calls = cloud_sdk
    host = tmp_path / "host"
    host.mkdir()
    config = ExecutionConfig(
        backend="modal",
        cloud_cwd=str(remote),
        python_binary=sys.executable,
        export_paths=("alias",),
    )
    tools = ExecutionToolset(config, cwd=host)
    with pytest.raises(ExceptionGroup, match="artifact export"):
        async with tools:
            (remote / "private").write_bytes(b"private")
            (remote / "alias").symlink_to(remote / "private")
    assert not remote.exists() and calls[-1] == ("cleanup",)
    assert tools.exported_files == []


async def test_export_destination_symlink_and_traversal_fail_before_creation(tmp_path, cloud_sdk):
    remote, _, calls = cloud_sdk
    host = tmp_path / "host"
    host.mkdir()
    (host / "alias").symlink_to(tmp_path, target_is_directory=True)
    for directory in ("alias", "../outside", ".harness/export"):
        config = ExecutionConfig(
            backend="modal",
            cloud_cwd=str(remote),
            export_directory=directory,
            export_paths=("result",),
        )
        with pytest.raises(ValueError):
            async with ExecutionToolset(config, cwd=host):
                pytest.fail("invalid export destination")
    assert not calls


async def test_import_total_byte_cap_fails_before_provider_creation(tmp_path, cloud_sdk):
    remote, _, calls = cloud_sdk
    (tmp_path / "one").write_bytes(b"a" * 800)
    (tmp_path / "two").write_bytes(b"b" * 800)
    config = ExecutionConfig(
        backend="modal", cloud_cwd=str(remote), transfer_max_bytes=1024, import_paths=("one", "two")
    )
    with pytest.raises(ValueError, match="total transfer byte limit"):
        async with ExecutionToolset(config, cwd=tmp_path):
            pytest.fail("oversized import")
    assert not calls
