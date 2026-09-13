"""Windows contracts run offline; native job-tree checks run on Windows CI."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import ctypes
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any

import pytest

from harness.core import ToolCall, ToolResult
from harness.tools.execution import ExecutionConfig, ExecutionToolset, transport, windows_job
from harness.tools.execution import backend as backend_module
from harness.tools.execution.backend import ExecutionBackend, encode_request
from harness.tools.execution.transfers import host_path


class FakeKernel:
    def __init__(self, *, assigned: bool = True, configured: bool = True):
        self.assigned = assigned
        self.configured = configured
        self.closed: list[int] = []
        self.flags: int | None = None
        self.attachments: list[tuple[int, int]] = []

    def CreateJobObjectW(self, *_: Any):
        return 123

    def SetInformationJobObject(self, handle, info_class, pointer, size):
        assert handle == 123 and info_class == 9
        assert size == ctypes.sizeof(windows_job._ExtendedLimits)
        limits = ctypes.cast(pointer, ctypes.POINTER(windows_job._ExtendedLimits)).contents
        self.flags = limits.BasicLimitInformation.LimitFlags
        return self.configured

    def OpenProcess(self, access, inherit, pid):
        assert access == 0x101 and inherit is False and pid == 456
        return 789

    def AssignProcessToJobObject(self, job, process):
        self.attachments.append((job, process))
        return self.assigned

    def CloseHandle(self, handle):
        self.closed.append(handle)
        return True


def test_job_object_owns_descendants_and_closes_handles(monkeypatch):
    api = FakeKernel()
    monkeypatch.setattr(windows_job, "_kernel32", lambda: api)
    job = windows_job.WindowsJob()
    job.assign(456)
    assert api.flags == 0x2000  # No BREAKAWAY/SILENT_BREAKAWAY flags.
    assert api.attachments == [(123, 789)] and api.closed == [789]
    job.close()
    job.close()
    assert api.closed == [789, 123]


@pytest.mark.parametrize("phase", ["configuration", "assignment"])
def test_job_object_failure_releases_owned_handles(monkeypatch, phase):
    api = FakeKernel(configured=phase != "configuration", assigned=phase != "assignment")
    monkeypatch.setattr(windows_job, "_kernel32", lambda: api)
    if phase == "configuration":
        with pytest.raises(OSError):
            windows_job.WindowsJob()
        assert api.closed == [123]
    else:
        job = windows_job.WindowsJob()
        with pytest.raises(OSError):
            job.assign(456)
        job.close()
        assert api.closed == [789, 123]


async def test_windows_gate_does_not_release_executable_before_job_assignment(monkeypatch):
    trace: list[Any] = []

    class Job:
        def assign(self, pid):
            trace.append(("assign", pid))

        def close(self):
            trace.append("close")

    class Input:
        def write(self, data):
            trace.append(("write", json.loads(base64.b64decode(data))))

        async def drain(self):
            pass

    class Process:
        pid = 456
        returncode = None
        stdin = Input()

        def kill(self):
            trace.append("kill")

        async def wait(self):
            return 0

    async def create(*args, **kwargs):
        trace.append(("create", args, kwargs))
        return Process()

    monkeypatch.setattr(transport, "is_windows", lambda: True)
    monkeypatch.setattr(transport, "WindowsJob", Job)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    process = await transport.spawn_transport(["selected.exe", "arg with spaces"])
    assert trace[0][0] == "create" and trace[0][1][0] == sys.executable
    assert trace[1:] == [("assign", 456), ("write", ["selected.exe", "arg with spaces"])]
    await transport.stop_transport(process)
    assert trace[-1] == "close" and not transport._JOBS


async def test_windows_assignment_denial_never_releases_gate(monkeypatch):
    writes = []
    killed = []

    class Job:
        def assign(self, pid):
            raise OSError("assignment denied")

        def close(self):
            pass

    class Input:
        def write(self, data):
            writes.append(data)

        async def drain(self):
            pass

    class Process:
        pid = 999
        returncode = None
        stdin = Input()

        def kill(self):
            killed.append(True)

        async def wait(self):
            return 1

    async def create(*args, **kwargs):
        return Process()

    monkeypatch.setattr(transport, "is_windows", lambda: True)
    monkeypatch.setattr(transport, "WindowsJob", Job)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", create)
    with pytest.raises(OSError, match="denied"):
        await transport.spawn_transport(["must-not-run.exe"])
    assert killed and not writes and not transport._JOBS


async def test_windows_requires_explicit_bash_before_worker_spawn(tmp_path, monkeypatch):
    monkeypatch.setattr("harness.tools.execution.backend.is_windows", lambda: True)
    backend = ExecutionBackend(ExecutionConfig(), tmp_path)
    with pytest.raises(ValueError, match="windows_shell"):
        await backend.open()
    assert not backend.active


@pytest.mark.skipif(
    os.name != "posix", reason="portable Windows pipe-pump test uses a POSIX shell fixture"
)
async def test_windows_pipe_supervisor_runs_input_and_deadlines_offline(tmp_path):
    # Execute the Windows implementation directly on this host. Win32 ownership
    # is separately tested above and by the native Windows integration below.
    worker = Path(__file__).parents[1] / "src/harness/tools/execution/worker.py"
    source = worker.read_text().replace('if os.name == "nt":', "if True:")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",
        "-c",
        source,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    request = {
        "operation": "shell",
        "root": str(tmp_path),
        "cwd": ".",
        "command": "read line; printf '%s' \"$line\"",
        "shell": "/bin/sh",
        "timeout": 2,
        "pty": False,
    }
    process.stdin.write(encode_request(request))
    first = json.loads(await asyncio.wait_for(process.stdout.readline(), 3))
    assert first["event"] == "started"
    process.stdin.write(
        json.dumps({"action": "input", "data": base64.b64encode(b"hello\n").decode()}).encode()
        + b"\n"
    )
    output = []
    while line := await asyncio.wait_for(process.stdout.readline(), 3):
        output.append(json.loads(line))
    await process.wait()
    assert b"hello" in b"".join(
        base64.b64decode(e["data"]) for e in output if e["event"] == "output"
    )
    assert output[-1]["event"] == "exit" and output[-1]["exit_code"] == 0
    process.stdin.close()


async def invoke(toolset: ExecutionToolset, name: str, **arguments) -> ToolResult:
    tool = next(tool for tool in toolset.tools if tool.name == name)
    return await tool(ToolCall(id="call", name=name, arguments=arguments))


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows Job Objects and explicit Bash")
async def test_native_windows_files_interaction_timeout_and_descendant_cleanup(tmp_path):
    shell = os.environ.get("HARNESS_WINDOWS_BASH")
    assert shell and await asyncio.to_thread(Path(shell).is_file), (
        "Windows CI must configure HARNESS_WINDOWS_BASH"
    )
    await exercise_windows_execution(tmp_path, shell)


@pytest.mark.skipif(os.name != "nt", reason="requires native Windows directory junctions")
async def test_windows_junction_cannot_export_private_state(tmp_path):
    private = tmp_path / ".harness"
    private.mkdir()
    (private / "secret").write_text("private state")
    alias = tmp_path / "public"
    process = await asyncio.create_subprocess_exec(
        "cmd.exe",
        "/c",
        "mklink",
        "/J",
        str(alias),
        str(private),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate()
    assert process.returncode == 0, (stdout, stderr)
    try:
        with pytest.raises(ValueError, match="reparse points"):
            host_path(tmp_path.resolve(), "public/secret")
        backend = ExecutionBackend(
            ExecutionConfig(windows_shell=os.environ["HARNESS_WINDOWS_BASH"]), tmp_path
        )
        await backend.open()
        try:
            with pytest.raises(RuntimeError, match="reparse points"):
                await backend.files("read_bytes", {"path": "public/secret"})
        finally:
            await backend.close()
    finally:
        alias.rmdir()


@pytest.mark.skipif(os.name != "posix", reason="offline Job Object stand-in uses POSIX groups")
async def test_windows_transport_pipeline_with_real_pipes_and_job_standin(tmp_path, monkeypatch):
    original = asyncio.create_subprocess_exec

    async def create_group(*args, **kwargs):
        kwargs["start_new_session"] = True
        return await original(*args, **kwargs)

    class Job:
        pid: int | None = None

        def assign(self, pid):
            self.pid = pid

        def close(self):
            if self.pid is not None:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(self.pid, signal.SIGKILL)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", create_group)
    monkeypatch.setattr(transport, "is_windows", lambda: True)
    monkeypatch.setattr(transport, "WindowsJob", Job)
    monkeypatch.setattr(backend_module, "is_windows", lambda: True)
    monkeypatch.setattr("harness.tools.execution.process.is_windows", lambda: True)
    monkeypatch.setattr(
        backend_module, "_WORKER", backend_module._WORKER.replace('if os.name == "nt":', "if True:")
    )
    await exercise_windows_execution(tmp_path, "/bin/bash")


async def exercise_windows_execution(tmp_path, shell: str):
    config = ExecutionConfig(windows_shell=shell, timeout_seconds=5)
    async with ExecutionToolset(config, cwd=tmp_path) as tools:
        assert not (
            await invoke(tools, "write_file", path="nested/hello.txt", content="before")
        ).is_error
        assert not (
            await invoke(tools, "edit_file", path="nested/hello.txt", old="before", new="after")
        ).is_error
        assert (await invoke(tools, "read_file", path="nested/hello.txt")).content == "after"
        assert "hello.txt" in (await invoke(tools, "list_dir", path="nested")).content
        assert "hello.txt" in (await invoke(tools, "glob", pattern="**/*.txt")).content
        assert (await invoke(tools, "read_file", path="../outside.txt")).is_error
        assert (await invoke(tools, "shell", command="pwd", cwd="..")).is_error
        shell_result = await invoke(tools, "shell", command="cat hello.txt", cwd="nested")
        assert not shell_result.is_error and "after" in shell_result.content
        start = await invoke(
            tools, "process", action="start", command="read line; printf '%s' \"$line\""
        )
        assert not start.is_error
        pid = json.loads(start.content)["process_id"]
        assert not (
            await invoke(tools, "process", action="input", process_id=pid, input="interactive\n")
        ).is_error
        result = await invoke(tools, "process", action="poll", process_id=pid, wait_seconds=2)
        assert "interactive" in result.content and '"running": false' in result.content
        timed = await invoke(tools, "shell", command="sleep 30", timeout=0.1)
        assert timed.is_error and timed.metadata and timed.metadata["status"] == "timed_out"
        background = await invoke(tools, "shell", command="(sleep 1; echo escaped > escaped.txt) &")
        assert not background.is_error
        await asyncio.sleep(1.2)
        assert not (tmp_path / "escaped.txt").exists()
        assert (
            await invoke(tools, "process", action="start", command="echo no", pty=True)
        ).is_error
        running = await invoke(tools, "process", action="start", command="sleep 30")
        assert not running.is_error
        running_id = json.loads(running.content)["process_id"]
        stopped = await invoke(tools, "process", action="terminate", process_id=running_id)
        assert '"running": false' in stopped.content
        verified = await invoke(
            tools, "verify_work", command='test "$(cat nested/hello.txt)" = after'
        )
        assert not verified.is_error
    assert not transport._JOBS
