from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import sys
from typing import cast

import pytest
from pydantic import ValidationError

from harness.core import ToolCall, ToolResult
from harness.tools.execution import ExecutionConfig, ExecutionToolset


def metadata(result: ToolResult):
    assert result.metadata is not None
    return result.metadata


async def invoke(toolset: ExecutionToolset, name: str, **arguments):
    tool = next(tool for tool in toolset.tools if tool.name == name)
    return await tool(ToolCall(id="call", name=name, arguments=arguments))


def python_command(source: str) -> str:
    return shlex.join([sys.executable, "-u", "-c", source])


def running(pid: int) -> bool:
    result = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return (
        result.returncode == 0
        and bool(result.stdout.strip())
        and not result.stdout.strip().startswith("Z")
    )


async def test_local_files_and_shell_share_workspace(tmp_path):
    async with ExecutionToolset(ExecutionConfig(), cwd=tmp_path) as tools:
        assert not (
            await invoke(tools, "write_file", path="nested/file.txt", content="before")
        ).is_error
        assert not (
            await invoke(tools, "edit_file", path="nested/file.txt", old="before", new="after")
        ).is_error
        assert (await invoke(tools, "read_file", path="nested/file.txt")).content == "after"
        assert "file.txt" in (await invoke(tools, "list_dir", path="nested")).content
        assert "nested/file.txt" in (await invoke(tools, "glob", pattern="**/*.txt")).content
        shell = await invoke(tools, "shell", command="cat file.txt", cwd="nested")
        assert not shell.is_error and "after" in shell.content
        assert metadata(shell)["cwd"] == str(tmp_path / "nested")
    assert (await invoke(tools, "read_file", path="nested/file.txt")).is_error


async def test_backend_resolves_symlink_and_cwd_boundaries(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (tmp_path / "secret").write_text("secret")
    (root / "escape").symlink_to(tmp_path, target_is_directory=True)
    async with ExecutionToolset(ExecutionConfig(), cwd=root) as tools:
        for name, args in (
            ("read_file", {"path": "../secret"}),
            ("read_file", {"path": "escape/secret"}),
            ("write_file", {"path": "escape/secret", "content": "changed"}),
            ("edit_file", {"path": "escape/secret", "old": "secret", "new": "changed"}),
            ("list_dir", {"path": "escape"}),
            ("glob", {"pattern": "../*"}),
            ("shell", {"command": "pwd", "cwd": "escape"}),
        ):
            assert (await invoke(tools, name, **args)).is_error, name
        assert "escape" not in (await invoke(tools, "list_dir")).content
        assert "secret" not in (await invoke(tools, "glob", pattern="**/*")).content
    assert (tmp_path / "secret").read_text() == "secret"


async def test_large_file_payload_uses_stdin_not_command_argument(tmp_path):
    content = "x" * 300000
    async with ExecutionToolset(ExecutionConfig(), cwd=tmp_path) as tools:
        assert not (await invoke(tools, "write_file", path="large.txt", content=content)).is_error
        assert (await invoke(tools, "read_file", path="large.txt")).content == content


async def test_process_handles_support_poll_and_input_across_calls(tmp_path):
    async with ExecutionToolset(ExecutionConfig(), cwd=tmp_path) as tools:
        start = await invoke(
            tools, "process", action="start", command="read value; printf 'received:%s' \"$value\""
        )
        assert not start.is_error and metadata(start)["running"]
        handle = json.loads(start.content)["process_id"]
        result = await invoke(
            tools, "process", action="input", process_id=handle, input="hello\n", wait_seconds=2
        )
        assert not result.is_error and json.loads(result.content)["output"] == "received:hello"
        assert metadata(result)["exit_code"] == 0
        result = await invoke(tools, "process", action="poll", process_id=handle)
        assert not metadata(result)["running"]
        other = await invoke(tools, "process", action="poll", process_id="foreign-handle")
        assert other.is_error


async def test_local_pty_has_controlling_terminal_and_accepts_input(tmp_path):
    command = python_command(
        "import os; print(os.isatty(0), os.isatty(1), flush=True); print(open('/dev/tty').isatty()); print('answer=' + input())"
    )
    async with ExecutionToolset(ExecutionConfig(), cwd=tmp_path) as tools:
        start = await invoke(tools, "process", action="start", command=command, pty=True)
        assert not start.is_error
        result = await invoke(
            tools,
            "process",
            action="input",
            process_id=metadata(start)["process_id"],
            input="yes\n",
            wait_seconds=2,
        )
        assert not result.is_error
        assert "True True" in result.content and "answer=yes" in result.content
        assert metadata(result)["exit_code"] == 0


async def test_timeout_reaps_group_and_output_is_bounded(tmp_path):
    command = python_command("import time; print('x' * 100000, flush=True); time.sleep(30)")
    async with ExecutionToolset(
        ExecutionConfig(timeout_seconds=0.4, max_output_bytes=1024), cwd=tmp_path
    ) as tools:
        result = await invoke(tools, "shell", command=command)
        assert result.is_error and metadata(result)["status"] == "timed_out"
        assert metadata(result)["truncated"] and len(metadata(result)["output"]) == 1024
        assert not running(metadata(result)["pid"])


async def test_full_stdin_does_not_block_timeout_or_termination(tmp_path):
    async with ExecutionToolset(ExecutionConfig(timeout_seconds=0.4), cwd=tmp_path) as tools:
        start = await invoke(tools, "process", action="start", command="sleep 30")
        result = await invoke(
            tools,
            "process",
            action="input",
            process_id=metadata(start)["process_id"],
            input="x" * 65536,
            wait_seconds=2,
        )
        assert metadata(result)["status"] == "timed_out"
        assert not running(metadata(start)["pid"])


async def test_successful_shell_reaps_background_descendants(tmp_path):
    async with ExecutionToolset(ExecutionConfig(), cwd=tmp_path) as tools:
        result = await invoke(tools, "shell", command="sleep 30 & echo $!")
        assert not result.is_error
        descendant = int(metadata(result)["output"].strip())
        assert not running(descendant)


async def test_shell_cancellation_and_context_exit_reap_processes(tmp_path, monkeypatch):
    async with ExecutionToolset(ExecutionConfig(), cwd=tmp_path) as tools:
        started = asyncio.Event()
        original = tools.start

        async def record_start(**kwargs):
            process = await original(**kwargs)
            started.set()
            return process

        monkeypatch.setattr(tools, "start", record_start)
        task = asyncio.create_task(invoke(tools, "shell", command="sleep 30"))
        await asyncio.wait_for(started.wait(), 3)
        process = next(iter(tools.processes.values()))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert process.pid is not None and not running(process.pid)
        start = await invoke(tools, "process", action="start", command="sleep 30")
        pid = metadata(start)["pid"]
        assert running(pid)
    assert not running(pid)


async def test_active_process_limit_and_termination(tmp_path):
    async with ExecutionToolset(ExecutionConfig(max_processes=1), cwd=tmp_path) as tools:
        first = await invoke(tools, "process", action="start", command="sleep 30")
        assert (await invoke(tools, "process", action="start", command="sleep 30")).is_error
        stopped = await invoke(
            tools, "process", action="terminate", process_id=metadata(first)["process_id"]
        )
        assert not stopped.is_error and metadata(stopped)["status"] == "terminated"


async def test_disconnected_control_channel_reaps_process_group(tmp_path):
    async with ExecutionToolset(ExecutionConfig(), cwd=tmp_path) as tools:
        process = await tools.start(command="sleep 30")
        assert process.process is not None and process.process.stdin is not None
        process.process.stdin.close()
        await asyncio.wait_for(process.done.wait(), 3)
        assert process.reason == "disconnected"
        assert process.pid is not None and not running(process.pid)


async def test_close_racing_with_start_does_not_leak_process(tmp_path):
    tools = ExecutionToolset(ExecutionConfig(), cwd=tmp_path)
    await tools.__aenter__()
    start = asyncio.create_task(tools.start(command="sleep 30"))
    closing = asyncio.create_task(tools.close())
    process = await start
    await closing
    assert process.pid is not None and not running(process.pid)
    with pytest.raises(RuntimeError, match="closed"):
        await tools.start(command="true")


@pytest.fixture
def fake_transport(tmp_path):
    remote = tmp_path / "remote"
    remote.mkdir()
    logs = tmp_path / "transport.jsonl"
    executable = tmp_path / "fake-transport"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        f"log = {str(logs)!r}\n"
        f"remote = {str(remote)!r}\n"
        "args = sys.argv[1:]\n"
        "with open(log, 'a') as stream: stream.write(json.dumps(args) + '\\n')\n"
        "if args[0] == 'run': print('owned-container-id'); sys.exit(0)\n"
        "if args[0] == 'rm': sys.exit(0)\n"
        "if args[0] == 'exec': worker = args[args.index('-c') + 1]\n"
        "else:\n"
        " import shlex\n"
        " worker = shlex.split(args[-1])[-1]\n"
        "worker = worker.replace('request = json.loads(base64.b64decode(sys.stdin.buffer.readline()))', 'request = json.loads(base64.b64decode(sys.stdin.buffer.readline())); request[\\\"root\\\"] = ' + repr(remote))\n"
        "os.execv(sys.executable, [sys.executable, '-u', '-c', worker])\n"
    )
    executable.chmod(0o755)
    return executable, remote, logs


@pytest.mark.parametrize("backend", ["docker", "ssh", "singularity"])
async def test_remote_transport_routes_every_file_and_process_operation(
    tmp_path, fake_transport, backend
):
    binary, remote, logs = fake_transport
    host = tmp_path / "host"
    host.mkdir()
    config = ExecutionConfig(
        backend=backend,
        docker_image="explicit-python-image",
        docker_binary=str(binary),
        ssh_host="example.test",
        ssh_user="tester",
        ssh_port=2222,
        ssh_cwd="/remote/project with spaces",
        ssh_binary=str(binary),
        singularity_image="explicit.sif",
        singularity_binary=str(binary),
    )
    async with ExecutionToolset(config, cwd=host) as tools:
        assert not (await invoke(tools, "write_file", path="file.txt", content="remote")).is_error
        assert not (host / "file.txt").exists()
        assert (remote / "file.txt").read_text() == "remote"
        assert not (
            await invoke(tools, "edit_file", path="file.txt", old="remote", new="selected")
        ).is_error
        assert (await invoke(tools, "read_file", path="file.txt")).content == "selected"
        assert "file.txt" in (await invoke(tools, "list_dir")).content
        assert "file.txt" in (await invoke(tools, "glob", pattern="*.txt")).content
        shell = await invoke(tools, "shell", command="cat file.txt")
        assert not shell.is_error and "selected" in shell.content
        verification = await invoke(
            tools, "verify_work", command='test "$(cat file.txt)" = selected'
        )
        assert not verification.is_error and verification.content.startswith("PASSED")
        assert metadata(verification)["backend"] == backend
        start = await invoke(
            tools, "process", action="start", command='read word; printf %s "$word"'
        )
        result = await invoke(
            tools,
            "process",
            action="input",
            process_id=metadata(start)["process_id"],
            input="transport\n",
            wait_seconds=2,
        )
        assert json.loads(result.content)["output"] == "transport"
        assert (await invoke(tools, "process", action="start", command="true", pty=True)).is_error
    commands = [json.loads(line) for line in logs.read_text().splitlines()]
    if backend == "docker":
        assert commands[0][0] == "run" and "explicit-python-image" in commands[0]
        assert f"type=bind,src={host},dst=/workspace" in commands[0]
        name = commands[0][commands[0].index("--name") + 1]
        assert commands[-1] == ["rm", "--force", name]
        assert all(command[0] == "exec" for command in commands[1:-1])
    elif backend == "singularity":
        assert all(command[0] == "exec" and "--containall" in command for command in commands)
        assert all(
            "explicit.sif" in command and f"{host}:/workspace" in command for command in commands
        )
    else:
        assert all(command[0] == "-T" for command in commands)
        assert all(
            "BatchMode=yes" in command and "tester@example.test" in command for command in commands
        )
        assert commands[0][commands[0].index("-p") + 1] == "2222"


async def test_failed_docker_probe_still_removes_only_owned_container(
    tmp_path, fake_transport, monkeypatch
):
    binary, _, logs = fake_transport
    from harness.tools.execution.backend import ExecutionBackend

    async def fail_probe(self, operation, arguments):
        raise RuntimeError("missing remote Python")

    monkeypatch.setattr(ExecutionBackend, "files", fail_probe)
    config = ExecutionConfig(backend="docker", docker_image="explicit", docker_binary=str(binary))
    with pytest.raises(RuntimeError, match="missing remote Python"):
        async with ExecutionToolset(config, cwd=tmp_path):
            pytest.fail("probe failure must not initialize")
    commands = [json.loads(line) for line in logs.read_text().splitlines()]
    name = commands[0][commands[0].index("--name") + 1]
    assert commands[-1] == ["rm", "--force", name]


@pytest.mark.parametrize("backend", ["docker", "ssh"])
async def test_remote_failure_never_falls_back_to_host(tmp_path, backend):
    config = ExecutionConfig(
        backend=backend,
        docker_image="explicit",
        docker_binary="/missing/docker",
        ssh_host="example.test",
        ssh_cwd="/remote",
        ssh_binary="/missing/ssh",
    )
    with pytest.raises((OSError, RuntimeError)):
        async with ExecutionToolset(config, cwd=tmp_path):
            pytest.fail("missing remote transport must not initialize")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "config",
    [
        {"backend": "docker"},
        {"backend": "ssh"},
        {"backend": "ssh", "ssh_host": "-oProxyCommand=bad", "ssh_cwd": "/remote"},
        {"backend": "ssh", "ssh_host": "host", "ssh_cwd": "relative"},
        {"backend": "unimplemented"},
    ],
)
def test_invalid_backend_configuration_fails_closed(config):
    with pytest.raises(ValidationError):
        ExecutionConfig.model_validate(config)


async def test_backend_verification_defaults_and_truthful_failure(tmp_path):
    config = ExecutionConfig()
    async with ExecutionToolset(config, cwd=tmp_path, verify_command="printf '1 passed'") as tools:
        passed = await invoke(tools, "verify_work")
        assert not passed.is_error and metadata(passed)["used_default_command"]
        assert (await invoke(tools, "verify_work", command="true")).is_error
        failed = await invoke(tools, "verify_work", command=python_command("print('1 failed')"))
        assert failed.is_error and metadata(failed)["output_reports_failure"]
        assert metadata(failed)["pipefail"] and metadata(failed)["errexit"]


def test_kill_group_tolerates_a_vanished_process_group(monkeypatch):
    """A reaped group reports ESRCH on Linux and EPERM on macOS; neither is fatal."""

    from harness.tools.execution import worker

    class _Reaped:
        pid = 4242
        returncode = 0

        def wait(self, timeout=None):
            del timeout
            return 0

    for error in (ProcessLookupError, PermissionError):
        signalled: list[int] = []

        def _vanished(pgid, sig, _error=error, _seen=signalled):
            _seen.append(sig)
            raise _error(1, "Operation not permitted")

        # worker resolves os.killpg at call time, so patching os covers it.
        monkeypatch.setattr(os, "killpg", _vanished)
        worker.kill_group(cast("subprocess.Popen[bytes]", _Reaped()))
        # Both the TERM and the KILL attempt are made and both are absorbed.
        assert len(signalled) == 2
