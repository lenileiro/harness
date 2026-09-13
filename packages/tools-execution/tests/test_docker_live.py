"""Explicit opt-in integration: only create/remove the context-owned container."""

import asyncio
import json
import os

import pytest

from harness.core import ToolCall, ToolResult
from harness.tools.execution import ExecutionConfig, ExecutionToolset


def metadata(result: ToolResult):
    assert result.metadata is not None
    return result.metadata


@pytest.mark.skipif(
    not os.environ.get("HARNESS_EXECUTION_TEST_IMAGE"),
    reason="requires an explicitly selected disposable Docker image with Python 3",
)
async def test_live_docker_files_shell_processes_and_owned_cleanup(tmp_path):
    config = ExecutionConfig(
        backend="docker",
        docker_image=os.environ["HARNESS_EXECUTION_TEST_IMAGE"],
        timeout_seconds=10,
    )
    async with ExecutionToolset(config, cwd=tmp_path) as toolset:
        tools = {tool.name: tool for tool in toolset.tools}

        async def call(name, **args):
            return await tools[name](ToolCall(id="live", name=name, arguments=args))

        assert not (await call("write_file", path="created.txt", content="container")).is_error
        assert (tmp_path / "created.txt").read_text() == "container"
        assert not (
            await call("edit_file", path="created.txt", old="container", new="edited")
        ).is_error
        assert (await call("read_file", path="created.txt")).content == "edited"
        assert "created.txt" in (await call("list_dir")).content
        assert "created.txt" in (await call("glob", pattern="*.txt")).content
        result = await call("shell", command="uname -s; cat created.txt")
        assert not result.is_error and "Linux" in result.content and "edited" in result.content
        result = await call("verify_work", command='test "$(cat created.txt)" = edited')
        assert not result.is_error and result.content.startswith("PASSED")
        process = await call("process", action="start", command='read value; printf %s "$value"')
        assert not process.is_error
        result = await call(
            "process",
            action="input",
            process_id=metadata(process)["process_id"],
            input="interactive\n",
            wait_seconds=3,
        )
        assert not result.is_error and json.loads(result.content)["output"] == "interactive"
        result = await call("shell", command="sleep 30", timeout=0.2)
        assert result.is_error and metadata(result)["status"] == "timed_out"
        owned = toolset.backend.container_name
        assert owned and owned.startswith("harness-execution-")
    assert toolset.backend.container_name is None and not toolset.backend.active
    inspected = await asyncio.create_subprocess_exec(
        "docker",
        "inspect",
        owned,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, error = await inspected.communicate()
    assert inspected.returncode != 0 and b"no such object" in error.lower()
