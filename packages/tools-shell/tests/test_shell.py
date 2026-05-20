"""Tests for ShellTool."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from harness.core import ToolCall
from harness.tools.shell import ShellTool


def _call(command: str, **extra: object) -> ToolCall:
    return ToolCall(id="c1", name="shell", arguments={"command": command, **extra})


@pytest.mark.asyncio
class TestShellTool:
    async def test_zero_exit_is_success(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path)
        result = await tool(_call("printf hello"))
        assert result.is_error is False
        assert "exit_code: 0" in result.content
        assert "hello" in result.content

    async def test_nonzero_exit_is_error(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path)
        result = await tool(_call("false"))
        assert result.is_error is True
        assert "exit_code: 1" in result.content

    async def test_stderr_captured(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path)
        result = await tool(_call("printf 'oops' 1>&2; exit 1"))
        assert result.is_error is True
        assert "stderr" in result.content
        assert "oops" in result.content

    async def test_runs_in_cwd(self, tmp_path: Path) -> None:
        (tmp_path / "marker.txt").write_text("found", encoding="utf-8")
        tool = ShellTool(cwd=tmp_path)
        result = await tool(_call("cat marker.txt"))
        assert result.is_error is False
        assert "found" in result.content

    async def test_timeout_kills_command(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path, default_timeout=0.2)
        result = await tool(_call(f"{sys.executable} -c 'import time; time.sleep(2)'"))
        assert result.is_error is True
        assert "timed out" in result.content

    async def test_missing_command_is_error(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path)
        result = await tool(ToolCall(id="c1", name="shell", arguments={"command": "  "}))
        assert result.is_error is True
        assert "command" in result.content

    async def test_per_call_timeout_overrides_default(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path, default_timeout=10.0)
        result = await tool(_call(f"{sys.executable} -c 'import time; time.sleep(2)'", timeout=0.1))
        assert result.is_error is True
        assert "timed out" in result.content

    async def test_stdout_truncated_at_cap(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path, max_output_bytes=32)
        result = await tool(_call(f"{sys.executable} -c 'print(\"x\" * 1000)'"))
        assert "truncated" in result.content

    async def test_default_approval_is_prompt(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path)
        assert tool.approval == "prompt"
