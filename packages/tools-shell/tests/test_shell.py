"""Tests for ShellTool."""

from __future__ import annotations

# pyright: reportOptionalSubscript=false
import sys
from pathlib import Path

import pytest

from harness.core import ActivityEvent, ActivityStore, ToolCall
from harness.core import activity as activity_kinds
from harness.tools.shell import ShellTool


def _call(command: str, **extra: object) -> ToolCall:
    return ToolCall(id="c1", name="shell", arguments={"command": command, **extra})


class _ActivitySink(ActivityStore):
    def __init__(self) -> None:
        self.events: list[ActivityEvent] = []

    async def append_activity(self, event: ActivityEvent) -> None:
        self.events.append(event)

    async def list_activity(
        self,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        kinds: tuple[str, ...] | None = None,
        limit: int = 200,
    ) -> list[ActivityEvent]:
        items = list(self.events)
        if task_id is not None:
            items = [event for event in items if event.task_id == task_id]
        if session_id is not None:
            items = [event for event in items if event.session_id == session_id]
        if kinds is not None:
            items = [event for event in items if event.kind in kinds]
        if limit <= 0:
            return []
        return items[-limit:]


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
        assert "empty failure" in result.content

    async def test_empty_missing_tool_check_is_actionable(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path)
        result = await tool(_call("command -v harness-definitely-missing-tool"))
        assert result.is_error is True
        assert "not found on PATH" in result.content

    async def test_empty_search_failure_is_actionable(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path)
        result = await tool(_call("printf abc | grep zzz"))
        assert result.is_error is True
        assert "search returned no matches" in result.content

    async def test_terse_test_failure_suggests_diagnostics(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path)
        result = await tool(_call("printf 'not ok: basic behavior\\n' >&2; exit 1"))
        assert result.is_error is True
        assert "diagnostic hint" in result.content
        assert "execution tracing" in result.content

    async def test_masked_failure_exit_zero_is_error(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path)
        result = await tool(_call("false || echo masked"))
        assert result.is_error is True
        assert "exit_code: 0" in result.content
        assert "unreliable result" in result.content
        assert result.metadata["masked_failure_exit_status"] is True

    async def test_nested_shell_status_echo_masking_is_error(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path)
        result = await tool(_call("bash -lc 'false; echo exit:$?'"))
        assert result.is_error is True
        assert "exit_code: 0" in result.content
        assert "exit:1" in result.content
        assert "unreliable result" in result.content
        assert result.metadata["masked_failure_exit_status"] is True

    async def test_non_initial_nested_shell_status_echo_masking_is_error(
        self,
        tmp_path: Path,
    ) -> None:
        tool = ShellTool(cwd=tmp_path)
        result = await tool(_call("command sh -c 'false; echo exit:$?'"))
        assert result.is_error is True
        assert "exit_code: 0" in result.content
        assert "exit:1" in result.content
        assert "unreliable result" in result.content
        assert result.metadata["masked_failure_exit_status"] is True

    async def test_zero_exit_with_shell_failure_on_stderr_is_error(
        self,
        tmp_path: Path,
    ) -> None:
        tool = ShellTool(cwd=tmp_path)
        result = await tool(
            _call("sh -c 'printf \"syntax error near token\\n\" >&2; exit 2' | cat")
        )
        assert result.is_error is True
        assert "exit_code: 0" in result.content
        assert "syntax error near token" in result.content
        assert "unreliable result" in result.content
        assert result.metadata["stderr_failure_exit_status"] is True

    async def test_zero_exit_with_heredoc_warning_on_stderr_is_error(
        self,
        tmp_path: Path,
    ) -> None:
        tool = ShellTool(cwd=tmp_path)
        result = await tool(
            _call(
                "sh -c 'printf \"script.sh: line 65: warning: here-document at "
                "line 7 delimited by end-of-file (wanted `INI`)\\n\" >&2; exit 0'"
            )
        )
        assert result.is_error is True
        assert "exit_code: 0" in result.content
        assert "here-document" in result.content
        assert "unreliable result" in result.content
        assert result.metadata["stderr_failure_exit_status"] is True

    async def test_pipefail_marks_failed_pipeline_as_error(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path, pipefail=True)
        result = await tool(_call("false | true"))
        assert result.is_error is True
        assert "exit_code: 1" in result.content
        assert result.metadata["pipefail"] is True

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

    async def test_prefers_local_venv_bin_when_present(self, tmp_path: Path) -> None:
        local_bin = tmp_path / ".venv" / "bin"
        local_bin.mkdir(parents=True)
        local_tool = local_bin / "local-tool"
        local_tool.write_text('#!/bin/sh\nprintf "$VIRTUAL_ENV|local-tool"\n', encoding="utf-8")
        local_tool.chmod(0o755)
        tool = ShellTool(cwd=tmp_path)

        result = await tool(_call("local-tool"))

        assert result.is_error is False
        assert f"{tmp_path / '.venv'}|local-tool" in result.content

    async def test_clean_env_hides_parent_process_variables(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARNESS_SHELL_LEAK_TEST", "present")
        tool = ShellTool(cwd=tmp_path, clean_env=True)

        result = await tool(_call('test -z "$HARNESS_SHELL_LEAK_TEST"'))

        assert result.is_error is False
        assert result.metadata["clean_env"] is True

    async def test_clean_env_still_prefers_workspace_venv(self, tmp_path: Path) -> None:
        local_bin = tmp_path / ".venv" / "bin"
        local_bin.mkdir(parents=True)
        local_tool = local_bin / "local-tool"
        local_tool.write_text('#!/bin/sh\nprintf "$VIRTUAL_ENV|local-tool"\n', encoding="utf-8")
        local_tool.chmod(0o755)
        tool = ShellTool(cwd=tmp_path, clean_env=True)

        result = await tool(_call("local-tool"))

        assert result.is_error is False
        assert f"{tmp_path / '.venv'}|local-tool" in result.content

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

    async def test_stdin_is_closed_for_non_interactive_commands(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path, default_timeout=2.0)
        result = await tool(_call(f"{sys.executable} -c 'input(\"city: \")'"))
        assert result.is_error is True
        assert "EOFError" in result.content

    async def test_stdout_truncated_at_cap(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path, max_output_bytes=32)
        result = await tool(_call(f"{sys.executable} -c 'print(\"x\" * 1000)'"))
        assert "truncated" in result.content

    async def test_default_approval_is_prompt(self, tmp_path: Path) -> None:
        tool = ShellTool(cwd=tmp_path)
        assert tool.approval == "prompt"

    async def test_default_timeout_is_long_enough_for_interactive_work(
        self, tmp_path: Path
    ) -> None:
        tool = ShellTool(cwd=tmp_path)
        assert tool.default_timeout == 120.0

    async def test_emits_running_event_with_pid(self, tmp_path: Path) -> None:
        sink = _ActivitySink()
        tool = ShellTool(cwd=tmp_path)
        tool.bind_activity_context(
            activity_store=sink,
            session_id="sess_shell",
            task_id="task_shell",
        )

        result = await tool(_call("printf hello"))

        assert result.is_error is False
        running = next(
            event for event in sink.events if event.kind == activity_kinds.TOOL_CALL_RUNNING
        )
        assert running.session_id == "sess_shell"
        assert running.task_id == "task_shell"
        assert running.data["tool_call_id"] == "c1"
        assert running.data["name"] == "shell"
        assert isinstance(running.data["pid"], int)
        assert running.data["command"] == "printf hello"
        assert running.data["cwd"] == str(tool.cwd)
        assert result.metadata["pid"] == running.data["pid"]
