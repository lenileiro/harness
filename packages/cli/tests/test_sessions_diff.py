"""Tests for _render_session_diff and `harness sessions diff` command."""

from __future__ import annotations

from io import StringIO
from unittest.mock import AsyncMock, patch

from rich.console import Console
from typer.testing import CliRunner

from harness.cli.__main__ import app
from harness.cli.render import _render_session_diff
from harness.tasks import ActivityEvent


def _console() -> tuple[Console, StringIO]:
    buf = StringIO()
    con = Console(file=buf, highlight=False, markup=True, no_color=True, width=120)
    return con, buf


def _activity(kind: str, **data: object) -> ActivityEvent:
    return ActivityEvent(session_id="s1", kind=kind, data=dict(data))


def _write_event(
    path: str,
    before: str | None,
    after: str,
    is_error: bool = False,
) -> ActivityEvent:
    return _activity(
        "tool_call.completed",
        name="write_file",
        is_error=is_error,
        metadata={
            "path": path,
            "content_before": before,
            "content_after": after,
        },
    )


def _edit_event(
    path: str,
    before: str,
    after: str,
) -> ActivityEvent:
    return _activity(
        "tool_call.completed",
        name="edit_file",
        is_error=False,
        metadata={
            "path": path,
            "content_before": before,
            "content_after": after,
        },
    )


def _shell_event(command: str, exit_code: int = 0) -> ActivityEvent:
    return _activity(
        "tool_call.completed",
        name="shell",
        is_error=False,
        arguments={"command": command},
        metadata={"exit_code": exit_code},
    )


# ---------------------------------------------------------------------------
# _render_session_diff
# ---------------------------------------------------------------------------


class TestRenderSessionDiff:
    def test_no_changes_prints_message(self) -> None:
        con, buf = _console()
        _render_session_diff([], con)
        assert "No file changes" in buf.getvalue()

    def test_write_new_file_shows_additions(self) -> None:
        con, buf = _console()
        _render_session_diff(
            [_write_event("out.txt", None, "hello\nworld\n")],
            con,
        )
        output = buf.getvalue()
        assert "write_file" in output
        assert "out.txt" in output
        # New lines show as additions in the diff
        assert "+" in output

    def test_edit_shows_unified_diff(self) -> None:
        con, buf = _console()
        _render_session_diff(
            [_edit_event("src.py", 'return "hello"\n', 'return "hi"\n')],
            con,
        )
        output = buf.getvalue()
        assert "edit_file" in output
        assert "src.py" in output

    def test_failed_write_not_shown(self) -> None:
        con, buf = _console()
        _render_session_diff(
            [_write_event("fail.txt", None, "content", is_error=True)],
            con,
        )
        assert "No file changes" in buf.getvalue()

    def test_shell_events_shown(self) -> None:
        con, buf = _console()
        _render_session_diff(
            [_shell_event("echo hello > /tmp/test.txt", exit_code=0)],
            con,
        )
        output = buf.getvalue()
        assert "shell" in output
        assert "echo hello" in output

    def test_no_diff_when_content_not_captured(self) -> None:
        con, buf = _console()
        evt = _activity(
            "tool_call.completed",
            name="write_file",
            is_error=False,
            metadata={"path": "foo.txt"},
        )
        _render_session_diff([evt], con)
        output = buf.getvalue()
        assert "no diff" in output.lower() or "content not captured" in output.lower()

    def test_unrelated_activity_ignored(self) -> None:
        con, buf = _console()
        _render_session_diff(
            [_activity("agent_run.started")],
            con,
        )
        assert "No file changes" in buf.getvalue()


# ---------------------------------------------------------------------------
# CLI: harness sessions diff command
# ---------------------------------------------------------------------------


runner = CliRunner()


def test_sessions_diff_no_changes() -> None:
    with patch("harness.cli.__main__._build_storage") as mock_storage_factory:
        mock_storage = AsyncMock()
        mock_storage.list_activity = AsyncMock(return_value=[])
        mock_storage.close = AsyncMock()
        mock_storage_factory.return_value = mock_storage

        result = runner.invoke(app, ["sessions", "diff", "sess_abc", "--in-memory"])

    assert result.exit_code == 0
    assert "No file changes" in result.output


def test_sessions_diff_with_write_activity() -> None:
    activity = [_write_event("hello.txt", None, "hello\n")]
    with patch("harness.cli.__main__._build_storage") as mock_storage_factory:
        mock_storage = AsyncMock()
        mock_storage.list_activity = AsyncMock(return_value=activity)
        mock_storage.close = AsyncMock()
        mock_storage_factory.return_value = mock_storage

        result = runner.invoke(app, ["sessions", "diff", "sess_abc", "--in-memory"])

    assert result.exit_code == 0
    assert "write_file" in result.output
    assert "hello.txt" in result.output
