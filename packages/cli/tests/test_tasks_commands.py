"""End-to-end CLI tests for `harness tasks ...` and `--task` on `run`.

Uses a tmp SQLite DB so the same backend is exercised across invocations.
The adapter is the FakeAdapter shared with other CLI tests (defined inline
here to keep this file self-contained).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import Result
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.core import Capabilities, Done, Event, Message, TextDelta


def text_turn(text: str) -> list[Event]:
    return [TextDelta(text=text), Done(final_message=Message(role="assistant", content=text))]


class FakeAdapter:
    name = "ollama"
    next_script: ClassVar[list[list[Event]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def stream(self, **_kwargs: Any) -> AsyncIterator[Event]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Event]:
        if not FakeAdapter.next_script:
            raise RuntimeError("FakeAdapter has no scripts left")
        for ev in FakeAdapter.next_script.pop(0):
            yield ev

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=True)

    async def cancel(self, session_id: str) -> None:
        pass


@pytest.fixture
def patch_adapter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli_main, "OllamaAdapter", FakeAdapter)

    def configure(scripts: list[list[Event]]) -> None:
        FakeAdapter.next_script = scripts

    yield configure
    FakeAdapter.next_script = []


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "tasks.db"


def _run(args: list[str], stdin: str | None = None) -> Result:
    return CliRunner().invoke(cli_main.app, args, input=stdin)


# ---------------------------------------------------------------------------
# `harness tasks` lifecycle
# ---------------------------------------------------------------------------


class TestTasksNew:
    def test_creates_task(self, db_path: Path, tmp_path: Path) -> None:
        result = _run(
            [
                "tasks",
                "new",
                "First task",
                "--db",
                str(db_path),
                "--cwd",
                str(tmp_path),
                "--priority",
                "high",
                "--labels",
                "phase-6,demo",
            ]
        )
        assert result.exit_code == 0, result.stdout
        assert "Created T-001" in result.stdout

    def test_sequential_refs(self, db_path: Path, tmp_path: Path) -> None:
        _run(["tasks", "new", "a", "--db", str(db_path), "--cwd", str(tmp_path)])
        _run(["tasks", "new", "b", "--db", str(db_path), "--cwd", str(tmp_path)])
        list_result = _run(["tasks", "list", "--db", str(db_path)])
        assert "T-001" in list_result.stdout
        assert "T-002" in list_result.stdout

    def test_unknown_parent_exits_1(self, db_path: Path, tmp_path: Path) -> None:
        result = _run(
            [
                "tasks",
                "new",
                "child",
                "--db",
                str(db_path),
                "--cwd",
                str(tmp_path),
                "--parent",
                "T-999",
            ]
        )
        assert result.exit_code == 1


class TestTasksList:
    def test_empty(self, db_path: Path) -> None:
        result = _run(["tasks", "list", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "No tasks" in result.stdout

    def test_filters_by_status(self, db_path: Path, tmp_path: Path) -> None:
        _run(["tasks", "new", "a", "--db", str(db_path), "--cwd", str(tmp_path)])
        _run(
            [
                "tasks",
                "update",
                "T-001",
                "--db",
                str(db_path),
                "--status",
                "done",
            ]
        )
        _run(["tasks", "new", "b", "--db", str(db_path), "--cwd", str(tmp_path)])
        done = _run(["tasks", "list", "--db", str(db_path), "--status", "done"])
        assert "T-001" in done.stdout
        assert "T-002" not in done.stdout


class TestTasksShow:
    def test_shows_task_and_activity(self, db_path: Path, tmp_path: Path) -> None:
        _run(["tasks", "new", "Demo", "--db", str(db_path), "--cwd", str(tmp_path)])
        show = _run(["tasks", "show", "T-001", "--db", str(db_path)])
        assert show.exit_code == 0
        assert "T-001" in show.stdout
        assert "Demo" in show.stdout
        assert "task.created" in show.stdout

    def test_missing_exits_1(self, db_path: Path) -> None:
        result = _run(["tasks", "show", "T-999", "--db", str(db_path)])
        assert result.exit_code == 1


class TestTasksUpdate:
    def test_status_change_logged(self, db_path: Path, tmp_path: Path) -> None:
        _run(["tasks", "new", "x", "--db", str(db_path), "--cwd", str(tmp_path)])
        _run(
            [
                "tasks",
                "update",
                "T-001",
                "--db",
                str(db_path),
                "--status",
                "in_progress",
            ]
        )
        show = _run(["tasks", "show", "T-001", "--db", str(db_path)])
        assert "task.status_changed" in show.stdout
        assert "in_progress" in show.stdout

    def test_unknown_ref_exits_1(self, db_path: Path) -> None:
        result = _run(
            [
                "tasks",
                "update",
                "T-999",
                "--db",
                str(db_path),
                "--status",
                "done",
            ]
        )
        assert result.exit_code == 1


class TestTasksLink:
    def test_link_records_activity(self, db_path: Path, tmp_path: Path) -> None:
        _run(["tasks", "new", "a", "--db", str(db_path), "--cwd", str(tmp_path)])
        _run(["tasks", "new", "b", "--db", str(db_path), "--cwd", str(tmp_path)])
        _run(
            [
                "tasks",
                "link",
                "T-002",
                "T-001",
                "--db",
                str(db_path),
                "--relation",
                "depends_on",
            ]
        )
        show = _run(["tasks", "show", "T-002", "--db", str(db_path)])
        assert "depends_on" in show.stdout
        assert "T-001" in show.stdout
        assert "task.linked" in show.stdout


class TestTasksRm:
    def test_yes_deletes(self, db_path: Path, tmp_path: Path) -> None:
        _run(["tasks", "new", "gone", "--db", str(db_path), "--cwd", str(tmp_path)])
        result = _run(["tasks", "rm", "T-001", "--db", str(db_path), "--yes"])
        assert result.exit_code == 0
        assert "Deleted" in result.stdout
        show = _run(["tasks", "show", "T-001", "--db", str(db_path)])
        assert show.exit_code == 1


# ---------------------------------------------------------------------------
# `harness run --task ...` attaches the session to a task
# ---------------------------------------------------------------------------


class TestRunWithTask:
    def test_attaches_session_to_task(self, patch_adapter, db_path: Path, tmp_path: Path) -> None:
        # Create a task first.
        _run(["tasks", "new", "Wire it up", "--db", str(db_path), "--cwd", str(tmp_path)])

        # Run with --task; verify success.
        patch_adapter([text_turn("hello")])
        run_result = _run(
            [
                "run",
                "say hi",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--task",
                "T-001",
                "--session",
                "sess_attached",
                "--yes",
            ]
        )
        assert run_result.exit_code == 0, run_result.stdout

        # The task's session_ids should now include the new session.
        show = _run(["tasks", "show", "T-001", "--db", str(db_path)])
        assert show.exit_code == 0
        assert "sess_attached" in show.stdout
        # Activity events from the agent run should appear scoped to the task.
        assert "agent_run.started" in show.stdout
        assert "agent_run.completed" in show.stdout

    def test_unknown_task_exits_1(self, patch_adapter, db_path: Path, tmp_path: Path) -> None:
        patch_adapter([text_turn("hello")])
        result = _run(
            [
                "run",
                "hi",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--task",
                "T-999",
                "--yes",
            ]
        )
        assert result.exit_code == 1
