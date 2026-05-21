"""End-to-end task lifecycle integration test.

Walks a task through its complete lifecycle in a single test:
  create (all fields) → child task → link → attach agent run →
  status transitions → field updates → verify show output →
  list filtering → delete

Uses a tmp SQLite DB and a FakeAdapter so the test is fast and
deterministic but exercises the real storage and CLI paths.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.core import Capabilities, Done, Event, Message, TextDelta


def _text_turn(text: str) -> list[Event]:
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


def _run(args: list[str]) -> Any:
    return CliRunner().invoke(cli_main.app, args)


class TestTaskLifecycle:
    def test_full_workflow(self, patch_adapter, tmp_path: Path) -> None:
        db = str(tmp_path / "test.db")

        # ── 1. Create parent task with all optional fields ─────────────────
        r = _run(
            [
                "tasks",
                "new",
                "Implement feature X",
                "--db",
                db,
                "--cwd",
                str(tmp_path),
                "--description",
                "End-to-end integration work",
                "--priority",
                "high",
                "--labels",
                "phase-6,integration",
            ]
        )
        assert r.exit_code == 0, r.stdout
        assert "Created T-001" in r.stdout

        # ── 2. Create child task under parent ──────────────────────────────
        r = _run(
            [
                "tasks",
                "new",
                "Write tests for feature X",
                "--db",
                db,
                "--cwd",
                str(tmp_path),
                "--parent",
                "T-001",
                "--labels",
                "testing",
            ]
        )
        assert r.exit_code == 0, r.stdout
        assert "Created T-002" in r.stdout

        # ── 3. Add a typed link: T-002 depends_on T-001 ────────────────────
        r = _run(["tasks", "link", "T-002", "T-001", "--db", db, "--relation", "depends_on"])
        assert r.exit_code == 0, r.stdout

        # ── 4. Attach an agent run to T-001 ────────────────────────────────
        patch_adapter([_text_turn("starting the work")])
        r = _run(
            [
                "run",
                "begin implementation",
                "--db",
                db,
                "--cwd",
                str(tmp_path),
                "--task",
                "T-001",
                "--session",
                "sess-feature-x",
                "--yes",
            ]
        )
        assert r.exit_code == 0, r.stdout

        # ── 5. Transition T-001 through statuses ───────────────────────────
        for status in ("todo", "in_progress", "done"):
            r = _run(["tasks", "update", "T-001", "--db", db, "--status", status])
            assert r.exit_code == 0, r.stdout
            assert "Updated T-001" in r.stdout

        # ── 6. Update title, priority, and labels ──────────────────────────
        r = _run(
            [
                "tasks",
                "update",
                "T-001",
                "--db",
                db,
                "--title",
                "Implement feature X (revised)",
                "--priority",
                "medium",
                "--labels",
                "phase-6,integration,shipped",
            ]
        )
        assert r.exit_code == 0, r.stdout

        # ── 7. show T-001: verify all fields and activity events ───────────
        show = _run(["tasks", "show", "T-001", "--db", db])
        assert show.exit_code == 0, show.stdout

        for fragment in (
            "T-001",
            "Implement feature X (revised)",
            "done",
            "medium",
            "phase-6",
            "shipped",
            "sess-feature-x",  # attached session visible
            "task.created",
            "task.status_changed",
            "task.updated",
            "agent_run.started",
            "agent_run.completed",
        ):
            assert fragment in show.stdout, f"expected {fragment!r} in show output"

        # ── 8. show T-002: link metadata present ───────────────────────────
        show2 = _run(["tasks", "show", "T-002", "--db", db])
        assert show2.exit_code == 0, show2.stdout
        assert "depends_on" in show2.stdout
        assert "T-001" in show2.stdout
        assert "task.linked" in show2.stdout

        # ── 9. list filtering across the lifecycle ─────────────────────────
        done_list = _run(["tasks", "list", "--db", db, "--status", "done"])
        assert "T-001" in done_list.stdout
        assert "T-002" not in done_list.stdout

        backlog_list = _run(["tasks", "list", "--db", db, "--status", "backlog"])
        assert "T-002" in backlog_list.stdout
        assert "T-001" not in backlog_list.stdout

        # ── 10. Delete T-001; T-002 is unaffected ──────────────────────────
        r = _run(["tasks", "rm", "T-001", "--db", db, "--yes"])
        assert r.exit_code == 0, r.stdout
        assert "Deleted" in r.stdout

        assert _run(["tasks", "show", "T-001", "--db", db]).exit_code == 1
        assert _run(["tasks", "show", "T-002", "--db", db]).exit_code == 0
