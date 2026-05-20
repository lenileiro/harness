"""`harness sessions ...` integration tests.

Each test uses a fresh tmp_path SQLite DB so commands persist state across
CliRunner invocations within the same test, exercising the real
SQLiteStorage end-to-end. The Ollama adapter is still swapped for FakeAdapter
to avoid any network I/O.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import Result
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.core import (
    Capabilities,
    Done,
    Event,
    Message,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)


def text_turn(text: str) -> list[Event]:
    """Script helper: emit one TextDelta then a Done with the assembled message."""
    return [TextDelta(text=text), Done(final_message=Message(role="assistant", content=text))]


# ---------------------------------------------------------------------------
# Fake adapter (shared shape with test_run_command)
# ---------------------------------------------------------------------------


class FakeAdapter:
    """The class-level queue lets a single test drive multiple CLI invocations.

    Scripts are popped on each `stream()` call across all instances, so a test
    can pre-load N scripts then invoke `harness run` then `harness sessions
    resume` and each invocation consumes the next script in order.
    """

    name = "ollama"
    next_script: ClassVar[list[list[Event]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def stream(self, **_kwargs: Any) -> AsyncIterator[Event]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Event]:
        if not FakeAdapter.next_script:
            raise RuntimeError("FakeAdapter has no scripts left for this stream call")
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
    return tmp_path / "sessions.db"


def _run(cli_args: list[str]) -> Result:
    return CliRunner().invoke(cli_main.app, cli_args)


# ---------------------------------------------------------------------------
# list / show / rm
# ---------------------------------------------------------------------------


class TestSessionsList:
    def test_empty_db(self, db_path: Path) -> None:
        result = _run(["sessions", "list", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "No sessions" in result.stdout

    def test_shows_created_session(self, patch_adapter, db_path: Path, tmp_path: Path) -> None:
        patch_adapter([text_turn("hi")])
        run_result = _run(
            [
                "run",
                "hello",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--session",
                "sess_first",
            ]
        )
        assert run_result.exit_code == 0, run_result.stdout

        list_result = _run(["sessions", "list", "--db", str(db_path)])
        assert list_result.exit_code == 0
        assert "sess_first" in list_result.stdout
        assert "ollama" in list_result.stdout


class TestSessionsShow:
    def test_missing_session_exits_1(self, db_path: Path) -> None:
        result = _run(["sessions", "show", "sess_nope", "--db", str(db_path)])
        assert result.exit_code == 1
        assert "not found" in result.stdout

    def test_shows_full_transcript(self, patch_adapter, db_path: Path, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("payload", encoding="utf-8")
        first_call = ToolCall(id="c1", name="read_file", arguments={"path": "f.txt"})
        patch_adapter(
            [
                [
                    ToolCallEvent(call=first_call),
                    Done(
                        final_message=Message(
                            role="assistant", content=None, tool_calls=[first_call]
                        )
                    ),
                ],
                text_turn("all done"),
            ]
        )
        run_result = _run(
            [
                "run",
                "read f.txt",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--session",
                "sess_show",
            ]
        )
        assert run_result.exit_code == 0, run_result.stdout

        show_result = _run(["sessions", "show", "sess_show", "--db", str(db_path)])
        assert show_result.exit_code == 0
        # Every role panel is present in the rendered output.
        for needle in ("user", "assistant", "tool", "read_file", "payload", "all done"):
            assert needle in show_result.stdout, f"expected {needle!r} in transcript"


class TestSessionsRm:
    def test_yes_skips_confirm_and_deletes(
        self, patch_adapter, db_path: Path, tmp_path: Path
    ) -> None:
        patch_adapter([text_turn("hi")])
        _run(
            [
                "run",
                "hi",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--session",
                "sess_rm",
            ]
        )

        result = _run(["sessions", "rm", "sess_rm", "--db", str(db_path), "--yes"])
        assert result.exit_code == 0
        assert "Deleted" in result.stdout

        show = _run(["sessions", "show", "sess_rm", "--db", str(db_path)])
        assert show.exit_code == 1


class TestSessionsResume:
    def test_resume_extends_history(self, patch_adapter, db_path: Path, tmp_path: Path) -> None:
        patch_adapter(
            [
                text_turn("first turn"),
                text_turn("second turn"),
            ]
        )

        # Initial run creates the session.
        first = _run(
            [
                "run",
                "first",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--session",
                "sess_resume",
            ]
        )
        assert first.exit_code == 0, first.stdout
        assert "first turn" in first.stdout

        # Resume with new prompt.
        second = _run(
            [
                "sessions",
                "resume",
                "sess_resume",
                "follow up",
                "--db",
                str(db_path),
                "--cwd",
                str(tmp_path),
            ]
        )
        assert second.exit_code == 0, second.stdout
        assert "second turn" in second.stdout

        # show now has both user prompts + both assistant turns.
        show = _run(["sessions", "show", "sess_resume", "--db", str(db_path)])
        assert show.exit_code == 0
        assert "first" in show.stdout
        assert "follow up" in show.stdout
        assert "first turn" in show.stdout
        assert "second turn" in show.stdout
