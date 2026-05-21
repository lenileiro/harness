"""Tests for `harness sessions fork` CLI command."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.core import Done, Event, Message, TextDelta


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

    async def capabilities(self):
        from harness.core import Capabilities

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


class TestSessionsFork:
    def test_fork_creates_new_session(self, tmp_path: Path, patch_adapter) -> None:
        db = str(tmp_path / "test.db")

        # Create a session via run
        patch_adapter([_text_turn("hello from parent")])
        r = _run(
            [
                "run",
                "hello",
                "--db",
                db,
                "--cwd",
                str(tmp_path),
                "--session",
                "sess_parent",
                "--yes",
            ]
        )
        assert r.exit_code == 0, r.stdout

        # Fork it
        r = _run(["sessions", "fork", "sess_parent", "--db", db])
        assert r.exit_code == 0, r.stdout
        assert "Forked" in r.stdout
        assert "sess_parent" in r.stdout

    def test_fork_with_explicit_session_id(self, tmp_path: Path, patch_adapter) -> None:
        db = str(tmp_path / "test.db")

        patch_adapter([_text_turn("hello")])
        _run(
            [
                "run",
                "hello",
                "--db",
                db,
                "--cwd",
                str(tmp_path),
                "--session",
                "sess_parent",
                "--yes",
            ]
        )

        r = _run(["sessions", "fork", "sess_parent", "--session", "sess_myfork", "--db", db])
        assert r.exit_code == 0, r.stdout
        assert "sess_myfork" in r.stdout

        # Verify the forked session exists
        show_r = _run(["sessions", "show", "sess_myfork", "--db", db])
        assert show_r.exit_code == 0, show_r.stdout

    def test_fork_with_prompt_runs_agent(self, tmp_path: Path, patch_adapter) -> None:
        db = str(tmp_path / "test.db")

        patch_adapter([_text_turn("parent response")])
        _run(
            [
                "run",
                "hello",
                "--db",
                db,
                "--cwd",
                str(tmp_path),
                "--session",
                "sess_parent",
                "--yes",
            ]
        )

        patch_adapter([_text_turn("fork response")])
        r = _run(["sessions", "fork", "sess_parent", "continue from fork", "--db", db, "--yes"])
        assert r.exit_code == 0, r.stdout
        assert "fork response" in r.stdout

    def test_fork_nonexistent_session(self, tmp_path: Path) -> None:
        db = str(tmp_path / "test.db")
        r = _run(["sessions", "fork", "sess_doesnotexist", "--db", db])
        assert r.exit_code == 1
        assert "not found" in r.stdout.lower()

    def test_fork_inherits_message_history(self, tmp_path: Path, patch_adapter) -> None:
        db = str(tmp_path / "test.db")

        patch_adapter([_text_turn("parent answer")])
        _run(
            [
                "run",
                "parent prompt",
                "--db",
                db,
                "--cwd",
                str(tmp_path),
                "--session",
                "sess_par",
                "--yes",
            ]
        )

        r = _run(["sessions", "fork", "sess_par", "--session", "sess_fork", "--db", db])
        assert r.exit_code == 0, r.stdout

        # The fork's show should include the parent message history
        show_r = _run(["sessions", "show", "sess_fork", "--db", db])
        assert show_r.exit_code == 0, show_r.stdout
        assert "parent prompt" in show_r.stdout
