"""Tests for `harness chat` REPL.

CliRunner's `input=` feeds lines to stdin so we can drive the REPL.
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


def _run(args: list[str], stdin: str) -> Result:
    return CliRunner().invoke(cli_main.app, args, input=stdin)


class TestSlashCommands:
    def test_quit_exits_cleanly(self, patch_adapter, tmp_path: Path) -> None:
        result = _run(["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"], stdin="/quit\n")
        assert result.exit_code == 0
        assert "bye" in result.stdout

    def test_help_lists_commands(self, patch_adapter, tmp_path: Path) -> None:
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="/help\n/quit\n",
        )
        assert result.exit_code == 0
        assert "/help" in result.stdout
        assert "/tools" in result.stdout

    def test_tools_lists_built_in_tools(self, patch_adapter, tmp_path: Path) -> None:
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="/tools\n/quit\n",
        )
        assert result.exit_code == 0
        assert "read_file" in result.stdout
        assert "shell" in result.stdout

    def test_unknown_command_warns(self, patch_adapter, tmp_path: Path) -> None:
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="/nope\n/quit\n",
        )
        assert result.exit_code == 0
        assert "Unknown command" in result.stdout


class TestRegularTurns:
    def test_single_turn_streams_response(self, patch_adapter, tmp_path: Path) -> None:
        patch_adapter([text_turn("hello there")])
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="hi\n/quit\n",
        )
        assert result.exit_code == 0, result.stdout
        assert "hello there" in result.stdout

    def test_multi_turn_uses_resume(self, patch_adapter, tmp_path: Path) -> None:
        # Two turns: first via agent.run(), second via agent.resume().
        patch_adapter([text_turn("first"), text_turn("second")])
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="hello\nfollow up\n/quit\n",
        )
        assert result.exit_code == 0, result.stdout
        assert "first" in result.stdout
        assert "second" in result.stdout


class TestSession:
    def test_session_command_reports_state(self, patch_adapter, tmp_path: Path) -> None:
        patch_adapter([text_turn("ok")])
        result = _run(
            [
                "chat",
                "--cwd",
                str(tmp_path),
                "--in-memory",
                "--yes",
                "--session",
                "sess_explicit",
            ],
            stdin="hello\n/session\n/quit\n",
        )
        assert result.exit_code == 0
        assert "sess_explicit" in result.stdout
        # After one turn: user + assistant = 2 messages
        assert "2 messages" in result.stdout
