"""End-to-end CLI tests for the `--verify rule|llm|none` flag."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import Result
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.core import Capabilities, Done, Event, Message, TextDelta, ToolCall, ToolCallEvent


def text_turn(text: str) -> list[Event]:
    return [TextDelta(text=text), Done(final_message=Message(role="assistant", content=text))]


def tool_call_turn(call_id: str, name: str, args: dict) -> list[Event]:
    call = ToolCall(id=call_id, name=name, arguments=args)
    return [
        ToolCallEvent(call=call),
        Done(final_message=Message(role="assistant", content=None, tool_calls=[call])),
    ]


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
    return tmp_path / "verify.db"


def _run(args: list[str]) -> Result:
    return CliRunner().invoke(cli_main.app, args)


class TestVerifyFlag:
    def test_no_verifier_by_default(self, patch_adapter, db_path: Path, tmp_path: Path) -> None:
        patch_adapter([text_turn("done")])
        result = _run(
            [
                "run",
                "hi",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--yes",
            ]
        )
        assert result.exit_code == 0
        # No verdict line.
        assert "verify" not in result.stdout

    def test_rule_verifier_renders_verdict_on_clean_run(
        self, patch_adapter, db_path: Path, tmp_path: Path
    ) -> None:
        patch_adapter([text_turn("answer")])
        result = _run(
            [
                "run",
                "hi",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--yes",
                "--verify",
                "rule",
            ]
        )
        assert result.exit_code == 0, result.stdout
        # Rich-rendered verdict line.
        assert "verify" in result.stdout
        assert "rule" in result.stdout
        # Clean run → can_finish=True; "no tools dispatched" reason.
        assert "no tools dispatched" in result.stdout

    def test_rule_verifier_blocks_on_tool_error(
        self, patch_adapter, db_path: Path, tmp_path: Path
    ) -> None:
        # Adapter calls read_file with a missing path → tool returns is_error=True
        patch_adapter(
            [
                tool_call_turn("c1", "read_file", {"path": "missing.txt"}),
                text_turn("done"),
            ]
        )
        result = _run(
            [
                "run",
                "x",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--yes",
                "--verify",
                "rule",
            ]
        )
        assert result.exit_code == 0
        # Verdict mentions the failing tool by name.
        assert "verify" in result.stdout
        assert "read_file" in result.stdout
        # ✗ marker shown for can_finish=False.
        assert "✗" in result.stdout

    def test_unknown_verify_value_exits_2(
        self, patch_adapter, db_path: Path, tmp_path: Path
    ) -> None:
        result = _run(
            [
                "run",
                "x",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--yes",
                "--verify",
                "magic",
            ]
        )
        # typer.BadParameter produces a usage error (exit code 2).
        assert result.exit_code == 2
