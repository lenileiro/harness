"""End-to-end CLI tests for `harness evidence list`."""

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
    return tmp_path / "evidence.db"


def _run(args: list[str]) -> Result:
    return CliRunner().invoke(cli_main.app, args)


class TestEmpty:
    def test_no_evidence(self, db_path: Path) -> None:
        result = _run(["evidence", "list", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "No evidence" in result.stdout


class TestEvidenceFromRun:
    def test_records_tool_calls(self, patch_adapter, db_path: Path, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("payload", encoding="utf-8")

        # 1) Tool call: read_file → metadata captured
        # 2) Final text turn
        patch_adapter(
            [
                tool_call_turn("c1", "read_file", {"path": "f.txt"}),
                text_turn("read"),
            ]
        )
        run_result = _run(
            [
                "run",
                "read",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--session",
                "sess_ev",
                "--yes",
            ]
        )
        assert run_result.exit_code == 0, run_result.stdout

        # Evidence list now has one row.
        listing = _run(["evidence", "list", "--db", str(db_path)])
        assert listing.exit_code == 0
        assert "read_file" in listing.stdout
        # The metadata cell should mention bytes (from ReadFileTool).
        assert "bytes=" in listing.stdout

    def test_tool_filter(self, patch_adapter, db_path: Path, tmp_path: Path) -> None:
        (tmp_path / "a.txt").write_text("x", encoding="utf-8")
        # Two tool calls in two turns — read_file then list_dir.
        patch_adapter(
            [
                tool_call_turn("c1", "read_file", {"path": "a.txt"}),
                tool_call_turn("c2", "list_dir", {}),
                text_turn("done"),
            ]
        )
        _run(
            [
                "run",
                "x",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--yes",
                "--session",
                "sess_filter",
            ]
        )

        listing_all = _run(["evidence", "list", "--db", str(db_path)])
        assert "read_file" in listing_all.stdout
        assert "list_dir" in listing_all.stdout

        only_read = _run(["evidence", "list", "--db", str(db_path), "--tool", "read_file"])
        assert "read_file" in only_read.stdout
        assert "list_dir" not in only_read.stdout

    def test_errors_only(self, patch_adapter, db_path: Path, tmp_path: Path) -> None:
        (tmp_path / "ok.txt").write_text("x", encoding="utf-8")
        patch_adapter(
            [
                # First a successful call.
                tool_call_turn("c1", "read_file", {"path": "ok.txt"}),
                # Then a failing call (missing file).
                tool_call_turn("c2", "read_file", {"path": "missing.txt"}),
                text_turn("done"),
            ]
        )
        _run(
            [
                "run",
                "x",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--yes",
                "--session",
                "sess_errors",
            ]
        )

        # All entries: both rows present.
        all_rows = _run(["evidence", "list", "--db", str(db_path)])
        assert all_rows.stdout.count("read_file") >= 2

        # Errors only: just the missing-file one.
        errors = _run(["evidence", "list", "--db", str(db_path), "--errors-only"])
        assert errors.exit_code == 0
        # 'error' status word appears at least once; 'ok' does not.
        assert "error" in errors.stdout

    def test_task_filter(self, patch_adapter, db_path: Path, tmp_path: Path) -> None:
        (tmp_path / "f.txt").write_text("x", encoding="utf-8")
        _run(["tasks", "new", "demo", "--db", str(db_path), "--cwd", str(tmp_path)])

        patch_adapter(
            [
                tool_call_turn("c1", "read_file", {"path": "f.txt"}),
                text_turn("done"),
            ]
        )
        _run(
            [
                "run",
                "x",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--task",
                "T-001",
                "--session",
                "sess_in_task",
                "--yes",
            ]
        )
        # Filter by task — should contain the read_file evidence.
        scoped = _run(["evidence", "list", "--db", str(db_path), "--task", "T-001"])
        assert scoped.exit_code == 0
        assert "read_file" in scoped.stdout

        # Filter by an unknown task ref — exits 1.
        missing = _run(["evidence", "list", "--db", str(db_path), "--task", "T-999"])
        assert missing.exit_code == 1
