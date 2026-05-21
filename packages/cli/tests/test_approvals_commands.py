"""End-to-end CLI tests for `harness approvals ...` and `--inbox` flag.

Uses a tmp SQLite DB across invocations so the queue → grant → replay flow
exercises real storage. The Ollama adapter is swapped for a FakeAdapter.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import Result
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.core import Capabilities, Done, Event, Message, TextDelta, ToolCall, ToolCallEvent
from harness.storage.sqlite import SQLiteStorage


def _first_approval_id(db_path: Path) -> str:
    """Query the SQLite store directly — avoids parsing the CLI table output
    (Rich may truncate the id column on narrow terminals)."""

    async def _go() -> str:
        s = SQLiteStorage(path=db_path)
        try:
            items = await s.list_approvals()
            assert items, "expected at least one approval in the inbox"
            return items[0].id
        finally:
            await s.close()

    return asyncio.run(_go())


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
    return tmp_path / "approvals.db"


def _run(args: list[str]) -> Result:
    return CliRunner().invoke(cli_main.app, args)


# ---------------------------------------------------------------------------
# Simple list / show / no-op cases (no agent involved)
# ---------------------------------------------------------------------------


class TestEmptyAndMissing:
    def test_list_empty(self, db_path: Path) -> None:
        result = _run(["approvals", "list", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "No approvals" in result.stdout

    def test_show_missing_exits_1(self, db_path: Path) -> None:
        result = _run(["approvals", "show", "appr_x", "--db", str(db_path)])
        assert result.exit_code == 1

    def test_grant_missing_exits_1(self, db_path: Path) -> None:
        result = _run(["approvals", "grant", "appr_x", "--db", str(db_path)])
        assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Inbox flow: run --inbox → list → grant → resume (replay) → real result
# ---------------------------------------------------------------------------


class TestInboxLifecycle:
    def test_queue_grant_replay(self, patch_adapter, db_path: Path, tmp_path: Path) -> None:
        # Create a file the eventual replay needs.
        (tmp_path / "note.txt").write_text("the answer is 42", encoding="utf-8")

        # First turn: model wants to read note.txt. ReadFileTool has
        # approval=auto by default, so we need to force it to prompt via
        # session-level override OR config. Easiest: use a config that
        # forces read_file to prompt.
        cfg = tmp_path / "config.toml"
        cfg.write_text('[approval]\nread_file = "prompt"\n', encoding="utf-8")

        patch_adapter(
            [
                tool_call_turn("c1", "read_file", {"path": "note.txt"}),
                text_turn("queued; will resume later"),
            ]
        )
        run_result = _run(
            [
                "run",
                "read the file",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--config",
                str(cfg),
                "--session",
                "sess_inbox",
                "--inbox",
            ]
        )
        assert run_result.exit_code == 0, run_result.stdout
        assert "queued for approval" in run_result.stdout

        # Confirm an approval landed in the inbox.
        list_result = _run(["approvals", "list", "--db", str(db_path)])
        assert list_result.exit_code == 0
        assert "read_file" in list_result.stdout
        assert "pending" in list_result.stdout

        # Read the approval id directly from storage (CLI output truncates on
        # narrow terminals).
        approval_id = _first_approval_id(db_path)

        # Grant it.
        grant = _run(["approvals", "grant", approval_id, "--db", str(db_path)])
        assert grant.exit_code == 0
        assert "Granted" in grant.stdout

        # Resume the session — the queued call should be replayed BEFORE
        # the next adapter turn. The adapter script for the resume turn is
        # just a text reply; we expect the model to see the real result in
        # transcript.
        FakeAdapter.next_script = [text_turn("now I know it's 42")]
        resume = _run(
            [
                "sessions",
                "resume",
                "sess_inbox",
                "what does it say?",
                "--db",
                str(db_path),
                "--cwd",
                str(tmp_path),
                "--config",
                str(cfg),
            ]
        )
        assert resume.exit_code == 0, resume.stdout

        # The transcript now has the real file contents at the tool message.
        show = _run(["sessions", "show", "sess_inbox", "--db", str(db_path)])
        assert show.exit_code == 0
        assert "the answer is 42" in show.stdout
        # The "queued" placeholder should NOT appear in the transcript any
        # more — it was overwritten.
        # (The activity log still shows approval.queued / approval.replayed,
        # which IS expected, so we check the assistant/tool panel area.)
        # Loose check: the panel content should contain the real text.

    def test_deny_does_not_replay(self, patch_adapter, db_path: Path, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text('[approval]\nread_file = "prompt"\n', encoding="utf-8")

        (tmp_path / "f.txt").write_text("hidden", encoding="utf-8")
        patch_adapter(
            [
                tool_call_turn("c1", "read_file", {"path": "f.txt"}),
                text_turn("queued"),
            ]
        )
        _run(
            [
                "run",
                "read",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--config",
                str(cfg),
                "--session",
                "sess_deny",
                "--inbox",
            ]
        )
        approval_id = _first_approval_id(db_path)

        deny = _run(["approvals", "deny", approval_id, "--db", str(db_path)])
        assert deny.exit_code == 0
        assert "Denied" in deny.stdout

        # Resume — the placeholder content should stay queued (no replay).
        FakeAdapter.next_script = [text_turn("ok")]
        _run(
            [
                "sessions",
                "resume",
                "sess_deny",
                "continue",
                "--db",
                str(db_path),
                "--cwd",
                str(tmp_path),
                "--config",
                str(cfg),
            ]
        )
        show = _run(["sessions", "show", "sess_deny", "--db", str(db_path)])
        # File contents should NOT be in the transcript.
        assert "hidden" not in show.stdout
