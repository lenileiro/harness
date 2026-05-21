"""End-to-end CLI tests for the `--max-context-tokens N` flag (Phase 6.6)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import Result
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.core import Capabilities, Done, Event, Message, TextDelta
from harness.storage.sqlite import SQLiteStorage


def _text_turn(text: str) -> list[Event]:
    return [TextDelta(text=text), Done(final_message=Message(role="assistant", content=text))]


class RecordingAdapter:
    """Fake adapter that records the messages it was asked to stream."""

    name = "ollama"
    next_script: ClassVar[list[list[Event]]] = []
    last_messages: ClassVar[list[Message]] = []
    call_count: ClassVar[int] = 0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def stream(self, *, messages: list[Message], **_kwargs: Any) -> AsyncIterator[Event]:
        RecordingAdapter.last_messages = list(messages)
        RecordingAdapter.call_count += 1
        return self._stream()

    async def _stream(self) -> AsyncIterator[Event]:
        if not RecordingAdapter.next_script:
            raise RuntimeError("RecordingAdapter has no scripts left")
        for ev in RecordingAdapter.next_script.pop(0):
            yield ev

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=True)

    async def cancel(self, session_id: str) -> None:
        pass


@pytest.fixture
def patch_adapter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli_main, "OllamaAdapter", RecordingAdapter)

    def configure(scripts: list[list[Event]]) -> None:
        RecordingAdapter.next_script = scripts
        RecordingAdapter.last_messages = []
        RecordingAdapter.call_count = 0

    yield configure
    RecordingAdapter.next_script = []
    RecordingAdapter.last_messages = []
    RecordingAdapter.call_count = 0


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "budget.db"


def _run(args: list[str]) -> Result:
    return CliRunner().invoke(cli_main.app, args)


async def _preload_chunky_session(db: Path, *, session_id: str, n_messages: int) -> None:
    """Save a session with `n_messages` bulky messages so the budget bites."""
    from harness.core import Session

    storage = SQLiteStorage(path=db)
    try:
        big = "word " * 200
        msgs: list[Message] = [Message(role="system", content="sys")]
        for i in range(n_messages):
            msgs.append(Message(role="user" if i % 2 == 0 else "assistant", content=big))
        session = Session(
            id=session_id,
            provider="ollama",
            model="m",
            cwd=Path.cwd(),
            messages=msgs,
        )
        await storage.save(session)
    finally:
        await storage.close()


class TestMaxContextTokensFlag:
    def test_flag_prunes_messages_on_resume(
        self, patch_adapter, db_path: Path, tmp_path: Path
    ) -> None:
        import asyncio

        asyncio.run(_preload_chunky_session(db_path, session_id="sess_chunk", n_messages=12))

        patch_adapter([_text_turn("ok")])
        result = _run(
            [
                "sessions",
                "resume",
                "sess_chunk",
                "more",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--yes",
                "--max-context-tokens",
                "200",
            ]
        )
        assert result.exit_code == 0, result.stdout
        # The pre-loaded session had 13 messages + the new user "more" = 14.
        # With a 200-token budget the pruner must drop several middle messages.
        assert RecordingAdapter.call_count == 1
        assert len(RecordingAdapter.last_messages) < 14

    def test_flag_without_resume_runs_clean(
        self, patch_adapter, db_path: Path, tmp_path: Path
    ) -> None:
        patch_adapter([_text_turn("answer")])
        result = _run(
            [
                "run",
                "hi",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--yes",
                "--max-context-tokens",
                "10000",
            ]
        )
        assert result.exit_code == 0, result.stdout
        # New session → only the new user message; no pruning happens.
        assert RecordingAdapter.call_count == 1
        assert len(RecordingAdapter.last_messages) == 1
        assert RecordingAdapter.last_messages[0].role == "user"

    def test_no_flag_no_pruning_even_with_huge_history(
        self, patch_adapter, db_path: Path, tmp_path: Path
    ) -> None:
        import asyncio

        asyncio.run(_preload_chunky_session(db_path, session_id="sess_big", n_messages=12))

        patch_adapter([_text_turn("ok")])
        result = _run(
            [
                "sessions",
                "resume",
                "sess_big",
                "more",
                "--cwd",
                str(tmp_path),
                "--db",
                str(db_path),
                "--yes",
            ]
        )
        assert result.exit_code == 0, result.stdout
        # Without --max-context-tokens, the full history (13 + new user "more")
        # is forwarded to the adapter untouched.
        assert RecordingAdapter.call_count == 1
        assert len(RecordingAdapter.last_messages) == 14
