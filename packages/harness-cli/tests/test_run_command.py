"""End-to-end CLI integration tests for `harness run`.

The Ollama adapter is swapped for a deterministic in-process fake so these
tests run without any external daemon, while exercising the real Agent +
ToolRegistry + InMemoryStorage + Rich-rendering wiring.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
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

# ---------------------------------------------------------------------------
# Fake adapter
# ---------------------------------------------------------------------------


class FakeAdapter:
    """Drop-in replacement for OllamaAdapter that replays a fixed event list.

    Scripts pop from the class-level queue on every stream() call so a test
    can drive multiple CLI invocations from a single pre-loaded list.
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
        for event in FakeAdapter.next_script.pop(0):
            yield event

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=True)

    async def cancel(self, session_id: str) -> None:
        pass


@pytest.fixture
def patch_adapter(monkeypatch: pytest.MonkeyPatch):
    """Swap the real OllamaAdapter for FakeAdapter and let the test pre-load scripts."""
    monkeypatch.setattr(cli_main, "OllamaAdapter", FakeAdapter)

    def configure(scripts: list[list[Event]]) -> None:
        FakeAdapter.next_script = scripts

    yield configure
    FakeAdapter.next_script = []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunCommand:
    def test_text_response_streams_to_stdout(self, patch_adapter, tmp_path: Path) -> None:
        patch_adapter(
            [
                [
                    TextDelta(text="hello "),
                    TextDelta(text="world"),
                    Done(final_message=Message(role="assistant", content="hello world")),
                ]
            ]
        )
        runner = CliRunner()
        result = runner.invoke(
            cli_main.app,
            [
                "run",
                "say hi",
                "--cwd",
                str(tmp_path),
                "--model",
                "test-model",
                "--in-memory",
            ],
        )
        assert result.exit_code == 0, result.stdout
        assert "hello world" in result.stdout

    def test_tool_call_then_final_answer(self, patch_adapter, tmp_path: Path) -> None:
        (tmp_path / "note.txt").write_text("the answer is 42", encoding="utf-8")

        first_call = ToolCall(id="c1", name="read_file", arguments={"path": "note.txt"})
        patch_adapter(
            [
                [
                    ToolCallEvent(call=first_call),
                    Done(
                        final_message=Message(
                            role="assistant",
                            content=None,
                            tool_calls=[first_call],
                        )
                    ),
                ],
                [
                    TextDelta(text="the answer is 42"),
                    Done(final_message=Message(role="assistant", content="the answer is 42")),
                ],
            ]
        )

        runner = CliRunner()
        result = runner.invoke(
            cli_main.app,
            ["run", "what is in note.txt?", "--cwd", str(tmp_path), "--in-memory"],
        )
        assert result.exit_code == 0, result.stdout
        # Tool name shows up in the rendered tool-call line.
        assert "read_file" in result.stdout
        # Final answer streams to stdout.
        assert "the answer is 42" in result.stdout

    def test_missing_cwd_exits_2(self, tmp_path: Path) -> None:
        ghost = tmp_path / "does-not-exist"
        runner = CliRunner()
        result = runner.invoke(cli_main.app, ["run", "x", "--cwd", str(ghost), "--in-memory"])
        assert result.exit_code == 2
