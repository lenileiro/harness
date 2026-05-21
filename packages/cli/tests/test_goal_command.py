"""Tests for `harness goal` and `harness run --goal`."""

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


def _plan_turn(steps: list[str]) -> list[Event]:
    import json

    content = json.dumps({"steps": [{"description": s} for s in steps]})
    return [TextDelta(text=content), Done(final_message=Message(role="assistant", content=content))]


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


class TestGoalCommand:
    def test_goal_emits_step_started(self, tmp_path: Path, patch_adapter) -> None:
        # First call: planner gets plan JSON
        # Second call: agent executes the single step
        patch_adapter(
            [
                _plan_turn(["Write a greeting", "Summarize it"]),
                _text_turn("Hello from step 1"),
                _text_turn("Summary done"),
            ]
        )
        r = _run(
            [
                "goal",
                "greet the user",
                "--in-memory",
                "--cwd",
                str(tmp_path),
                "--yes",
            ]
        )
        assert r.exit_code == 0, r.stdout
        assert "Step" in r.stdout or "Hello from step 1" in r.stdout

    def test_run_with_goal_flag(self, tmp_path: Path, patch_adapter) -> None:
        patch_adapter(
            [
                _plan_turn(["Step one"]),
                _text_turn("done"),
            ]
        )
        r = _run(
            [
                "run",
                "--goal",
                "accomplish something",
                "--in-memory",
                "--cwd",
                str(tmp_path),
                "--yes",
            ]
        )
        assert r.exit_code == 0, r.stdout
        assert "done" in r.stdout

    def test_goal_fallback_on_bad_plan(self, tmp_path: Path, patch_adapter) -> None:
        # Planner returns bad JSON — LLMPlanner falls back to NoOpPlanner (1 step)
        patch_adapter(
            [
                [
                    TextDelta(text="this is not json"),
                    Done(final_message=Message(role="assistant", content="not json")),
                ],
                _text_turn("executed anyway"),
            ]
        )
        r = _run(
            [
                "goal",
                "do the thing",
                "--in-memory",
                "--cwd",
                str(tmp_path),
                "--yes",
            ]
        )
        assert r.exit_code == 0, r.stdout
        assert "executed anyway" in r.stdout
