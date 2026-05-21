"""Tests for LLMPlanner."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from harness.core.events import Done, Event, TextDelta
from harness.core.planner import LLMPlanner, PlanContext
from harness.core.schemas import Capabilities, Message


def _make_context() -> PlanContext:
    return PlanContext(
        session_id="sess_test",
        messages=[Message(role="user", content="do something")],
        available_tools=["read_file", "write_file"],
    )


class ScriptedAdapter:
    """Returns a scripted response."""

    def __init__(self, content: str) -> None:
        self._content = content

    async def stream(self, **_kwargs: Any) -> AsyncIterator[Event]:  # type: ignore[override]
        yield TextDelta(text=self._content)
        yield Done(final_message=Message(role="assistant", content=self._content))

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=False)

    async def cancel(self, session_id: str) -> None:
        pass


class FailingAdapter:
    """Always raises on stream."""

    async def stream(self, **_kwargs: Any) -> AsyncIterator[Event]:  # type: ignore[override]
        raise RuntimeError("adapter broken")
        yield  # make it a generator


@pytest.mark.asyncio
async def test_parses_valid_json_plan() -> None:
    content = '{"steps": [{"description": "Step A"}, {"description": "Step B"}]}'
    planner = LLMPlanner(adapter=ScriptedAdapter(content), model="m")
    plan = await planner.plan("do something", _make_context())

    assert len(plan.steps) == 2
    assert plan.steps[0].description == "Step A"
    assert plan.steps[1].description == "Step B"


@pytest.mark.asyncio
async def test_falls_back_on_invalid_json() -> None:
    planner = LLMPlanner(adapter=ScriptedAdapter("not json at all"), model="m")
    plan = await planner.plan("do something", _make_context())

    # Falls back to NoOpPlanner — single step wrapping the goal
    assert len(plan.steps) == 1
    assert plan.steps[0].description == "do something"


@pytest.mark.asyncio
async def test_falls_back_on_empty_steps() -> None:
    planner = LLMPlanner(adapter=ScriptedAdapter('{"steps": []}'), model="m")
    plan = await planner.plan("do something", _make_context())

    assert len(plan.steps) == 1
    assert plan.steps[0].description == "do something"


@pytest.mark.asyncio
async def test_falls_back_on_adapter_error() -> None:
    planner = LLMPlanner(adapter=FailingAdapter(), model="m")
    plan = await planner.plan("do something", _make_context())

    assert len(plan.steps) == 1
    assert plan.steps[0].description == "do something"


@pytest.mark.asyncio
async def test_falls_back_on_missing_steps_key() -> None:
    planner = LLMPlanner(adapter=ScriptedAdapter('{"result": "ok"}'), model="m")
    plan = await planner.plan("my goal", _make_context())

    assert len(plan.steps) == 1
    assert plan.steps[0].description == "my goal"


@pytest.mark.asyncio
async def test_five_steps_parsed() -> None:
    steps = [{"description": f"Step {i}"} for i in range(5)]
    content = f'{{"steps": {steps!r}}}'.replace("'", '"')
    planner = LLMPlanner(adapter=ScriptedAdapter(content), model="m")
    plan = await planner.plan("big goal", _make_context())

    assert len(plan.steps) == 5
