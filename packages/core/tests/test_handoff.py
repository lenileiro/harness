"""Tests for agent handoff (Swarm-style control delegation)."""

from __future__ import annotations

import pytest

from harness.core import (
    Done,
    Handoff,
    HandoffEvent,
    RunRequest,
    ToolResultEvent,
)
from harness.core.handoff import HandoffTool
from harness.core.schemas import ToolCall

from .conftest import MockAdapter, MockStorage, text_turn, tool_call_turn


def make_agent(adapters, tools=None, default_cwd="/tmp"):
    from harness.core import Agent, FailoverPolicy, ToolRegistry

    storage = MockStorage()
    registry = ToolRegistry()
    for t in tools or []:
        registry.register(t)
    failover = FailoverPolicy(chain=list(adapters), max_attempts=1)
    return Agent(
        adapters=adapters,
        tools=registry,
        storage=storage,
        failover=failover,
        default_model="test-model",
        default_cwd=default_cwd,
    )


async def collect(it):
    out = []
    async for e in it:
        out.append(e)
    return out


@pytest.mark.asyncio
class TestHandoff:
    async def test_handoff_emits_handoff_event(self, tmp_path) -> None:
        """Raising Handoff from a tool produces a HandoffEvent in the stream."""
        specialist_adapter = MockAdapter(
            "specialist",
            scripts=[text_turn("specialist answer")],
        )
        specialist = make_agent({"specialist": specialist_adapter}, default_cwd=str(tmp_path))

        handoff_tool = HandoffTool(specialist, name="to_specialist")

        router_adapter = MockAdapter(
            "router",
            scripts=[
                tool_call_turn(
                    call_id="h1", name="to_specialist", arguments={"reason": "needs expertise"}
                ),
            ],
        )
        router = make_agent(
            {"router": router_adapter},
            tools=[handoff_tool],
            default_cwd=str(tmp_path),
        )

        events = await collect(router.run(RunRequest(prompt="help me")))

        handoff_events = [e for e in events if isinstance(e, HandoffEvent)]
        assert len(handoff_events) == 1
        assert handoff_events[0].reason == "needs expertise"

    async def test_specialist_produces_final_answer(self, tmp_path) -> None:
        """After handoff, the specialist's Done event is yielded."""
        specialist_adapter = MockAdapter(
            "specialist",
            scripts=[text_turn("the expert answer")],
        )
        specialist = make_agent({"specialist": specialist_adapter}, default_cwd=str(tmp_path))

        handoff_tool = HandoffTool(specialist)

        router_adapter = MockAdapter(
            "router",
            scripts=[
                tool_call_turn(call_id="h1", name="handoff", arguments={"reason": "delegating"}),
            ],
        )
        router = make_agent(
            {"router": router_adapter},
            tools=[handoff_tool],
            default_cwd=str(tmp_path),
        )

        events = await collect(router.run(RunRequest(prompt="question")))

        done_events = [e for e in events if isinstance(e, Done)]
        assert done_events, "expected a Done event from specialist"
        assert done_events[-1].final_message is not None
        assert "expert answer" in (done_events[-1].final_message.content or "")

    async def test_handoff_target_name_in_event(self, tmp_path) -> None:
        """HandoffEvent.target_name reflects the specialist agent type name."""
        specialist_adapter = MockAdapter("sp", scripts=[text_turn("ok")])
        specialist = make_agent({"sp": specialist_adapter}, default_cwd=str(tmp_path))

        handoff_tool = HandoffTool(specialist, name="delegate", description="delegate")

        router_adapter = MockAdapter(
            "router",
            scripts=[tool_call_turn(call_id="h1", name="delegate", arguments={"reason": "reason"})],
        )
        router = make_agent(
            {"router": router_adapter}, tools=[handoff_tool], default_cwd=str(tmp_path)
        )

        events = await collect(router.run(RunRequest(prompt="hi")))

        he = next((e for e in events if isinstance(e, HandoffEvent)), None)
        assert he is not None
        # target_name should be a non-empty string
        assert he.target_name

    async def test_handoff_tool_raises_handoff_directly(self, tmp_path) -> None:
        """HandoffTool.__call__ raises Handoff, not returns a ToolResult."""
        specialist_adapter = MockAdapter("sp", scripts=[text_turn("done")])
        specialist = make_agent({"sp": specialist_adapter}, default_cwd=str(tmp_path))

        tool = HandoffTool(specialist)
        call = ToolCall(id="x1", name="handoff", arguments={"reason": "test"})

        with pytest.raises(Handoff) as exc_info:
            await tool(call)

        assert exc_info.value.target is specialist
        assert exc_info.value.reason == "test"

    async def test_no_handoff_tool_result_in_stream(self, tmp_path) -> None:
        """When handoff fires, no ToolResultEvent for the handoff call is emitted."""
        specialist_adapter = MockAdapter("sp", scripts=[text_turn("answer")])
        specialist = make_agent({"sp": specialist_adapter}, default_cwd=str(tmp_path))

        handoff_tool = HandoffTool(specialist)

        router_adapter = MockAdapter(
            "router",
            scripts=[tool_call_turn(call_id="h1", name="handoff", arguments={"reason": "go"})],
        )
        router = make_agent(
            {"router": router_adapter}, tools=[handoff_tool], default_cwd=str(tmp_path)
        )

        events = await collect(router.run(RunRequest(prompt="hi")))

        # No ToolResultEvent for the handoff call itself — control just shifts.
        tool_results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert not tool_results, "handoff should not emit a ToolResultEvent"
