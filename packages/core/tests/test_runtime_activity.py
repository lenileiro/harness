"""Tests that the Agent emits ActivityEvents to a configured ActivityStore."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.core import (
    ActivityEvent,
    ActivityStore,
    Agent,
    AutoApprove,
    FailoverPolicy,
    NetworkError,
    RunRequest,
    ToolRegistry,
)
from harness.core import activity as activity_kinds

from .conftest import MockAdapter, MockStorage, MockTool, text_turn, tool_call_turn


class InMemoryActivitySink(ActivityStore):
    """Minimal ActivityStore for tests; collects all events in order."""

    def __init__(self) -> None:
        self.events: list[ActivityEvent] = []

    async def append_activity(self, event: ActivityEvent) -> None:
        self.events.append(event)

    async def list_activity(
        self,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        kinds: tuple[str, ...] | None = None,
        limit: int = 200,
    ) -> list[ActivityEvent]:
        items = list(self.events)
        if task_id is not None:
            items = [e for e in items if e.task_id == task_id]
        if session_id is not None:
            items = [e for e in items if e.session_id == session_id]
        if kinds is not None:
            items = [e for e in items if e.kind in kinds]
        return items[:limit]


def _build_agent(adapters: dict, *, tools=None, activity_store=None) -> tuple[Agent, MockStorage]:
    storage = MockStorage()
    registry = ToolRegistry()
    for t in tools or []:
        registry.register(t)
    agent = Agent(
        adapters=adapters,  # type: ignore[arg-type]
        tools=registry,
        storage=storage,
        failover=FailoverPolicy(chain=list(adapters), max_attempts=2),
        approval_handler=AutoApprove(),
        activity_store=activity_store,
        default_model="test-model",
    )
    return agent, storage


async def _drain(it):
    async for _ in it:
        pass


@pytest.mark.asyncio
class TestAgentActivityEmission:
    async def test_happy_path_emits_lifecycle(self, tmp_path: Path) -> None:
        sink = InMemoryActivitySink()
        adapter = MockAdapter("mock", scripts=[text_turn("hi")])
        agent, _ = _build_agent({"mock": adapter}, activity_store=sink)
        await _drain(agent.run(RunRequest(prompt="ping", session_id="sess_demo", model="m")))

        kinds = [e.kind for e in sink.events]
        # Required lifecycle markers in order:
        assert kinds[0] == activity_kinds.AGENT_RUN_STARTED
        assert activity_kinds.STEP_STARTED in kinds
        assert activity_kinds.STEP_COMPLETED in kinds
        assert kinds[-1] == activity_kinds.AGENT_RUN_COMPLETED
        # Session id propagates to every event.
        assert all(e.session_id == "sess_demo" for e in sink.events)

    async def test_tool_loop_emits_dispatch_and_completion(self, tmp_path: Path) -> None:
        sink = InMemoryActivitySink()
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "x"}),
                text_turn("done"),
            ],
        )
        tool = MockTool(name="echo", approval="auto")
        agent, _ = _build_agent({"mock": adapter}, tools=[tool], activity_store=sink)
        await _drain(agent.run(RunRequest(prompt="echo x", model="m")))

        kinds = [e.kind for e in sink.events]
        assert activity_kinds.TOOL_CALL_DISPATCHED in kinds
        assert activity_kinds.TOOL_CALL_COMPLETED in kinds
        # No approval event when the tool is auto-approved.
        assert activity_kinds.APPROVAL_REQUESTED not in kinds

    async def test_prompt_approval_emits_requested_and_granted(self, tmp_path: Path) -> None:
        sink = InMemoryActivitySink()
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="ask", arguments={"text": "x"}),
                text_turn("ok"),
            ],
        )
        tool = MockTool(name="ask", approval="prompt")
        agent, _ = _build_agent({"mock": adapter}, tools=[tool], activity_store=sink)
        await _drain(agent.run(RunRequest(prompt="ask", model="m")))

        kinds = [e.kind for e in sink.events]
        assert activity_kinds.APPROVAL_REQUESTED in kinds
        assert activity_kinds.APPROVAL_GRANTED in kinds

    async def test_failure_emits_agent_run_failed(self, tmp_path: Path) -> None:
        sink = InMemoryActivitySink()
        adapter = MockAdapter("mock", error=NetworkError("down"))
        agent, _ = _build_agent({"mock": adapter}, activity_store=sink)
        await _drain(agent.run(RunRequest(prompt="x", model="m")))

        kinds = [e.kind for e in sink.events]
        assert activity_kinds.AGENT_RUN_FAILED in kinds
        # The failed event carries the classified kind.
        fail = next(e for e in sink.events if e.kind == activity_kinds.AGENT_RUN_FAILED)
        assert fail.data.get("kind") == "network"

    async def test_task_id_propagates_to_events(self, tmp_path: Path) -> None:
        sink = InMemoryActivitySink()
        adapter = MockAdapter("mock", scripts=[text_turn("hi")])
        agent, _ = _build_agent({"mock": adapter}, activity_store=sink)
        await _drain(
            agent.run(RunRequest(prompt="hi", session_id="s1", task_id="task_abc", model="m"))
        )

        assert all(e.task_id == "task_abc" for e in sink.events)

    async def test_no_emission_without_store(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("hi")])
        # No activity_store passed; should still run cleanly with no side effect.
        storage = MockStorage()
        registry = ToolRegistry()
        agent = Agent(
            adapters={"mock": adapter},  # type: ignore[arg-type]
            tools=registry,
            storage=storage,
            failover=FailoverPolicy(chain=["mock"], max_attempts=1),
            approval_handler=AutoApprove(),
            default_model="m",
        )
        # Just verify it doesn't blow up.
        await _drain(agent.run(RunRequest(prompt="hi", model="m")))
