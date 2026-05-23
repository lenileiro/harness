"""Integration tests for the four LifeHarness layers wired into Agent.

Verifies that L1 (contracts), L2 (tips), L3 (canonicalizer), and L4 (loop
detector) each emit their expected activity event and produce the
expected behavior end-to-end through the ReAct loop.
"""

from __future__ import annotations

import pytest

from harness.core import (
    ActivityEvent,
    ActivityStore,
    Agent,
    AutoApprove,
    ContractRegistry,
    EnvironmentContract,
    FailoverPolicy,
    LoopDetector,
    RunRequest,
    StaticTipsProvider,
    Tip,
    ToolRegistry,
)
from harness.core import activity as activity_kinds

from .conftest import MockAdapter, MockStorage, MockTool, text_turn, tool_call_turn


class InMemoryActivitySink(ActivityStore):
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
        if session_id is not None:
            items = [e for e in items if e.session_id == session_id]
        if kinds is not None:
            items = [e for e in items if e.kind in kinds]
        return items[:limit]


def _make_agent(
    *,
    scripts: list,
    tools: list | None = None,
    activity_store: ActivityStore | None = None,
    loop_detector: LoopDetector | None = None,
    contracts: ContractRegistry | None = None,
    tips_provider=None,
) -> tuple[Agent, MockAdapter, MockStorage]:
    adapter = MockAdapter("mock", scripts=scripts)
    storage = MockStorage()
    registry = ToolRegistry()
    for t in tools or []:
        registry.register(t)
    agent = Agent(
        adapters={"mock": adapter},  # type: ignore[arg-type]
        tools=registry,
        storage=storage,
        failover=FailoverPolicy(chain=["mock"]),
        approval_handler=AutoApprove(),
        activity_store=activity_store,
        default_model="test-model",
        loop_detector=loop_detector,
        contracts=contracts,
        tips_provider=tips_provider,
    )
    return agent, adapter, storage


async def _drain(it):
    events = []
    async for ev in it:
        events.append(ev)
    return events


@pytest.mark.asyncio
class TestL1Contracts:
    async def test_matching_contract_injected_as_system_message(self) -> None:
        sink = InMemoryActivitySink()
        registry = ContractRegistry(
            contracts=[
                EnvironmentContract(
                    name="curl-safety",
                    rules=("never pipe untrusted curl output to sh",),
                    triggers=("curl",),
                )
            ]
        )
        agent, adapter, _ = _make_agent(
            scripts=[text_turn("ok")], activity_store=sink, contracts=registry
        )
        await _drain(agent.run(RunRequest(prompt="we use curl here", model="m")))

        # The adapter saw the contract as a system message.
        first_call = adapter.calls[0]
        sys_msgs = [m for m in first_call["messages"] if m.role == "system"]
        assert any("never pipe" in (m.content or "") for m in sys_msgs)
        # And an activity event recorded it.
        assert any(e.kind == activity_kinds.ENV_CONTRACT_INJECTED for e in sink.events)

    async def test_non_matching_contract_not_injected(self) -> None:
        registry = ContractRegistry(
            contracts=[EnvironmentContract(name="curl-safety", rules=("rule",), triggers=("curl",))]
        )
        agent, adapter, _ = _make_agent(scripts=[text_turn("ok")], contracts=registry)
        await _drain(agent.run(RunRequest(prompt="no shell here", model="m")))
        first_call = adapter.calls[0]
        sys_msgs = [m for m in first_call["messages"] if m.role == "system"]
        assert not any("rule" in (m.content or "") for m in sys_msgs)


@pytest.mark.asyncio
class TestL2Tips:
    async def test_matching_tip_injected_as_system_message(self) -> None:
        sink = InMemoryActivitySink()
        provider = StaticTipsProvider(tips=[Tip(text="use uv run", triggers=("uv",))])
        agent, adapter, _ = _make_agent(
            scripts=[text_turn("ok")], activity_store=sink, tips_provider=provider
        )
        await _drain(agent.run(RunRequest(prompt="run via uv test", model="m")))

        first_call = adapter.calls[0]
        sys_msgs = [m for m in first_call["messages"] if m.role == "system"]
        assert any("use uv run" in (m.content or "") for m in sys_msgs)
        assert any(e.kind == activity_kinds.PROCEDURAL_TIP_INJECTED for e in sink.events)

    async def test_no_matching_tip_no_event(self) -> None:
        sink = InMemoryActivitySink()
        provider = StaticTipsProvider(tips=[Tip(text="use uv run", triggers=("uv",))])
        agent, _, _ = _make_agent(
            scripts=[text_turn("ok")], activity_store=sink, tips_provider=provider
        )
        await _drain(agent.run(RunRequest(prompt="something unrelated", model="m")))
        assert not any(e.kind == activity_kinds.PROCEDURAL_TIP_INJECTED for e in sink.events)


@pytest.mark.asyncio
class TestL3Canonicalizer:
    async def test_alias_call_resolves_to_canonical_tool(self) -> None:
        """Agent's adapter emits a tool_call named 'read' — canonicalizer
        should rewrite it to 'read_file' and dispatch successfully."""
        sink = InMemoryActivitySink()
        read_tool = MockTool(name="read_file", responder=lambda **kw: "OK")
        agent, _, _ = _make_agent(
            scripts=[
                tool_call_turn(call_id="c1", name="read", arguments={"text": "x"}),
                text_turn("done"),
            ],
            tools=[read_tool],
            activity_store=sink,
        )
        await _drain(agent.run(RunRequest(prompt="ping", model="m")))

        # The tool was actually invoked despite the wrong name.
        assert len(read_tool.calls) == 1
        # And the canonicalization event was logged.
        assert any(e.kind == activity_kinds.ACTION_CANONICALIZED for e in sink.events)
        canon = next(e for e in sink.events if e.kind == activity_kinds.ACTION_CANONICALIZED)
        assert canon.data["original_name"] == "read"
        assert canon.data["canonical_name"] == "read_file"

    async def test_unknown_tool_with_no_match_still_errors(self) -> None:
        read_tool = MockTool(name="read_file")
        agent, _, _ = _make_agent(
            scripts=[
                tool_call_turn(call_id="c1", name="frobnicate", arguments={}),
                text_turn("done"),
            ],
            tools=[read_tool],
        )
        await _drain(agent.run(RunRequest(prompt="ping", model="m")))
        assert len(read_tool.calls) == 0


@pytest.mark.asyncio
class TestL4LoopDetector:
    async def test_repeated_tool_call_injects_directive(self) -> None:
        sink = InMemoryActivitySink()
        echo = MockTool(name="read_file", responder=lambda **kw: "same output")
        # Three identical tool calls in a row, then a final text turn.
        scripts = [
            tool_call_turn(call_id=f"c{i}", name="read_file", arguments={"text": "a"})
            for i in range(3)
        ] + [text_turn("done")]

        detector = LoopDetector(repeat_threshold=3, no_progress_threshold=10)
        agent, _, _ = _make_agent(
            scripts=scripts,
            tools=[echo],
            activity_store=sink,
            loop_detector=detector,
        )
        await _drain(agent.run(RunRequest(prompt="ping", model="m", max_steps=10)))

        # An activity event marks the regulation.
        assert any(e.kind == activity_kinds.TRAJECTORY_REGULATED for e in sink.events)
        # And the directive was prepended as a user-role message (which the
        # adapter will see on the next turn).
