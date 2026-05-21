"""Runtime tests for the approval-inbox flow: queue → grant → replay."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.core import (
    Agent,
    ApprovalStore,
    AutoApprove,
    FailoverPolicy,
    InboxApprovalHandler,
    PendingApproval,
    RunRequest,
    ToolRegistry,
    ToolResultEvent,
)
from harness.core import activity as activity_kinds
from harness.core.activity import ActivityEvent, ActivityStore
from harness.core.approval import ApprovalStatus

from .conftest import MockAdapter, MockStorage, MockTool, text_turn, tool_call_turn


class InMemoryApprovalStore(ApprovalStore):
    """Minimal ApprovalStore for runtime tests."""

    def __init__(self) -> None:
        self._items: dict[str, PendingApproval] = {}

    async def create_approval(self, approval: PendingApproval) -> PendingApproval:
        self._items[approval.id] = approval.model_copy(deep=True)
        return approval

    async def get_approval(self, approval_id: str) -> PendingApproval | None:
        s = self._items.get(approval_id)
        return s.model_copy(deep=True) if s else None

    async def list_approvals(
        self,
        *,
        session_id: str | None = None,
        task_id: str | None = None,
        status: ApprovalStatus | None = None,
        limit: int = 100,
    ) -> list[PendingApproval]:
        items = list(self._items.values())
        if session_id is not None:
            items = [a for a in items if a.session_id == session_id]
        if task_id is not None:
            items = [a for a in items if a.task_id == task_id]
        if status is not None:
            items = [a for a in items if a.status == status]
        return [a.model_copy(deep=True) for a in items[:limit]]

    async def resolve_approval(
        self,
        approval_id: str,
        *,
        status: ApprovalStatus,
        resolved_by: str | None = None,
    ) -> PendingApproval | None:
        stored = self._items.get(approval_id)
        if stored is None:
            return None
        from datetime import UTC, datetime

        stored.status = status
        stored.resolved_at = datetime.now(UTC)
        stored.resolved_by = resolved_by
        return stored.model_copy(deep=True)

    async def mark_replayed(self, approval_id: str) -> None:
        from datetime import UTC, datetime

        stored = self._items.get(approval_id)
        if stored is None or stored.replayed_at is not None:
            return
        stored.replayed_at = datetime.now(UTC)

    async def list_unreplayed_granted(self, *, session_id: str) -> list[PendingApproval]:
        items = [
            a
            for a in self._items.values()
            if a.session_id == session_id and a.status == "granted" and a.replayed_at is None
        ]
        return [a.model_copy(deep=True) for a in items]


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
        return list(self.events)[:limit]


async def _drain(it):
    async for _ in it:
        pass


def _build_agent(
    *,
    adapter: MockAdapter,
    tool: MockTool,
    approval_store: ApprovalStore | None = None,
    activity_store: ActivityStore | None = None,
    inbox: bool = True,
    storage: MockStorage | None = None,
) -> tuple[Agent, MockStorage]:
    sess_store = storage or MockStorage()
    registry = ToolRegistry()
    registry.register(tool)
    handler: object
    if inbox:
        assert approval_store is not None
        handler = InboxApprovalHandler(approval_store=approval_store)
    else:
        handler = AutoApprove()
    return (
        Agent(
            adapters={"mock": adapter},  # type: ignore[arg-type]
            tools=registry,
            storage=sess_store,
            failover=FailoverPolicy(chain=["mock"], max_attempts=1),
            approval_handler=handler,  # type: ignore[arg-type]
            activity_store=activity_store,
            approval_store=approval_store,
            default_model="m",
        ),
        sess_store,
    )


@pytest.mark.asyncio
class TestQueueFlow:
    async def test_prompt_tool_is_queued_not_executed(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="shell", arguments={"text": "x"}),
                text_turn("done"),
            ],
        )
        tool = MockTool(name="shell", approval="prompt")
        approvals = InMemoryApprovalStore()
        activity = InMemoryActivitySink()
        agent, _storage = _build_agent(
            adapter=adapter,
            tool=tool,
            approval_store=approvals,
            activity_store=activity,
        )

        events: list = []
        async for e in agent.run(RunRequest(prompt="hi", session_id="s1", model="m")):
            events.append(e)

        # Tool was NOT executed.
        assert tool.calls == []
        # The user sees a "queued" tool_result.
        result = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result.result.is_error is True
        assert "queued for approval" in result.result.content
        # A PendingApproval was created.
        pending = await approvals.list_approvals()
        assert len(pending) == 1
        assert pending[0].tool_name == "shell"
        # The right activity kinds were emitted.
        kinds = [e.kind for e in activity.events]
        assert activity_kinds.APPROVAL_REQUESTED in kinds
        assert activity_kinds.APPROVAL_QUEUED in kinds


@pytest.mark.asyncio
class TestReplayFlow:
    async def test_granted_approval_replays_on_resume(self, tmp_path: Path) -> None:
        # First turn: queue the tool call.
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "ping"}),
                text_turn("queued, awaiting approval"),
            ],
        )
        tool = MockTool(
            name="echo",
            approval="prompt",
            responder=lambda **kw: f"REAL: {kw.get('text')}",
        )
        approvals = InMemoryApprovalStore()
        activity = InMemoryActivitySink()
        agent, storage = _build_agent(
            adapter=adapter,
            tool=tool,
            approval_store=approvals,
            activity_store=activity,
        )

        await _drain(agent.run(RunRequest(prompt="echo", session_id="s1", model="m")))
        assert tool.calls == []

        # Grant the approval out-of-band.
        [pending] = await approvals.list_approvals()
        await approvals.resolve_approval(pending.id, status="granted", resolved_by="test")

        # Second turn: resume. The runtime must replay the queued call
        # before sending anything to the model.
        adapter.scripts = [text_turn("ok now")]
        await _drain(agent.resume("s1", prompt="continue"))

        # Tool was executed with the original arguments.
        assert tool.calls == [{"text": "ping"}]

        # The corresponding tool message in transcript now carries the real
        # result (not the queued placeholder).
        stored = await storage.get("s1")
        assert stored is not None
        tool_msg = next(m for m in stored.messages if m.role == "tool" and m.tool_call_id == "c1")
        assert tool_msg.content == "REAL: ping"

        # Approval is marked replayed.
        replayed = await approvals.get_approval(pending.id)
        assert replayed is not None
        assert replayed.replayed_at is not None

        # APPROVAL_REPLAYED activity fired.
        assert any(e.kind == activity_kinds.APPROVAL_REPLAYED for e in activity.events)

    async def test_denied_approval_is_not_replayed(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="shell", arguments={"text": "x"}),
                text_turn("queued"),
            ],
        )
        tool = MockTool(name="shell", approval="prompt")
        approvals = InMemoryApprovalStore()
        agent, _ = _build_agent(adapter=adapter, tool=tool, approval_store=approvals)

        await _drain(agent.run(RunRequest(prompt="x", session_id="s1", model="m")))
        [pending] = await approvals.list_approvals()
        await approvals.resolve_approval(pending.id, status="denied", resolved_by="t")

        adapter.scripts = [text_turn("done")]
        await _drain(agent.resume("s1", prompt="continue"))

        # Tool still hasn't run.
        assert tool.calls == []
        # Approval was not replayed.
        denied = await approvals.get_approval(pending.id)
        assert denied is not None
        assert denied.replayed_at is None

    async def test_replay_handles_unknown_tool_gracefully(self, tmp_path: Path) -> None:
        # Build a session by queueing a call, then "unregister" the tool by
        # constructing a new agent without it.
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "x"}),
                text_turn("queued"),
            ],
        )
        tool = MockTool(name="echo", approval="prompt")
        approvals = InMemoryApprovalStore()
        agent, storage = _build_agent(adapter=adapter, tool=tool, approval_store=approvals)

        await _drain(agent.run(RunRequest(prompt="echo", session_id="s1", model="m")))
        [pending] = await approvals.list_approvals()
        await approvals.resolve_approval(pending.id, status="granted")

        # New agent with an empty registry — replay must not crash, just
        # write a sensible placeholder.
        empty_registry = ToolRegistry()
        agent2 = Agent(
            adapters={"mock": adapter},  # type: ignore[arg-type]
            tools=empty_registry,
            storage=storage,
            failover=FailoverPolicy(chain=["mock"], max_attempts=1),
            approval_handler=AutoApprove(),
            approval_store=approvals,
            default_model="m",
        )
        adapter.scripts = [text_turn("ok")]
        await _drain(agent2.resume("s1", prompt="continue"))

        stored = await storage.get("s1")
        assert stored is not None
        tool_msg = next(m for m in stored.messages if m.tool_call_id == "c1")
        assert "replay failed" in (tool_msg.content or "")
        # Replay still marked done so we don't loop forever.
        replayed = await approvals.get_approval(pending.id)
        assert replayed is not None
        assert replayed.replayed_at is not None
