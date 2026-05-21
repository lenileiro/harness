"""Tests for WorkQueue, AgentRole, OrchestratorEvent types, and MultiAgentOrchestrator."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from harness.core.events import Done, TextDelta
from harness.core.orchestrator import (
    AgentDoneEvent,
    AgentEventWrapper,
    AgentRole,
    AgentStartedEvent,
    MultiAgentOrchestrator,
    OrchestratorEvent,
    WorkItemClaimedEvent,
    WorkQueue,
)
from harness.storage.memory import InMemoryStorage

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _store() -> InMemoryStorage:
    return InMemoryStorage()


async def _make_queue(store: InMemoryStorage, n_items: int = 1) -> tuple[WorkQueue, str]:
    """Return (queue, parent_id) with n_items todo work items already pushed."""
    parent_id = f"job_{uuid.uuid4().hex[:8]}"
    q = WorkQueue(store, parent_id=parent_id)
    for i in range(n_items):
        await q.push(title=f"item {i}", cwd=Path("/tmp"))
    return q, parent_id


# ---------------------------------------------------------------------------
# WorkQueue tests
# ---------------------------------------------------------------------------


class TestWorkQueue:
    @pytest.mark.asyncio
    async def test_push_and_claim(self) -> None:
        store = _store()
        q, _ = await _make_queue(store, n_items=2)
        t1 = await q.claim(claimed_by="worker-0")
        t2 = await q.claim(claimed_by="worker-1")
        t3 = await q.claim(claimed_by="worker-2")
        assert t1 is not None
        assert t2 is not None
        assert t1.id != t2.id
        assert t3 is None  # queue empty

    @pytest.mark.asyncio
    async def test_claim_sets_status_in_progress(self) -> None:
        store = _store()
        q, _ = await _make_queue(store, n_items=1)
        task = await q.claim(claimed_by="worker-0")
        assert task is not None
        assert task.status == "in_progress"
        assert task.metadata.get("claimed_by") == "worker-0"

    @pytest.mark.asyncio
    async def test_is_drained_false_while_in_progress(self) -> None:
        store = _store()
        q, _ = await _make_queue(store, n_items=1)
        await q.claim(claimed_by="worker-0")  # now in_progress, not done
        assert await q.is_drained() is False

    @pytest.mark.asyncio
    async def test_is_drained_true_after_all_done(self) -> None:
        store = _store()
        q, _ = await _make_queue(store, n_items=2)
        t1 = await q.claim(claimed_by="w0")
        t2 = await q.claim(claimed_by="w1")
        assert t1 and t2
        await q.complete(t1.id, summary="done 1")
        await q.complete(t2.id, summary="done 2")
        assert await q.is_drained() is True

    @pytest.mark.asyncio
    async def test_is_drained_true_when_empty(self) -> None:
        store = _store()
        parent_id = "job_empty"
        q = WorkQueue(store, parent_id=parent_id)
        assert await q.is_drained() is True

    @pytest.mark.asyncio
    async def test_list_all_items(self) -> None:
        store = _store()
        q, _ = await _make_queue(store, n_items=3)
        items = await q.list()
        assert len(items) == 3

    @pytest.mark.asyncio
    async def test_list_filtered_by_status(self) -> None:
        store = _store()
        q, _ = await _make_queue(store, n_items=3)
        await q.claim(claimed_by="w0")
        todo = await q.list(status="todo")
        in_progress = await q.list(status="in_progress")
        assert len(todo) == 2
        assert len(in_progress) == 1

    @pytest.mark.asyncio
    async def test_complete_missing_task_raises(self) -> None:
        store = _store()
        q = WorkQueue(store, parent_id="job_x")
        with pytest.raises(KeyError):
            await q.complete("nonexistent_id")

    @pytest.mark.asyncio
    async def test_concurrent_claim_no_double_claim(self) -> None:
        store = _store()
        q, _ = await _make_queue(store, n_items=1)

        results = await asyncio.gather(
            q.claim(claimed_by="worker-A"),
            q.claim(claimed_by="worker-B"),
        )
        claimed = [r for r in results if r is not None]
        nones = [r for r in results if r is None]
        assert len(claimed) == 1
        assert len(nones) == 1

    @pytest.mark.asyncio
    async def test_concurrent_claim_distributes_3_items_to_5_workers(self) -> None:
        store = _store()
        q, _ = await _make_queue(store, n_items=3)

        tasks = [q.claim(claimed_by=f"worker-{i}") for i in range(5)]
        results = await asyncio.gather(*tasks)
        claimed = [r for r in results if r is not None]
        # Each claimed item should have a distinct id
        claimed_ids = {t.id for t in claimed}
        assert len(claimed_ids) == 3


# ---------------------------------------------------------------------------
# AgentRole tests
# ---------------------------------------------------------------------------


class TestAgentRole:
    def test_fields(self) -> None:
        role = AgentRole(
            name="planner",
            system_prompt="You are a planner.",
            max_instances=1,
        )
        assert role.name == "planner"
        assert role.max_instances == 1
        assert role.tool_names is None
        assert role.model is None
        assert role.current_phase is None

    def test_copy_with_name_override(self) -> None:
        role = AgentRole(name="worker", system_prompt="You are a worker.")
        updated = role.model_copy(update={"name": "worker-2"})
        assert updated.name == "worker-2"
        assert role.name == "worker"  # original unchanged


# ---------------------------------------------------------------------------
# OrchestratorEvent tests
# ---------------------------------------------------------------------------


class TestOrchestratorEvents:
    def test_agent_started_fields(self) -> None:
        e = AgentStartedEvent(role="planner", session_id="sess_001")
        assert e.type == "agent_started"
        assert e.role == "planner"

    def test_agent_done_default_turn_count(self) -> None:
        e = AgentDoneEvent(role="reporter", session_id="sess_002")
        assert e.turn_count == 0

    def test_work_item_claimed_fields(self) -> None:
        e = WorkItemClaimedEvent(task_id="t1", task_ref="T-001", worker_session_id="w0_abc")
        assert e.task_ref == "T-001"
        assert e.type == "work_item_claimed"

    def test_agent_event_wrapper_wraps_core_event(self) -> None:
        inner = TextDelta(text="hello")
        wrapper = AgentEventWrapper(role="planner", event=inner)
        assert wrapper.type == "agent_event"
        assert wrapper.role == "planner"
        assert isinstance(wrapper.event, TextDelta)

    def test_agent_event_wrapper_roundtrip(self) -> None:
        inner = Done(final_message=None, usage=None)
        wrapper = AgentEventWrapper(role="reporter", event=inner)
        data = wrapper.model_dump(mode="json")
        restored = AgentEventWrapper.model_validate(data)
        assert restored.role == "reporter"
        assert isinstance(restored.event, Done)


# ---------------------------------------------------------------------------
# MultiAgentOrchestrator integration
# ---------------------------------------------------------------------------


def _make_simple_agent_factory(store: InMemoryStorage) -> Callable[[AgentRole], Any]:
    """Return an agent_factory that produces a stub agent yielding a Done event.

    For worker roles, also marks the claimed task as done so orphan-reset does
    not re-queue the item indefinitely during tests.
    """

    class StubAgent:
        def __init__(self, item_id: str | None) -> None:
            self._item_id = item_id

        async def run(self, request):
            if self._item_id:
                task = await store.get_task(self._item_id)
                if task and task.status == "in_progress":
                    updated = task.model_copy(
                        update={
                            "status": "done",
                            "metadata": {**task.metadata, "result_summary": "stub done"},
                        }
                    )
                    await store.update_task(updated)
            yield TextDelta(text="thinking...")
            yield Done(final_message=None, usage=None)

    def factory(role: AgentRole) -> StubAgent:
        return StubAgent(item_id=role.item_id)

    return factory


class TestMultiAgentOrchestrator:
    @pytest.mark.asyncio
    async def test_orchestrator_emits_lifecycle_events(self) -> None:
        store = _store()
        # Pre-populate 2 work items so workers have something to do
        parent_id = f"job_{uuid.uuid4().hex[:8]}"
        q = WorkQueue(store, parent_id=parent_id)
        await q.push(title="task A", cwd=Path("/tmp"))
        await q.push(title="task B", cwd=Path("/tmp"))

        factory = _make_simple_agent_factory(store)
        planner_role = AgentRole(name="planner", system_prompt="Plan.")
        worker_role = AgentRole(name="worker", system_prompt="Work.")
        reporter_role = AgentRole(name="reporter", system_prompt="Report.")

        orchestrator = MultiAgentOrchestrator(
            agent_factory=factory,
            store=store,
            planner_role=planner_role,
            worker_role=worker_role,
            reporter_role=reporter_role,
            max_workers=2,
            job_cwd=Path("/tmp"),
            provider="ollama",
            model="llama3.2",
        )

        events: list[OrchestratorEvent] = []
        async for event in orchestrator.run("do something", job_id=parent_id):
            events.append(event)

        types = [type(e).__name__ for e in events]

        assert "AgentStartedEvent" in types
        assert "AgentDoneEvent" in types

    @pytest.mark.asyncio
    async def test_orchestrator_claims_work_items(self) -> None:
        store = _store()
        parent_id = f"job_{uuid.uuid4().hex[:8]}"
        q = WorkQueue(store, parent_id=parent_id)
        await q.push(title="eval model A", cwd=Path("/tmp"))
        await q.push(title="eval model B", cwd=Path("/tmp"))
        await q.push(title="eval model C", cwd=Path("/tmp"))

        factory = _make_simple_agent_factory(store)
        orchestrator = MultiAgentOrchestrator(
            agent_factory=factory,
            store=store,
            planner_role=AgentRole(name="planner", system_prompt="Plan."),
            worker_role=AgentRole(name="worker", system_prompt="Work."),
            reporter_role=AgentRole(name="reporter", system_prompt="Report."),
            max_workers=3,
            job_cwd=Path("/tmp"),
            provider="ollama",
            model="llama3.2",
        )

        claimed_events: list[WorkItemClaimedEvent] = []
        async for event in orchestrator.run("evaluate models", job_id=parent_id):
            if isinstance(event, WorkItemClaimedEvent):
                claimed_events.append(event)

        # 3 items, each should be claimed exactly once
        assert len(claimed_events) == 3
        claimed_refs = {e.task_ref for e in claimed_events}
        assert len(claimed_refs) == 3  # no duplicates

    @pytest.mark.asyncio
    async def test_orchestrator_full_flow_phases(self) -> None:
        store = _store()
        parent_id = f"job_{uuid.uuid4().hex[:8]}"
        q = WorkQueue(store, parent_id=parent_id)
        await q.push(title="item 1", cwd=Path("/tmp"))

        factory = _make_simple_agent_factory(store)
        orchestrator = MultiAgentOrchestrator(
            agent_factory=factory,
            store=store,
            planner_role=AgentRole(name="planner", system_prompt="Plan."),
            worker_role=AgentRole(name="worker", system_prompt="Work."),
            reporter_role=AgentRole(name="reporter", system_prompt="Report."),
            max_workers=1,
            job_cwd=Path("/tmp"),
            provider="ollama",
            model="llama3.2",
        )

        roles_seen: list[str] = []
        async for event in orchestrator.run("do work", job_id=parent_id):
            if isinstance(event, AgentStartedEvent):
                roles_seen.append(event.role)

        # Planner runs first, then worker(s), then reporter
        assert roles_seen[0] == "planner"
        assert roles_seen[-1] == "reporter"
        assert any("worker" in r for r in roles_seen)
