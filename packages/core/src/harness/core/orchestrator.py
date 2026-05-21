"""Multi-agent orchestration: planner → workers → reporter.

The orchestrator coordinates three sequential phases:

1. Planner — decomposes the user prompt into a queue of work items via
   CreateWorkItemTool calls.
2. Workers — N parallel agents each atomically claim one work item, execute
   it, and loop until the queue is empty.
3. Reporter — reads all completed items and synthesizes a final report.

All agent events are wrapped in AgentEventWrapper so callers can distinguish
which role produced each event.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from harness.core.events import Event
from harness.core.schemas import RunRequest
from harness.tasks.schemas import Task, TaskStatus
from harness.tasks.store import TaskStore

if TYPE_CHECKING:
    from harness.core.runtime import Agent


# ---------------------------------------------------------------------------
# AgentRole
# ---------------------------------------------------------------------------


class AgentRole(BaseModel):
    """Configuration for a role in a multi-agent job."""

    name: str
    description: str = ""
    system_prompt: str
    tool_names: list[str] | None = None
    max_instances: int = 1
    current_phase: str | None = None
    model: str | None = None
    # Injected by orchestrator at run time — not set by callers
    job_id: str | None = None
    item_id: str | None = None


# ---------------------------------------------------------------------------
# OrchestratorEvent types
# ---------------------------------------------------------------------------


class AgentStartedEvent(BaseModel):
    type: Literal["agent_started"] = "agent_started"
    role: str
    session_id: str


class AgentDoneEvent(BaseModel):
    type: Literal["agent_done"] = "agent_done"
    role: str
    session_id: str
    turn_count: int = 0


class WorkItemCreatedEvent(BaseModel):
    type: Literal["work_item_created"] = "work_item_created"
    task_id: str
    task_ref: str
    title: str


class WorkItemClaimedEvent(BaseModel):
    type: Literal["work_item_claimed"] = "work_item_claimed"
    task_id: str
    task_ref: str
    worker_session_id: str


class WorkItemCompletedEvent(BaseModel):
    type: Literal["work_item_completed"] = "work_item_completed"
    task_id: str
    task_ref: str
    summary: str = ""


class AgentEventWrapper(BaseModel):
    type: Literal["agent_event"] = "agent_event"
    role: str
    event: Event


OrchestratorEvent = (
    AgentStartedEvent
    | AgentDoneEvent
    | WorkItemCreatedEvent
    | WorkItemClaimedEvent
    | WorkItemCompletedEvent
    | AgentEventWrapper
)


# ---------------------------------------------------------------------------
# WorkQueue
# ---------------------------------------------------------------------------


class WorkQueue:
    """Thin wrapper over TaskStore scoped to a parent job task."""

    def __init__(self, store: TaskStore, parent_id: str) -> None:
        self._store = store
        self._parent_id = parent_id

    async def push(self, title: str, description: str | None = None, cwd: Path = Path(".")) -> Task:
        task = Task(
            ref="",
            title=title,
            description=description,
            status="todo",
            parent_id=self._parent_id,
            cwd=cwd,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        return await self._store.create_task(task)

    async def claim(self, claimed_by: str, worker_session_id: str | None = None) -> Task | None:
        return await self._store.claim_task(
            parent_id=self._parent_id,
            claimed_by=claimed_by,
            worker_session_id=worker_session_id,
        )

    async def complete(self, task_id: str, summary: str = "") -> Task:
        task = await self._store.get_task(task_id)
        if task is None:
            raise KeyError(f"task {task_id!r} not found")
        updated = task.model_copy(
            update={
                "status": "done",
                "metadata": {**task.metadata, "result_summary": summary},
                "updated_at": datetime.now(UTC),
            }
        )
        return await self._store.update_task(updated)

    async def list(self, *, status: TaskStatus | None = None) -> list[Task]:
        return await self._store.list_tasks(parent_id=self._parent_id, status=status)

    async def is_drained(self) -> bool:
        remaining = await self._store.list_tasks(parent_id=self._parent_id, status="todo")
        in_progress = await self._store.list_tasks(parent_id=self._parent_id, status="in_progress")
        return len(remaining) == 0 and len(in_progress) == 0


# ---------------------------------------------------------------------------
# MultiAgentOrchestrator
# ---------------------------------------------------------------------------


class MultiAgentOrchestrator:
    """Runs planner → N parallel workers → reporter, yielding OrchestratorEvent.

    Args:
        agent_factory: Callable[[AgentRole], Agent] that constructs a fresh
            Agent for the given role. The factory is called once per agent
            invocation and is responsible for wiring tools and storage.
        store: TaskStore for work-item persistence.
        planner_role / worker_role / reporter_role: Role configurations.
        max_workers: Number of concurrent worker agents.
        job_cwd: Working directory for the job root task.
        provider / model: Default LLM provider and model for RunRequest.
    """

    def __init__(
        self,
        *,
        agent_factory: Callable[[AgentRole], Agent],
        store: TaskStore,
        planner_role: AgentRole,
        worker_role: AgentRole,
        reporter_role: AgentRole,
        max_workers: int = 2,
        max_worker_steps: int = 10,
        job_cwd: Path = Path("."),
        provider: str,
        model: str,
    ) -> None:
        self._factory = agent_factory
        self._store = store
        self._planner_role = planner_role
        self._worker_role = worker_role
        self._reporter_role = reporter_role
        self._max_workers = max_workers
        self._max_worker_steps = max_worker_steps
        self._job_cwd = job_cwd
        self._provider = provider
        self._model = model

    async def run(
        self,
        prompt: str,
        *,
        job_id: str | None = None,
    ) -> AsyncIterator[OrchestratorEvent]:
        job_id = job_id or f"job_{uuid.uuid4().hex[:12]}"
        queue = WorkQueue(self._store, parent_id=job_id)

        root = Task(
            id=job_id,
            ref="",
            title=f"job: {prompt[:60]}",
            status="in_progress",
            parent_id=None,
            cwd=self._job_cwd,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        await self._store.create_task(root)

        async for event in self._run_planner(prompt, queue):
            yield event

        async for event in self._run_workers(queue):
            yield event

        async for event in self._run_reporter(queue):
            yield event

    async def _run_planner(self, prompt: str, queue: WorkQueue) -> AsyncIterator[OrchestratorEvent]:
        session_id = f"planner_{uuid.uuid4().hex[:8]}"
        role = self._planner_role.model_copy(update={"job_id": queue._parent_id})
        agent = self._factory(role)
        yield AgentStartedEvent(role="planner", session_id=session_id)
        request = RunRequest(
            prompt=prompt,
            session_id=session_id,
            provider=self._provider,
            model=self._planner_role.model or self._model,
        )
        turn_count = 0
        async for event in agent.run(request):
            turn_count += 1
            yield AgentEventWrapper(role="planner", event=event)
        yield AgentDoneEvent(role="planner", session_id=session_id, turn_count=turn_count)

    async def _run_workers(self, queue: WorkQueue) -> AsyncIterator[OrchestratorEvent]:
        output_queue: asyncio.Queue[OrchestratorEvent] = asyncio.Queue()
        done_event = asyncio.Event()

        async def worker(idx: int) -> None:
            worker_name = f"worker-{idx}"
            while True:
                session_id = f"w{idx}_{uuid.uuid4().hex[:6]}"
                task = await queue.claim(claimed_by=worker_name, worker_session_id=session_id)
                if task is None:
                    break
                await output_queue.put(
                    WorkItemClaimedEvent(
                        task_id=task.id,
                        task_ref=task.ref,
                        worker_session_id=session_id,
                    )
                )
                role = self._worker_role.model_copy(
                    update={
                        "name": worker_name,
                        "job_id": queue._parent_id,
                        "item_id": task.id,
                    }
                )
                agent = self._factory(role)
                item_prompt = (
                    f"Complete this work item.\n\n"
                    f"Ref: {task.ref}\n"
                    f"Title: {task.title}\n"
                    f"Description: {task.description or '(none)'}\n\n"
                    f"When finished, call complete_work_item with a summary of what you did."
                )
                request = RunRequest(
                    prompt=item_prompt,
                    session_id=session_id,
                    provider=self._provider,
                    model=role.model or self._model,
                    max_steps=self._max_worker_steps,
                )
                await output_queue.put(AgentStartedEvent(role=worker_name, session_id=session_id))
                async for event in agent.run(request):
                    await output_queue.put(AgentEventWrapper(role=worker_name, event=event))
                await output_queue.put(AgentDoneEvent(role=worker_name, session_id=session_id))

        worker_tasks = [asyncio.create_task(worker(i)) for i in range(self._max_workers)]

        async def _set_done() -> None:
            await asyncio.gather(*worker_tasks, return_exceptions=True)
            done_event.set()

        _drain_task = asyncio.create_task(_set_done())  # noqa: RUF006

        while not done_event.is_set() or not output_queue.empty():
            try:
                event = await asyncio.wait_for(output_queue.get(), timeout=0.05)
                yield event
            except TimeoutError:
                continue

    async def _run_reporter(self, queue: WorkQueue) -> AsyncIterator[OrchestratorEvent]:
        items = await queue.list()
        if not items:
            return
        summary_lines = [
            f"- {t.ref} [{t.status}] {t.title}: {t.metadata.get('result_summary', '')}"
            for t in items
        ]
        report_prompt = (
            "Synthesize these completed work items into a final report:\n\n"
            + "\n".join(summary_lines)
        )
        session_id = f"reporter_{uuid.uuid4().hex[:8]}"
        role = self._reporter_role.model_copy(update={"job_id": queue._parent_id})
        agent = self._factory(role)
        yield AgentStartedEvent(role="reporter", session_id=session_id)
        request = RunRequest(
            prompt=report_prompt,
            session_id=session_id,
            provider=self._provider,
            model=self._reporter_role.model or self._model,
        )
        turn_count = 0
        async for event in agent.run(request):
            turn_count += 1
            yield AgentEventWrapper(role="reporter", event=event)
        yield AgentDoneEvent(role="reporter", session_id=session_id, turn_count=turn_count)


__all__ = [
    "AgentDoneEvent",
    "AgentEventWrapper",
    "AgentRole",
    "AgentStartedEvent",
    "MultiAgentOrchestrator",
    "OrchestratorEvent",
    "WorkItemClaimedEvent",
    "WorkItemCompletedEvent",
    "WorkItemCreatedEvent",
    "WorkQueue",
]
