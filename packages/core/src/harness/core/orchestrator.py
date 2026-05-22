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
from typing import TYPE_CHECKING, Literal, Protocol

from pydantic import BaseModel

from harness.core.events import Event
from harness.core.schemas import RunRequest
from harness.tasks.schemas import Task, TaskStatus
from harness.tasks.store import TaskStore

if TYPE_CHECKING:
    from harness.core.activity import ActivityEvent
    from harness.core.runtime import Agent
    from harness.core.verification import WorkItemJudge


class _ActivityStore(Protocol):
    """Minimal duck-type for fetching activity events by session."""

    async def list_activity(self, *, session_id: str, limit: int) -> list[ActivityEvent]: ...


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
    max_steps: int | None = None
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


class WorkItemVerifiedEvent(BaseModel):
    type: Literal["work_item_verified"] = "work_item_verified"
    task_id: str
    task_ref: str
    confidence: float | None = None


class WorkItemRejectedEvent(BaseModel):
    type: Literal["work_item_rejected"] = "work_item_rejected"
    task_id: str
    task_ref: str
    reason: str
    attempt: int


class WorkItemOrphanedEvent(BaseModel):
    type: Literal["work_item_orphaned"] = "work_item_orphaned"
    task_id: str
    task_ref: str
    attempt: int


class AgentEventWrapper(BaseModel):
    type: Literal["agent_event"] = "agent_event"
    role: str
    event: Event


class PlanRejectedEvent(BaseModel):
    type: Literal["plan_rejected"] = "plan_rejected"
    reason: str
    attempt: int


class StallDetectedEvent(BaseModel):
    """Emitted when enough workers have exited without completing their items."""

    type: Literal["stall_detected"] = "stall_detected"
    stall_count: int
    max_stalls: int
    stalled_item_ids: list[str]


class ReplanRequestedEvent(BaseModel):
    """Emitted when the orchestrator triggers a replan after stall detection."""

    type: Literal["replan_requested"] = "replan_requested"
    attempt: int
    reason: str


OrchestratorEvent = (
    AgentStartedEvent
    | AgentDoneEvent
    | WorkItemCreatedEvent
    | WorkItemClaimedEvent
    | WorkItemCompletedEvent
    | WorkItemVerifiedEvent
    | WorkItemRejectedEvent
    | WorkItemOrphanedEvent
    | PlanRejectedEvent
    | StallDetectedEvent
    | ReplanRequestedEvent
    | AgentEventWrapper
)


# ---------------------------------------------------------------------------
# ProgressLedger
# ---------------------------------------------------------------------------


class ProgressLedger:
    """Tracks per-item completion to detect systemic stalls.

    A stall is defined as a worker agent completing its run without marking
    its work item done. When ``stall_count`` reaches ``max_stalls``, the
    orchestrator should replan. One successful completion resets the counter.
    """

    def __init__(self, max_stalls: int = 2) -> None:
        self.max_stalls = max_stalls
        self.stall_count: int = 0
        self._stalled_ids: list[str] = []

    def record_completion(self, task_id: str, *, completed: bool) -> None:
        """Record whether a work item was completed or stalled."""
        if completed:
            self.stall_count = 0
            self._stalled_ids.clear()
        else:
            self.stall_count += 1
            self._stalled_ids.append(task_id)

    @property
    def is_stalled(self) -> bool:
        return self.stall_count >= self.max_stalls

    @property
    def stalled_item_ids(self) -> list[str]:
        return list(self._stalled_ids)

    def reset(self) -> None:
        self.stall_count = 0
        self._stalled_ids.clear()


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
        work_item_judge: WorkItemJudge | None = None,
        max_judge_retries: int = 2,
        activity_store: _ActivityStore | None = None,
        planner_validator: Callable[[list[Task]], str | None] | None = None,
        max_planner_retries: int = 1,
        max_stalls: int = 2,
        max_replan_attempts: int = 1,
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
        self._work_item_judge = work_item_judge
        self._max_judge_retries = max_judge_retries
        self._activity_store = activity_store
        self._planner_validator = planner_validator
        self._max_planner_retries = max_planner_retries
        self._max_stalls = max_stalls
        self._max_replan_attempts = max_replan_attempts

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

        # Validate planner output; retry up to max_planner_retries times.
        if self._planner_validator is not None:
            for attempt in range(1, self._max_planner_retries + 2):
                items = await queue.list(status="todo")
                error = self._planner_validator(items)
                if error is None:
                    break
                yield PlanRejectedEvent(reason=error, attempt=attempt)
                if attempt > self._max_planner_retries:
                    break
                # Cancel bad items and retry the planner with feedback
                for item in items:
                    cancelled = item.model_copy(
                        update={"status": "cancelled", "updated_at": datetime.now(UTC)}
                    )
                    await self._store.update_task(cancelled)
                retry_prompt = (
                    f"{prompt}\n\n"
                    f"VALIDATION FEEDBACK (attempt {attempt}): {error}\n"
                    "Fix: create one work item per subdirectory, not one item for everything."
                )
                async for event in self._run_planner(retry_prompt, queue):
                    yield event

        ledger = ProgressLedger(self._max_stalls)
        replan_attempts = 0
        while True:
            stall_detected = False
            async for event in self._run_workers(queue, ledger=ledger):
                yield event
                if isinstance(event, StallDetectedEvent):
                    stall_detected = True

            if not stall_detected or replan_attempts >= self._max_replan_attempts:
                break

            replan_attempts += 1
            reason = f"{ledger.stall_count} worker(s) exited without completing their items"
            yield ReplanRequestedEvent(attempt=replan_attempts, reason=reason)

            for item_id in ledger.stalled_item_ids:
                stalled_task = await self._store.get_task(item_id)
                if stalled_task and stalled_task.status == "in_progress":
                    await self._store.update_task(
                        stalled_task.model_copy(
                            update={"status": "todo", "updated_at": datetime.now(UTC)}
                        )
                    )

            ledger.reset()
            replan_prompt = (
                f"{prompt}\n\n"
                f"[REPLAN {replan_attempts}]: {reason}. "
                "Revise the plan for the remaining items only."
            )
            async for event in self._run_planner(replan_prompt, queue):
                yield event

        async for event in self._run_reporter(queue):
            yield event

        # Mark the root job task done
        done_root = root.model_copy(update={"status": "done", "updated_at": datetime.now(UTC)})
        await self._store.update_task(done_root)

    async def resume(self, job_id: str) -> AsyncIterator[OrchestratorEvent]:
        """Resume an interrupted job from persistent storage.

        Resets any in_progress tasks (interrupted mid-run) back to todo, then
        re-runs the worker and reporter phases. The planner phase is skipped —
        the work queue already exists from the original run.
        """
        root = await self._store.get_task(job_id)
        if root is None:
            raise KeyError(f"job {job_id!r} not found in store")

        queue = WorkQueue(self._store, parent_id=job_id)

        # Reset any tasks that were in_progress when the job was interrupted
        in_progress = await self._store.list_tasks(parent_id=job_id, status="in_progress")
        for task in in_progress:
            reset = task.model_copy(
                update={
                    "status": "todo",
                    "metadata": {
                        **task.metadata,
                        "_resume_reset": True,
                    },
                    "updated_at": datetime.now(UTC),
                }
            )
            await self._store.update_task(reset)

        # Re-mark root as in_progress if it was marked done prematurely
        if root.status == "done":
            await self._store.update_task(
                root.model_copy(update={"status": "in_progress", "updated_at": datetime.now(UTC)})
            )

        async for event in self._run_workers(queue):
            yield event

        async for event in self._run_reporter(queue):
            yield event

        done_root = (await self._store.get_task(job_id)) or root
        await self._store.update_task(
            done_root.model_copy(update={"status": "done", "updated_at": datetime.now(UTC)})
        )

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

    async def _run_workers(
        self, queue: WorkQueue, ledger: ProgressLedger | None = None
    ) -> AsyncIterator[OrchestratorEvent]:
        output_queue: asyncio.Queue[OrchestratorEvent] = asyncio.Queue()
        done_event = asyncio.Event()
        stall_triggered = asyncio.Event()

        async def worker(idx: int) -> None:
            worker_name = f"worker-{idx}"
            while not stall_triggered.is_set():
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
                    max_steps=role.max_steps or self._max_worker_steps,
                )
                await output_queue.put(AgentStartedEvent(role=worker_name, session_id=session_id))
                async for event in agent.run(request):
                    await output_queue.put(AgentEventWrapper(role=worker_name, event=event))
                await output_queue.put(AgentDoneEvent(role=worker_name, session_id=session_id))

                # Post-run check: inspect task status and run judge / handle orphan
                async for post_event in self._post_run_check(task, session_id):
                    await output_queue.put(post_event)

                # Stall detection: record whether the work item ended as done
                if ledger is not None:
                    refreshed = await self._store.get_task(task.id)
                    completed = refreshed is not None and refreshed.status == "done"
                    ledger.record_completion(task.id, completed=completed)
                    if ledger.is_stalled:
                        await output_queue.put(
                            StallDetectedEvent(
                                stall_count=ledger.stall_count,
                                max_stalls=ledger.max_stalls,
                                stalled_item_ids=ledger.stalled_item_ids,
                            )
                        )
                        stall_triggered.set()
                        return

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

    async def _post_run_check(
        self, task: Task, session_id: str
    ) -> AsyncIterator[OrchestratorEvent]:
        """Inspect task status after a worker agent exits and take corrective action.

        - done   → run the isolated judge (if configured); on rejection, reset to todo
        - in_progress → worker hit max_steps without calling complete_work_item; reset to todo
        """
        current = await self._store.get_task(task.id)
        if current is None:
            return

        if current.status == "done":
            async for event in self._run_item_judge(current, session_id):
                yield event

        elif current.status == "in_progress":
            # Worker exited without completing — reset for retry
            orphan_count = current.metadata.get("_orphan_count", 0)
            attempt = orphan_count + 1
            yield WorkItemOrphanedEvent(task_id=current.id, task_ref=current.ref, attempt=attempt)
            if attempt >= self._max_judge_retries:
                # Give up — mark cancelled so it doesn't block workers forever
                abandoned = current.model_copy(
                    update={
                        "status": "cancelled",
                        "metadata": {**current.metadata, "_orphan_count": attempt},
                        "updated_at": datetime.now(UTC),
                    }
                )
                await self._store.update_task(abandoned)
            else:
                feedback = (
                    f"\n\n[RETRY {attempt}]: Previous attempt hit the step limit without"
                    " completing. Focus on calling complete_work_item before running out of steps."
                )
                reset = current.model_copy(
                    update={
                        "status": "todo",
                        "description": (task.description or "") + feedback,
                        "metadata": {**current.metadata, "_orphan_count": attempt},
                        "updated_at": datetime.now(UTC),
                    }
                )
                await self._store.update_task(reset)

    async def _run_item_judge(
        self, task: Task, session_id: str
    ) -> AsyncIterator[OrchestratorEvent]:
        """Run the isolated judge on a self-reported completion.

        On pass: emit WorkItemVerifiedEvent.
        On fail: reset task to todo with judge feedback; emit WorkItemRejectedEvent.
        Once retries are exhausted, leave the task as done (accept with warning).
        """
        if self._work_item_judge is None:
            return

        summary = task.metadata.get("result_summary", "")
        retry_count = task.metadata.get("_judge_retries", 0)

        activity: list[ActivityEvent] = []
        if self._activity_store is not None:
            activity = await self._activity_store.list_activity(session_id=session_id, limit=200)

        result = await self._work_item_judge.judge(
            task_title=task.title,
            task_description=task.description,
            result_summary=summary,
            activity=activity,
        )

        if result.can_finish:
            yield WorkItemVerifiedEvent(
                task_id=task.id,
                task_ref=task.ref,
                confidence=result.confidence,
            )
            return

        attempt = retry_count + 1
        yield WorkItemRejectedEvent(
            task_id=task.id,
            task_ref=task.ref,
            reason=result.reason,
            attempt=attempt,
        )

        if attempt >= self._max_judge_retries:
            # Exhausted retries — accept the completion as-is
            return

        # Reset to todo with judge feedback injected into description
        feedback = (
            f"\n\n[REVIEWER FEEDBACK (attempt {attempt})]: {result.reason}"
            " Revise your approach and try again."
        )
        reset = task.model_copy(
            update={
                "status": "todo",
                "description": (task.description or "") + feedback,
                "metadata": {
                    **task.metadata,
                    "_judge_retries": attempt,
                    "result_summary": None,
                },
                "updated_at": datetime.now(UTC),
            }
        )
        await self._store.update_task(reset)

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
    "PlanRejectedEvent",
    "ProgressLedger",
    "ReplanRequestedEvent",
    "StallDetectedEvent",
    "WorkItemClaimedEvent",
    "WorkItemCompletedEvent",
    "WorkItemCreatedEvent",
    "WorkItemOrphanedEvent",
    "WorkItemRejectedEvent",
    "WorkItemVerifiedEvent",
    "WorkQueue",
]
