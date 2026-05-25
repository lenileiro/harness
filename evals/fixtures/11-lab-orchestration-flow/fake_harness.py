from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from harness.cli import __main__ as cli_main
from harness.core.orchestrator import (
    AgentDoneEvent,
    AgentStartedEvent,
    WorkItemClaimedEvent,
    WorkItemCompletedEvent,
    WorkItemCreatedEvent,
    WorkItemVerifiedEvent,
)
from harness.tasks import Task


class FakeOrchestrator:
    def __init__(self, *args, **kwargs):
        self._store = kwargs["store"]
        self._job_cwd = Path(kwargs["job_cwd"])

    async def run(self, prompt: str):
        root = await self._store.get_task("task_root_eval")
        if root is None:
            now = datetime.now(UTC)
            root = await self._store.create_task(
                Task(
                    id="task_root_eval",
                    ref="",
                    title=f"job: {prompt[:60]}",
                    status="in_progress",
                    parent_id=None,
                    cwd=self._job_cwd,
                    created_at=now,
                    updated_at=now,
                )
            )
        items = await self._store.list_tasks(parent_id=root.id)
        if not items:
            now = datetime.now(UTC)
            draft = await self._store.create_task(
                Task(
                    ref="",
                    title="Draft plan",
                    description="Planner decomposes the work.",
                    status="done",
                    parent_id=root.id,
                    cwd=self._job_cwd,
                    created_at=now,
                    updated_at=now,
                    metadata={"result_summary": "planner split work"},
                )
            )
            execute = await self._store.create_task(
                Task(
                    ref="",
                    title="Execute change",
                    description="Worker completes the pending execution step.",
                    status="in_progress",
                    parent_id=root.id,
                    cwd=self._job_cwd,
                    created_at=now,
                    updated_at=now,
                    metadata={},
                )
            )
        else:
            by_title = {item.title: item for item in items}
            draft = by_title["Draft plan"]
            execute = by_title["Execute change"]

        yield AgentStartedEvent(role="planner", session_id="sess_planner_eval")
        yield WorkItemCreatedEvent(task_id=draft.id, task_ref=draft.ref, title=draft.title)
        yield WorkItemCreatedEvent(task_id=execute.id, task_ref=execute.ref, title=execute.title)
        yield AgentDoneEvent(role="planner", session_id="sess_planner_eval", turn_count=1)
        yield AgentStartedEvent(role="worker-1", session_id="sess_worker_1")
        yield WorkItemClaimedEvent(
            task_id=draft.id, task_ref=draft.ref, worker_session_id="sess_worker_1"
        )
        yield WorkItemCompletedEvent(
            task_id=draft.id, task_ref=draft.ref, summary="planner split work"
        )
        yield WorkItemVerifiedEvent(task_id=draft.id, task_ref=draft.ref, confidence=1.0)
        yield AgentDoneEvent(role="worker-1", session_id="sess_worker_1", turn_count=1)
        yield AgentStartedEvent(role="worker-2", session_id="sess_worker_2")
        yield WorkItemClaimedEvent(
            task_id=execute.id, task_ref=execute.ref, worker_session_id="sess_worker_2"
        )
        yield AgentDoneEvent(role="worker-2", session_id="sess_worker_2", turn_count=1)

    async def resume(self, job_id: str):
        root = await self._store.get_task(job_id)
        if root is None:
            return
        items = await self._store.list_tasks(parent_id=job_id)
        by_title = {item.title: item for item in items}
        execute = by_title["Execute change"]
        done_execute = execute.model_copy(
            update={
                "status": "done",
                "updated_at": datetime.now(UTC),
                "metadata": {"result_summary": "worker finished execution"},
            }
        )
        await self._store.update_task(done_execute)
        done_root = root.model_copy(update={"status": "done", "updated_at": datetime.now(UTC)})
        await self._store.update_task(done_root)
        report = self._job_cwd / "lab_report.md"
        report.write_text(
            "# Reporter summary\n\n- Draft plan\n- Execute change\n",
            encoding="utf-8",
        )

        yield AgentStartedEvent(role="worker-2", session_id="sess_worker_2")
        yield WorkItemClaimedEvent(
            task_id=done_execute.id, task_ref=done_execute.ref, worker_session_id="sess_worker_2"
        )
        yield WorkItemCompletedEvent(
            task_id=done_execute.id,
            task_ref=done_execute.ref,
            summary="worker finished execution",
        )
        yield WorkItemVerifiedEvent(
            task_id=done_execute.id,
            task_ref=done_execute.ref,
            confidence=1.0,
        )
        yield AgentDoneEvent(role="worker-2", session_id="sess_worker_2", turn_count=1)
        yield AgentStartedEvent(role="reporter", session_id="sess_reporter_eval")
        yield AgentDoneEvent(role="reporter", session_id="sess_reporter_eval", turn_count=1)


cli_main.MultiAgentOrchestrator = FakeOrchestrator


if __name__ == "__main__":
    cli_main.app()
