"""In-memory Storage implementation for Harness.

Implements four Protocols:

- `harness.core.Storage`         — sessions
- `harness.tasks.TaskStore`      — tasks
- `harness.tasks.ActivityStore`  — append-only activity log
- `harness.tasks.ApprovalStore`  — pending tool-call approvals

Sessions, tasks, activity, and approvals all live in process memory.
All mutations defensively deep-copy so callers cannot mutate the store by
retaining references.
"""

from __future__ import annotations

from datetime import UTC, datetime

from harness.core import Session, SessionStatus
from harness.tasks import ActivityEvent, ApprovalStatus, PendingApproval, Task, TaskStatus

__version__ = "0.0.0"


class InMemoryStorage:
    """In-memory backend covering sessions, tasks, activity, and approvals."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._tasks: dict[str, Task] = {}
        self._task_ref_counter: int = 0
        self._activity: list[ActivityEvent] = []
        self._activity_ids: set[str] = set()
        self._approvals: dict[str, PendingApproval] = {}

    # ------------------------------------------------------------------ #
    # SessionStore (harness.core.Storage)                                 #
    # ------------------------------------------------------------------ #

    async def get(self, session_id: str) -> Session | None:
        stored = self._sessions.get(session_id)
        return stored.model_copy(deep=True) if stored else None

    async def save(self, session: Session) -> None:
        self._sessions[session.id] = session.model_copy(deep=True)

    async def list(
        self,
        *,
        limit: int = 50,
        before: datetime | None = None,
        status: SessionStatus | None = None,
    ) -> list[Session]:
        items = sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)
        if before is not None:
            items = [s for s in items if s.updated_at < before]
        if status is not None:
            items = [s for s in items if s.status == status]
        return [s.model_copy(deep=True) for s in items[:limit]]

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------ #
    # TaskStore (harness.tasks.TaskStore)                                 #
    # ------------------------------------------------------------------ #

    async def create_task(self, task: Task) -> Task:
        self._task_ref_counter += 1
        updated = task.model_copy(update={"ref": f"T-{self._task_ref_counter:03d}"})
        self._tasks[updated.id] = updated.model_copy(deep=True)
        return updated.model_copy(deep=True)

    async def get_task(self, task_id: str) -> Task | None:
        stored = self._tasks.get(task_id)
        return stored.model_copy(deep=True) if stored else None

    async def get_task_by_ref(self, ref: str) -> Task | None:
        for stored in self._tasks.values():
            if stored.ref == ref:
                return stored.model_copy(deep=True)
        return None

    async def list_tasks(
        self,
        *,
        limit: int = 50,
        status: TaskStatus | None = None,
        parent_id: str | None = None,
    ) -> list[Task]:
        items = sorted(self._tasks.values(), key=lambda t: t.updated_at, reverse=True)
        if status is not None:
            items = [t for t in items if t.status == status]
        if parent_id is not None:
            items = [t for t in items if t.parent_id == parent_id]
        return [t.model_copy(deep=True) for t in items[:limit]]

    async def update_task(self, task: Task) -> Task:
        if task.id not in self._tasks:
            raise KeyError(f"task {task.id!r} not found")
        self._tasks[task.id] = task.model_copy(deep=True)
        return task.model_copy(deep=True)

    async def delete_task(self, task_id: str) -> None:
        self._tasks.pop(task_id, None)

    # ------------------------------------------------------------------ #
    # ActivityStore (harness.tasks.ActivityStore)                         #
    # ------------------------------------------------------------------ #

    async def append_activity(self, event: ActivityEvent) -> None:
        if event.id in self._activity_ids:
            return
        self._activity.append(event.model_copy(deep=True))
        self._activity_ids.add(event.id)

    async def list_activity(
        self,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        kinds: tuple[str, ...] | None = None,
        limit: int = 200,
    ) -> list[ActivityEvent]:
        items = list(self._activity)
        if task_id is not None:
            items = [e for e in items if e.task_id == task_id]
        if session_id is not None:
            items = [e for e in items if e.session_id == session_id]
        if kinds is not None:
            kinds_set = set(kinds)
            items = [e for e in items if e.kind in kinds_set]
        items.sort(key=lambda e: e.timestamp)
        return [e.model_copy(deep=True) for e in items[:limit]]

    # ------------------------------------------------------------------ #
    # ApprovalStore                                                       #
    # ------------------------------------------------------------------ #

    async def create_approval(self, approval: PendingApproval) -> PendingApproval:
        self._approvals[approval.id] = approval.model_copy(deep=True)
        return approval.model_copy(deep=True)

    async def get_approval(self, approval_id: str) -> PendingApproval | None:
        stored = self._approvals.get(approval_id)
        return stored.model_copy(deep=True) if stored else None

    async def list_approvals(
        self,
        *,
        session_id: str | None = None,
        task_id: str | None = None,
        status: ApprovalStatus | None = None,
        limit: int = 100,
    ) -> list[PendingApproval]:
        items = list(self._approvals.values())
        if session_id is not None:
            items = [a for a in items if a.session_id == session_id]
        if task_id is not None:
            items = [a for a in items if a.task_id == task_id]
        if status is not None:
            items = [a for a in items if a.status == status]
        items.sort(key=lambda a: a.requested_at, reverse=True)
        return [a.model_copy(deep=True) for a in items[:limit]]

    async def resolve_approval(
        self,
        approval_id: str,
        *,
        status: ApprovalStatus,
        resolved_by: str | None = None,
    ) -> PendingApproval | None:
        stored = self._approvals.get(approval_id)
        if stored is None:
            return None
        stored.status = status
        stored.resolved_at = datetime.now(UTC)
        stored.resolved_by = resolved_by
        return stored.model_copy(deep=True)

    async def mark_replayed(self, approval_id: str) -> None:
        stored = self._approvals.get(approval_id)
        if stored is None:
            return
        if stored.replayed_at is None:
            stored.replayed_at = datetime.now(UTC)

    async def list_unreplayed_granted(self, *, session_id: str) -> list[PendingApproval]:
        items = [
            a
            for a in self._approvals.values()
            if a.session_id == session_id and a.status == "granted" and a.replayed_at is None
        ]
        items.sort(key=lambda a: a.requested_at)
        return [a.model_copy(deep=True) for a in items]


__all__ = ["InMemoryStorage", "__version__"]
