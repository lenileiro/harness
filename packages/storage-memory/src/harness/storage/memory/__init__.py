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

import asyncio
from datetime import UTC, datetime

from harness.core import Session, SessionStatus
from harness.core.memory import MemoryEntry, MemoryKind, MemoryScope, MemoryStore
from harness.core.session_search import (
    SessionSearchResult,
    search_limit,
    search_terms,
    session_scope,
    session_text,
)
from harness.tasks import ActivityEvent, ApprovalStatus, PendingApproval, Task, TaskStatus

__version__ = "0.0.0"


class InMemoryStorage(MemoryStore):
    """In-memory backend covering sessions, tasks, activity, approvals, and memory."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._tasks: dict[str, Task] = {}
        self._task_ref_counter: int = 0
        self._activity: list[ActivityEvent] = []
        self._activity_ids: set[str] = set()
        self._approvals: dict[str, PendingApproval] = {}
        self._memory: list[MemoryEntry] = []
        self._claim_lock = asyncio.Lock()

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

    async def search_sessions(
        self, query: str, *, scope: MemoryScope, limit: int = 10
    ) -> list[SessionSearchResult]:
        from harness.core.session_search import matching_messages

        limit = search_limit(limit)
        terms = search_terms(query)
        if not terms:
            return []
        found: list[SessionSearchResult] = []
        for session in sorted(
            self._sessions.values(), key=lambda item: item.updated_at, reverse=True
        ):
            if session_scope(session) != scope:
                continue
            text = session_text(session)
            lowered = text.casefold()
            if not all(term in lowered for term in terms):
                continue
            offset = max(0, min(lowered.index(term) for term in terms) - 100)
            found.append(
                SessionSearchResult(
                    session_id=session.id,
                    excerpt=text[offset : offset + 1200],
                    updated_at=session.updated_at,
                    matches=matching_messages(session.id, session.messages, terms),
                )
            )
            if len(found) >= limit:
                break
        return found

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

    async def claim_task(
        self,
        *,
        parent_id: str,
        claimed_by: str,
        worker_session_id: str | None = None,
    ) -> Task | None:
        async with self._claim_lock:
            candidates = [
                t for t in self._tasks.values() if t.parent_id == parent_id and t.status == "todo"
            ]
            if not candidates:
                return None
            oldest = min(candidates, key=lambda t: t.created_at)
            updated = oldest.model_copy(
                update={
                    "status": "in_progress",
                    "metadata": {
                        **oldest.metadata,
                        "claimed_by": claimed_by,
                        "worker_session_id": worker_session_id,
                    },
                    "updated_at": datetime.now(UTC),
                }
            )
            self._tasks[updated.id] = updated.model_copy(deep=True)
            return updated.model_copy(deep=True)

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
        if limit <= 0:
            return []
        return [e.model_copy(deep=True) for e in items[-limit:]]

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
        if status not in ("granted", "denied"):
            raise ValueError("approval resolution must be granted or denied")
        stored = self._approvals.get(approval_id)
        if stored is None or stored.status != "pending":
            return None
        stored.status = status
        stored.resolved_at = datetime.now(UTC)
        stored.resolved_by = resolved_by
        return stored.model_copy(deep=True)

    async def claim_replay(self, approval_id: str, *, session_id: str) -> bool:
        stored = self._approvals.get(approval_id)
        if (
            stored is None
            or stored.session_id != session_id
            or stored.status != "granted"
            or stored.replayed_at is not None
            or stored.replay_claimed_at is not None
        ):
            return False
        stored.replay_claimed_at = datetime.now(UTC)
        return True

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

    # ------------------------------------------------------------------ #
    # MemoryStore                                                         #
    # ------------------------------------------------------------------ #

    async def save_memory(self, entry: MemoryEntry) -> MemoryEntry:
        if (
            entry.scope is None
            and entry.session_id is not None
            and entry.session_id in self._sessions
        ):
            entry = entry.model_copy(
                update={"scope": session_scope(self._sessions[entry.session_id])}
            )
        self._memory = [e for e in self._memory if e.id != entry.id]
        self._memory.append(entry.model_copy(deep=True))
        return entry.model_copy(deep=True)

    async def list_memory(
        self, *, kind: MemoryKind | None = None, limit: int = 50
    ) -> list[MemoryEntry]:
        items = list(self._memory)
        if kind is not None:
            items = [e for e in items if e.kind == kind]
        items.sort(key=lambda e: e.created_at, reverse=True)
        return [e.model_copy(deep=True) for e in items[:limit]]

    async def search_memory(self, query: str, *, limit: int = 20) -> list[MemoryEntry]:
        q = query.lower()
        items = [e for e in self._memory if q in e.text.lower()]
        items.sort(key=lambda e: e.created_at, reverse=True)
        return [e.model_copy(deep=True) for e in items[:limit]]

    async def delete_memory(self, entry_id: str) -> None:
        self._memory = [e for e in self._memory if e.id != entry_id]

    async def save_scoped_memory(self, entry: MemoryEntry, *, scope: MemoryScope) -> MemoryEntry:
        stored = next((item for item in self._memory if item.id == entry.id), None)
        if stored is not None and stored.scope != scope:
            raise KeyError("Memory not found")
        return await self.save_memory(entry.model_copy(update={"scope": scope}, deep=True))

    async def get_scoped_memory(self, entry_id: str, *, scope: MemoryScope) -> MemoryEntry | None:
        entry = next(
            (item for item in self._memory if item.id == entry_id and item.scope == scope), None
        )
        return entry.model_copy(deep=True) if entry else None

    async def list_scoped_memory(
        self, *, scope: MemoryScope, kind: MemoryKind | None = None, limit: int = 50
    ) -> list[MemoryEntry]:
        items = [
            entry
            for entry in self._memory
            if entry.scope == scope and (kind is None or entry.kind == kind)
        ]
        items.sort(key=lambda entry: entry.created_at, reverse=True)
        return [entry.model_copy(deep=True) for entry in items[: search_limit(limit)]]

    async def search_scoped_memory(
        self, query: str, *, scope: MemoryScope, limit: int = 20
    ) -> list[MemoryEntry]:
        items = [
            entry
            for entry in self._memory
            if entry.scope == scope and query.lower() in entry.text.lower()
        ]
        items.sort(key=lambda entry: entry.created_at, reverse=True)
        return [entry.model_copy(deep=True) for entry in items[: search_limit(limit)]]

    async def delete_scoped_memory(self, entry_id: str, *, scope: MemoryScope) -> bool:
        count = len(self._memory)
        self._memory = [
            entry for entry in self._memory if not (entry.id == entry_id and entry.scope == scope)
        ]
        return len(self._memory) != count


__all__ = ["InMemoryStorage", "__version__"]
