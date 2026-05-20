"""Storage Protocol for `Task` instances.

The activity ledger Protocol (`ActivityStore`) lives in
`harness.core.activity` so the runtime can emit events without taking a
dependency on this package. Storage backends typically implement both
`TaskStore` and `ActivityStore` (plus `harness.core.Storage` for sessions)
in a single class.

`create_task` is responsible for assigning the next sequential `ref`
(`T-001`, `T-002`, ...). Refs are unique per store.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from harness.tasks.schemas import Task, TaskStatus


@runtime_checkable
class TaskStore(Protocol):
    """Durable storage for `Task` instances."""

    async def create_task(self, task: Task) -> Task:
        """Insert the task and assign its `ref`. Returns the persisted copy.

        Implementations must populate `task.ref` with the next sequential
        identifier (`T-001`, `T-002`, ...). If `task.ref` is already set,
        the reference impls (`InMemory`, `SQLite`) overwrite with the next
        counter value rather than honoring it.
        """
        ...

    async def get_task(self, task_id: str) -> Task | None:
        """Load by id (`task_<uuid>`) or return None."""
        ...

    async def get_task_by_ref(self, ref: str) -> Task | None:
        """Load by human ref (`T-001`) or return None."""
        ...

    async def list_tasks(
        self,
        *,
        limit: int = 50,
        status: TaskStatus | None = None,
        parent_id: str | None = None,
    ) -> list[Task]:
        """List tasks, newest-updated first."""
        ...

    async def update_task(self, task: Task) -> Task:
        """Persist updates. Caller is responsible for bumping `updated_at`."""
        ...

    async def delete_task(self, task_id: str) -> None:
        """Idempotent delete by id."""
        ...


__all__ = ["TaskStore"]
