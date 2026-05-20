"""TaskStore + ActivityStore tests for InMemoryStorage."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from harness.storage.memory import InMemoryStorage
from harness.tasks import ActivityEvent, Task, TaskLink


def _make_task(*, title: str = "test", **overrides: object) -> Task:
    return Task(ref="", title=title, cwd=Path.cwd(), **overrides)  # type: ignore[arg-type]


@pytest.mark.asyncio
class TestTaskStore:
    async def test_create_assigns_sequential_ref(self) -> None:
        storage = InMemoryStorage()
        first = await storage.create_task(_make_task(title="first"))
        second = await storage.create_task(_make_task(title="second"))
        assert first.ref == "T-001"
        assert second.ref == "T-002"

    async def test_create_ignores_caller_ref(self) -> None:
        storage = InMemoryStorage()
        attempted = Task(ref="T-999", title="forced", cwd=Path.cwd())
        saved = await storage.create_task(attempted)
        # Counter wins; T-999 is dropped.
        assert saved.ref == "T-001"

    async def test_get_round_trip(self) -> None:
        storage = InMemoryStorage()
        saved = await storage.create_task(_make_task(title="t"))
        loaded = await storage.get_task(saved.id)
        assert loaded is not None
        assert loaded.title == "t"

    async def test_get_by_ref(self) -> None:
        storage = InMemoryStorage()
        saved = await storage.create_task(_make_task())
        loaded = await storage.get_task_by_ref(saved.ref)
        assert loaded is not None
        assert loaded.id == saved.id

    async def test_get_missing_returns_none(self) -> None:
        storage = InMemoryStorage()
        assert await storage.get_task("nope") is None
        assert await storage.get_task_by_ref("T-999") is None

    async def test_list_newest_first(self) -> None:
        storage = InMemoryStorage()
        a = await storage.create_task(_make_task(title="a"))
        await asyncio.sleep(0.001)
        b = await storage.create_task(_make_task(title="b"))
        listed = await storage.list_tasks()
        assert [t.id for t in listed] == [b.id, a.id]

    async def test_list_filters_by_status(self) -> None:
        storage = InMemoryStorage()
        a = await storage.create_task(_make_task(title="a"))
        b = await storage.create_task(_make_task(title="b"))
        a.status = "done"
        a.touch()
        await storage.update_task(a)
        result = await storage.list_tasks(status="done")
        assert [t.id for t in result] == [a.id]
        result = await storage.list_tasks(status="backlog")
        assert [t.id for t in result] == [b.id]

    async def test_list_filters_by_parent_id(self) -> None:
        storage = InMemoryStorage()
        parent = await storage.create_task(_make_task(title="p"))
        child = await storage.create_task(_make_task(title="c", parent_id=parent.id))
        siblings = await storage.list_tasks(parent_id=parent.id)
        assert [t.id for t in siblings] == [child.id]

    async def test_update_persists_changes(self) -> None:
        storage = InMemoryStorage()
        t = await storage.create_task(_make_task(title="initial"))
        t.title = "updated"
        t.touch()
        await storage.update_task(t)
        loaded = await storage.get_task(t.id)
        assert loaded is not None
        assert loaded.title == "updated"

    async def test_update_missing_raises(self) -> None:
        storage = InMemoryStorage()
        with pytest.raises(KeyError):
            await storage.update_task(Task(id="nope", ref="T-X", title="x", cwd=Path.cwd()))

    async def test_delete_removes(self) -> None:
        storage = InMemoryStorage()
        t = await storage.create_task(_make_task())
        await storage.delete_task(t.id)
        assert await storage.get_task(t.id) is None

    async def test_save_isolates_external_mutations(self) -> None:
        storage = InMemoryStorage()
        t = await storage.create_task(
            _make_task(links=[TaskLink(target_ref="T-002", relation="blocks")])
        )
        t.links.append(TaskLink(target_ref="T-003", relation="depends_on"))
        loaded = await storage.get_task(t.id)
        assert loaded is not None
        # External mutation after create must not bleed into the store.
        assert len(loaded.links) == 1


@pytest.mark.asyncio
class TestActivityStore:
    async def test_append_and_list(self) -> None:
        storage = InMemoryStorage()
        await storage.append_activity(
            ActivityEvent(task_id="t1", kind="task.created", data={"ref": "T-001"})
        )
        await storage.append_activity(ActivityEvent(task_id="t1", kind="task.updated", data={}))
        events = await storage.list_activity(task_id="t1")
        assert [e.kind for e in events] == ["task.created", "task.updated"]

    async def test_append_is_idempotent_on_id(self) -> None:
        storage = InMemoryStorage()
        event = ActivityEvent(task_id="t1", kind="task.created")
        await storage.append_activity(event)
        await storage.append_activity(event)
        events = await storage.list_activity(task_id="t1")
        assert len(events) == 1

    async def test_filter_by_session(self) -> None:
        storage = InMemoryStorage()
        await storage.append_activity(ActivityEvent(session_id="s1", kind="agent_run.started"))
        await storage.append_activity(ActivityEvent(session_id="s2", kind="agent_run.started"))
        s1_events = await storage.list_activity(session_id="s1")
        assert len(s1_events) == 1

    async def test_filter_by_kinds(self) -> None:
        storage = InMemoryStorage()
        await storage.append_activity(ActivityEvent(kind="task.created"))
        await storage.append_activity(ActivityEvent(kind="task.updated"))
        await storage.append_activity(ActivityEvent(kind="agent_run.started"))
        result = await storage.list_activity(kinds=("task.created", "task.updated"))
        assert {e.kind for e in result} == {"task.created", "task.updated"}
