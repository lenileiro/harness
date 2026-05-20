"""TaskStore + ActivityStore tests for SQLiteStorage."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from harness.storage.sqlite import SQLiteStorage
from harness.tasks import ActivityEvent, Task, TaskLink


def _make_task(*, title: str = "test", **overrides: object) -> Task:
    return Task(ref="", title=title, cwd=Path.cwd(), **overrides)  # type: ignore[arg-type]


@pytest.fixture
async def storage(tmp_path: Path):
    s = SQLiteStorage(path=tmp_path / "tasks.db")
    try:
        yield s
    finally:
        await s.close()


@pytest.mark.asyncio
class TestTaskStore:
    async def test_create_assigns_sequential_ref(self, storage: SQLiteStorage) -> None:
        first = await storage.create_task(_make_task(title="first"))
        second = await storage.create_task(_make_task(title="second"))
        assert first.ref == "T-001"
        assert second.ref == "T-002"

    async def test_create_ignores_caller_ref(self, storage: SQLiteStorage) -> None:
        attempted = Task(ref="T-999", title="forced", cwd=Path.cwd())
        saved = await storage.create_task(attempted)
        assert saved.ref == "T-001"

    async def test_get_round_trip(self, storage: SQLiteStorage) -> None:
        saved = await storage.create_task(_make_task(title="t"))
        loaded = await storage.get_task(saved.id)
        assert loaded is not None
        assert loaded.title == "t"

    async def test_get_by_ref(self, storage: SQLiteStorage) -> None:
        saved = await storage.create_task(_make_task())
        loaded = await storage.get_task_by_ref(saved.ref)
        assert loaded is not None
        assert loaded.id == saved.id

    async def test_get_missing_returns_none(self, storage: SQLiteStorage) -> None:
        assert await storage.get_task("nope") is None
        assert await storage.get_task_by_ref("T-999") is None

    async def test_list_newest_first(self, storage: SQLiteStorage) -> None:
        a = await storage.create_task(_make_task(title="a"))
        await asyncio.sleep(0.005)
        b = await storage.create_task(_make_task(title="b"))
        listed = await storage.list_tasks()
        assert [t.id for t in listed] == [b.id, a.id]

    async def test_list_filters_by_status(self, storage: SQLiteStorage) -> None:
        a = await storage.create_task(_make_task(title="a"))
        await storage.create_task(_make_task(title="b"))
        a.status = "done"
        a.touch()
        await storage.update_task(a)
        result = await storage.list_tasks(status="done")
        assert [t.id for t in result] == [a.id]

    async def test_update_persists_changes(self, storage: SQLiteStorage) -> None:
        t = await storage.create_task(_make_task(title="initial"))
        t.title = "updated"
        t.touch()
        await storage.update_task(t)
        loaded = await storage.get_task(t.id)
        assert loaded is not None
        assert loaded.title == "updated"

    async def test_update_missing_raises(self, storage: SQLiteStorage) -> None:
        with pytest.raises(KeyError):
            await storage.update_task(Task(id="nope", ref="T-X", title="x", cwd=Path.cwd()))

    async def test_delete_removes(self, storage: SQLiteStorage) -> None:
        t = await storage.create_task(_make_task())
        await storage.delete_task(t.id)
        assert await storage.get_task(t.id) is None

    async def test_complex_task_round_trips(self, storage: SQLiteStorage) -> None:
        t = await storage.create_task(
            _make_task(
                title="complex",
                description="multi-line\ndescription",
                priority="high",
                labels=["a", "b"],
                links=[TaskLink(target_ref="T-002", relation="blocks")],
                session_ids=["s1", "s2"],
                metadata={"key": "value"},
            )
        )
        loaded = await storage.get_task(t.id)
        assert loaded is not None
        assert loaded.priority == "high"
        assert loaded.labels == ["a", "b"]
        assert loaded.links == t.links
        assert loaded.session_ids == ["s1", "s2"]
        assert loaded.metadata == {"key": "value"}

    async def test_persists_across_storage_instances(self, tmp_path: Path) -> None:
        db_path = tmp_path / "persistent.db"
        a = SQLiteStorage(path=db_path)
        try:
            saved = await a.create_task(_make_task(title="persist"))
            ref = saved.ref
        finally:
            await a.close()

        b = SQLiteStorage(path=db_path)
        try:
            loaded = await b.get_task_by_ref(ref)
            assert loaded is not None
            assert loaded.title == "persist"
            # Counter continues from where the first instance left off.
            next_task = await b.create_task(_make_task(title="next"))
            assert next_task.ref == "T-002"
        finally:
            await b.close()


@pytest.mark.asyncio
class TestActivityStore:
    async def test_append_and_list(self, storage: SQLiteStorage) -> None:
        await storage.append_activity(
            ActivityEvent(task_id="t1", kind="task.created", data={"ref": "T-001"})
        )
        await storage.append_activity(ActivityEvent(task_id="t1", kind="task.updated", data={}))
        events = await storage.list_activity(task_id="t1")
        assert [e.kind for e in events] == ["task.created", "task.updated"]

    async def test_append_is_idempotent_on_id(self, storage: SQLiteStorage) -> None:
        event = ActivityEvent(task_id="t1", kind="task.created")
        await storage.append_activity(event)
        await storage.append_activity(event)
        events = await storage.list_activity(task_id="t1")
        assert len(events) == 1

    async def test_filter_by_session_and_kinds(self, storage: SQLiteStorage) -> None:
        await storage.append_activity(ActivityEvent(session_id="s1", kind="agent_run.started"))
        await storage.append_activity(ActivityEvent(session_id="s1", kind="tool_call.dispatched"))
        await storage.append_activity(ActivityEvent(session_id="s2", kind="agent_run.started"))
        only_s1 = await storage.list_activity(session_id="s1")
        assert len(only_s1) == 2

        only_kind = await storage.list_activity(kinds=("tool_call.dispatched",))
        assert [e.kind for e in only_kind] == ["tool_call.dispatched"]

    async def test_data_round_trips(self, storage: SQLiteStorage) -> None:
        await storage.append_activity(
            ActivityEvent(
                task_id="t1",
                kind="custom.thing",
                data={"nested": {"k": 1}, "list": [1, 2, 3], "str": "hello"},
            )
        )
        events = await storage.list_activity(task_id="t1")
        assert events[0].data == {"nested": {"k": 1}, "list": [1, 2, 3], "str": "hello"}
