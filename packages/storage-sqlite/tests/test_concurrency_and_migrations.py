"""Storage transactions remain isolated across workers and failed upgrades."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import aiosqlite
import pytest

from harness.core import Session
from harness.storage import sqlite as sqlite_module
from harness.storage.sqlite import SQLiteStorage
from harness.tasks import ActivityEvent, Task


async def test_parallel_creates_claims_and_unrelated_writes(tmp_path: Path) -> None:
    storage = SQLiteStorage(path=tmp_path / "tasks.db")
    session = Session(provider="ollama", model="test", cwd=tmp_path)
    try:
        operations = [
            storage.create_task(
                Task(ref="", title=str(i), cwd=tmp_path, status="todo", parent_id="p")
            )
            for i in range(20)
        ]
        await asyncio.gather(
            *operations,
            storage.save(session),
            storage.append_activity(ActivityEvent(session_id=session.id, kind="test", data={})),
        )
        tasks = await storage.list_tasks()
        assert {task.ref for task in tasks} == {f"T-{i:03d}" for i in range(1, 21)}
        claims = await asyncio.gather(
            *(storage.claim_task(parent_id="p", claimed_by=f"worker-{i}") for i in range(25))
        )
        claimed = [task for task in claims if task is not None]
        assert len(claimed) == 20
        assert {task.id for task in claimed} == {task.id for task in tasks}
        assert all(task.status == "in_progress" for task in claimed)
        assert await storage.get(session.id) is not None
        assert len(await storage.list_activity(session_id=session.id)) == 1
    finally:
        await storage.close()


async def test_separate_connections_assign_distinct_references(tmp_path: Path) -> None:
    stores = [SQLiteStorage(path=tmp_path / "shared.db") for _ in range(2)]
    try:
        tasks = await asyncio.gather(
            *(
                stores[i % 2].create_task(Task(ref="", title=str(i), cwd=tmp_path))
                for i in range(12)
            )
        )
        assert len({task.ref for task in tasks}) == 12
    finally:
        for store in stores:
            await store.close()


async def test_cancelled_transaction_rolls_back_before_next_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = SQLiteStorage(path=tmp_path / "cancel.db")
    await storage.list()
    entered = asyncio.Event()
    release = asyncio.Event()
    original_commit = aiosqlite.Connection.commit
    first_commit = True

    async def pause_commit(db: aiosqlite.Connection) -> None:
        nonlocal first_commit
        if first_commit:
            first_commit = False
            entered.set()
            await release.wait()
        await original_commit(db)

    monkeypatch.setattr(aiosqlite.Connection, "commit", pause_commit)
    creating = asyncio.create_task(
        storage.create_task(Task(ref="", title="cancel me", cwd=tmp_path))
    )
    try:
        await asyncio.wait_for(entered.wait(), 2)
        session = Session(provider="ollama", model="test", cwd=tmp_path)
        saving = asyncio.create_task(storage.save(session))
        creating.cancel()
        with pytest.raises(asyncio.CancelledError):
            await creating
        await asyncio.wait_for(saving, 2)
        assert await storage.list_tasks() == []
        assert await storage.get(session.id) is not None
    finally:
        release.set()
        await storage.close()


async def test_legacy_sessions_migrate_without_data_loss(tmp_path: Path) -> None:
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE sessions (
            id TEXT PRIMARY KEY, provider TEXT, model TEXT, cwd TEXT, status TEXT,
            created_at TEXT, updated_at TEXT, messages TEXT,
            approval_overrides TEXT, metadata TEXT)""")
        db.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy",
                "ollama",
                "test",
                str(tmp_path),
                "done",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                "[]",
                "{}",
                "{}",
            ),
        )
    for _ in range(2):
        storage = SQLiteStorage(path=path)
        try:
            session = await storage.get("legacy")
            assert session is not None
            assert session.status == "done"
            assert session.task_id is None
            assert session.forked_from is None
            await storage.save(session)
        finally:
            await storage.close()
    with sqlite3.connect(path) as db:
        indexes = {row[1] for row in db.execute("PRAGMA index_list(sessions)")}
        assert "idx_sessions_task_id" in indexes


async def test_failed_migration_is_atomic_and_retryable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "failure.db"
    storage = SQLiteStorage(path=path)
    try:
        with monkeypatch.context() as context:
            context.setattr(
                sqlite_module, "_MIGRATIONS", (*sqlite_module._MIGRATIONS, "invalid SQL")
            )
            with pytest.raises(aiosqlite.OperationalError):
                await storage.list()
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
        assert await storage.list() == []
    finally:
        await storage.close()


async def test_cancellation_before_queued_begin_executes_does_not_leave_transaction(
    tmp_path: Path,
) -> None:
    import threading

    storage = SQLiteStorage(path=tmp_path / "queued.db")
    await storage.list()
    db = storage._db
    assert db is not None
    entered = threading.Event()
    release = threading.Event()

    def hold_worker() -> int:
        entered.set()
        release.wait(timeout=2)
        return 1

    await db.create_function("hold_worker", 0, hold_worker)
    blocker = asyncio.ensure_future(db.execute("SELECT hold_worker()"))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        creating = asyncio.create_task(
            storage.create_task(Task(ref="", title="cancel before begin", cwd=tmp_path))
        )
        # The worker is occupied, so BEGIN is queued but in_transaction is false.
        await asyncio.sleep(0)
        creating.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(creating, 2)
        cursor = await blocker
        await cursor.close()
        saved = await storage.create_task(Task(ref="", title="next writer", cwd=tmp_path))
        assert saved.ref == "T-001"
        assert len(await storage.list_tasks()) == 1
    finally:
        release.set()
        await storage.close()
