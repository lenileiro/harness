"""FTS5 migration, atomic indexing, persistent session context and scope isolation."""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from harness.core.memory import MemoryEntry, MemoryScope
from harness.core.schemas import Message, Note, PhaseStatus, Session
from harness.storage.sqlite import SQLiteStorage
from harness.storage.sqlite.search import index_session


async def test_sqlite_preserves_notes_phases_and_search_after_reopening(tmp_path):
    path = tmp_path / "sessions.db"
    store = SQLiteStorage(path=path)
    session = Session(
        provider="mock",
        model="mock",
        cwd=tmp_path,
        notes=[Note(text="finding")],
        phases=[PhaseStatus(name="implement", notes=["in progress"])],
        messages=[Message(role="user", content="persistent telescope")],
    )
    await store.save(session)
    await store.close()
    store = SQLiteStorage(path=path)
    loaded = await store.get(session.id)
    assert loaded is not None
    assert loaded.notes == session.notes
    assert loaded.phases == session.phases
    assert [
        match.session_id
        for match in await store.search_sessions(
            "telescope", scope=MemoryScope(workspace=str(tmp_path))
        )
    ] == [session.id]
    await store.close()


async def test_backfill_indexes_legacy_transcripts_and_migrates_only_owned_memories(tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE sessions(id TEXT PRIMARY KEY, provider TEXT, model TEXT, cwd TEXT, status TEXT,
              created_at TEXT, updated_at TEXT, messages TEXT, approval_overrides TEXT, metadata TEXT);
            CREATE TABLE memory(id TEXT PRIMARY KEY, kind TEXT, text TEXT, session_id TEXT, task_id TEXT, created_at TEXT);
        """)
        db.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "old",
                "mock",
                "mock",
                str(tmp_path),
                "done",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
                json.dumps([{"role": "user", "content": "nebula migration"}]),
                "{}",
                "{}",
            ),
        )
        for entry_id, session_id in (("owned", "old"), ("unknown", None)):
            db.execute(
                "INSERT INTO memory VALUES (?, 'project_fact', 'legacy', ?, NULL, ?)",
                (entry_id, session_id, "2026-01-01T00:00:00+00:00"),
            )
    for _ in range(2):
        store = SQLiteStorage(path=path)
        scope = MemoryScope(workspace=str(tmp_path))
        assert [
            match.session_id for match in await store.search_sessions("nebula", scope=scope)
        ] == ["old"]
        assert [entry.id for entry in await store.list_scoped_memory(scope=scope)] == ["owned"]
        assert len(await store.list_memory()) == 2
        await store.close()
    with sqlite3.connect(path) as db:
        assert (
            "fts5"
            in db.execute("SELECT sql FROM sqlite_master WHERE name = 'session_search'").fetchone()[
                0
            ]
        )
        assert db.execute("SELECT COUNT(*) FROM session_search").fetchone()[0] == 1


async def test_workspace_database_can_scope_legacy_memory_without_session(tmp_path):
    path = tmp_path / ".harness" / "harness.db"
    store = SQLiteStorage(path=path)
    await store.save_memory(MemoryEntry(kind="project_fact", text="local legacy"))
    assert len(await store.list_scoped_memory(scope=MemoryScope(workspace=str(tmp_path)))) == 1
    assert not await store.list_scoped_memory(
        scope=MemoryScope(workspace=str(tmp_path), user_id="alice")
    )
    await store.close()


async def test_conflicting_scoped_inserts_are_atomic_across_connections(tmp_path):
    path = tmp_path / "memory.db"
    stores = [SQLiteStorage(path=path), SQLiteStorage(path=path)]
    scopes = [MemoryScope(workspace=str(tmp_path), user_id=user) for user in ("alice", "bob")]
    entry = MemoryEntry(id="same", kind="user_fact", text="winner")
    results = await asyncio.gather(
        *(
            store.save_scoped_memory(entry, scope=scope)
            for store, scope in zip(stores, scopes, strict=True)
        ),
        return_exceptions=True,
    )
    assert sum(isinstance(result, MemoryEntry) for result in results) == 1
    assert sum(isinstance(result, KeyError) for result in results) == 1
    for store in stores:
        await store.close()


async def test_failed_index_update_rolls_back_transcript_and_search(tmp_path, monkeypatch):
    store = SQLiteStorage(path=tmp_path / "sessions.db")
    session = Session(
        provider="mock",
        model="mock",
        cwd=tmp_path,
        messages=[Message(role="user", content="original nebula")],
    )
    await store.save(session)

    async def fail_after_index(db, changed):
        await index_session(db, changed)
        raise RuntimeError("index failure")

    monkeypatch.setattr("harness.storage.sqlite.index_session", fail_after_index)
    session.messages = [Message(role="user", content="replacement telescope")]
    with pytest.raises(RuntimeError, match="index failure"):
        await store.save(session)
    saved = await store.get(session.id)
    assert saved is not None
    assert saved.messages[0].content == "original nebula"
    scope = MemoryScope(workspace=str(tmp_path))
    assert len(await store.search_sessions("nebula", scope=scope)) == 1
    assert not await store.search_sessions("telescope", scope=scope)
    await store.close()
