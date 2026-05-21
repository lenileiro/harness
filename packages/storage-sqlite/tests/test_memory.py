"""Tests for SQLiteStorage memory CRUD + search + persistence."""

from __future__ import annotations

import pytest

from harness.core.memory import MemoryEntry
from harness.storage.sqlite import SQLiteStorage


@pytest.fixture
async def store():  # type: ignore[return]
    s = SQLiteStorage(path=":memory:")
    yield s
    await s.close()


@pytest.mark.asyncio
async def test_save_and_list(store: SQLiteStorage) -> None:
    e1 = MemoryEntry(kind="project_fact", text="uses uv workspace")
    e2 = MemoryEntry(kind="user_preference", text="prefers concise responses")
    await store.save_memory(e1)
    await store.save_memory(e2)

    all_entries = await store.list_memory()
    assert len(all_entries) == 2


@pytest.mark.asyncio
async def test_list_filter_by_kind(store: SQLiteStorage) -> None:
    await store.save_memory(MemoryEntry(kind="project_fact", text="uses uv"))
    await store.save_memory(MemoryEntry(kind="user_preference", text="concise"))
    await store.save_memory(MemoryEntry(kind="project_fact", text="python 3.12"))

    facts = await store.list_memory(kind="project_fact")
    assert len(facts) == 2
    assert all(e.kind == "project_fact" for e in facts)


@pytest.mark.asyncio
async def test_list_limit(store: SQLiteStorage) -> None:
    for i in range(5):
        await store.save_memory(MemoryEntry(kind="project_fact", text=f"fact {i}"))

    limited = await store.list_memory(limit=3)
    assert len(limited) == 3


@pytest.mark.asyncio
async def test_search(store: SQLiteStorage) -> None:
    await store.save_memory(MemoryEntry(kind="project_fact", text="uses uv workspace"))
    await store.save_memory(MemoryEntry(kind="project_fact", text="Python version 3.12"))

    results = await store.search_memory("uv")
    assert len(results) == 1
    assert "uv" in results[0].text


@pytest.mark.asyncio
async def test_search_case_insensitive(store: SQLiteStorage) -> None:
    await store.save_memory(MemoryEntry(kind="project_fact", text="Uses UV Workspace"))
    results = await store.search_memory("uv workspace")
    # SQLite LIKE is case-insensitive for ASCII
    assert len(results) == 1


@pytest.mark.asyncio
async def test_search_no_results(store: SQLiteStorage) -> None:
    await store.save_memory(MemoryEntry(kind="project_fact", text="something else"))
    results = await store.search_memory("xyznotfound")
    assert results == []


@pytest.mark.asyncio
async def test_delete(store: SQLiteStorage) -> None:
    entry = await store.save_memory(MemoryEntry(kind="user_fact", text="to delete"))
    entries = await store.list_memory()
    assert len(entries) == 1

    await store.delete_memory(entry.id)
    entries = await store.list_memory()
    assert entries == []


@pytest.mark.asyncio
async def test_delete_nonexistent_is_noop(store: SQLiteStorage) -> None:
    await store.delete_memory("mem_doesnotexist")  # should not raise


@pytest.mark.asyncio
async def test_save_overwrites_same_id(store: SQLiteStorage) -> None:
    entry = MemoryEntry(kind="project_fact", text="original")
    await store.save_memory(entry)
    updated = entry.model_copy(update={"text": "updated"})
    await store.save_memory(updated)

    entries = await store.list_memory()
    assert len(entries) == 1
    assert entries[0].text == "updated"


@pytest.mark.asyncio
async def test_persistence_across_connections(tmp_path) -> None:
    db_path = tmp_path / "test_memory.db"

    s1 = SQLiteStorage(path=db_path)
    entry = MemoryEntry(kind="project_context", text="persisted context")
    await s1.save_memory(entry)
    await s1.close()

    s2 = SQLiteStorage(path=db_path)
    entries = await s2.list_memory()
    await s2.close()

    assert len(entries) == 1
    assert entries[0].text == "persisted context"
    assert entries[0].id == entry.id
