"""Tests for InMemoryStorage memory CRUD + search."""

from __future__ import annotations

import pytest

from harness.core.memory import MemoryEntry
from harness.storage.memory import InMemoryStorage


@pytest.fixture
def store() -> InMemoryStorage:
    return InMemoryStorage()


@pytest.mark.asyncio
async def test_save_and_list(store: InMemoryStorage) -> None:
    e1 = MemoryEntry(kind="project_fact", text="uses uv workspace")
    e2 = MemoryEntry(kind="user_preference", text="prefers concise responses")
    await store.save_memory(e1)
    await store.save_memory(e2)

    all_entries = await store.list_memory()
    assert len(all_entries) == 2


@pytest.mark.asyncio
async def test_list_filter_by_kind(store: InMemoryStorage) -> None:
    await store.save_memory(MemoryEntry(kind="project_fact", text="uses uv"))
    await store.save_memory(MemoryEntry(kind="user_preference", text="concise"))
    await store.save_memory(MemoryEntry(kind="project_fact", text="python 3.12"))

    facts = await store.list_memory(kind="project_fact")
    assert len(facts) == 2
    assert all(e.kind == "project_fact" for e in facts)

    prefs = await store.list_memory(kind="user_preference")
    assert len(prefs) == 1


@pytest.mark.asyncio
async def test_list_limit(store: InMemoryStorage) -> None:
    for i in range(5):
        await store.save_memory(MemoryEntry(kind="project_fact", text=f"fact {i}"))

    limited = await store.list_memory(limit=3)
    assert len(limited) == 3


@pytest.mark.asyncio
async def test_list_sorted_newest_first(store: InMemoryStorage) -> None:
    from datetime import UTC, datetime, timedelta

    old = MemoryEntry(kind="project_fact", text="old", created_at=datetime(2020, 1, 1, tzinfo=UTC))
    new = MemoryEntry(
        kind="project_fact", text="new", created_at=datetime.now(UTC) + timedelta(seconds=1)
    )
    await store.save_memory(old)
    await store.save_memory(new)

    entries = await store.list_memory()
    assert entries[0].text == "new"
    assert entries[1].text == "old"


@pytest.mark.asyncio
async def test_search_case_insensitive(store: InMemoryStorage) -> None:
    await store.save_memory(MemoryEntry(kind="project_fact", text="Uses UV Workspace"))
    await store.save_memory(MemoryEntry(kind="project_fact", text="Python version 3.12"))

    results = await store.search_memory("uv")
    assert len(results) == 1
    assert "UV" in results[0].text


@pytest.mark.asyncio
async def test_search_no_results(store: InMemoryStorage) -> None:
    await store.save_memory(MemoryEntry(kind="project_fact", text="something else"))
    results = await store.search_memory("xyznotfound")
    assert results == []


@pytest.mark.asyncio
async def test_delete(store: InMemoryStorage) -> None:
    entry = await store.save_memory(MemoryEntry(kind="user_fact", text="to delete"))
    entries = await store.list_memory()
    assert len(entries) == 1

    await store.delete_memory(entry.id)
    entries = await store.list_memory()
    assert entries == []


@pytest.mark.asyncio
async def test_delete_nonexistent_is_noop(store: InMemoryStorage) -> None:
    await store.delete_memory("mem_doesnotexist")  # should not raise


@pytest.mark.asyncio
async def test_save_overwrites_same_id(store: InMemoryStorage) -> None:
    entry = MemoryEntry(kind="project_fact", text="original")
    await store.save_memory(entry)
    updated = entry.model_copy(update={"text": "updated"})
    await store.save_memory(updated)

    entries = await store.list_memory()
    assert len(entries) == 1
    assert entries[0].text == "updated"


@pytest.mark.asyncio
async def test_deep_copy_isolation(store: InMemoryStorage) -> None:
    entry = MemoryEntry(kind="project_fact", text="original")
    saved = await store.save_memory(entry)
    # Mutating the returned copy should not affect the store
    saved.text = "mutated"  # type: ignore[misc]
    entries = await store.list_memory()
    assert entries[0].text == "original"
