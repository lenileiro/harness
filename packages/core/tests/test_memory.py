"""Tests for MemoryEntry and MemoryStore protocol."""

from __future__ import annotations

from harness.core.memory import MemoryEntry, MemoryKind, MemoryStore


def test_memory_entry_defaults() -> None:
    entry = MemoryEntry(kind="project_fact", text="uses uv")
    assert entry.id.startswith("mem_")
    assert len(entry.id) == 16  # "mem_" + 12 hex chars
    assert entry.kind == "project_fact"
    assert entry.text == "uses uv"
    assert entry.session_id is None
    assert entry.task_id is None
    assert entry.created_at is not None


def test_memory_entry_with_session() -> None:
    entry = MemoryEntry(kind="user_preference", text="prefers concise", session_id="sess_abc")
    assert entry.session_id == "sess_abc"
    assert entry.task_id is None


def test_memory_entry_model_dump_row() -> None:
    entry = MemoryEntry(kind="user_fact", text="senior engineer")
    row = entry.model_dump_row()
    assert row["id"] == entry.id
    assert row["kind"] == "user_fact"
    assert row["text"] == "senior engineer"
    assert "T" in row["created_at"]  # ISO datetime has T separator


def test_memory_store_is_protocol() -> None:
    assert hasattr(MemoryStore, "__protocol_attrs__")


def test_memory_store_runtime_checkable() -> None:
    class FakeStore:
        async def save_memory(self, entry: MemoryEntry) -> MemoryEntry:
            return entry

        async def list_memory(
            self, *, kind: MemoryKind | None = None, limit: int = 50
        ) -> list[MemoryEntry]:
            return []

        async def search_memory(self, query: str, *, limit: int = 20) -> list[MemoryEntry]:
            return []

        async def delete_memory(self, entry_id: str) -> None:
            pass

    assert isinstance(FakeStore(), MemoryStore)


def test_all_kinds_valid() -> None:
    for kind in ("user_preference", "user_fact", "project_fact", "project_context"):
        entry = MemoryEntry(kind=kind, text="test")  # type: ignore[arg-type]
        assert entry.kind == kind
