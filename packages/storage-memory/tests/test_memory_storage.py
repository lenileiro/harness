"""Storage protocol tests for InMemoryStorage."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from harness.core import Message, Session
from harness.storage.memory import InMemoryStorage


def _make_session(*, id_: str = "sess_x", status: str = "pending") -> Session:
    return Session(
        id=id_,
        provider="ollama",
        model="llama3.2",
        cwd=Path.cwd(),
        messages=[Message(role="user", content="hi")],
        status=status,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
class TestInMemoryStorage:
    async def test_get_missing_returns_none(self) -> None:
        storage = InMemoryStorage()
        assert await storage.get("nope") is None

    async def test_save_then_get_round_trip(self) -> None:
        storage = InMemoryStorage()
        s = _make_session()
        await storage.save(s)
        loaded = await storage.get(s.id)
        assert loaded is not None
        assert loaded.id == s.id
        assert loaded.messages == s.messages

    async def test_save_isolates_external_mutations(self) -> None:
        storage = InMemoryStorage()
        s = _make_session()
        await storage.save(s)
        s.messages.append(Message(role="assistant", content="injected"))

        loaded = await storage.get(s.id)
        assert loaded is not None
        # Mutation after save must not leak into the stored copy.
        assert len(loaded.messages) == 1

    async def test_get_isolates_loaded_mutations(self) -> None:
        storage = InMemoryStorage()
        await storage.save(_make_session())
        first = await storage.get("sess_x")
        assert first is not None
        first.messages.append(Message(role="assistant", content="local"))

        second = await storage.get("sess_x")
        assert second is not None
        assert len(second.messages) == 1

    async def test_list_returns_newest_first(self) -> None:
        storage = InMemoryStorage()
        a = _make_session(id_="sess_a")
        await storage.save(a)
        await asyncio.sleep(0.001)  # ensure distinct updated_at
        b = _make_session(id_="sess_b")
        await storage.save(b)

        ids = [s.id for s in await storage.list()]
        assert ids == ["sess_b", "sess_a"]

    async def test_list_respects_limit(self) -> None:
        storage = InMemoryStorage()
        for i in range(5):
            await storage.save(_make_session(id_=f"sess_{i}"))
            await asyncio.sleep(0.001)
        results = await storage.list(limit=2)
        assert len(results) == 2

    async def test_list_filters_by_status(self) -> None:
        storage = InMemoryStorage()
        await storage.save(_make_session(id_="sess_a", status="done"))
        await storage.save(_make_session(id_="sess_b", status="failed"))
        await storage.save(_make_session(id_="sess_c", status="done"))
        results = await storage.list(status="done")
        ids = {s.id for s in results}
        assert ids == {"sess_a", "sess_c"}

    async def test_list_filters_by_before(self) -> None:
        storage = InMemoryStorage()
        s = _make_session(id_="sess_a")
        await storage.save(s)
        future = datetime.now(UTC) + timedelta(seconds=1)
        past = datetime.now(UTC) - timedelta(seconds=1)
        assert len(await storage.list(before=future)) == 1
        assert len(await storage.list(before=past)) == 0

    async def test_delete(self) -> None:
        storage = InMemoryStorage()
        await storage.save(_make_session())
        await storage.delete("sess_x")
        assert await storage.get("sess_x") is None

    async def test_delete_missing_is_noop(self) -> None:
        storage = InMemoryStorage()
        await storage.delete("never_existed")  # must not raise
