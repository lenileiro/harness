"""SQLiteStorage tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from harness.core import Message, Session, ToolCall
from harness.storage.sqlite import SQLiteStorage, default_db_path


def _make_session(
    *,
    id_: str = "sess_x",
    status: str = "pending",
    messages: list[Message] | None = None,
) -> Session:
    return Session(
        id=id_,
        provider="ollama",
        model="llama3.2",
        cwd=Path.cwd(),
        messages=messages or [Message(role="user", content="hi")],
        status=status,  # type: ignore[arg-type]
    )


@pytest.fixture
async def storage(tmp_path: Path):
    s = SQLiteStorage(path=tmp_path / "test.db")
    try:
        yield s
    finally:
        await s.close()


@pytest.mark.asyncio
class TestSQLiteStorage:
    async def test_get_missing_returns_none(self, storage: SQLiteStorage) -> None:
        assert await storage.get("nope") is None

    async def test_save_then_get_round_trip(self, storage: SQLiteStorage) -> None:
        s = _make_session()
        await storage.save(s)
        loaded = await storage.get(s.id)
        assert loaded is not None
        assert loaded.id == s.id
        assert loaded.provider == s.provider
        assert loaded.messages == s.messages

    async def test_complex_message_with_tool_calls_round_trips(
        self, storage: SQLiteStorage
    ) -> None:
        s = _make_session(
            messages=[
                Message(role="user", content="ping x"),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id="c1", name="ping", arguments={"host": "x.com"})],
                ),
                Message(role="tool", tool_call_id="c1", name="ping", content="pong"),
                Message(role="assistant", content="done"),
            ]
        )
        await storage.save(s)
        loaded = await storage.get(s.id)
        assert loaded is not None
        assert loaded.messages == s.messages

    async def test_save_updates_existing(self, storage: SQLiteStorage) -> None:
        s = _make_session()
        await storage.save(s)
        s.status = "done"
        s.messages.append(Message(role="assistant", content="ok"))
        await storage.save(s)

        loaded = await storage.get(s.id)
        assert loaded is not None
        assert loaded.status == "done"
        assert len(loaded.messages) == 2

    async def test_list_returns_newest_first(self, storage: SQLiteStorage) -> None:
        a = _make_session(id_="sess_a")
        await storage.save(a)
        await asyncio.sleep(0.005)
        b = _make_session(id_="sess_b")
        await storage.save(b)
        ids = [s.id for s in await storage.list()]
        assert ids == ["sess_b", "sess_a"]

    async def test_list_respects_limit(self, storage: SQLiteStorage) -> None:
        for i in range(5):
            await storage.save(_make_session(id_=f"sess_{i}"))
            await asyncio.sleep(0.001)
        assert len(await storage.list(limit=2)) == 2

    async def test_list_filters_by_status(self, storage: SQLiteStorage) -> None:
        await storage.save(_make_session(id_="sess_a", status="done"))
        await storage.save(_make_session(id_="sess_b", status="failed"))
        await storage.save(_make_session(id_="sess_c", status="done"))
        ids = {s.id for s in await storage.list(status="done")}
        assert ids == {"sess_a", "sess_c"}

    async def test_list_filters_by_before(self, storage: SQLiteStorage) -> None:
        await storage.save(_make_session(id_="sess_a"))
        future = datetime.now(UTC) + timedelta(seconds=1)
        past = datetime.now(UTC) - timedelta(seconds=1)
        assert len(await storage.list(before=future)) == 1
        assert len(await storage.list(before=past)) == 0

    async def test_delete(self, storage: SQLiteStorage) -> None:
        await storage.save(_make_session())
        await storage.delete("sess_x")
        assert await storage.get("sess_x") is None

    async def test_delete_missing_is_noop(self, storage: SQLiteStorage) -> None:
        await storage.delete("never")  # must not raise

    async def test_persists_across_storage_instances(self, tmp_path: Path) -> None:
        db_path = tmp_path / "persistent.db"

        a = SQLiteStorage(path=db_path)
        try:
            await a.save(_make_session(id_="sess_persist"))
        finally:
            await a.close()

        b = SQLiteStorage(path=db_path)
        try:
            loaded = await b.get("sess_persist")
            assert loaded is not None
            assert loaded.id == "sess_persist"
        finally:
            await b.close()


class TestDefaultDbPath:
    def test_uses_xdg_state_home_when_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_STATE_HOME", "/tmp/xdg-test")
        assert default_db_path() == Path("/tmp/xdg-test/harness/sessions.db")

    def test_falls_back_to_home_local_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XDG_STATE_HOME", raising=False)
        path = default_db_path()
        assert path.parts[-2:] == ("harness", "sessions.db")
        assert ".local/state" in str(path)
