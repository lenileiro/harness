"""ApprovalStore tests for SQLiteStorage."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from harness.storage.sqlite import SQLiteStorage
from harness.tasks import PendingApproval


def _approval(**overrides: object) -> PendingApproval:
    defaults: dict[str, object] = {
        "session_id": "sess_1",
        "tool_call_id": "call_1",
        "tool_name": "shell",
        "arguments": {"command": "pwd"},
    }
    defaults.update(overrides)
    return PendingApproval(**defaults)  # type: ignore[arg-type]


@pytest.fixture
async def storage(tmp_path: Path):
    s = SQLiteStorage(path=tmp_path / "approvals.db")
    try:
        yield s
    finally:
        await s.close()


@pytest.mark.asyncio
class TestApprovalStore:
    async def test_create_round_trip(self, storage: SQLiteStorage) -> None:
        saved = await storage.create_approval(_approval())
        loaded = await storage.get_approval(saved.id)
        assert loaded is not None
        assert loaded.tool_name == "shell"
        assert loaded.arguments == {"command": "pwd"}
        assert loaded.status == "pending"

    async def test_get_missing_returns_none(self, storage: SQLiteStorage) -> None:
        assert await storage.get_approval("nope") is None

    async def test_list_filters(self, storage: SQLiteStorage) -> None:
        a = await storage.create_approval(_approval(session_id="s1"))
        b = await storage.create_approval(_approval(session_id="s2", task_id="t1"))
        await asyncio.sleep(0)  # ensure flush

        assert {x.id for x in await storage.list_approvals(session_id="s1")} == {a.id}
        assert {x.id for x in await storage.list_approvals(task_id="t1")} == {b.id}

        await storage.resolve_approval(a.id, status="granted")
        granted = await storage.list_approvals(status="granted")
        assert [x.id for x in granted] == [a.id]

    async def test_resolve_round_trip(self, storage: SQLiteStorage) -> None:
        saved = await storage.create_approval(_approval())
        updated = await storage.resolve_approval(saved.id, status="denied", resolved_by="ci")
        assert updated is not None
        assert updated.status == "denied"
        assert updated.resolved_at is not None
        assert updated.resolved_by == "ci"

    async def test_resolve_missing_returns_none(self, storage: SQLiteStorage) -> None:
        assert await storage.resolve_approval("nope", status="granted") is None

    async def test_mark_replayed_idempotent(self, storage: SQLiteStorage) -> None:
        saved = await storage.create_approval(_approval())
        await storage.resolve_approval(saved.id, status="granted")
        await storage.mark_replayed(saved.id)
        first = await storage.get_approval(saved.id)
        first_replayed = first.replayed_at  # type: ignore[union-attr]
        await asyncio.sleep(0.01)
        await storage.mark_replayed(saved.id)  # no-op
        second = await storage.get_approval(saved.id)
        assert second.replayed_at == first_replayed  # type: ignore[union-attr]

    async def test_list_unreplayed_granted(self, storage: SQLiteStorage) -> None:
        pending = await storage.create_approval(_approval(tool_call_id="p"))
        granted = await storage.create_approval(_approval(tool_call_id="g"))
        await storage.resolve_approval(granted.id, status="granted")
        replayed = await storage.create_approval(_approval(tool_call_id="r"))
        await storage.resolve_approval(replayed.id, status="granted")
        await storage.mark_replayed(replayed.id)

        result = await storage.list_unreplayed_granted(session_id="sess_1")
        assert [a.id for a in result] == [granted.id]
        ids = {a.id for a in result}
        assert pending.id not in ids
        assert replayed.id not in ids

    async def test_persists_across_instances(self, tmp_path: Path) -> None:
        db_path = tmp_path / "persist.db"
        a = SQLiteStorage(path=db_path)
        try:
            saved = await a.create_approval(_approval())
            saved_id = saved.id
        finally:
            await a.close()

        b = SQLiteStorage(path=db_path)
        try:
            loaded = await b.get_approval(saved_id)
            assert loaded is not None
            assert loaded.tool_name == "shell"
        finally:
            await b.close()
