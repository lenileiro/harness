"""ApprovalStore tests for InMemoryStorage."""

from __future__ import annotations

import asyncio

import pytest

from harness.storage.memory import InMemoryStorage
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


@pytest.mark.asyncio
class TestApprovalStore:
    async def test_create_round_trip(self) -> None:
        storage = InMemoryStorage()
        saved = await storage.create_approval(_approval())
        loaded = await storage.get_approval(saved.id)
        assert loaded is not None
        assert loaded.tool_name == "shell"
        assert loaded.status == "pending"

    async def test_get_missing_returns_none(self) -> None:
        storage = InMemoryStorage()
        assert await storage.get_approval("nope") is None

    async def test_list_newest_first(self) -> None:
        storage = InMemoryStorage()
        a = await storage.create_approval(_approval(tool_call_id="c1"))
        await asyncio.sleep(0.001)
        b = await storage.create_approval(_approval(tool_call_id="c2"))
        items = await storage.list_approvals()
        assert [x.id for x in items] == [b.id, a.id]

    async def test_filter_by_session_status_task(self) -> None:
        storage = InMemoryStorage()
        await storage.create_approval(_approval(session_id="s1"))
        await storage.create_approval(_approval(session_id="s2", task_id="t1"))
        assert len(await storage.list_approvals(session_id="s1")) == 1
        assert len(await storage.list_approvals(task_id="t1")) == 1

        only = await storage.create_approval(_approval(session_id="s3"))
        await storage.resolve_approval(only.id, status="granted", resolved_by="cli")
        granted = await storage.list_approvals(status="granted")
        assert [a.id for a in granted] == [only.id]

    async def test_resolve_sets_resolved_fields(self) -> None:
        storage = InMemoryStorage()
        saved = await storage.create_approval(_approval())
        updated = await storage.resolve_approval(saved.id, status="granted", resolved_by="bob")
        assert updated is not None
        assert updated.status == "granted"
        assert updated.resolved_at is not None
        assert updated.resolved_by == "bob"

    async def test_resolve_missing_returns_none(self) -> None:
        storage = InMemoryStorage()
        assert await storage.resolve_approval("nope", status="granted") is None

    async def test_mark_replayed_is_idempotent(self) -> None:
        storage = InMemoryStorage()
        saved = await storage.create_approval(_approval())
        await storage.resolve_approval(saved.id, status="granted")
        await storage.mark_replayed(saved.id)
        first = await storage.get_approval(saved.id)
        first_replayed_at = first.replayed_at  # type: ignore[union-attr]
        await storage.mark_replayed(saved.id)  # second call no-ops
        second = await storage.get_approval(saved.id)
        assert second.replayed_at == first_replayed_at  # type: ignore[union-attr]

    async def test_list_unreplayed_granted_filters_correctly(self) -> None:
        storage = InMemoryStorage()
        pending = await storage.create_approval(_approval(tool_call_id="p"))
        granted = await storage.create_approval(_approval(tool_call_id="g"))
        await storage.resolve_approval(granted.id, status="granted")
        denied = await storage.create_approval(_approval(tool_call_id="d"))
        await storage.resolve_approval(denied.id, status="denied")
        replayed = await storage.create_approval(_approval(tool_call_id="r"))
        await storage.resolve_approval(replayed.id, status="granted")
        await storage.mark_replayed(replayed.id)

        result = await storage.list_unreplayed_granted(session_id="sess_1")
        assert [a.id for a in result] == [granted.id]
        assert pending.id not in {a.id for a in result}
