import asyncio

import pytest

from harness.core import PendingApproval
from harness.storage.memory import InMemoryStorage
from harness.storage.sqlite import SQLiteStorage


@pytest.mark.parametrize("backend", ["sqlite", "memory"])
async def test_resolution_and_claim_have_one_winner(tmp_path, backend):
    first = SQLiteStorage(path=tmp_path / "db") if backend == "sqlite" else InMemoryStorage()
    second = SQLiteStorage(path=tmp_path / "db") if backend == "sqlite" else first
    try:
        approval = await first.create_approval(
            PendingApproval(session_id="s", tool_call_id="c", tool_name="write")
        )
        decisions = await asyncio.gather(
            first.resolve_approval(approval.id, status="granted"),
            second.resolve_approval(approval.id, status="granted"),
        )
        assert sum(item is not None for item in decisions) == 1
        assert await first.resolve_approval(approval.id, status="denied") is None
        assert not await first.claim_replay(approval.id, session_id="other")
        claims = await asyncio.gather(
            first.claim_replay(approval.id, session_id="s"),
            second.claim_replay(approval.id, session_id="s"),
        )
        assert claims.count(True) == 1
        persisted = await second.get_approval(approval.id)
        assert persisted is not None and persisted.replay_claimed_at is not None
        if isinstance(second, SQLiteStorage):
            await second.close()
            second = SQLiteStorage(path=tmp_path / "db")
            assert not await second.claim_replay(approval.id, session_id="s")
            assert len(await second.list_unreplayed_granted(session_id="s")) == 1
    finally:
        if isinstance(first, SQLiteStorage):
            await first.close()
        if isinstance(second, SQLiteStorage):
            await second.close()
