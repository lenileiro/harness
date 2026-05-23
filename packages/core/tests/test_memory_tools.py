"""Tests for the Memory-as-Action tools: NotesTool + PruneLedgerTool."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.core import NotesTool, PruneLedgerTool, Session
from harness.core.schemas import Message, ToolCall


def _new_session() -> Session:
    return Session(provider="x", model="y", cwd=Path("/tmp"))


def _call(name: str, args: dict, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=args)


@pytest.mark.asyncio
class TestNotesTool:
    async def test_add_and_list(self) -> None:
        sess = _new_session()
        tool = NotesTool(session=sess)
        result = await tool(_call("notes", {"action": "add", "text": "hello"}))
        assert not result.is_error
        listed = await tool(_call("notes", {"action": "list"}, call_id="c2"))
        assert "hello" in (listed.content or "")
        assert len(sess.notes) == 1

    async def test_add_with_tags(self) -> None:
        sess = _new_session()
        tool = NotesTool(session=sess)
        await tool(_call("notes", {"action": "add", "text": "scoped", "tags": ["t1", "t2"]}))
        assert sess.notes[0].tags == ["t1", "t2"]

    async def test_add_rejects_empty(self) -> None:
        sess = _new_session()
        tool = NotesTool(session=sess)
        result = await tool(_call("notes", {"action": "add", "text": ""}))
        assert result.is_error

    async def test_add_rejects_too_long(self) -> None:
        sess = _new_session()
        tool = NotesTool(session=sess)
        result = await tool(_call("notes", {"action": "add", "text": "x" * 2000}))
        assert result.is_error

    async def test_delete_by_id(self) -> None:
        sess = _new_session()
        tool = NotesTool(session=sess)
        added = await tool(_call("notes", {"action": "add", "text": "rm me"}))
        assert added.metadata is not None
        note_id = added.metadata["id"]
        deleted = await tool(_call("notes", {"action": "delete", "id": note_id}, call_id="c2"))
        assert not deleted.is_error
        assert sess.notes == []

    async def test_delete_unknown_id_errors(self) -> None:
        sess = _new_session()
        tool = NotesTool(session=sess)
        result = await tool(_call("notes", {"action": "delete", "id": "nope"}))
        assert result.is_error

    async def test_list_empty_session(self) -> None:
        sess = _new_session()
        tool = NotesTool(session=sess)
        result = await tool(_call("notes", {"action": "list"}))
        assert "no notes" in (result.content or "")

    async def test_invalid_action(self) -> None:
        sess = _new_session()
        tool = NotesTool(session=sess)
        result = await tool(_call("notes", {"action": "frobnicate"}))
        assert result.is_error


@pytest.mark.asyncio
class TestPruneLedgerTool:
    async def test_no_op_when_few_pairs(self) -> None:
        sess = _new_session()
        sess.messages = [
            Message(role="user", content="do thing"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="t1", name="read_file", arguments={})],
            ),
            Message(role="tool", content="ok", tool_call_id="t1", name="read_file"),
        ]
        tool = PruneLedgerTool(session=sess)
        result = await tool(_call("prune_ledger", {"keep_recent_turns": 4}))
        assert not result.is_error
        assert result.metadata is not None
        assert result.metadata["dropped_pairs"] == 0
        assert len(sess.messages) == 3

    async def test_drops_old_pairs_keeps_recent(self) -> None:
        sess = _new_session()
        sess.messages.append(Message(role="user", content="task"))
        for i in range(5):
            sess.messages.append(
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[ToolCall(id=f"t{i}", name="read_file", arguments={})],
                )
            )
            sess.messages.append(
                Message(role="tool", content=f"result-{i}", tool_call_id=f"t{i}", name="read_file")
            )
        # 5 pairs + 1 user message = 11 messages total
        tool = PruneLedgerTool(session=sess)
        result = await tool(_call("prune_ledger", {"keep_recent_turns": 2}))
        assert not result.is_error
        assert result.metadata is not None
        assert result.metadata["dropped_pairs"] == 3
        # User message kept + 2 pairs kept = 1 + 4 = 5 messages
        assert len(sess.messages) == 5
        assert sess.messages[0].role == "user"
        # Most recent results stay
        assert any("result-4" in (m.content or "") for m in sess.messages)
        # Older results dropped
        assert not any("result-0" in (m.content or "") for m in sess.messages)

    async def test_never_drops_system_or_user(self) -> None:
        sess = _new_session()
        sess.messages = [
            Message(role="system", content="sys"),
            Message(role="user", content="u1"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="t1", name="read_file", arguments={})],
            ),
            Message(role="tool", content="r1", tool_call_id="t1", name="read_file"),
            Message(role="user", content="u2"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="t2", name="read_file", arguments={})],
            ),
            Message(role="tool", content="r2", tool_call_id="t2", name="read_file"),
        ]
        tool = PruneLedgerTool(session=sess)
        await tool(_call("prune_ledger", {"keep_recent_turns": 1}))
        roles_kept = [m.role for m in sess.messages]
        # System + both users always preserved
        assert "system" in roles_kept
        assert roles_kept.count("user") == 2
