"""Tests for the context budget governor (Phase 6.6)."""

from __future__ import annotations

import pytest

from harness.core import (
    Agent,
    AutoApprove,
    ContextBudget,
    FailoverPolicy,
    Message,
    RunRequest,
    ToolCall,
    ToolRegistry,
    count_tokens,
    prune,
)
from harness.core import activity as activity_kinds
from harness.core.budget import _atomic_blocks

from .conftest import MockAdapter, MockStorage, text_turn
from .test_runtime_activity import InMemoryActivitySink

# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------


class TestCountTokens:
    def test_empty_history_is_zero(self) -> None:
        assert count_tokens([], "gpt-4o-mini") == 0

    def test_single_short_message(self) -> None:
        msgs = [Message(role="user", content="hello")]
        # Should be small but positive.
        total = count_tokens(msgs, "gpt-4o-mini")
        assert 1 <= total <= 20

    def test_token_count_grows_with_content(self) -> None:
        short = [Message(role="user", content="hi")]
        long = [Message(role="user", content="hi " * 500)]
        assert count_tokens(long, "gpt-4o-mini") > count_tokens(short, "gpt-4o-mini")

    def test_unknown_model_falls_back(self) -> None:
        # Ollama / Gemini / random model ids should not raise.
        msgs = [Message(role="user", content="hello world")]
        total = count_tokens(msgs, "llama3.1:8b-instruct")
        assert total > 0

    def test_tool_call_arguments_counted(self) -> None:
        call = ToolCall(id="c1", name="echo", arguments={"text": "abc"})
        with_call = [Message(role="assistant", content=None, tool_calls=[call])]
        plain = [Message(role="assistant", content="abc")]
        # The serialized JSON + call id push the count above the plain string.
        assert count_tokens(with_call, "gpt-4o-mini") >= count_tokens(plain, "gpt-4o-mini")


# ---------------------------------------------------------------------------
# _atomic_blocks
# ---------------------------------------------------------------------------


class TestAtomicBlocks:
    def test_simple_messages_one_per_block(self) -> None:
        msgs = [
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
            Message(role="assistant", content="hello"),
        ]
        blocks = _atomic_blocks(msgs)
        assert [len(b) for b in blocks] == [1, 1, 1]

    def test_tool_call_groups_with_results(self) -> None:
        call = ToolCall(id="c1", name="echo", arguments={"t": "x"})
        msgs = [
            Message(role="user", content="run echo"),
            Message(role="assistant", content=None, tool_calls=[call]),
            Message(role="tool", content="x", tool_call_id="c1", name="echo"),
            Message(role="assistant", content="done"),
        ]
        blocks = _atomic_blocks(msgs)
        # user | (assistant+tool) | assistant
        assert [len(b) for b in blocks] == [1, 2, 1]
        assert blocks[1][0].role == "assistant"
        assert blocks[1][1].role == "tool"

    def test_multiple_tool_calls_in_one_assistant(self) -> None:
        c1 = ToolCall(id="c1", name="a", arguments={})
        c2 = ToolCall(id="c2", name="b", arguments={})
        msgs = [
            Message(role="assistant", content=None, tool_calls=[c1, c2]),
            Message(role="tool", content="a-out", tool_call_id="c1", name="a"),
            Message(role="tool", content="b-out", tool_call_id="c2", name="b"),
            Message(role="user", content="thanks"),
        ]
        blocks = _atomic_blocks(msgs)
        assert [len(b) for b in blocks] == [3, 1]

    def test_orphan_tool_message_gets_own_block(self) -> None:
        # The runtime shouldn't produce this, but the pruner must not crash.
        msgs = [Message(role="tool", content="?", tool_call_id="nope", name="x")]
        blocks = _atomic_blocks(msgs)
        assert [len(b) for b in blocks] == [1]


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)  # type: ignore[arg-type]


class TestPrune:
    def test_under_budget_no_op(self) -> None:
        msgs = [_msg("system", "sys"), _msg("user", "hi")]
        budget = ContextBudget(max_tokens=10_000)
        out = prune(msgs, budget=budget, model="gpt-4o-mini")
        assert out == msgs

    def test_returns_new_list(self) -> None:
        msgs = [_msg("user", "hi")]
        out = prune(msgs, budget=ContextBudget(max_tokens=10_000), model="gpt-4o-mini")
        assert out is not msgs

    def test_over_budget_drops_middle(self) -> None:
        # 12 messages each ~50 tokens → ~600 tokens total.
        big = "word " * 50
        msgs = [_msg("system", "sys")]
        msgs += [_msg("user" if i % 2 == 0 else "assistant", big) for i in range(10)]
        msgs += [_msg("user", "final")]
        budget = ContextBudget(max_tokens=200, keep_first_n=1, keep_last_n=2)
        out = prune(msgs, budget=budget, model="gpt-4o-mini")
        # System prompt + final two messages must survive.
        assert out[0] is msgs[0]  # system
        assert out[-1] is msgs[-1]  # final
        assert out[-2] is msgs[-2]
        # We dropped some middle blocks.
        assert len(out) < len(msgs)

    def test_tool_pairs_never_split(self) -> None:
        """Crucial invariant: a pruned list cannot orphan a tool result."""
        call = ToolCall(id="c1", name="echo", arguments={"t": "x"})
        big = "word " * 100
        msgs = [
            _msg("system", "sys"),
            # Big middle blocks to push us over budget.
            _msg("user", big),
            Message(role="assistant", content=None, tool_calls=[call]),
            Message(role="tool", content=big, tool_call_id="c1", name="echo"),
            _msg("user", big),
            _msg("assistant", big),
            _msg("user", "final"),
        ]
        budget = ContextBudget(max_tokens=80, keep_first_n=1, keep_last_n=1)
        out = prune(msgs, budget=budget, model="gpt-4o-mini")
        _assert_no_orphans(out)

    def test_extreme_budget_returns_head_and_tail(self) -> None:
        big = "word " * 200
        msgs = [_msg("system", big), _msg("user", big), _msg("user", big), _msg("user", big)]
        budget = ContextBudget(max_tokens=10, keep_first_n=1, keep_last_n=1)
        out = prune(msgs, budget=budget, model="gpt-4o-mini")
        # Even though we still overshoot, the function returns head+tail rather
        # than crash or return an empty list.
        assert out[0] is msgs[0]
        assert out[-1] is msgs[-1]
        assert len(out) == 2


def _assert_no_orphans(msgs: list[Message]) -> None:
    """Every role=tool message must follow an assistant whose tool_calls
    include its tool_call_id; every assistant.tool_calls id must have a
    matching tool message somewhere after it in this list."""
    pending: set[str] = set()
    for m in msgs:
        if m.role == "assistant" and m.tool_calls:
            for tc in m.tool_calls:
                pending.add(tc.id)
        elif m.role == "tool":
            assert m.tool_call_id in pending, f"orphan tool message: {m.tool_call_id!r}"
            pending.discard(m.tool_call_id or "")
    assert not pending, f"assistant tool_calls without matching tool result: {pending}"


# ---------------------------------------------------------------------------
# Agent integration — context.pruned event
# ---------------------------------------------------------------------------


def _build_agent_with_budget(
    *,
    budget: ContextBudget | None,
    sink: InMemoryActivitySink,
    adapter: MockAdapter,
) -> tuple[Agent, MockStorage]:
    storage = MockStorage()
    agent = Agent(
        adapters={"mock": adapter},  # type: ignore[arg-type]
        tools=ToolRegistry(),
        storage=storage,
        failover=FailoverPolicy(chain=["mock"], max_attempts=1),
        approval_handler=AutoApprove(),
        activity_store=sink,
        budget=budget,
        default_model="gpt-4o-mini",
    )
    return agent, storage


async def _drain(it) -> None:
    async for _ in it:
        pass


@pytest.mark.asyncio
class TestAgentBudget:
    async def test_no_budget_no_pruned_event(self) -> None:
        sink = InMemoryActivitySink()
        adapter = MockAdapter("mock", scripts=[text_turn("hi")])
        agent, _ = _build_agent_with_budget(budget=None, sink=sink, adapter=adapter)
        await _drain(agent.run(RunRequest(prompt="ping", model="gpt-4o-mini")))
        kinds = [e.kind for e in sink.events]
        assert activity_kinds.CONTEXT_PRUNED not in kinds

    async def test_under_budget_no_pruned_event(self) -> None:
        sink = InMemoryActivitySink()
        adapter = MockAdapter("mock", scripts=[text_turn("hi")])
        budget = ContextBudget(max_tokens=10_000)
        agent, _ = _build_agent_with_budget(budget=budget, sink=sink, adapter=adapter)
        await _drain(agent.run(RunRequest(prompt="ping", model="gpt-4o-mini")))
        kinds = [e.kind for e in sink.events]
        assert activity_kinds.CONTEXT_PRUNED not in kinds

    async def test_over_budget_emits_pruned_event(self) -> None:
        sink = InMemoryActivitySink()
        adapter = MockAdapter("mock", scripts=[text_turn("ok")])
        # Stuff the session ahead-of-time with many bulky messages so the
        # very first turn already overshoots the budget.
        storage = MockStorage()
        big = "word " * 200
        pre_msgs = [_msg("system", "sys")] + [
            _msg("user" if i % 2 == 0 else "assistant", big) for i in range(12)
        ]
        from harness.core import Session

        session = Session(
            id="sess_budget",
            provider="mock",
            model="gpt-4o-mini",
            cwd=__import__("pathlib").Path.cwd(),
            messages=pre_msgs,
        )
        await storage.save(session)

        budget = ContextBudget(max_tokens=200, keep_first_n=1, keep_last_n=2)
        agent = Agent(
            adapters={"mock": adapter},  # type: ignore[arg-type]
            tools=ToolRegistry(),
            storage=storage,
            failover=FailoverPolicy(chain=["mock"], max_attempts=1),
            approval_handler=AutoApprove(),
            activity_store=sink,
            budget=budget,
            default_model="gpt-4o-mini",
        )
        await _drain(
            agent.run(RunRequest(prompt="ping", session_id="sess_budget", model="gpt-4o-mini"))
        )

        pruned = [e for e in sink.events if e.kind == activity_kinds.CONTEXT_PRUNED]
        assert pruned, "expected a context.pruned activity event"
        data = pruned[0].data
        assert data["max_tokens"] == 200
        assert data["messages_after"] < data["messages_before"]
        assert data["tokens_after"] <= data["tokens_before"]

        # The adapter must have been called with the pruned message list,
        # not the original 13+ message history.
        assert len(adapter.calls) == 1
        sent = adapter.calls[0]["messages"]
        assert len(sent) == data["messages_after"]
