"""The model budget includes dynamically loaded instructions and tool schemas."""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from harness.core import (
    Agent,
    ErrorEvent,
    FailoverPolicy,
    Message,
    RunRequest,
    Session,
    ToolCall,
    ToolRegistry,
)
from harness.core.budget import ContextBudget, count_tokens
from harness.core.skills import SkillLibrary
from harness.storage.memory import InMemoryStorage

from .conftest import MockAdapter, MockTool, text_turn


def _library(root: Path, body: str) -> SkillLibrary:
    directory = root / "budget-skill"
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        "---\nname: budget-skill\ndescription: Instructions used in the budget test.\n---\n" + body
    )
    return SkillLibrary.load([root])


@pytest.mark.parametrize("source", ["instruction", "skill", "schema"])
async def test_oversized_injected_context_fails_before_model_call_and_closes_toolset(
    tmp_path: Path,
    source: str,
) -> None:
    store = InMemoryStorage()
    registry = ToolRegistry()
    adapter = MockAdapter("mock", scripts=[text_turn("must not run")])
    managed = MockTool(
        name="managed",
        description=("schema detail " * 600 if source == "schema" else "A managed tool."),
    )
    closed: list[bool] = []

    @asynccontextmanager
    async def factory():
        try:
            yield [managed]
        finally:
            closed.append(True)

    library = (
        _library(tmp_path / "skills", "skill instruction " * 600) if source == "skill" else None
    )
    await store.save(
        Session(
            id="budget-session",
            provider="mock",
            model="test-model",
            cwd=tmp_path,
            metadata={"active_skills": ["budget-skill"]} if library else {},
        )
    )
    agent = Agent(
        adapters={"mock": adapter},
        tools=registry,
        storage=store,
        failover=FailoverPolicy(chain=["mock"]),
        toolset_factory=factory,
        skill_library=library,
        system_prompt="system instruction " * 600 if source == "instruction" else None,
        budget=ContextBudget(max_tokens=200),
    )
    events = [
        event
        async for event in agent.run(
            RunRequest(prompt="continue", model="test-model", session_id="budget-session")
        )
    ]
    errors = [event for event in events if isinstance(event, ErrorEvent)]
    assert len(errors) == 1 and errors[0].kind == "configuration"
    assert "exceed the configured context budget" in errors[0].error
    assert not adapter.calls and not managed.calls
    assert closed == [True] and not registry.has("managed")
    session = await store.get("budget-session")
    assert session is not None and session.status == "failed"


async def test_prefix_reservation_prunes_history_but_keeps_tool_pairs_and_persisted_history(
    tmp_path: Path,
) -> None:
    store = InMemoryStorage()
    registry = ToolRegistry()
    adapter = MockAdapter("mock", scripts=[text_turn("done")])
    managed = MockTool(name="managed", description="Read a small result from a managed server.")

    @asynccontextmanager
    async def factory():
        yield [managed]

    library = _library(tmp_path / "skills", "Keep the newest verified result with its tool call.")
    oldest = ToolCall(id="old-call", name="managed", arguments={"text": "old"})
    newest = ToolCall(id="new-call", name="managed", arguments={"text": "new"})
    history = [
        Message(role="user", content="Keep track of this task."),
        Message(role="assistant", tool_calls=[oldest]),
        Message(
            role="tool", name="managed", tool_call_id=oldest.id, content="obsolete-history " * 1000
        ),
        Message(role="assistant", tool_calls=[newest]),
        Message(role="tool", name="managed", tool_call_id=newest.id, content="newest-evidence"),
    ]
    request = RunRequest(prompt="continue", model="test-model", session_id="history-session")
    # The unprefixed transcript fits. Only reserving instructions/tool schemas
    # makes pruning necessary, so this catches the previous history-only budget.
    budget = ContextBudget(
        max_tokens=count_tokens(
            [*history, Message(role="user", content=request.prompt)], "test-model"
        )
        + 10,
        keep_first_n=1,
        keep_last_n=2,
    )
    await store.save(
        Session(
            id=request.session_id,
            provider="mock",
            model="test-model",
            cwd=tmp_path,
            messages=history,
            metadata={"active_skills": ["budget-skill"]},
        )
    )
    agent = Agent(
        adapters={"mock": adapter},
        tools=registry,
        storage=store,
        activity_store=store,
        failover=FailoverPolicy(chain=["mock"]),
        toolset_factory=factory,
        skill_library=library,
        system_prompt="Follow the configured approval policy and retain evidence.",
        budget=budget,
    )
    events = [event async for event in agent.run(request)]
    assert not any(isinstance(event, ErrorEvent) for event in events)
    assert len(adapter.calls) == 1
    sent = adapter.calls[0]["messages"]
    assert any("Active skill budget-skill" in (message.content or "") for message in sent)
    assert any("configured approval policy" in (message.content or "") for message in sent)
    assert not any("obsolete-history" in (message.content or "") for message in sent)
    calls = {call.id for message in sent for call in message.tool_calls or []}
    results = {message.tool_call_id for message in sent if message.role == "tool"}
    assert calls == results == {"new-call"}
    schema_tokens = count_tokens(
        [Message(role="system", content=json.dumps(adapter.calls[0]["tools"]))], "test-model"
    )
    assert count_tokens(sent, "test-model") + schema_tokens <= budget.max_tokens
    saved = await store.get(request.session_id)
    assert saved is not None and any(
        "obsolete-history" in (message.content or "") for message in saved.messages
    )
    assert not registry.has("managed")
