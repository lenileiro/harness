"""Tests for VerifierRouter."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from harness.core.activity import ActivityEvent
from harness.core.events import Done, Event, TextDelta
from harness.core.schemas import Capabilities, Message, Session
from harness.core.verification import LLMJudgeVerifier, RuleVerifier, VerifierRouter


def _make_session() -> Session:
    return Session(provider="ollama", model="llama3.2", cwd=Path("/tmp"))


def _tool_event(name: str, is_error: bool = False) -> ActivityEvent:
    return ActivityEvent(
        kind="tool_call.completed",
        data={"name": name, "is_error": is_error},
    )


class AlwaysPassLLMAdapter:
    """Fake adapter that returns a passing judge response."""

    async def stream(self, **_kwargs: Any) -> AsyncIterator[Event]:  # type: ignore[override]
        content = '{"can_finish": true, "reason": "looks good", "confidence": 0.9}'
        yield TextDelta(text=content)
        yield Done(final_message=Message(role="assistant", content=content))

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=False)

    async def cancel(self, session_id: str) -> None:
        pass


class AlwaysFailLLMAdapter:
    """Fake adapter that returns a failing judge response."""

    async def stream(self, **_kwargs: Any) -> AsyncIterator[Event]:  # type: ignore[override]
        content = '{"can_finish": false, "reason": "something wrong", "confidence": 0.8}'
        yield TextDelta(text=content)
        yield Done(final_message=Message(role="assistant", content=content))

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=False)

    async def cancel(self, session_id: str) -> None:
        pass


@pytest.mark.asyncio
async def test_no_tools_routes_to_llm_judge() -> None:
    # No tools dispatched → router must use the LLM judge, not rule.
    # The model may have verbally claimed to do work without calling any tool.
    router = VerifierRouter(
        rule=RuleVerifier(),
        llm=LLMJudgeVerifier(adapter=AlwaysFailLLMAdapter(), model="m"),  # type: ignore[arg-type]
    )
    session = _make_session()
    result = await router.verify(session=session, activity=[])
    # AlwaysFailLLMAdapter returns can_finish=False → proves judge was called, not rule
    assert result.can_finish is False
    assert result.verifier_name == "router"


@pytest.mark.asyncio
async def test_no_tools_llm_judge_can_pass() -> None:
    # When the LLM judge says it's fine (e.g. a factual Q&A with no tools needed), trust it
    router = VerifierRouter(
        rule=RuleVerifier(),
        llm=LLMJudgeVerifier(adapter=AlwaysPassLLMAdapter(), model="m"),  # type: ignore[arg-type]
    )
    session = _make_session()
    result = await router.verify(session=session, activity=[])
    assert result.can_finish is True
    assert result.verifier_name == "router"


@pytest.mark.asyncio
async def test_readonly_tools_routes_to_rule() -> None:
    router = VerifierRouter(
        rule=RuleVerifier(),
        llm=LLMJudgeVerifier(adapter=AlwaysFailLLMAdapter(), model="m"),  # type: ignore[arg-type]
    )
    session = _make_session()
    activity = [_tool_event("read_file"), _tool_event("list_dir")]
    result = await router.verify(session=session, activity=activity)
    assert result.can_finish is True
    assert result.verifier_name == "router"


@pytest.mark.asyncio
async def test_write_file_routes_to_llm() -> None:
    router = VerifierRouter(
        rule=RuleVerifier(),
        llm=LLMJudgeVerifier(adapter=AlwaysPassLLMAdapter(), model="m"),  # type: ignore[arg-type]
    )
    session = _make_session()
    activity = [_tool_event("read_file"), _tool_event("write_file")]
    result = await router.verify(session=session, activity=activity)
    assert result.can_finish is True
    assert result.verifier_name == "router"


@pytest.mark.asyncio
async def test_edit_file_routes_to_llm() -> None:
    router = VerifierRouter(
        rule=RuleVerifier(),
        llm=LLMJudgeVerifier(adapter=AlwaysPassLLMAdapter(), model="m"),  # type: ignore[arg-type]
    )
    session = _make_session()
    activity = [_tool_event("edit_file")]
    result = await router.verify(session=session, activity=activity)
    assert result.verifier_name == "router"


@pytest.mark.asyncio
async def test_shell_routes_to_llm() -> None:
    router = VerifierRouter(
        rule=RuleVerifier(),
        llm=LLMJudgeVerifier(adapter=AlwaysPassLLMAdapter(), model="m"),  # type: ignore[arg-type]
    )
    session = _make_session()
    activity = [_tool_event("shell")]
    result = await router.verify(session=session, activity=activity)
    assert result.verifier_name == "router"


@pytest.mark.asyncio
async def test_llm_failure_surfaced_through_router() -> None:
    router = VerifierRouter(
        rule=RuleVerifier(),
        llm=LLMJudgeVerifier(adapter=AlwaysFailLLMAdapter(), model="m"),  # type: ignore[arg-type]
    )
    session = _make_session()
    activity = [_tool_event("write_file")]
    result = await router.verify(session=session, activity=activity)
    assert result.can_finish is False
    assert result.verifier_name == "router"


@pytest.mark.asyncio
async def test_rule_error_propagated_through_router() -> None:
    router = VerifierRouter(
        rule=RuleVerifier(),
        llm=LLMJudgeVerifier(adapter=AlwaysPassLLMAdapter(), model="m"),  # type: ignore[arg-type]
    )
    session = _make_session()
    activity = [_tool_event("read_file", is_error=True)]
    result = await router.verify(session=session, activity=activity)
    assert result.can_finish is False
    assert result.verifier_name == "router"
