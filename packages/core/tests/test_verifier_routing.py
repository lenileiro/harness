"""Tests for VerifierRouter and LLMJudgeVerifier retry logic."""

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


# ---------------------------------------------------------------------------
# LLMJudgeVerifier retry logic
# ---------------------------------------------------------------------------


async def _instant_sleep(_: float) -> None:
    """No-op sleep for tests — avoids patching the global asyncio.sleep."""


class AlwaysRaiseLLMAdapter:
    """Fake adapter that always raises on stream()."""

    async def stream(self, **_kwargs: Any) -> AsyncIterator[Event]:  # type: ignore[override]
        raise RuntimeError("network error")
        yield  # make it an async generator

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=False)

    async def cancel(self, session_id: str) -> None:
        pass


class BadJSONLLMAdapter:
    """Fake adapter that returns unparseable content."""

    async def stream(self, **_kwargs: Any) -> AsyncIterator[Event]:  # type: ignore[override]
        content = "sorry I cannot answer in JSON right now"
        yield TextDelta(text=content)
        yield Done(final_message=Message(role="assistant", content=content))

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=False)

    async def cancel(self, session_id: str) -> None:
        pass


class FlakyThenPassLLMAdapter:
    """Fails on the first call, succeeds on the second."""

    def __init__(self) -> None:
        self._calls = 0

    async def stream(self, **_kwargs: Any) -> AsyncIterator[Event]:  # type: ignore[override]
        self._calls += 1
        if self._calls == 1:
            raise RuntimeError("transient error")
            yield  # make it an async generator
        content = '{"can_finish": true, "reason": "ok on retry", "confidence": 0.9}'
        yield TextDelta(text=content)
        yield Done(final_message=Message(role="assistant", content=content))

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=False)

    async def cancel(self, session_id: str) -> None:
        pass


@pytest.mark.asyncio
async def test_judge_retries_on_exception_then_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("harness.core.verification.asyncio.sleep", _instant_sleep)
    adapter = FlakyThenPassLLMAdapter()
    verifier = LLMJudgeVerifier(adapter=adapter, model="m", max_retries=3)  # type: ignore[arg-type]
    session = _make_session()
    result = await verifier.verify(session=session, activity=[])
    assert result.can_finish is True
    assert adapter._calls == 2


@pytest.mark.asyncio
async def test_judge_exhausts_retries_and_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("harness.core.verification.asyncio.sleep", _instant_sleep)
    verifier = LLMJudgeVerifier(adapter=AlwaysRaiseLLMAdapter(), model="m", max_retries=3)  # type: ignore[arg-type]
    session = _make_session()
    result = await verifier.verify(session=session, activity=[])
    assert result.can_finish is False
    assert result.confidence == 0.0
    assert "attempt 3" in result.reason


@pytest.mark.asyncio
async def test_judge_retries_on_bad_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("harness.core.verification.asyncio.sleep", _instant_sleep)
    verifier = LLMJudgeVerifier(adapter=BadJSONLLMAdapter(), model="m", max_retries=2)  # type: ignore[arg-type]
    session = _make_session()
    result = await verifier.verify(session=session, activity=[])
    assert result.can_finish is False
    assert "non-JSON" in result.reason
    assert "attempt 2" in result.reason
