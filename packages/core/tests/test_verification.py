"""Tests for RuleVerifier, LLMJudgeVerifier, and Agent verifier wiring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.core import (
    ActivityEvent,
    ActivityStore,
    Agent,
    AutoApprove,
    ErrorEvent,
    FailoverPolicy,
    LLMJudgeVerifier,
    Message,
    RuleVerifier,
    RunRequest,
    Session,
    StallError,
    ToolRegistry,
    Verification,
    VerificationResult,
)
from harness.core import activity as activity_kinds
from harness.core.verification import _is_repetitive

from .conftest import MockAdapter, MockStorage, text_turn, tool_call_turn


def _activity(*, kind: str, **data: object) -> ActivityEvent:
    return ActivityEvent(session_id="s1", kind=kind, data=dict(data))


def _session(*, messages: list[Message] | None = None) -> Session:
    return Session(
        id="s1",
        provider="mock",
        model="m",
        cwd=Path.cwd(),
        messages=messages or [Message(role="user", content="hello")],
    )


# ---------------------------------------------------------------------------
# RuleVerifier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRuleVerifier:
    async def test_empty_activity_can_finish(self) -> None:
        result = await RuleVerifier().verify(session=_session(), activity=[])
        assert result.can_finish is True
        assert "no tools dispatched" in result.reason
        assert result.verifier_name == "rule"

    async def test_all_tools_succeeded(self) -> None:
        activity = [
            _activity(kind="tool_call.completed", name="read_file", is_error=False),
            _activity(kind="tool_call.completed", name="list_dir", is_error=False),
            _activity(kind="agent_run.started"),  # ignored
        ]
        result = await RuleVerifier().verify(session=_session(), activity=activity)
        assert result.can_finish is True
        assert "2 tool calls" in result.reason
        assert result.evidence_event_ids == []

    async def test_one_tool_failed_blocks_finish(self) -> None:
        ok = _activity(kind="tool_call.completed", name="read_file", is_error=False)
        bad = _activity(kind="tool_call.completed", name="shell", is_error=True)
        result = await RuleVerifier().verify(session=_session(), activity=[ok, bad])
        assert result.can_finish is False
        assert "shell" in result.reason
        assert result.evidence_event_ids == [bad.id]

    async def test_multiple_failing_tools_dedup_names(self) -> None:
        activity = [
            _activity(kind="tool_call.completed", name="shell", is_error=True),
            _activity(kind="tool_call.completed", name="shell", is_error=True),
            _activity(kind="tool_call.completed", name="write_file", is_error=True),
        ]
        result = await RuleVerifier().verify(session=_session(), activity=activity)
        assert result.can_finish is False
        # Names dedup'd and sorted.
        assert "shell" in result.reason
        assert "write_file" in result.reason
        assert len(result.evidence_event_ids) == 3

    async def test_repetitive_output_fails(self) -> None:
        repeated = "I do not know the answer to that question. " * 20
        session = _session(
            messages=[
                Message(role="user", content="hello"),
                Message(role="assistant", content=repeated),
            ]
        )
        result = await RuleVerifier().verify(session=session, activity=[])
        assert result.can_finish is False
        assert "loop" in result.reason.lower() or "repetit" in result.reason.lower()

    async def test_verbal_refusal_with_no_tools_fails(self) -> None:
        session = _session(
            messages=[
                Message(role="user", content="deep dive on the code"),
                Message(
                    role="assistant",
                    content=(
                        "I do not have direct access to the entire source code repository, "
                        "only the information I have been given."
                    ),
                ),
            ]
        )
        result = await RuleVerifier().verify(session=session, activity=[])
        assert result.can_finish is False
        assert "verbal refusal" in result.reason.lower() or "claimed" in result.reason.lower()

    async def test_verbal_refusal_phrase_with_tools_used_passes_through(self) -> None:
        # If tools were used alongside a refusal phrase, fall through to normal rules.
        session = _session(
            messages=[
                Message(role="user", content="deep dive on the code"),
                Message(
                    role="assistant",
                    content="I cannot access the file directly but I used read_file.",
                ),
            ]
        )
        activity = [_activity(kind="tool_call.completed", name="read_file", is_error=False)]
        result = await RuleVerifier().verify(session=session, activity=activity)
        # Falls through to rule 5: all tools succeeded → can_finish=True
        assert result.can_finish is True

    async def test_short_clean_no_tools_passes(self) -> None:
        # Simple text answer with no refusal patterns and no tools → passes with low confidence
        session = _session(
            messages=[
                Message(role="user", content="What is 2+2?"),
                Message(role="assistant", content="4"),
            ]
        )
        result = await RuleVerifier().verify(session=session, activity=[])
        assert result.can_finish is True
        assert result.confidence is not None and result.confidence <= 0.5


# ---------------------------------------------------------------------------
# _is_repetitive helper
# ---------------------------------------------------------------------------


class TestIsRepetitive:
    def test_highly_repetitive_text(self) -> None:
        block = "I do not have direct access to the source code. " * 20
        assert _is_repetitive(block) is True

    def test_unique_text(self) -> None:
        text = " ".join(str(i) for i in range(500))
        assert _is_repetitive(text) is False

    def test_short_text_not_flagged(self) -> None:
        # Below the window*threshold threshold
        assert _is_repetitive("hello world") is False

    def test_threshold_exactly_met(self) -> None:
        # A 200-char block repeated exactly 4 times → window=40 sees 20+ hits → True
        block = "x" * 200
        assert _is_repetitive(block * 4) is True

    def test_threshold_just_below(self) -> None:
        # "a"*40 repeated exactly 4 times (non-overlapping count = 4 < threshold 5)
        # followed by unique suffix so total length exceeds window*threshold guard
        block = "a" * 40
        unique_suffix = " ".join(str(i) for i in range(30))  # "0 1 2 ... 29" — no repeats
        assert _is_repetitive(block * 4 + unique_suffix) is False


# ---------------------------------------------------------------------------
# LLMJudgeVerifier
# ---------------------------------------------------------------------------


def _judge_response(payload: dict | str) -> list:
    """Helper: an adapter script that emits exactly one text message + Done."""
    from harness.core import Done, TextDelta

    body = payload if isinstance(payload, str) else json.dumps(payload)
    return [
        TextDelta(text=body),
        Done(final_message=Message(role="assistant", content=body)),
    ]


@pytest.mark.asyncio
class TestLLMJudgeVerifier:
    async def test_can_finish_true(self) -> None:
        adapter = MockAdapter(
            "judge",
            scripts=[
                _judge_response({"can_finish": True, "reason": "answer matches", "confidence": 0.9})
            ],
        )
        verifier = LLMJudgeVerifier(adapter=adapter, model="judge-m")
        result = await verifier.verify(
            session=_session(
                messages=[
                    Message(role="user", content="ping"),
                    Message(role="assistant", content="pong"),
                ]
            ),
            activity=[],
        )
        assert result.can_finish is True
        assert result.reason == "answer matches"
        assert result.confidence == pytest.approx(0.9)
        assert result.verifier_name == "llm"

    async def test_can_finish_false(self) -> None:
        adapter = MockAdapter(
            "judge",
            scripts=[
                _judge_response({"can_finish": False, "reason": "off-topic", "confidence": 0.7})
            ],
        )
        verifier = LLMJudgeVerifier(adapter=adapter, model="m")
        result = await verifier.verify(session=_session(), activity=[])
        assert result.can_finish is False
        assert result.confidence == pytest.approx(0.7)

    async def test_non_json_response_falls_back(self) -> None:
        adapter = MockAdapter(
            "judge",
            scripts=[_judge_response("I think yes but I'm not sure")],
        )
        verifier = LLMJudgeVerifier(adapter=adapter, model="m", max_retries=1)
        result = await verifier.verify(session=_session(), activity=[])
        assert result.can_finish is False
        assert "non-JSON" in result.reason
        assert result.confidence == 0.0

    async def test_json_fenced_response_parses(self) -> None:
        body = "```json\n" + json.dumps({"can_finish": True, "reason": "ok"}) + "\n```"
        adapter = MockAdapter("judge", scripts=[_judge_response(body)])
        verifier = LLMJudgeVerifier(adapter=adapter, model="m")
        result = await verifier.verify(session=_session(), activity=[])
        assert result.can_finish is True
        assert result.reason == "ok"

    async def test_adapter_failure_returns_can_finish_false(self) -> None:
        from harness.core import NetworkError

        adapter = MockAdapter("judge", error=NetworkError("judge offline"))
        verifier = LLMJudgeVerifier(adapter=adapter, model="m")
        result = await verifier.verify(session=_session(), activity=[])
        assert result.can_finish is False
        assert "judge call failed" in result.reason
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Agent verifier wiring
# ---------------------------------------------------------------------------


class _Sink(ActivityStore):
    def __init__(self) -> None:
        self.events: list[ActivityEvent] = []

    async def append_activity(self, event: ActivityEvent) -> None:
        self.events.append(event)

    async def list_activity(
        self,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        kinds: tuple[str, ...] | None = None,
        limit: int = 200,
    ) -> list[ActivityEvent]:
        items = list(self.events)
        if session_id is not None:
            items = [e for e in items if e.session_id == session_id]
        return items[:limit]


def _agent(*, adapter: MockAdapter, verifier, sink: ActivityStore) -> Agent:
    return Agent(
        adapters={"mock": adapter},  # type: ignore[arg-type]
        tools=ToolRegistry(),
        storage=MockStorage(),
        failover=FailoverPolicy(chain=["mock"], max_attempts=1),
        approval_handler=AutoApprove(),
        activity_store=sink,
        verifier=verifier,
        default_model="m",
    )


@pytest.mark.asyncio
class TestAgentWiring:
    async def test_emits_verification_event_after_done(self) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("answer")])
        sink = _Sink()
        agent = _agent(adapter=adapter, verifier=RuleVerifier(), sink=sink)

        events: list = []
        async for e in agent.run(RunRequest(prompt="hi", session_id="s1", model="m")):
            events.append(e)

        # Verification event appears in the stream.
        verifications = [e for e in events if isinstance(e, Verification)]
        assert len(verifications) == 1
        assert verifications[0].result.verifier_name == "rule"
        assert verifications[0].result.can_finish is True

        # Activity ledger has verification.completed too.
        kinds = [e.kind for e in sink.events]
        assert activity_kinds.VERIFICATION_COMPLETED in kinds

    async def test_no_verifier_means_no_verification_event(self) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("answer")])
        sink = _Sink()
        agent = _agent(adapter=adapter, verifier=None, sink=sink)

        events: list = []
        async for e in agent.run(RunRequest(prompt="hi", session_id="s1", model="m")):
            events.append(e)

        assert not [e for e in events if isinstance(e, Verification)]
        assert activity_kinds.VERIFICATION_COMPLETED not in [e.kind for e in sink.events]

    async def test_verifier_receives_real_activity(self) -> None:
        """RuleVerifier should see the tool_call.completed events from the run."""

        # The adapter scripts a single tool call that fails (unknown tool).
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="ghost", arguments={}),
                text_turn("done"),
            ],
        )
        sink = _Sink()
        agent = _agent(adapter=adapter, verifier=RuleVerifier(), sink=sink)

        events: list = []
        async for e in agent.run(RunRequest(prompt="hi", session_id="s1", model="m")):
            events.append(e)

        verdict = next(e for e in events if isinstance(e, Verification)).result
        assert verdict.can_finish is False
        assert "ghost" in verdict.reason

    async def test_verifier_exception_yields_failure_result(self) -> None:
        """A verifier that raises should not crash the run."""

        class _Broken:
            name = "broken"

            async def verify(self, *, session, activity):  # type: ignore[no-untyped-def]
                raise RuntimeError("boom")

        adapter = MockAdapter("mock", scripts=[text_turn("answer")])
        sink = _Sink()
        agent = _agent(adapter=adapter, verifier=_Broken(), sink=sink)

        events: list = []
        async for e in agent.run(RunRequest(prompt="hi", session_id="s1", model="m")):
            events.append(e)

        verdict = next(e for e in events if isinstance(e, Verification)).result
        assert verdict.can_finish is False
        assert "raised" in verdict.reason
        assert isinstance(verdict, VerificationResult)
