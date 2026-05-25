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
    BugfixCommentRewriteVerifier,
    FailoverPolicy,
    FileScopeVerifier,
    LLMJudgeVerifier,
    Message,
    NegativeConstraintVerifier,
    PromptSurfaceRevertVerifier,
    ResearchPromotionFlowVerifier,
    RuleVerifier,
    RunRequest,
    Session,
    ToolRegistry,
    Verification,
    VerificationResult,
    VerifyBeforeDoneVerifier,
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
        max_repair_attempts=0,
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


# ---------------------------------------------------------------------------
# VerifyBeforeDoneVerifier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestVerifyBeforeDoneVerifier:
    def _activity(self, *, kind: str, **data: object) -> ActivityEvent:
        return ActivityEvent(session_id="s1", kind=kind, data=dict(data))

    async def test_no_writes_passes(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [self._activity(kind="tool_call.completed", name="read_file", is_error=False)]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is True

    async def test_write_without_verify_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [self._activity(kind="tool_call.completed", name="write_file", is_error=False)]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "verify_work" in result.reason

    async def test_write_then_verify_passes(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(kind="tool_call.completed", name="verify_work", is_error=False),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is True

    async def test_write_then_failed_verify_passes_through(self) -> None:
        # Once verify_work was called (even if it failed), VerifyBeforeDoneVerifier
        # defers to downstream verifiers that have the real test output.
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="shell", is_error=False),
            self._activity(kind="tool_call.completed", name="verify_work", is_error=True),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is True
        assert "downstream" in result.reason.lower() or "deferring" in result.reason.lower()

    async def test_promotion_artifact_pr_flow_skips_generic_verify_requirement(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={
                    "path": ".harness/research/promotions/promo-test/promotion_candidate.json"
                },
            ),
            self._activity(
                kind="tool_call.completed",
                name="edit_file",
                is_error=False,
                arguments={"path": ".harness/research/promotions/promo-test/PR_BODY.md"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="shell",
                is_error=False,
                arguments={
                    "command": "harness research pr --candidate promo-test --base-branch main --push --open --draft"
                },
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is True
        assert "promotion artifacts" in result.reason.lower()

    async def test_empty_activity_passes(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        result = await verifier.verify(session=_session(), activity=[])
        assert result.can_finish is True


@pytest.mark.asyncio
class TestResearchPromotionFlowVerifier:
    def _activity(self, *, kind: str, **data: object) -> ActivityEvent:
        return ActivityEvent(session_id="s1", kind=kind, data=dict(data))

    async def test_allows_harness_native_promotion_flow(self) -> None:
        verifier = ResearchPromotionFlowVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={
                    "path": ".harness/research/promotions/promo-test/promotion_candidate.json"
                },
            ),
            self._activity(
                kind="tool_call.completed",
                name="shell",
                is_error=False,
                arguments={"command": "harness research create-candidate --title demo"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="shell",
                is_error=False,
                arguments={"command": "harness research promote --candidate promo-test"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="shell",
                is_error=False,
                arguments={"command": "harness research pr --candidate promo-test --push --open"},
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is True
        assert "harness promotion flow" in result.reason.lower()

    async def test_blocks_manual_pr_flow_without_harness_commands(self) -> None:
        verifier = ResearchPromotionFlowVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": ".harness/research/promotions/candidate.json"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="shell",
                is_error=False,
                arguments={"command": "git checkout -b research/openapi-promotion"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="shell",
                is_error=False,
                arguments={"command": "gh pr create --draft --base main --title demo"},
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "create-candidate" in result.reason
        assert "harness research pr" in result.reason


# ---------------------------------------------------------------------------
# PromptSurfaceRevertVerifier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPromptSurfaceRevertVerifier:
    async def test_blocks_disproven_prompt_surface_edit_left_in_diff(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "harness.core.verification_behavioral._git_diff_unified_zero",
            lambda _cwd: (
                "diff --git a/src/cache.py b/src/cache.py\n"
                "@@ -11 +11 @@\n"
                "-TIMEOUT_SECONDS = 5\n"
                "+TIMEOUT_SECONDS = 30\n"
                "@@ -46,0 +47,8 @@\n"
                "+        if key in self._in_flight:\n"
                "+            return await self._in_flight[key]\n"
                "+        task = asyncio.create_task(_fetch_and_cache())\n"
            ),
        )
        session = Session(
            id="s1",
            provider="mock",
            model="m",
            cwd=tmp_path,
            messages=[
                Message(
                    role="user",
                    content=(
                        "# Fix batch endpoint timeout\n\n"
                        "Increase the timeout from 5 seconds to 30 seconds.\n\n"
                        "File to change: `src/cache.py` (the `TIMEOUT_SECONDS` constant).\n"
                    ),
                )
            ],
        )
        activity = [
            _activity(
                kind="tool_call.completed",
                name="edit_file",
                is_error=False,
                arguments={"path": "src/cache.py", "new": "TIMEOUT_SECONDS = 30"},
                content_preview="TIMEOUT_SECONDS = 30",
            ),
            _activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=True,
                content_preview=(
                    "FAILED tests/test_cache.py::test_concurrent_requests_deduplicated "
                    "- AssertionError"
                ),
            ),
            _activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={
                    "path": "src/cache.py",
                    "content": (
                        "TIMEOUT_SECONDS = 30\n"
                        "self._in_flight = {}\n"
                        "return await self._in_flight[key]\n"
                    ),
                },
                content_preview="TIMEOUT_SECONDS = 30 ... _in_flight",
            ),
            _activity(kind="tool_call.completed", name="verify_work", is_error=False),
        ]

        result = await PromptSurfaceRevertVerifier().verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "Revert the prompt-surface edit" in result.reason
        assert result.verifier_name == "prompt_surface_revert"

    async def test_passes_when_current_diff_only_contains_root_cause_fix(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "harness.core.verification_behavioral._git_diff_unified_zero",
            lambda _cwd: (
                "diff --git a/src/cache.py b/src/cache.py\n"
                "@@ -46,0 +47,8 @@\n"
                "+        if key in self._in_flight:\n"
                "+            return await self._in_flight[key]\n"
                "+        task = asyncio.create_task(_fetch_and_cache())\n"
            ),
        )
        session = Session(
            id="s1",
            provider="mock",
            model="m",
            cwd=tmp_path,
            messages=[
                Message(
                    role="user",
                    content=(
                        "# Fix batch endpoint timeout\n\n"
                        "Increase the timeout from 5 seconds to 30 seconds.\n\n"
                        "File to change: `src/cache.py` (the `TIMEOUT_SECONDS` constant).\n"
                    ),
                )
            ],
        )
        activity = [
            _activity(
                kind="tool_call.completed",
                name="edit_file",
                is_error=False,
                arguments={"path": "src/cache.py", "new": "TIMEOUT_SECONDS = 30"},
                content_preview="TIMEOUT_SECONDS = 30",
            ),
            _activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=True,
                content_preview=(
                    "FAILED tests/test_cache.py::test_concurrent_requests_deduplicated "
                    "- AssertionError"
                ),
            ),
            _activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={
                    "path": "src/cache.py",
                    "content": ("self._in_flight = {}\nreturn await self._in_flight[key]\n"),
                },
                content_preview="_in_flight",
            ),
            _activity(kind="tool_call.completed", name="verify_work", is_error=False),
        ]

        result = await PromptSurfaceRevertVerifier().verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "no longer contains" in result.reason

    async def test_ignores_generic_symptom_tokens_from_function_call_prompt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "harness.core.verification_behavioral._git_diff_unified_zero",
            lambda _cwd: (
                "diff --git a/src/format.py b/src/format.py\n"
                "@@ -28,0 +29,2 @@\n"
                "+    if amount is None:\n"
                '+        return "—"\n'
            ),
        )
        session = Session(
            id="s1",
            provider="mock",
            model="m",
            cwd=tmp_path,
            messages=[
                Message(
                    role="user",
                    content=(
                        "# Fix null handling in format_price\n\n"
                        "`format_price(None)` raises a `TypeError`.\n"
                        'Return the string `"—"` when amount is None.\n'
                    ),
                )
            ],
        )
        activity = [
            _activity(
                kind="tool_call.completed",
                name="edit_file",
                is_error=False,
                arguments={"path": "src/format.py", "new": 'if amount is None:\n    return "—"'},
                content_preview='if amount is None: return "—"',
            ),
            _activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=True,
                content_preview="FAILED tests/test_format.py::test_format_price_none - TypeError",
            ),
            _activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={
                    "path": "src/format.py",
                    "content": 'if amount is None:\n    return "—"',
                },
                content_preview='if amount is None: return "—"',
            ),
            _activity(kind="tool_call.completed", name="verify_work", is_error=False),
        ]

        result = await PromptSurfaceRevertVerifier().verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "no longer contains" in result.reason or "protect" in result.reason

    async def test_allows_legitimate_timeout_mentions_after_constant_is_reverted(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "harness.core.verification_behavioral._git_diff_unified_zero",
            lambda _cwd: (
                "diff --git a/src/cache.py b/src/cache.py\n"
                "@@ -20,14 +20,14 @@ class SimpleCache:\n"
                "-    Note: concurrent requests for the same key each trigger their own\n"
                "+    Note: concurrent requests for the same key are deduplicated in-flight.\n"
                "@@ -46,12 +46,20 @@ class SimpleCache:\n"
                "+        if key in self._in_flight:\n"
                "+            return await self._in_flight[key]\n"
                "+        async def _fetch_and_cache() -> Any:\n"
                "+            try:\n"
                "+                value = await asyncio.wait_for(fetch(key), timeout=fetch_timeout)\n"
                "+                self._store[key] = value\n"
                "+                return value\n"
                "+            finally:\n"
                "+                self._in_flight.pop(key, None)\n"
                "+        task = asyncio.create_task(_fetch_and_cache())\n"
                "+        self._in_flight[key] = task\n"
                "+        return await task\n"
            ),
        )
        session = Session(
            id="s1",
            provider="mock",
            model="m",
            cwd=tmp_path,
            messages=[
                Message(
                    role="user",
                    content=(
                        "# Fix batch endpoint timeout\n\n"
                        "Increase the timeout from 5 seconds to 30 seconds.\n\n"
                        "File to change: `src/cache.py` (the `TIMEOUT_SECONDS` constant).\n"
                    ),
                )
            ],
        )
        activity = [
            _activity(
                kind="tool_call.completed",
                name="edit_file",
                is_error=False,
                arguments={"path": "src/cache.py", "new": "TIMEOUT_SECONDS = 30"},
                content_preview="TIMEOUT_SECONDS = 30",
            ),
            _activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=True,
                content_preview=(
                    "FAILED tests/test_cache.py::test_concurrent_requests_deduplicated "
                    "- AssertionError"
                ),
            ),
            _activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={
                    "path": "src/cache.py",
                    "content": (
                        "TIMEOUT_SECONDS = 5\n"
                        "if key in self._in_flight:\n"
                        "    return await self._in_flight[key]\n"
                        "value = await asyncio.wait_for(fetch(key), timeout=fetch_timeout)\n"
                    ),
                },
                content_preview="TIMEOUT_SECONDS = 5 ... timeout=fetch_timeout",
            ),
            _activity(kind="tool_call.completed", name="verify_work", is_error=False),
        ]

        result = await PromptSurfaceRevertVerifier().verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "no longer contains" in result.reason


@pytest.mark.asyncio
class TestNegativeConstraintVerifier:
    async def test_blocks_comment_style_cleanup_when_prompt_forbids_formatting(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "harness.core.verification_behavioral._git_diff_unified_zero",
            lambda _cwd: (
                "diff --git a/tests/test_calc.py b/tests/test_calc.py\n"
                "@@ -10,0 +11,1 @@\n"
                "+# -- power -----------------------------------------------------------\n"
                "@@ -20,0 +21,3 @@\n"
                "+def test_power():\n"
                "+    assert 2 ** 3 == 8\n"
                "+\n"
            ),
        )
        session = Session(
            id="s1",
            provider="mock",
            model="m",
            cwd=tmp_path,
            messages=[
                Message(
                    role="user",
                    content=(
                        "Add a feature.\n\n"
                        "Do not fix pre-existing typos, inconsistent formatting, or unused imports."
                    ),
                )
            ],
        )
        activity = [_activity(kind="tool_call.completed", name="verify_work", is_error=False)]

        result = await NegativeConstraintVerifier().verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "comment-style changes" in result.reason

    async def test_passes_when_only_requested_code_changes_remain(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "harness.core.verification_behavioral._git_diff_unified_zero",
            lambda _cwd: (
                "diff --git a/src/feature.py b/src/feature.py\n"
                "@@ -5,0 +6,2 @@\n"
                "+def power(base, exponent):\n"
                "+    return base ** exponent\n"
            ),
        )
        session = Session(
            id="s1",
            provider="mock",
            model="m",
            cwd=tmp_path,
            messages=[
                Message(
                    role="user",
                    content=(
                        "Add a feature.\n\n"
                        "Do not fix pre-existing typos, inconsistent formatting, or unused imports."
                    ),
                )
            ],
        )
        activity = [_activity(kind="tool_call.completed", name="verify_work", is_error=False)]

        result = await NegativeConstraintVerifier().verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "no explicit negative-constraint violations" in result.reason

    async def test_comment_banner_feedback_is_actionable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "harness.core.verification_behavioral._git_diff_unified_zero",
            lambda _cwd: (
                "diff --git a/tests/test_calc.py b/tests/test_calc.py\n"
                "@@ -10,0 +11,1 @@\n"
                "+# -- power -----------------------------------------------------------\n"
                "@@ -20,0 +21,3 @@\n"
                "+def test_power():\n"
                "+    assert 2 ** 3 == 8\n"
                "+\n"
            ),
        )
        session = Session(
            id="s1",
            provider="mock",
            model="m",
            cwd=tmp_path,
            messages=[
                Message(
                    role="user",
                    content=(
                        "Add a feature.\n\n"
                        "Do not fix pre-existing typos, inconsistent formatting, or unused imports."
                    ),
                )
            ],
        )
        activity = [_activity(kind="tool_call.completed", name="verify_work", is_error=False)]

        result = await NegativeConstraintVerifier().verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "delete only those new `# ...` lines" in result.reason


@pytest.mark.asyncio
class TestFileScopeVerifier:
    async def test_ignores_test_only_paths_for_bugfix_prompt_with_function_call(
        self, tmp_path: Path
    ) -> None:
        session = Session(
            id="s1",
            provider="mock",
            model="m",
            cwd=tmp_path,
            messages=[
                Message(
                    role="user",
                    content=(
                        "# Fix null handling in render_amount\n\n"
                        "`render_amount(None)` raises a `TypeError`.\n\n"
                        "Also add one regression test in `tests/test_format.py`.\n"
                    ),
                )
            ],
        )
        activity = [
            _activity(
                kind="tool_call.completed",
                name="edit_file",
                is_error=False,
                arguments={"path": "src/format.py", "new": 'if amount is None:\n    return "—"'},
                content_preview='if amount is None: return "—"',
            ),
            _activity(
                kind="tool_call.completed",
                name="edit_file",
                is_error=False,
                arguments={
                    "path": "tests/test_format.py",
                    "new": 'def test_render_amount_none():\n    assert render_amount(None) == "—"',
                },
                content_preview="def test_render_amount_none(): ...",
            ),
        ]

        result = await FileScopeVerifier().verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "no file-scope constraint" in result.reason

    async def test_enforces_explicit_named_source_file_scope(self, tmp_path: Path) -> None:
        session = Session(
            id="s1",
            provider="mock",
            model="m",
            cwd=tmp_path,
            messages=[
                Message(
                    role="user",
                    content=(
                        "# Fix batch endpoint timeout\n\n"
                        "Increase the timeout from 5 seconds to 30 seconds.\n\n"
                        "File to change: `src/cache.py`.\n"
                    ),
                )
            ],
        )
        activity = [
            _activity(
                kind="tool_call.completed",
                name="edit_file",
                is_error=False,
                arguments={"path": "src/cache.py", "new": "TIMEOUT_SECONDS = 30"},
                content_preview="TIMEOUT_SECONDS = 30",
            ),
            _activity(
                kind="tool_call.completed",
                name="edit_file",
                is_error=False,
                arguments={"path": "tests/test_cache.py", "new": "assert True"},
                content_preview="assert True",
            ),
        ]

        result = await FileScopeVerifier().verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "src/cache.py" in result.reason


@pytest.mark.asyncio
class TestBugfixCommentRewriteVerifier:
    async def test_blocks_new_source_comment_on_bugfix_prompt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "harness.core.verification_behavioral._git_diff_unified_zero",
            lambda _cwd: (
                "diff --git a/src/db.py b/src/db.py\n"
                "@@ -14,3 +14,2 @@\n"
                "+    # We no longer strip hyphens as they are a valid part of IDs.\n"
            ),
        )
        session = Session(
            id="s1",
            provider="mock",
            model="m",
            cwd=tmp_path,
            messages=[
                Message(
                    role="user",
                    content="# Fix hyphenated user ID lookup\n\n`get_user('abc-def')` returns None.\n",
                )
            ],
        )

        result = await BugfixCommentRewriteVerifier().verify(session=session, activity=[])

        assert result.can_finish is False
        assert "source comment lines" in result.reason

    async def test_passes_when_bugfix_adds_only_code(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "harness.core.verification_behavioral._git_diff_unified_zero",
            lambda _cwd: (
                "diff --git a/src/db.py b/src/db.py\n"
                "@@ -14,3 +14,1 @@\n"
                "+    return _USERS.get(user_id)\n"
            ),
        )
        session = Session(
            id="s1",
            provider="mock",
            model="m",
            cwd=tmp_path,
            messages=[
                Message(
                    role="user",
                    content="# Fix hyphenated user ID lookup\n\n`get_user('abc-def')` returns None.\n",
                )
            ],
        )

        result = await BugfixCommentRewriteVerifier().verify(session=session, activity=[])

        assert result.can_finish is True
        assert "no new source comment lines" in result.reason
