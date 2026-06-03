"""Tests for RuleVerifier, LLMJudgeVerifier, and Agent verifier wiring."""

from __future__ import annotations

# pyright: reportArgumentType=false
import asyncio
import json
import subprocess
from pathlib import Path

import pytest

from harness.core import (
    ActivityEvent,
    ActivityStore,
    Agent,
    AutoApprove,
    BugfixCommentRewriteVerifier,
    ChainedVerifier,
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
    ToolCall,
    ToolRegistry,
    Verification,
    VerificationResult,
    VerifierRouter,
    VerifyBeforeDoneVerifier,
)
from harness.core import activity as activity_kinds
from harness.core.tools_verification import VerifyWorkTool
from harness.core.verification import _is_repetitive
from harness.core.verification_structural import tool_event_changes_state

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


def test_failed_shell_event_with_unchanged_workspace_metadata_does_not_change_state() -> None:
    event = _activity(
        kind=activity_kinds.TOOL_CALL_COMPLETED,
        name="shell",
        is_error=True,
        arguments={"command": "go build -o abs main.go"},
        metadata={
            "exit_code": 1,
            "workspace_changed": False,
            "workspace_fingerprint_changed": False,
        },
    )

    assert tool_event_changes_state(event, frozenset({"shell"})) is False


def test_failed_shell_event_with_changed_workspace_metadata_changes_state() -> None:
    event = _activity(
        kind=activity_kinds.TOOL_CALL_COMPLETED,
        name="shell",
        is_error=True,
        arguments={"command": "go build -o abs main.go"},
        metadata={
            "exit_code": 1,
            "workspace_changed": True,
            "workspace_fingerprint_changed": True,
        },
    )

    assert tool_event_changes_state(event, frozenset({"shell"})) is True


class _StaticVerifier:
    def __init__(self, result: VerificationResult) -> None:
        self._result = result

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        return self._result


@pytest.mark.asyncio
async def test_chained_verifier_can_aggregate_independent_failures() -> None:
    verifier = ChainedVerifier(
        _StaticVerifier(
            VerificationResult(
                can_finish=False,
                reason="remove unrelated comment",
                confidence=0.8,
                verifier_name="comments",
            )
        ),
        _StaticVerifier(
            VerificationResult(
                can_finish=False,
                reason="revert prompt-surface edit",
                confidence=0.9,
                verifier_name="surface",
            )
        ),
        fail_fast=False,
    )

    result = await verifier.verify(session=_session(), activity=[])

    assert result.can_finish is False
    assert "Multiple independent verification checks failed" in result.reason
    assert "remove unrelated comment" in result.reason
    assert "revert prompt-surface edit" in result.reason


@pytest.mark.asyncio
async def test_chained_verifier_handles_missing_failure_confidence() -> None:
    verifier = ChainedVerifier(
        _StaticVerifier(
            VerificationResult(
                can_finish=False,
                reason="first issue",
                confidence=None,
                verifier_name="first",
            )
        ),
        _StaticVerifier(
            VerificationResult(
                can_finish=False,
                reason="second issue",
                confidence=0.7,
                verifier_name="second",
            )
        ),
        fail_fast=False,
    )

    result = await verifier.verify(session=_session(), activity=[])

    assert result.can_finish is False
    assert result.confidence == 0.7
    assert "first issue" in result.reason
    assert "second issue" in result.reason


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

    async def test_empty_assistant_final_answer_fails(self) -> None:
        session = _session(
            messages=[
                Message(role="user", content="Good morning"),
                Message(role="assistant", content=None),
            ]
        )

        result = await RuleVerifier().verify(session=session, activity=[])

        assert result.can_finish is False
        assert "final answer is empty" in result.reason

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


@pytest.mark.asyncio
async def test_router_blocks_ungrounded_read_only_numeric_claim() -> None:
    session = _session(
        messages=[
            Message(role="user", content="What is the weather in Tokyo right now?"),
            Message(role="assistant", content="The current temperature in Tokyo is 23°C."),
        ]
    )
    activity = [
        _activity(
            kind="tool_call.completed",
            name="web_search",
            is_error=False,
            content_preview="Weather in Tokyo: temp_c 22.0 condition Clear",
            metadata={},
        )
    ]
    verifier = VerifierRouter(
        rule=RuleVerifier(),
        llm=_StaticVerifier(
            VerificationResult(
                can_finish=True,
                reason="unused",
                confidence=1.0,
                verifier_name="unused",
            )
        ),
    )

    result = await verifier.verify(session=session, activity=activity)

    assert result.can_finish is False
    assert result.verifier_name == "router"
    assert "23" in result.reason


@pytest.mark.asyncio
async def test_router_allows_read_only_numeric_claim_grounded_in_metadata() -> None:
    session = _session(
        messages=[
            Message(role="user", content="What is the weather in Tokyo right now?"),
            Message(role="assistant", content="The current temperature in Tokyo is 22°C."),
        ]
    )
    activity = [
        _activity(
            kind="tool_call.completed",
            name="web_search",
            is_error=False,
            content_preview="Weather result preview was truncated before the numeric value",
            metadata={"results": [{"content": "{'current': {'temp_c': 22.0}}"}]},
        )
    ]
    verifier = VerifierRouter(
        rule=RuleVerifier(),
        llm=_StaticVerifier(
            VerificationResult(
                can_finish=True,
                reason="unused",
                confidence=1.0,
                verifier_name="unused",
            )
        ),
    )

    result = await verifier.verify(session=session, activity=activity)

    assert result.can_finish is True
    assert "all claims grounded" in result.reason


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

    async def test_uses_latest_user_message_for_multiturn_sessions(self) -> None:
        adapter = MockAdapter(
            "judge",
            scripts=[
                _judge_response({"can_finish": True, "reason": "latest goal ok", "confidence": 0.9})
            ],
        )
        verifier = LLMJudgeVerifier(adapter=adapter, model="m")

        await verifier.verify(
            session=_session(
                messages=[
                    Message(role="user", content="Create result.txt containing exactly old."),
                    Message(role="assistant", content="done"),
                    Message(role="user", content="Now append the new sentence."),
                    Message(role="assistant", content="appended"),
                ]
            ),
            activity=[],
        )

        prompt = adapter.calls[0]["messages"][1].content
        assert "Now append the new sentence." in prompt
        assert "Create result.txt containing exactly old." not in prompt

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
        if limit <= 0:
            return []
        return items[-limit:]


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
async def test_verify_work_preserves_raw_stdout_in_metadata(tmp_path: Path) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)
    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "printf 'harness-ok\\n'"},
        )
    )

    assert result.is_error is False
    assert result.content == "PASSED\n\nharness-ok"
    assert result.metadata is not None
    assert result.metadata["stdout"] == "harness-ok\n"
    assert result.metadata["exit_code"] == 0


@pytest.mark.asyncio
async def test_verify_work_can_use_configured_default_command(tmp_path: Path) -> None:
    tool = VerifyWorkTool(cwd=tmp_path, default_command="printf 'default-ok\\n'")

    result = await tool(ToolCall(id="v1", name="verify_work", arguments={}))

    assert result.is_error is False
    assert result.content == "PASSED\n\ndefault-ok"
    assert result.metadata is not None
    assert result.metadata["used_default_command"] is True
    assert "required" not in tool.parameters_schema
    assert "default verifier" in tool.description
    assert "clean command environment" in tool.description
    assert "containerized command" in tool.description


@pytest.mark.asyncio
async def test_verify_work_rejects_noop_commands(tmp_path: Path) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    direct = await tool(ToolCall(id="v1", name="verify_work", arguments={"command": "true"}))
    chained = await tool(
        ToolCall(id="v2", name="verify_work", arguments={"command": "cd . && true"})
    )
    container_wrapped = await tool(
        ToolCall(
            id="v3",
            name="verify_work",
            arguments={
                "command": (
                    "docker run --rm -v \"$PWD\":/app -w /app image bash -lc 'cd /app && true'"
                )
            },
        )
    )

    assert direct.is_error is True
    assert chained.is_error is True
    assert container_wrapped.is_error is True
    assert direct.metadata is not None
    assert chained.metadata is not None
    assert container_wrapped.metadata is not None
    assert direct.metadata["reason"] == "noop_verification_command"
    assert chained.metadata["reason"] == "noop_verification_command"
    assert container_wrapped.metadata["reason"] == "noop_verification_command"
    assert "meaningful check" in direct.content


@pytest.mark.asyncio
async def test_verify_work_runs_with_clean_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HARNESS_VERIFY_LEAK_TEST", "present")
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": 'test -z "$HARNESS_VERIFY_LEAK_TEST"'},
        )
    )

    assert result.is_error is False
    assert result.metadata is not None
    assert result.metadata["clean_env"] is True


@pytest.mark.asyncio
async def test_verify_work_fails_pipeline_when_left_side_fails(tmp_path: Path) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "false | cat"},
        )
    )

    assert result.is_error is True
    assert result.metadata is not None
    assert result.metadata["pipefail"] is True
    assert result.metadata["exit_code"] != 0


@pytest.mark.asyncio
async def test_verify_work_fails_multiline_command_when_earlier_line_fails(tmp_path: Path) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "false\necho masked"},
        )
    )

    assert result.is_error is True
    assert result.metadata is not None
    assert result.metadata["errexit"] is True
    assert result.metadata["exit_code"] != 0


@pytest.mark.asyncio
async def test_verify_work_rejects_masked_failure_fallback(tmp_path: Path) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "test -f missing.txt || echo 'Verification failed'"},
        )
    )

    assert result.is_error is True
    assert "failure branch can exit successfully" in result.content
    assert result.metadata is not None
    assert result.metadata["invalid_verification_command"] is True


@pytest.mark.asyncio
async def test_verify_work_rejects_nested_shell_status_echo_masking(tmp_path: Path) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "bash -lc 'false; echo exit:$?'"},
        )
    )

    assert result.is_error is True
    assert "failure branch can exit successfully" in result.content
    assert result.metadata is not None
    assert result.metadata["reason"] == "masked_failure_exit_status"


@pytest.mark.asyncio
async def test_verify_work_rejects_container_nested_shell_status_echo_masking(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": ("docker run --rm example/image sh -c 'false; echo status:$?'")},
        )
    )

    assert result.is_error is True
    assert "failure branch can exit successfully" in result.content
    assert result.metadata is not None
    assert result.metadata["reason"] == "masked_failure_exit_status"


@pytest.mark.asyncio
async def test_verify_work_allows_quoted_or_text(tmp_path: Path) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "printf 'a||b\\n'"},
        )
    )

    assert result.is_error is False
    assert result.content == "PASSED\n\na||b"
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0


@pytest.mark.asyncio
async def test_verify_work_allows_quoted_if_text(tmp_path: Path) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "printf 'if ! test -f missing.txt; then echo ok; fi\\n'"},
        )
    )

    assert result.is_error is False
    assert "if ! test -f missing.txt; then echo ok; fi" in result.content
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0


@pytest.mark.asyncio
async def test_verify_work_allows_commented_if_text(tmp_path: Path) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "printf ok # if ! test -f missing.txt; then echo ok; fi"},
        )
    )

    assert result.is_error is False
    assert result.content == "PASSED\n\nok"
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0


@pytest.mark.asyncio
async def test_verify_work_rejects_exit_zero_failure_fallback(tmp_path: Path) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "test -f missing.txt >/dev/null 2>&1 || exit 0"},
        )
    )

    assert result.is_error is True
    assert "failure branch can exit successfully" in result.content
    assert result.metadata is not None
    assert result.metadata["reason"] == "masked_failure_exit_status"


@pytest.mark.asyncio
async def test_verify_work_rejects_exit_before_later_test_command(tmp_path: Path) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "exit 0; pytest tests"},
        )
    )

    assert result.is_error is True
    assert "exits before a later check can run" in result.content
    assert result.metadata is not None
    assert result.metadata["reason"] == "unreachable_verification_command"


@pytest.mark.asyncio
async def test_verify_work_rejects_compound_exit_zero_failure_fallback(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "test -f missing.txt || { echo 'not really checked'; exit 0; }"},
        )
    )

    assert result.is_error is True
    assert "failure branch can exit successfully" in result.content
    assert result.metadata is not None
    assert result.metadata["reason"] == "masked_failure_exit_status"


@pytest.mark.asyncio
async def test_verify_work_rejects_if_not_successful_failure_branch(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "if ! test -f missing.txt; then echo ok; fi"},
        )
    )

    assert result.is_error is True
    assert "failure branch can exit successfully" in result.content
    assert result.metadata is not None
    assert result.metadata["reason"] == "masked_failure_exit_status"


@pytest.mark.asyncio
async def test_verify_work_rejects_positive_if_without_else(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "if test -f missing.txt; then echo ok; fi"},
        )
    )

    assert result.is_error is True
    assert "failure branch can exit successfully" in result.content
    assert result.metadata is not None
    assert result.metadata["reason"] == "masked_failure_exit_status"


@pytest.mark.asyncio
async def test_verify_work_rejects_multiline_positive_if_without_else(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "if test -f missing.txt\nthen echo ok\nfi"},
        )
    )

    assert result.is_error is True
    assert "failure branch can exit successfully" in result.content
    assert result.metadata is not None
    assert result.metadata["reason"] == "masked_failure_exit_status"


@pytest.mark.asyncio
async def test_verify_work_rejects_while_loop_condition_masking(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "while test -f missing.txt; do echo ok; done"},
        )
    )

    assert result.is_error is True
    assert "failure branch can exit successfully" in result.content
    assert result.metadata is not None
    assert result.metadata["reason"] == "masked_failure_exit_status"


@pytest.mark.asyncio
async def test_verify_work_rejects_until_loop_condition_masking(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "until test -f missing.txt; do echo ok; break; done"},
        )
    )

    assert result.is_error is True
    assert "failure branch can exit successfully" in result.content
    assert result.metadata is not None
    assert result.metadata["reason"] == "masked_failure_exit_status"


@pytest.mark.asyncio
async def test_verify_work_rejects_empty_for_loop_masking(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "for item in ; do echo ok; done"},
        )
    )

    assert result.is_error is True
    assert "failure branch can exit successfully" in result.content
    assert result.metadata is not None
    assert result.metadata["reason"] == "masked_failure_exit_status"


@pytest.mark.asyncio
async def test_verify_work_rejects_dynamic_for_loop_masking(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "ITEMS=''; for item in $ITEMS; do echo ok; done"},
        )
    )

    assert result.is_error is True
    assert "failure branch can exit successfully" in result.content
    assert result.metadata is not None
    assert result.metadata["reason"] == "masked_failure_exit_status"


@pytest.mark.asyncio
async def test_verify_work_allows_static_for_loop(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": 'for item in present; do test "$item" = present; done'},
        )
    )

    assert result.is_error is False
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0


@pytest.mark.asyncio
async def test_verify_work_rejects_case_without_nonzero_default(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "case missing in present) echo ok;; esac"},
        )
    )

    assert result.is_error is True
    assert "failure branch can exit successfully" in result.content
    assert result.metadata is not None
    assert result.metadata["reason"] == "masked_failure_exit_status"


@pytest.mark.asyncio
async def test_verify_work_allows_case_with_nonzero_default(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "case present in present) echo ok;; *) exit 1;; esac"},
        )
    )

    assert result.is_error is False
    assert result.content == "PASSED\n\nok"
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0


@pytest.mark.asyncio
async def test_verify_work_rejects_multiline_if_not_successful_failure_branch(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "if ! test -f missing.txt\nthen echo ok\nfi"},
        )
    )

    assert result.is_error is True
    assert "failure branch can exit successfully" in result.content
    assert result.metadata is not None
    assert result.metadata["reason"] == "masked_failure_exit_status"


@pytest.mark.asyncio
async def test_verify_work_allows_if_not_with_nonzero_failure_branch(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "if ! test -f missing.txt; then exit 1; fi"},
        )
    )

    assert result.is_error is True
    assert result.content.startswith("FAILED (exit")
    assert result.metadata is not None
    assert result.metadata.get("reason") != "masked_failure_exit_status"


@pytest.mark.asyncio
async def test_verify_work_rejects_successful_command_that_reports_failure(tmp_path: Path) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "printf 'Verification failed\\n'"},
        )
    )

    assert result.is_error is True
    assert result.content == "FAILED (output reports failure)\n\nVerification failed"
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is True


@pytest.mark.asyncio
async def test_verify_work_rejects_successful_command_that_reports_nonzero_status(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "printf 'make_rc:2\\n'"},
        )
    )

    assert result.is_error is True
    assert result.content == "FAILED (output reports failure)\n\nmake_rc:2"
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is True


@pytest.mark.asyncio
async def test_verify_work_rejects_successful_command_that_prints_traceback(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={
                "command": (
                    "printf 'Traceback (most recent call last):\\nAssertionError: wrong result\\n'"
                )
            },
        )
    )

    assert result.is_error is True
    assert result.content.startswith("FAILED (output reports failure)")
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is True


@pytest.mark.asyncio
async def test_verify_work_rejects_successful_command_that_prints_error(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "printf 'ERROR: expected harness-ok got wrong\\n'"},
        )
    )

    assert result.is_error is True
    assert result.content.startswith("FAILED (output reports failure)")
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is True


@pytest.mark.asyncio
async def test_verify_work_allows_successful_zero_error_summary(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "printf '3 passed, 0 errors\\n'"},
        )
    )

    assert result.is_error is False
    assert result.content == "PASSED\n\n3 passed, 0 errors"
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is False


@pytest.mark.asyncio
async def test_verify_work_rejects_command_that_changes_workspace(tmp_path: Path) -> None:
    await asyncio.to_thread(
        subprocess.run,
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "printf ok && mkdir -p src && touch src/after_verify.py"},
        )
    )

    assert result.is_error is True
    assert result.content.startswith("FAILED (verification changed workspace)")
    assert result.metadata is not None
    assert result.metadata["workspace_changed"] is True
    assert (tmp_path / "src" / "after_verify.py").is_file()


@pytest.mark.asyncio
async def test_verify_work_ignores_generated_test_artifacts_for_workspace_change(
    tmp_path: Path,
) -> None:
    await asyncio.to_thread(
        subprocess.run,
        ["git", "init"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={
                "command": "mkdir -p .pytest_cache src/__pycache__ && "
                "touch .pytest_cache/v src/__pycache__/app.pyc && printf ok"
            },
        )
    )

    assert result.is_error is False
    assert result.content == "PASSED\n\nok"
    assert result.metadata is not None
    assert result.metadata["workspace_changed"] is False


@pytest.mark.asyncio
async def test_verify_work_rejects_successful_command_that_runs_no_tests(tmp_path: Path) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={
                "command": (
                    "printf '============================= test session starts "
                    "=============================\\ncollected 0 items\\n\\n"
                    "============================ no tests ran in 0.01s "
                    "============================\\n'"
                )
            },
        )
    )

    assert result.is_error is True
    assert result.content.startswith("FAILED (output reports failure)")
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is True


@pytest.mark.asyncio
async def test_verify_work_rejects_successful_command_with_only_skipped_tests(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "printf '1 skipped, 0 passed, 0 failed\\n'"},
        )
    )

    assert result.is_error is True
    assert result.content.startswith("FAILED (output reports failure)")
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is True


@pytest.mark.asyncio
async def test_verify_work_rejects_successful_command_with_all_skipped_tests(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "printf '3 skipped in 0.02s\\n'"},
        )
    )

    assert result.is_error is True
    assert result.content.startswith("FAILED (output reports failure)")
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is True


@pytest.mark.asyncio
async def test_verify_work_allows_successful_passed_with_skipped_summary(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "printf '2 passed, 1 skipped, 0 failed\\n'"},
        )
    )

    assert result.is_error is False
    assert result.content == "PASSED\n\n2 passed, 1 skipped, 0 failed"
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is False


@pytest.mark.asyncio
async def test_verify_work_rejects_successful_command_with_all_deselected_tests(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "printf '3 deselected in 0.02s\\n'"},
        )
    )

    assert result.is_error is True
    assert result.content.startswith("FAILED (output reports failure)")
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is True


@pytest.mark.asyncio
async def test_verify_work_allows_successful_passed_with_deselected_summary(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "printf '2 passed, 1 deselected, 0 failed\\n'"},
        )
    )

    assert result.is_error is False
    assert result.content == "PASSED\n\n2 passed, 1 deselected, 0 failed"
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        "testing: warning: no tests to run\nPASS\nok  example.com/pkg 0.003s",
        "?   \texample.com/pkg\t[no test files]",
        "running 0 tests\n\ntest result: ok. 0 passed; 0 failed; 0 ignored",
        "0 passed, 0 failed",
        "0 passing (1ms)",
        "Test Suites: 0 passed, 0 total\nTests: 0 total",
        "All tests skipped",
    ],
)
async def test_verify_work_rejects_successful_common_no_test_outputs(
    tmp_path: Path,
    output: str,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "cat <<'EOF'\n" + output + "\nEOF"},
        )
    )

    assert result.is_error is True
    assert result.content.startswith("FAILED (output reports failure)")
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is True


@pytest.mark.asyncio
async def test_verify_work_allows_go_package_sweep_with_some_no_test_packages(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)
    output = (
        "ok  github.com/mattn/anko 0.123s\n"
        "?   \tgithub.com/mattn/anko/ast\t[no test files]\n"
        "ok  github.com/mattn/anko/core 0.456s\n"
        "?   \tgithub.com/mattn/anko/parser\t[no test files]\n"
    )

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "cat <<'EOF'\n" + output + "\nEOF"},
        )
    )

    assert result.is_error is False
    assert result.content.startswith("PASSED")
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is False


@pytest.mark.asyncio
async def test_verify_work_allows_go_package_sweep_with_some_no_matching_tests(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)
    output = (
        "ok  github.com/go-git/go-git/v6 0.012s\n"
        "ok  github.com/go-git/go-git/v6/backend/http 0.003s [no tests to run]\n"
        "?   \tgithub.com/go-git/go-git/v6/internal/pathutil\t[no test files]\n"
        "ok  github.com/go-git/go-git/v6/config 0.004s [no tests to run]\n"
    )

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "cat <<'EOF'\n" + output + "\nEOF"},
        )
    )

    assert result.is_error is False
    assert result.content.startswith("PASSED")
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is False


@pytest.mark.asyncio
async def test_verify_work_rejects_go_package_sweep_with_only_no_matching_tests(
    tmp_path: Path,
) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)
    output = (
        "ok  github.com/go-git/go-git/v6 0.012s [no tests to run]\n"
        "ok  github.com/go-git/go-git/v6/config 0.004s [no tests to run]\n"
        "?   \tgithub.com/go-git/go-git/v6/internal/pathutil\t[no test files]\n"
    )

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "cat <<'EOF'\n" + output + "\nEOF"},
        )
    )

    assert result.is_error is True
    assert result.content.startswith("FAILED (output reports failure)")
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is True


@pytest.mark.asyncio
async def test_verify_work_rejects_successful_go_output_with_failures(tmp_path: Path) -> None:
    tool = VerifyWorkTool(cwd=tmp_path)
    output = (
        "--- FAIL: TestRunInteractive (0.00s)\n"
        "    anko_test.go:109: OpenFile error\n"
        "FAIL\n"
        "FAIL\tgithub.com/mattn/anko\t0.008s\n"
        "ok  \tgithub.com/mattn/anko/vm\t0.152s\n"
    )

    result = await tool(
        ToolCall(
            id="v1",
            name="verify_work",
            arguments={"command": "cat <<'EOF'\n" + output + "\nEOF"},
        )
    )

    assert result.is_error is True
    assert result.content.startswith("FAILED (output reports failure)")
    assert result.metadata is not None
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is True


@pytest.mark.asyncio
class TestVerifyBeforeDoneVerifier:
    def _activity(self, *, kind: str, **data: object) -> ActivityEvent:
        return ActivityEvent(session_id="s1", kind=kind, data=dict(data))

    async def test_no_writes_passes(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [self._activity(kind="tool_call.completed", name="read_file", is_error=False)]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is True

    async def test_exact_file_request_without_tools_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[Message(role="user", content="Create result.txt containing exactly truth.")]
        )

        result = await verifier.verify(session=session, activity=[])

        assert result.can_finish is False
        assert "exact file or exact program output" in result.reason

    async def test_exact_stdout_request_without_tools_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[Message(role="user", content="Create hello.py that prints exactly truth.")]
        )

        result = await verifier.verify(session=session, activity=[])

        assert result.can_finish is False
        assert "exact file or exact program output" in result.reason

    async def test_exact_request_without_state_change_accepts_verify_evidence(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[Message(role="user", content="Create hello.py that prints exactly truth.")]
        )
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 hello.py"},
                content_preview="PASSED\n\ntruth",
            )
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "confirmed the exact request" in result.reason

    async def test_read_only_shell_does_not_require_verify_work(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="shell",
                is_error=False,
                arguments={"command": 'curl -s "https://wttr.in/Tokyo?format=3"'},
            )
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is True
        assert "verification not required" in result.reason.lower()

    async def test_mutating_shell_without_verify_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="shell",
                is_error=False,
                arguments={"command": "touch weather.txt"},
            )
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "verify_work" in result.reason

    async def test_write_without_verify_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [self._activity(kind="tool_call.completed", name="write_file", is_error=False)]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "verify_work" in result.reason

    async def test_write_without_verify_mentions_configured_default(self) -> None:
        verifier = VerifyBeforeDoneVerifier(default_verify_command_available=True)
        activity = [self._activity(kind="tool_call.completed", name="write_file", is_error=False)]

        result = await verifier.verify(session=_session(), activity=activity)

        assert result.can_finish is False
        assert "configured default verifier" in result.reason
        assert "verify_work with no arguments" not in result.reason

    async def test_write_then_final_user_decision_handoff_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(role="user", content="Fix the bug."),
                Message(
                    role="assistant",
                    content=(
                        "Please confirm one of these paths: reset and proceed, "
                        "or continue from the current state."
                    ),
                ),
            ]
        )
        activity = [self._activity(kind="tool_call.completed", name="write_file", is_error=False)]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "asks the user to choose" in result.reason
        assert "Continue autonomously" in result.reason

    async def test_write_then_verify_passes(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "pytest tests"},
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is True

    async def test_write_then_valid_verify_allows_optional_follow_up(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(role="user", content="Fix the bug."),
                Message(role="assistant", content="Done. Do you want me to also update docs?"),
            ]
        )
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "pytest tests"},
                content_preview="PASSED\n\n1 passed",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True

    async def test_write_then_python_unittest_output_passes(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "slugify_text.py"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 test_slugify.py"},
                content_preview="PASSED\n\n......\nRan 6 tests in 0.001s\n\nOK",
                metadata={"exit_code": 0, "stdout": "......\nRan 6 tests in 0.001s\n\nOK\n"},
            ),
        ]

        result = await verifier.verify(session=_session(), activity=activity)

        assert result.can_finish is True

    async def test_read_only_shell_sequence_does_not_require_verify_work(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="shell",
                is_error=False,
                arguments={"command": "date; TZ='Asia/Tokyo' date"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="shell",
                is_error=False,
                arguments={"command": "node -v"},
            ),
        ]

        result = await verifier.verify(session=_session(), activity=activity)

        assert result.can_finish is True
        assert "No modifying tool calls" in result.reason

    async def test_write_then_test_script_all_tests_passed_output_passes(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "node test_sum_cli.js"},
                content_preview="PASSED\n\nRunning tests...\nAll tests passed successfully!",
                metadata={
                    "exit_code": 0,
                    "stdout": "Running tests...\nAll tests passed successfully!\n",
                },
            ),
        ]

        result = await verifier.verify(session=_session(), activity=activity)

        assert result.can_finish is True

    async def test_write_then_test_script_all_test_cases_passed_output_passes(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 test_slug.py"},
                content_preview="PASSED\n\nAll test cases passed!",
                metadata={"exit_code": 0, "stdout": "All test cases passed!\n"},
            ),
        ]

        result = await verifier.verify(session=_session(), activity=activity)

        assert result.can_finish is True

    async def test_write_then_test_script_verification_successful_output_passes(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 test_weather.py"},
                content_preview="PASSED\n\nVerification Successful: tokyo: 22C",
                metadata={"exit_code": 0, "stdout": "Verification Successful: tokyo: 22C\n"},
            ),
        ]

        result = await verifier.verify(session=_session(), activity=activity)

        assert result.can_finish is True

    async def test_multiturn_uses_latest_user_prompt_for_exact_requests(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(
                    role="user",
                    content="Create project_note.txt containing exactly Harness validates real work.",
                ),
                Message(role="assistant", content="done"),
                Message(
                    role="user",
                    content=(
                        "In the same workspace, append a second sentence to project_note.txt: "
                        "Verified twice."
                    ),
                ),
            ]
        )
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={
                    "path": "project_note.txt",
                    "content": "Harness validates real work. Verified twice.",
                },
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 test_project_note.py"},
                content_preview="PASSED\n\n..\nRan 2 tests in 0.001s\n\nOK",
                metadata={"exit_code": 0, "stdout": "..\nRan 2 tests in 0.001s\n\nOK\n"},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True

    async def test_write_then_verify_with_benign_env_assignment_passes(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "env CI=true PYTHONPATH=src pytest tests"},
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is True

    async def test_write_then_node_builtin_test_runner_passes(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "src/urlJoin.js"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "test/urlJoin.test.js"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "node --test"},
                content_preview="PASSED\n\nok joins URLs\n# tests 16\n# pass 16",
                metadata={
                    "exit_code": 0,
                    "stdout": "ok joins URLs\n# tests 16\n# pass 16\n",
                },
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is True

    async def test_write_then_verify_without_command_or_default_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(kind="tool_call.completed", name="verify_work", is_error=False),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_trivial_verify_command_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "printf ok"},
                content_preview="PASSED\n\nok",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_unrelated_assertion_after_program_run_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "weather.py"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        "python3 weather.py && "
                        "curl -s https://example.test/weather.json | grep -q temperature"
                    )
                },
                content_preview="PASSED\n\nCurrent weather in Tokyo: 25C",
                metadata={
                    "stdout": "Current weather in Tokyo: 25C\n",
                    "exit_code": 0,
                    "workspace_changed": False,
                },
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason
        assert "changed work" in result.reason

    async def test_write_then_env_assignment_non_executing_verify_command_blocks(
        self,
    ) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "PYTEST_ADDOPTS=--collect-only pytest tests"},
                content_preview="PASSED\n\ncollected 3 items",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_masked_generic_verify_command_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "pytest tests || { echo 'not really checked'; exit 0; }"},
                content_preview="PASSED\n\nnot really checked",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_traceback_verify_output_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "pytest tests"},
                content_preview=(
                    "PASSED\n\nTraceback (most recent call last):\nAssertionError: wrong result"
                ),
                metadata={
                    "stdout": (
                        "Traceback (most recent call last):\nAssertionError: wrong result\n"
                    ),
                    "exit_code": 0,
                },
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_error_verify_output_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "pytest tests"},
                content_preview="PASSED\n\nERROR: expected harness-ok got wrong",
                metadata={
                    "stdout": "ERROR: expected harness-ok got wrong\n",
                    "exit_code": 0,
                },
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_only_skipped_verify_output_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "pytest tests"},
                content_preview="PASSED\n\n1 skipped, 0 passed, 0 failed",
                metadata={
                    "stdout": "1 skipped, 0 passed, 0 failed\n",
                    "exit_code": 0,
                },
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_all_skipped_verify_output_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "pytest tests"},
                content_preview="PASSED\n\n3 skipped in 0.02s",
                metadata={
                    "stdout": "3 skipped in 0.02s\n",
                    "exit_code": 0,
                },
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_all_deselected_verify_output_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "pytest tests"},
                content_preview="PASSED\n\n3 deselected in 0.02s",
                metadata={
                    "stdout": "3 deselected in 0.02s\n",
                    "exit_code": 0,
                },
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_zero_pass_verify_output_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "pytest tests"},
                content_preview="PASSED\n\n0 passed, 0 failed",
                metadata={
                    "stdout": "0 passed, 0 failed\n",
                    "exit_code": 0,
                },
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    @pytest.mark.parametrize(
        "command",
        [
            "false && pytest tests",
            "exit 0; pytest tests",
            "false && grep expected src/app.py",
        ],
    )
    async def test_write_then_unreachable_generic_verify_command_blocks(
        self,
        command: str,
    ) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": command},
                content_preview="PASSED",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_if_not_masked_verify_command_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "if ! pytest tests; then echo ok; fi"},
                content_preview="PASSED\n\nok",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_multiline_if_not_masked_verify_command_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "if ! pytest tests\nthen echo ok\nfi"},
                content_preview="PASSED\n\nok",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_quoted_test_runner_text_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "printf 'pytest passed'"},
                content_preview="PASSED\n\npytest passed",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_inline_comment_assertion_verify_command_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "printf ok # assert the changed behavior"},
                content_preview="PASSED\n\nok",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_bare_assert_word_verify_command_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "src/app.py"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "assert src/app.py"},
                content_preview="PASSED",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_fake_package_script_verify_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "npm run echo-ok"},
                content_preview="PASSED\n\nok",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_package_test_script_verify_passes(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "npm run test:unit"},
                content_preview="PASSED\n\nunit tests passed",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is True

    async def test_write_then_nonzero_exit_verify_metadata_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "pytest -q"},
                content_preview="PASSED",
                metadata={"exit_code": 1},
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_skipped_broad_check_verify_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "false && true && pytest -q"},
                content_preview="PASSED",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    @pytest.mark.parametrize(
        "command",
        [
            "pytest --version",
            "ruff --version",
            "pytest --collect-only tests",
            "go test -list TestThing ./...",
            "npm test -- --help",
        ],
    )
    async def test_write_then_non_executing_verify_command_blocks(
        self,
        command: str,
    ) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": command},
                content_preview="PASSED",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_write_then_unrelated_assertion_verify_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "src/app.py"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "test -f unrelated.txt"},
                content_preview="PASSED",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "changed work" in result.reason

    @pytest.mark.parametrize(
        "command",
        [
            "test -f src/app.py",
            "[ -s src/app.py ]",
            "test -f src/app.py && echo ok",
        ],
    )
    async def test_write_then_presence_only_changed_path_assertion_blocks(
        self,
        command: str,
    ) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "src/app.py"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": command},
                content_preview="PASSED",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "changed work" in result.reason

    async def test_write_then_changed_path_assertion_verify_passes(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "src/app.py"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "grep expected src/app.py"},
                content_preview="PASSED\n\nexpected",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is True

    async def test_write_then_changed_path_presence_with_unrelated_content_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "src/app.py"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "test -f src/app.py && grep expected unrelated.txt"},
                content_preview="PASSED",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "changed work" in result.reason

    async def test_write_then_changed_path_presence_plus_content_verify_passes(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "src/app.py"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "test -f src/app.py && grep expected src/app.py"},
                content_preview="PASSED",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is True

    async def test_write_then_statically_failing_verify_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "src/app.py"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "pytest -q && false && true && grep expected src/app.py"},
                content_preview="PASSED",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "meaningful test/check command" in result.reason

    async def test_apply_patch_then_unrelated_assertion_verify_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        patch = (
            "diff --git a/src/app.py b/src/app.py\n"
            "--- a/src/app.py\n"
            "+++ b/src/app.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+expected\n"
        )
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="apply_patch",
                is_error=False,
                arguments={"patch": patch},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "test -f unrelated.txt"},
                content_preview="PASSED",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "changed work" in result.reason

    async def test_shell_mutation_then_unrelated_assertion_verify_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="shell",
                is_error=False,
                arguments={"command": "touch src/app.py"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "test -f unrelated.txt"},
                content_preview="PASSED",
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "changed work" in result.reason

    async def test_write_then_failed_verify_blocks(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="shell", is_error=False),
            self._activity(kind="tool_call.completed", name="verify_work", is_error=True),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "latest verify_work after the final state change failed" in result.reason

    async def test_write_after_verify_blocks_until_verify_runs_again(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(kind="tool_call.completed", name="verify_work", is_error=False),
            self._activity(kind="tool_call.completed", name="edit_file", is_error=False),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "after the final change" in result.reason

    async def test_mutating_verify_work_does_not_satisfy_verification(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "printf '%s' truth > result.txt"},
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "non-mutating" in result.reason

    async def test_container_internal_setup_verify_does_not_count_as_workspace_mutation(
        self,
    ) -> None:
        verifier = VerifyBeforeDoneVerifier()
        command = "docker run --rm image bash -lc 'mkdir -p /root/go/bin && go test ./...'"
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "src/app.go"},
                metadata={"path": "src/app.go"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": command},
                content_preview="PASSED\n\nok  \tgithub.com/example/project\t0.123s",
                metadata={
                    "command": command,
                    "exit_code": 0,
                    "stdout": "ok  \tgithub.com/example/project\t0.123s\n",
                    "output_reports_failure": False,
                    "workspace_changed": False,
                    "workspace_fingerprint_changed": False,
                },
            ),
        ]

        result = await verifier.verify(session=_session(), activity=activity)

        assert result.can_finish is True
        assert "passing verify_work ran after the last state change" in result.reason

    async def test_verify_work_workspace_changed_metadata_does_not_satisfy_verification(
        self,
    ) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "pytest tests"},
                metadata={"workspace_changed": True},
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "non-mutating" in result.reason

    async def test_truncate_verify_work_does_not_satisfy_verification(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "truncate -s 0 result.txt"},
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "non-mutating" in result.reason

    async def test_mutating_verify_work_then_read_only_verify_passes(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "printf '%s' truth > result.txt"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "grep -qx truth result.txt"},
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is True
        assert "after the last state change" in result.reason

    async def test_failed_shell_workspace_change_invalidates_prior_verify(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(kind="tool_call.completed", name="verify_work", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="shell",
                is_error=True,
                arguments={"command": "pytest tests"},
                metadata={"workspace_changed": True},
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "after the final change" in result.reason

    async def test_configured_default_verify_work_is_treated_as_read_only(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={},
                metadata={"used_default_command": True},
            ),
        ]

        result = await verifier.verify(session=_session(), activity=activity)

        assert result.can_finish is True
        assert "after the last state change" in result.reason

    async def test_exact_content_task_rejects_print_only_verify_work(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Create result.txt containing exactly shelltruth with no trailing "
                        "newline. Verify the file content and byte size before finishing."
                    ),
                )
            ]
        )
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "cat result.txt"},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "must assert the requested file content and byte size" in result.reason

    async def test_exact_content_task_rejects_echoed_substitution_verify_work(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Create result.txt containing exactly shelltruth with no trailing "
                        "newline. Verify the file content and byte size before finishing."
                    ),
                )
            ]
        )
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        'test "$(cat result.txt >/dev/null; echo shelltruth)" = "shelltruth" '
                        "&& test $(wc -c < result.txt) -eq 10"
                    )
                },
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "must assert the requested file content and byte size" in result.reason

    async def test_exact_content_task_rejects_grep_side_input_verify_work(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Create result.txt containing exactly shelltruth with no trailing "
                        "newline. Verify the file content and byte size before finishing."
                    ),
                )
            ]
        )
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        "grep -qx 'shelltruth' <(printf %s shelltruth) result.txt "
                        "&& test $(wc -c < result.txt) -eq 10"
                    )
                },
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "must assert the requested file content and byte size" in result.reason

    async def test_exact_content_task_rejects_unreachable_grep_verify_work(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(
                    role="user",
                    content="Create result.txt containing exactly shelltruth.",
                )
            ]
        )
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "false && grep -qx shelltruth result.txt"},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "must assert the requested file content" in result.reason

    async def test_exact_content_task_rejects_unreachable_byte_size_verify_work(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Create result.txt containing exactly shelltruth with no trailing "
                        "newline. Verify the file content and byte size before finishing."
                    ),
                )
            ]
        )
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        'test "$(cat result.txt)" = "shelltruth" '
                        "&& false && test $(wc -c < result.txt) -eq 10"
                    )
                },
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "must assert the requested file content and byte size" in result.reason

    async def test_no_trailing_newline_task_requires_byte_assertion_without_verify_prompt(
        self,
    ) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Create a file named result.txt with content exactly shelltruth "
                        "with no trailing newline."
                    ),
                )
            ]
        )
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": 'test "$(cat result.txt)" = "shelltruth"'},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "must assert the requested file content and byte size" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        'test "$(cat result.txt)" = "shelltruth" '
                        "&& test $(wc -c < result.txt) -eq 10"
                    )
                },
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "after the last state change" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": 'test "$(cat result.txt)" = "shelltruth"'},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "test $(wc -c < result.txt) -eq 10"},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "after the last state change" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        '([ "$(< result.txt)" = "shelltruth" ] '
                        "&& [ \"$(wc -c < result.txt | tr -d ' ')\" -eq 10 ]) || exit 1"
                    )
                },
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "after the last state change" in result.reason

    async def test_exact_content_task_accepts_generated_assertion_script_after_final_state_change(
        self,
    ) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Create a file named exact_message.txt whose exact content is "
                        "Hello verified harness with no trailing newline."
                    ),
                )
            ]
        )
        test_script = (
            "def test_file_content():\n"
            "    with open('exact_message.txt', 'rb') as f:\n"
            "        content = f.read()\n"
            "    expected = b'Hello verified harness'\n"
            "    assert content == expected\n"
            "    assert not content.endswith(b'\\n')\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_file_content()\n"
            "    print('Verification PASSED')\n"
        )
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "exact_message.txt", "content": "Hello verified harness"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "test_exact_message.py", "content": test_script},
                metadata={"path": "test_exact_message.py", "content_after": test_script},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 test_exact_message.py"},
                content_preview="PASSED\n\nVerification PASSED",
                metadata={"exit_code": 0, "stdout": "Verification PASSED\n"},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "after the last state change" in result.reason

    async def test_exact_content_task_accepts_inline_python_assertion_verify_work(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Create a file named exact_message.txt whose exact content is "
                        "Hello verified harness with no trailing newline."
                    ),
                )
            ]
        )
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="shell",
                is_error=False,
                arguments={"command": 'printf "Hello verified harness" > exact_message.txt'},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        "python3 -c \"content = open('exact_message.txt', 'rb').read(); "
                        "assert content == b'Hello verified harness'\""
                    )
                },
                content_preview="PASSED",
                metadata={"exit_code": 0, "stdout": ""},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "after the last state change" in result.reason

    async def test_exact_content_task_rejects_print_only_generated_script(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Create a file named exact_message.txt whose exact content is "
                        "Hello verified harness with no trailing newline."
                    ),
                )
            ]
        )
        test_script = (
            "with open('exact_message.txt', 'rb') as f:\n"
            "    content = f.read()\n"
            "expected = b'Hello verified harness'\n"
            "print('Verification PASSED')\n"
        )
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "exact_message.txt", "content": "Hello verified harness"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "test_exact_message.py", "content": test_script},
                metadata={"path": "test_exact_message.py", "content_after": test_script},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 test_exact_message.py"},
                content_preview="PASSED\n\nVerification PASSED",
                metadata={"exit_code": 0, "stdout": "Verification PASSED\n"},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "must assert the requested file content and byte size" in result.reason

    async def test_exact_content_task_rejects_strip_based_generated_script(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Create a file named exact_message.txt whose exact content is "
                        "Hello verified harness with no trailing newline."
                    ),
                )
            ]
        )
        test_script = (
            "with open('exact_message.txt', 'r') as f:\n"
            "    content = f.read().strip()\n"
            "expected = 'Hello verified harness'\n"
            "assert content == expected\n"
            "print('Verification PASSED')\n"
        )
        activity = [
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "exact_message.txt", "content": "Hello verified harness"},
            ),
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "test_exact_message.py", "content": test_script},
                metadata={"path": "test_exact_message.py", "content_after": test_script},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 test_exact_message.py"},
                content_preview="PASSED\n\nVerification PASSED",
                metadata={"exit_code": 0, "stdout": "Verification PASSED\n"},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "must assert the requested file content and byte size" in result.reason

    async def test_exact_content_task_rejects_masked_assertion_failure(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Create result.txt containing exactly shelltruth with no trailing "
                        "newline. Verify the file content and byte size before finishing."
                    ),
                )
            ]
        )
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        'test "$(cat result.txt)" = "shelltruth" || true && '
                        "test $(wc -c < result.txt) -eq 10"
                    )
                },
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "must assert the requested file content and byte size" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        'test "$(cat result.txt)" = "shelltruth" '
                        "&& test $(wc -c < result.txt) -eq 10 && (false)"
                    )
                },
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "must assert the requested file content and byte size" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        'test "$(cat result.txt)" = "shelltruth" '
                        "&& test $(wc -c < result.txt) -eq 10 && sh -c false"
                    )
                },
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "must assert the requested file content and byte size" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        'test "$(cat result.txt)" = "shelltruth" '
                        "&& test $(wc -c < result.txt) -eq 10 & false"
                    )
                },
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "must assert the requested file content and byte size" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        'test "$(cat result.txt)" = "shelltruth" '
                        "&& test $(wc -c < result.txt) -eq 10 && env false"
                    )
                },
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "must assert the requested file content and byte size" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        'test "$(cat result.txt)" = "shelltruth" '
                        "&& test $(wc -c < result.txt) -eq 10 && command false"
                    )
                },
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "must assert the requested file content and byte size" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        'test "$(cat result.txt)" = "shelltruth" '
                        "&& test $(wc -c < result.txt) -eq 10 && true | false"
                    )
                },
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "must assert the requested file content and byte size" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        "false && true && "
                        'test "$(cat result.txt)" = "shelltruth" '
                        "&& test $(wc -c < result.txt) -eq 10"
                    )
                },
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "must assert the requested file content and byte size" in result.reason

    async def test_exact_content_task_accepts_assertive_verify_work(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Create result.txt containing exactly shelltruth with no trailing "
                        "newline. Verify the file content and byte size before finishing."
                    ),
                )
            ]
        )
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        'test "$(cat result.txt)" = "shelltruth" '
                        "&& test $(wc -c < result.txt) -eq 10"
                    )
                },
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "after the last state change" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={
                    "command": (
                        'test "$(cat result.txt)" = shelltruth && test $(wc -c < result.txt) -eq 10'
                    )
                },
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "after the last state change" in result.reason

    async def test_exact_stdout_task_requires_program_output_evidence(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(
                    role="user",
                    content="Create hello.py that prints exactly harness-ok.",
                )
            ]
        )
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 hello.py"},
                content_preview="PASSED\n\nwrong",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "exact-output tasks" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 hello.py"},
                content_preview="PASSED\n\nharness-ok",
                metadata={
                    "stdout": "harness-ok\n",
                    "stderr": (
                        "Traceback (most recent call last):\nAssertionError: wrong result\n"
                    ),
                    "exit_code": 0,
                },
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "exact-output tasks" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 hello.py"},
                content_preview="PASSED\n\nharness-ok",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "after the last state change" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 hello.py # || false"},
                content_preview="PASSED\n\nharness-ok",
                metadata={"stdout": "harness-ok\n", "exit_code": 0},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "after the last state change" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 hello.py"},
                content_preview="PASSED\n\nharness-ok",
                metadata={"stdout": "harness-ok\n", "exit_code": 1},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "exact-output tasks" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": 'test "$(python3 hello.py)" = harness-ok'},
                content_preview="PASSED",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "after the last state change" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": 'test "$(python3 hello.py)" = harness-ok # || true'},
                content_preview="PASSED",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "after the last state change" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": 'test "$(python3 hello.py)" = harness-ok && false'},
                content_preview="PASSED",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "exact-output tasks" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 hello.py && false"},
                content_preview="PASSED\n\nharness-ok",
                metadata={"stdout": "harness-ok\n", "exit_code": 0},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "exact-output tasks" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": 'test "$(python3 hello.py)" = harness-ok && (false)'},
                content_preview="PASSED",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "exact-output tasks" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": 'test "$(python3 hello.py)" = harness-ok && sh -c false'},
                content_preview="PASSED",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "exact-output tasks" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": 'test "$(python3 hello.py)" = harness-ok & false'},
                content_preview="PASSED",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "exact-output tasks" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 hello.py & false"},
                content_preview="PASSED\n\nharness-ok",
                metadata={"stdout": "harness-ok\n", "exit_code": 0},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "exact-output tasks" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": 'test "$(python3 hello.py)" = harness-ok && env false'},
                content_preview="PASSED",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "exact-output tasks" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": 'test "$(python3 hello.py)" = harness-ok && command false'},
                content_preview="PASSED",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "exact-output tasks" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": 'test "$(python3 hello.py)" = harness-ok && true | false'},
                content_preview="PASSED",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "exact-output tasks" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": 'false && true && test "$(python3 hello.py)" = harness-ok'},
                content_preview="PASSED",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "exact-output tasks" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": 'if false; then test "$(python3 hello.py)" = harness-ok; fi'},
                content_preview="PASSED",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "exact-output tasks" in result.reason

    async def test_exact_stdout_no_trailing_newline_requires_raw_stdout_match(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(
                    role="user",
                    content=(
                        "Create hello.py that prints exactly harness-ok with no trailing newline."
                    ),
                )
            ]
        )
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 hello.py"},
                content_preview="PASSED\n\nharness-ok",
                metadata={"stdout": "harness-ok\n", "exit_code": 0},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "no trailing newline" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 hello.py | tr -d '\\n'"},
                content_preview="PASSED\n\nharness-ok",
                metadata={"stdout": "harness-ok", "exit_code": 0},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "no trailing newline" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": 'test "$(python3 hello.py)" = harness-ok'},
                content_preview="PASSED",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "no trailing newline" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 hello.py"},
                content_preview="PASSED\n\nharness-ok",
                metadata={"stdout": "harness-ok", "exit_code": 0},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "after the last state change" in result.reason

        test_script = (
            "import subprocess\n"
            "\n"
            "def test_emit_code():\n"
            "    result = subprocess.run(['python3', 'hello.py'], capture_output=True, text=True)\n"
            "    assert result.stdout == 'harness-ok'\n"
            "    assert not result.stdout.endswith('\\n')\n"
            "    assert result.returncode == 0\n"
            "\n"
            "if __name__ == '__main__':\n"
            "    test_emit_code()\n"
            "    print('Tests passed!')\n"
        )
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "test_hello.py", "content": test_script},
                metadata={"path": "test_hello.py", "content_after": test_script},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 test_hello.py"},
                content_preview="PASSED\n\nTests passed!",
                metadata={"stdout": "Tests passed!\n", "exit_code": 0},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "after the last state change" in result.reason

        strip_script = (
            "import subprocess\n"
            "result = subprocess.run(['python3', 'hello.py'], capture_output=True, text=True)\n"
            "assert result.stdout.strip() == 'harness-ok'\n"
            "print('Tests passed!')\n"
        )
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="write_file",
                is_error=False,
                arguments={"path": "test_hello.py", "content": strip_script},
                metadata={"path": "test_hello.py", "content_after": strip_script},
            ),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 test_hello.py"},
                content_preview="PASSED\n\nTests passed!",
                metadata={"stdout": "Tests passed!\n", "exit_code": 0},
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "no trailing newline" in result.reason

    async def test_named_exact_stdout_task_requires_program_output_evidence(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        session = _session(
            messages=[
                Message(
                    role="user",
                    content="Create a Python script named hello.py that prints exactly harness-ok.",
                )
            ]
        )
        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 hello.py"},
                content_preview="PASSED\n\nwrong",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "exact-output tasks" in result.reason

        activity = [
            self._activity(kind="tool_call.completed", name="write_file", is_error=False),
            self._activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=False,
                arguments={"command": "python3 hello.py"},
                content_preview="PASSED\n\nharness-ok",
            ),
        ]

        result = await verifier.verify(session=session, activity=activity)

        assert result.can_finish is True
        assert "after the last state change" in result.reason

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

    async def test_payload_only_pr_flow_still_requires_verify_work(self) -> None:
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
                name="shell",
                is_error=False,
                arguments={"command": "harness research pr --candidate promo-test"},
            ),
        ]
        result = await verifier.verify(session=_session(), activity=activity)
        assert result.can_finish is False
        assert "never ran verify_work" in result.reason

    async def test_shell_driven_open_pr_flow_skips_generic_verify_requirement(self) -> None:
        verifier = VerifyBeforeDoneVerifier()
        activity = [
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
                arguments={
                    "command": "harness research pr --candidate promo-test --push --open --draft"
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

    async def test_blocks_payload_only_pr_then_raw_gh_create(self) -> None:
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
                arguments={"command": "harness research pr --candidate promo-test"},
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
        assert "harness research pr --push --open" in result.reason

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
        assert "harness research pr --push --open" in result.reason

    async def test_blocks_shell_driven_payload_then_raw_gh_flow(self) -> None:
        verifier = ResearchPromotionFlowVerifier()
        activity = [
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
                arguments={"command": "harness research pr --candidate promo-test"},
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
        assert "harness research pr --push --open" in result.reason


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

    async def test_blocks_disproven_prompt_surface_edit_before_latest_verify_is_green(
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
            _activity(
                kind="tool_call.completed",
                name="verify_work",
                is_error=True,
                content_preview="FAILED tests/test_cache.py::test_timeout_constant_unchanged",
            ),
        ]

        result = await PromptSurfaceRevertVerifier().verify(session=session, activity=activity)

        assert result.can_finish is False
        assert "latest verify_work is still failing" in result.reason
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
