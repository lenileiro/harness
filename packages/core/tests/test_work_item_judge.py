"""Tests for WorkItemJudge — isolated post-completion verifier."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from harness.core.activity import ActivityEvent
from harness.core.schemas import Message
from harness.core.verification import WorkItemJudge
from harness.storage.memory import InMemoryStorage

from .conftest import MockAdapter


def _activity_event(
    kind: str,
    *,
    tool_name: str = "shell",
    is_error: bool = False,
    session_id: str = "sess_test",
) -> ActivityEvent:
    return ActivityEvent(
        id=f"evt_{uuid.uuid4().hex[:8]}",
        session_id=session_id,
        kind=kind,
        data={"name": tool_name, "is_error": is_error},
    )


def _judge_response(*, passed: bool, reason: str = "looks good", confidence: float = 0.9) -> list:
    payload = json.dumps({"can_finish": passed, "reason": reason, "confidence": confidence})
    from harness.core.events import Done, TextDelta

    return [TextDelta(text=payload), Done(final_message=None, usage=None)]


def _make_judge(adapter: MockAdapter) -> WorkItemJudge:
    return WorkItemJudge(adapter=adapter, model="test-model", max_retries=2)


# ---------------------------------------------------------------------------
# Pass / fail on summary content
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_passes_on_substantive_summary() -> None:
    adapter = MockAdapter("mock", scripts=[_judge_response(passed=True, reason="task completed")])
    judge = _make_judge(adapter)

    activity = [_activity_event("tool_call.completed", tool_name="write_file")]
    result = await judge.judge(
        task_title="Write a hello world file",
        task_description=None,
        result_summary="Created hello.py with print('hello world')",
        activity=activity,
    )

    assert result.can_finish is True
    assert "task completed" in result.reason
    assert result.verifier_name == "work_item_judge"


@pytest.mark.asyncio
async def test_judge_fails_on_empty_summary() -> None:
    adapter = MockAdapter(
        "mock",
        scripts=[_judge_response(passed=False, reason="summary is empty", confidence=0.95)],
    )
    judge = _make_judge(adapter)

    result = await judge.judge(
        task_title="Write a hello world file",
        task_description=None,
        result_summary="",
        activity=[],
    )

    assert result.can_finish is False
    assert result.verifier_name == "work_item_judge"


@pytest.mark.asyncio
async def test_judge_fails_when_no_tools_called() -> None:
    adapter = MockAdapter(
        "mock",
        scripts=[_judge_response(passed=False, reason="no tools were called", confidence=0.88)],
    )
    judge = _make_judge(adapter)

    result = await judge.judge(
        task_title="Analyze the codebase",
        task_description="Read all Python files and summarize",
        result_summary="done",
        activity=[],  # no activity events
    )

    assert result.can_finish is False


# ---------------------------------------------------------------------------
# Tool activity summary appears in the prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_prompt_includes_tool_calls() -> None:
    from harness.core.events import Done, TextDelta

    captured_messages: list[list[Message]] = []

    class CapturingAdapter:
        name = "capturing"

        async def stream(self, *, model: str, messages: list[Message]):  # type: ignore[misc]
            captured_messages.append(messages)
            yield TextDelta(text='{"can_finish": true, "reason": "ok", "confidence": 0.9}')
            yield Done(final_message=None, usage=None)

    judge = WorkItemJudge(adapter=CapturingAdapter(), model="test", max_retries=1)  # type: ignore[arg-type]
    activity = [
        _activity_event("tool_call.completed", tool_name="write_file", is_error=False),
        _activity_event("tool_call.completed", tool_name="shell", is_error=False),
    ]
    await judge.judge(
        task_title="Build something",
        task_description="Do the thing",
        result_summary="did it",
        activity=activity,
    )

    assert captured_messages
    prompt_text = captured_messages[0][-1].content or ""
    assert "write_file" in prompt_text
    assert "shell" in prompt_text
    assert "Build something" in prompt_text
    assert "Do the thing" in prompt_text


# ---------------------------------------------------------------------------
# Retry logic
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_retries_on_non_json_response() -> None:
    from harness.core.events import Done, TextDelta

    good_payload = json.dumps({"can_finish": True, "reason": "ok on retry", "confidence": 0.8})
    adapter = MockAdapter(
        "mock",
        scripts=[
            [TextDelta(text="not json at all"), Done(final_message=None, usage=None)],
            [TextDelta(text=good_payload), Done(final_message=None, usage=None)],
        ],
    )
    judge = _make_judge(adapter)
    result = await judge.judge(
        task_title="Test task",
        task_description=None,
        result_summary="summary",
        activity=[],
    )
    assert result.can_finish is True
    assert "ok on retry" in result.reason


@pytest.mark.asyncio
async def test_judge_fails_gracefully_when_all_retries_exhausted() -> None:
    from harness.core.events import Done, TextDelta

    bad_script = [TextDelta(text="garbage"), Done(final_message=None, usage=None)]
    adapter = MockAdapter("mock", scripts=[bad_script, bad_script])
    judge = _make_judge(adapter)

    result = await judge.judge(
        task_title="Test task",
        task_description=None,
        result_summary="summary",
        activity=[],
    )
    assert result.can_finish is False
    assert result.confidence == 0.0


@pytest.mark.asyncio
async def test_judge_fails_gracefully_on_adapter_error() -> None:
    from harness.core.events import Done, TextDelta

    good_payload = json.dumps({"can_finish": True, "reason": "recovered", "confidence": 0.7})
    adapter = MockAdapter(
        "mock",
        scripts=[
            [TextDelta(text=good_payload), Done(final_message=None, usage=None)],
        ],
        error=RuntimeError("connection failed"),
    )
    judge = _make_judge(adapter)

    # error on first attempt, good response on second
    result = await judge.judge(
        task_title="Test task",
        task_description=None,
        result_summary="summary",
        activity=[],
    )
    # With error adapter, first attempt raises → falls through to graceful fail
    # (MockAdapter with error= raises on ALL calls, so both retries fail)
    assert result.can_finish is False


# ---------------------------------------------------------------------------
# Confidence propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_judge_confidence_propagated() -> None:
    adapter = MockAdapter(
        "mock",
        scripts=[_judge_response(passed=True, confidence=0.77)],
    )
    judge = _make_judge(adapter)
    result = await judge.judge(
        task_title="T",
        task_description=None,
        result_summary="done it",
        activity=[],
    )
    assert result.confidence == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# Orchestrator integration: judge wires into _post_run_check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestrator_emits_verified_event_on_judge_pass() -> None:
    from harness.core.orchestrator import (
        AgentRole,
        MultiAgentOrchestrator,
        WorkItemVerifiedEvent,
        WorkQueue,
    )

    store = InMemoryStorage()
    parent_id = f"job_{uuid.uuid4().hex[:8]}"
    q = WorkQueue(store, parent_id=parent_id)
    await q.push(title="do something", cwd=Path("/tmp"))

    # Judge always passes
    judge_adapter = MockAdapter(
        "judge",
        scripts=[_judge_response(passed=True, reason="verified")] * 10,
    )
    judge = WorkItemJudge(adapter=judge_adapter, model="test", max_retries=1)

    # Stub agent factory: agent does nothing except complete the work item via store
    def completing_factory(role: AgentRole):  # type: ignore[return]
        class _StubAgent:
            async def run(self, request):  # type: ignore[misc]
                from harness.core.events import Done, TextDelta

                if role.name.startswith("worker") and role.item_id:
                    task = await store.get_task(role.item_id)
                    if task:
                        updated = task.model_copy(
                            update={
                                "status": "done",
                                "metadata": {
                                    **task.metadata,
                                    "result_summary": "I wrote the file",
                                },
                            }
                        )
                        await store.update_task(updated)
                yield TextDelta(text="done")
                yield Done(final_message=None, usage=None)

        return _StubAgent()

    orchestrator = MultiAgentOrchestrator(
        agent_factory=completing_factory,  # type: ignore[arg-type]
        store=store,
        planner_role=AgentRole(name="planner", system_prompt="plan"),
        worker_role=AgentRole(name="worker", system_prompt="work"),
        reporter_role=AgentRole(name="reporter", system_prompt="report"),
        max_workers=1,
        job_cwd=Path("/tmp"),
        provider="mock",
        model="test",
        work_item_judge=judge,
    )

    events = []
    async for event in orchestrator.run("do something", job_id=parent_id):
        events.append(event)

    verified = [e for e in events if isinstance(e, WorkItemVerifiedEvent)]
    assert len(verified) == 1
    assert verified[0].task_ref != ""


@pytest.mark.asyncio
async def test_orchestrator_emits_rejected_event_and_resets_task() -> None:
    from harness.core.orchestrator import (
        AgentRole,
        MultiAgentOrchestrator,
        WorkItemRejectedEvent,
        WorkQueue,
    )

    store = InMemoryStorage()
    parent_id = f"job_{uuid.uuid4().hex[:8]}"
    q = WorkQueue(store, parent_id=parent_id)
    await q.push(title="do something", cwd=Path("/tmp"))

    call_count = 0

    # Judge always rejects (to exhaust retries quickly)
    judge_adapter = MockAdapter(
        "judge",
        scripts=[_judge_response(passed=False, reason="bad work")] * 10,
    )
    judge = WorkItemJudge(adapter=judge_adapter, model="test", max_retries=2)

    def completing_factory(role: AgentRole):  # type: ignore[return]
        nonlocal call_count

        class _StubAgent:
            async def run(self, request):  # type: ignore[misc]
                from harness.core.events import Done, TextDelta

                nonlocal call_count
                call_count += 1
                if role.name.startswith("worker") and role.item_id:
                    t = await store.get_task(role.item_id)
                    if t:
                        updated = t.model_copy(
                            update={
                                "status": "done",
                                "metadata": {**t.metadata, "result_summary": "done"},
                            }
                        )
                        await store.update_task(updated)
                yield TextDelta(text="done")
                yield Done(final_message=None, usage=None)

        return _StubAgent()

    orchestrator = MultiAgentOrchestrator(
        agent_factory=completing_factory,  # type: ignore[arg-type]
        store=store,
        planner_role=AgentRole(name="planner", system_prompt="plan"),
        worker_role=AgentRole(name="worker", system_prompt="work"),
        reporter_role=AgentRole(name="reporter", system_prompt="report"),
        max_workers=1,
        job_cwd=Path("/tmp"),
        provider="mock",
        model="test",
        work_item_judge=judge,
        max_judge_retries=2,
    )

    events = []
    async for event in orchestrator.run("do something", job_id=parent_id):
        events.append(event)

    rejected = [e for e in events if isinstance(e, WorkItemRejectedEvent)]
    assert len(rejected) >= 1
    assert rejected[0].reason == "bad work"
    # Worker was called more than once (task was re-queued)
    assert call_count >= 2
