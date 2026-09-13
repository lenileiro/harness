from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta

import aiosqlite
import pytest

from harness.core import (
    Agent,
    Done,
    Event,
    FailoverPolicy,
    Message,
    RunRequest,
    ToolCall,
    ToolRegistry,
)
from harness.core.clarification import (
    PendingQuestion,
    QuestionSpec,
    QuestionStore,
    apply_question_answers,
)
from harness.core.errors import ConfigurationError
from harness.core.memory import MemoryScope
from harness.core.runtime import fork_session
from harness.core.tools_clarify import ClarifyTool
from harness.storage.sqlite import SQLiteStorage

from .conftest import MockAdapter, MockTool, text_turn


def question_turn(*, extra=False) -> list[Event]:
    calls = [
        ToolCall(
            id="ask",
            name="clarify",
            arguments={
                "questions": [
                    {"question": "Which database?", "choices": ["SQLite", "Postgres"]},
                    {
                        "question": "Which checks?",
                        "choices": ["Unit", "Integration", "Types"],
                        "multi_select": True,
                    },
                    {"question": "Any other constraints?"},
                ]
            },
        )
    ]
    if extra:
        calls.append(ToolCall(id="mutation", name="write", arguments={}))
    return [Done(final_message=Message(role="assistant", tool_calls=calls))]


def agent_for(storage, questions, scope, adapter, tools=None):
    return Agent(
        adapters={"mock": adapter},
        tools=tools or ToolRegistry(),
        storage=storage,
        failover=FailoverPolicy(chain=["mock"]),
        default_model="fake",
        default_cwd=scope.workspace,
        memory_scope=scope,
        question_store=questions,
    )


async def test_actual_agent_sqlite_restart_answers_resume_original_tool_result(tmp_path):
    path = tmp_path / "session.db"
    scope = MemoryScope(workspace=str(tmp_path), user_id="owner")
    storage, questions = SQLiteStorage(path=path), QuestionStore(path)
    adapter = MockAdapter("mock", scripts=[question_turn(extra=True)])
    tool = MockTool(name="write")
    tools = ToolRegistry()
    tools.register(tool)
    agent = agent_for(storage, questions, scope, adapter, tools)
    request = RunRequest(session_id="session", prompt="Build the service")
    events = [event async for event in agent.run(request)]
    record = questions.list_pending(scope=scope, session_id="session")[0]
    session = await storage.get("session")
    assert session is not None and session.status == "paused"
    assert session.metadata["pause_reason"] == "clarification"
    terminal = events[-1]
    assert isinstance(terminal, Done) and terminal.structured_result is not None
    assert terminal.structured_result["question_id"] == record.id
    assert len(adapter.calls) == 1 and not tool.calls
    # An ordinary resume cannot bypass the pause or generate another model call.
    [event async for event in agent.resume("session", "please continue")]
    assert len(adapter.calls) == 1
    assert (
        questions.get(record.id, scope=MemoryScope(workspace=str(tmp_path), user_id="other"))
        is None
    )
    with pytest.raises(ValueError, match="owner"):
        questions.answer(
            record.id,
            scope=MemoryScope(workspace=str(tmp_path), user_id="other"),
            session_id="session",
            answers={"q0": "1"},
        )
    partial = questions.answer(record.id, scope=scope, session_id="session", answers={"q0": "1"})
    assert partial.status == "pending"
    questions.close()
    await storage.close()
    storage, questions = SQLiteStorage(path=path), QuestionStore(path)
    try:
        answered = questions.answer(
            record.id,
            scope=scope,
            session_id="session",
            answers={"q1": ["1", "3"], "q2": "Keep it local"},
        )
        assert answered.status == "answered"
        next_adapter = MockAdapter(
            "mock", scripts=[text_turn("Using SQLite with unit/type checks.")]
        )
        resumed = agent_for(storage, questions, scope, next_adapter)
        [event async for event in resumed.resume("session")]
        transcript = next_adapter.calls[0]["messages"]
        result = next(m for m in transcript if m.role == "tool" and m.tool_call_id == "ask")
        payload = json.loads(result.content)
        assert [r["user_response"] for r in payload["responses"]] == [
            "SQLite",
            ["Unit", "Types"],
            "Keep it local",
        ]
        assert not questions.list_pending(scope=scope)
        session = await storage.get("session")
        assert session is not None and session.status == "done"
        assert "pending_question_id" not in session.metadata
        skipped = next(m for m in session.messages if m.tool_call_id == "mutation")
        assert "Not executed" in (skipped.content or "")
    finally:
        questions.close()
        await storage.close()


async def test_expiry_resumes_with_partial_answers_and_timeout_without_busy_calls(tmp_path):
    path = tmp_path / "session.db"
    scope = MemoryScope(workspace=str(tmp_path))
    storage, questions = SQLiteStorage(path=path), QuestionStore(path)
    adapter = MockAdapter(
        "mock", scripts=[question_turn(), text_turn("Using the available answers")]
    )
    agent = agent_for(storage, questions, scope, adapter)
    try:
        [event async for event in agent.run(RunRequest(session_id="s", prompt="task"))]
        record = questions.list_pending(scope=scope)[0]
        questions.answer(record.id, scope=scope, session_id="s", answers={"q0": "2"})
        # Public payload timestamps simulate elapsed wall time, without sleeping.
        row = questions.db.execute(
            "SELECT payload FROM clarifications WHERE id=?", (record.id,)
        ).fetchone()
        expired = PendingQuestion.model_validate_json(row["payload"])
        expired.expires_at = datetime.now(UTC) - timedelta(seconds=1)
        with questions.db:
            questions.db.execute(
                "UPDATE clarifications SET payload=?,expires_at=? WHERE id=?",
                (expired.model_dump_json(), expired.expires_at.timestamp(), record.id),
            )
        [event async for event in agent.resume("s")]
        result = next(m for m in adapter.calls[-1]["messages"] if m.tool_call_id == "ask")
        assert json.loads(result.content)["timed_out"] is True
        assert json.loads(result.content)["responses"][0]["user_response"] == "Postgres"
        assert len(adapter.calls) == 2
    finally:
        questions.close()
        await storage.close()


def test_store_atomic_partial_answers_limits_cross_session_and_no_overwrites(tmp_path):
    scope = MemoryScope(workspace=str(tmp_path), user_id="owner")
    store = QuestionStore(tmp_path / "questions.db")
    try:
        question = store.create(
            session_id="s",
            scope=scope,
            tool_call_id="call",
            questions=[
                QuestionSpec(question="Choice", choices=["A", "B"]),
                QuestionSpec(question="Other"),
            ],
        )
        with pytest.raises(ValueError):
            store.answer(question.id, scope=scope, session_id="different", answers={"q0": "1"})
        with pytest.raises(ValueError):
            store.answer(
                question.id, scope=scope, session_id="s", answers={"q0": "1", "q99": "invalid"}
            )
        unchanged = store.get(question.id, scope=scope)
        assert unchanged is not None and unchanged.answers == {}
        store.answer(question.id, scope=scope, session_id="s", answers={"q0": "1"})
        with pytest.raises(ValueError, match="cannot be changed"):
            store.answer(question.id, scope=scope, session_id="s", answers={"q0": "2"})
        with pytest.raises(ValueError):
            store.create(
                session_id="s",
                scope=scope,
                tool_call_id="call2",
                questions=[QuestionSpec(question="Another")],
            )
        store.cancel(question.id, scope=scope, session_id="s")
        with pytest.raises(ValueError, match="cancelled"):
            store.answer(question.id, scope=scope, session_id="s", answers={"q1": "late"})
    finally:
        store.close()
        store.close()


async def test_question_tool_waits_for_sqlite_writer_without_blocking_commit(tmp_path):
    path = tmp_path / "questions.db"
    store = QuestionStore(path)
    scope = MemoryScope(workspace=str(tmp_path))
    tool = ClarifyTool(store, session_id="session", scope=scope)
    try:
        async with aiosqlite.connect(path) as writer:
            await writer.execute("BEGIN IMMEDIATE")
            task = asyncio.create_task(
                tool(
                    ToolCall(
                        id="q", name="clarify", arguments={"questions": [{"question": "Choice?"}]}
                    )
                )
            )
            await asyncio.sleep(0.02)
            assert not task.done()
            await writer.commit()
            result = await asyncio.wait_for(task, timeout=1)
            assert not result.is_error
    finally:
        store.close()


async def test_concurrent_partial_answers_merge_without_lost_updates(tmp_path):
    store = QuestionStore(tmp_path / "questions.db")
    scope = MemoryScope(workspace=str(tmp_path))
    try:
        record = store.create(
            session_id="s",
            scope=scope,
            tool_call_id="c",
            questions=[QuestionSpec(question="First?"), QuestionSpec(question="Second?")],
        )
        await asyncio.gather(
            asyncio.to_thread(
                store.answer, record.id, scope=scope, session_id="s", answers={"q0": "one"}
            ),
            asyncio.to_thread(
                store.answer, record.id, scope=scope, session_id="s", answers={"q1": "two"}
            ),
        )
        result = store.get(record.id, scope=scope)
        assert (
            result is not None
            and result.status == "answered"
            and result.answers == {"q0": "one", "q1": "two"}
        )
    finally:
        store.close()


async def test_cancelled_question_creation_cannot_leave_late_pending_record(tmp_path):
    path = tmp_path / "questions.db"
    store = QuestionStore(path)
    scope = MemoryScope(workspace=str(tmp_path))
    tool = ClarifyTool(store, session_id="s", scope=scope)
    try:
        async with aiosqlite.connect(path) as writer:
            await writer.execute("BEGIN IMMEDIATE")
            task = asyncio.create_task(
                tool(
                    ToolCall(
                        id="q", name="clarify", arguments={"questions": [{"question": "Which?"}]}
                    )
                )
            )
            await asyncio.sleep(0.02)
            task.cancel()
            await writer.commit()
            with pytest.raises(asyncio.CancelledError):
                await task
        rows = store.list_pending(scope=scope)
        assert len(rows) == 1 and rows[0].status == "cancelled"
    finally:
        store.close()


@pytest.mark.parametrize(
    "arguments",
    [
        {"questions": []},
        {"questions": [{"question": "Why?"}] * 6},
        {"questions": [{"question": "Why?", "choices": ["a", "b", "c", "d", "e"]}]},
        {"questions": [{"question": "Why?"}], "session_id": "other-session"},
        {"questions": [{"question": "Why?"}], "scope": {"user_id": "other"}},
    ],
)
async def test_clarify_contract_rejects_invalid_batches_and_model_identity_overrides(
    tmp_path, arguments
):
    scope = MemoryScope(workspace=str(tmp_path))
    store = QuestionStore(tmp_path / "questions.db")
    try:
        result = await ClarifyTool(store, session_id="s", scope=scope)(
            ToolCall(id="ask", name="clarify", arguments=arguments)
        )
        assert result.is_error and not store.list_pending(scope=scope)
    finally:
        store.close()


def test_pure_answer_merge_does_not_mutate_input_on_success_or_validation_failure(tmp_path):
    original = PendingQuestion(
        session_id="s",
        scope=MemoryScope(workspace=str(tmp_path)),
        tool_call_id="ask",
        questions=[
            QuestionSpec(question="First?", choices=["A", "B"]),
            QuestionSpec(question="Second?"),
        ],
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
    )
    with pytest.raises(ValueError, match="unknown question"):
        apply_question_answers(original, {"q0": "1", "q99": "invalid"})
    updated = apply_question_answers(original, {"q0": "2", "q1": "Other"})
    assert original.answers == {} and original.status == "pending"
    assert updated.answers == {"q0": "B", "q1": "Other"} and updated.status == "answered"
    original.expires_at = datetime.now(UTC) - timedelta(seconds=1)
    with pytest.raises(ValueError, match="expired"):
        apply_question_answers(original, {"q0": "1"})


async def test_pending_question_cannot_fork_or_resume_with_missing_ledger(tmp_path):
    path = tmp_path / "sessions.db"
    scope = MemoryScope(workspace=str(tmp_path))
    storage, questions = SQLiteStorage(path=path), QuestionStore(path)
    adapter = MockAdapter("mock", scripts=[question_turn()])
    agent = agent_for(storage, questions, scope, adapter)
    try:
        [event async for event in agent.run(RunRequest(session_id="s", prompt="task"))]
        with pytest.raises(ConfigurationError, match="original session"):
            await fork_session(storage, "s", new_session_id="fork")
        assert await storage.get("fork") is None
        with questions.db:
            questions.db.execute("DELETE FROM clarifications")
        with pytest.raises(ConfigurationError, match="original database"):
            [event async for event in agent.resume("s")]
        assert len(adapter.calls) == 1
    finally:
        questions.close()
        await storage.close()
