"""Regression coverage for reused agents and observable terminal outcomes."""

import asyncio
from pathlib import Path

import pytest

from harness.core import (
    Agent,
    Done,
    FailoverPolicy,
    FailureStep,
    NetworkError,
    RunRequest,
    ToolRegistry,
    VerificationResult,
    VerificationStep,
)
from harness.core import activity as activity_kinds

from .conftest import MockAdapter, MockStorage, text_turn, tool_call_turn


def make_agent(tmp_path: Path, adapter: MockAdapter, **kwargs):
    store = MockStorage()
    return Agent(
        adapters={"mock": adapter},
        tools=ToolRegistry(),
        storage=store,
        failover=FailoverPolicy(chain=["mock"], max_attempts=1),
        default_model="mock",
        default_cwd=str(tmp_path),
        **kwargs,
    ), store


def add_note(call_id: str, text: str):
    return tool_call_turn(call_id=call_id, name="notes", arguments={"action": "add", "text": text})


async def collect(stream):
    return [event async for event in stream]


async def test_resume_model_override_becomes_the_persisted_model(tmp_path):
    adapter = MockAdapter("mock", scripts=[text_turn("one"), text_turn("two"), text_turn("three")])
    agent, store = make_agent(tmp_path, adapter)
    await collect(agent.run(RunRequest(prompt="start", session_id="same", model="first")))
    await collect(agent.resume("same", "switch", model="second"))
    stored = await store.get("same")
    assert stored is not None
    assert stored.model == "second"
    await collect(agent.resume("same", "continue"))
    assert [call["model"] for call in adapter.calls] == ["first", "second", "second"]


async def test_execution_context_survives_restart_without_changing_task_or_other_sessions(tmp_path):
    from harness.core.verification_structural import PublicSourceEvidenceVerifier, first_user_prompt
    from harness.storage.sqlite import SQLiteStorage

    context = (
        "General context guidance: research current package versions from official public "
        "sources when the task needs them. Private finding: use the existing reports directory."
    )
    adapter = MockAdapter("mock", scripts=[text_turn("Hello")])

    def build(store, provider):
        return Agent(
            adapters={"mock": provider},
            tools=ToolRegistry(),
            storage=store,
            failover=FailoverPolicy(chain=["mock"], max_attempts=1),
            default_model="mock",
            default_cwd=str(tmp_path),
            verifier=PublicSourceEvidenceVerifier(),
        )

    database = tmp_path / "context.db"
    store = SQLiteStorage(path=database)
    try:
        await collect(
            build(store, adapter).run(
                RunRequest(
                    prompt="hello",
                    session_id="own",
                    execution_context=context,
                )
            )
        )
        saved = await store.get("own")
        assert saved is not None and saved.status == "done"
        assert first_user_prompt(saved) == "hello"
        assert saved.metadata["execution_context"] == context
        assert all(context not in (message.content or "") for message in saved.messages)
        assert any(
            context in (message.content or "") and message.role == "system"
            for message in adapter.calls[0]["messages"]
        )
    finally:
        await store.close()

    adapter = MockAdapter(
        "mock", scripts=[text_turn("Continued"), text_turn("Other"), text_turn("Cleared")]
    )
    store = SQLiteStorage(path=database)
    try:
        agent = build(store, adapter)
        await collect(agent.resume("own", "continue"))
        assert any(context in (message.content or "") for message in adapter.calls[0]["messages"])
        await collect(agent.run(RunRequest(prompt="hello", session_id="other")))
        assert all(
            context not in (message.content or "") for message in adapter.calls[1]["messages"]
        )
        await collect(agent.resume("own", "clear context", execution_context=""))
        assert all(
            context not in (message.content or "") for message in adapter.calls[2]["messages"]
        )
        saved = await store.get("own")
        assert saved is not None and "execution_context" not in saved.metadata
        assert saved.status == "done" and first_user_prompt(saved) == "hello"
    finally:
        await store.close()


async def test_notes_survive_same_agent_resume_and_do_not_cross_sessions(tmp_path):
    adapter = MockAdapter(
        "mock",
        scripts=[
            add_note("n1", "first"),
            text_turn("ok"),
            add_note("n2", "second"),
            text_turn("ok"),
            tool_call_turn(call_id="n3", name="notes", arguments={"action": "list"}),
            text_turn("ok"),
        ],
    )
    agent, store = make_agent(tmp_path, adapter, memory_tools_enabled=True)
    await collect(agent.run(RunRequest(prompt="remember", session_id="same")))
    await collect(agent.resume("same", "remember more"))
    stored = await store.get("same")
    assert stored is not None
    assert [note.text for note in stored.notes] == ["first", "second"]
    events = await collect(agent.run(RunRequest(prompt="list notes", session_id="other")))
    assert [event.result.content for event in events if event.type == "tool_result"] == [
        "(no notes yet)"
    ]


async def test_overlapping_runs_do_not_rebind_tools_until_first_finishes(tmp_path):
    entered = asyncio.Event()
    release = asyncio.Event()

    class DelayedAdapter(MockAdapter):
        async def _stream(self):
            if len(self.calls) == 1:
                entered.set()
                await release.wait()
            async for event in super()._stream():
                yield event

    adapter = DelayedAdapter(
        "mock",
        scripts=[
            add_note("n1", "first"),
            text_turn("ok"),
            add_note("n2", "second"),
            text_turn("ok"),
        ],
    )
    agent, store = make_agent(tmp_path, adapter, memory_tools_enabled=True)
    first = asyncio.create_task(collect(agent.run(RunRequest(prompt="a", session_id="a"))))
    await entered.wait()
    second = asyncio.create_task(collect(agent.run(RunRequest(prompt="b", session_id="b"))))
    try:
        await asyncio.sleep(0)
        assert len(adapter.calls) == 1
    finally:
        release.set()
        await asyncio.gather(first, second)
    a, b = await store.get("a"), await store.get("b")
    assert a is not None and b is not None
    assert [note.text for note in a.notes] == ["first"]
    assert [note.text for note in b.notes] == ["second"]


class RejectVerifier:
    name = "reject"

    async def verify(self, **kwargs):
        return VerificationResult(
            can_finish=False, reason="Required check failed", verifier_name=self.name
        )


async def test_exhausted_verification_persists_failure(tmp_path):
    agent, store = make_agent(
        tmp_path,
        MockAdapter("mock", scripts=[text_turn("claimed done")]),
        verifier=RejectVerifier(),
        max_repair_attempts=0,
    )
    events = await collect(agent.run(RunRequest(prompt="work", session_id="failed-check")))
    stored = await store.get("failed-check")
    assert stored is not None and stored.status == "failed"
    assert any(event.type == "verification" and not event.result.can_finish for event in events)
    assert any(event.type == "error" and event.kind == "verification" for event in events)
    activity = agent._ephemeral_activity[stored.id]
    assert activity[-1].kind == activity_kinds.AGENT_RUN_FAILED


async def test_iterator_exposes_failure_and_verification(tmp_path):
    agent, _ = make_agent(tmp_path, MockAdapter("mock", error=NetworkError("offline")))
    async with agent.iter(RunRequest(prompt="work")) as run:
        steps = [step async for step in run]
    failures = [step for step in steps if isinstance(step, FailureStep)]
    assert len(failures) == 1
    assert failures[0].kind == "network" and failures[0].error == "offline"

    agent, _ = make_agent(
        tmp_path,
        MockAdapter("mock", scripts=[text_turn("claimed done")]),
        verifier=RejectVerifier(),
        max_repair_attempts=0,
    )
    async with agent.iter(RunRequest(prompt="work")) as run:
        steps = [step async for step in run]
    verdicts = [step for step in steps if isinstance(step, VerificationStep)]
    assert len(verdicts) == 1 and not verdicts[0].result.can_finish
    assert type(steps[-1]).__name__ == "FailureStep"


async def test_iterator_close_settles_run_and_allows_reuse(tmp_path):
    agent, store = make_agent(tmp_path, MockAdapter("mock", scripts=[text_turn("next run")]))
    async with agent.iter(RunRequest(prompt="stop", session_id="stopped")) as run:
        async for _step in run:
            break
    stored = await store.get("stopped")
    assert stored is not None and stored.status == "cancelled"
    events = await asyncio.wait_for(
        collect(agent.run(RunRequest(prompt="continue", session_id="new"))), timeout=1
    )
    assert any(isinstance(event, Done) for event in events)


async def test_iterator_guardrail_stop_is_a_failed_outcome(tmp_path):
    from harness.core.guardrails import GuardrailResult

    class Stop:
        name = "stop"
        mode = "blocking"

        async def __call__(self, messages):
            return GuardrailResult(tripped=True, reason="stop requested")

    agent, store = make_agent(tmp_path, MockAdapter("mock"), guardrails=[Stop()])
    async with agent.iter(RunRequest(prompt="work", session_id="guard")) as run:
        steps = [step async for step in run]
    assert any(isinstance(step, FailureStep) and step.kind == "guardrail" for step in steps)
    stored = await store.get("guard")
    assert stored is not None and stored.status == "failed"


async def test_cancel_during_verification_marks_session_cancelled(tmp_path):
    entered = asyncio.Event()

    class WaitingVerifier:
        name = "waiting"

        async def verify(self, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    agent, store = make_agent(
        tmp_path, MockAdapter("mock", scripts=[text_turn("done")]), verifier=WaitingVerifier()
    )
    task = asyncio.create_task(collect(agent.run(RunRequest(prompt="work", session_id="cancel"))))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    stored = await store.get("cancel")
    assert stored is not None and stored.status == "cancelled"
