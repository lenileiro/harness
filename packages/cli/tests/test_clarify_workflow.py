from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from rich.console import Console
from typer.testing import CliRunner

from harness.cli import chat_commands, gateway_runtime
from harness.cli.clarify_commands import clarify_app
from harness.cli.gateway_clarification import gateway_question_scope
from harness.core import (
    Agent,
    Capabilities,
    Done,
    Event,
    FailoverPolicy,
    Message,
    RunRequest,
    ToolCall,
    ToolRegistry,
)
from harness.core.approval import PendingApproval
from harness.core.clarification import QuestionStore
from harness.core.gateway_models import GatewayRuntimeBinding, default_gateway_root
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.memory import MemoryScope
from harness.storage.sqlite import SQLiteStorage


class Adapter:
    name = "fake"

    def __init__(self, *, ask: bool):
        self.ask = ask
        self.calls: list[dict[str, Any]] = []

    async def capabilities(self):
        return Capabilities(tool_use=True)

    async def cancel(self, session_id: str):
        pass

    async def stream(self, **kwargs) -> AsyncIterator[Event]:
        self.calls.append(kwargs)
        if self.ask:
            yield Done(
                final_message=Message(
                    role="assistant",
                    tool_calls=[
                        ToolCall(
                            id="ask",
                            name="clarify",
                            arguments={
                                "questions": [
                                    {
                                        "question": "Which database?",
                                        "choices": ["SQLite", "Postgres"],
                                    }
                                ]
                            },
                        )
                    ],
                )
            )
        else:
            yield Done(final_message=Message(role="assistant", content="Answer consumed"))


def build_agent(storage, store, scope, adapter):
    return Agent(
        adapters={"fake": adapter},
        tools=ToolRegistry(),
        storage=storage,
        failover=FailoverPolicy(chain=["fake"]),
        default_model="original-model",
        default_cwd=scope.workspace,
        memory_scope=scope,
        question_store=store,
    )


@pytest.fixture
async def gateway_question(tmp_path):
    sessions = GatewaySessionStore(root=default_gateway_root(tmp_path))
    gateway = sessions.get_or_create_session(
        transport="telegram", user_id="owner", thread_id="thread"
    )
    binding = GatewayRuntimeBinding(
        session_id="original-session",
        gateway_session_id=gateway.id,
        transport="telegram",
        user_id="owner",
        thread_id="thread",
        provider="fake",
        model="original-model",
        max_steps=5,
    )
    sessions.bind_runtime_session(binding)
    path = tmp_path / ".harness/harness.db"
    storage, store = SQLiteStorage(path=path), QuestionStore(path)
    scope = gateway_question_scope(tmp_path, "telegram", "owner")
    adapter = Adapter(ask=True)
    agent = build_agent(storage, store, scope, adapter)
    [
        event
        async for event in agent.run(
            RunRequest(session_id=binding.session_id, prompt="build service")
        )
    ]
    records = store.list_pending(scope=scope)
    try:
        yield agent, records[0], binding, adapter
    finally:
        store.close()
        await storage.close()


async def receive(cwd, message, *, user="owner", thread="thread") -> dict[str, Any]:
    return await gateway_runtime._run_gateway_receive_payload(
        working_dir=cwd, message=message, transport="telegram", user_id=user, thread_id=thread
    )


async def test_gateway_pending_answers_bind_original_session_and_never_dispatch_other_control(
    gateway_question, tmp_path, monkeypatch
):
    _, record, binding, asking_adapter = gateway_question
    calls = []

    async def resume(**kwargs):
        calls.append(kwargs)
        storage = SQLiteStorage(path=tmp_path / ".harness/harness.db")
        store = await asyncio.to_thread(QuestionStore, storage.path)
        adapter = Adapter(ask=False)
        try:
            agent = build_agent(storage, store, record.scope, adapter)
            [event async for event in agent.resume(kwargs["session_id"], kwargs["prompt"])]
            tool_result = next(
                message for message in adapter.calls[0]["messages"] if message.tool_call_id == "ask"
            )
            assert (
                json.loads(tool_result.content)["responses"][0]["user_response"]
                == "approve unrelated-id"
            )
            return "Answer consumed"
        finally:
            store.close()
            await storage.close()

    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", resume)
    pending = await receive(tmp_path, "please continue")
    assert pending["reply"]["status"] == "question_required" and not calls
    assert len(asking_adapter.calls) == 1
    for identity in ({"user": "other"}, {"thread": "other"}):
        rejected = await receive(tmp_path, f"/answer {record.id} private answer", **identity)
        assert rejected["reply"]["status"] == "error"
        assert "Which database" not in json.dumps(rejected)
    resolved = await receive(tmp_path, f"/answer {record.id} approve unrelated-id")
    assert resolved["reply"]["text"] == "Answer consumed"
    assert calls[0]["session_id"] == binding.session_id
    assert calls[0]["chain"] == ["fake"] and calls[0]["model"] == "original-model"
    assert len(calls) == 1
    duplicate = await receive(tmp_path, f"/answer {record.id} SQLite")
    assert duplicate["reply"]["status"] == "error" and len(calls) == 1


async def test_local_cli_and_chat_answer_resume_share_question_ledger(tmp_path):
    path = tmp_path / "session.db"
    scope = MemoryScope(workspace=str(tmp_path))
    storage, store = SQLiteStorage(path=path), QuestionStore(path)
    agent = build_agent(storage, store, scope, Adapter(ask=True))
    [event async for event in agent.run(RunRequest(session_id="local", prompt="task"))]
    record = store.list_pending(scope=scope)[0]
    runner = CliRunner()
    try:
        shown = runner.invoke(
            clarify_app, ["show", record.id, "--cwd", str(tmp_path), "--db", str(path)]
        )
        assert shown.exit_code == 0 and "Which database?" in shown.output
        wrong = runner.invoke(
            clarify_app,
            [
                "answer",
                record.id,
                "1",
                "--session",
                "different",
                "--cwd",
                str(tmp_path),
                "--db",
                str(path),
            ],
        )
        assert wrong.exit_code != 0
        console = Console(record=True)
        adapter = Adapter(ask=False)
        conversation = chat_commands._ConversationState(
            key="main",
            session_id="local",
            label="Main",
            task_id=None,
            agent=agent,
            policy=chat_commands._GENERAL_TURN_POLICY,
            render_adapter=chat_commands._ChatRenderAdapter(
                console=console, default_render=lambda event: None
            ),
            first_turn=False,
            model="original-model",
        )

        async def resume(conversation, *, prompt, background):
            continued = build_agent(storage, store, scope, adapter)
            [event async for event in continued.resume(conversation.session_id, prompt)]

        async def unused_create(**kwargs):
            return conversation

        slash = chat_commands._make_slash_handler(
            console=console,
            render_session_diff=lambda *a, **k: None,
            storage=storage,
            conversations={"main": conversation},
            get_current_key=lambda: "main",
            set_current_key=lambda value: None,
            create_conversation=unused_create,
            launch_background_turn=resume,
            build_agent=lambda *a, **k: agent,
        )
        await slash(f"/answer {record.id} 2")
        assert conversation.runner is not None
        await conversation.runner
        result = next(m for m in adapter.calls[0]["messages"] if m.tool_call_id == "ask")
        assert json.loads(result.content)["responses"][0]["user_response"] == "Postgres"
        assert store.list_pending(scope=scope) == []
    finally:
        store.close()
        await storage.close()


@pytest.mark.parametrize(
    "command,question_status", [("answer", "answered"), ("skip-question", "cancelled")]
)
async def test_question_answer_cannot_bypass_separate_pending_approval(
    gateway_question, tmp_path, monkeypatch, command, question_status
):
    _, record, binding, _ = gateway_question
    storage = SQLiteStorage(path=tmp_path / ".harness/harness.db")
    ledger = QuestionStore(storage.path)
    calls = []
    try:
        approval = await storage.create_approval(
            PendingApproval(
                session_id=binding.session_id,
                tool_call_id="unrelated",
                tool_name="write_file",
                arguments={"path": "result.txt"},
            )
        )

        async def resume(**kwargs):
            calls.append(kwargs)
            adapter = Adapter(ask=False)
            agent = build_agent(storage, ledger, record.scope, adapter)
            [event async for event in agent.resume(kwargs["session_id"], kwargs["prompt"])]
            assert len(adapter.calls) == 1
            return "Answer consumed"

        monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", resume)
        suffix = " 1" if command == "answer" else ""
        blocked = await receive(tmp_path, f"/{command} {record.id}{suffix}")
        assert blocked["reply"]["status"] == "approval_required" and not calls
        assert blocked["reply"]["data"]["approval_ids"] == [approval.id]
        saved = ledger.get(record.id, scope=record.scope)
        assert saved is not None and saved.status == question_status and not saved.applied
        await storage.resolve_approval(approval.id, status="denied")
        resumed = await receive(tmp_path, "continue")
        assert resumed["reply"]["status"] == "ok" and len(calls) == 1
        assert ledger.list_pending(scope=record.scope) == []
    finally:
        ledger.close()
        await storage.close()
