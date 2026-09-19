from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

from harness.cli import __main__ as cli_main
from harness.cli import gateway_runtime, run_commands
from harness.cli.config import HarnessConfig
from harness.cli.gateway_hooks import WhatsAppNotificationHook
from harness.cli.gateway_runtime import GatewayApprovalPolicy
from harness.core import (
    Agent,
    ApprovalDecision,
    Capabilities,
    Done,
    FailoverPolicy,
    InboxApprovalHandler,
    Message,
    PendingApproval,
    RunRequest,
    Session,
    ToolCall,
    ToolRegistry,
    ToolResult,
)
from harness.core.gateway_models import GatewayMessage, GatewayRuntimeBinding, default_gateway_root
from harness.core.gateway_router import dispatch_gateway_message, is_gateway_control_message
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.memory import MemoryEntry, MemoryScope
from harness.core.scheduler.store import SchedulerStore
from harness.storage.sqlite import SQLiteStorage


def _owner(tmp_path, *, session_id="runtime"):
    sessions = GatewaySessionStore(root=default_gateway_root(tmp_path))
    session = sessions.get_or_create_session(transport="test", user_id="owner", thread_id="thread")
    binding = GatewayRuntimeBinding(
        session_id=session_id,
        gateway_session_id=session.id,
        transport="test",
        user_id="owner",
        thread_id="thread",
        provider="fake",
        model="original-model",
    )
    sessions.bind_runtime_session(binding)
    return sessions, binding


async def _dispatch(tmp_path, storage, text, **kwargs):
    return await dispatch_gateway_message(
        cwd=tmp_path,
        session_store=GatewaySessionStore(root=default_gateway_root(tmp_path)),
        scheduler_store=SchedulerStore(root=tmp_path / ".harness/scheduler"),
        approval_store=storage,
        message=GatewayMessage(
            id="message",
            transport=kwargs.pop("transport", "test"),
            user_id=kwargs.pop("user_id", "owner"),
            thread_id=kwargs.pop("thread_id", "thread"),
            text=text,
        ),
        **kwargs,
    )


@pytest.mark.parametrize(
    "actor", [{"user_id": "other"}, {"thread_id": "other"}, {"transport": "other"}]
)
async def test_remote_decisions_and_listing_are_scoped_to_original_actor(tmp_path, actor):
    _owner(tmp_path)
    storage = SQLiteStorage(path=tmp_path / "db")
    try:
        approval = await storage.create_approval(
            PendingApproval(
                session_id="runtime",
                tool_call_id="call",
                tool_name="shell",
                arguments={"command": "private action"},
            )
        )
        for command in ("approve", "deny"):
            reply, _ = await _dispatch(tmp_path, storage, f"{command} {approval.id}", **actor)
            assert reply.status == "error"
            assert "private action" not in reply.text
        listing, _ = await _dispatch(tmp_path, storage, "approvals", **actor)
        assert approval.id not in listing.text
        current = await storage.get_approval(approval.id)
        assert current is not None and current.status == "pending"
    finally:
        await storage.close()


async def test_unbound_terminal_approval_cannot_be_granted_from_gateway(tmp_path):
    storage = SQLiteStorage(path=tmp_path / "db")
    try:
        approval = await storage.create_approval(
            PendingApproval(session_id="terminal", tool_call_id="call", tool_name="shell")
        )
        reply, _ = await _dispatch(tmp_path, storage, f"approve {approval.id}")
        assert reply.status == "error"
        current = await storage.get_approval(approval.id)
        assert current is not None and current.status == "pending"
    finally:
        await storage.close()


@pytest.mark.parametrize("decision", ["deny", "expire"])
async def test_denied_and_expired_actions_never_resume(tmp_path, decision):
    _owner(tmp_path)
    storage = SQLiteStorage(path=tmp_path / "db")
    resumed = []

    async def resume(binding):
        resumed.append(binding.session_id)
        return "unexpected"

    try:
        requested = datetime.now(UTC) - timedelta(minutes=16 if decision == "expire" else 0)
        approval = await storage.create_approval(
            PendingApproval(
                session_id="runtime", tool_call_id="call", tool_name="shell", requested_at=requested
            )
        )
        command = "approve" if decision == "expire" else "deny"
        reply, _ = await _dispatch(
            tmp_path, storage, f"{command} {approval.id}", resume_approval=resume
        )
        assert reply.status == ("expired" if decision == "expire" else "ok")
        repeated, _ = await _dispatch(
            tmp_path, storage, f"approve {approval.id}", resume_approval=resume
        )
        assert repeated.status == "duplicate"
        assert not resumed
    finally:
        await storage.close()


async def test_concurrent_and_duplicate_decisions_resume_once(tmp_path):
    _owner(tmp_path)
    storage = SQLiteStorage(path=tmp_path / "db")
    started = asyncio.Event()
    finish = asyncio.Event()
    resumed = []

    async def resume(binding):
        resumed.append(binding)
        started.set()
        await finish.wait()
        return "completed"

    try:
        approval = await storage.create_approval(
            PendingApproval(session_id="runtime", tool_call_id="call", tool_name="shell")
        )
        first = asyncio.create_task(
            _dispatch(tmp_path, storage, f"approve {approval.id}", resume_approval=resume)
        )
        await asyncio.wait_for(started.wait(), timeout=5)
        concurrent, _ = await _dispatch(
            tmp_path, storage, f"approve {approval.id}", resume_approval=resume
        )
        assert concurrent.status == "busy"
        finish.set()
        assert (await first)[0].status == "ok"
        duplicate, _ = await _dispatch(
            tmp_path, storage, f"approve {approval.id}", resume_approval=resume
        )
        assert duplicate.status == "duplicate"
        assert len(resumed) == 1
        assert resumed[0].model == "original-model"
    finally:
        finish.set()
        await storage.close()


def test_gateway_policy_does_not_inherit_blanket_auto_approval():
    class Tool:
        name = "mutate"
        approval: ApprovalDecision = "auto"
        effect_scope = "workspace_durable"
        description = ""

        def __init__(self):
            self.parameters_schema: dict[str, Any] = {}

        async def __call__(self, call):
            raise AssertionError("policy inspection must not call the tool")

    policy = GatewayApprovalPolicy(per_tool={"mutate": "auto"})
    assert policy.decide(Tool(), session_overrides={"mutate": "auto"}) == "prompt"
    assert GatewayApprovalPolicy(per_tool={"mutate": "deny"}).decide(Tool()) == "deny"


async def test_gateway_queue_restart_approve_runs_exact_original_action_once(tmp_path, monkeypatch):
    turns = []
    calls = []

    class AppendTool:
        name = "append_marker"
        description = "Append one marker"
        approval: ApprovalDecision = "auto"
        effect_scope = "workspace_durable"

        def __init__(self):
            self.parameters_schema: dict[str, Any] = {
                "type": "object",
                "properties": {"value": {"type": "string"}},
            }

        async def __call__(self, call):
            calls.append(dict(call.arguments))
            return ToolResult(tool_call_id=call.id, name=self.name, content="marker appended")

    class FakeAdapter:
        name = "fake"

        async def capabilities(self):
            return Capabilities(streaming=True, tool_use=True)

        async def cancel(self, session_id):
            pass

        async def stream(self, **kwargs):
            if any(
                item.role == "tool" and item.content == "marker appended"
                for item in kwargs["messages"]
            ):
                yield Done(
                    final_message=Message(role="assistant", content="Completed original action.")
                )
            else:
                yield Done(
                    final_message=Message(
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                id="call-original",
                                name="append_marker",
                                arguments={"value": "exact original"},
                            )
                        ],
                    )
                )

    async def turn(**kwargs):
        turns.append((kwargs["session_id"], kwargs["model"]))
        storage = SQLiteStorage(path=tmp_path / ".harness/harness.db")
        tools = ToolRegistry()
        tools.register(AppendTool())
        try:
            agent = Agent(
                adapters={"fake": FakeAdapter()},
                tools=tools,
                storage=storage,
                approval_store=storage,
                approval_handler=InboxApprovalHandler(approval_store=storage),
                approval_policy=GatewayApprovalPolicy(),
                failover=FailoverPolicy(chain=["fake"], max_attempts=1),
                default_model=kwargs["model"],
                default_cwd=str(tmp_path),
                pause_on_approval=True,
            )
            text = ""
            async for event in agent.run(
                RunRequest(
                    prompt=kwargs["prompt"],
                    model=kwargs["model"],
                    session_id=kwargs["session_id"],
                    max_steps=3,
                )
            ):
                if isinstance(event, Done) and event.final_message is not None:
                    text = event.final_message.content or ""
            return text
        finally:
            await storage.close()

    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", turn)
    monkeypatch.setattr(
        gateway_runtime,
        "_load_cli_config",
        lambda _: HarnessConfig(default_provider="fake", default_model="original-model"),
    )
    monkeypatch.setattr(gateway_runtime, "_gateway_env_override", lambda _: "")
    monkeypatch.setattr(gateway_runtime, "_load_hooks", lambda _: ())
    first = await gateway_runtime._run_gateway_receive_payload(
        working_dir=tmp_path,
        message="Append exactly one marker",
        transport="test",
        user_id="owner",
        thread_id="thread",
    )
    reply = first["reply"]
    assert isinstance(reply, dict)
    assert reply["status"] == "approval_required"
    assert '"value": "exact original"' in reply["text"]
    approval_id = reply["data"]["approval_ids"][0]
    assert calls == []
    assert len(turns) == 1

    # New stores on each receive simulate a gateway restart; changing the
    # defaults must not send the approved action to another runtime session.
    monkeypatch.setattr(
        gateway_runtime,
        "_load_cli_config",
        lambda _: HarnessConfig(default_provider="fake", default_model="changed-model"),
    )
    listed = await gateway_runtime._run_gateway_receive_payload(
        working_dir=tmp_path,
        message="approvals",
        transport="test",
        user_id="owner",
        thread_id="thread",
    )
    listed_reply = listed["reply"]
    assert isinstance(listed_reply, dict)
    assert approval_id in listed_reply["text"]
    approved = await gateway_runtime._run_gateway_receive_payload(
        working_dir=tmp_path,
        message=f"approve {approval_id}",
        transport="test",
        user_id="owner",
        thread_id="thread",
    )
    approved_reply = approved["reply"]
    assert isinstance(approved_reply, dict)
    assert "Completed original action" in approved_reply["text"]
    duplicate = await gateway_runtime._run_gateway_receive_payload(
        working_dir=tmp_path,
        message=f"approve {approval_id}",
        transport="test",
        user_id="owner",
        thread_id="thread",
    )
    duplicate_reply = duplicate["reply"]
    assert isinstance(duplicate_reply, dict)
    assert duplicate_reply["status"] == "duplicate"
    assert calls == [{"value": "exact original"}]
    assert len(turns) == 2 and turns[0] == turns[1]


def test_whatsapp_approval_notification_uses_bound_thread_not_latest_user(tmp_path, monkeypatch):
    sessions = GatewaySessionStore(root=default_gateway_root(tmp_path))
    session = sessions.get_or_create_session(
        transport="whatsapp", user_id="owner", thread_id="original@g.us"
    )
    sessions.bind_runtime_session(
        GatewayRuntimeBinding(
            session_id="runtime",
            gateway_session_id=session.id,
            transport="whatsapp",
            user_id="owner",
            thread_id="original@g.us",
            provider="fake",
            model="fake",
        )
    )
    sessions.get_or_create_session(
        transport="whatsapp", user_id="unrelated", thread_id="wrong-thread"
    )
    monkeypatch.setenv("HARNESS_WHATSAPP_NOTIFY_TO", "wrong-target")
    sent = []
    monkeypatch.setattr(
        "harness.cli.gateway_hooks.send_whatsapp_text_message", lambda **kwargs: sent.append(kwargs)
    )
    approval = PendingApproval(
        session_id="runtime",
        tool_call_id="call",
        tool_name="shell",
        arguments={"command": "echo exact"},
    )
    WhatsAppNotificationHook().on_approval_requested(cwd=tmp_path, approval=approval)
    assert sent[0]["to"] == "original@g.us"
    assert "echo exact" in sent[0]["text"]
    assert f"deny {approval.id}" in sent[0]["text"]


def test_approval_commands_are_routed_to_control_plane():
    assert is_gateway_control_message("approvals")
    assert is_gateway_control_message("deny appr_123")


async def test_gateway_chat_wires_inbox_pause_and_restrictive_policy(tmp_path, monkeypatch):
    captured = {}
    removed = []
    agent = SimpleNamespace(tools=SimpleNamespace(unregister=removed.append), storage=object())
    monkeypatch.setattr(cli_main, "_build_agent", lambda **kwargs: agent)
    boundaries = []
    monkeypatch.setattr(
        "harness.cli.gateway_tool_boundary.install_gateway_tool_boundary",
        lambda agent, cwd, **kwargs: boundaries.append(kwargs),
    )

    async def run_once(**kwargs):
        captured.update(kwargs)
        kwargs["build_agent"]()
        return "pending"

    monkeypatch.setattr(run_commands, "run_once", run_once)
    await gateway_runtime._run_gateway_chat_turn(
        cwd=tmp_path,
        transport="test",
        user_id="owner",
        prompt="do work",
        chain=["fake"],
        model="fake",
        session_id="session",
        max_steps=3,
        config=HarnessConfig(),
        system_prompt="system",
    )
    assert captured["yes"] is False
    assert captured["inbox"] is True
    assert agent.pause_on_approval is True
    assert isinstance(agent.approval_policy, GatewayApprovalPolicy)
    assert agent.memory_scope.user_id == '["test","owner"]'
    assert agent.memory_store is agent.storage
    assert set(removed) == {"spawn_agents", "shell", "verify_work"}
    assert boundaries == [{"local_only": False}]


async def test_gateway_rejects_native_tool_provider_before_model_call(tmp_path, monkeypatch):
    async def unexpected(**kwargs):
        raise AssertionError("native tool adapter must not execute")

    monkeypatch.setattr(run_commands, "run_once", unexpected)
    reply = await gateway_runtime._run_gateway_chat_turn(
        cwd=tmp_path,
        transport="test",
        user_id="owner",
        prompt="do work",
        chain=["codex"],
        model="fake",
        session_id="session",
        max_steps=3,
        config=HarnessConfig(),
        system_prompt="system",
    )
    assert "outside this approval boundary" in reply


async def test_real_gateway_agent_injects_and_searches_only_its_user_scope(tmp_path, monkeypatch):
    storage = SQLiteStorage(path=tmp_path / "memory.db")
    seen = []

    class SearchAdapter:
        name = "fake"
        turns = 0

        async def capabilities(self):
            return Capabilities(streaming=True, tool_use=True)

        async def cancel(self, session_id):
            pass

        async def stream(self, **kwargs):
            self.turns += 1
            seen.extend(item.content or "" for item in kwargs["messages"])
            if self.turns == 1:
                yield Done(
                    final_message=Message(
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                id="lookup", name="search_sessions", arguments={"query": "needle"}
                            )
                        ],
                    )
                )
            else:
                result = next(
                    item.content for item in reversed(kwargs["messages"]) if item.role == "tool"
                )
                yield Done(final_message=Message(role="assistant", content=result))

    def build(**kwargs):
        return Agent(
            adapters={"fake": SearchAdapter()},
            tools=ToolRegistry(),
            storage=storage,
            approval_store=storage,
            failover=FailoverPolicy(chain=["fake"], max_attempts=1),
            default_model="fake",
            default_cwd=str(tmp_path),
        )

    async def run_once(**kwargs):
        agent = kwargs["build_agent"]()
        text = ""
        async for event in agent.run(
            RunRequest(prompt="Find needle", session_id="gateway-memory", model="fake", max_steps=3)
        ):
            if isinstance(event, Done) and event.final_message is not None:
                text = event.final_message.content or ""
        return text

    try:
        for user, label in (("owner", "own"), ("other", "foreign")):
            scope = MemoryScope(workspace=str(tmp_path), user_id=f'["test","{user}"]')
            await storage.save_scoped_memory(
                MemoryEntry(kind="user_fact", text=f"{label}-memory-needle"), scope=scope
            )
            await storage.save(
                Session(
                    id=f"{label}-conversation",
                    provider="fake",
                    model="fake",
                    cwd=tmp_path,
                    metadata={"memory_scope": scope.model_dump(mode="json")},
                    messages=[Message(role="user", content=f"{label}-transcript-needle")],
                )
            )
        monkeypatch.setattr(cli_main, "_build_agent", build)
        monkeypatch.setattr(run_commands, "run_once", run_once)
        result = await gateway_runtime._run_gateway_chat_turn(
            cwd=tmp_path,
            prompt="Find needle",
            chain=["fake"],
            model="fake",
            session_id="gateway-memory",
            max_steps=3,
            config=HarnessConfig(),
            system_prompt="system",
            transport="test",
            user_id="owner",
        )
        assert "own-memory-needle" in "\n".join(seen)
        assert "own-transcript-needle" in result
        assert "foreign-memory-needle" not in "\n".join(seen)
        assert "foreign-transcript-needle" not in result
    finally:
        await storage.close()
