"""Actual runtime/service coverage for autonomous, identity-bound read access."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from rich.console import Console

from harness.cli.common import _build_tools
from harness.cli.config import HarnessConfig
from harness.cli.gateway_runtime import GatewayApprovalPolicy
from harness.cli.runtime_agent import build_agent
from harness.core import (
    Agent,
    Capabilities,
    Done,
    FailoverPolicy,
    InboxApprovalHandler,
    Message,
    RunRequest,
    Session,
    ToolCall,
    ToolRegistry,
    ToolResultEvent,
)
from harness.core.memory import MemoryEntry, MemoryScope
from harness.server.delegation import DelegationTool, LocalDelegationToolset
from harness.storage.sqlite import SQLiteStorage


class ScriptAdapter:
    name = "fake"

    def __init__(self, actions: list[tuple[str, dict[str, Any]]]):
        self.actions = actions
        self.calls: list[dict[str, Any]] = []

    async def capabilities(self):
        return Capabilities(tool_use=True)

    async def cancel(self, session_id):
        pass

    async def stream(self, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        if index < len(self.actions):
            name, arguments = self.actions[index]
            yield Done(
                final_message=Message(
                    role="assistant",
                    tool_calls=[ToolCall(id=f"call-{index}", name=name, arguments=arguments)],
                )
            )
        else:
            yield Done(final_message=Message(role="assistant", content="Discovery complete"))


async def seed_history(storage, scope, session_id, content):
    await storage.save(
        Session(
            id=session_id,
            provider="fake",
            model="test",
            cwd=Path(scope.workspace),
            metadata={"memory_scope": scope.model_dump(mode="json")},
            messages=[Message(role="user", content=content)],
        )
    )


@pytest.mark.parametrize("decision", [None, "prompt", "deny"])
async def test_local_memory_writes_follow_configured_approval_policy(tmp_path, decision):
    storage = SQLiteStorage(path=tmp_path / "sessions.db")
    adapter = ScriptAdapter([("memory", {"action": "add", "text": "local telescope fact"})])
    config = HarnessConfig(approval={} if decision is None else {"memory": decision})
    agent = build_agent(
        chain=["fake"],
        base_url=None,
        model="test",
        storage=storage,
        cwd=tmp_path,
        config=config,
        yes=False,
        inbox=True,
        pause_on_approval=True,
        approval_store=storage,
        build_adapter=lambda *args, **kwargs: adapter,
        build_tools=lambda path: ToolRegistry(),
        build_search_fn=lambda: None,
        console=Console(quiet=True),
        memory_store=storage,
        auxiliary_tools_enabled=False,
        project_context_enabled=False,
        skip_builtin_verify_before_done=True,
    )
    try:
        async with agent:
            events = [
                event
                async for event in agent.run(
                    RunRequest(prompt="Remember this task fact", session_id="local", max_steps=3)
                )
            ]
        memories = await storage.search_scoped_memory(
            "telescope", scope=MemoryScope(workspace=str(tmp_path))
        )
        approvals = await storage.list_approvals(session_id="local")
        session = await storage.get("local")
        assert session is not None
        if decision == "prompt":
            assert not memories and session.status == "paused"
            assert len(approvals) == 1
            assert approvals[0].tool_name == "memory" and approvals[0].status == "pending"
        elif decision == "deny":
            assert not memories and not approvals and session.status == "done"
            assert any(
                isinstance(event, ToolResultEvent)
                and event.result.is_error
                and "denied" in event.result.content.lower()
                for event in events
            )
        else:
            assert [memory.text for memory in memories] == ["local telescope fact"]
            assert not approvals and session.status == "done"
    finally:
        await agent.aclose()
        await storage.close()


async def test_actual_agent_recall_is_automatic_but_memory_mutation_still_pauses(tmp_path):
    storage = SQLiteStorage(path=tmp_path / "sessions.db")
    scope = MemoryScope(workspace=str(tmp_path), user_id='["telegram","alice"]')
    adapter = ScriptAdapter(
        [
            ("recall_memory", {"action": "search", "query": "telescope"}),
            ("recall_memory", {"action": "get", "id": "foreign"}),
            ("memory", {"action": "add", "text": "new fact requires review"}),
        ]
    )
    agent = Agent(
        adapters={"fake": adapter},
        tools=ToolRegistry(),
        storage=storage,
        memory_store=storage,
        memory_scope=scope,
        approval_store=storage,
        approval_handler=InboxApprovalHandler(approval_store=storage),
        approval_policy=GatewayApprovalPolicy(),
        pause_on_approval=True,
        failover=FailoverPolicy(chain=["fake"]),
        default_model="test",
        default_cwd=str(tmp_path),
    )
    try:
        await storage.save_scoped_memory(
            MemoryEntry(kind="project_fact", text="own telescope fact"), scope=scope
        )
        await storage.save_scoped_memory(
            MemoryEntry(id="foreign", kind="project_fact", text="foreign telescope secret"),
            scope=scope.model_copy(update={"user_id": "bob"}),
        )
        events = [
            event
            async for event in agent.run(
                RunRequest(prompt="Recall prior facts", session_id="remote", max_steps=5)
            )
        ]
        results = [event.result for event in events if isinstance(event, ToolResultEvent)]
        assert json.loads(results[0].content)["memories"][0]["text"] == "own telescope fact"
        assert not results[0].is_error
        assert results[1].is_error and "foreign telescope secret" not in results[1].content
        assert all(
            "foreign telescope secret" not in (message.content or "")
            for call in adapter.calls
            for message in call["messages"]
        )
        approvals = await storage.list_approvals(session_id="remote")
        assert [approval.tool_name for approval in approvals] == ["memory"]
        assert approvals[0].status == "pending"
        assert not await storage.search_scoped_memory("new fact", scope=scope)
        saved = await storage.get("remote")
        assert saved is not None and saved.status == "paused"
    finally:
        await agent.aclose()
        await storage.close()


async def test_managed_child_discovers_own_history_without_parent_or_foreign_scope(
    tmp_path, monkeypatch
):
    child_adapter = ScriptAdapter(
        [
            ("recall_memory", {"action": "list"}),
            ("search_sessions", {"query": "telescope"}),
            ("conversation_window", {"session_id": "child-history"}),
            ("conversation_window", {"session_id": "parent-history"}),
        ]
    )
    parent_adapter = ScriptAdapter([])
    adapters = iter([parent_adapter, child_adapter])
    scopes = []

    class SeededToolset(LocalDelegationToolset):
        def __init__(self, database, workspace, agent_builder, **kwargs):
            async def seeded(context):
                own_scope = MemoryScope(workspace=str(context.workspace), user_id="local")
                scopes.append(own_scope)
                for label, scope in (
                    ("child", own_scope),
                    ("parent", MemoryScope(workspace=str(workspace), user_id="local")),
                    ("foreign", own_scope.model_copy(update={"user_id": "bob"})),
                ):
                    await context.storage.save_scoped_memory(
                        MemoryEntry(kind="project_fact", text=f"{label} telescope memory"),
                        scope=scope,
                    )
                    await seed_history(
                        context.storage, scope, f"{label}-history", f"{label} telescope transcript"
                    )
                return agent_builder(context)

            super().__init__(database, workspace, seeded, **kwargs)

    monkeypatch.setattr("harness.server.delegation.LocalDelegationToolset", SeededToolset)
    storage = SQLiteStorage(path=tmp_path / "parent.db")
    agent = build_agent(
        chain=["fake"],
        base_url=None,
        model="test",
        storage=storage,
        cwd=tmp_path,
        config=HarnessConfig(delegation_enabled=True),
        yes=False,
        build_adapter=lambda *args, **kwargs: next(adapters),
        build_tools=lambda path: _build_tools(path, config=HarnessConfig()),
        build_search_fn=lambda: None,
        console=Console(quiet=True),
        memory_store=storage,
        skip_builtin_verify_before_done=True,
        project_context_enabled=False,
    )
    try:
        async with agent:
            [event async for event in agent.run(RunRequest(prompt="Coordinate", session_id="p"))]
            tool = agent.tools.get("delegate")
            assert isinstance(tool, DelegationTool)
            result = await tool(
                ToolCall(
                    id="delegate",
                    name="delegate",
                    arguments={"prompt": "Recall telescope work", "max_steps": 6},
                )
            )
            assert not result.is_error
            job = json.loads(result.content)
            service = tool.manager.service
            async with asyncio.timeout(10):
                async for _ in service.events(tool.owner, job["run"]["id"]):
                    pass
            status = await tool.manager.status(tool.owner, job["id"], tool.parent_session_id)
            assert status["state"] == "completed" and status["approvals"] == []
            saved = await service.storage.get(job["session_id"])
            assert saved is not None
            results = [
                message.content or "" for message in saved.messages if message.role == "tool"
            ]
            assert len(results) == 4
            assert json.loads(results[0])["memories"][0]["text"] == "child telescope memory"
            matches = {item["session_id"] for item in json.loads(results[1])["sessions"]}
            assert "child-history" in matches
            assert matches <= {"child-history", job["session_id"]}
            assert "child telescope transcript" in results[2]
            assert results[3] == "Conversation not found"
            assert scopes and scopes[0].workspace != str(tmp_path)
            seen = "\n".join(
                message.content or ""
                for call in child_adapter.calls
                for message in call["messages"]
            )
            assert "parent telescope" not in seen and "foreign telescope" not in seen
    finally:
        await storage.close()
