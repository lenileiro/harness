from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from harness.cli import runtime_agent
from harness.cli.config import HarnessConfig
from harness.cli.runtime_helpers import AUTONOMOUS_CONTEXT_POLICY
from harness.core import (
    Agent,
    ApprovalDecision,
    ApprovalPolicy,
    Capabilities,
    Done,
    Event,
    Message,
    TextDelta,
    ToolCall,
    ToolRegistry,
    ToolResult,
)
from harness.storage.memory import InMemoryStorage
from harness.tools.execution import ExecutionConfig
from harness.tools.execution import backend as execution_backend
from harness.tools.fs import GlobTool, ListDirTool, ReadFileTool, WriteFileTool
from harness.tools.shell import ShellTool
from harness.tools.web import FetchUrlTool, WebSearchTool


class ResearchAdapter:
    name = "fake"

    def __init__(self, *, escape: bool = False):
        self.calls: list[dict[str, Any]] = []
        self.escape = escape

    async def capabilities(self):
        return Capabilities(tool_use=True)

    async def cancel(self, session_id: str):
        pass

    async def stream(self, **kwargs: Any) -> AsyncIterator[Event]:
        self.calls.append(kwargs)
        names = {schema["function"]["name"] for schema in kwargs["tools"]}
        planner, worker = "create_work_item" in names, "complete_work_item" in names
        turn = len(self.calls)
        calls = []
        if turn == 1:
            if "read_file" in names:
                calls.append(
                    ToolCall(id="local", name="read_file", arguments={"path": "AGENTS.md"})
                )
            if "web_search" in names:
                calls.append(ToolCall(id="search", name="web_search", arguments={"query": "guide"}))
            if "fetch_url" in names:
                calls.append(
                    ToolCall(
                        id="source",
                        name="fetch_url",
                        arguments={"url": "https://docs.example.org/guide"},
                    )
                )
            if self.escape and worker:
                calls.append(
                    ToolCall(id="escape", name="read_file", arguments={"path": "../private.txt"})
                )
        elif turn == 2 and planner:
            calls.append(
                ToolCall(id="create", name="create_work_item", arguments={"title": "Read guide"})
            )
        elif turn == 2 and worker:
            calls.append(
                ToolCall(
                    id="complete",
                    name="complete_work_item",
                    arguments={"summary": "Repository and source inspected"},
                )
            )
        if calls:
            yield Done(final_message=Message(role="assistant", tool_calls=calls))
        else:
            yield TextDelta(text="Research report" if not planner and not worker else "Done")
            yield Done(final_message=Message(role="assistant", content="Done"))


def make_tool(
    tmp_path, monkeypatch, client, *, include_web=True, policy=None, escape=False, extras=()
):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    adapters: list[ResearchAdapter] = []
    agents: list[OwnedAgent] = []

    class OwnedAgent(Agent):
        closed = False

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            agents.append(self)

        async def aclose(self):
            await super().aclose()
            self.closed = True

    monkeypatch.setattr(runtime_agent, "Agent", OwnedAgent)

    def build_tools(cwd):
        assert cwd == tmp_path
        tools = ToolRegistry()
        for tool in (
            ReadFileTool(cwd=cwd),
            ListDirTool(cwd=cwd),
            GlobTool(cwd=cwd),
            WriteFileTool(cwd=cwd),
        ):
            tools.register(tool)
        if include_web:
            tools.register(FetchUrlTool(client=client))
            tools.register(WebSearchTool(client=client))
        for tool in extras:
            if tools.has(tool.name):
                tools.unregister(tool.name)
            tools.register(tool)
        return tools

    def build_adapter(*args, **kwargs):
        adapter = ResearchAdapter(escape=escape)
        adapters.append(adapter)
        return adapter

    def forbidden_raw_search():
        raise AssertionError("A raw search callback would bypass the scoped registry")

    tool = runtime_agent.SpawnAgentsTool(
        provider="fake",
        model="fake",
        cwd=tmp_path,
        config=HarnessConfig(),
        build_adapter=build_adapter,
        build_tools=build_tools,
        build_search_fn=forbidden_raw_search,
        approval_policy=policy,
        max_workers=1,
    )
    (tmp_path / "AGENTS.md").write_text("Use the bounded official guide for this assignment.")
    return tool, adapters, agents


def run_call():
    return ToolCall(id="spawn", name="spawn_agents", arguments={"goal": "Research the guide"})


def response(request):
    if request.url.host == "duckduckgo.com":
        return httpx.Response(
            200,
            text='<a class="result__a" href="https://docs.example.org/guide">Guide</a>',
            headers={"Content-Type": "text/html"},
        )
    return httpx.Response(200, text="Official source evidence")


async def test_every_role_researches_through_actual_tools_and_preserves_workspace_scope(
    tmp_path: Path, monkeypatch
):
    requests = []

    def transport(request):
        requests.append(request)
        return response(request)

    (tmp_path.parent / "private.txt").write_text("Private outside-workspace sentinel")
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        tool, adapters, agents = make_tool(tmp_path, monkeypatch, client, escape=True)
        result = await tool(run_call())
        assert not client.is_closed  # Borrowed tool clients remain owned by the caller.
    assert result.content == "Research report"
    assert len(adapters) == len(agents) == 3
    assert len(requests) == 6  # Actual search and fetch for planner, worker, reporter.
    for adapter, agent in zip(adapters, agents, strict=True):
        assert agent.closed
        first = adapter.calls[0]
        system = first["messages"][0].content
        assert system.startswith(AUTONOMOUS_CONTEXT_POLICY)
        names = {schema["function"]["name"] for schema in first["tools"]}
        assert {"read_file", "glob", "list_dir", "web_search", "fetch_url"} <= names
        assert not ({"write_file", "edit_file", "clarify", "spawn_agents", "shell"} & names)
        results = {message.tool_call_id: message for message in adapter.calls[1]["messages"]}
        assert "bounded official guide" in results["local"].content
        assert "Official source evidence" in results["source"].content
        if "escape" in results:
            assert "Private outside-workspace sentinel" not in results["escape"].content
            assert "outside" in results["escape"].content


async def test_parent_denial_is_applied_before_research_dispatch(tmp_path, monkeypatch):
    requests = []

    def transport(request):
        requests.append(request)
        return response(request)

    policy = ApprovalPolicy(per_tool={"web_search": "deny", "fetch_url": "deny"})
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as client:
        tool, adapters, agents = make_tool(tmp_path, monkeypatch, client, policy=policy)
        await tool(run_call())
    assert requests == []
    for adapter, agent in zip(adapters, agents, strict=True):
        assert agent.approval_policy is policy
        results = {message.tool_call_id: message for message in adapter.calls[1]["messages"]}
        assert "denied" in results["search"].content.lower()
        assert "denied" in results["source"].content.lower()
        assert agent.closed


async def test_filtered_discovery_tools_stay_unavailable_and_local_research_still_runs(
    tmp_path, monkeypatch
):
    tool, adapters, agents = make_tool(tmp_path, monkeypatch, None, include_web=False)
    assert (await tool(run_call())).content == "Research report"
    for adapter, agent in zip(adapters, agents, strict=True):
        first = adapter.calls[0]
        names = {schema["function"]["name"] for schema in first["tools"]}
        assert not {"web_search", "fetch_url", "shell"} & names
        assert "unavailable in the inherited tool configuration: web_search, fetch_url" in (
            first["messages"][0].content
        )
        assert agent.closed


async def test_research_backend_failure_is_observable_to_each_role(tmp_path, monkeypatch):
    def failed(request):
        raise httpx.ConnectError("offline fixture", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(failed)) as client:
        tool, adapters, agents = make_tool(tmp_path, monkeypatch, client)
        await tool(run_call())
    for adapter, agent in zip(adapters, agents, strict=True):
        results = {message.tool_call_id: message for message in adapter.calls[1]["messages"]}
        assert "offline fixture" in results["search"].content
        assert "offline fixture" in results["source"].content
        assert agent.closed


async def test_cancellation_closes_active_research_before_owned_agents(tmp_path, monkeypatch):
    entered, cancelled = asyncio.Event(), asyncio.Event()

    async def blocked(request):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        return response(request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(blocked)) as client:
        tool, _, agents = make_tool(tmp_path, monkeypatch, client)
        task = asyncio.create_task(tool(run_call()))
        await asyncio.wait_for(entered.wait(), timeout=5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=5)
        assert cancelled.is_set()
        assert agents and all(agent.closed for agent in agents)


async def test_discovery_names_do_not_grant_write_effects_or_expand_shell_roles(
    tmp_path, monkeypatch
):
    class MutatingSearch:
        name = "web_search"
        description = "A configured plugin with a misleading discovery name"
        effect_scope = "external_write"
        approval = "auto"
        phases = ("*",)

        def __init__(self):
            self.parameters_schema = {"type": "object"}

        async def __call__(self, call):
            raise AssertionError("Discovery must not invoke write-effect plugins")

    policy = ApprovalPolicy(per_tool={"shell": "deny"})
    tool, adapters, agents = make_tool(
        tmp_path,
        monkeypatch,
        None,
        include_web=False,
        policy=policy,
        extras=(MutatingSearch(), ShellTool(cwd=tmp_path)),
    )
    assert (await tool(run_call())).content == "Research report"
    for adapter, agent in zip(adapters, agents, strict=True):
        names = {schema["function"]["name"] for schema in adapter.calls[0]["tools"]}
        assert "web_search" not in names
        assert ("shell" in names) == ("complete_work_item" in names)
        assert agent.approval_policy is policy


def build_parent(tmp_path, monkeypatch, *, execution=None):
    adapters: list[ResearchAdapter] = []
    agents: list[Agent] = []
    build_calls: list[Path] = []

    class OwnedAgent(Agent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            agents.append(self)

    class ShellResearchAdapter(ResearchAdapter):
        async def stream(self, **kwargs: Any) -> AsyncIterator[Event]:
            first = not self.calls
            worker = any(
                tool["function"]["name"] == "complete_work_item" for tool in kwargs["tools"]
            )
            async for event in super().stream(**kwargs):
                if first and worker and isinstance(event, Done):
                    assert event.final_message is not None
                    assert event.final_message.tool_calls is not None
                    event.final_message.tool_calls.append(
                        ToolCall(id="shell", name="shell", arguments={"command": "pwd"})
                    )
                yield event

    monkeypatch.setattr(runtime_agent, "Agent", OwnedAgent)

    def build_tools(cwd):
        build_calls.append(cwd)
        tools = ToolRegistry()
        for tool in (
            ReadFileTool(cwd=cwd),
            ListDirTool(cwd=cwd),
            GlobTool(cwd=cwd),
            ShellTool(cwd=cwd),
        ):
            tools.register(tool)
        return tools

    def build_adapter(*args, **kwargs):
        adapter = ShellResearchAdapter()
        adapters.append(adapter)
        return adapter

    parent = runtime_agent.build_agent(
        chain=["fake"],
        base_url=None,
        model="fake",
        storage=InMemoryStorage(),
        cwd=tmp_path,
        config=HarnessConfig(execution=execution, approval={"shell": "deny"}),
        yes=True,
        build_adapter=build_adapter,
        build_tools=build_tools,
        build_search_fn=lambda: None,
        console=None,
        project_context_enabled=False,
    )
    return parent, adapters, agents, build_calls


@pytest.fixture
def docker_transport(monkeypatch):
    calls: list[tuple[str, dict[str, Any] | None]] = []

    async def transport(argv, *, payload=None, timeout_seconds):
        assert argv[0] == "docker"
        operation = argv[1]
        if operation in {"run", "rm"}:
            calls.append((operation, None))
            return 0, b"owned-container", b""
        assert operation == "exec"
        assert payload is not None
        request = json.loads(base64.b64decode(payload))
        calls.append((request["operation"], request))
        assert request["root"] == "/workspace"
        assert request["operation"] in {"probe", "read_file"}
        result = {
            "event": "result",
            "content": "DOCKER_SENTINEL" if request["operation"] == "read_file" else "",
            "metadata": {"cwd": "/workspace"},
        }
        return 0, json.dumps(result).encode(), b""

    monkeypatch.setattr(execution_backend, "transport_command", transport)
    return calls


async def test_spawn_borrows_docker_tools_and_policy_without_opening_child_backends(
    tmp_path, monkeypatch, docker_transport
):
    (tmp_path / "AGENTS.md").write_text("HOST_SENTINEL")
    parent, adapters, agents, build_calls = build_parent(
        tmp_path, monkeypatch, execution=ExecutionConfig(backend="docker", docker_image="fixture")
    )
    async with parent:
        backend_read = parent.tools.get("read_file")
        backend_shell = parent.tools.get("shell")
        result = await parent.tools.get("spawn_agents")(run_call())
        assert result.content == "Research report"
        assert len(agents) == 4  # The actual parent, planner, worker, and reporter.
        for adapter, child in zip(adapters[1:], agents[1:], strict=True):
            results = {message.tool_call_id: message for message in adapter.calls[1]["messages"]}
            assert "DOCKER_SENTINEL" in results["local"].content
            assert "HOST_SENTINEL" not in results["local"].content
            assert child.tools.get("read_file") is backend_read
            assert child.approval_policy is parent.approval_policy
            assert child.approval_handler is parent.approval_handler
            if child.tools.has("complete_work_item"):
                assert child.tools.get("shell") is backend_shell
                assert "denied" in results["shell"].content.lower()
            else:
                assert not child.tools.has("shell")
        # Closing borrowed child agents leaves the owning backend usable.
        again = await backend_read(
            ToolCall(id="after-children", name="read_file", arguments={"path": "AGENTS.md"})
        )
        assert again.content == "DOCKER_SENTINEL"
        assert not any(operation == "rm" for operation, _ in docker_transport)
    assert build_calls == [tmp_path]
    operations = [operation for operation, _ in docker_transport]
    assert operations == ["run", "probe", *(["read_file"] * 4), "rm"]


async def test_spawn_inherits_registry_wrappers_removals_and_current_policy(tmp_path, monkeypatch):
    (tmp_path / "AGENTS.md").write_text("HOST_SENTINEL")
    parent, adapters, agents, build_calls = build_parent(tmp_path, monkeypatch)

    class ScopedRead:
        name = "read_file"
        description = "Parent-scoped read"
        effect_scope = "read_only"
        approval: ApprovalDecision = "auto"
        phases = ("*",)

        def __init__(self):
            self.parameters_schema = ReadFileTool(cwd=tmp_path).parameters_schema

        async def __call__(self, call):
            return ToolResult(tool_call_id=call.id, name=self.name, content="SCOPED_SENTINEL")

    parent.tools.unregister("read_file")
    scoped = ScopedRead()
    parent.tools.register(scoped)
    parent.tools.unregister("glob")
    parent.approval_policy = ApprovalPolicy(default="auto", per_tool={"shell": "deny"})
    async with parent:
        assert (await parent.tools.get("spawn_agents")(run_call())).content == "Research report"
    for adapter, child in zip(adapters[1:], agents[1:], strict=True):
        results = {message.tool_call_id: message for message in adapter.calls[1]["messages"]}
        assert results["local"].content == "SCOPED_SENTINEL"
        assert child.tools.get("read_file") is scoped
        assert not child.tools.has("glob")
        assert child.approval_policy is parent.approval_policy
    assert build_calls == [tmp_path]


async def test_spawn_requires_active_parent_execution_context(
    tmp_path, monkeypatch, docker_transport
):
    parent, adapters, _, build_calls = build_parent(
        tmp_path, monkeypatch, execution=ExecutionConfig(backend="docker", docker_image="fixture")
    )
    spawn = parent.tools.get("spawn_agents")
    before = await spawn(run_call())
    assert before.is_error
    assert "active" in before.content
    assert docker_transport == []
    async with parent:
        pass
    after = await spawn(run_call())
    assert after.is_error
    assert "active" in after.content
    assert len(adapters) == 1  # Neither invalid call constructed children or host tools.
    assert build_calls == [tmp_path]
    assert [operation for operation, _ in docker_transport] == ["run", "probe", "rm"]


async def test_standalone_spawn_cannot_rebuild_host_tools_for_configured_execution(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("An unbound managed spawn must not construct tools or adapters")

    spawn = runtime_agent.SpawnAgentsTool(
        provider="fake",
        model="fake",
        cwd=tmp_path,
        config=HarnessConfig(execution=ExecutionConfig(backend="docker", docker_image="fixture")),
        build_adapter=forbidden,
        build_tools=forbidden,
        build_search_fn=forbidden,
    )
    result = await spawn(run_call())
    assert result.is_error
    assert "active parent tool context" in result.content
