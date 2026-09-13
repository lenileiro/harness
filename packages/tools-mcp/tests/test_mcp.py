from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

from harness.core import ToolCall
from harness.core.tools import ToolRegistry
from harness.tools.mcp import MCPConnectionError, MCPServerConfig, MCPToolset, parse_mcp_servers

SERVER = Path(__file__).with_name("stdio_server.py")


def _stdio(**kwargs: Any) -> MCPServerConfig:
    return MCPServerConfig(name="local", command=sys.executable, args=(str(SERVER),), **kwargs)


def _call(tool_name: str, **arguments: Any) -> ToolCall:
    return ToolCall(id="offline-call", name=tool_name, arguments=arguments)


def _assert_stopped(cwd: Path) -> None:
    pid = int((cwd / "server.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_configuration_requires_explicit_secret_references(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_TEST_TOKEN", "offline-secret")
    config = MCPServerConfig(
        name="remote",
        transport="streamable-http",
        url="https://mcp.invalid/mcp",
        bearer_token_env="MCP_TEST_TOKEN",
        headers_from={"X-Test": "MCP_TEST_TOKEN"},
    )
    assert config.resolve_headers() == {
        "Authorization": "Bearer offline-secret",
        "X-Test": "offline-secret",
    }
    assert "offline-secret" not in repr(config)
    with pytest.raises(ValueError, match="MISSING_TEST_TOKEN"):
        config.model_copy(update={"bearer_token_env": "MISSING_TEST_TOKEN"}).resolve_headers(
            {"MCP_TEST_TOKEN": "offline-secret"}
        )
    with pytest.raises(ValueError) as caught:
        MCPServerConfig(name="bad", command="python", url="https://offline-secret.invalid")
    assert "offline-secret" not in str(caught.value)


@pytest.mark.parametrize(
    "settings",
    [
        {"name": "bad/name", "command": "python"},
        {"name": "bad", "transport": "stdio"},
        {"name": "bad", "transport": "streamable-http", "url": "file:///tmp/server"},
        {
            "name": "bad",
            "transport": "streamable-http",
            "url": "https://user:secret@example.invalid",
        },
        {"name": "bad", "command": "python", "timeout": float("nan")},
        {
            "name": "bad",
            "transport": "streamable-http",
            "url": "https://example.invalid",
            "headers": {"Authorization": "x"},
            "bearer_token_env": "TOKEN",
        },
    ],
)
def test_configuration_rejects_invalid_transport_fields(settings: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        MCPServerConfig.model_validate(settings)


def test_configuration_table_and_list_forms() -> None:
    table = parse_mcp_servers({"demo": {"command": "python", "include_tools": ["read*"]}})
    rows = parse_mcp_servers([{"name": "demo", "command": "python", "include_tools": ["read*"]}])
    assert table == rows
    with pytest.raises(ValueError, match="unique"):
        parse_mcp_servers([{"name": "same", "command": "python"}] * 2)
    with pytest.raises(ValueError, match="table key"):
        parse_mcp_servers({"demo": {"name": "other", "command": "python"}})


async def test_real_stdio_tools_structured_results_errors_env_and_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_MCP_TEST_TOKEN", "selected-value")
    monkeypatch.setenv("HARNESS_MCP_TEST_UNFORWARDED", "must-not-cross")
    registry = ToolRegistry()
    config = _stdio(
        env_from={"EXPLICIT": "HARNESS_MCP_TEST_TOKEN"}, env={"LITERAL": "literal-value"}
    )
    async with MCPToolset([config], cwd=tmp_path, registry=registry) as toolset:
        echo = registry.get("mcp__local__tool__echo")
        assert echo.approval == "prompt"  # readOnlyHint never grants approval.
        assert echo.effect_scope == "external_side_effect"  # type: ignore[attr-defined]
        result = await echo(_call(echo.name, text="hello"))
        assert not result.is_error and '"echo": "hello"' in result.content
        assert result.metadata is not None and result.metadata["structured_content"] == {
            "echo": "hello"
        }
        env_tool = registry.get("mcp__local__tool__environment")
        result = await env_tool(_call(env_tool.name))
        assert result.metadata is not None
        assert result.metadata["structured_content"] == {
            "explicit": "selected-value",
            "literal": "literal-value",
            "unforwarded": None,
            "cwd": str(tmp_path),
        }
        failure = registry.get("mcp__local__tool__fail")
        failed = await failure(_call(failure.name))
        assert failed.is_error and "deliberate server failure" in failed.content
        assert not (await echo(_call(echo.name, text="still connected"))).is_error
        assert registry.names() == sorted(tool.name for tool in toolset.tools)
    assert registry.names() == []
    _assert_stopped(tmp_path)


async def test_real_stdio_resource_templates_and_prompts(tmp_path: Path) -> None:
    registry = ToolRegistry()
    async with MCPToolset(
        [_stdio(expose_resources=True, expose_prompts=True, include_tools=("echo",))],
        cwd=tmp_path,
        registry=registry,
    ):
        assert "mcp__local__tool__fail" not in registry.names()
        resources = registry.get("mcp__local__list_resources")
        listing = await resources(_call(resources.name))
        assert "test://message" in listing.content and "test://greeting/{name}" in listing.content
        read = registry.get("mcp__local__read_resource")
        assert "offline resource" in (await read(_call(read.name, uri="test://message"))).content
        assert "Hello Ada" in (await read(_call(read.name, uri="test://greeting/Ada"))).content
        prompts = registry.get("mcp__local__list_prompts")
        assert "summarize" in (await prompts(_call(prompts.name))).content
        get = registry.get("mcp__local__get_prompt")
        result = await get(_call(get.name, name="summarize", arguments={"subject": "MCP"}))
        assert "Summarize MCP" in result.content and not result.is_error
    _assert_stopped(tmp_path)


async def _wait_for_file(path: Path) -> None:
    async with asyncio.timeout(5):
        # The signal comes from a separate process; it cannot set an asyncio.Event.
        while not await asyncio.to_thread(path.exists):  # noqa: ASYNC110
            await asyncio.sleep(0.01)


async def test_cancelled_call_terminates_stdio_and_never_replays(tmp_path: Path) -> None:
    registry = ToolRegistry()
    async with MCPToolset([_stdio()], cwd=tmp_path, registry=registry):
        slow = registry.get("mcp__local__tool__slow")
        pending = asyncio.create_task(slow(_call(slow.name)))
        await _wait_for_file(tmp_path / "started")
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        _assert_stopped(tmp_path)
        assert (tmp_path / "started").read_text().splitlines() == ["call"]
        after = await slow(_call(slow.name))
        assert after.is_error and after.metadata is not None
        assert after.metadata["outcome"] == "not_started"
    assert registry.names() == []


async def test_cancelled_context_unregisters_tools_and_reaps_server(tmp_path: Path) -> None:
    registry = ToolRegistry()
    ready = asyncio.Event()

    async def run() -> None:
        async with MCPToolset([_stdio()], cwd=tmp_path, registry=registry):
            ready.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run())
    await asyncio.wait_for(ready.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert registry.names() == []
    _assert_stopped(tmp_path)


async def test_timeout_closes_connection_without_retry(tmp_path: Path) -> None:
    async with MCPToolset([_stdio(timeout=2)], cwd=tmp_path) as toolset:
        slow = next(tool for tool in toolset.tools if tool.name.endswith("__slow"))
        result = await slow(_call(slow.name))
        assert result.is_error and "outcome may be unknown" in result.content
        assert result.metadata is not None and result.metadata["retryable"] is False
        assert (tmp_path / "started").read_text().splitlines() == ["call"]
        _assert_stopped(tmp_path)


async def test_failed_second_server_rolls_back_first_connection(tmp_path: Path) -> None:
    registry = ToolRegistry()
    missing = MCPServerConfig(name="missing", command=str(tmp_path / "no-such-command"))
    with pytest.raises(MCPConnectionError):
        async with MCPToolset([_stdio(), missing], cwd=tmp_path, registry=registry):
            pytest.fail("startup should fail")
    assert registry.names() == []
    _assert_stopped(tmp_path)


class HTTPServer:
    def __init__(self, *, fail_call: bool = False, repeat_cursor: bool = False):
        self.requests: list[httpx.Request] = []
        self.fail_call = fail_call
        self.repeat_cursor = repeat_cursor
        self.clients: list[httpx.AsyncClient] = []

    async def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "DELETE":
            return httpx.Response(204)
        if request.method == "GET":
            return httpx.Response(405)
        payload = json.loads(request.content)
        method = payload["method"]
        if "id" not in payload:
            return httpx.Response(202)
        if method == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": "2025-06-18",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "offline", "version": "1"},
            }
        elif method == "tools/list":
            if not payload.get("params", {}).get("cursor") or self.repeat_cursor:
                result = {
                    "tools": [{"name": "echo/unsafe", "inputSchema": {"type": "object"}}],
                    "nextCursor": "page-2",
                }
            else:
                result = {"tools": [{"name": "second", "inputSchema": {"type": "object"}}]}
        elif method == "tools/call":
            if self.fail_call:
                raise httpx.ReadError("offline response lost after mutation")
            result = {
                "content": [{"type": "text", "text": "HTTP result"}],
                "structuredContent": {"ok": True},
                "isError": False,
            }
        else:
            raise AssertionError(method)
        return httpx.Response(
            200,
            headers={"Mcp-Session-Id": "offline-session"},
            json={"jsonrpc": "2.0", "id": payload["id"], "result": result},
        )

    def client(self, config: MCPServerConfig, headers: dict[str, str]) -> httpx.AsyncClient:
        client = httpx.AsyncClient(
            transport=httpx.MockTransport(self.handle), headers=headers, timeout=config.timeout
        )
        self.clients.append(client)
        return client


def _http(**kwargs: Any) -> MCPServerConfig:
    return MCPServerConfig(
        name="remote", transport="streamable-http", url="https://mcp.invalid/mcp", **kwargs
    )


async def test_streamable_http_pagination_headers_and_session_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HARNESS_MCP_HTTP_TOKEN", "offline-bearer")
    server = HTTPServer()
    async with MCPToolset(
        [_http(bearer_token_env="HARNESS_MCP_HTTP_TOKEN")],
        cwd=tmp_path,
        http_client_factory=server.client,
    ) as toolset:
        assert len(toolset.tools) == 2
        assert all(
            len(tool.name) <= 64 and re.fullmatch(r"[A-Za-z0-9_-]+", tool.name)
            for tool in toolset.tools
        )
        tool = toolset.tools[0]
        result = await tool(_call(tool.name))
        assert not result.is_error and "HTTP result" in result.content
        assert result.metadata is not None and result.metadata["structured_content"] == {"ok": True}
    assert any(request.method == "DELETE" for request in server.requests)
    assert all(
        request.headers["authorization"] == "Bearer offline-bearer" for request in server.requests
    )
    assert all(client.is_closed for client in server.clients)


async def test_http_disconnection_does_not_replay_mutation(tmp_path: Path) -> None:
    server = HTTPServer(fail_call=True)
    async with MCPToolset(
        [_http(timeout=1)], cwd=tmp_path, http_client_factory=server.client
    ) as toolset:
        tool = toolset.tools[0]
        result = await tool(_call(tool.name))
        assert result.is_error and "Do not automatically retry" in result.content
    calls = [
        request
        for request in server.requests
        if request.method == "POST" and json.loads(request.content).get("method") == "tools/call"
    ]
    assert len(calls) == 1
    assert all(client.is_closed for client in server.clients)


async def test_repeated_discovery_cursor_fails_and_cleans_up(tmp_path: Path) -> None:
    server = HTTPServer(repeat_cursor=True)
    with pytest.raises(MCPConnectionError, match="repeated pagination"):
        async with MCPToolset([_http()], cwd=tmp_path, http_client_factory=server.client):
            pytest.fail("discovery should fail")
    assert all(client.is_closed for client in server.clients)


async def test_filters_and_disabled_servers_do_not_expose_unselected_tools(tmp_path: Path) -> None:
    server = HTTPServer()
    async with MCPToolset(
        [
            _http(include_tools=("*",), exclude_tools=("echo*",)),
            MCPServerConfig(name="disabled", command="no-such-command", enabled=False),
        ],
        cwd=tmp_path,
        http_client_factory=server.client,
    ) as toolset:
        assert [tool.name for tool in toolset.tools] == ["mcp__remote__tool__second"]


async def test_registration_collision_preserves_existing_registry(tmp_path: Path) -> None:
    server = HTTPServer()
    registry = ToolRegistry()
    async with MCPToolset(
        [_http()], cwd=tmp_path, registry=registry, http_client_factory=server.client
    ) as first:
        originals = list(first.tools)
        with pytest.raises(ValueError, match="already registered"):
            async with MCPToolset(
                [_http()], cwd=tmp_path, registry=registry, http_client_factory=server.client
            ):
                pytest.fail("collision should fail")
        assert [registry.get(tool.name) for tool in originals] == originals
        assert not (await originals[0](_call(originals[0].name))).is_error
    assert registry.names() == []
