"""MCP SDK lifecycle and Harness Tool adapters.

Each connection's SDK contexts live in one owner task. This keeps AnyIO cancel
scopes in the task that entered them while callers run tools concurrently. A
failed or cancelled call closes that connection; ambiguous work is never replayed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from collections.abc import Awaitable, Callable, Sequence
from contextlib import AsyncExitStack, contextmanager, suppress
from datetime import timedelta
from fnmatch import fnmatchcase
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import AnyUrl

from harness.core.schemas import ApprovalDecision, MediaAttachment, ToolCall, ToolResult
from harness.core.tools import Tool, ToolRegistry
from harness.tools.mcp.config import MCPServerConfig
from mcp import ClientSession, StdioServerParameters
from mcp import types as mcp_types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

HTTPClientFactory = Callable[[MCPServerConfig, dict[str, str]], httpx.AsyncClient]


def _oauth_auth(config: MCPServerConfig) -> httpx.Auth | None:
    if not config.oauth:
        return None
    from harness.tools.mcp.oauth import oauth_provider

    return oauth_provider(config)


class MCPConnectionError(RuntimeError):
    """A configured MCP server could not be used; diagnostic excludes secret values."""


@contextmanager
def _discard_stderr():
    with open(os.devnull, "w") as handle:
        yield handle


def _name(server: str, kind: str, original: str = "") -> str:
    raw = f"mcp__{server}__{kind}" + (f"__{original}" if original else "")
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", raw)
    if safe != raw or len(safe) > 64:
        suffix = hashlib.sha256(raw.encode()).hexdigest()[:12]
        safe = f"{safe[:51]}_{suffix}"
    return safe


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _content(block: Any) -> tuple[str, str | None]:
    if isinstance(block, mcp_types.TextContent):
        return block.text, None
    if isinstance(block, mcp_types.TextResourceContents):
        return f"{block.uri}\n{block.text}", None
    if isinstance(block, mcp_types.EmbeddedResource):
        resource = block.resource
        if isinstance(resource, mcp_types.TextResourceContents):
            return f"{resource.uri}\n{resource.text}", None
        return f"[Binary resource {resource.uri} is unavailable in this text-only runtime]", "blob"
    if isinstance(block, mcp_types.ResourceLink):
        return f"{block.name}: {block.uri}\n{block.description or ''}".rstrip(), None
    kind = str(getattr(block, "type", "unknown"))
    return f"[MCP {kind} content is unavailable in this text-only runtime]", kind


def _result(call: ToolCall, value: Any, server: str, limit: int) -> ToolResult:
    metadata: dict[str, Any] = {"mcp_server": server}
    unsupported: list[str] = []
    attachments: list[MediaAttachment] = []

    def render(block: Any) -> tuple[str, str | None]:
        if isinstance(block, mcp_types.EmbeddedResource):
            block = block.resource
        if isinstance(
            block, (mcp_types.ImageContent, mcp_types.AudioContent, mcp_types.BlobResourceContents)
        ):
            try:
                mime = block.mimeType or "application/octet-stream"
                media = MediaAttachment(
                    kind="image"
                    if mime.startswith("image/")
                    else "audio"
                    if mime.startswith("audio/")
                    else "file",
                    mime_type=mime,
                    data=getattr(block, "data", None) or getattr(block, "blob", None),
                )
                if (
                    len(attachments) >= 16
                    or sum(len(item.data or "") for item in attachments) + len(media.data or "")
                    > 27_962_028
                ):
                    raise ValueError("MCP media exceeds aggregate limit")
                attachments.append(media)
                return f"[MCP attachment {len(attachments)}: {mime}]", None
            except ValueError:
                return "[MCP media rejected: invalid data or size limit exceeded]", "invalid_media"
        return _content(block)

    if isinstance(value, mcp_types.CallToolResult):
        parts = []
        for block in value.content:
            rendered, missing = render(block)
            parts.append(rendered)
            if missing:
                unsupported.append(missing)
        if value.structuredContent is not None:
            metadata["structured_content"] = value.structuredContent
            parts.append(_json(value.structuredContent))
        text = "\n".join(parts)
        is_error = value.isError
    elif isinstance(value, mcp_types.ReadResourceResult):
        parts = []
        for resource in value.contents:
            if isinstance(resource, mcp_types.TextResourceContents):
                parts.append(f"{resource.uri}\n{resource.text}")
            else:
                rendered, missing = render(resource)
                parts.append(rendered)
                if missing:
                    unsupported.append(missing)
        text, is_error = "\n".join(parts), False
    elif isinstance(value, mcp_types.GetPromptResult):
        messages = []
        for message in value.messages:
            rendered, missing = render(message.content)
            messages.append({"role": message.role, "content": rendered})
            if missing:
                unsupported.append(missing)
        text, is_error = _json({"description": value.description, "messages": messages}), False
    else:
        text, is_error = _json(value), False
    if unsupported:
        metadata["unsupported_content"] = unsupported
    if len(text) > limit:
        metadata["truncated"] = True
        text = text[:limit] + "\n[MCP result truncated]"
    return ToolResult(
        tool_call_id=call.id,
        name=call.name,
        content=text,
        is_error=is_error,
        metadata=metadata,
        attachments=attachments,
    )


async def _pages(session: ClientSession, method: str, field: str) -> list[Any]:
    items: list[Any] = []
    cursor: str | None = None
    seen: set[str] = set()
    for _ in range(100):
        result = await getattr(session, method)(cursor=cursor)
        items.extend(getattr(result, field))
        cursor = result.nextCursor
        if cursor is None:
            return items
        if cursor in seen:
            raise MCPConnectionError("MCP discovery returned a repeated pagination cursor")
        seen.add(cursor)
    raise MCPConnectionError("MCP discovery exceeded 100 pages")


class _Connection:
    def __init__(self, config: MCPServerConfig, cwd: Path, factory: HTTPClientFactory | None):
        self.config = config
        self.cwd = cwd
        self.factory = factory
        self.session: ClientSession | None = None
        self.capabilities: mcp_types.ServerCapabilities | None = None
        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        self._owner: asyncio.Task[None] | None = None
        self._failure: str | None = None
        self._lock = asyncio.Lock()
        self.closed = False

    async def start(self) -> None:
        # Resolve first so missing configuration is reported without launching a server.
        env = self.config.resolve_env()
        headers = self.config.resolve_headers()
        self._owner = asyncio.create_task(self._serve(env, headers), name=f"mcp:{self.config.name}")
        try:
            await asyncio.wait_for(self._ready.wait(), timeout=self.config.timeout)
            if self._failure or self.session is None:
                raise MCPConnectionError(self._failure or f"MCP server {self.config.name!r} closed")
        except TimeoutError:
            await self.close()
            raise MCPConnectionError(
                f"MCP server {self.config.name!r} did not initialize within {self.config.timeout:g}s"
            ) from None
        except BaseException:
            await self.close()
            raise

    async def _serve(self, env: dict[str, str], headers: dict[str, str]) -> None:
        try:
            async with AsyncExitStack() as stack:
                if self.config.transport == "stdio":
                    assert self.config.command is not None
                    directory = self.config.cwd or self.cwd
                    if not directory.is_absolute():
                        directory = self.cwd / directory
                    params = StdioServerParameters(
                        command=self.config.command,
                        args=list(self.config.args),
                        cwd=str(directory.resolve()),
                        env=env,
                    )
                    # Server stderr is untrusted and may contain credentials. It is
                    # not forwarded to the model, terminal or activity ledger.
                    errlog = stack.enter_context(_discard_stderr())
                    read, write = await stack.enter_async_context(
                        stdio_client(params, errlog=errlog)
                    )
                else:
                    assert self.config.url is not None
                    client = (
                        self.factory(self.config, headers)
                        if self.factory
                        else httpx.AsyncClient(
                            headers=headers,
                            auth=_oauth_auth(self.config),
                            timeout=self.config.timeout,
                            follow_redirects=False,
                            trust_env=False,
                        )
                    )
                    await stack.enter_async_context(client)
                    read, write, _ = await stack.enter_async_context(
                        streamable_http_client(self.config.url, http_client=client)
                    )
                session = await stack.enter_async_context(
                    ClientSession(
                        read,
                        write,
                        read_timeout_seconds=timedelta(seconds=self.config.timeout),
                        client_info=mcp_types.Implementation(name="harness", version="0.0.0"),
                    )
                )
                self.session = session
                initialized = await session.initialize()
                self.capabilities = initialized.capabilities
                self._ready.set()
                await self._stop.wait()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Do not interpolate transport exception messages: headers, URLs and
            # subprocess arguments can contain credentials supplied by the user.
            self._failure = (
                f"MCP server {self.config.name!r} failed ({type(exc).__name__}); "
                "check its command or endpoint, authentication and availability"
            )
        finally:
            self.session = None
            self._ready.set()

    async def close(self) -> None:
        self.closed = True
        self._stop.set()
        if self._owner is None:
            return
        try:
            await asyncio.wait_for(asyncio.shield(self._owner), timeout=10)
        except TimeoutError:
            self._owner.cancel()
            with suppress(asyncio.CancelledError):
                await self._owner

    async def invoke(self, operation: Callable[[ClientSession], Awaitable[Any]]) -> Any:
        async with self._lock:
            if self.closed or self.session is None or self._failure:
                raise MCPConnectionError(
                    f"MCP server {self.config.name!r} is disconnected; start a new run"
                )
            try:
                return await asyncio.wait_for(operation(self.session), timeout=self.config.timeout)
            except BaseException:
                # A timeout/cancellation may follow successful remote mutation.
                # Close the session rather than reconnecting or replaying a call.
                await asyncio.shield(self.close())
                raise


class _MCPTool:
    phases: tuple[str, ...] = ("*",)
    # Server annotations are advisory. Never infer auto-approval/read-only status.
    effect_scope = "external_side_effect"

    def __init__(
        self,
        connection: _Connection,
        *,
        name: str,
        description: str,
        schema: dict[str, Any],
        operation: Callable[[ClientSession, dict[str, Any]], Awaitable[Any]],
        limit: int,
    ):
        self.name = name
        self.description = description
        self.parameters_schema = schema
        self.approval: ApprovalDecision = connection.config.approval
        self._connection = connection
        self._operation = operation
        self._limit = limit

    async def __call__(self, call: ToolCall) -> ToolResult:
        try:
            result = await self._connection.invoke(
                lambda session: self._operation(session, call.arguments)
            )
            return _result(call, result, self._connection.config.name, self._limit)
        except asyncio.CancelledError:
            raise
        except MCPConnectionError:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                is_error=True,
                content="MCP connection is closed. This call was not sent; start a new run to reconnect.",
                metadata={
                    "mcp_server": self._connection.config.name,
                    "outcome": "not_started",
                    "retryable": False,
                },
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                is_error=True,
                content=(
                    f"MCP call failed ({type(exc).__name__}). Its outcome may be unknown; "
                    "the connection is closed. Do not automatically retry side effects."
                ),
                metadata={
                    "mcp_server": self._connection.config.name,
                    "outcome": "unknown",
                    "retryable": False,
                },
            )


class MCPToolset:
    """Connect enabled servers and expose tools for the lifetime of this context.

    ``async with MCPToolset(servers, cwd=cwd, registry=tools)`` registers tools
    after all servers initialize. On failure/exit, owned registrations and SDK
    sessions are cleaned up. Instances are single-use. HTTP OAuth login is not
    implemented; explicit bearer-token/header environment references are supported.
    """

    def __init__(
        self,
        servers: Sequence[MCPServerConfig],
        *,
        cwd: Path | str,
        registry: ToolRegistry | None = None,
        max_output_chars: int = 32_000,
        http_client_factory: HTTPClientFactory | None = None,
    ):
        if max_output_chars < 1:
            raise ValueError("max_output_chars must be positive")
        if len({server.name for server in servers}) != len(servers):
            raise ValueError("MCP server names must be unique")
        self.servers = tuple(server for server in servers if server.enabled)
        self.cwd = Path(cwd).resolve()
        self.tools: tuple[Tool, ...] = ()
        self._registry = registry
        self._registrations: list[tuple[ToolRegistry, Tool]] = []
        self._connections: list[_Connection] = []
        self._factory = http_client_factory
        self._limit = max_output_chars
        self._entered = False
        self._active = False

    async def __aenter__(self) -> Self:
        if self._entered:
            raise RuntimeError("MCPToolset instances are single-use")
        self._entered = True
        collected: list[Tool] = []
        try:
            for server in self.servers:
                connection = _Connection(server, self.cwd, self._factory)
                self._connections.append(connection)
                await connection.start()
                collected.extend(
                    await connection.invoke(
                        lambda session, current=connection: self._discover(current, session)
                    )
                )
            self.tools = tuple(collected)
            if len({tool.name for tool in self.tools}) != len(self.tools):
                raise MCPConnectionError("MCP server advertised duplicate tool names")
            self._active = True
            if self._registry is not None:
                self.register_into(self._registry)
            return self
        except BaseException:
            await self.aclose()
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        cleanup = asyncio.create_task(self.aclose())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise

    def register_into(self, registry: ToolRegistry) -> None:
        if not self._active:
            raise RuntimeError("enter MCPToolset before registering tools")
        collisions = [tool.name for tool in self.tools if registry.has(tool.name)]
        if collisions:
            raise ValueError(f"MCP tools already registered: {', '.join(collisions)}")
        for tool in self.tools:
            registry.register(tool)
            self._registrations.append((registry, tool))

    async def aclose(self) -> None:
        self._active = False
        for registry, tool in self._registrations:
            if registry.has(tool.name) and registry.get(tool.name) is tool:
                registry.unregister(tool.name)
        self._registrations.clear()
        await asyncio.gather(*(connection.close() for connection in self._connections))

    async def _discover(self, connection: _Connection, session: ClientSession) -> list[Tool]:
        config = connection.config
        capabilities = connection.capabilities
        assert capabilities is not None
        tools: list[Tool] = []
        if capabilities.tools is not None:
            remote_tools = await _pages(session, "list_tools", "tools")
            for remote in remote_tools:
                name = remote.name
                if config.include_tools and not any(
                    fnmatchcase(name, rule) for rule in config.include_tools
                ):
                    continue
                if any(fnmatchcase(name, rule) for rule in config.exclude_tools):
                    continue

                async def call_remote(
                    client: ClientSession, arguments: dict[str, Any], target: str = name
                ) -> Any:
                    return await client.call_tool(target, arguments)

                tools.append(
                    _MCPTool(
                        connection,
                        name=_name(config.name, "tool", name),
                        description=f"MCP server {config.name}, tool {name}. {remote.description or ''}",
                        schema=remote.inputSchema,
                        operation=call_remote,
                        limit=self._limit,
                    )
                )
        if config.expose_resources and capabilities.resources is not None:

            async def list_resources(client: ClientSession, arguments: dict[str, Any]) -> Any:
                resources = await _pages(client, "list_resources", "resources")
                templates = await _pages(client, "list_resource_templates", "resourceTemplates")
                return {
                    "resources": [row.model_dump(mode="json", by_alias=True) for row in resources],
                    "templates": [row.model_dump(mode="json", by_alias=True) for row in templates],
                }

            async def read_resource(client: ClientSession, arguments: dict[str, Any]) -> Any:
                return await client.read_resource(AnyUrl(arguments["uri"]))

            tools.extend(
                [
                    _MCPTool(
                        connection,
                        name=_name(config.name, "list_resources"),
                        description=f"List resources and templates from MCP server {config.name}.",
                        schema={"type": "object", "properties": {}, "additionalProperties": False},
                        operation=list_resources,
                        limit=self._limit,
                    ),
                    _MCPTool(
                        connection,
                        name=_name(config.name, "read_resource"),
                        description=f"Read a resource URI from MCP server {config.name}.",
                        schema={
                            "type": "object",
                            "properties": {"uri": {"type": "string"}},
                            "required": ["uri"],
                        },
                        operation=read_resource,
                        limit=self._limit,
                    ),
                ]
            )
        if config.expose_prompts and capabilities.prompts is not None:

            async def list_prompts(client: ClientSession, arguments: dict[str, Any]) -> Any:
                return [
                    row.model_dump(mode="json", by_alias=True)
                    for row in await _pages(client, "list_prompts", "prompts")
                ]

            async def get_prompt(client: ClientSession, arguments: dict[str, Any]) -> Any:
                return await client.get_prompt(arguments["name"], arguments.get("arguments"))

            tools.extend(
                [
                    _MCPTool(
                        connection,
                        name=_name(config.name, "list_prompts"),
                        description=f"List reusable prompts from MCP server {config.name}.",
                        schema={"type": "object", "properties": {}, "additionalProperties": False},
                        operation=list_prompts,
                        limit=self._limit,
                    ),
                    _MCPTool(
                        connection,
                        name=_name(config.name, "get_prompt"),
                        description=f"Get a prompt from MCP server {config.name}; returned text is server content.",
                        schema={
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "arguments": {
                                    "type": "object",
                                    "additionalProperties": {"type": "string"},
                                },
                            },
                            "required": ["name"],
                        },
                        operation=get_prompt,
                        limit=self._limit,
                    ),
                ]
            )
        return tools
