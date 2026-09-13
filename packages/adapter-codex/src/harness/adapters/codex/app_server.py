"""Harness-managed tool bridge over the real Codex app-server protocol.

A process lives only while a Codex turn is waiting for a Harness tool result.
Harness owns the transcript; changed context is rebuilt in a fresh ephemeral
thread instead of replaying effects or silently retaining stale instructions.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from harness.adapters.codex.app_server_protocol import (
    MAX_WIRE_BYTES,
    NAMESPACE,
    JsonRpcProcess,
    dynamic_tools,
    isolation_config,
    response_items,
    tool_response,
)
from harness.core import (
    Capabilities,
    ConfigurationError,
    Done,
    Event,
    InternalError,
    Message,
    NetworkError,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    Usage,
)
from harness.core.errors import TimeoutError as HarnessTimeoutError


@dataclass
class _Turn:
    rpc: JsonRpcProcess
    directory: tempfile.TemporaryDirectory[str]
    model: str
    schemas: list[dict[str, Any]]
    messages: list[Message]
    thread_id: str = ""
    turn_id: str = ""
    request_id: str | int | None = None
    call: ToolCall | None = None
    result: ToolResult | None = None
    assistant: Message | None = None
    usage: Usage = field(default_factory=Usage)
    reported_usage: Usage = field(default_factory=Usage)
    dispatched_calls: set[str] = field(default_factory=set)

    async def close(self) -> None:
        try:
            if not self.rpc.closed and self.turn_id:
                with contextlib.suppress(Exception):
                    async with asyncio.timeout(1):
                        await self.rpc.send(
                            {
                                "id": "harness-interrupt",
                                "method": "turn/interrupt",
                                "params": {"threadId": self.thread_id, "turnId": self.turn_id},
                            }
                        )
            await self.rpc.close()
        finally:
            self.directory.cleanup()

    def usage_delta(self) -> Usage:
        delta = Usage(
            **{
                key: max(0, value - getattr(self.reported_usage, key))
                for key, value in self.usage.model_dump().items()
            }
        )
        self.reported_usage = self.usage.model_copy()
        return delta


class AppServerBridge:
    def __init__(self, *, codex_bin: str, timeout: float, idle_timeout: float) -> None:
        self.codex_bin = codex_bin
        self.timeout = timeout
        self.idle_timeout = idle_timeout
        self.turns: dict[str, _Turn] = {}
        self.streaming: set[str] = set()
        self.closing: dict[str, asyncio.Task[None]] = {}

    async def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True, tool_use=True, external_tools=True, input_media=["image", "audio"]
        )

    async def submit_tool_result(self, session_id: str, result: ToolResult) -> None:
        turn = self.turns.get(session_id)
        if turn is None or turn.call is None:
            return  # A resumed session reconstructs its transcript without a live RPC.
        if turn.call.id != result.tool_call_id or turn.call.name != result.name:
            raise InternalError("Codex bridge received a mismatched tool result")
        turn.result = result.model_copy(deep=True)

    async def end_run(self, session_id: str) -> None:
        turn = self.turns.pop(session_id, None)
        if turn:
            cleanup = asyncio.create_task(turn.close())
            self.closing[session_id] = cleanup
        else:
            cleanup = self.closing.get(session_id)
        if cleanup is not None:
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                await cleanup
                raise
            finally:
                if cleanup.done() and self.closing.get(session_id) is cleanup:
                    self.closing.pop(session_id, None)

    def _can_continue(
        self, turn: _Turn, model: str, schemas: list[dict[str, Any]], messages: list[Message]
    ) -> bool:
        if (
            turn.model != model
            or turn.schemas != schemas
            or turn.assistant is None
            or turn.result is None
            or turn.rpc.process.returncode is not None
        ):
            return False
        prefix = [*turn.messages, turn.assistant]
        if messages[: len(prefix)] != prefix:
            return False
        suffix = messages[len(prefix) :]
        # Skill activation, compaction and repair directives rebuild from the exact
        # new context. Never let a hidden native transcript override those changes.
        return (
            len(suffix) == 1
            and suffix[0].role == "tool"
            and suffix[0].tool_call_id == turn.result.tool_call_id
            and suffix[0].content == turn.result.content
            and suffix[0].attachments == turn.result.attachments
        )

    async def _open(
        self, session_id: str, model: str, schemas: list[dict[str, Any]], messages: list[Message]
    ) -> _Turn:
        directory = tempfile.TemporaryDirectory(prefix="harness-codex-")
        config = isolation_config()
        command = [self.codex_bin, "app-server", "--listen", "stdio://", "--strict-config"]
        for key, value in config.items():
            command.extend(["-c", f"{key}={json.dumps(value)}"])
        creation = asyncio.create_task(
            asyncio.create_subprocess_exec(
                *command,
                cwd=directory.name,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                limit=MAX_WIRE_BYTES,
                start_new_session=os.name == "posix",
            )
        )
        try:
            process = await asyncio.shield(creation)
        except asyncio.CancelledError:
            try:
                process = await creation
                await JsonRpcProcess(process, idle_timeout=self.idle_timeout).close()
            finally:
                directory.cleanup()
            raise
        except BaseException:
            directory.cleanup()
            raise
        turn = _Turn(
            rpc=JsonRpcProcess(process, idle_timeout=self.idle_timeout),
            directory=directory,
            model=model,
            schemas=schemas,
            messages=[message.model_copy(deep=True) for message in messages],
        )
        self.turns[session_id] = turn
        rpc = turn.rpc
        await rpc.request(
            "initialize",
            {
                "clientInfo": {"name": "harness", "version": "0.0.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        await rpc.send({"method": "initialized"})
        effective = await rpc.request("config/read", {"includeLayers": False})
        values = effective.get("config", {})
        for key, expected in config.items():
            # ConfigRead's ToolsV2 projection omits these two utility settings;
            # strict startup config still validates their spelling and values.
            if key in {
                "tools.update_plan.enabled",
                "tools.experimental_request_user_input.enabled",
            }:
                continue
            value: Any = values
            for part in key.split("."):
                value = value.get(part) if isinstance(value, dict) else None
            if value != expected:
                raise ConfigurationError(f"Codex cannot enforce bridge config: {key}")
        # The private empty cwd prevents automatic project config/instruction loads;
        # the selected Harness backend remains the only workspace tool authority.
        native_mcp = values.get("mcp_servers", {})
        if not isinstance(native_mcp, dict):
            raise ConfigurationError("Codex returned an invalid native MCP configuration")
        disabled_mcp = {"mcp_servers": {name: {"enabled": False} for name in native_mcp}}
        started = await rpc.request(
            "thread/start",
            {
                "model": model.removeprefix("openai/"),
                "ephemeral": True,
                "cwd": directory.name,
                "environments": [],
                "sandbox": "read-only",
                "approvalPolicy": "never",
                "dynamicTools": dynamic_tools(schemas),
                "config": disabled_mcp,
                "developerInstructions": "Use the Harness tool namespace for all external actions. "
                "The hosting runtime controls tools, approval and workspace access.",
            },
        )
        thread = started.get("thread", {})
        if thread.get("environments") != []:
            raise ConfigurationError(
                "Codex app-server did not disable native execution environments"
            )
        turn.thread_id = thread["id"]
        await rpc.request(
            "thread/inject_items", {"threadId": turn.thread_id, "items": response_items(messages)}
        )
        started_turn = await rpc.request(
            "turn/start",
            {
                "threadId": turn.thread_id,
                "input": [],
                "environments": [],
                "model": model.removeprefix("openai/"),
            },
        )
        turn.turn_id = started_turn["turn"]["id"]
        return turn

    async def stream(
        self,
        *,
        session_id: str,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Event]:
        if not session_id:
            raise ConfigurationError("Codex bridge requires a session_id and end_run lifecycle")
        if temperature is not None or max_tokens is not None:
            raise ConfigurationError(
                "Codex app-server does not expose temperature or max_tokens; "
                "leave them unset or choose an API adapter"
            )
        if kwargs.get("tool_choice") not in (None, "auto", "required"):
            raise ConfigurationError("Codex bridge supports auto/required tool choice only")
        schemas = tools or []
        dynamic_tools(schemas)
        response_items(messages)  # Validate media before starting any subprocess.
        if session_id in self.streaming:
            raise ConfigurationError("concurrent Codex streams for one session are not supported")
        self.streaming.add(session_id)
        keep_pending = False
        try:
            async with asyncio.timeout(self.timeout):
                turn = self.turns.get(session_id)
                if turn and not self._can_continue(turn, model, schemas, messages):
                    await self.end_run(session_id)
                    turn = None
                if turn is None:
                    turn = await self._open(session_id, model, schemas, messages)
                else:
                    assert turn.result is not None
                    await turn.rpc.send(
                        {"id": turn.request_id, "result": tool_response(turn.result)}
                    )
                    turn.request_id = None
                    turn.call = None
                    turn.result = None
                    turn.messages = [message.model_copy(deep=True) for message in messages]
                text: list[str] = []
                text_items: dict[str, str] = {}
                while True:
                    item = await turn.rpc.next_event()
                    method = item.get("method")
                    params = item.get("params", {})
                    if "id" in item and method:
                        if method != "item/tool/call":
                            await turn.rpc.send(
                                {
                                    "id": item["id"],
                                    "error": {
                                        "code": -32601,
                                        "message": "Harness bridge denies native tool and approval requests",
                                    },
                                }
                            )
                            raise ConfigurationError(
                                "Codex requested a native operation outside Harness"
                            )
                        allowed = {spec["function"]["name"] for spec in schemas}
                        if (
                            params.get("namespace") != NAMESPACE
                            or params.get("tool") not in allowed
                            or params.get("threadId") != turn.thread_id
                            or params.get("turnId") != turn.turn_id
                            or not isinstance(params.get("arguments"), dict)
                        ):
                            raise InternalError(
                                "Codex returned an invalid or unregistered Harness tool call"
                            )
                        call = ToolCall(
                            id=params["callId"], name=params["tool"], arguments=params["arguments"]
                        )
                        if not call.id or call.id in turn.dispatched_calls:
                            raise InternalError("Codex returned a duplicate or empty tool call ID")
                        turn.dispatched_calls.add(call.id)
                        turn.request_id, turn.call = item["id"], call
                        turn.assistant = Message(
                            role="assistant", content="".join(text) or None, tool_calls=[call]
                        )
                        yield ToolCallEvent(call=call)
                        keep_pending = True
                        yield Done(final_message=turn.assistant, usage=turn.usage_delta())
                        return
                    if params.get("threadId") not in (None, turn.thread_id):
                        continue
                    if method == "item/agentMessage/delta":
                        delta = params.get("delta", "")
                        if not isinstance(delta, str):
                            raise InternalError("Codex returned an invalid text delta")
                        text.append(delta)
                        text_items[str(params.get("itemId"))] = (
                            text_items.get(str(params.get("itemId")), "") + delta
                        )
                        yield TextDelta(text=delta)
                    elif method == "item/completed":
                        completed = params.get("item", {})
                        if (
                            completed.get("type") == "agentMessage"
                            and completed.get("id") not in text_items
                        ):
                            delta = completed.get("text", "")
                            if delta:
                                text.append(delta)
                                yield TextDelta(text=delta)
                    elif method == "thread/tokenUsage/updated":
                        usage = params.get("tokenUsage", {}).get("total", {})
                        turn.usage = Usage(
                            prompt_tokens=usage.get("inputTokens", 0),
                            completion_tokens=usage.get("outputTokens", 0),
                            total_tokens=usage.get("totalTokens", 0),
                            cache_read_input_tokens=usage.get("cachedInputTokens", 0),
                        )
                    elif method == "turn/completed":
                        status = params.get("turn", {}).get("status")
                        if status != "completed":
                            raise InternalError(
                                f"Codex app-server turn ended with status {status!r}"
                            )
                        yield Done(
                            final_message=Message(role="assistant", content="".join(text)),
                            usage=turn.usage_delta(),
                        )
                        return
        except TimeoutError as exc:
            raise HarnessTimeoutError("Codex app-server stream timed out") from exc
        except OSError as exc:
            raise NetworkError("failed to launch or communicate with Codex app-server") from exc
        finally:
            self.streaming.discard(session_id)
            if not keep_pending:
                await self.end_run(session_id)
