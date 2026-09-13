"""Codex 0.154 app-server wire conversions and owned JSONL transport."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
from collections import deque
from typing import Any

from harness.core import ConfigurationError, InternalError, Message, NetworkError, ToolResult

NAMESPACE = "harness"
MAX_WIRE_BYTES = 64 * 1024 * 1024

# These are operator overrides, applied before app-server initializes. In addition,
# every thread and turn selects environments=[]: upstream excludes native shell,
# apply_patch and view_image when no execution environment is selected.
DISABLED_FEATURES = (
    "apps",
    "plugins",
    "remote_plugin",
    "recommended_plugins",
    "tool_suggest",
    "hooks",
    "codex_hooks",
    "plugin_hooks",
    "shell_tool",
    "shell_snapshot",
    "shell_snapshot_v2",
    "unified_exec",
    "js_repl",
    "code_mode",
    "code_mode_host",
    "code_mode_only",
    "multi_agent",
    "multi_agent_v2",
    "browser_use",
    "computer_use",
    "image_generation",
    "imagegenext",
    "web_search",
    "standalone_web_search",
    "memories",
    "memory_tool",
    "external_agent_memory_import",
    "deferred_executor",
    "request_permissions_tool",
    "view_image",
    "remote_control",
    "goals",
)


def isolation_config() -> dict[str, Any]:
    return {
        **{f"features.{name}": False for name in DISABLED_FEATURES},
        "features.skip_host_skill_discovery": True,
        "agents.enabled": False,
        "orchestrator.mcp.enabled": False,
        "orchestrator.skills.enabled": False,
        "tools.update_plan.enabled": False,
        "tools.experimental_request_user_input.enabled": False,
        "web_search": "disabled",
        "project_doc_max_bytes": 0,
        "include_environment_context": False,
        "notify": [],
        "analytics.enabled": False,
    }


def dynamic_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function", {})
        name = function.get("name", "")
        if tool.get("type") != "function" or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
            raise ConfigurationError("Codex bridge requires named OpenAI function tool schemas")
        if name in names:
            raise ConfigurationError(f"duplicate Codex bridge tool: {name}")
        names.add(name)
        specs.append(
            {
                "type": "function",
                "name": name,
                "description": function.get("description", ""),
                "inputSchema": function.get("parameters", {"type": "object"}),
                "deferLoading": False,
            }
        )
    return (
        [
            {
                "type": "namespace",
                "name": NAMESPACE,
                "description": "Tools executed by Harness with its approval and workspace policies.",
                "tools": specs,
            }
        ]
        if specs
        else []
    )


def media_content(message: Message | ToolResult, *, dynamic: bool = False) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for attachment in message.attachments:
        if not attachment.model_visible:
            continue
        if attachment.kind not in {"image", "audio"} or attachment.url is not None:
            raise ConfigurationError(
                "Codex app-server supports inline image/audio media; load a local file with "
                "MediaAttachment.from_file instead of a remote URL or generic file"
            )
        if dynamic:
            kind = "inputImage" if attachment.kind == "image" else "inputAudio"
            field = "imageUrl" if attachment.kind == "image" else "audioUrl"
        else:
            kind = "input_" + attachment.kind
            field = attachment.kind + "_url"
        content.append({"type": kind, field: attachment.data_uri()})
    return content


def response_items(messages: list[Message]) -> list[dict[str, Any]]:
    """Preserve roles, tool IDs and inline media when rebuilding a fresh thread."""
    items: list[dict[str, Any]] = []
    for message in messages:
        media = media_content(message)
        if message.role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_call_id,
                    "output": [{"type": "input_text", "text": message.content or ""}, *media],
                }
            )
            continue
        if message.content or media:
            role = "developer" if message.role == "system" else message.role
            text_kind = "output_text" if role == "assistant" else "input_text"
            content = [{"type": text_kind, "text": message.content}] if message.content else []
            items.append({"type": "message", "role": role, "content": [*content, *media]})
        for call in message.tool_calls or []:
            items.append(
                {
                    "type": "function_call",
                    "call_id": call.id,
                    "namespace": NAMESPACE,
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                }
            )
    return items


def tool_response(result: ToolResult) -> dict[str, Any]:
    return {
        "success": not result.is_error,
        "contentItems": [
            {"type": "inputText", "text": result.content},
            *media_content(result, dynamic=True),
        ],
    }


class JsonRpcProcess:
    """One private stdio transport; no shared Codex daemon or external listener."""

    def __init__(self, process: asyncio.subprocess.Process, *, idle_timeout: float) -> None:
        self.process = process
        self.idle_timeout = idle_timeout
        self.sequence = 0
        self.notifications: deque[dict[str, Any]] = deque()
        self.stderr_task = asyncio.create_task(self._drain_stderr())
        self.closed = False

    async def _drain_stderr(self) -> None:
        # Drain without retaining logs: auth/provider diagnostics can contain secrets.
        if self.process.stderr:
            while await self.process.stderr.read(8192):
                pass

    async def send(self, message: dict[str, Any]) -> None:
        if self.process.stdin is None:
            raise NetworkError("Codex app-server stdin is unavailable")
        data = json.dumps(message).encode() + b"\n"
        if len(data) > MAX_WIRE_BYTES:
            raise ConfigurationError("Codex app-server request exceeds 64 MiB wire limit")
        try:
            self.process.stdin.write(data)
            await self.process.stdin.drain()
        except (BrokenPipeError, ConnectionError) as exc:
            raise NetworkError("Codex app-server closed its input") from exc

    async def read(self) -> dict[str, Any]:
        if self.process.stdout is None:
            raise NetworkError("Codex app-server stdout is unavailable")
        try:
            line = await asyncio.wait_for(self.process.stdout.readline(), self.idle_timeout)
        except ValueError as exc:
            raise InternalError("Codex app-server exceeded its message size limit") from exc
        if not line:
            raise NetworkError("Codex app-server closed its output")
        try:
            item = json.loads(line)
        except (ValueError, UnicodeError) as exc:
            raise InternalError("Codex app-server returned malformed JSON") from exc
        if not isinstance(item, dict):
            raise InternalError("Codex app-server returned a non-object message")
        return item

    async def request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.sequence += 1
        request_id = f"harness-{self.sequence}"
        await self.send({"id": request_id, "method": method, "params": params})
        while True:
            item = await self.read()
            if item.get("id") == request_id and "method" not in item:
                if "error" in item:
                    # Do not include arbitrary server diagnostics/credentials in evidence.
                    raise ConfigurationError(f"Codex app-server rejected {method}")
                result = item.get("result")
                if not isinstance(result, dict):
                    raise InternalError(f"Codex app-server returned an invalid {method} result")
                return result
            if len(self.notifications) >= 1000:
                raise InternalError("Codex app-server notification buffer exceeded 1000 messages")
            self.notifications.append(item)

    async def next_event(self) -> dict[str, Any]:
        return self.notifications.popleft() if self.notifications else await self.read()

    async def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        proc = self.process
        if proc.stdin:
            proc.stdin.close()
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                if os.name == "posix":
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), 3)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    if os.name == "posix":
                        os.killpg(proc.pid, signal.SIGKILL)
                    else:
                        proc.kill()
                await proc.wait()
        self.stderr_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self.stderr_task
