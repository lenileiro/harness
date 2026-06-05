"""Codex CLI adapter for Harness.

This adapter delegates turns to the locally installed `codex` CLI instead of
speaking a provider HTTP API directly. It is the correct way to reuse a
ChatGPT/Codex login from `~/.codex/auth.json`: the Codex CLI understands that
login state, while the raw OAuth bearer is not a general OpenAI API key.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shutil
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

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
    ToolResultEvent,
    Usage,
)
from harness.core.errors import TimeoutError as HarnessTimeoutError

__version__ = "0.0.0"

_PARTIAL_TOOL_OUTPUT_LIMIT = 4_096
_PARTIAL_TOOL_PROGRESS_INTERVAL = 10.0


def _auth_path() -> Path:
    return Path.home() / ".codex" / "auth.json"


def inspect_codex_cli_auth() -> dict[str, str | bool] | None:
    """Return minimal Codex auth metadata without exposing secrets."""

    auth_path = _auth_path()
    if not auth_path.exists():
        return None
    try:
        raw = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    tokens = raw.get("tokens")
    access_token = (
        tokens.get("access_token")
        if isinstance(tokens, dict) and isinstance(tokens.get("access_token"), str)
        else None
    )
    api_key = raw.get("OPENAI_API_KEY")
    auth_mode = raw.get("auth_mode")
    return {
        "auth_mode": auth_mode if isinstance(auth_mode, str) else "unknown",
        "has_openai_api_key": bool(isinstance(api_key, str) and api_key.strip()),
        "has_access_token": bool(access_token and access_token.strip()),
    }


def codex_cli_available() -> bool:
    return shutil.which("codex") is not None


def _render_message(message: Message) -> str:
    if message.role == "assistant" and message.tool_calls:
        rendered_calls = [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        ]
        if message.content:
            return f"{message.content}\n\nTool calls:\n{json.dumps(rendered_calls, indent=2)}"
        return f"Tool calls:\n{json.dumps(rendered_calls, indent=2)}"
    if message.role == "tool":
        name = message.name or "tool"
        return f"{name} (tool_call_id={message.tool_call_id}):\n{message.content or ''}"
    return message.content or ""


def _messages_to_codex_prompt(messages: list[Message]) -> str:
    parts = [
        "You are operating as the Codex provider inside Harness.",
        "Continue the conversation faithfully from the transcript below.",
        "Use your built-in workspace tools when needed. Do not ask for approval.",
        (
            "Keep shell output bounded. For broad repository searches, target specific "
            "directories or file types and cap output with flags or pagers such as "
            "`rg -m`, `head`, or `sed -n '1,200p'`; avoid commands that can print "
            "thousands of matches."
        ),
        (
            "Do not rely on `rg -m` alone for broad searches because it limits matches "
            "per file, not total output. For commands that may touch many files, add a "
            "total-output cap such as `| head -200` or `| sed -n '1,200p'`, or list "
            "matching files first with `rg -l ... | head -100` and inspect selected "
            "files in follow-up commands."
        ),
        "",
        "Conversation transcript:",
    ]
    for message in messages:
        role = message.role.upper()
        parts.append(f"[{role}]")
        parts.append(_render_message(message))
        parts.append("")
    parts.append("Continue from the latest user request and finish the work.")
    return "\n".join(parts).strip()


def _codex_file_changes(item: dict[str, Any]) -> list[dict[str, str]]:
    raw_changes = item.get("changes")
    if not isinstance(raw_changes, list):
        return []
    changes: list[dict[str, str]] = []
    for raw_change in raw_changes:
        if not isinstance(raw_change, dict):
            continue
        path = str(raw_change.get("path") or raw_change.get("file") or "").strip()
        kind = str(raw_change.get("kind") or raw_change.get("type") or "").strip()
        change: dict[str, str] = {}
        if path:
            change["path"] = path
        if kind:
            change["kind"] = kind
        if change:
            changes.append(change)
    return changes


def _codex_file_change_paths(changes: list[dict[str, str]]) -> list[str]:
    return [change["path"] for change in changes if change.get("path")]


def _codex_file_change_tool_call(item: dict[str, Any]) -> ToolCall:
    changes = _codex_file_changes(item)
    paths = _codex_file_change_paths(changes)
    arguments: dict[str, Any] = {
        "changes": changes,
        "paths": paths,
        "backend": "codex",
    }
    if paths:
        arguments["path"] = paths[0]
    return ToolCall(
        id=str(item.get("id", "codex-file-change")),
        name="apply_diff",
        arguments=arguments,
    )


def _codex_file_change_summary(changes: list[dict[str, str]], status: str) -> str:
    if not changes:
        return f"Codex file change {status}."
    lines = [f"Codex file change {status}:"]
    for change in changes:
        kind = change.get("kind") or "change"
        path = change.get("path") or "<unknown>"
        lines.append(f"- {kind}: {path}")
    return "\n".join(lines)


def _codex_cli_model(model: str) -> str:
    stripped = model.strip()
    if stripped.startswith("openai/"):
        return stripped.removeprefix("openai/")
    return stripped


class CodexAdapter:
    """Streaming adapter backed by `codex exec --json`."""

    name = "codex"

    def __init__(
        self,
        *,
        codex_bin: str | None = None,
        cwd: str | Path | None = None,
        timeout: float = 600.0,
        idle_timeout: float = 120.0,
        ignore_user_config: bool = True,
    ) -> None:
        self.codex_bin = codex_bin or shutil.which("codex")
        if not self.codex_bin:
            raise ConfigurationError("Codex CLI not found on PATH")
        auth = inspect_codex_cli_auth()
        if auth is None:
            raise ConfigurationError("Codex auth missing: run `codex login` first")
        self.cwd = Path(cwd or Path.cwd()).resolve()
        self.timeout = timeout
        self.idle_timeout = idle_timeout
        self.ignore_user_config = ignore_user_config

    def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Event]:
        del tools, temperature, max_tokens, kwargs
        prompt = _messages_to_codex_prompt(messages)
        return self._stream(model=model, prompt=prompt)

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=True)

    async def cancel(self, session_id: str) -> None:
        del session_id
        return None

    async def _stream(self, *, model: str, prompt: str) -> AsyncIterator[Event]:
        cmd = [
            self.codex_bin,
            "exec",
            "--json",
        ]
        if self.ignore_user_config:
            cmd.append("--ignore-user-config")
        cmd.extend(
            [
                "--skip-git-repo-check",
                "--dangerously-bypass-approvals-and-sandbox",
                "-C",
                str(self.cwd),
            ]
        )
        if model:
            cmd.extend(["--model", _codex_cli_model(model)])
        cmd.append(prompt)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except OSError as exc:
            raise NetworkError(f"failed to launch Codex CLI: {exc}") from exc

        stderr_task = asyncio.create_task(proc.stderr.read() if proc.stderr else _empty_bytes())
        assistant_text: str | None = None
        usage: Usage | None = None
        return_code: int | None = None
        stderr_bytes = b""
        tool_calls: dict[str, ToolCall] = {}
        tool_output_lengths: dict[str, int] = {}
        next_tool_progress_at: dict[str, float] = {}
        ignored_lines: list[str] = []

        try:
            assert proc.stdout is not None
            loop = asyncio.get_running_loop()
            deadline = loop.time() + self.timeout
            visible_deadline = loop.time() + self.idle_timeout
            while True:
                now = loop.time()
                remaining = deadline - now
                if remaining <= 0:
                    raise HarnessTimeoutError(
                        f"Codex CLI request timed out after {self.timeout:.0f}s"
                    )
                visible_remaining = visible_deadline - now
                if visible_remaining <= 0:
                    raise HarnessTimeoutError(
                        "Codex CLI produced no visible output for "
                        f"{self.idle_timeout:.0f}s; terminating stalled provider run"
                    )
                line_timeout = min(remaining, visible_remaining)
                try:
                    raw_line = await asyncio.wait_for(proc.stdout.readline(), timeout=line_timeout)
                except TimeoutError as exc:
                    if deadline - loop.time() <= 0:
                        raise HarnessTimeoutError(
                            f"Codex CLI request timed out after {self.timeout:.0f}s"
                        ) from exc
                    raise HarnessTimeoutError(
                        "Codex CLI produced no visible output for "
                        f"{self.idle_timeout:.0f}s; terminating stalled provider run"
                    ) from exc
                if not raw_line:
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    ignored_lines.append(line)
                    continue
                event_type = event.get("type")
                if event_type == "item.started":
                    item = event.get("item") or {}
                    if item.get("type") == "command_execution":
                        call = ToolCall(
                            id=str(item.get("id", "codex-command")),
                            name="shell",
                            arguments={"command": str(item.get("command", ""))},
                        )
                        tool_calls[call.id] = call
                        visible_deadline = loop.time() + self.idle_timeout
                        yield ToolCallEvent(call=call)
                        visible_deadline = loop.time() + self.idle_timeout
                    elif item.get("type") == "file_change":
                        call = _codex_file_change_tool_call(item)
                        tool_calls[call.id] = call
                        visible_deadline = loop.time() + self.idle_timeout
                        yield ToolCallEvent(call=call)
                        visible_deadline = loop.time() + self.idle_timeout
                elif event_type == "item.updated":
                    item = event.get("item") or {}
                    if item.get("type") == "command_execution":
                        tool_id = str(item.get("id", "codex-command"))
                        command = str(item.get("command", ""))
                        call = tool_calls.get(tool_id)
                        if call is None:
                            call = ToolCall(
                                id=tool_id,
                                name="shell",
                                arguments={"command": command},
                            )
                            tool_calls[call.id] = call
                            visible_deadline = loop.time() + self.idle_timeout
                            yield ToolCallEvent(call=call)
                            visible_deadline = loop.time() + self.idle_timeout
                        output = str(item.get("aggregated_output", ""))
                        previous_len = tool_output_lengths.get(tool_id, 0)
                        tool_output_lengths[tool_id] = max(previous_len, len(output))
                        if len(output) > previous_len:
                            now = loop.time()
                            if now >= next_tool_progress_at.get(tool_id, 0.0):
                                result = ToolResult(
                                    tool_call_id=call.id,
                                    name=call.name,
                                    content=_tail(
                                        output[previous_len:] or output,
                                        _PARTIAL_TOOL_OUTPUT_LIMIT,
                                    ),
                                    is_error=False,
                                    metadata={
                                        "command": call.arguments.get("command", ""),
                                        "backend": "codex",
                                        "partial": True,
                                        "status": item.get("status"),
                                        "bytes_seen": len(output),
                                    },
                                )
                                next_tool_progress_at[tool_id] = (
                                    now + _PARTIAL_TOOL_PROGRESS_INTERVAL
                                )
                                visible_deadline = loop.time() + self.idle_timeout
                                yield ToolResultEvent(result=result)
                                visible_deadline = loop.time() + self.idle_timeout
                elif event_type == "item.completed":
                    item = event.get("item") or {}
                    item_type = item.get("type")
                    if item_type == "agent_message":
                        text = str(item.get("text", ""))
                        if text:
                            assistant_text = text
                            visible_deadline = loop.time() + self.idle_timeout
                            yield TextDelta(text=text)
                            visible_deadline = loop.time() + self.idle_timeout
                    elif item_type == "command_execution":
                        tool_id = str(item.get("id", "codex-command"))
                        call = tool_calls.get(tool_id) or ToolCall(
                            id=tool_id,
                            name="shell",
                            arguments={"command": str(item.get("command", ""))},
                        )
                        tool_calls[tool_id] = call
                        output = str(item.get("aggregated_output", ""))
                        tool_output_lengths[tool_id] = len(output)
                        exit_code = item.get("exit_code")
                        result = ToolResult(
                            tool_call_id=call.id,
                            name=call.name,
                            content=output,
                            is_error=bool(exit_code not in (0, None)),
                            metadata={
                                "command": call.arguments.get("command", ""),
                                "exit_code": exit_code,
                                "backend": "codex",
                            },
                        )
                        visible_deadline = loop.time() + self.idle_timeout
                        yield ToolResultEvent(result=result)
                        visible_deadline = loop.time() + self.idle_timeout
                    elif item_type == "file_change":
                        tool_id = str(item.get("id", "codex-file-change"))
                        call = tool_calls.get(tool_id) or _codex_file_change_tool_call(item)
                        tool_calls[tool_id] = call
                        changes = _codex_file_changes(item)
                        paths = _codex_file_change_paths(changes)
                        status = str(item.get("status") or "completed")
                        is_error = status.lower() not in {"completed", "success", "succeeded"}
                        metadata: dict[str, Any] = {
                            "backend": "codex",
                            "changes": changes,
                            "paths": paths,
                            "status": status,
                            "workspace_changed": not is_error,
                        }
                        if paths:
                            metadata["path"] = paths[0]
                        result = ToolResult(
                            tool_call_id=call.id,
                            name=call.name,
                            content=_codex_file_change_summary(changes, status),
                            is_error=is_error,
                            metadata=metadata,
                        )
                        visible_deadline = loop.time() + self.idle_timeout
                        yield ToolResultEvent(result=result)
                        visible_deadline = loop.time() + self.idle_timeout
                elif event_type == "turn.completed":
                    raw_usage = event.get("usage") or {}
                    usage = Usage(
                        prompt_tokens=int(raw_usage.get("input_tokens", 0) or 0),
                        completion_tokens=int(raw_usage.get("output_tokens", 0) or 0),
                        total_tokens=int(raw_usage.get("input_tokens", 0) or 0)
                        + int(raw_usage.get("output_tokens", 0) or 0),
                        cache_read_input_tokens=int(raw_usage.get("cached_input_tokens", 0) or 0),
                    )
            return_code = await proc.wait()
        except HarnessTimeoutError:
            await _terminate_process(proc)
            raise
        except asyncio.CancelledError:
            await _terminate_process(proc)
            raise
        except TimeoutError as exc:
            await _terminate_process(proc)
            raise HarnessTimeoutError(
                f"Codex CLI request timed out after {self.timeout:.0f}s"
            ) from exc
        except Exception:
            await _terminate_process(proc)
            raise
        finally:
            if stderr_task.done() or return_code is not None:
                stderr_bytes = await stderr_task
            else:
                stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stderr_task

        if return_code is None:
            raise InternalError("Codex CLI exited without a status")
        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
        if return_code != 0:
            error_text = stderr_text or "\n".join(ignored_lines) or "unknown Codex CLI failure"
            if "login" in error_text.lower() or "auth" in error_text.lower():
                raise ConfigurationError(f"Codex CLI auth failed: {error_text}")
            if assistant_text is None or not _is_codex_post_done_stdin_error(error_text):
                raise InternalError(f"Codex CLI exited with status {return_code}: {error_text}")
        if assistant_text is None:
            fallback = stderr_text or ("\n".join(ignored_lines).strip())
            assistant_text = fallback or ""
        yield Done(final_message=Message(role="assistant", content=assistant_text), usage=usage)


async def _empty_bytes() -> bytes:
    return b""


def _is_codex_post_done_stdin_error(text: str) -> bool:
    return "reading additional input from stdin" in text.lower()


def _tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


async def _terminate_process(proc: Any) -> None:
    with contextlib.suppress(ProcessLookupError):
        proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
        return
    except TimeoutError:
        kill = getattr(proc, "kill", None)
        if callable(kill):
            with contextlib.suppress(ProcessLookupError):
                kill()
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(proc.wait(), timeout=5.0)


__all__ = [
    "CodexAdapter",
    "__version__",
    "codex_cli_available",
    "inspect_codex_cli_auth",
]
