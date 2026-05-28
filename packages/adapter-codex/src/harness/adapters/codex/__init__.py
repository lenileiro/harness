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
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "-C",
            str(self.cwd),
        ]
        if model:
            cmd.extend(["--model", model])
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
        tool_calls: dict[str, ToolCall] = {}
        ignored_lines: list[str] = []

        try:
            assert proc.stdout is not None
            deadline = asyncio.get_running_loop().time() + self.timeout
            while True:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    raise HarnessTimeoutError(
                        f"Codex CLI request timed out after {self.timeout:.0f}s"
                    )
                line_timeout = min(remaining, self.idle_timeout)
                try:
                    raw_line = await asyncio.wait_for(proc.stdout.readline(), timeout=line_timeout)
                except TimeoutError as exc:
                    raise HarnessTimeoutError(
                        "Codex CLI produced no output for "
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
                        yield ToolCallEvent(call=call)
                elif event_type == "item.completed":
                    item = event.get("item") or {}
                    item_type = item.get("type")
                    if item_type == "agent_message":
                        text = str(item.get("text", ""))
                        if text:
                            assistant_text = text
                            yield TextDelta(text=text)
                    elif item_type == "command_execution":
                        tool_id = str(item.get("id", "codex-command"))
                        call = tool_calls.get(tool_id) or ToolCall(
                            id=tool_id,
                            name="shell",
                            arguments={"command": str(item.get("command", ""))},
                        )
                        tool_calls[tool_id] = call
                        output = str(item.get("aggregated_output", ""))
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
                        yield ToolResultEvent(result=result)
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
        except TimeoutError as exc:
            await _terminate_process(proc)
            raise HarnessTimeoutError(
                f"Codex CLI request timed out after {self.timeout:.0f}s"
            ) from exc
        finally:
            stderr_bytes = await stderr_task

        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()
        if return_code != 0:
            error_text = stderr_text or "\n".join(ignored_lines) or "unknown Codex CLI failure"
            if "login" in error_text.lower() or "auth" in error_text.lower():
                raise ConfigurationError(f"Codex CLI auth failed: {error_text}")
            raise InternalError(f"Codex CLI exited with status {return_code}: {error_text}")
        if assistant_text is None:
            fallback = stderr_text or ("\n".join(ignored_lines).strip())
            assistant_text = fallback or ""
        yield Done(final_message=Message(role="assistant", content=assistant_text), usage=usage)


async def _empty_bytes() -> bytes:
    return b""


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
