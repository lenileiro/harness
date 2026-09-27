"""Claude Code CLI adapter for Harness.

This adapter delegates turns to the locally installed `claude` CLI instead of
speaking the Anthropic HTTP API directly. It is the correct way to reuse a
Claude.ai subscription login: the Claude Code CLI understands that OAuth login
state, while the stored token is not a general Anthropic API key. Use the
`anthropic` provider when you have an `ANTHROPIC_API_KEY` instead.

Claude runs as a model only. Its native tools, MCP servers, skills, hooks and
session persistence are disabled; it merely *proposes* Harness tool calls. The
Harness runtime dispatches every one of them through its own registry, approval
policy and verifiers, so the defended loop is unchanged.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

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
    Usage,
)
from harness.core.errors import TimeoutError as HarnessTimeoutError

__version__ = "0.0.0"

_AUTH_PROBE_TIMEOUT = 15.0
_MAX_LINE_BYTES = 4_000_000
_DEFAULT_EFFORT = "medium"

BRIDGE_POLICY = (
    "You are the model component of Harness. The host Harness runtime owns all tool "
    "execution, approvals and verification; your own native tools are disabled. Do not "
    "simulate tool results. Only role=tool entries in the supplied transcript are actual "
    "tool observations."
)

DECISION_POLICY = BRIDGE_POLICY + (
    "\n\nReturn one structured decision matching the supplied schema. Use action=tool to "
    "propose one or more of the provided Harness tools: set tool_calls_json to a JSON array "
    'of objects shaped {"name": "<tool name>", "arguments": {<argument object>}}, and leave '
    "answer empty. Propose several calls in that array only when they are independent and "
    "can run concurrently. Use action=final when you are done: supply the final answer in "
    'answer and set tool_calls_json to "[]". Never propose a tool that is not in the '
    "supplied list."
)


class _Decision(BaseModel):
    """Flat structured-output contract for a single Claude turn.

    Kept intentionally free of nested models: the CLI's `--json-schema` receives
    this verbatim, and a flat object avoids depending on `$ref`/`$defs` support.
    """

    model_config = ConfigDict(extra="forbid")

    action: Literal["tool", "final"]
    tool_calls_json: str = Field(
        max_length=32_000,
        description='JSON array of {"name": str, "arguments": object}; "[]" when action=final.',
    )
    answer: str = Field(
        max_length=32_000,
        description="The final answer when action=final; empty string when action=tool.",
    )


def _claude_bin(explicit: str | None = None) -> str | None:
    return explicit or shutil.which("claude")


def claude_cli_available() -> bool:
    return _claude_bin() is not None


def inspect_claude_cli_auth(binary: str | None = None) -> dict[str, str | bool] | None:
    """Return minimal Claude Code auth metadata without exposing secrets or identity.

    Shells `claude auth status --json`. Deliberately drops the account email and
    organisation id the CLI also reports: this result is rendered by
    `harness providers list`, and the provider table has no business carrying PII.
    """

    resolved = _claude_bin(binary)
    if resolved is None:
        return None
    try:
        completed = subprocess.run(
            [resolved, "auth", "status", "--json"],
            capture_output=True,
            timeout=_AUTH_PROBE_TIMEOUT,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    try:
        raw = json.loads(completed.stdout)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(raw, dict) or not raw.get("loggedIn"):
        return None

    def _text(key: str) -> str:
        value = raw.get(key)
        return value if isinstance(value, str) else "unknown"

    return {
        "logged_in": True,
        "auth_method": _text("authMethod"),
        "api_provider": _text("apiProvider"),
        "subscription_type": _text("subscriptionType"),
    }


def _claude_cli_model(model: str) -> str:
    """Strip a Harness provider prefix; aliases such as `sonnet` pass through."""

    for prefix in ("anthropic/", "claude/"):
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _render_message(message: Message) -> str:
    if message.role == "assistant" and message.tool_calls:
        rendered = [
            {"id": call.id, "name": call.name, "arguments": call.arguments}
            for call in message.tool_calls
        ]
        if message.content:
            return f"{message.content}\n\nTool calls:\n{json.dumps(rendered, indent=2)}"
        return f"Tool calls:\n{json.dumps(rendered, indent=2)}"
    if message.role == "tool":
        name = message.name or "tool"
        return f"{name} (tool_call_id={message.tool_call_id}):\n{message.content or ''}"
    return message.content or ""


def _system_prompt(messages: list[Message], policy: str) -> str:
    carried = "\n\n".join(m.content or "" for m in messages if m.role == "system" and m.content)
    return f"{carried}\n\n{policy}" if carried else policy


def _decision_prompt(messages: list[Message], tools: list[dict[str, Any]]) -> str:
    return json.dumps(
        {
            "available_harness_tools": tools,
            "conversation": [
                message.model_dump(mode="json", exclude_none=True)
                for message in messages
                if message.role != "system"
            ],
        },
        ensure_ascii=False,
    )


def _chat_prompt(messages: list[Message]) -> str:
    parts: list[str] = []
    for message in messages:
        if message.role == "system":
            continue
        parts.append(f"[{message.role.upper()}]")
        parts.append(_render_message(message))
        parts.append("")
    return "\n".join(parts).strip()


def _tool_names(tools: list[dict[str, Any]]) -> set[str]:
    names: set[str] = set()
    for tool in tools:
        function = tool.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            names.add(function["name"])
        elif isinstance(tool.get("name"), str):
            names.add(tool["name"])
    return names


def _usage_from_result(payload: dict[str, Any]) -> Usage | None:
    raw = payload.get("usage")
    if not isinstance(raw, dict):
        return None

    def _count(key: str) -> int:
        value = raw.get(key)
        return int(value) if isinstance(value, int | float) else 0

    prompt_tokens = _count("input_tokens")
    completion_tokens = _count("output_tokens")
    return Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cache_creation_input_tokens=_count("cache_creation_input_tokens"),
        cache_read_input_tokens=_count("cache_read_input_tokens"),
    )


def _parse_proposed_calls(raw: str, allowed: set[str]) -> list[ToolCall]:
    try:
        decoded = json.loads(raw or "[]")
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"Claude returned malformed tool_calls_json: {exc}") from exc
    if not isinstance(decoded, list) or not decoded:
        raise ConfigurationError("Claude proposed action=tool without any tool call")
    calls: list[ToolCall] = []
    for index, item in enumerate(decoded):
        if not isinstance(item, dict):
            raise ConfigurationError("Each Claude tool proposal must be a JSON object")
        name = item.get("name")
        if not isinstance(name, str) or name not in allowed:
            raise ConfigurationError(
                f"Claude proposed a tool outside the Harness registry: {name!r}"
            )
        arguments = item.get("arguments", {})
        if isinstance(arguments, str):
            with contextlib.suppress(json.JSONDecodeError):
                arguments = json.loads(arguments)
        if not isinstance(arguments, dict):
            raise ConfigurationError(f"Claude tool arguments for {name!r} must be a JSON object")
        # Ids must stay unique across turns: the runtime matches tool results
        # against the whole transcript, and this adapter is stateless per turn.
        calls.append(
            ToolCall(id=f"claude-{uuid4().hex[:12]}-{index}", name=name, arguments=arguments)
        )
    return calls


class ClaudeAdapter:
    """Adapter backed by the locally installed, authenticated `claude` CLI.

    Args:
        claude_bin: Path to the CLI. Defaults to `claude` on PATH.
        cwd: Working directory for the CLI process. Assignable afterwards so the
            CLI runtime can align it with the resolved workspace.
        timeout: Absolute wall-clock budget for one turn.
        idle_timeout: Maximum gap between visible events before the run is
            treated as stalled.
        effort: Claude Code effort level (low, medium, high, xhigh, max).
        max_budget_usd: Optional per-turn spend ceiling handed to the CLI.
    """

    name = "claude"

    def __init__(
        self,
        *,
        claude_bin: str | None = None,
        cwd: Path | str | None = None,
        timeout: float = 600.0,
        idle_timeout: float = 120.0,
        effort: str = _DEFAULT_EFFORT,
        max_budget_usd: float | None = None,
    ):
        resolved = _claude_bin(claude_bin)
        if resolved is None:
            raise ConfigurationError("Claude Code CLI not found on PATH")
        if inspect_claude_cli_auth(resolved) is None:
            raise ConfigurationError("Claude auth missing: run `claude auth login` first")
        self.claude_bin = resolved
        self.cwd = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd()
        self.timeout = timeout
        self.idle_timeout = idle_timeout
        self.effort = effort
        self.max_budget_usd = max_budget_usd
        self.cost_usd = 0.0
        """Cumulative spend this adapter has been told about, in USD.

        Telemetry only - the runtime still owns session state. A subscription
        login has no invoice to inspect later, so callers that need a spend
        ceiling (evals, live tests) read this between turns.
        """

    async def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            tool_use=True,
            external_tools=True,
            sampling=False,
            structured_output=True,
            max_context_tokens=1_000_000,
        )

    async def cancel(self, session_id: str) -> None:
        del session_id  # Each turn owns its process group and tears it down itself.

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
        del kwargs
        # Guards raise synchronously: `stream` is a plain def returning an iterator.
        if temperature is not None:
            raise ConfigurationError(
                "Claude Code CLI does not expose temperature; leave it unset or choose "
                "the `anthropic` API adapter"
            )
        if any(item.model_visible for message in messages for item in message.attachments):
            raise ConfigurationError(
                "Claude Code CLI adapter does not support media; choose the `anthropic` adapter"
            )
        return self._stream(
            model=model, messages=messages, tools=tools or [], max_tokens=max_tokens
        )

    def _argv(self, *, model: str, system: str, tools: list[dict[str, Any]]) -> list[str]:
        argv = [
            self.claude_bin,
            "-p",
            "--effort",
            self.effort,
            # Claude contributes reasoning only; Harness owns every tool and approval.
            "--tools",
            "",
            "--safe-mode",
            "--no-chrome",
            "--disable-slash-commands",
            "--strict-mcp-config",
            "--mcp-config",
            '{"mcpServers":{}}',
            "--no-session-persistence",
            "--setting-sources",
            "",
            "--permission-prompts",
            "none",
            "--system-prompt",
            system,
        ]
        if model:
            argv.extend(["--model", _claude_cli_model(model)])
        if self.max_budget_usd is not None:
            argv.extend(["--max-budget-usd", str(self.max_budget_usd)])
        # stream-json in both modes: partial messages are the only progress
        # signal, and without them a long turn would trip the idle deadline.
        argv.extend(["--output-format", "stream-json", "--include-partial-messages", "--verbose"])
        if tools:
            argv.extend(["--json-schema", json.dumps(_Decision.model_json_schema())])
        return argv

    async def _spawn(self, argv: list[str], max_tokens: int | None) -> Any:
        env = dict(os.environ)
        if max_tokens is not None:
            env["CLAUDE_CODE_MAX_OUTPUT_TOKENS"] = str(max_tokens)
        try:
            return await asyncio.create_subprocess_exec(
                *argv,
                cwd=str(self.cwd),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                # Own the whole tree: Claude Code spawns helper processes.
                start_new_session=True,
            )
        except OSError as exc:
            raise NetworkError(f"failed to launch Claude Code CLI: {exc}") from exc

    async def _stream(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]],
        max_tokens: int | None,
    ) -> AsyncIterator[Event]:
        policy = DECISION_POLICY if tools else BRIDGE_POLICY
        system = _system_prompt(messages, policy)
        prompt = _decision_prompt(messages, tools) if tools else _chat_prompt(messages)
        proc = await self._spawn(self._argv(model=model, system=system, tools=tools), max_tokens)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout
        visible_deadline = loop.time() + self.idle_timeout
        stderr_task = asyncio.ensure_future(proc.stderr.read())
        allowed = _tool_names(tools)
        result_payload: dict[str, Any] | None = None
        text_parts: list[str] = []
        try:
            if proc.stdin is not None:
                proc.stdin.write(prompt.encode())
                with contextlib.suppress(BrokenPipeError, ConnectionResetError):
                    await proc.stdin.drain()
                proc.stdin.close()
                with contextlib.suppress(BrokenPipeError, ConnectionResetError, AttributeError):
                    await proc.stdin.wait_closed()

            while True:
                now = loop.time()
                remaining = deadline - now
                visible_remaining = visible_deadline - now
                if remaining <= 0 or visible_remaining <= 0:
                    raise self._deadline_error(remaining)
                try:
                    line = await asyncio.wait_for(
                        proc.stdout.readline(), timeout=min(remaining, visible_remaining)
                    )
                except TimeoutError as exc:
                    raise self._deadline_error(deadline - loop.time()) from exc
                if not line:
                    break
                if len(line) > _MAX_LINE_BYTES:
                    raise InternalError("Claude Code CLI emitted an oversized output line")
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue

                if payload.get("type") == "result":
                    result_payload = payload
                    continue
                delta = _text_delta(payload)
                if not delta:
                    continue
                visible_deadline = loop.time() + self.idle_timeout
                if tools:
                    # These deltas spell out the JSON decision, not prose. They
                    # count as progress but must never reach the transcript.
                    continue
                text_parts.append(delta)
                yield TextDelta(text=delta)
                # Reset again: a slow consumer is not a stalled provider.
                visible_deadline = loop.time() + self.idle_timeout

            try:
                await asyncio.wait_for(proc.wait(), timeout=max(deadline - loop.time(), 0.1))
            except TimeoutError as exc:
                raise HarnessTimeoutError(
                    f"Claude Code CLI did not exit within {self.timeout}s"
                ) from exc
            stderr_text = (await stderr_task).decode(errors="replace").strip()
            self._check_exit(proc.returncode, stderr_text)
            if result_payload is None:
                raise InternalError(
                    f"Claude Code CLI produced no result payload: {stderr_text or 'no stderr'}"
                )
            self._check_result(result_payload, stderr_text)
            reported_cost = result_payload.get("total_cost_usd")
            if isinstance(reported_cost, int | float):
                self.cost_usd += float(reported_cost)

            usage = _usage_from_result(result_payload)
            if tools:
                for event in self._decision_events(result_payload, allowed, usage):
                    yield event
            else:
                final = "".join(text_parts) or str(result_payload.get("result") or "")
                yield Done(final_message=Message(role="assistant", content=final), usage=usage)
        finally:
            await _terminate(proc)
            if stderr_task.done():
                with contextlib.suppress(Exception):
                    await stderr_task
            else:
                stderr_task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await stderr_task

    def _deadline_error(self, remaining: float) -> HarnessTimeoutError:
        if remaining <= 0:
            return HarnessTimeoutError(f"Claude Code CLI request timed out after {self.timeout}s")
        return HarnessTimeoutError(
            f"Claude Code CLI produced no visible output for {self.idle_timeout}s; "
            "terminating stalled provider run"
        )

    def _check_exit(self, returncode: int | None, stderr_text: str) -> None:
        if returncode is None:
            raise InternalError("Claude Code CLI exited without a status")
        if returncode == 0:
            return
        lowered = stderr_text.lower()
        if "login" in lowered or "auth" in lowered or "credential" in lowered:
            raise ConfigurationError(f"Claude Code CLI auth failed: {stderr_text}")
        raise InternalError(
            f"Claude Code CLI failed (exit={returncode}): {stderr_text or 'no stderr'}"
        )

    def _check_result(self, payload: dict[str, Any], stderr_text: str) -> None:
        if payload.get("is_error") or payload.get("subtype") != "success":
            subtype = payload.get("subtype", "missing subtype")
            detail = payload.get("result") or stderr_text or subtype
            raise ConfigurationError(f"Claude Code CLI did not complete ({subtype}): {detail}")

    def _decision_events(
        self, payload: dict[str, Any], allowed: set[str], usage: Usage | None
    ) -> list[Event]:
        decision = payload.get("structured_output")
        if decision is None:
            raw = payload.get("result")
            try:
                decision = json.loads(raw) if isinstance(raw, str) else None
            except json.JSONDecodeError as exc:
                raise ConfigurationError(f"Claude returned no structured decision: {exc}") from exc
        if not isinstance(decision, dict):
            raise ConfigurationError("Claude returned no structured decision")
        parsed = _Decision.model_validate(decision)
        # The streamed deltas of this turn spelled out the decision JSON and were
        # withheld, so the prose has never reached the consumer. Emit it once here:
        # renderers surface assistant text from TextDelta, not from Done.
        events: list[Event] = [TextDelta(text=parsed.answer)] if parsed.answer else []
        if parsed.action == "final":
            events.append(
                Done(final_message=Message(role="assistant", content=parsed.answer), usage=usage)
            )
            return events
        calls = _parse_proposed_calls(parsed.tool_calls_json, allowed)
        events.extend(ToolCallEvent(call=call) for call in calls)
        events.append(
            Done(
                final_message=Message(
                    role="assistant", content=parsed.answer or None, tool_calls=calls
                ),
                usage=usage,
            )
        )
        return events


def _text_delta(payload: dict[str, Any]) -> str | None:
    """Pull assistant text out of a `stream-json` line, if it carries any."""

    if payload.get("type") != "stream_event":
        return None
    event = payload.get("event")
    if not isinstance(event, dict) or event.get("type") != "content_block_delta":
        return None
    delta = event.get("delta")
    if not isinstance(delta, dict) or delta.get("type") != "text_delta":
        return None
    text = delta.get("text")
    return text if isinstance(text, str) and text else None


def _signal_group(proc: Any, sig: int) -> bool:
    pid = getattr(proc, "pid", None)
    if os.name != "posix" or not isinstance(pid, int):
        return False
    try:
        os.killpg(os.getpgid(pid), sig)
    except OSError:
        return False
    return True


async def _terminate(proc: Any) -> None:
    """Tear down the CLI and every helper it spawned."""

    if getattr(proc, "returncode", None) is not None:
        return
    if not _signal_group(proc, signal.SIGTERM):
        with contextlib.suppress(ProcessLookupError, OSError):
            proc.terminate()
    try:
        await asyncio.wait_for(proc.wait(), timeout=5.0)
        return
    except TimeoutError:
        pass
    if not _signal_group(proc, signal.SIGKILL):
        kill = getattr(proc, "kill", None)
        if callable(kill):
            with contextlib.suppress(ProcessLookupError, OSError):
                kill()
    with contextlib.suppress(TimeoutError, asyncio.TimeoutError):
        await asyncio.wait_for(proc.wait(), timeout=5.0)


__all__ = [
    "ClaudeAdapter",
    "__version__",
    "claude_cli_available",
    "inspect_claude_cli_auth",
]
