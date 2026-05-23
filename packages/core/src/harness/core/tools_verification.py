"""Agent-callable verification and critique tools.

These tools are always registered in the agent's tool registry — no flags
required. They give the LLM the ability to self-verify and self-critique
proactively during its work, not just receive feedback at the end.

``verify_work``
    Run a verification command chosen by the agent (language-agnostic). The
    agent supplies the command; the tool runs it and returns stdout+stderr plus
    a pass/fail verdict. Designed to be called BEFORE the agent declares done.

``request_critique``
    Ask a second LLM reviewer to challenge the agent's proposed approach. The
    agent describes what it plans to do and why; the critic returns a pointed
    question or objection the agent must address before proceeding. Designed
    for moments of uncertainty: "am I about to fix the right thing?"
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from harness.core.adapter import Adapter
from harness.core.critic import SearchFn
from harness.core.events import Done, TextDelta
from harness.core.schemas import ApprovalDecision, Message, ToolCall, ToolResult

_CRITIQUE_SYSTEM = """\
You are a code review critic. An AI agent is about to make a change and wants \
a second opinion.

Your job: identify whether the agent's proposed approach actually addresses the \
problem as described, then ask a pointed question if something looks wrong.

Rules:
- Be concise: 3-5 sentences maximum
- If the approach looks correct, say so briefly and confirm the agent should proceed
- If the approach has a flaw: name it and ask one specific question the agent must \
answer before proceeding
- Do NOT provide the correct solution unprompted
- Tone: direct and collegial — "have you considered..." not "you are wrong"\
"""

_CRITIQUE_USER = """\
## Proposed approach

{approach}

## Problem context / failure output

{context}

Does this approach address what the problem actually requires? Critique in 3-5 \
sentences. If you spot a flaw, ask one specific question.\
"""

_VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": (
                "The verification command to run, appropriate for this project's language "
                "and test framework. Examples: 'pytest tests/ -v', 'npm test', "
                "'cargo test', 'go test ./...', 'make test'."
            ),
        },
    },
    "required": ["command"],
}

_CRITIQUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approach": {
            "type": "string",
            "description": "Describe what you plan to change and why you think it fixes the problem.",
        },
        "context": {
            "type": "string",
            "description": "Paste the relevant error, test failure, or problem description you're working with.",
        },
    },
    "required": ["approach", "context"],
}


class VerifyWorkTool:
    """Run a verification command and return pass/fail + output.

    The agent chooses the appropriate command for the project (pytest, npm test,
    cargo test, etc.). Call this before declaring a task complete.
    """

    name = "verify_work"
    description = (
        "Run the project's test suite to check whether your current code is correct. "
        "Use this as your inner feedback loop: write or edit code, call verify_work, "
        "read the failure output, revise your approach, call verify_work again. "
        "Repeat until all tests pass. Do NOT declare the task complete until "
        "verify_work returns PASSED."
    )
    # `verify_work` runs an arbitrary shell command — same blast radius as
    # `shell`. Keep the approval level consistent so an agent can't bypass the
    # shell approval gate by passing its command through here instead.
    approval: ApprovalDecision = "prompt"
    # NOTE: declared read_only because the typical use-case is "run the test
    # suite," which is read-only. But the underlying mechanism is subprocess
    # shell, so the structural denylist (check_dangerous_command) is applied
    # below as a backstop regardless of the declared scope.
    effect_scope = "read_only"

    def __init__(self, cwd: Path, *, timeout: float = 120.0) -> None:
        self._cwd = cwd
        self._timeout = timeout
        self.parameters_schema = _VERIFY_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        from harness.core.shell_safety import check_dangerous_command

        command = call.arguments.get("command", "")
        if not command or not command.strip():
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="command argument is required",
                is_error=True,
            )

        denial = check_dangerous_command(command)
        if denial is not None:
            tier, reason = denial
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=(
                    f"refused [{tier} deny]: {reason}. verify_work is for "
                    f"test commands (pytest, npm test, cargo test, etc.), "
                    f"not arbitrary shell. Use a test runner."
                ),
                is_error=True,
                metadata={"refused_reason": reason, "deny_tier": tier, "denied": True},
            )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=self._cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
            output = stdout.decode(errors="replace").strip()
            passed = proc.returncode == 0
            verdict = "PASSED" if passed else f"FAILED (exit {proc.returncode})"
            content = f"{verdict}\n\n{output}" if output else verdict
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=content,
                is_error=not passed,
            )
        except TimeoutError:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"command timed out after {self._timeout}s",
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"error running command: {exc}",
                is_error=True,
            )


class RequestCritiqueTool:
    """Ask a second LLM reviewer to challenge your proposed approach.

    Describe what you plan to change and why; the critic returns a pointed
    challenge or confirms you're on the right track. Call this when you're
    uncertain about your diagnosis before making changes.
    """

    name = "request_critique"
    description = (
        "Get a second opinion on your proposed approach before making changes. "
        "Describe what you plan to do and why — the critic will identify any flaws "
        "in your reasoning and ask you a specific question if something looks wrong. "
        "Call this when you're uncertain about your diagnosis."
    )
    approval: ApprovalDecision = "auto"
    effect_scope = "read_only"

    def __init__(
        self,
        adapter: Adapter,
        model: str,
        *,
        max_tokens: int = 400,
        temperature: float = 0.3,
        search_fn: SearchFn | None = None,
        max_searches: int = 2,
    ) -> None:
        self._adapter = adapter
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._search_fn = search_fn
        self._max_searches = max_searches
        self.parameters_schema = _CRITIQUE_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        approach = str(call.arguments.get("approach", "")).strip()
        context = str(call.arguments.get("context", "")).strip()

        if not approach:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="approach argument is required",
                is_error=True,
            )

        # Pre-fetch web context without LLM tool-calling (works with local models).
        research_lines: list[str] = []
        if self._search_fn is not None:
            queries = [approach[:120], context[:120] if context else ""]
            for q in queries[: self._max_searches]:
                if not q.strip():
                    continue
                try:
                    result = await self._search_fn(q.strip())  # type: ignore[misc]
                    if result:
                        research_lines.append(f"Search: {q.strip()[:80]}\n{result[:800]}")
                except Exception:
                    pass

        research_block = (
            "\n## Web research\n\n" + "\n\n".join(research_lines) + "\n\n" if research_lines else ""
        )

        messages: list[Message] = [
            Message(role="system", content=_CRITIQUE_SYSTEM),
            Message(
                role="user",
                content=research_block
                + _CRITIQUE_USER.format(
                    approach=approach[:2000],
                    context=context[:2000] if context else "(no context provided)",
                ),
            ),
        ]

        try:
            text_parts: list[str] = []
            async for event in self._adapter.stream(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            ):
                if isinstance(event, TextDelta):
                    text_parts.append(event.text)
                elif isinstance(event, Done):
                    if event.final_message and event.final_message.content:
                        return ToolResult(
                            tool_call_id=call.id,
                            name=self.name,
                            content=event.final_message.content.strip()
                            or "(critic produced no output)",
                        )
                    break
            critique = "".join(text_parts).strip()
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=critique or "(critic produced no output)",
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"critic unavailable: {exc}",
                is_error=True,
            )


__all__ = ["RequestCritiqueTool", "VerifyWorkTool"]
