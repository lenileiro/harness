from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from harness.cli.config import HarnessConfig
from harness.cli.research_commands import build_context_packet_prompt
from harness.core import (
    Agent,
    ContextBudget,
    ContextCompactor,
    Done,
    ErrorEvent,
    HandoffEvent,
    HandoffTool,
    Message,
    RunRequest,
    Session,
    StepCompleted,
    StepStarted,
    Storage,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
)
from harness.storage.sqlite import SQLiteStorage

_HELP_TEXT = (
    "/help              show this help\n"
    "/quit, /exit, /q   exit the chat\n"
    "/tools             list registered tools and effective approval\n"
    "/session           show current session id and turn count\n"
    "/sessions          list known conversations and running state\n"
    "/new [name]        create and switch to a new conversation\n"
    "/switch <id|name>  switch the active conversation\n"
    "/send <id|name|.> <message>  run a background turn on a conversation\n"
    "/diff              show file changes made this session\n"
    "/clear             clear the terminal\n"
    "/model [name]      show or switch the active model mid-session\n"
)

SlashHandler = Callable[..., Awaitable[bool]]


_REVIEW_SYSTEM_PROMPT = """
Review the proposed code or codebase. Stay read-only.

Focus on:
- correctness bugs
- behavioral regressions
- missing tests
- unsafe assumptions

Ignore style-only nits. Keep the output short and high-signal.

Return JSON only with this shape:
{
  "summary": "one short paragraph",
  "findings": [
    {
      "severity": "high|medium|low",
      "file": "relative/path.py",
      "line": 12,
      "issue": "what is wrong",
      "rationale": "why it matters",
      "suggested_fix": "optional concrete fix"
    }
  ]
}

Additional constraints:
- Keep the summary to 2 sentences max
- Return at most 3 findings
""".strip()


_WORKFLOW_BOOTSTRAP_SYSTEM_PROMPT = """
Set up Harness-native durable work. Stay inside Harness primitives.

Goal:
- bootstrap long-running, repeated, resumable, or mission-style work from chat
- use the real Harness CLI command shapes
- prefer executing the bootstrap flow over merely describing it

Required workflow:
1. If there is any uncertainty about a command shape, inspect `harness <group> --help` or `harness <group> <command> --help` first.
2. Use `harness mission launch --title <title> --goal <goal> [--feature <name>] [--every 30m] [--run-now]` to create durable workflow state.
3. Reuse ids emitted by `mission launch`, especially `mission_id`. Do not invent ids or flags.
4. After `mission launch`, inspect continuity with `harness resume show`. Do not call `harness resume init` after launch unless the resume file is actually missing.
5. Inspect mission state with exact forms:
   - `harness mission show <mission_id>`
   - `harness mission show-contract --mission <mission_id>`
   - `harness mission summarize --mission <mission_id>`
6. Inspect future execution state with:
   - `harness scheduler list`
   - `harness scheduler list-runs`
7. Inspect human gates when relevant:
   - `harness approvals list`
   - `harness evidence list`
8. Mention contracts or tips only when they materially affect the workflow:
   - `harness contracts list|test`
   - `harness tips list|test`

Execution rules:
- Do the bootstrap work when feasible; do not stop at a conceptual explanation.
- Keep the final user-facing summary brief and concrete.
- Report which mission, scheduler, resume, and gate artifacts you created or inspected.
""".strip()


_RESEARCH_SYSTEM_PROMPT = """
Answer by reading and inspecting the codebase or local workspace. Stay read-only.

Use tools to find the answer, but do not edit files or act like this is a bugfix task.
When useful, cite the relevant file paths briefly. Keep the answer short and factual.

When the user asks to be caught up, understand an unfamiliar area, or align their
mental model before work, structure the answer around comprehension: mental
model, file/component map, flow or trace, local conventions, evidence inspected,
and any important next questions.

When the user asks for a context packet or agent onboarding context, produce
targeted, conflict-aware context rather than a raw search dump. Check sources of
truth, local patterns, visible permission/data boundaries, and validation risks,
then return only the compact packet the next agent needs.
""".strip()


@dataclass(frozen=True, slots=True)
class _ChatTurnPolicy:
    name: str
    allowed_tools: tuple[str, ...] | None = None
    system_prompt: str | None = None
    profile: str = "minimal"
    suppress_tool_trace: bool = False
    disable_spawn_agents: bool = False
    disable_grounding_verify: bool = False
    disable_verify: bool = False


@dataclass(slots=True)
class _ConversationState:
    key: str
    session_id: str
    label: str
    task_id: str | None
    agent: Agent
    policy: _ChatTurnPolicy
    render_adapter: _ChatRenderAdapter
    first_turn: bool
    runner: asyncio.Task[None] | None = None
    execution: _ExecutionState = field(default_factory=lambda: _ExecutionState())


@dataclass(slots=True)
class _ExecutionState:
    edited_files: set[str] = field(default_factory=set)
    pending_verification: bool = False
    verification_attempted: bool = False
    verification_passed: bool = False
    last_verify_command: str | None = None
    last_verify_error: str | None = None
    active_hypothesis: str | None = None
    active_goal: str | None = None
    recent_tool_outcomes: list[str] = field(default_factory=list)
    adjacent_review_done: bool = False


_GENERAL_TURN_POLICY = _ChatTurnPolicy(name="general")
_GENERAL_CONTEXT_TURN_POLICY = _ChatTurnPolicy(name="general-context")
_RESEARCH_TURN_POLICY = _ChatTurnPolicy(
    name="research",
    allowed_tools=("read_file", "list_dir", "glob", "shell", "fetch_url", "web_search"),
    system_prompt=_RESEARCH_SYSTEM_PROMPT,
    profile="bare",
    disable_spawn_agents=True,
    disable_verify=True,
)
_REVIEW_TURN_POLICY = _ChatTurnPolicy(
    name="review",
    allowed_tools=("read_file", "list_dir", "glob", "fetch_url", "web_search"),
    system_prompt=_REVIEW_SYSTEM_PROMPT,
    profile="bare",
    suppress_tool_trace=True,
    disable_spawn_agents=True,
    disable_grounding_verify=True,
)
_WORKFLOW_TURN_POLICY = _ChatTurnPolicy(
    name="workflow",
    system_prompt=_WORKFLOW_BOOTSTRAP_SYSTEM_PROMPT,
    profile="bare",
    disable_spawn_agents=True,
    disable_verify=True,
)


def _is_general_execution_policy(policy: _ChatTurnPolicy) -> bool:
    return policy in {_GENERAL_TURN_POLICY, _GENERAL_CONTEXT_TURN_POLICY}


def _handoff_tool_name_for_policy(policy: _ChatTurnPolicy) -> str | None:
    if policy == _RESEARCH_TURN_POLICY:
        return "handoff_to_research_specialist"
    if policy == _REVIEW_TURN_POLICY:
        return "handoff_to_review_specialist"
    if policy == _WORKFLOW_TURN_POLICY:
        return "handoff_to_workflow_specialist"
    return None


def _inject_handoff_routing_directive(prompt: str, policy: _ChatTurnPolicy) -> str:
    tool_name = _handoff_tool_name_for_policy(policy)
    if tool_name is None:
        return prompt
    return (
        "SYSTEM ROUTING DIRECTIVE (not user-visible): "
        f"This request is best handled by `{tool_name}`. "
        f"Call `{tool_name}` with a concise reason, then let the specialist complete the task.\n\n"
        f"User request:\n{prompt}"
    )


def _route_label_for_policy(policy: _ChatTurnPolicy) -> str | None:
    if policy == _RESEARCH_TURN_POLICY:
        return "research specialist"
    if policy == _REVIEW_TURN_POLICY:
        return "review specialist"
    if policy == _WORKFLOW_TURN_POLICY:
        return "workflow specialist"
    return None


def _retry_prompt_for_policy(prompt: str, policy: _ChatTurnPolicy) -> str:
    if policy == _WORKFLOW_TURN_POLICY:
        prefix = (
            "The previous attempt did not produce visible workflow output or tool execution. "
            "Act now: inspect help if needed, run the relevant Harness primitives, and report the resulting artifacts."
        )
    elif policy == _REVIEW_TURN_POLICY:
        prefix = (
            "The previous attempt did not produce visible review output. "
            "Inspect the relevant files, identify the top issues, and answer directly."
        )
    elif policy == _RESEARCH_TURN_POLICY:
        prefix = (
            "The previous attempt did not produce visible research output. "
            "Read the relevant files and answer the question directly with file citations."
        )
    else:
        prefix = (
            "The previous attempt produced no visible output. Complete the user's request directly."
        )
    return f"{prefix}\n\nOriginal request:\n{prompt}"


def _inject_general_task_scope_directive(prompt: str) -> str:
    return (
        "SYSTEM TASK-SCOPING DIRECTIVE (not user-visible): "
        "For broad coding requests, pick the smallest relevant existing tracked files and focused tests first. "
        "Prefer editing an existing file over creating a new file when the current repo already has an obvious home for the change. "
        "Do not create new fixtures, scratch directories, or parallel test trees unless the user explicitly asked for them or the repo has no suitable existing target. "
        "Avoid drifting into eval fixtures, generated files, or broad repo searches when a narrower existing file is the likely home. "
        "After one or two discovery steps, if there are still multiple materially different valid targets, ask the user a short options question instead of guessing.\n\n"
        f"User request:\n{prompt}"
    )


def _should_prefetch_context_packet(execution: _ExecutionState) -> bool:
    return not (execution.pending_verification and not execution.verification_passed)


def _inject_context_packet_directive(prompt: str, context_packet: str) -> str:
    return (
        "SYSTEM CONTEXT PACKET (not user-visible): "
        "Use this read-only context packet to plan and execute the user request. "
        "Do not treat it as a substitute for checking fresh evidence when needed, "
        "but prefer its sources of truth, boundaries, validation guidance, and conflict notes.\n\n"
        f"{context_packet.strip()}\n\n"
        f"User request and execution directives:\n{prompt}"
    )


def _track_execution_tool_call(
    execution: _ExecutionState,
    event: ToolCallEvent,
) -> None:
    tool_name = event.call.name
    arguments = event.call.arguments
    if tool_name in {"write_file", "edit_file"}:
        raw_path = arguments.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            execution.edited_files.add(raw_path.strip())
        execution.pending_verification = True
        execution.verification_passed = False
    if tool_name == "verify_work":
        execution.verification_attempted = True
        raw_command = arguments.get("command")
        if isinstance(raw_command, str) and raw_command.strip():
            execution.last_verify_command = raw_command.strip()


def _track_execution_tool_result(
    execution: _ExecutionState,
    event: ToolResultEvent,
) -> None:
    result = event.result
    summary = result.content.strip()
    preview = summary.splitlines()[0] if summary else ""
    marker = "error" if result.is_error else "ok"
    if preview:
        execution.recent_tool_outcomes.append(f"{result.name}:{marker}:{preview}")
        execution.recent_tool_outcomes = execution.recent_tool_outcomes[-5:]
    if result.name in {"write_file", "edit_file"} and not result.is_error:
        execution.pending_verification = True
        execution.verification_passed = False
        return
    if result.name != "verify_work":
        return
    execution.verification_attempted = True
    if not result.is_error and "PASSED" in result.content.upper():
        execution.pending_verification = False
        execution.verification_passed = True
        execution.last_verify_error = None
        execution.active_hypothesis = None
        return
    execution.pending_verification = bool(execution.edited_files)
    execution.verification_passed = False
    execution.last_verify_error = summary.splitlines()[0] if summary else "verification failed"
    execution.active_hypothesis = execution.last_verify_error


def _inject_active_work_directive(prompt: str, execution: _ExecutionState) -> str:
    if not execution.pending_verification or execution.verification_passed:
        return prompt
    edited = ", ".join(f"`{path}`" for path in sorted(execution.edited_files)[:5])
    goal = execution.active_goal or "the current task"
    hypothesis = execution.active_hypothesis or execution.last_verify_error
    recent = "; ".join(execution.recent_tool_outcomes[-3:])

    directive = (
        "SYSTEM CONTINUITY DIRECTIVE (not user-visible): "
        f"You already have unfinished active work on {goal}. "
        f"Files already changed: {edited or 'unknown'}. "
        f"{f'Active hypothesis: {hypothesis}. ' if hypothesis else ''}"
        f"{f'Recent tool outcomes: {recent}. ' if recent else ''}"
        "Do not lose the thread. Continue that work, verify it, and only then address any extra user request.\n\n"
    )

    if execution.last_verify_command:
        directive += f"Verification is still required for `{execution.last_verify_command}`. "

    return directive + f"User request:\n{prompt}"


def _build_autonomous_followup_prompt(execution: _ExecutionState) -> str | None:
    if not execution.pending_verification or execution.verification_passed:
        return None
    edited = ", ".join(f"`{path}`" for path in sorted(execution.edited_files)[:5])
    goal = execution.active_goal or "the active task"
    hypothesis = execution.active_hypothesis or execution.last_verify_error
    recent = "; ".join(execution.recent_tool_outcomes[-3:])
    if execution.last_verify_command:
        failure = execution.last_verify_error or "the last verification attempt did not pass"
        return (
            "SYSTEM CONTINUITY DIRECTIVE (not user-visible): "
            f"You already changed {edited or 'workspace files'} while working on {goal}. "
            f"Verification is still incomplete because `{execution.last_verify_command}` failed: {failure}. "
            f"{f'Current hypothesis: {hypothesis}. ' if hypothesis else ''}"
            f"{f'Recent tool outcomes: {recent}. ' if recent else ''}"
            "Inspect the failure, fix the relevant files, then call verify_work again. "
            "Repeat until verify_work returns PASSED or you hit a concrete blocker."
        )
    return (
        "SYSTEM CONTINUITY DIRECTIVE (not user-visible): "
        f"You already changed {edited or 'workspace files'} while working on {goal}. "
        f"{f'Recent tool outcomes: {recent}. ' if recent else ''}"
        "The task is not complete until verification passes. "
        "Choose the most relevant focused verify_work command now, call verify_work, "
        "repair any failures, and continue until verify_work returns PASSED or you hit a concrete blocker."
    )


def _build_adjacent_review_prompt(execution: _ExecutionState) -> str | None:
    if not execution.verification_passed or not execution.edited_files:
        return None
    edited = ", ".join(f"`{path}`" for path in sorted(execution.edited_files)[:5])
    goal = execution.active_goal or "the recent task"
    return (
        "Inspect the nearby code around the recent verified change. Stay read-only and keep this bounded.\n"
        f"Primary task just completed: {goal}\n"
        f"Edited files: {edited}\n\n"
        "Look for only adjacent issues or opportunities:\n"
        "- missing or weak nearby tests\n"
        "- repeated logic that should probably be centralized\n"
        "- inconsistencies in nearby files touched by the same concept\n"
        "- follow-on risks exposed by the change\n\n"
        "Do not propose a broad rewrite. Return only the top 0-3 high-signal nearby findings."
    )


def _format_code_review_output(text: str) -> Markdown | None:
    payload: dict[str, Any] | None = None
    candidates = [text.strip()]
    stripped = text.strip()
    if "```" in stripped:
        fence_parts = stripped.split("```")
        for part in fence_parts:
            part = part.strip()
            if not part:
                continue
            if part.startswith("json"):
                part = part[4:].strip()
            candidates.append(part)

    brace_start = stripped.find("{")
    brace_end = stripped.rfind("}")
    if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
        candidates.append(stripped[brace_start : brace_end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            payload = parsed
            break
    if payload is None:
        return None

    summary = str(payload.get("summary") or "").strip()
    findings_raw = payload.get("findings")
    findings = findings_raw if isinstance(findings_raw, list) else []

    lines: list[str] = []
    if summary:
        lines.append(summary)

    if findings:
        if lines:
            lines.append("")
        for idx, item in enumerate(findings[:3], start=1):
            if not isinstance(item, dict):
                continue
            severity = str(item.get("severity") or "").strip().lower() or "unknown"
            file = str(item.get("file") or "").strip()
            line = item.get("line")
            issue = str(item.get("issue") or "").strip()
            rationale = str(item.get("rationale") or "").strip()

            location = file
            if file and isinstance(line, int):
                location = f"{file}:{line}"
            elif isinstance(line, int):
                location = f"line {line}"
            prefix = f"**{idx}. [{severity}]**"
            if location:
                prefix += f" `{location}`"
            if issue:
                prefix += f" {issue}"
            lines.append(prefix)
            if rationale:
                lines.append(f"   {rationale}")
    elif not lines:
        lines.append("No review findings.")

    return Markdown("\n".join(lines))


def _extract_keyed_artifacts(text: str) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key and value:
            artifacts[key] = value
    return artifacts


def _format_workflow_bootstrap_fallback(tool_results: list[str]) -> Markdown | None:
    if not tool_results:
        return None
    artifacts: dict[str, str] = {}
    for result in tool_results:
        artifacts.update(_extract_keyed_artifacts(result))

    mission_id = artifacts.get("mission_id")
    task_ref = artifacts.get("task_ref")
    feature_id = artifacts.get("feature_id")
    scheduler_job_id = artifacts.get("scheduler_job_id")
    resume_feature = artifacts.get("resume_feature") or artifacts.get("feature")

    if not any((mission_id, task_ref, feature_id, scheduler_job_id, resume_feature)):
        return None

    lines = ["Harness bootstrapped the durable workflow."]
    lines.append("")
    if mission_id:
        lines.append(f"- Mission: `{mission_id}`")
    if task_ref:
        lines.append(f"- Task: `{task_ref}`")
    if resume_feature:
        lines.append(f"- Resume feature: `{resume_feature}`")
    if feature_id:
        lines.append(f"- Feature: `{feature_id}`")
    if scheduler_job_id:
        lines.append(f"- Scheduler job: `{scheduler_job_id}`")
    return Markdown("\n".join(lines))


class _ChatRenderAdapter:
    def __init__(self, *, console: Console, default_render: Callable[[Any], None]) -> None:
        self._console = console
        self._default_render = default_render
        self._policy = _GENERAL_TURN_POLICY
        self._text_buf = ""
        self._tool_results: list[str] = []

    def set_policy(self, policy: _ChatTurnPolicy) -> None:
        self._policy = policy
        self._text_buf = ""
        self._tool_results = []

    def render(self, event: Any) -> None:
        if isinstance(event, HandoffEvent):
            self._console.print()
            self._console.print(
                f"[magenta]⇢ handoff[/magenta] [bold]{event.target_name}[/bold]"
                f" [dim]({event.reason})[/dim]"
            )
            return
        if not self._policy.suppress_tool_trace:
            if isinstance(event, ToolResultEvent) and event.result.content:
                self._tool_results.append(event.result.content)
            if isinstance(event, Done) and event.final_message is not None:
                content = event.final_message.content
                if isinstance(content, str):
                    rendered = _format_code_review_output(content)
                    if rendered is not None:
                        self._console.print()
                        self._console.print(rendered)
                        if event.usage:
                            u = event.usage
                            self._console.print(
                                f"[dim]tokens: {u.prompt_tokens:,} in / {u.completion_tokens:,} out[/dim]"
                            )
                        return
                if self._policy.name == "workflow":
                    rendered = _format_workflow_bootstrap_fallback(self._tool_results)
                    if rendered is not None:
                        self._console.print()
                        self._console.print(rendered)
                        if event.usage:
                            u = event.usage
                            self._console.print(
                                f"[dim]tokens: {u.prompt_tokens:,} in / {u.completion_tokens:,} out[/dim]"
                            )
                        return
            self._default_render(event)
            return

        if isinstance(event, TextDelta):
            self._text_buf += event.text
            return
        if isinstance(event, ToolResultEvent) and event.result.content:
            self._tool_results.append(event.result.content)
        if isinstance(event, ToolCallEvent | ToolResultEvent | StepStarted | StepCompleted):
            return
        if isinstance(event, Done):
            text = self._text_buf
            self._text_buf = ""
            if (
                not text
                and event.final_message is not None
                and isinstance(event.final_message.content, str)
            ):
                text = event.final_message.content
            rendered = _format_code_review_output(text)
            if rendered is not None:
                self._console.print()
                self._console.print(rendered)
                if event.usage:
                    u = event.usage
                    self._console.print(
                        f"[dim]tokens: {u.prompt_tokens:,} in / {u.completion_tokens:,} out[/dim]"
                    )
                return
            if self._policy.name == "workflow":
                rendered = _format_workflow_bootstrap_fallback(self._tool_results)
                if rendered is not None:
                    self._console.print()
                    self._console.print(rendered)
                    if event.usage:
                        u = event.usage
                        self._console.print(
                            f"[dim]tokens: {u.prompt_tokens:,} in / {u.completion_tokens:,} out[/dim]"
                        )
                    return
            if text:
                self._console.print()
                self._console.print(Markdown(text))
                if event.usage:
                    u = event.usage
                    self._console.print(
                        f"[dim]tokens: {u.prompt_tokens:,} in / {u.completion_tokens:,} out[/dim]"
                    )
                return
        if isinstance(event, ErrorEvent):
            self._text_buf = ""
            self._default_render(event)
            return
        self._default_render(event)


async def _classify_chat_turn_policy(
    *,
    adapter: Any,
    model: str,
    prompt: str,
) -> _ChatTurnPolicy:
    messages = [
        Message(
            role="system",
            content=(
                "You are a routing classifier for an interactive general-purpose work agent. "
                "Choose exactly one behavior label.\n"
                "- Return `review` when the user asks to review, audit, inspect, or find issues in code.\n"
                "- Return `research` when the user asks a read-only question about the repo, such as where something lives, how something works, what a file does, or to research/explain/cite code without editing it.\n"
                "- Return `research` when the user asks to be caught up, build a mental model, understand architecture/conventions/testing/history, or act as a new contributor before making changes.\n"
                "- Return `research` when the user only asks for a context packet, source-of-truth map, conflict-aware context, or agent onboarding context before work.\n"
                "- Return `workflow` when the user asks to set up long-running, durable, resumable, repeated, scheduled, reminder, or mission-style work in Harness.\n"
                "- Return `general-context` when the user asks for hands-on work and the work is broad, open-ended, unfamiliar, cross-cutting, or explicitly asks to use context before execution.\n"
                "- Return `general` when the user asks to build, edit, debug, explain, run, hand off, delegate, or otherwise do hands-on task work.\n"
                "- If the user explicitly asks for a specialist handoff or delegation, return `general` so the main agent can use handoff tools.\n"
                "Return only one token: general, general-context, research, review, or workflow."
            ),
        ),
        Message(role="user", content=prompt),
    ]
    final_text = ""
    async for event in adapter.stream(model=model, messages=messages, max_tokens=16):
        if (
            getattr(event, "type", "") == "done"
            and getattr(event, "final_message", None) is not None
        ):
            content = event.final_message.content
            if isinstance(content, str):
                final_text = content.strip().lower()
    if "general-context" in final_text or "general_context" in final_text:
        return _GENERAL_CONTEXT_TURN_POLICY
    if "workflow" in final_text:
        return _WORKFLOW_TURN_POLICY
    if "research" in final_text:
        return _RESEARCH_TURN_POLICY
    if "review" in final_text:
        return _REVIEW_TURN_POLICY
    return _GENERAL_TURN_POLICY


def run_chat_command(
    *,
    model: str | None,
    provider: str | None,
    base_url: str | None,
    cwd: Path | None,
    db: Path | None,
    in_memory: bool,
    session_id: str | None,
    task_ref: str | None,
    max_steps: int,
    failover: str | None,
    yes: bool,
    inbox: bool,
    verify: str | None,
    require_tools: bool,
    max_context_tokens: int | None,
    config_path: Path | None,
    auto_compact: bool,
    verbose: bool,
    console: Console,
    configure_logging: Callable[..., None],
    load_cli_config: Callable[[Path | None], HarnessConfig],
    resolve_chain: Callable[..., list[str]],
    run_async: Callable[[Any], Any],
    build_storage: Callable[..., Storage],
    resolve_task_attachment: Callable[..., Any],
    build_verifier: Callable[..., Any],
    build_adapter: Callable[..., Any],
    build_tools: Callable[..., Any],
    build_agent: Callable[..., Agent],
    render: Callable[[Any], None],
    render_session_diff: Callable[[Any, Console], None],
    default_system_prompt: str,
) -> None:
    configure_logging(level="DEBUG" if verbose else "INFO")
    if not yes and os.environ.get("HARNESS_YES"):
        yes = True
    if verify == "none":
        verify = None
    cfg = load_cli_config(config_path)
    chain = resolve_chain(failover_flag=failover, provider_flag=provider, config=cfg)
    effective_model = model or cfg.default_model or "llama3.2"
    working_dir = (cwd or Path.cwd()).resolve()
    if not working_dir.exists() or not working_dir.is_dir():
        console.print(f"[red]--cwd does not exist or is not a directory: {working_dir}[/red]")
        raise typer.Exit(2)

    try:
        run_async(
            chat_loop(
                chain=chain,
                base_url=base_url,
                model=effective_model,
                cwd=working_dir,
                db=db,
                in_memory=in_memory,
                session_id=session_id,
                task_ref=task_ref,
                max_steps=max_steps,
                yes=yes,
                inbox=inbox,
                verify=verify,
                require_tools=require_tools,
                max_context_tokens=max_context_tokens,
                auto_compact=auto_compact,
                config=cfg,
                console=console,
                build_storage=build_storage,
                resolve_task_attachment=resolve_task_attachment,
                build_verifier=build_verifier,
                build_adapter=build_adapter,
                build_tools=build_tools,
                build_agent=build_agent,
                render=render,
                render_session_diff=render_session_diff,
                default_system_prompt=default_system_prompt,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]bye[/yellow]")
        raise typer.Exit(130) from None


async def chat_loop(
    *,
    chain: list[str],
    base_url: str | None,
    model: str,
    cwd: Path,
    db: Path | None,
    in_memory: bool,
    session_id: str | None,
    task_ref: str | None,
    max_steps: int,
    yes: bool,
    inbox: bool,
    verify: str | None,
    require_tools: bool = False,
    max_context_tokens: int | None,
    auto_compact: bool = False,
    config: HarnessConfig,
    console: Console,
    build_storage: Callable[..., Storage],
    resolve_task_attachment: Callable[..., Any],
    build_verifier: Callable[..., Any],
    build_adapter: Callable[..., Any],
    build_tools: Callable[..., Any],
    build_agent: Callable[..., Agent],
    render: Callable[[Any], None],
    render_session_diff: Callable[[Any, Console], None],
    default_system_prompt: str,
) -> None:
    from uuid import uuid4

    storage = build_storage(db=db, in_memory=in_memory, cwd=cwd)
    try:
        budget = (
            ContextBudget(max_tokens=max_context_tokens) if max_context_tokens is not None else None
        )
        compactor: ContextCompactor | None = None
        if auto_compact:
            adapter = build_adapter(chain[0], base_url=base_url, config=config)
            compactor = ContextCompactor(adapter=adapter, model=model)
        classifier_adapter = build_adapter(chain[0], base_url=base_url, config=config)
        specialist_cache: dict[str, Agent] = {}

        def get_specialist_agent(policy: _ChatTurnPolicy) -> Agent:
            cache_key = f"policy:{policy.name}"
            specialist = specialist_cache.get(cache_key)
            if specialist is None:
                specialist = build_chat_agent(policy, allow_handoffs=False)
                specialist_cache[cache_key] = specialist
            return specialist

        def build_chat_agent(policy: _ChatTurnPolicy, *, allow_handoffs: bool = True) -> Agent:
            effective_verify = verify
            if policy.disable_verify:
                effective_verify = None
            if policy.disable_grounding_verify and effective_verify == "grounding":
                effective_verify = None
            verifier = build_verifier(
                effective_verify,
                chain=chain,
                model=model,
                config=config,
                build_adapter=build_adapter,
                cwd=cwd,
            )
            allowed_tools = set(policy.allowed_tools) if policy.allowed_tools else None

            def scoped_build_tools(tool_cwd: Path) -> Any:
                return build_tools(tool_cwd, config=config, include=allowed_tools)

            built_agent = build_agent(
                chain=chain,
                base_url=base_url,
                model=model,
                storage=storage,
                cwd=cwd,
                config=config,
                yes=yes,
                build_tools=scoped_build_tools,
                inbox=inbox,
                activity_store=storage,  # type: ignore[arg-type]
                approval_store=storage,  # type: ignore[arg-type]
                verifier=verifier,
                budget=budget,
                memory_store=storage,  # type: ignore[arg-type]
                system_prompt=policy.system_prompt or default_system_prompt,
                compactor=compactor,
                profile=policy.profile,
            )
            if policy.disable_spawn_agents:
                cast(Any, built_agent.tools).unregister("spawn_agents")
            if allow_handoffs and _is_general_execution_policy(policy):
                specialist_specs = (
                    (
                        "handoff_to_research_specialist",
                        _RESEARCH_TURN_POLICY,
                        "Hand off a read-only repository research or explanation task to the research specialist.",
                    ),
                    (
                        "handoff_to_review_specialist",
                        _REVIEW_TURN_POLICY,
                        "Hand off a read-only code review or audit task to the review specialist.",
                    ),
                    (
                        "handoff_to_workflow_specialist",
                        _WORKFLOW_TURN_POLICY,
                        "Hand off durable workflow bootstrap or orchestration work to the workflow specialist.",
                    ),
                )
                for tool_name, specialist_policy, description in specialist_specs:
                    specialist = get_specialist_agent(specialist_policy)
                    built_agent.tools.register(
                        cast(
                            Any,
                            HandoffTool(
                                specialist,
                                name=tool_name,
                                description=description,
                            ),
                        )
                    )
            return built_agent

        async def make_conversation(
            *,
            requested_session_id: str | None,
            label: str | None = None,
        ) -> _ConversationState:
            existing: Session | None = None
            if requested_session_id:
                existing = await storage.get(requested_session_id)
            resolved_session_id = requested_session_id or f"sess_{uuid4().hex[:12]}"
            task_id, _task = await resolve_task_attachment(storage, task_ref, resolved_session_id)
            policy = _GENERAL_TURN_POLICY
            render_adapter = _ChatRenderAdapter(console=console, default_render=render)
            render_adapter.set_policy(policy)
            return _ConversationState(
                key=label or resolved_session_id,
                session_id=resolved_session_id,
                label=label or resolved_session_id,
                task_id=task_id,
                agent=build_chat_agent(policy),
                policy=policy,
                render_adapter=render_adapter,
                first_turn=existing is None,
            )

        def print_turn_banner(conversation: _ConversationState, *, background: bool) -> None:
            mode = "background" if background else "active"
            console.print(
                f"\n[bold cyan]{escape(conversation.label)}[/bold cyan] [dim]{mode} turn[/dim]"
            )

        async def build_context_packet_for_turn(
            conversation: _ConversationState,
            prompt: str,
        ) -> str | None:
            console.print(
                "[cyan]↻ context packet[/cyan] "
                "[dim]building read-only repo context before execution[/dim]"
            )
            research_agent = get_specialist_agent(_RESEARCH_TURN_POLICY)
            research_render = _ChatRenderAdapter(console=console, default_render=render)
            research_render.set_policy(_RESEARCH_TURN_POLICY)
            context_session_id = f"{conversation.session_id}_context_{uuid4().hex[:8]}"
            context_request = RunRequest(
                prompt=build_context_packet_prompt(task=prompt),
                session_id=context_session_id,
                model=model,
                max_steps=min(max_steps, 18),
                require_tool_use=False,
            )
            text_deltas: list[str] = []
            final_text: str | None = None
            saw_error = False
            try:
                async for event in research_agent.run(context_request):
                    if isinstance(event, TextDelta):
                        text_deltas.append(event.text)
                    if (
                        isinstance(event, Done)
                        and event.final_message is not None
                        and isinstance(event.final_message.content, str)
                    ):
                        final_text = event.final_message.content
                    if isinstance(event, ErrorEvent):
                        saw_error = True
                    research_render.render(event)
            except Exception as exc:
                console.print(
                    f"[yellow]context packet skipped[/yellow] [dim]({escape(str(exc))})[/dim]"
                )
                return None
            packet = (final_text or "".join(text_deltas)).strip()
            if saw_error or not packet:
                return None
            return packet

        async def run_turn(
            conversation: _ConversationState,
            *,
            prompt: str,
            background: bool,
        ) -> None:
            try:
                conversation.execution.active_goal = prompt
                inferred_policy = await _classify_chat_turn_policy(
                    adapter=classifier_adapter,
                    model=model,
                    prompt=prompt,
                )
                if inferred_policy != conversation.policy:
                    conversation.policy = inferred_policy
                    conversation.agent = build_chat_agent(inferred_policy)
                    conversation.render_adapter.set_policy(inferred_policy)
                effective_agent = conversation.agent
                effective_prompt = prompt
                if _is_general_execution_policy(inferred_policy):
                    effective_prompt = _inject_active_work_directive(prompt, conversation.execution)
                    if effective_prompt == prompt:
                        effective_prompt = _inject_general_task_scope_directive(prompt)
                route_label = _route_label_for_policy(inferred_policy)
                print_turn_banner(conversation, background=background)
                if route_label is not None:
                    console.print(f"[magenta]⇢ routed[/magenta] [bold]{route_label}[/bold]")

                if (
                    inferred_policy == _GENERAL_CONTEXT_TURN_POLICY
                    and _should_prefetch_context_packet(conversation.execution)
                ):
                    context_packet = await build_context_packet_for_turn(conversation, prompt)
                    if context_packet:
                        effective_prompt = _inject_context_packet_directive(
                            effective_prompt,
                            context_packet,
                        )

                first_attempt = conversation.first_turn
                route_attempts = 2 if route_label is not None else 1
                current_prompt = effective_prompt
                auto_followups = 0
                while True:
                    saw_text = False
                    saw_tool_result = False
                    saw_error = False
                    saw_done_content = False
                    route_visible_work = False
                    for attempt_index in range(route_attempts):
                        if first_attempt:
                            request_kwargs: dict[str, object] = {
                                "prompt": current_prompt,
                                "session_id": conversation.session_id,
                                "model": model,
                                "max_steps": max_steps,
                                "require_tool_use": require_tools,
                            }
                            if conversation.task_id:
                                request_kwargs["task_id"] = conversation.task_id
                            request = RunRequest(**request_kwargs)  # type: ignore[arg-type]
                            stream = effective_agent.run(request)
                        else:
                            stream = effective_agent.resume(
                                conversation.session_id,
                                prompt=current_prompt,
                                max_steps=max_steps,
                            )
                        async for event in stream:
                            if isinstance(event, TextDelta) and event.text.strip():
                                saw_text = True
                            if isinstance(event, ToolCallEvent):
                                _track_execution_tool_call(conversation.execution, event)
                            if isinstance(event, ToolResultEvent):
                                saw_tool_result = True
                                _track_execution_tool_result(conversation.execution, event)
                            if isinstance(event, ErrorEvent):
                                saw_error = True
                            if (
                                isinstance(event, Done)
                                and event.final_message is not None
                                and isinstance(event.final_message.content, str)
                                and event.final_message.content.strip()
                            ):
                                saw_done_content = True
                            conversation.render_adapter.render(event)
                        first_attempt = False
                        conversation.first_turn = False
                        route_visible_work = (
                            saw_error or saw_tool_result or saw_text or saw_done_content
                        )
                        if (
                            route_label is None
                            or route_visible_work
                            or attempt_index + 1 >= route_attempts
                        ):
                            break
                        console.print(
                            f"[yellow]↻ retrying[/yellow] [bold]{route_label}[/bold] "
                            "[dim](previous attempt produced no visible work)[/dim]"
                        )
                        current_prompt = _retry_prompt_for_policy(prompt, inferred_policy)

                    if not _is_general_execution_policy(inferred_policy):
                        break
                    if saw_error:
                        break
                    followup_prompt = _build_autonomous_followup_prompt(conversation.execution)
                    if followup_prompt is None or auto_followups >= 3:
                        break
                    auto_followups += 1
                    console.print(
                        "[cyan]↻ continuing[/cyan] [dim]active work is not verified yet; "
                        "running the next relevant gate automatically[/dim]"
                    )
                    current_prompt = followup_prompt

                adjacent_review_prompt = _build_adjacent_review_prompt(conversation.execution)
                if (
                    _is_general_execution_policy(inferred_policy)
                    and adjacent_review_prompt is not None
                    and not conversation.execution.adjacent_review_done
                ):
                    conversation.execution.adjacent_review_done = True
                    console.print(
                        "[cyan]↻ adjacent review[/cyan] [dim]checking nearby code for gaps or follow-on opportunities[/dim]"
                    )
                    review_agent = get_specialist_agent(_REVIEW_TURN_POLICY)
                    review_render = _ChatRenderAdapter(console=console, default_render=render)
                    review_render.set_policy(_REVIEW_TURN_POLICY)
                    review_session_id = f"{conversation.session_id}_adjacent_review"
                    review_request = RunRequest(
                        prompt=adjacent_review_prompt,
                        session_id=review_session_id,
                        model=model,
                        max_steps=max_steps,
                        require_tool_use=False,
                    )
                    async for event in review_agent.run(review_request):
                        review_render.render(event)
            except (KeyboardInterrupt, asyncio.CancelledError):
                console.print(f"\n[yellow][{conversation.label}] cancelled[/yellow]")
                raise
            except Exception as exc:
                console.print(f"\n[red][{conversation.label}] Error:[/red] {exc!s}")
            finally:
                conversation.runner = None

        conversations: dict[str, _ConversationState] = {}
        initial_conversation = await make_conversation(
            requested_session_id=session_id,
            label=session_id,
        )
        conversations[initial_conversation.key] = initial_conversation
        current_key = initial_conversation.key

        chain_label = chain[0]
        if len(chain) > 1:
            chain_label += "  [dim](failover: " + ", ".join(chain[1:]) + ")[/dim]"
        intro = (
            f"[bold]Session:[/bold] {initial_conversation.session_id}"
            + f"\n[bold]Provider:[/bold] {chain_label}"
            f"\n[bold]Model:[/bold] {model}"
            f"\n[bold]CWD:[/bold] {cwd}\n\n"
            f"[dim]Type /help for commands. /quit to exit.[/dim]"
        )
        console.print(Panel(intro, title="harness chat", expand=False))

        def set_current(value: str) -> None:
            nonlocal current_key
            current_key = value

        slash_handler = _make_slash_handler(
            console=console,
            render_session_diff=render_session_diff,
            storage=storage,
            conversations=conversations,
            get_current_key=lambda: current_key,
            set_current_key=set_current,
            create_conversation=make_conversation,
            launch_background_turn=run_turn,
        )

        while True:
            try:
                prompt_label = conversations[current_key].label
                user_input = (
                    await asyncio.to_thread(
                        console.input, f"\n[bold cyan]{prompt_label}> [/bold cyan]"
                    )
                ).strip()
            except EOFError:
                for conversation in conversations.values():
                    if conversation.runner is not None:
                        conversation.runner.cancel()
                console.print("\n[yellow]bye[/yellow]")
                return
            except KeyboardInterrupt:
                for conversation in conversations.values():
                    if conversation.runner is not None:
                        conversation.runner.cancel()
                console.print("\n[yellow]bye[/yellow]")
                return

            if not user_input:
                continue

            if user_input.startswith("/"):
                keep_going = await slash_handler(user_input)
                if not keep_going:
                    for conversation in conversations.values():
                        if conversation.runner is not None:
                            conversation.runner.cancel()
                    return
                continue

            conversation = conversations[current_key]
            if conversation.runner is not None:
                console.print(
                    f"[yellow]{conversation.label} is already running. "
                    "Use /send on another conversation, /new, or wait for it to finish.[/yellow]"
                )
                continue
            await run_turn(conversation, prompt=user_input, background=False)
    finally:
        if isinstance(storage, SQLiteStorage):
            await storage.close()


def _make_slash_handler(
    *,
    console: Console,
    render_session_diff: Callable[[Any, Console], None],
    storage: Storage,
    conversations: dict[str, _ConversationState],
    get_current_key: Callable[[], str],
    set_current_key: Callable[[str], None],
    create_conversation: Callable[..., Awaitable[_ConversationState]],
    launch_background_turn: Callable[..., Coroutine[Any, Any, None]],
) -> Callable[[str], Awaitable[bool]]:
    registry: dict[str, SlashHandler] = {}

    def slash(name: str) -> Callable[[SlashHandler], SlashHandler]:
        def decorator(fn: SlashHandler) -> SlashHandler:
            registry[name] = fn
            return fn

        return decorator

    @slash("/quit")
    @slash("/exit")
    @slash("/q")
    async def slash_quit(line: str) -> bool:
        console.print("[yellow]bye[/yellow]")
        return False

    @slash("/help")
    async def slash_help(line: str) -> bool:
        console.print(Panel(_HELP_TEXT.rstrip(), title="commands", expand=False))
        return True

    @slash("/tools")
    async def slash_tools(line: str) -> bool:
        conversation = conversations[get_current_key()]
        agent = conversation.agent
        table = Table(show_header=True, header_style="bold")
        table.add_column("Tool")
        table.add_column("Approval")
        for tool in agent.tools.all():
            effective = agent.approval_policy.decide(tool)
            color = {"auto": "green", "prompt": "yellow", "deny": "red"}.get(effective, "white")
            table.add_row(tool.name, f"[{color}]{effective}[/{color}]")
        console.print(table)
        return True

    @slash("/session")
    async def slash_session(line: str) -> bool:
        conversation = conversations[get_current_key()]
        session = await storage.get(conversation.session_id)
        if session is None:
            console.print(f"[dim]Session {conversation.session_id} (no turns yet)[/dim]")
        else:
            console.print(
                f"[dim]Session {conversation.session_id}, status: {session.status}, "
                f"{len(session.messages)} messages[/dim]"
            )
        return True

    @slash("/sessions")
    async def slash_sessions(line: str) -> bool:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Current")
        table.add_column("Conversation")
        table.add_column("Session")
        table.add_column("State")
        for key, conversation in conversations.items():
            current = "*" if key == get_current_key() else ""
            state = "running" if conversation.runner is not None else "idle"
            table.add_row(current, conversation.label, conversation.session_id, state)
        console.print(table)
        return True

    @slash("/new")
    async def slash_new(line: str) -> bool:
        parts = line.split(None, 1)
        label = parts[1].strip() if len(parts) > 1 and parts[1].strip() else None
        conversation = await create_conversation(requested_session_id=None, label=label)
        conversations[conversation.key] = conversation
        set_current_key(conversation.key)
        console.print(
            f"[green]Switched to new conversation:[/green] "
            f"{conversation.label} ({conversation.session_id})"
        )
        return True

    @slash("/switch")
    async def slash_switch(line: str) -> bool:
        parts = line.split(None, 1)
        if len(parts) == 1 or not parts[1].strip():
            console.print("[red]Usage:[/red] /switch <id|name>")
            return True
        target = parts[1].strip()
        conversation = conversations.get(target)
        if conversation is None:
            matched = next(
                (item for item in conversations.values() if item.session_id == target),
                None,
            )
            conversation = matched
        if conversation is None:
            console.print(f"[red]Unknown conversation:[/red] {target}")
            return True
        set_current_key(conversation.key)
        console.print(f"[green]Switched to:[/green] {conversation.label}")
        return True

    @slash("/send")
    async def slash_send(line: str) -> bool:
        parts = line.split(None, 2)
        if len(parts) < 3:
            console.print("[red]Usage:[/red] /send <id|name|.> <message>")
            return True
        target, message = parts[1].strip(), parts[2].strip()
        if not message:
            console.print("[red]Usage:[/red] /send <id|name|.> <message>")
            return True
        if target == ".":
            conversation = conversations[get_current_key()]
        else:
            conversation = conversations.get(target)
            if conversation is None:
                conversation = next(
                    (item for item in conversations.values() if item.session_id == target),
                    None,
                )
        if conversation is None:
            console.print(f"[red]Unknown conversation:[/red] {target}")
            return True
        if conversation.runner is not None:
            console.print(f"[yellow]{conversation.label} is already running.[/yellow]")
            return True
        conversation.runner = asyncio.create_task(
            launch_background_turn(conversation, prompt=message, background=True)
        )
        console.print(f"[green]Started background turn:[/green] {conversation.label}")
        return True

    @slash("/diff")
    async def slash_diff(line: str) -> bool:
        conversation = conversations[get_current_key()]
        activity = await storage.list_activity(session_id=conversation.session_id)  # type: ignore[attr-defined]
        render_session_diff(activity, console)
        return True

    @slash("/clear")
    async def slash_clear(line: str) -> bool:
        console.clear()
        return True

    @slash("/model")
    async def slash_model(line: str) -> bool:
        conversation = conversations[get_current_key()]
        agent = conversation.agent
        parts = line.split(None, 1)
        if len(parts) == 1:
            console.print(f"[dim]Active model: {agent.default_model}[/dim]")
        else:
            new_model = parts[1].strip()
            agent.default_model = new_model
            console.print(f"[green]Switched model to:[/green] {new_model}")
        return True

    async def handle_slash(line: str) -> bool:
        cmd = line.split(None, 1)[0].lower()
        handler = registry.get(cmd)
        if handler is None:
            console.print(f"[red]Unknown command:[/red] {cmd}.  Try /help.")
            return True
        return await handler(line)

    return handle_slash
