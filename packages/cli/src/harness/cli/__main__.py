"""Harness CLI entry point.

Phase 4 surface:
- `harness run "prompt"`         — one-shot prompt with the full tool set
- `harness sessions list`        — list saved sessions
- `harness sessions show <id>`   — print full transcript
- `harness sessions resume <id>` — continue an existing session
- `harness sessions rm <id>`     — delete a session
- `harness version`              — print the installed CLI version

Providers: ollama, codex, openai, openrouter.
Tools: read_file, write_file, edit_file, list_dir, glob, shell, fetch_url.

Config: `$XDG_CONFIG_HOME/harness/config.toml` (or ~/.config/harness/config.toml)
provides defaults for provider, model, per-provider settings, and per-tool
approval levels. CLI flags override the config.

Tool approvals default to `prompt` for any tool that mutates state or makes
network calls; the CLI shows a Rich prompt. Pass `--yes` to auto-approve
everything (handy for non-interactive use), or set approvals in config.
"""

from __future__ import annotations

# Load .env from the working directory (or any parent) before anything reads env vars.
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(override=False)
except ImportError:
    pass
from pathlib import Path
from typing import Annotated, Any

import typer

from harness.adapters.anthropic import AnthropicAdapter
from harness.adapters.codex import CodexAdapter
from harness.adapters.ollama import OllamaAdapter
from harness.adapters.openai import OpenAIAdapter
from harness.adapters.openrouter import OpenRouterAdapter
from harness.cli.approvals_evidence_commands import (
    approvals_deny_command as _approvals_deny_command,
)
from harness.cli.approvals_evidence_commands import (
    approvals_grant_command as _approvals_grant_command,
)
from harness.cli.approvals_evidence_commands import (
    approvals_list_command as _approvals_list_command,
)
from harness.cli.approvals_evidence_commands import (
    approvals_show_command as _approvals_show_command,
)
from harness.cli.approvals_evidence_commands import (
    evidence_list_command as _evidence_list_command,
)
from harness.cli.chat_commands import run_chat_command as _run_chat_command
from harness.cli.common import (
    _ago,
    _build_tools,
    _load_cli_config,
    _resolve_chain,
    _run_async,
    _truncate,
    console,
)
from harness.cli.common import (
    _args_preview as _common_args_preview,
)
from harness.cli.config import HarnessConfig, default_config_path
from harness.cli.docs_commands import docs_audit_command as _docs_audit_command
from harness.cli.evals import eval_app
from harness.cli.experience_commands import experience_app
from harness.cli.gateway_commands import gateway_app
from harness.cli.introspection import plugins_app, providers_app, tools_app
from harness.cli.lab_commands import lab_list_command as _lab_list_command
from harness.cli.lab_commands import lab_resume_command as _lab_resume_command
from harness.cli.lab_commands import lab_run_command as _lab_run_command
from harness.cli.lab_commands import lab_status_command as _lab_status_command
from harness.cli.lifecycle_commands import contracts_list_command as _contracts_list_command
from harness.cli.lifecycle_commands import contracts_test_command as _contracts_test_command
from harness.cli.lifecycle_commands import phase_complete_command as _phase_complete_command
from harness.cli.lifecycle_commands import phase_declare_command as _phase_declare_command
from harness.cli.lifecycle_commands import phase_status_command as _phase_status_command
from harness.cli.lifecycle_commands import resume_add_feature_command as _resume_add_feature_command
from harness.cli.lifecycle_commands import resume_init_command as _resume_init_command
from harness.cli.lifecycle_commands import resume_set_current_command as _resume_set_current_command
from harness.cli.lifecycle_commands import resume_show_command as _resume_show_command
from harness.cli.lifecycle_commands import tips_add_command as _tips_add_command
from harness.cli.lifecycle_commands import tips_list_command as _tips_list_command
from harness.cli.lifecycle_commands import tips_mine_command as _tips_mine_command
from harness.cli.lifecycle_commands import tips_test_command as _tips_test_command
from harness.cli.markdown_render import Renderer
from harness.cli.markdown_render import (
    _preprocess_markdown as _preprocess_markdown_impl,
)
from harness.cli.markdown_render import (
    _render_mermaid as _render_mermaid_impl,
)
from harness.cli.mission_commands import mission_app
from harness.cli.render import (
    _approval_status_style,
    _render_approval,
    _render_session,
    _render_session_diff,
    _render_task,
    _status_style,
    _task_status_style,
)
from harness.cli.research_commands import research_app, vision_app
from harness.cli.review_commands import review_command as _review_command
from harness.cli.run_commands import run_command as _run_command
from harness.cli.run_commands import run_once as _run_once_impl
from harness.cli.runtime_agent import (
    _SPAWN_SCHEMA as _RUNTIME_SPAWN_SCHEMA,
)
from harness.cli.runtime_agent import (
    SpawnAgentsTool as _RuntimeSpawnAgentsTool,
)
from harness.cli.runtime_agent import (
    build_agent as _build_agent_impl,
)
from harness.cli.runtime_agent import (
    load_project_context as _load_project_context_impl,
)
from harness.cli.runtime_helpers import (
    build_critic as _build_critic,
)
from harness.cli.runtime_helpers import (
    build_search_fn as _build_search_fn,
)
from harness.cli.runtime_helpers import (
    build_storage as _build_storage,
)
from harness.cli.runtime_helpers import (
    build_verifier as _build_verifier,
)
from harness.cli.runtime_helpers import (
    print_defense_ledger as _print_defense_ledger,
)
from harness.cli.runtime_helpers import (
    resolve_runtime_strategy as _resolve_runtime_strategy,
)
from harness.cli.runtime_helpers import workspace_db
from harness.cli.scheduler_commands import scheduler_app
from harness.cli.sessions_commands import (
    sessions_diff_command as _sessions_diff_command,
)
from harness.cli.sessions_commands import (
    sessions_fork_command as _sessions_fork_command,
)
from harness.cli.sessions_commands import (
    sessions_list_command as _sessions_list_command,
)
from harness.cli.sessions_commands import (
    sessions_resume_command as _sessions_resume_command,
)
from harness.cli.sessions_commands import (
    sessions_rm_command as _sessions_rm_command,
)
from harness.cli.sessions_commands import (
    sessions_show_command as _sessions_show_command,
)
from harness.cli.tasks_commands import (
    close_if_sqlite as _close_if_sqlite,
)
from harness.cli.tasks_commands import (
    tasks_link_command as _tasks_link_command,
)
from harness.cli.tasks_commands import (
    tasks_list_command as _tasks_list_command,
)
from harness.cli.tasks_commands import (
    tasks_new_command as _tasks_new_command,
)
from harness.cli.tasks_commands import (
    tasks_rm_command as _tasks_rm_command,
)
from harness.cli.tasks_commands import (
    tasks_show_command as _tasks_show_command,
)
from harness.cli.tasks_commands import (
    tasks_update_command as _tasks_update_command,
)
from harness.cli.tune_commands import tune_list_command as _tune_list_command
from harness.cli.tune_commands import tune_propose_command as _tune_propose_command
from harness.cli.tune_commands import tune_rollback_command as _tune_rollback_command
from harness.cli.tune_commands import tune_show_command as _tune_show_command
from harness.cli.workspace_commands import (
    init_workspace as _init_workspace,
)
from harness.cli.workspace_commands import (
    memory_list_command as _memory_list_command,
)
from harness.cli.workspace_commands import (
    memory_rm_command as _memory_rm_command,
)
from harness.cli.workspace_commands import (
    memory_save_command as _memory_save_command,
)
from harness.cli.workspace_commands import (
    memory_search_command as _memory_search_command,
)
from harness.cli.workspace_commands import (
    run_goal_command as _run_goal_command,
)
from harness.core import (
    Adapter,
    Agent,
    ApprovalDecision,
    ApprovalStore,
    ConsequencePredictor,
    ContextBudget,
    Critic,
    MultiAgentOrchestrator,
    Planner,
    RepairOrchestrator,
    Storage,
    Verifier,
    WorkItemJudge,
    configure_logging,
    fork_session,
)
from harness.storage.sqlite import default_db_path
from harness.tasks import (
    ActivityStore,
    Task,
    TaskStore,
)

_args_preview = _common_args_preview
_preprocess_markdown = _preprocess_markdown_impl
_render_mermaid = _render_mermaid_impl
_workspace_db = workspace_db

app = typer.Typer(
    name="harness",
    help="Harness — Python agent runtime over OpenAI-compatible providers and Ollama.",
    no_args_is_help=True,
    add_completion=False,
)


sessions_app = typer.Typer(
    name="sessions", help="Inspect, resume, and remove saved sessions.", no_args_is_help=True
)
app.add_typer(sessions_app, name="sessions")
app.add_typer(providers_app, name="providers")
app.add_typer(tools_app, name="tools")
app.add_typer(plugins_app, name="plugins")
app.add_typer(gateway_app, name="gateway")
app.add_typer(mission_app, name="mission")
app.add_typer(scheduler_app, name="scheduler")
app.add_typer(vision_app, name="vision")
app.add_typer(research_app, name="research")

tasks_app = typer.Typer(
    name="tasks", help="Create, list, and update durable tasks.", no_args_is_help=True
)
app.add_typer(tasks_app, name="tasks")

approvals_app = typer.Typer(
    name="approvals",
    help="Inspect and resolve tool-call approvals queued via --inbox.",
    no_args_is_help=True,
)
app.add_typer(approvals_app, name="approvals")

evidence_app = typer.Typer(
    name="evidence",
    help="Inspect the tool-call evidence ledger.",
    no_args_is_help=True,
)
app.add_typer(evidence_app, name="evidence")

lab_app = typer.Typer(
    name="lab",
    help="Multi-agent lab: planner → workers → reporter.",
    no_args_is_help=True,
)
app.add_typer(lab_app, name="lab")

memory_app = typer.Typer(
    name="memory",
    help="Manage persistent workspace memories injected into every agent run.",
    no_args_is_help=True,
)
app.add_typer(memory_app, name="memory")

app.add_typer(eval_app, name="eval")
app.add_typer(experience_app, name="experience")

phase_app = typer.Typer(
    name="phase",
    help=(
        "Coordination primitive for multi-step tasks. External agents "
        "(Claude Code, Cursor, etc.) can shell out to `harness phase` "
        "to record SDLC progress against the most recent session, and "
        "the structural PhaseGateVerifier reads the same activity log."
    ),
    no_args_is_help=True,
)
app.add_typer(phase_app, name="phase")

tips_app = typer.Typer(
    name="tips",
    help=(
        "L2 procedural-skill tips. Lessons mined from past failure traces "
        "and authored hints, injected into the system prompt when their "
        "triggers match the task text."
    ),
    no_args_is_help=True,
)
app.add_typer(tips_app, name="tips")

tune_app = typer.Typer(
    name="tune",
    help=(
        "ACON-style verifier/critic prompt tuner. Given a current prompt "
        "plus paired A/B trajectories, ask an LLM to propose a revised "
        "prompt. Always advisory — proposals must be reviewed before "
        "being applied."
    ),
    no_args_is_help=True,
)
app.add_typer(tune_app, name="tune")

resume_app = typer.Typer(
    name="resume",
    help=(
        "Cross-session resume contract. JSON file at .harness/resume.json "
        "describing the workspace roadmap and the in-flight feature; "
        "injected into the system prompt at run start."
    ),
    no_args_is_help=True,
)
app.add_typer(resume_app, name="resume")

contracts_app = typer.Typer(
    name="contracts",
    help=(
        "L1 environment contracts. Hard rules (YAML/JSON) loaded from "
        ".harness/contracts/ and ~/.harness/contracts/ and prepended to "
        "matching runs."
    ),
    no_args_is_help=True,
)
app.add_typer(contracts_app, name="contracts")

_DEFAULT_SYSTEM_PROMPT = (
    "You are Harness, a general-purpose AI work agent with access to filesystem, shell, and workflow tools. "
    "Complete the task fully before responding — do not stop mid-task to ask "
    "questions or offer options. Use tools to find everything you need.\n\n"
    "Harness is general, not coding-only. Coding is one capability, alongside read-only research, review, "
    "workflow orchestration, reminders, gateway work, and other operational tasks. "
    "Choose the lightest execution style that fits the user's request. Do not force coding workflows onto "
    "read-only questions, explanations, or orchestration tasks.\n\n"
    "If specialist handoff tools are available and the user explicitly asks for delegation or the task is an obvious "
    "fit for a specialist, use the handoff tool instead of pretending to do specialist work directly.\n\n"
    "## Comprehension-first policy\n\n"
    "In large or unfamiliar codebases, the first high-value outcome is often understanding, not code generation. "
    "When the user asks to be caught up, understand how something works, review architecture/conventions/testing/history, "
    "or align their mental model before implementation, stay read-only and build that mental model first. "
    "Show the evidence you inspected, the file/component map, and the flow through the system before proposing changes.\n\n"
    "## Context-engine policy\n\n"
    "Do not confuse access with understanding. Raw MCP/tool access, naive document search, or a huge context window "
    "is not enough for broad or unfamiliar work. Before planning, implementation, or review on open-ended tasks, "
    "build or request a compact context packet when the needed context is not already clear.\n\n"
    "A good context packet includes current sources of truth, local patterns to reuse, conflicts between code/docs/history, "
    "visible permission or data-governance boundaries, local expert signals from repo evidence when available, "
    "and the validation or review risks the execution agent should use. Search beyond the first plausible hit, "
    "do not dump raw logs, and do not rely on stale cached answers without checking current evidence.\n\n"
    "## Execution-first policy (required)\n\n"
    "If the user asks for something executable or directly observable, do not stop at creation. "
    "Carry the task through the obvious next step before closing.\n\n"
    "Examples:\n"
    "- If you write a script, run it unless blocked.\n"
    "- If you modify an app or command, exercise it and report the real outcome.\n"
    "- If the user asks a read-only repo or system question, inspect the relevant sources and answer directly.\n"
    "- If the user asks to set up durable work, use Harness workflow primitives and report the resulting artifacts.\n"
    "- If the user asked for live behavior, prefer a real run over mock-only verification.\n\n"
    "For executable requests, done means:\n"
    "1. Create or modify the artifact.\n"
    "2. Execute or verify the real behavior when feasible.\n"
    "3. Summarize the real output or blocker back to the user.\n\n"
    "Do not make the user ask you to run the thing you just created when that run is the natural next step.\n\n"
    "## Code-change workflow (apply only when the task actually requires code edits or bug fixing)\n\n"
    "Work like an engineer at a terminal: write code, run tests, read the output, fix, repeat.\n\n"
    "1. Read the relevant source files to understand the problem.\n"
    "1a. For broad or open-ended coding tasks, first choose the smallest relevant existing tracked files and focused tests. "
    "Prefer editing an existing file over creating a new file when the repo already has an obvious home for the change. "
    "Do not create new fixtures, scratch trees, or parallel test directories unless the user explicitly asked for them or no suitable existing target exists.\n"
    "2. Form a hypothesis about what is wrong.\n"
    "3. If uncertain, call request_critique to stress-test your hypothesis.\n"
    "4. Implement a fix.\n"
    "5. Call verify_work immediately with the project's test command "
    "(e.g. 'pytest tests/', 'npm test', 'cargo test', 'go test ./...').\n"
    "6. READ THE OUTPUT. If tests fail:\n"
    "   - Find the specific failing test and what it asserts.\n"
    "   - Understand why your fix does not address what the test checks.\n"
    "   - Revise your implementation.\n"
    "   - Call verify_work again.\n"
    "7. Repeat steps 4-6 until verify_work returns PASSED.\n"
    "8. After verification passes, do one bounded adjacent review of nearby code for missing tests, inconsistencies, or follow-on risks. "
    "Report the best 0-3 nearby findings separately without silently expanding the task.\n"
    "9. Only then declare the task complete.\n\n"
    "Do NOT declare done after a single attempt on coding tasks. "
    "Do NOT assume your first fix is correct without running verify_work. "
    "Iteration is expected — most fixes take 2-4 attempts.\n\n"
    "If there are multiple materially different valid implementation paths after a small amount of inspection, "
    "ask the user a short option question instead of guessing blindly.\n\n"
    "## Other tools\n\n"
    "- request_critique: Get a second opinion on your approach before making changes. "
    "Describe what you plan to do and why — the critic will identify flaws. "
    "Call this when your diagnosis feels uncertain.\n\n"
    "Shell hygiene rules (follow these on every shell call):\n"
    "- Exclude .venv, __pycache__, node_modules, .git from find/glob commands:\n"
    "  find . -name '*.py' -not -path './.venv/*' -not -path './__pycache__/*'\n"
    "- Use pipes to count/sort/summarize large output: | wc -l, | sort | uniq -c | sort -rn\n"
    "- Never run long-running background processes.\n\n"
    "For long-running or durable work, do not invent an ad hoc background loop. "
    "Use Harness primitives instead.\n"
    "- Create durable task state when the work should outlive the current turn.\n"
    "- Use mission/resume/contracts/tips/memory primitives to capture the workflow state.\n"
    "- Use scheduler jobs for repeated or future execution.\n"
    "- Prefer `harness mission launch ...` over shelling a custom daemon or sleep loop.\n"
    "- Do not invent CLI flags. If you are unsure, inspect `--help` and then use the exact command shape.\n\n"
    "Harness-native long-running workflow patterns:\n"
    "- Launch a durable workflow: `harness mission launch --title <title> --goal <goal> [--feature <name>] [--every 30m] [--run-now]`\n"
    "- Inspect or advance mission state: `harness mission show`, `list-milestones`, `list-features`, `show-contract`, `execute-next`, `execute-burst`, `summarize`\n"
    "- Inspect scheduler state: `harness scheduler list`, `list-runs`, `run-now`, `start`, `pause`, `resume`\n"
    "- Manage continuity: `harness resume show`, `resume init`, `resume set-current`, `resume add-feature`\n"
    "- Manage hard rules and soft hints: `harness contracts list|test` and `harness tips list|test|add|mine`\n"
    "- Inspect human-in-the-loop gates: `harness approvals list|show|grant|deny` and `harness evidence list`\n\n"
    "When the user asks to set up durable, repeated, or resumable work from chat, bootstrap it in this order when feasible:\n"
    "1. Launch the workflow and capture its ids: `harness mission launch --title <title> --goal <goal> [--feature <name>] [--every 30m] [--run-now]`\n"
    "2. Reuse the emitted `mission_id` in follow-up commands. Do not invent ids or flags.\n"
    "3. Confirm continuity state with `harness resume show`. Do not call `resume init` after `mission launch` unless the resume file is actually missing.\n"
    "4. Inspect mission state with exact command shapes: `harness mission show <mission_id>`, `harness mission show-contract --mission <mission_id>`, `harness mission summarize --mission <mission_id>`\n"
    "5. Inspect scheduler state for future work: `harness scheduler list` and `harness scheduler list-runs`\n"
    "6. Check human gates if the workflow may block: `harness approvals list` and `harness evidence list`\n"
    "7. Mention contracts/tips when they matter: `harness contracts list|test` and `harness tips list|test`\n"
    "If you use these primitives, tell the user which mission, scheduler, resume, and gate artifacts you created or inspected.\n\n"
    "When a task requires reading source files:\n"
    "1. Use the shell tool to check total size: "
    "`find <path> -name '*.py' | xargs wc -c 2>/dev/null | tail -1` "
    "(note: 'find' is a shell command, not a standalone tool — always call it via shell)\n"
    "2. If total exceeds ~200 KB, call spawn_agents with a goal that names "
    "the exact path and what analysis you need.\n"
    "3. If under 200 KB, read the key files and synthesize a complete answer.\n\n"
    "When asked about architecture, design, or how components interact: "
    "read the relevant source files first, then include a Mermaid diagram "
    "(flowchart or sequence diagram) that shows the actual relationships and "
    "data flow found in the code. Fence it with ```mermaid ... ```."
)

_SPAWN_SCHEMA: dict[str, Any] = _RUNTIME_SPAWN_SCHEMA


class SpawnAgentsTool(_RuntimeSpawnAgentsTool):
    """CLI compatibility shim for the extracted runtime-agent module."""


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _build_adapter(provider: str, *, base_url: str | None, config: HarnessConfig) -> Adapter:
    """CLI-local adapter factory kept for monkeypatch-friendly test compatibility."""
    settings = config.provider(provider)
    effective_base_url = base_url or settings.get("base_url")
    if provider == "ollama":
        timeout = float(settings.get("timeout", 120.0))
        return (
            OllamaAdapter(base_url=effective_base_url, timeout=timeout)
            if effective_base_url
            else OllamaAdapter(timeout=timeout)
        )
    if provider == "openrouter":
        return OpenRouterAdapter(
            base_url=effective_base_url,
            http_referer=settings.get("http_referer"),
            x_title=settings.get("x_title"),
        )
    if provider == "codex":
        timeout = float(settings.get("timeout", 600.0))
        idle_timeout = float(settings.get("idle_timeout", 120.0))
        cwd = settings.get("cwd")
        return CodexAdapter(cwd=cwd, timeout=timeout, idle_timeout=idle_timeout)
    if provider == "openai":
        timeout = float(settings.get("timeout", 120.0))
        return OpenAIAdapter(base_url=effective_base_url, timeout=timeout)
    if provider == "anthropic":
        return AnthropicAdapter(base_url=effective_base_url)
    raise typer.BadParameter(f"unknown provider: {provider!r}")


def _load_project_context(cwd: Path) -> str:
    return _load_project_context_impl(cwd)


def _build_agent(
    *,
    chain: list[str],
    base_url: str | None,
    model: str,
    storage: Storage,
    cwd: Path,
    config: HarnessConfig,
    yes: bool,
    inbox: bool = False,
    activity_store: ActivityStore | None = None,
    approval_store: ApprovalStore | None = None,
    verifier: Verifier | None = None,
    critic: Critic | None = None,
    budget: ContextBudget | None = None,
    memory_store: Any | None = None,
    planner: Planner | None = None,
    session_overrides: dict[str, ApprovalDecision] | None = None,
    predictor: ConsequencePredictor | None = None,
    repair: RepairOrchestrator | None = None,
    system_prompt: str | None = None,
    compactor: Any | None = None,
    max_repair_attempts: int = 3,
    profile: str = "minimal",
    phases_enabled: bool = False,
    loop_detector: Any | None = None,
    contracts: Any | None = None,
    tips_provider: Any | None = None,
    resume: Any | None = None,
    build_tools: Any = _build_tools,
) -> Agent:
    return _build_agent_impl(
        chain=chain,
        base_url=base_url,
        model=model,
        storage=storage,
        cwd=cwd,
        config=config,
        yes=yes,
        build_adapter=_build_adapter,
        build_tools=build_tools,
        build_search_fn=_build_search_fn,
        console=console,
        inbox=inbox,
        activity_store=activity_store,
        approval_store=approval_store,
        verifier=verifier,
        critic=critic,
        budget=budget,
        memory_store=memory_store,
        planner=planner,
        session_overrides=session_overrides,
        predictor=predictor,
        repair=repair,
        system_prompt=system_prompt,
        compactor=compactor,
        max_repair_attempts=max_repair_attempts,
        profile=profile,
        phases_enabled=phases_enabled,
        loop_detector=loop_detector,
        contracts=contracts,
        tips_provider=tips_provider,
        resume=resume,
    )


async def _resolve_task_attachment(
    storage: object, task_ref: str | None, session_id: str | None
) -> tuple[str | None, Task | None]:
    """If `task_ref` is set, look it up and attach `session_id` to its session_ids.

    Returns `(task.id, task)` so callers can pass `task_id` into RunRequest.
    Raises `typer.Exit(1)` if the ref doesn't resolve.
    """
    if task_ref is None:
        return None, None
    store: TaskStore = storage  # type: ignore[assignment]
    task = await store.get_task_by_ref(task_ref)
    if task is None:
        console.print(f"[red]Task not found:[/red] {task_ref}")
        raise typer.Exit(1)
    if session_id and session_id not in task.session_ids:
        task.session_ids.append(session_id)
        task.touch()
        await store.update_task(task)
    return task.id, task


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the installed Harness CLI version."""
    from harness.cli import __version__

    typer.echo(__version__)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    prompt: Annotated[str, typer.Argument(help="The user prompt for the agent.")],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model identifier (overrides config)."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider",
            "-p",
            help="Provider: 'ollama', 'codex', 'openai', or 'openrouter' (overrides config).",
        ),
    ] = None,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="Override the provider's base URL.")
    ] = None,
    cwd: Annotated[
        Path | None,
        typer.Option(
            "--cwd", help="Working directory for filesystem tools (default: current dir)."
        ),
    ] = None,
    max_steps: Annotated[
        int, typer.Option("--max-steps", help="Maximum ReAct turns before giving up.")
    ] = 25,
    max_output_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-output-tokens",
            help="Cap model output tokens for each adapter call.",
        ),
    ] = None,
    max_repair: Annotated[
        int,
        typer.Option(
            "--max-repair",
            help=(
                "Max repair attempts after a verifier fails. Lower this for slow "
                "local models — each attempt re-runs the agent + critic."
            ),
        ),
    ] = 3,
    failover: Annotated[
        str | None,
        typer.Option(
            "--failover",
            help="Comma-separated provider chain (e.g. 'ollama,openai,openrouter'). Overrides --provider.",
        ),
    ] = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session", help="Reuse / create a session with this id. Required for resume later."
        ),
    ] = None,
    task_ref: Annotated[
        str | None,
        typer.Option("--task", help="Attach this session to an existing task ref (e.g. T-001)."),
    ] = None,
    db: Annotated[
        Path | None,
        typer.Option("--db", help=f"SQLite session db path. Default: {default_db_path()}."),
    ] = None,
    in_memory: Annotated[
        bool, typer.Option("--in-memory", help="Use in-memory storage (session lost on exit).")
    ] = False,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Auto-approve all tool calls (non-interactive)."),
    ] = False,
    inbox: Annotated[
        bool,
        typer.Option(
            "--inbox",
            help="Queue prompt-approval tool calls in the durable inbox instead of asking.",
        ),
    ] = False,
    verify: Annotated[
        str | None,
        typer.Option(
            "--verify",
            help=(
                "Post-run verifier: grounding (claim check, free) | state (filesystem check) | "
                "rule (heuristic) | shell (run --verify-command) | llm (extra adapter call) | "
                "auto (all chained) | none."
            ),
        ),
    ] = "grounding",
    verify_command: Annotated[
        str | None,
        typer.Option(
            "--verify-command",
            help="Shell command for --verify shell. Exit 0 = pass, non-zero = fail + repair.",
        ),
    ] = None,
    critic: Annotated[
        str | None,
        typer.Option(
            "--critic",
            help=(
                "Critic mode: llm (challenge agent hypothesis after each failed repair) | "
                "llm+search (same + Tavily web research, requires TAVILY_API_KEY) | none."
            ),
        ),
    ] = None,
    require_tools: Annotated[
        bool,
        typer.Option(
            "--require-tools/--no-require-tools",
            help="Force model to call at least one tool before answering (prevents memory-only replies).",
        ),
    ] = False,
    goal: Annotated[
        bool,
        typer.Option(
            "--goal",
            help="Use LLMPlanner to generate a multi-step plan before running.",
        ),
    ] = False,
    max_context_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-context-tokens",
            help="Prune session history before each adapter call to fit this token budget.",
        ),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help=f"Override config path (default: {default_config_path()})."),
    ] = None,
    predict: Annotated[
        bool,
        typer.Option(
            "--predict",
            help="Enable ConsequencePredictor: commit a prediction before each tool executes.",
        ),
    ] = False,
    auto_compact: Annotated[
        bool,
        typer.Option(
            "--auto-compact",
            help="Summarize old messages via LLM when context exceeds 80% of max_tokens.",
        ),
    ] = False,
    profile: Annotated[
        str,
        typer.Option(
            "--profile",
            help=(
                "Structural defense level. 'bare' = no chain, no critic "
                "(model + tools only). 'adaptive' (default) chooses minimal "
                "or stricter paths from the task shape. 'diagnostic' = "
                "verify + diagnosis-alignment + misdirected-suggestion + prompt-surface revert. "
                "'minimal' = only the tests-before-done check. 'strict' = full chain (file scope, "
                "minimal fix, tests-first, verify-before-done, diagnosis "
                "alignment, misdirected suggestion, prompt-surface revert). See evals/EVAL.md for "
                "the cross-model A/B data behind the default."
            ),
        ),
    ] = "adaptive",
    domain: Annotated[
        str,
        typer.Option(
            "--domain",
            help=(
                "Task domain profile. Currently: coding, code-review, comprehension, "
                "research, docs-audit, mission-planning."
            ),
        ),
    ] = "coding",
    bare: Annotated[
        bool,
        typer.Option(
            "--bare",
            help="Deprecated alias for --profile bare.",
            hidden=True,
        ),
    ] = False,
    phases: Annotated[
        str | None,
        typer.Option(
            "--phases",
            help=(
                "Comma-separated phase names to pre-declare on the session "
                "(e.g. --phases implement,test,document,verify). Runtime "
                "tracks state natively; PhaseGateVerifier (in strict profile) "
                "refuses Done until all phases are completed via the phase "
                "tool."
            ),
        ),
    ] = None,
    loop_detect: Annotated[
        bool,
        typer.Option(
            "--loop-detect/--no-loop-detect",
            help=(
                "L4 trajectory regulation. Detects repeated identical tool "
                "calls and read-only spinning, injecting a corrective user "
                "message. Default on in minimal/strict profiles, off in bare."
            ),
        ),
    ] = True,
    contracts: Annotated[
        bool,
        typer.Option(
            "--contracts/--no-contracts",
            help=(
                "L1 environment contracts. Loads YAML/JSON from "
                ".harness/contracts/ and ~/.harness/contracts/ and prepends "
                "matching contracts as system rules. Default on in "
                "minimal/strict, off in bare."
            ),
        ),
    ] = True,
    tips: Annotated[
        bool,
        typer.Option(
            "--tips/--no-tips",
            help=(
                "L2 procedural skill. Loads tips from .harness/tips.jsonl / "
                "~/.harness/tips.jsonl and injects matching ones as system "
                "context. Default on in minimal/strict, off in bare."
            ),
        ),
    ] = True,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable DEBUG logging to stderr.")
    ] = False,
) -> None:
    _run_command(
        prompt=prompt,
        model=model,
        provider=provider,
        failover=failover,
        base_url=base_url,
        cwd=cwd,
        max_steps=max_steps,
        max_output_tokens=max_output_tokens,
        session_id=session_id,
        task_ref=task_ref,
        db=db,
        in_memory=in_memory,
        yes=yes,
        inbox=inbox,
        verify=verify,
        verify_command=verify_command,
        critic=critic,
        require_tools=require_tools,
        goal=goal,
        max_context_tokens=max_context_tokens,
        predict=predict,
        auto_compact=auto_compact,
        max_repair=max_repair,
        profile=profile,
        domain=domain,
        bare=bare,
        phases=phases,
        loop_detect=loop_detect,
        contracts=contracts,
        tips=tips,
        verbose=verbose,
        config_path=config_path,
        console=console,
        load_cli_config=_load_cli_config,
        resolve_chain=_resolve_chain,
        run_async=_run_async,
        run_once=_run_once,
    )


@app.command("review")
def review(
    base: Annotated[
        str,
        typer.Option("--base", help="Base git ref to diff against."),
    ] = "HEAD~1",
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    provider: Annotated[str | None, typer.Option("--provider", "-p")] = None,
    failover: Annotated[str | None, typer.Option("--failover")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    cwd: Annotated[Path | None, typer.Option("--cwd")] = None,
    max_steps: Annotated[int, typer.Option("--max-steps")] = 20,
    max_output_tokens: Annotated[int | None, typer.Option("--max-output-tokens")] = None,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help=f"Override config path (default: {default_config_path()})."),
    ] = None,
) -> None:
    _review_command(
        base=base,
        model=model,
        provider=provider,
        failover=failover,
        base_url=base_url,
        cwd=cwd,
        max_steps=max_steps,
        max_output_tokens=max_output_tokens,
        db=db,
        in_memory=in_memory,
        yes=yes,
        verbose=verbose,
        json_output=json_output,
        config_path=config_path,
        console=console,
        load_cli_config=_load_cli_config,
        resolve_chain=_resolve_chain,
        run_async=_run_async,
        run_once=_run_once,
    )


@app.command("docs-audit")
def docs_audit(
    focus: Annotated[
        str | None,
        typer.Argument(help="Optional documentation area or question to focus on."),
    ] = None,
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    provider: Annotated[str | None, typer.Option("--provider", "-p")] = None,
    failover: Annotated[str | None, typer.Option("--failover")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    cwd: Annotated[Path | None, typer.Option("--cwd")] = None,
    max_steps: Annotated[int, typer.Option("--max-steps")] = 20,
    max_output_tokens: Annotated[int | None, typer.Option("--max-output-tokens")] = None,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help=f"Override config path (default: {default_config_path()})."),
    ] = None,
) -> None:
    _docs_audit_command(
        focus=focus,
        model=model,
        provider=provider,
        failover=failover,
        base_url=base_url,
        cwd=cwd,
        max_steps=max_steps,
        max_output_tokens=max_output_tokens,
        db=db,
        in_memory=in_memory,
        yes=yes,
        verbose=verbose,
        json_output=json_output,
        config_path=config_path,
        console=console,
        load_cli_config=_load_cli_config,
        resolve_chain=_resolve_chain,
        run_async=_run_async,
        run_once=_run_once,
    )


async def _run_once(**kwargs: Any) -> str | None:
    return await _run_once_impl(
        **kwargs,
        build_storage=_build_storage,
        resolve_task_attachment=_resolve_task_attachment,
        resolve_runtime_strategy=_resolve_runtime_strategy,
        build_verifier=_build_verifier,
        build_critic=_build_critic,
        build_adapter=_build_adapter,
        build_tools=_build_tools,
        build_agent=_build_agent,
        print_defense_ledger=_print_defense_ledger,
        render=_render,
        default_system_prompt=_DEFAULT_SYSTEM_PROMPT,
        console=console,
    )


# ---------------------------------------------------------------------------
# sessions subcommands
# ---------------------------------------------------------------------------


@sessions_app.command("list")
def sessions_list(
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max sessions to show.")] = 25,
    status: Annotated[
        str | None,
        typer.Option(
            "--status",
            help="Filter: pending | running | paused | done | failed | cancelled.",
        ),
    ] = None,
) -> None:
    _sessions_list_command(
        db=db,
        in_memory=in_memory,
        limit=limit,
        status=status,
        console=console,
        build_storage=_build_storage,
        run_async=_run_async,
        ago=_ago,
        status_style=_status_style,
    )


@sessions_app.command("show")
def sessions_show(
    session_id: Annotated[str, typer.Argument(help="Session id to display.")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    _sessions_show_command(
        session_id=session_id,
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        run_async=_run_async,
        render_session=_render_session,
    )


@sessions_app.command("resume")
def sessions_resume(
    session_id: Annotated[str, typer.Argument(help="Session id to continue.")],
    prompt: Annotated[
        str | None,
        typer.Argument(help="New user prompt. Omit to continue without new input."),
    ] = None,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Working directory for filesystem tools."),
    ] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    max_steps: Annotated[int, typer.Option("--max-steps")] = 25,
    failover: Annotated[
        str | None,
        typer.Option(
            "--failover",
            help="Comma-separated provider chain. Overrides the session's recorded provider.",
        ),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Auto-approve all tool calls.")] = False,
    inbox: Annotated[
        bool, typer.Option("--inbox", help="Queue prompt-approval tool calls to the inbox.")
    ] = False,
    verify: Annotated[
        str | None,
        typer.Option("--verify", help="Post-run verifier: rule | llm | none."),
    ] = None,
    max_context_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-context-tokens",
            help="Prune session history before each adapter call to fit this token budget.",
        ),
    ] = None,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    _sessions_resume_command(
        session_id=session_id,
        prompt=prompt,
        db=db,
        in_memory=in_memory,
        cwd=cwd,
        base_url=base_url,
        max_steps=max_steps,
        failover=failover,
        yes=yes,
        inbox=inbox,
        verify=verify,
        max_context_tokens=max_context_tokens,
        config_path=config_path,
        verbose=verbose,
        console=console,
        load_cli_config=_load_cli_config,
        resolve_chain=_resolve_chain,
        build_storage=_build_storage,
        build_verifier=_build_verifier,
        build_adapter=_build_adapter,
        build_agent=_build_agent,
        render=_render,
        run_async=_run_async,
    )


@sessions_app.command("rm")
def sessions_rm(
    session_id: Annotated[str, typer.Argument(help="Session id to delete.")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    _sessions_rm_command(
        session_id=session_id,
        db=db,
        in_memory=in_memory,
        yes=yes,
        console=console,
        build_storage=_build_storage,
        run_async=_run_async,
    )


@sessions_app.command("fork")
def sessions_fork(
    session_id: Annotated[str, typer.Argument(help="Session id to fork from.")],
    prompt: Annotated[
        str | None,
        typer.Argument(help="Optional prompt to run immediately in the new fork."),
    ] = None,
    new_id: Annotated[
        str | None,
        typer.Option("--session", help="Explicit id for the new session."),
    ] = None,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Auto-approve tool calls if prompt is given.")
    ] = False,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    _sessions_fork_command(
        session_id=session_id,
        prompt=prompt,
        new_id=new_id,
        db=db,
        in_memory=in_memory,
        yes=yes,
        config_path=config_path,
        verbose=verbose,
        console=console,
        load_cli_config=_load_cli_config,
        build_storage=_build_storage,
        build_agent=_build_agent,
        render=_render,
        run_async=_run_async,
        fork_session_fn=fork_session,
    )


@sessions_app.command("diff")
def sessions_diff_cmd(
    session_id: Annotated[str, typer.Argument(help="Session id to diff.")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    _sessions_diff_command(
        session_id=session_id,
        db=db,
        in_memory=in_memory,
        build_storage=_build_storage,
        render_session_diff=_render_session_diff,
        console=console,
        run_async=_run_async,
    )


# ---------------------------------------------------------------------------
# tasks subcommands
# ---------------------------------------------------------------------------


@tasks_app.command("new")
def tasks_new_cmd(
    title: Annotated[str, typer.Argument(help="Short title for the task.")],
    description: Annotated[
        str | None, typer.Option("--description", "-d", help="Longer description.")
    ] = None,
    priority: Annotated[
        str | None,
        typer.Option("--priority", help="low | medium | high"),
    ] = None,
    labels: Annotated[
        str | None, typer.Option("--labels", help="Comma-separated label list.")
    ] = None,
    parent: Annotated[
        str | None,
        typer.Option("--parent", help="Parent task ref (e.g. T-001)."),
    ] = None,
    cwd: Annotated[Path | None, typer.Option("--cwd")] = None,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    _tasks_new_command(
        title=title,
        description=description,
        priority=priority,
        labels=labels,
        parent=parent,
        cwd=cwd,
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        run_async=_run_async,
    )


@tasks_app.command("list")
def tasks_list_cmd(
    status: Annotated[
        str | None,
        typer.Option("--status", help="Filter: backlog|todo|in_progress|waiting|done|cancelled."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 25,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    _tasks_list_command(
        status=status,
        limit=limit,
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        run_async=_run_async,
        task_status_style=_task_status_style,
        truncate=_truncate,
        ago=_ago,
    )


@tasks_app.command("show")
def tasks_show_cmd(
    ref: Annotated[str, typer.Argument(help="Task ref, e.g. T-001.")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    _tasks_show_command(
        ref=ref,
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        run_async=_run_async,
        render_task=_render_task,
    )


@tasks_app.command("update")
def tasks_update_cmd(
    ref: Annotated[str, typer.Argument(help="Task ref.")],
    status: Annotated[str | None, typer.Option("--status")] = None,
    title: Annotated[str | None, typer.Option("--title")] = None,
    description: Annotated[str | None, typer.Option("--description", "-d")] = None,
    priority: Annotated[str | None, typer.Option("--priority")] = None,
    labels: Annotated[
        str | None,
        typer.Option("--labels", help="Comma-separated label list (replaces existing)."),
    ] = None,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    _tasks_update_command(
        ref=ref,
        status=status,
        title=title,
        description=description,
        priority=priority,
        labels=labels,
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        run_async=_run_async,
    )


@tasks_app.command("link")
def tasks_link_cmd(
    ref: Annotated[str, typer.Argument(help="Task ref (source of the link).")],
    target: Annotated[str, typer.Argument(help="Target task ref.")],
    relation: Annotated[
        str,
        typer.Option(
            "--relation",
            "-r",
            help="One of: blocks | depends_on | duplicates | fixes | tests | relates_to.",
        ),
    ] = "relates_to",
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    _tasks_link_command(
        ref=ref,
        target=target,
        relation=relation,
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        run_async=_run_async,
    )


@tasks_app.command("rm")
def tasks_rm_cmd(
    ref: Annotated[str, typer.Argument(help="Task ref.")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    _tasks_rm_command(
        ref=ref,
        db=db,
        in_memory=in_memory,
        yes=yes,
        console=console,
        build_storage=_build_storage,
        run_async=_run_async,
    )


# ---------------------------------------------------------------------------
# approvals subcommands
# ---------------------------------------------------------------------------


@approvals_app.command("list")
def approvals_list_cmd(
    pending_only: Annotated[
        bool, typer.Option("--pending", help="Only list pending approvals.")
    ] = False,
    task: Annotated[
        str | None, typer.Option("--task", help="Filter by task ref (e.g. T-001).")
    ] = None,
    session_id: Annotated[
        str | None, typer.Option("--session", help="Filter by session id.")
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 50,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    _approvals_list_command(
        pending_only=pending_only,
        task=task,
        session_id=session_id,
        limit=limit,
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        close_if_sqlite=_close_if_sqlite,
        run_async=_run_async,
        approval_status_style=_approval_status_style,
        truncate=_truncate,
        ago=_ago,
    )


@approvals_app.command("show")
def approvals_show_cmd(
    approval_id: Annotated[str, typer.Argument(help="Approval id (appr_...).")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    _approvals_show_command(
        approval_id=approval_id,
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        close_if_sqlite=_close_if_sqlite,
        run_async=_run_async,
        render_approval=_render_approval,
    )


@approvals_app.command("grant")
def approvals_grant_cmd(
    approval_id: Annotated[str, typer.Argument(help="Approval id (appr_...).")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    _approvals_grant_command(
        approval_id=approval_id,
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        close_if_sqlite=_close_if_sqlite,
        run_async=_run_async,
    )


@approvals_app.command("deny")
def approvals_deny_cmd(
    approval_id: Annotated[str, typer.Argument(help="Approval id (appr_...).")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    _approvals_deny_command(
        approval_id=approval_id,
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        close_if_sqlite=_close_if_sqlite,
        run_async=_run_async,
    )


# ---------------------------------------------------------------------------
# evidence subcommands
# ---------------------------------------------------------------------------


@evidence_app.command("list")
def evidence_list_cmd(
    task: Annotated[
        str | None, typer.Option("--task", help="Filter by task ref (e.g. T-001).")
    ] = None,
    session_id: Annotated[
        str | None, typer.Option("--session", help="Filter by session id.")
    ] = None,
    tool_name: Annotated[
        str | None, typer.Option("--tool", help="Filter by tool name (e.g. shell).")
    ] = None,
    errors_only: Annotated[
        bool, typer.Option("--errors-only", help="Show only entries with is_error=True.")
    ] = False,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 50,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    _evidence_list_command(
        task=task,
        session_id=session_id,
        tool_name=tool_name,
        errors_only=errors_only,
        limit=limit,
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        close_if_sqlite=_close_if_sqlite,
        run_async=_run_async,
        truncate=_truncate,
        ago=_ago,
    )


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


@app.command()
def chat(
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model identifier (overrides config)."),
    ] = None,
    provider: Annotated[
        str | None, typer.Option("--provider", "-p", help="Primary provider (overrides config).")
    ] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Working directory for filesystem tools."),
    ] = None,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
    session_id: Annotated[
        str | None,
        typer.Option("--session", help="Resume an existing session, or create one with this id."),
    ] = None,
    task_ref: Annotated[
        str | None,
        typer.Option("--task", help="Attach this session to an existing task ref (e.g. T-001)."),
    ] = None,
    max_steps: Annotated[int, typer.Option("--max-steps")] = 25,
    failover: Annotated[
        str | None,
        typer.Option("--failover", help="Comma-separated provider chain."),
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Auto-approve all tool calls.")] = False,
    inbox: Annotated[
        bool, typer.Option("--inbox", help="Queue prompt-approval tool calls to the inbox.")
    ] = False,
    verify: Annotated[
        str | None,
        typer.Option("--verify", help="Post-run verifier: rule | llm | none."),
    ] = "grounding",
    require_tools: Annotated[
        bool,
        typer.Option(
            "--require-tools/--no-require-tools",
            help="Force the model to call at least one tool before answering.",
        ),
    ] = False,
    max_context_tokens: Annotated[
        int | None,
        typer.Option("--max-context-tokens", help="Token budget for pruning per turn."),
    ] = None,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
    auto_compact: Annotated[
        bool,
        typer.Option(
            "--auto-compact",
            help="Summarize old messages via LLM when context exceeds 80% of max_tokens.",
        ),
    ] = False,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Interactive REPL: chat with the agent, drive tools, resume across turns."""
    _run_chat_command(
        model=model,
        provider=provider,
        base_url=base_url,
        cwd=cwd,
        db=db,
        in_memory=in_memory,
        session_id=session_id,
        task_ref=task_ref,
        max_steps=max_steps,
        failover=failover,
        yes=yes,
        inbox=inbox,
        verify=verify,
        require_tools=require_tools,
        max_context_tokens=max_context_tokens,
        config_path=config_path,
        auto_compact=auto_compact,
        verbose=verbose,
        console=console,
        configure_logging=configure_logging,
        load_cli_config=_load_cli_config,
        resolve_chain=_resolve_chain,
        run_async=_run_async,
        build_storage=_build_storage,
        resolve_task_attachment=_resolve_task_attachment,
        build_verifier=_build_verifier,
        build_adapter=_build_adapter,
        build_tools=_build_tools,
        build_agent=_build_agent,
        render=_render,
        render_session_diff=_render_session_diff,
        default_system_prompt=_DEFAULT_SYSTEM_PROMPT,
    )


_renderer = Renderer(console)


def _render(event: Any) -> None:
    _renderer.render(event)


# ---------------------------------------------------------------------------
# goal command
# ---------------------------------------------------------------------------


@app.command()
def goal(
    prompt: Annotated[str, typer.Argument(help="The goal for the agent to accomplish.")],
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    provider: Annotated[str | None, typer.Option("--provider", "-p")] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    cwd: Annotated[Path | None, typer.Option("--cwd")] = None,
    max_steps: Annotated[int, typer.Option("--max-steps")] = 25,
    db: Annotated[
        Path | None,
        typer.Option("--db", help=f"SQLite session db path. Default: {default_db_path()}."),
    ] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    _run_goal_command(
        prompt=prompt,
        model=model,
        provider=provider,
        base_url=base_url,
        cwd=cwd,
        max_steps=max_steps,
        db=db,
        in_memory=in_memory,
        yes=yes,
        config_path=config_path,
        verbose=verbose,
        console=console,
        load_cli_config=_load_cli_config,
        resolve_chain=_resolve_chain,
        run_async=_run_async,
        run_once=_run_once,
    )


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------


@app.command()
def init(
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Directory to initialise (default: current directory)."),
    ] = None,
) -> None:
    _init_workspace(cwd=cwd, console=console)


# ---------------------------------------------------------------------------
# memory subcommands
# ---------------------------------------------------------------------------


@memory_app.command("save")
def memory_save(
    text: Annotated[str, typer.Argument(help="Memory text to store.")],
    kind: Annotated[
        str,
        typer.Option(
            "--kind",
            "-k",
            help="user_preference | user_fact | project_fact | project_context",
        ),
    ] = "project_fact",
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    _memory_save_command(
        text=text,
        kind=kind,
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        run_async=_run_async,
    )


@memory_app.command("list")
def memory_list(
    kind: Annotated[
        str | None,
        typer.Option("--kind", "-k", help="Filter by kind."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 50,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    _memory_list_command(
        kind=kind,
        limit=limit,
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        run_async=_run_async,
        ago=_ago,
    )


@memory_app.command("search")
def memory_search(
    query: Annotated[str, typer.Argument(help="Search query (substring match).")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    _memory_search_command(
        query=query,
        limit=limit,
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        run_async=_run_async,
        ago=_ago,
    )


@memory_app.command("rm")
def memory_rm(
    entry_id: Annotated[str, typer.Argument(help="Memory entry ID to delete.")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    _memory_rm_command(
        entry_id=entry_id,
        db=db,
        in_memory=in_memory,
        yes=yes,
        console=console,
        build_storage=_build_storage,
        run_async=_run_async,
    )


# ---------------------------------------------------------------------------
# lab subcommands — multi-agent orchestration
# ---------------------------------------------------------------------------


@lab_app.command("run")
def lab_run(
    prompt: Annotated[str, typer.Argument(help="Top-level task prompt for the planner.")],
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider", "-p", help="LLM provider (ollama, codex, openai, openrouter, …)."
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name."),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Number of parallel worker agents."),
    ] = 2,
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Auto-approve all tool calls."),
    ] = False,
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Working directory for agents (default: current)."),
    ] = None,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to harness config TOML."),
    ] = None,
    no_judge: Annotated[
        bool,
        typer.Option("--no-judge", help="Disable post-completion judge verification."),
    ] = False,
    db: Annotated[
        Path | None,
        typer.Option(
            "--db", help="SQLite database path for durable job storage (default: in-memory)."
        ),
    ] = None,
    max_context_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-context-tokens", help="Token budget for worker context pruning per turn."
        ),
    ] = None,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", help="Max tool-call steps per worker per work item."),
    ] = 20,
    planner_model: Annotated[
        str | None,
        typer.Option("--planner-model", help="Model for the planner (overrides --model)."),
    ] = None,
    worker_model: Annotated[
        str | None,
        typer.Option("--worker-model", help="Model for workers (overrides --model)."),
    ] = None,
    reporter_model: Annotated[
        str | None,
        typer.Option("--reporter-model", help="Model for the reporter (overrides --model)."),
    ] = None,
) -> None:
    """Run a multi-agent job: planner decomposes, workers execute in parallel, reporter synthesizes."""
    _lab_run_command(
        prompt=prompt,
        provider=provider,
        model=model,
        workers=workers,
        yes=yes,
        cwd=cwd,
        config_path=config_path,
        no_judge=no_judge,
        db=db,
        max_context_tokens=max_context_tokens,
        max_steps=max_steps,
        planner_model=planner_model,
        worker_model=worker_model,
        reporter_model=reporter_model,
        console=console,
        load_cli_config=_load_cli_config,
        build_adapter=_build_adapter,
        run_async=_run_async,
        args_preview=_args_preview,
        truncate=_truncate,
        orchestrator_cls=MultiAgentOrchestrator,
        work_item_judge_cls=WorkItemJudge,
    )


@lab_app.command("status")
def lab_status(
    job_id: Annotated[str, typer.Argument(help="Job ID to inspect.")],
    db: Annotated[
        Path,
        typer.Option("--db", help="SQLite database path."),
    ] = Path("harness.db"),
) -> None:
    """Show work item status for a job stored in a SQLite database."""
    _lab_status_command(job_id=job_id, db=db, console=console, run_async=_run_async)


@lab_app.command("list")
def lab_list(
    db: Annotated[
        Path,
        typer.Option("--db", help="SQLite database path."),
    ] = Path("harness.db"),
) -> None:
    """List all jobs in a SQLite database."""
    _lab_list_command(db=db, console=console, run_async=_run_async)


@lab_app.command("resume")
def lab_resume(
    job_id: Annotated[str, typer.Argument(help="Job ID to resume.")],
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="LLM provider."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model name."),
    ] = None,
    workers: Annotated[
        int,
        typer.Option("--workers", "-w", help="Number of parallel worker agents."),
    ] = 2,
    db: Annotated[
        Path,
        typer.Option("--db", help="SQLite database path."),
    ] = Path("harness.db"),
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help="Path to harness config TOML."),
    ] = None,
    no_judge: Annotated[
        bool,
        typer.Option("--no-judge", help="Disable post-completion judge verification."),
    ] = False,
    max_steps: Annotated[
        int,
        typer.Option("--max-steps", help="Max tool-call steps per worker per work item."),
    ] = 20,
    planner_model: Annotated[
        str | None,
        typer.Option("--planner-model", help="Model for the judge (overrides --model)."),
    ] = None,
    worker_model: Annotated[
        str | None,
        typer.Option("--worker-model", help="Model for workers (overrides --model)."),
    ] = None,
) -> None:
    """Resume an interrupted job from a SQLite database, skipping already-done work items."""
    _lab_resume_command(
        job_id=job_id,
        provider=provider,
        model=model,
        workers=workers,
        db=db,
        config_path=config_path,
        no_judge=no_judge,
        max_steps=max_steps,
        planner_model=planner_model,
        worker_model=worker_model,
        console=console,
        load_cli_config=_load_cli_config,
        build_adapter=_build_adapter,
        run_async=_run_async,
        args_preview=_args_preview,
        truncate=_truncate,
        orchestrator_cls=MultiAgentOrchestrator,
        work_item_judge_cls=WorkItemJudge,
    )


# ---------------------------------------------------------------------------
# phase subcommands — external coordination primitive
# ---------------------------------------------------------------------------


@phase_app.command("declare")
def phase_declare(
    name: Annotated[str, typer.Argument(help="Phase name (e.g. implement, test).")],
    notes: Annotated[
        str | None,
        typer.Option("--notes", "-n", help="Optional one-line note about this phase start."),
    ] = None,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    """Record the start of a phase against the most recent session."""
    _phase_declare_command(
        name=name,
        notes=notes,
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        run_async=_run_async,
    )


@phase_app.command("complete")
def phase_complete(
    name: Annotated[str, typer.Argument(help="Phase name to mark complete.")],
    notes: Annotated[
        str | None,
        typer.Option("--notes", "-n", help="Optional note about what was achieved."),
    ] = None,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    """Mark a phase as complete against the most recent session."""
    _phase_complete_command(
        name=name,
        notes=notes,
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        run_async=_run_async,
    )


@phase_app.command("status")
def phase_status(
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    """List declared / completed phases for the most recent session."""
    _phase_status_command(
        db=db,
        in_memory=in_memory,
        console=console,
        build_storage=_build_storage,
        run_async=_run_async,
    )


# ---------------------------------------------------------------------------
# contracts subcommands — L1 environment contract inspection
# ---------------------------------------------------------------------------


@contracts_app.command("list")
def contracts_list(
    cwd: Annotated[
        Path | None, typer.Option("--cwd", help="Working dir whose .harness/contracts/ to load.")
    ] = None,
) -> None:
    """Show all loaded contracts and the paths they came from."""
    _contracts_list_command(cwd=cwd, console=console)


@contracts_app.command("test")
def contracts_test(
    task: Annotated[str, typer.Argument(help="Task text to match against contracts.")],
    cwd: Annotated[Path | None, typer.Option("--cwd")] = None,
) -> None:
    """Show which contracts would fire for a given task string."""
    _contracts_test_command(task=task, cwd=cwd, console=console)


# ---------------------------------------------------------------------------
# tips subcommands — L2 procedural skill library
# ---------------------------------------------------------------------------


@tips_app.command("list")
def tips_list(
    cwd: Annotated[Path | None, typer.Option("--cwd")] = None,
) -> None:
    """Show all loaded tips, ordered by weight desc."""
    _tips_list_command(cwd=cwd, console=console)


@tips_app.command("add")
def tips_add(
    text: Annotated[str, typer.Argument(help="The tip body (imperative one-liner).")],
    triggers: Annotated[
        str | None,
        typer.Option(
            "--triggers",
            help="Comma-separated trigger substrings. Omit for an always-on tip.",
        ),
    ] = None,
    weight: Annotated[float, typer.Option("--weight")] = 1.0,
    scope: Annotated[
        str,
        typer.Option(
            "--scope",
            help="Where to write: 'repo' (.harness/tips.jsonl) or 'user' (~/.harness/tips.jsonl).",
        ),
    ] = "repo",
) -> None:
    """Append a tip to the library."""
    _tips_add_command(
        text=text,
        triggers=triggers,
        weight=weight,
        scope=scope,
        console=console,
    )


@tips_app.command("test")
def tips_test(
    task: Annotated[str, typer.Argument(help="Task text to match against tips.")],
    top_k: Annotated[int, typer.Option("--top-k")] = 3,
    cwd: Annotated[Path | None, typer.Option("--cwd")] = None,
) -> None:
    """Show which tips would fire for a given task string."""
    _tips_test_command(task=task, top_k=top_k, cwd=cwd, console=console)


@tips_app.command("mine")
def tips_mine(
    session_id: Annotated[
        str,
        typer.Argument(help="Session ID to mine for tips (must be a failed session)."),
    ],
    model: Annotated[
        str | None, typer.Option("--model", "-m", help="Model for the extractor.")
    ] = None,
    provider: Annotated[str | None, typer.Option("--provider", "-p")] = None,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    scope: Annotated[str, typer.Option("--scope")] = "repo",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Print the tips that would be added without writing."),
    ] = False,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Mine a failed session's transcript for reusable tips via an LLM extractor.

    Reads the session's task text + transcript tail, asks the configured
    model for short procedural tips that would have prevented the failure,
    and appends them to the tip library. Skipped tips (bad JSON, over-long
    bodies) are logged but never crash the command.
    """
    _tips_mine_command(
        session_id=session_id,
        model=model,
        provider=provider,
        db=db,
        scope=scope,
        dry_run=dry_run,
        config_path=config_path,
        console=console,
        load_cli_config=_load_cli_config,
        resolve_chain=_resolve_chain,
        build_adapter=_build_adapter,
        run_async=_run_async,
    )


# ---------------------------------------------------------------------------
# tune subcommands — ACON-style prompt tuner
# ---------------------------------------------------------------------------


@tune_app.command("list")
def tune_list(
    cwd: Annotated[Path | None, typer.Option("--cwd")] = None,
) -> None:
    """List versioned tunable prompts under `.harness/tuned-prompts/`."""
    _tune_list_command(cwd=cwd, console=console)


@tune_app.command("show")
def tune_show(
    key: Annotated[str, typer.Argument(help="Prompt key (e.g. minimal_fix_verifier).")],
    version: Annotated[int | None, typer.Option("--version", "-v")] = None,
    cwd: Annotated[Path | None, typer.Option("--cwd")] = None,
) -> None:
    """Print one version (default: current) of a tunable prompt."""
    _tune_show_command(key=key, version=version, cwd=cwd, console=console)


@tune_app.command("propose")
def tune_propose(
    key: Annotated[str, typer.Argument(help="Prompt key to tune.")],
    current_prompt_file: Annotated[
        Path,
        typer.Option("--current", help="File containing the current prompt text."),
    ],
    pairs_file: Annotated[
        Path,
        typer.Option(
            "--pairs",
            help="JSON file with a list of trajectory pairs (see verifier_tuner.TrajectoryPair).",
        ),
    ],
    model: Annotated[str | None, typer.Option("--model", "-m")] = None,
    provider: Annotated[str | None, typer.Option("--provider", "-p")] = None,
    notes: Annotated[str | None, typer.Option("--notes")] = None,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Print the proposal without writing to .harness/tuned-prompts/.",
        ),
    ] = False,
) -> None:
    """Ask the configured LLM for a prompt-delta proposal.

    The proposal is printed for review. Pass --dry-run to skip writing;
    otherwise the new prompt is appended as a new version to
    `.harness/tuned-prompts/<key>.json` (still requires explicit
    runtime opt-in to actually use it).
    """
    _tune_propose_command(
        key=key,
        current_prompt_file=current_prompt_file,
        pairs_file=pairs_file,
        model=model,
        provider=provider,
        notes=notes,
        config_path=config_path,
        dry_run=dry_run,
        console=console,
        load_cli_config=_load_cli_config,
        resolve_chain=_resolve_chain,
        build_adapter=_build_adapter,
        run_async=_run_async,
    )


@tune_app.command("rollback")
def tune_rollback(
    key: Annotated[str, typer.Argument()],
    cwd: Annotated[Path | None, typer.Option("--cwd")] = None,
) -> None:
    """Drop the latest version, restoring the previous one as current."""
    _tune_rollback_command(key=key, cwd=cwd, console=console)


# ---------------------------------------------------------------------------
# resume subcommands — cross-session continuity
# ---------------------------------------------------------------------------


@resume_app.command("show")
def resume_show(
    cwd: Annotated[Path | None, typer.Option("--cwd")] = None,
) -> None:
    """Print the current resume contract (or a hint when missing)."""
    _resume_show_command(cwd=cwd, console=console)


@resume_app.command("init")
def resume_init(
    cwd: Annotated[Path | None, typer.Option("--cwd")] = None,
    feature: Annotated[
        str | None,
        typer.Option("--feature", help="Name of the initial feature to seed."),
    ] = None,
    description: Annotated[
        str | None,
        typer.Option("--description", help="Description for the initial feature."),
    ] = None,
) -> None:
    """Create a fresh `.harness/resume.json` with a single starter feature."""
    _resume_init_command(
        cwd=cwd,
        feature=feature,
        description=description,
        console=console,
    )


@resume_app.command("set-current")
def resume_set_current(
    feature_name: Annotated[str, typer.Argument()],
    cwd: Annotated[Path | None, typer.Option("--cwd")] = None,
) -> None:
    """Point the resume contract at a different feature for the next session."""
    _resume_set_current_command(feature_name=feature_name, cwd=cwd, console=console)


@resume_app.command("add-feature")
def resume_add_feature(
    name: Annotated[str, typer.Argument(help="Kebab-case feature name.")],
    description: Annotated[str | None, typer.Option("--description")] = None,
    phases: Annotated[
        str | None,
        typer.Option(
            "--phases",
            help="Comma-separated phase plan (e.g. implement,test,document,verify).",
        ),
    ] = None,
    cwd: Annotated[Path | None, typer.Option("--cwd")] = None,
) -> None:
    """Append a new pending feature to the roadmap."""
    _resume_add_feature_command(
        name=name,
        description=description,
        phases=phases,
        cwd=cwd,
        console=console,
    )


if __name__ == "__main__":
    app()
