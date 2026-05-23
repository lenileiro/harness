"""Harness CLI entry point.

Phase 4 surface:
- `harness run "prompt"`         — one-shot prompt with the full tool set
- `harness sessions list`        — list saved sessions
- `harness sessions show <id>`   — print full transcript
- `harness sessions resume <id>` — continue an existing session
- `harness sessions rm <id>`     — delete a session
- `harness version`              — print the installed CLI version

Providers: ollama, openrouter.
Tools: read_file, write_file, edit_file, list_dir, glob, shell, fetch_url.

Config: `$XDG_CONFIG_HOME/harness/config.toml` (or ~/.config/harness/config.toml)
provides defaults for provider, model, per-provider settings, and per-tool
approval levels. CLI flags override the config.

Tool approvals default to `prompt` for any tool that mutates state or makes
network calls; the CLI shows a Rich prompt. Pass `--yes` to auto-approve
everything (handy for non-interactive use), or set approvals in config.
"""

from __future__ import annotations

import asyncio

# Load .env from the working directory (or any parent) before anything reads env vars.
try:
    from dotenv import load_dotenv as _load_dotenv

    _load_dotenv(override=False)
except ImportError:
    pass
import difflib
import os
import re
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
import unicodeitplus as _unicodeit
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.spinner import Spinner
from rich.table import Table

from harness.adapters.anthropic import AnthropicAdapter
from harness.adapters.ollama import OllamaAdapter
from harness.adapters.openrouter import OpenRouterAdapter
from harness.cli.approval import RichApprovalHandler
from harness.cli.config import HarnessConfig, default_config_path, load_config
from harness.core import (
    Adapter,
    Agent,
    AgentDoneEvent,
    AgentEventWrapper,
    AgentRole,
    AgentStartedEvent,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalPolicy,
    ApprovalStore,
    AutoApprove,
    ChainedVerifier,
    ClaimGroundingVerifier,
    CompleteWorkItemTool,
    ConsequencePredictor,
    ContextBudget,
    ContextCompactor,
    CreateWorkItemTool,
    Critic,
    Critique,
    DiagnosisAlignmentVerifier,
    Done,
    ErrorEvent,
    FailoverPolicy,
    FileScopeVerifier,
    InboxApprovalHandler,
    ListWorkItemsTool,
    LLMJudgeVerifier,
    LLMPlanner,
    MemoryEntry,
    MinimalFixVerifier,
    MisdirectedSuggestionVerifier,
    MultiAgentOrchestrator,
    PendingApproval,
    Planner,
    PlanRejectedEvent,
    PredictionEvent,
    PredictionMismatchEvent,
    RepairOrchestrator,
    RequestCritiqueTool,
    RuleVerifier,
    RunRequest,
    Session,
    ShellVerifier,
    StateVerifier,
    StepCompleted,
    StepStarted,
    Storage,
    TestsBeforeEditVerifier,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolRegistry,
    ToolResult,
    ToolResultEvent,
    Verification,
    Verifier,
    VerifierRouter,
    VerifyBeforeDoneVerifier,
    VerifyWorkTool,
    WorkItemClaimedEvent,
    WorkItemCompletedEvent,
    WorkItemCreatedEvent,
    WorkItemJudge,
    WorkItemOrphanedEvent,
    WorkItemRejectedEvent,
    WorkItemVerifiedEvent,
    build_ledger,
    configure_logging,
    correlate_defenses,
    fork_session,
    format_ledger,
    make_multi_critic,
    parse_ledger_text,
)
from harness.storage.memory import InMemoryStorage
from harness.storage.sqlite import SQLiteStorage, default_db_path
from harness.tasks import (
    ActivityEvent,
    ActivityStore,
    Task,
    TaskLink,
    TaskStore,
)
from harness.tasks import activity as task_activity
from harness.tools.fs import (
    EditFileTool,
    GlobTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from harness.tools.shell import ShellTool
from harness.tools.web import FetchUrlTool, TavilySearchTool

app = typer.Typer(
    name="harness",
    help="Harness — Python agent runtime over OpenRouter and Ollama.",
    no_args_is_help=True,
    add_completion=False,
)

sessions_app = typer.Typer(
    name="sessions", help="Inspect, resume, and remove saved sessions.", no_args_is_help=True
)
app.add_typer(sessions_app, name="sessions")

providers_app = typer.Typer(
    name="providers", help="Inspect available providers.", no_args_is_help=True
)
app.add_typer(providers_app, name="providers")

tools_app = typer.Typer(name="tools", help="Inspect the built-in tools.", no_args_is_help=True)
app.add_typer(tools_app, name="tools")

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

eval_app = typer.Typer(
    name="eval",
    help="Behavioral eval harness: run fixtures and score agent output.",
    no_args_is_help=True,
)
app.add_typer(eval_app, name="eval")

console = Console()

KNOWN_PROVIDERS: tuple[str, ...] = ("ollama", "openrouter")

_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful AI agent with access to filesystem and shell tools. "
    "Complete the task fully before responding — do not stop mid-task to ask "
    "questions or offer options. Use tools to find everything you need.\n\n"
    "## How to fix bugs (required workflow)\n\n"
    "Work like an engineer at a terminal: write code, run tests, read the output, fix, repeat.\n\n"
    "1. Read the relevant source files to understand the problem.\n"
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
    "8. Only then declare the task complete.\n\n"
    "Do NOT declare done after a single attempt. "
    "Do NOT assume your first fix is correct without running verify_work. "
    "Iteration is expected — most fixes take 2-4 attempts.\n\n"
    "## Other tools\n\n"
    "- request_critique: Get a second opinion on your approach before making changes. "
    "Describe what you plan to do and why — the critic will identify flaws. "
    "Call this when your diagnosis feels uncertain.\n\n"
    "Shell hygiene rules (follow these on every shell call):\n"
    "- Exclude .venv, __pycache__, node_modules, .git from find/glob commands:\n"
    "  find . -name '*.py' -not -path './.venv/*' -not -path './__pycache__/*'\n"
    "- Use pipes to count/sort/summarize large output: | wc -l, | sort | uniq -c | sort -rn\n"
    "- Never run long-running background processes.\n\n"
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

_SPAWN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "string",
            "description": (
                "Clear description of what to analyze or produce. "
                "Include which files or directories to read, what output format "
                "you need, and any constraints."
            ),
        },
    },
    "required": ["goal"],
}


class SpawnAgentsTool:
    """Spawn a multi-agent job to analyze files that would overflow the context window."""

    name = "spawn_agents"
    description = (
        "Spawn a multi-agent analysis job when you need to read and synthesize many large "
        "files that would overflow the context window. A Planner breaks the goal into "
        "independent work items, Workers read and analyze their assigned files, and a "
        "Reporter synthesizes the results. Use this when total file content exceeds ~200 KB."
    )
    effect_scope = "task_durable"
    approval: ApprovalDecision = "auto"
    phases: tuple[str, ...] = ("*",)

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        cwd: Path,
        config: HarnessConfig,
        max_workers: int = 3,
        approval_policy: ApprovalPolicy | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._cwd = cwd
        self._config = config
        self._max_workers = max_workers
        # Inherit the parent's approval policy/handler. If the parent runs
        # interactively (RichApprovalHandler), so do the children. If --yes,
        # children get AutoApprove. Without this, sub-agents silently
        # AutoApprove regardless of the parent's policy — letting a model
        # bypass approval prompts by spawning a worker to run shell.
        self._approval_policy = approval_policy or ApprovalPolicy(default="auto")
        self._approval_handler = approval_handler or AutoApprove()
        self.parameters_schema = _SPAWN_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        args: dict[str, Any] = call.arguments if isinstance(call.arguments, dict) else {}
        goal = args.get("goal", "").strip()
        if not goal:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="'goal' is required",
                is_error=True,
            )

        store = InMemoryStorage()

        def agent_factory(role: AgentRole) -> Agent:
            job_id = role.job_id or "_job_"
            item_id = role.item_id or "_item_"
            sub_tools = ToolRegistry()

            if role.name == "planner":
                sub_tools.register(ListDirTool(cwd=self._cwd))
                sub_tools.register(CreateWorkItemTool(store, parent_id=job_id, cwd=self._cwd))
                sub_tools.register(ListWorkItemsTool(store, job_id))
            elif role.name.startswith("worker"):
                sub_tools.register(ReadFileTool(cwd=self._cwd))
                sub_tools.register(ListDirTool(cwd=self._cwd))
                sub_tools.register(GlobTool(cwd=self._cwd))
                sub_tools.register(ShellTool(cwd=self._cwd))
                sub_tools.register(ListWorkItemsTool(store, job_id))
                sub_tools.register(CompleteWorkItemTool(store, item_id))
            else:  # reporter
                sub_tools.register(ReadFileTool(cwd=self._cwd))
                sub_tools.register(ListDirTool(cwd=self._cwd))
                sub_tools.register(GlobTool(cwd=self._cwd))
                sub_tools.register(ListWorkItemsTool(store, job_id))

            adapter = _build_adapter(self._provider, base_url=None, config=self._config)
            return Agent(
                adapters={self._provider: adapter},
                tools=sub_tools,
                storage=store,
                failover=FailoverPolicy(chain=[self._provider]),
                approval_policy=self._approval_policy,
                approval_handler=self._approval_handler,
                default_model=role.model or self._model,
                default_cwd=str(self._cwd),
                system_prompt=role.system_prompt,
            )

        planner_role = AgentRole(
            name="planner",
            system_prompt=(
                "You are a Planner. Read the goal carefully and decompose it into "
                "independent work items — one per distinct area the goal explicitly "
                "asks about. Do NOT explore the whole project. Use list_dir or glob "
                "only when you need to confirm which specific paths exist for a part "
                "of the goal. Create as few items as needed. Stop immediately after "
                "calling create_work_item for each part."
            ),
        )
        worker_role = AgentRole(
            name="worker",
            max_steps=15,
            system_prompt=(
                "You are a Worker. Read the assigned files, perform the analysis, "
                "and write a clear result summary. "
                "CRITICAL: Call complete_work_item as a tool call when done — "
                "do not write it as plain text."
            ),
        )
        reporter_role = AgentRole(
            name="reporter",
            system_prompt=(
                "You are a Reporter. Read the completed work item summaries and "
                "synthesize a clear, structured final answer for the user."
            ),
        )

        orchestrator = MultiAgentOrchestrator(
            agent_factory=agent_factory,
            store=store,
            planner_role=planner_role,
            worker_role=worker_role,
            reporter_role=reporter_role,
            max_workers=self._max_workers,
            job_cwd=self._cwd,
            provider=self._provider,
            model=self._model,
        )

        reporter_text: list[str] = []
        async for event in orchestrator.run(goal):
            if (
                isinstance(event, AgentEventWrapper)
                and event.role == "reporter"
                and isinstance(event.event, TextDelta)
            ):
                reporter_text.append(event.event.text)

        result = "".join(reporter_text).strip()
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=result or "No output from agents.",
        )


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _workspace_db(cwd: Path) -> Path | None:
    """Return `.harness/harness.db` in cwd if it exists, else None."""
    candidate = cwd / ".harness" / "harness.db"
    return candidate if candidate.exists() else None


def _build_storage(*, db: Path | None, in_memory: bool, cwd: Path | None = None) -> Storage:
    if in_memory:
        return InMemoryStorage()
    resolved = db or (cwd and _workspace_db(cwd)) or default_db_path()
    return SQLiteStorage(path=resolved)


def _build_adapter(provider: str, *, base_url: str | None, config: HarnessConfig) -> Adapter:
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
    if provider == "anthropic":
        return AnthropicAdapter(base_url=effective_base_url)
    raise typer.BadParameter(f"unknown provider: {provider!r}")


def _build_tools(cwd: Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadFileTool(cwd=cwd))
    registry.register(WriteFileTool(cwd=cwd))
    registry.register(EditFileTool(cwd=cwd))
    registry.register(ListDirTool(cwd=cwd))
    registry.register(GlobTool(cwd=cwd))
    registry.register(ShellTool(cwd=cwd))
    registry.register(TavilySearchTool())
    registry.register(FetchUrlTool())
    return registry


def _build_verifier(
    verify: str | None,
    *,
    chain: list[str],
    model: str,
    config: HarnessConfig,
    cwd: Path | None = None,
    verify_command: str | None = None,
) -> Verifier | None:
    """Resolve --verify value to a Verifier instance (or None).

    Options:
      grounding  — ClaimGroundingVerifier only (free, no LLM call)
      state      — StateVerifier only (filesystem + shell re-run checks)
      rule       — RuleVerifier only (heuristic: stalls, refusals, tool errors)
      shell      — ShellVerifier: runs --verify-command and checks exit code
      llm        — LLMJudgeVerifier only (one extra adapter call)
      auto       — ChainedVerifier: grounding → state → rule/llm router
      none       — disabled
    """
    if not verify or verify == "none":
        return None
    if verify == "grounding":
        return ClaimGroundingVerifier()
    if verify == "state":
        return StateVerifier(cwd=cwd or Path.cwd())
    if verify == "rule":
        return RuleVerifier()
    if verify == "shell":
        if not verify_command:
            raise typer.BadParameter("--verify shell requires --verify-command <cmd>")
        return ShellVerifier(verify_command, cwd=cwd)
    if verify == "llm":
        adapter = _build_adapter(chain[0], base_url=None, config=config)
        return LLMJudgeVerifier(adapter=adapter, model=model)
    if verify == "auto":
        adapter = _build_adapter(chain[0], base_url=None, config=config)
        return ChainedVerifier(
            ClaimGroundingVerifier(),
            StateVerifier(cwd=cwd or Path.cwd()),
            VerifierRouter(
                rule=RuleVerifier(),
                llm=LLMJudgeVerifier(adapter=adapter, model=model),
            ),
        )
    raise typer.BadParameter(
        f"unknown --verify value: {verify!r} (use grounding|state|rule|shell|llm|auto|none)"
    )


def _build_search_fn() -> Any:
    """Return a TavilySearchTool wrapper if TAVILY_API_KEY is set, else None."""
    if not os.environ.get("TAVILY_API_KEY"):
        return None
    try:
        from harness.tools.web import TavilySearchTool

        _searcher = TavilySearchTool()

        async def _search(query: str) -> str:
            call = ToolCall(id=f"s_{query[:8]}", name="web_search", arguments={"query": query})
            result: ToolResult = await _searcher(call)
            return result.content or ""

        return _search
    except Exception:
        return None


def _build_critic(
    critic: str | None,
    *,
    chain: list[str],
    model: str,
    config: HarnessConfig,
) -> Critic | None:
    """Resolve --critic value to a Critic instance (or None).

    Options:
      llm        — MultiCritic without web search
      llm+search — MultiCritic with Tavily web search (requires TAVILY_API_KEY)
      none       — disabled (default)
    """
    if not critic or critic == "none":
        return None
    if critic in ("llm", "llm+search"):
        adapter = _build_adapter(chain[0], base_url=None, config=config)
        search_fn = _build_search_fn() if critic == "llm+search" else None
        return make_multi_critic(adapter=adapter, model=model, search_fn=search_fn)
    raise typer.BadParameter(f"unknown --critic value: {critic!r} (use llm|llm+search|none)")


def _load_project_context(cwd: Path) -> str:
    """Walk from cwd up to filesystem root, collecting CLAUDE.md and AGENTS.md files.

    Returns an XML-tagged block suitable for injection into the system prompt, or an
    empty string if no files are found.
    """
    target_names = {"CLAUDE.md", "AGENTS.md"}
    collected: list[str] = []
    current = cwd.resolve()
    visited: set[Path] = set()
    while True:
        if current in visited:
            break
        visited.add(current)
        for name in sorted(target_names):
            candidate = current / name
            if candidate.is_file():
                try:
                    text = candidate.read_text(encoding="utf-8", errors="replace").strip()
                    if text:
                        collected.append(f"# {candidate}\n{text}")
                except OSError:
                    pass
        parent = current.parent
        if parent == current:
            break
        current = parent

    if not collected:
        return ""
    body = "\n\n---\n\n".join(reversed(collected))
    return f"<project_instructions>\n{body}\n</project_instructions>"


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
) -> Agent:
    """Build an Agent over a provider chain. `chain[0]` is the primary.

    Pass `activity_store` / `approval_store` (typically the same storage
    instance) to enable activity-ledger emission and approval-replay on
    resume.

    Handler precedence: `--yes` (AutoApprove) > `--inbox` (InboxApprovalHandler)
    > default (RichApprovalHandler).

    `profile` selects the structural defense level:
      - "bare":    no chain, no critic. Model + tools only.
      - "minimal": only VerifyBeforeDoneVerifier (catches "forgot to test").
      - "strict":  full chain (FileScope, MinimalFix, TestsBeforeEdit,
                   VerifyBeforeDone, DiagnosisAlignment, MisdirectedSuggestion).

    Default is "minimal" — cross-model A/B data (see evals/EVAL.md) showed
    the full strict chain regressed pass rate on capable hosted models. The
    minimal profile keeps the one defense that consistently correlates with
    PASS (forgot-to-test) without the over-engineering surface of the rest.
    """
    if not chain:
        raise typer.BadParameter("provider chain is empty")
    if inbox and approval_store is None:
        raise typer.BadParameter("--inbox requires an approval_store (passed by _build_agent)")

    # --base-url applies to the primary provider only; others use their defaults.
    adapters: dict[str, Adapter] = {}
    for i, provider in enumerate(chain):
        provider_base_url = base_url if i == 0 else None
        adapters[provider] = _build_adapter(provider, base_url=provider_base_url, config=config)

    project_ctx = _load_project_context(cwd)
    if project_ctx and system_prompt:
        system_prompt = f"{system_prompt}\n\n{project_ctx}"
    elif project_ctx:
        system_prompt = project_ctx

    tools = _build_tools(cwd)
    tools.register(VerifyWorkTool(cwd=cwd))
    primary_adapter = adapters[chain[0]]
    tools.register(
        RequestCritiqueTool(
            adapter=primary_adapter,
            model=model,
            search_fn=_build_search_fn(),
        )
    )

    # SpawnAgentsTool registration is deferred until after we've built the
    # parent's approval policy + handler, so we can pass them down to sub-
    # agents. See the registration block lower in this function.

    # Always enforce six structural defenses (unless bare=True):
    #   1. If the prompt named specific files, the agent may only modify those.
    #   2. If the prompt asked for a "minimal fix", the agent may not write a large diff.
    #   3. If the agent edited, it must have run the tests first.
    #   4. If the agent modified files, it must call verify_work.
    #   5. If verify_work still shows failing tests, the agent's edits must
    #      share vocabulary with those failing test names — otherwise it's
    #      editing the wrong layer.
    #   6. Once tests pass, every edit must share vocabulary with at least
    #      one historical failing test — otherwise it's scope creep driven
    #      by the prompt rather than the bug.
    # All deterministic — no LLM, no false positives on no-op turns.
    # Profile controls which run:
    #   bare    → none of these wire in.
    #   minimal → only VerifyBeforeDoneVerifier (the "forgot to test" catch).
    #   strict  → full chain.
    if profile == "strict":
        enforce_scope = FileScopeVerifier()
        enforce_minimal = MinimalFixVerifier()
        enforce_tests_first = TestsBeforeEditVerifier()
        enforce_verify = VerifyBeforeDoneVerifier()
        enforce_alignment = DiagnosisAlignmentVerifier()
        enforce_no_scope_creep = MisdirectedSuggestionVerifier()
        structural = ChainedVerifier(
            enforce_scope,
            enforce_minimal,
            enforce_tests_first,
            enforce_verify,
            enforce_alignment,
            enforce_no_scope_creep,
        )
        verifier = ChainedVerifier(structural, verifier) if verifier is not None else structural
    elif profile == "minimal":
        # Just the catch for "agent edited but never ran tests" — the one
        # defense that correlated with PASS in every A/B run so far.
        enforce_verify = VerifyBeforeDoneVerifier()
        verifier = (
            ChainedVerifier(enforce_verify, verifier) if verifier is not None else enforce_verify
        )
    # else profile == "bare": nothing structural, just whatever user passed.

    approval_policy = ApprovalPolicy(default="prompt", per_tool=dict(config.approval))

    approval_handler: ApprovalHandler
    if yes:
        approval_handler = AutoApprove()
    elif inbox:
        assert approval_store is not None  # checked above
        approval_handler = InboxApprovalHandler(approval_store=approval_store)
    else:
        approval_handler = RichApprovalHandler(console=console, session_overrides=session_overrides)

    # Register SpawnAgentsTool now that we have the parent's approval policy
    # and handler — sub-agents will inherit both, closing the previous
    # silent-AutoApprove escape path.
    tools.register(
        SpawnAgentsTool(
            provider=chain[0],
            model=model,
            cwd=cwd,
            config=config,
            approval_policy=approval_policy,
            approval_handler=approval_handler,
        )
    )

    multi = len(chain) > 1
    return Agent(
        adapters=adapters,
        tools=tools,
        storage=storage,
        failover=FailoverPolicy(
            chain=chain,
            max_attempts=max(len(chain), 1),
            backoff_base=0.5 if multi else 0.0,
            backoff_max=10.0,
            backoff_jitter=0.2 if multi else 0.0,
        ),
        approval_policy=approval_policy,
        approval_handler=approval_handler,
        activity_store=activity_store,
        approval_store=approval_store,
        verifier=verifier,
        critic=critic,
        budget=budget,
        default_model=model,
        default_cwd=str(cwd),
        memory_store=memory_store,
        planner=planner,
        predictor=predictor,
        repair=repair,
        system_prompt=system_prompt,
        compactor=compactor,
        max_repair_attempts=max_repair_attempts,
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


def _resolve_chain(
    *,
    failover_flag: str | None,
    provider_flag: str | None,
    config: HarnessConfig,
) -> list[str]:
    """Resolve the provider chain from --failover > --provider > config > 'ollama'."""
    if failover_flag:
        chain = [p.strip() for p in failover_flag.split(",") if p.strip()]
        if not chain:
            raise typer.BadParameter("--failover chain is empty")
        return chain
    return [provider_flag or config.default_provider or "ollama"]


def _load_cli_config(config_path: Path | None) -> HarnessConfig:
    try:
        return load_config(config_path)
    except Exception as exc:
        console.print(f"[red]Bad config:[/red] {exc}")
        raise typer.Exit(2) from None


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
            "--provider", "-p", help="Provider: 'ollama' or 'openrouter' (overrides config)."
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
            help="Comma-separated provider chain (e.g. 'ollama,openrouter'). Overrides --provider.",
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
                "(model + tools only). 'minimal' (default) = only the "
                "tests-before-done check. 'strict' = full chain (file scope, "
                "minimal fix, tests-first, verify-before-done, diagnosis "
                "alignment, misdirected suggestion). See evals/EVAL.md for "
                "the cross-model A/B data behind the default."
            ),
        ),
    ] = "minimal",
    bare: Annotated[
        bool,
        typer.Option(
            "--bare",
            help="Deprecated alias for --profile bare.",
            hidden=True,
        ),
    ] = False,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable DEBUG logging to stderr.")
    ] = False,
) -> None:
    """Run a single prompt through the agent and stream the result to stdout."""
    configure_logging(level="DEBUG" if verbose else "INFO")

    # HARNESS_YES=1 in the environment is equivalent to --yes, making it easy
    # to run the agent autonomously without repeating the flag every invocation.
    if not yes and os.environ.get("HARNESS_YES"):
        yes = True

    # --bare is a deprecated alias for --profile bare. If both are given,
    # --bare takes precedence (legacy behavior). If neither, use --profile.
    if bare:
        profile = "bare"
    if profile not in ("bare", "minimal", "strict"):
        console.print(
            f"[red]Invalid --profile {profile!r}; expected bare, minimal, or strict.[/red]"
        )
        raise typer.Exit(2)

    cfg = _load_cli_config(config_path)
    chain = _resolve_chain(failover_flag=failover, provider_flag=provider, config=cfg)
    effective_model = model or cfg.default_model or "llama3.2"

    working_dir = (cwd or Path.cwd()).resolve()
    if not working_dir.exists() or not working_dir.is_dir():
        console.print(f"[red]--cwd does not exist or is not a directory: {working_dir}[/red]")
        raise typer.Exit(2)

    try:
        asyncio.run(
            _run_once(
                prompt=prompt,
                model=effective_model,
                chain=chain,
                base_url=base_url,
                cwd=working_dir,
                max_steps=max_steps,
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
                config=cfg,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user.[/yellow]")
        raise typer.Exit(130) from None


async def _run_once(
    *,
    prompt: str,
    model: str,
    chain: list[str],
    base_url: str | None,
    cwd: Path,
    max_steps: int,
    session_id: str | None,
    task_ref: str | None,
    db: Path | None,
    in_memory: bool,
    yes: bool,
    inbox: bool,
    verify: str | None,
    verify_command: str | None = None,
    critic: str | None = None,
    require_tools: bool = False,
    goal: bool = False,
    max_context_tokens: int | None = None,
    predict: bool = False,
    auto_compact: bool = False,
    max_repair: int = 3,
    profile: str = "minimal",
    config: HarnessConfig,
) -> None:
    storage = _build_storage(db=db, in_memory=in_memory, cwd=cwd)
    try:
        # Resolve the optional task attachment first (validates ref exists and
        # appends session_id to task.session_ids).
        task_id, _task = await _resolve_task_attachment(storage, task_ref, session_id)

        verifier = _build_verifier(
            verify, chain=chain, model=model, config=config, cwd=cwd, verify_command=verify_command
        )
        # The "bare" profile disables the critic too — the A/B baseline arm
        # is the model + tools alone, with no harness-side reasoning support.
        critic_obj = (
            None
            if profile == "bare"
            else _build_critic(critic, chain=chain, model=model, config=config)
        )
        budget = (
            ContextBudget(max_tokens=max_context_tokens) if max_context_tokens is not None else None
        )
        planner: Planner | None = None
        if goal:
            adapter = _build_adapter(chain[0], base_url=base_url, config=config)
            planner = LLMPlanner(adapter=adapter, model=model)
        compactor: ContextCompactor | None = None
        if auto_compact:
            adapter = _build_adapter(chain[0], base_url=base_url, config=config)
            compactor = ContextCompactor(adapter=adapter, model=model)
        agent = _build_agent(
            chain=chain,
            base_url=base_url,
            model=model,
            storage=storage,
            cwd=cwd,
            config=config,
            yes=yes,
            inbox=inbox,
            activity_store=storage,  # type: ignore[arg-type]
            approval_store=storage,  # type: ignore[arg-type]
            verifier=verifier,
            critic=critic_obj,
            budget=budget,
            memory_store=storage,  # type: ignore[arg-type]
            planner=planner,
            predictor=ConsequencePredictor() if predict else None,
            repair=RepairOrchestrator() if predict else None,
            system_prompt=_DEFAULT_SYSTEM_PROMPT,
            compactor=compactor,
            max_repair_attempts=max_repair,
            profile=profile,
        )

        request_kwargs: dict[str, object] = {
            "prompt": prompt,
            "model": model,
            "max_steps": max_steps,
            "require_tool_use": require_tools,
        }
        if session_id:
            request_kwargs["session_id"] = session_id
        if task_id:
            request_kwargs["task_id"] = task_id
        request = RunRequest(**request_kwargs)  # type: ignore[arg-type]

        last_verification: Verification | None = None
        try:
            async for event in agent.run(request):
                _render(event)
                if isinstance(event, Verification):
                    last_verification = event
        except Exception as exc:
            console.print(f"\n[red]Unhandled error:[/red] {exc!s}")
            raise typer.Exit(1) from None
        await _print_defense_ledger(storage, session_id)
    finally:
        if isinstance(storage, SQLiteStorage):
            await storage.close()

    # Surface the final verifier verdict to the shell. The repair loop has
    # already exhausted its retries by this point — a blocking final verdict
    # means the harness ran out of budget while the work was still wrong.
    # Eval tooling reads this exit code to distinguish "agent succeeded" from
    # "agent gave up after every defense fired."
    if last_verification is not None and not last_verification.result.can_finish:
        raise typer.Exit(2)


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
    """List saved sessions, newest first."""

    async def _go() -> None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            sessions = await storage.list(limit=limit, status=status)  # type: ignore[arg-type]
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

        if not sessions:
            console.print("[dim]No sessions.[/dim]")
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("ID")
        table.add_column("Status")
        table.add_column("Provider")
        table.add_column("Model")
        table.add_column("Updated")
        table.add_column("Turns", justify="right")
        for s in sessions:
            table.add_row(
                s.id,
                _status_style(s.status),
                s.provider,
                s.model,
                _ago(s.updated_at),
                str(len(s.messages)),
            )
        console.print(table)

    asyncio.run(_go())


@sessions_app.command("show")
def sessions_show(
    session_id: Annotated[str, typer.Argument(help="Session id to display.")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    """Print a session's full transcript."""

    async def _go() -> None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            session = await storage.get(session_id)
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

        if session is None:
            console.print(f"[red]Session not found:[/red] {session_id}")
            raise typer.Exit(1)
        _render_session(session)

    asyncio.run(_go())


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
    """Continue a saved session, optionally with a new user prompt."""
    configure_logging(level="DEBUG" if verbose else "INFO")

    cfg = _load_cli_config(config_path)

    async def _go() -> None:
        working_dir_hint = cwd.resolve() if cwd else None
        storage = _build_storage(db=db, in_memory=in_memory, cwd=working_dir_hint)
        try:
            session = await storage.get(session_id)
            if session is None:
                console.print(f"[red]Session not found:[/red] {session_id}")
                raise typer.Exit(1)

            working_dir = (cwd or session.cwd).resolve()
            chain = _resolve_chain(
                failover_flag=failover, provider_flag=session.provider, config=cfg
            )
            verifier = _build_verifier(verify, chain=chain, model=session.model, config=cfg)
            budget = (
                ContextBudget(max_tokens=max_context_tokens)
                if max_context_tokens is not None
                else None
            )
            agent = _build_agent(
                chain=chain,
                base_url=base_url,
                model=session.model,
                storage=storage,
                cwd=working_dir,
                config=cfg,
                yes=yes,
                inbox=inbox,
                activity_store=storage,  # type: ignore[arg-type]
                approval_store=storage,  # type: ignore[arg-type]
                verifier=verifier,
                budget=budget,
                memory_store=storage,  # type: ignore[arg-type]
            )
            try:
                async for event in agent.resume(session_id, prompt=prompt, max_steps=max_steps):
                    _render(event)
            except Exception as exc:
                console.print(f"\n[red]Unhandled error:[/red] {exc!s}")
                raise typer.Exit(1) from None
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    asyncio.run(_go())


@sessions_app.command("rm")
def sessions_rm(
    session_id: Annotated[str, typer.Argument(help="Session id to delete.")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Delete a saved session."""
    if not yes and not Confirm.ask(f"Delete session [bold]{session_id}[/bold]?", default=False):
        console.print("[yellow]Aborted.[/yellow]")
        raise typer.Exit(1)

    async def _go() -> None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            await storage.delete(session_id)
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    asyncio.run(_go())
    console.print(f"[green]Deleted[/green] {session_id}")


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
    """Branch a new session from an existing session's message history."""
    configure_logging(level="DEBUG" if verbose else "INFO")
    cfg = _load_cli_config(config_path)

    async def _go() -> None:
        from harness.core.errors import ConfigurationError as _CE

        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            try:
                forked = await fork_session(storage, session_id, new_session_id=new_id)
            except _CE as exc:
                console.print(f"[red]Error:[/red] {exc}")
                raise typer.Exit(1) from None
            console.print(f"[green]Forked[/green] {session_id} → {forked.id}")
            if not prompt:
                console.print(
                    f'[dim]Resume with:[/dim] harness sessions resume {forked.id} "<prompt>"'
                )
                return
            # Run immediately with the given prompt.
            chain = [forked.provider]
            agent = _build_agent(
                chain=chain,
                base_url=None,
                model=forked.model,
                storage=storage,
                cwd=forked.cwd,
                config=cfg,
                yes=yes,
                activity_store=storage,  # type: ignore[arg-type]
                approval_store=storage,  # type: ignore[arg-type]
                memory_store=storage,  # type: ignore[arg-type]
            )
            async for event in agent.resume(forked.id, prompt=prompt):
                _render(event)
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    try:
        asyncio.run(_go())
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user.[/yellow]")
        raise typer.Exit(130) from None


@sessions_app.command("diff")
def sessions_diff_cmd(
    session_id: Annotated[str, typer.Argument(help="Session id to diff.")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    """Show file changes made during a session."""

    async def _go() -> None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            activity = await storage.list_activity(session_id=session_id)  # type: ignore[attr-defined]
            _render_session_diff(activity, console)
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    asyncio.run(_go())


# ---------------------------------------------------------------------------
# providers subcommands
# ---------------------------------------------------------------------------


@providers_app.command("list")
def providers_list_cmd(
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """List known providers and their configuration status."""
    cfg = _load_cli_config(config_path)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Provider")
    table.add_column("Status")
    table.add_column("Notes")

    settings = cfg.provider("ollama")
    ollama_base = settings.get("base_url") or os.environ.get(
        "OLLAMA_HOST", "http://localhost:11434"
    )
    table.add_row(
        "ollama",
        "[green]ready[/green]",
        f"base_url: {ollama_base}",
    )

    has_or_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    or_settings = cfg.provider("openrouter")
    or_status = "[green]ready[/green]" if has_or_key else "[red]missing OPENROUTER_API_KEY[/red]"
    or_notes_parts = []
    if has_or_key:
        or_notes_parts.append("env: OPENROUTER_API_KEY set")
    if "http_referer" in or_settings:
        or_notes_parts.append(f"http_referer: {or_settings['http_referer']}")
    if "x_title" in or_settings:
        or_notes_parts.append(f"x_title: {or_settings['x_title']}")
    table.add_row("openrouter", or_status, ", ".join(or_notes_parts) or "—")

    console.print(table)


@providers_app.command("capabilities")
def providers_capabilities_cmd(
    name: Annotated[str, typer.Argument(help="Provider name (ollama or openrouter).")],
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Print a provider's reported Capabilities."""
    cfg = _load_cli_config(config_path)
    if name not in KNOWN_PROVIDERS:
        console.print(f"[red]Unknown provider:[/red] {name}")
        raise typer.Exit(2)

    async def _go() -> None:
        try:
            adapter = _build_adapter(name, base_url=None, config=cfg)
        except Exception as exc:
            console.print(f"[red]Could not construct adapter:[/red] {exc}")
            raise typer.Exit(2) from None
        caps = await adapter.capabilities()
        table = Table(show_header=False)
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("streaming", str(caps.streaming))
        table.add_row("tool_use", str(caps.tool_use))
        table.add_row("structured_output", str(caps.structured_output))
        table.add_row(
            "max_context_tokens",
            "—" if caps.max_context_tokens is None else str(caps.max_context_tokens),
        )
        table.add_row(
            "models",
            "—" if caps.models is None else ", ".join(caps.models),
        )
        console.print(table)

    asyncio.run(_go())


# ---------------------------------------------------------------------------
# tools subcommands
# ---------------------------------------------------------------------------


@tools_app.command("list")
def tools_list_cmd(
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Working directory used to construct fs/shell tools."),
    ] = None,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """List built-in tools with their effective approval levels."""
    cfg = _load_cli_config(config_path)
    working_dir = (cwd or Path.cwd()).resolve()
    registry = _build_tools(working_dir)
    policy = ApprovalPolicy(default="prompt", per_tool=dict(cfg.approval))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Tool")
    table.add_column("Approval")
    table.add_column("Description")
    for tool in registry.all():
        effective = policy.decide(tool)
        color = {"auto": "green", "prompt": "yellow", "deny": "red"}.get(effective, "white")
        table.add_row(
            tool.name,
            f"[{color}]{effective}[/{color}]",
            _truncate(tool.description, 80),
        )
    console.print(table)


# ---------------------------------------------------------------------------
# tasks subcommands
# ---------------------------------------------------------------------------


async def _append_task_activity(
    storage: ActivityStore, *, task_id: str, kind: str, data: dict
) -> None:
    """Append a task-domain activity event to the ledger."""
    event = ActivityEvent(task_id=task_id, kind=kind, data=data)
    await storage.append_activity(event)


def _close_if_sqlite(storage: object) -> bool:
    return isinstance(storage, SQLiteStorage)


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
    """Create a new task."""
    working_dir = (cwd or Path.cwd()).resolve()
    label_list: list[str] = [s.strip() for s in labels.split(",") if s.strip()] if labels else []

    async def _go() -> Task:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            store: TaskStore = storage  # type: ignore[assignment]
            parent_id: str | None = None
            if parent:
                parent_task = await store.get_task_by_ref(parent)
                if parent_task is None:
                    console.print(f"[red]Parent task not found:[/red] {parent}")
                    raise typer.Exit(1)
                parent_id = parent_task.id

            draft = Task(
                ref="",  # filled in by the store
                title=title,
                description=description,
                priority=priority,  # type: ignore[arg-type]
                labels=label_list,
                parent_id=parent_id,
                cwd=working_dir,
            )
            saved = await store.create_task(draft)
            await _append_task_activity(
                storage,  # type: ignore[arg-type]
                task_id=saved.id,
                kind=task_activity.TASK_CREATED,
                data={"ref": saved.ref, "title": saved.title},
            )
            return saved
        finally:
            if _close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    task = asyncio.run(_go())
    console.print(f"[green]Created[/green] {task.ref}  {task.title}")


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
    """List tasks, newest-updated first."""

    async def _go() -> list[Task]:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            store: TaskStore = storage  # type: ignore[assignment]
            return await store.list_tasks(limit=limit, status=status)  # type: ignore[arg-type]
        finally:
            if _close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    tasks = asyncio.run(_go())
    if not tasks:
        console.print("[dim]No tasks.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Ref")
    table.add_column("Status")
    table.add_column("Title")
    table.add_column("Labels")
    table.add_column("Updated")
    for t in tasks:
        table.add_row(
            t.ref,
            _task_status_style(t.status),
            _truncate(t.title, 60),
            ", ".join(t.labels) if t.labels else "—",
            _ago(t.updated_at),
        )
    console.print(table)


@tasks_app.command("show")
def tasks_show_cmd(
    ref: Annotated[str, typer.Argument(help="Task ref, e.g. T-001.")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    """Print a task's full details + activity log."""

    async def _go() -> tuple[Task | None, list[ActivityEvent]]:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            store: TaskStore = storage  # type: ignore[assignment]
            task = await store.get_task_by_ref(ref)
            if task is None:
                return None, []
            activity_store: ActivityStore = storage  # type: ignore[assignment]
            events = await activity_store.list_activity(task_id=task.id, limit=200)
            return task, events
        finally:
            if _close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    task, events = asyncio.run(_go())
    if task is None:
        console.print(f"[red]Task not found:[/red] {ref}")
        raise typer.Exit(1)
    _render_task(task, events)


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
    """Update a task."""

    async def _go() -> Task | None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            store: TaskStore = storage  # type: ignore[assignment]
            task = await store.get_task_by_ref(ref)
            if task is None:
                return None
            old_status = task.status
            if status is not None:
                task.status = status  # type: ignore[assignment]
            if title is not None:
                task.title = title
            if description is not None:
                task.description = description
            if priority is not None:
                task.priority = priority  # type: ignore[assignment]
            if labels is not None:
                task.labels = [s.strip() for s in labels.split(",") if s.strip()]
            task.touch()
            saved = await store.update_task(task)
            kind = (
                task_activity.TASK_STATUS_CHANGED
                if status is not None and status != old_status
                else task_activity.TASK_UPDATED
            )
            data: dict[str, object] = {"ref": saved.ref}
            if status is not None and status != old_status:
                data["from"] = old_status
                data["to"] = status
            await _append_task_activity(storage, task_id=saved.id, kind=kind, data=data)  # type: ignore[arg-type]
            return saved
        finally:
            if _close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    task = asyncio.run(_go())
    if task is None:
        console.print(f"[red]Task not found:[/red] {ref}")
        raise typer.Exit(1)
    console.print(f"[green]Updated[/green] {task.ref}")


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
    """Add a typed link from one task to another."""

    async def _go() -> Task | None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            store: TaskStore = storage  # type: ignore[assignment]
            task = await store.get_task_by_ref(ref)
            if task is None:
                return None
            target_task = await store.get_task_by_ref(target)
            if target_task is None:
                console.print(f"[red]Target task not found:[/red] {target}")
                raise typer.Exit(1)
            task.links.append(TaskLink(target_ref=target, relation=relation))  # type: ignore[arg-type]
            task.touch()
            saved = await store.update_task(task)
            await _append_task_activity(
                storage,  # type: ignore[arg-type]
                task_id=saved.id,
                kind=task_activity.TASK_LINKED,
                data={"ref": saved.ref, "target_ref": target, "relation": relation},
            )
            return saved
        finally:
            if _close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    task = asyncio.run(_go())
    if task is None:
        console.print(f"[red]Task not found:[/red] {ref}")
        raise typer.Exit(1)
    console.print(f"[green]Linked[/green] {task.ref} --{relation}--> {target}")


@tasks_app.command("rm")
def tasks_rm_cmd(
    ref: Annotated[str, typer.Argument(help="Task ref.")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Delete a task."""
    if not yes and not Confirm.ask(f"Delete task [bold]{ref}[/bold]?", default=False):
        console.print("[yellow]Aborted.[/yellow]")
        raise typer.Exit(1)

    async def _go() -> bool:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            store: TaskStore = storage  # type: ignore[assignment]
            task = await store.get_task_by_ref(ref)
            if task is None:
                return False
            await store.delete_task(task.id)
            await _append_task_activity(
                storage,  # type: ignore[arg-type]
                task_id=task.id,
                kind=task_activity.TASK_DELETED,
                data={"ref": ref},
            )
            return True
        finally:
            if _close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    if not asyncio.run(_go()):
        console.print(f"[red]Task not found:[/red] {ref}")
        raise typer.Exit(1)
    console.print(f"[green]Deleted[/green] {ref}")


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
    """List queued tool-call approvals, newest-requested first."""

    async def _go() -> list[PendingApproval]:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            store: ApprovalStore = storage  # type: ignore[assignment]
            task_id: str | None = None
            if task:
                task_obj = await storage.get_task_by_ref(task)  # type: ignore[union-attr]
                if task_obj is None:
                    console.print(f"[red]Task not found:[/red] {task}")
                    raise typer.Exit(1)
                task_id = task_obj.id
            return await store.list_approvals(
                session_id=session_id,
                task_id=task_id,
                status="pending" if pending_only else None,
                limit=limit,
            )
        finally:
            if _close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    items = asyncio.run(_go())
    if not items:
        console.print("[dim]No approvals.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", no_wrap=True)
    table.add_column("Status")
    table.add_column("Tool", no_wrap=True)
    table.add_column("Args")
    table.add_column("Session", no_wrap=True)
    table.add_column("Requested")
    for a in items:
        table.add_row(
            a.id,
            _approval_status_style(a.status),
            a.tool_name,
            _truncate(repr(a.arguments), 40),
            a.session_id,
            _ago(a.requested_at),
        )
    console.print(table)


@approvals_app.command("show")
def approvals_show_cmd(
    approval_id: Annotated[str, typer.Argument(help="Approval id (appr_...).")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    """Print full details for one approval."""

    async def _go() -> PendingApproval | None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            store: ApprovalStore = storage  # type: ignore[assignment]
            return await store.get_approval(approval_id)
        finally:
            if _close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    approval = asyncio.run(_go())
    if approval is None:
        console.print(f"[red]Approval not found:[/red] {approval_id}")
        raise typer.Exit(1)
    _render_approval(approval)


@approvals_app.command("grant")
def approvals_grant_cmd(
    approval_id: Annotated[str, typer.Argument(help="Approval id (appr_...).")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    """Mark an approval as granted. Replay happens on the next `sessions resume`."""

    async def _go() -> PendingApproval | None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            store: ApprovalStore = storage  # type: ignore[assignment]
            return await store.resolve_approval(approval_id, status="granted", resolved_by="cli")
        finally:
            if _close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    updated = asyncio.run(_go())
    if updated is None:
        console.print(f"[red]Approval not found:[/red] {approval_id}")
        raise typer.Exit(1)
    console.print(
        f"[green]Granted[/green] {updated.id}  "
        f"[dim]({updated.tool_name})[/dim]  — "
        f"resume the session to dispatch."
    )


@approvals_app.command("deny")
def approvals_deny_cmd(
    approval_id: Annotated[str, typer.Argument(help="Approval id (appr_...).")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    """Mark an approval as denied. No replay; the queued result stays in transcript."""

    async def _go() -> PendingApproval | None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            store: ApprovalStore = storage  # type: ignore[assignment]
            return await store.resolve_approval(approval_id, status="denied", resolved_by="cli")
        finally:
            if _close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    updated = asyncio.run(_go())
    if updated is None:
        console.print(f"[red]Approval not found:[/red] {approval_id}")
        raise typer.Exit(1)
    console.print(f"[yellow]Denied[/yellow] {updated.id}  [dim]({updated.tool_name})[/dim]")


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
    """List the tool-call evidence ledger.

    Each row is a `tool_call.completed` activity event — the runtime emits
    one per dispatched tool call, carrying timing, exit codes, sizes, and
    tool-specific metadata.
    """

    async def _go() -> list[ActivityEvent]:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            task_id: str | None = None
            if task:
                task_obj = await storage.get_task_by_ref(task)  # type: ignore[union-attr]
                if task_obj is None:
                    console.print(f"[red]Task not found:[/red] {task}")
                    raise typer.Exit(1)
                task_id = task_obj.id
            store: ActivityStore = storage  # type: ignore[assignment]
            events = await store.list_activity(
                task_id=task_id,
                session_id=session_id,
                kinds=("tool_call.completed",),
                limit=limit,
            )
        finally:
            if _close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

        if tool_name is not None:
            events = [e for e in events if e.data.get("name") == tool_name]
        if errors_only:
            events = [e for e in events if e.data.get("is_error") is True]
        return events

    items = asyncio.run(_go())
    if not items:
        console.print("[dim]No evidence.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("When")
    table.add_column("Tool")
    table.add_column("Args")
    table.add_column("Status")
    table.add_column("ms", justify="right")
    table.add_column("Evidence")
    for e in items:
        is_error = bool(e.data.get("is_error"))
        status = "[red]error[/red]" if is_error else "[green]ok[/green]"
        duration = e.data.get("duration_ms")
        duration_str = "—" if duration is None else str(duration)
        meta = e.data.get("metadata") or {}
        meta_str = _truncate(
            " ".join(f"{k}={v}" for k, v in meta.items()) or "—",
            60,
        )
        table.add_row(
            _ago(e.timestamp),
            str(e.data.get("name", "?")),
            _truncate(repr(e.data.get("arguments", {})), 30),
            status,
            duration_str,
            meta_str,
        )
    console.print(table)


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
    configure_logging(level="DEBUG" if verbose else "INFO")
    if not yes and os.environ.get("HARNESS_YES"):
        yes = True
    if verify == "none":
        verify = None
    cfg = _load_cli_config(config_path)
    chain = _resolve_chain(failover_flag=failover, provider_flag=provider, config=cfg)
    effective_model = model or cfg.default_model or "llama3.2"
    working_dir = (cwd or Path.cwd()).resolve()
    if not working_dir.exists() or not working_dir.is_dir():
        console.print(f"[red]--cwd does not exist or is not a directory: {working_dir}[/red]")
        raise typer.Exit(2)

    try:
        asyncio.run(
            _chat_loop(
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
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]bye[/yellow]")
        raise typer.Exit(130) from None


async def _chat_loop(
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
) -> None:
    from uuid import uuid4

    storage = _build_storage(db=db, in_memory=in_memory, cwd=cwd)
    try:
        existing: Session | None = None
        if session_id:
            existing = await storage.get(session_id)
        current_session_id = session_id or f"sess_{uuid4().hex[:12]}"

        task_id, _task = await _resolve_task_attachment(storage, task_ref, current_session_id)

        verifier = _build_verifier(verify, chain=chain, model=model, config=config, cwd=cwd)
        budget = (
            ContextBudget(max_tokens=max_context_tokens) if max_context_tokens is not None else None
        )
        compactor: ContextCompactor | None = None
        if auto_compact:
            adapter = _build_adapter(chain[0], base_url=base_url, config=config)
            compactor = ContextCompactor(adapter=adapter, model=model)
        agent = _build_agent(
            chain=chain,
            base_url=base_url,
            model=model,
            storage=storage,
            cwd=cwd,
            config=config,
            yes=yes,
            inbox=inbox,
            activity_store=storage,  # type: ignore[arg-type]
            approval_store=storage,  # type: ignore[arg-type]
            verifier=verifier,
            budget=budget,
            memory_store=storage,  # type: ignore[arg-type]
            system_prompt=_DEFAULT_SYSTEM_PROMPT,
            compactor=compactor,
        )

        first_turn = existing is None

        chain_label = chain[0]
        if len(chain) > 1:
            chain_label += "  [dim](failover: " + ", ".join(chain[1:]) + ")[/dim]"
        intro = (
            f"[bold]Session:[/bold] {current_session_id}"
            + (" [dim](resumed)[/dim]" if existing else "")
            + f"\n[bold]Provider:[/bold] {chain_label}"
            f"\n[bold]Model:[/bold] {model}"
            f"\n[bold]Tools:[/bold] {', '.join(agent.tools.names())}"
            f"\n[bold]CWD:[/bold] {cwd}\n\n"
            f"[dim]Type /help for commands. /quit to exit.[/dim]"
        )
        console.print(Panel(intro, title="harness chat", expand=False))

        while True:
            try:
                user_input = console.input("\n[bold cyan]> [/bold cyan]").strip()
            except EOFError:
                console.print("\n[yellow]bye[/yellow]")
                return
            except KeyboardInterrupt:
                console.print("\n[yellow]bye[/yellow]")
                return

            if not user_input:
                continue

            if user_input.startswith("/"):
                keep_going = await _handle_slash(
                    user_input, agent=agent, session_id=current_session_id, storage=storage
                )
                if not keep_going:
                    return
                continue

            try:
                if first_turn:
                    request_kwargs: dict[str, object] = {
                        "prompt": user_input,
                        "session_id": current_session_id,
                        "model": model,
                        "max_steps": max_steps,
                        "require_tool_use": require_tools,
                    }
                    if task_id:
                        request_kwargs["task_id"] = task_id
                    request = RunRequest(**request_kwargs)  # type: ignore[arg-type]
                    async for event in agent.run(request):
                        _render(event)
                    first_turn = False
                else:
                    async for event in agent.resume(
                        current_session_id, prompt=user_input, max_steps=max_steps
                    ):
                        _render(event)
            except (KeyboardInterrupt, asyncio.CancelledError):
                console.print("\n[yellow]cancelled[/yellow]")
            except Exception as exc:
                console.print(f"\n[red]Error:[/red] {exc!s}")
    finally:
        if isinstance(storage, SQLiteStorage):
            await storage.close()


_HELP_TEXT = (
    "/help              show this help\n"
    "/quit, /exit, /q   exit the chat\n"
    "/tools             list registered tools and effective approval\n"
    "/session           show current session id and turn count\n"
    "/diff              show file changes made this session\n"
    "/clear             clear the terminal\n"
    "/model [name]      show or switch the active model mid-session\n"
)

# ---------------------------------------------------------------------------
# Slash command registry
# ---------------------------------------------------------------------------

SlashHandler = Callable[..., Awaitable[bool]]

_SLASH_REGISTRY: dict[str, SlashHandler] = {}


def _slash(name: str) -> Callable[[SlashHandler], SlashHandler]:
    """Decorator to register a slash command handler."""

    def decorator(fn: SlashHandler) -> SlashHandler:
        _SLASH_REGISTRY[name] = fn
        return fn

    return decorator


@_slash("/quit")
@_slash("/exit")
@_slash("/q")
async def _slash_quit(line: str, *, agent: Agent, session_id: str, storage: Storage) -> bool:
    console.print("[yellow]bye[/yellow]")
    return False


@_slash("/help")
async def _slash_help(line: str, *, agent: Agent, session_id: str, storage: Storage) -> bool:
    console.print(Panel(_HELP_TEXT.rstrip(), title="commands", expand=False))
    return True


@_slash("/tools")
async def _slash_tools(line: str, *, agent: Agent, session_id: str, storage: Storage) -> bool:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Tool")
    table.add_column("Approval")
    for tool in agent.tools.all():
        effective = agent.approval_policy.decide(tool)
        color = {"auto": "green", "prompt": "yellow", "deny": "red"}.get(effective, "white")
        table.add_row(tool.name, f"[{color}]{effective}[/{color}]")
    console.print(table)
    return True


@_slash("/session")
async def _slash_session(line: str, *, agent: Agent, session_id: str, storage: Storage) -> bool:
    session = await storage.get(session_id)
    if session is None:
        console.print(f"[dim]Session {session_id} (no turns yet)[/dim]")
    else:
        console.print(
            f"[dim]Session {session_id}, status: {session.status}, "
            f"{len(session.messages)} messages[/dim]"
        )
    return True


@_slash("/diff")
async def _slash_diff(line: str, *, agent: Agent, session_id: str, storage: Storage) -> bool:
    activity = await storage.list_activity(session_id=session_id)  # type: ignore[attr-defined]
    _render_session_diff(activity, console)
    return True


@_slash("/clear")
async def _slash_clear(line: str, *, agent: Agent, session_id: str, storage: Storage) -> bool:
    console.clear()
    return True


@_slash("/model")
async def _slash_model(line: str, *, agent: Agent, session_id: str, storage: Storage) -> bool:
    parts = line.split(None, 1)
    if len(parts) == 1:
        console.print(f"[dim]Active model: {agent.default_model}[/dim]")
    else:
        new_model = parts[1].strip()
        agent.default_model = new_model
        console.print(f"[green]Switched model to:[/green] {new_model}")
    return True


async def _handle_slash(line: str, *, agent: Agent, session_id: str, storage: Storage) -> bool:
    """Dispatch a /command via the registry. Returns False to terminate the REPL."""
    cmd = line.split(None, 1)[0].lower()
    handler = _SLASH_REGISTRY.get(cmd)
    if handler is None:
        console.print(f"[red]Unknown command:[/red] {cmd}.  Try /help.")
        return True
    return await handler(line, agent=agent, session_id=session_id, storage=storage)


# ---------------------------------------------------------------------------
# Markdown preprocessing
# ---------------------------------------------------------------------------

_DOLLAR = re.escape("$")
_MATH_DISPLAY = re.compile(_DOLLAR * 2 + r"(.+?)" + _DOLLAR * 2, re.DOTALL)
_MATH_INLINE = re.compile(_DOLLAR + r"([^\n]+?)" + _DOLLAR)
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_MERMAID_FENCE = re.compile(r"```mermaid\n(.*?)\n```", re.DOTALL)

_mermaid_render_cache: dict[str, str] = {}


def _render_mermaid(source: str) -> str:
    """Convert mermaid source to an ASCII-art fenced block.

    Tries `mermaid_ascii.mermaid_to_ascii` (optional pip dep), then the
    `mermaid-ascii -i -` subprocess, then falls back to a plain code block
    so the diagram is still visible even without the optional dependency.
    """
    if source in _mermaid_render_cache:
        return _mermaid_render_cache[source]

    ascii_art: str | None = None
    try:
        from mermaid_ascii import mermaid_to_ascii  # type: ignore[import-untyped]  # optional

        result = mermaid_to_ascii(source)
        if result and result.strip():
            ascii_art = result.strip()
    except ImportError:
        import shutil
        import subprocess

        if shutil.which("mermaid-ascii"):
            try:
                proc = subprocess.run(
                    ["mermaid-ascii", "-i", "-"],
                    input=source,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    ascii_art = proc.stdout.strip()
            except Exception:
                pass
    except Exception:
        pass

    rendered = f"```\n{ascii_art}\n```" if ascii_art else f"```\n{source}\n```"
    _mermaid_render_cache[source] = rendered
    return rendered


def _convert_math(inner: str) -> str:
    """Convert the body of a $...$ or $$...$$ span via unicodeitplus."""
    return _unicodeit.replace(inner)


def _preprocess_markdown(text: str) -> str:
    """Prepare LLM output for Rich Markdown rendering.

    - Renders ```mermaid fences to ASCII art via mermaid_ascii (optional dep)
    - Converts LaTeX math spans ($...$, $$...$$) to Unicode via unicodeitplus
      (2 566-symbol table, handles subscripts/superscripts as Unicode chars)
    - Wraps <think>...</think> blocks in a dim blockquote
    """
    # Mermaid diagrams — intercept complete fences before Markdown sees them
    text = _MERMAID_FENCE.sub(lambda m: _render_mermaid(m.group(1)), text)
    # Display math first ($$...$$) to avoid partial matches
    text = _MATH_DISPLAY.sub(lambda m: f"`{_convert_math(m.group(1))}`", text)
    # Inline math ($...$)
    text = _MATH_INLINE.sub(lambda m: _convert_math(m.group(1)), text)
    # <think>...</think> blocks from reasoning models
    text = _THINK_BLOCK.sub(
        lambda m: "> *thinking: " + m.group(1).strip().replace("\n", " ")[:200] + "…*\n",
        text,
    )
    return text


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class Renderer:
    """Stateful event renderer.

    Text deltas are buffered and rendered as Markdown in a live context so
    the response updates in real-time while still applying syntax highlighting
    and other rich formatting. Spinners run between ToolCallEvent and
    ToolResultEvent; only one Live context is active at a time.
    """

    def __init__(self, con: Console) -> None:
        self._console = con
        self._live: Live | None = None
        self._live_kind: str = ""  # "spinner" | "text"
        self._pending_name: str = ""
        self._pending_start: float = 0.0
        self._text_buf: str = ""

    def render(self, event: Any) -> None:
        if isinstance(event, TextDelta):
            self._stop_spinner()
            self._text_buf += event.text
            rendered = Markdown(_preprocess_markdown(self._text_buf))
            if self._live is None:
                self._live = Live(
                    rendered,
                    console=self._console,
                    refresh_per_second=12,
                    vertical_overflow="visible",
                )
                self._live_kind = "text"
                self._live.start()
            else:
                self._live.update(rendered)
        elif isinstance(event, ToolCallEvent):
            self._flush_text()
            self._console.print()
            self._console.print(
                f"[blue]→[/blue] [bold]{event.call.name}[/bold]({_args_preview(event.call.arguments)})",
                style="dim",
            )
            self._start_spinner(event.call.name)
        elif isinstance(event, ToolResultEvent):
            elapsed = time.monotonic() - self._pending_start if self._pending_start else 0.0
            self._stop_spinner()
            marker = "[red]✗[/red]" if event.result.is_error else "[green]✓[/green]"
            full_len = len(event.result.content)
            preview = _truncate(event.result.content, 200)
            suffix = f"  [dim]… {full_len:,} bytes[/dim]" if full_len > 200 else ""
            self._console.print(
                f"{marker} {event.result.name}: {preview}{suffix}  [dim]({elapsed:.1f}s)[/dim]",
                style="dim",
            )
        elif isinstance(event, StepStarted):
            self._flush_text()
            if event.total_steps > 1:
                label = f"Step {event.step + 1}/{event.total_steps}"
                if event.description:
                    label += f": {event.description}"
                self._console.print(f"\n[bold blue]●[/bold blue] {label}")
        elif isinstance(event, StepCompleted):
            pass
        elif isinstance(event, ErrorEvent):
            self._stop_spinner()
            self._flush_text()
            self._console.print()
            self._console.print(f"[red]Error ({event.kind}):[/red] {event.error}")
        elif isinstance(event, Verification):
            self._flush_text()
            r = event.result
            marker = "[green]✓[/green]" if r.can_finish else "[red]✗[/red]"
            conf = (
                f"  [dim](confidence {r.confidence:.2f})[/dim]" if r.confidence is not None else ""
            )
            self._console.print()
            self._console.print(
                f"{marker} [bold]verify[/bold] ({r.verifier_name})  {r.reason}{conf}"
            )
        elif isinstance(event, Critique):
            self._flush_text()
            self._console.print()
            self._console.print(
                f"[yellow bold]critic[/yellow bold] [dim](attempt {event.attempt})[/dim]"
            )
            for line in event.text.splitlines():
                self._console.print(f"  [yellow]{line}[/yellow]")
            self._console.print()
        elif isinstance(event, PredictionEvent):
            p = event.prediction
            scope = p.effect_scope or "unknown"
            self._console.print(
                f"[dim]  ⟳ predict scope={scope} confidence={p.confidence:.2f} "
                f"expected={p.expected_status} reversibility={p.reversibility}[/dim]"
            )
        elif isinstance(event, PredictionMismatchEvent):
            o = event.outcome
            self._console.print(
                f"[yellow]  ⚠ mismatch severity={o.severity} actual={o.actual_status} "
                f"lesson={o.lesson}[/yellow]"
            )
        elif isinstance(event, Done):
            self._stop_spinner()
            self._flush_text()
            self._console.print()
            if event.usage:
                u = event.usage
                self._console.print(
                    f"[dim]tokens: {u.prompt_tokens:,} in / {u.completion_tokens:,} out[/dim]"
                )

    def _flush_text(self) -> None:
        """Stop the text Live context (leaving rendered markdown on screen) and reset buffer."""
        if self._live is not None and self._live_kind == "text":
            self._live.stop()
            self._live = None
            self._live_kind = ""
        self._text_buf = ""

    def _start_spinner(self, name: str) -> None:
        self._pending_name = name
        self._pending_start = time.monotonic()
        self._live = Live(
            Spinner("dots", text=f"[dim]{name}[/dim]"),
            console=self._console,
            refresh_per_second=10,
            transient=True,
        )
        self._live_kind = "spinner"
        self._live.start()

    def _stop_spinner(self) -> None:
        if self._live is not None and self._live_kind == "spinner":
            self._live.stop()
            self._live = None
            self._live_kind = ""
        self._pending_start = 0.0


_renderer = Renderer(console)


def _render(event: Any) -> None:
    _renderer.render(event)


async def _print_defense_ledger(storage: Storage, session_id: str | None) -> None:
    """List the activity ledger for the just-completed run and print a summary.

    If `session_id` was supplied to the run, filter to that session. Otherwise
    grab the most recent session from storage (the one we just created) and
    filter to it. Failures here must not bubble up — the ledger is observability,
    not control flow.
    """
    try:
        target_session_id = session_id
        if target_session_id is None:
            sessions = await storage.list(limit=1)  # type: ignore[attr-defined]
            if sessions:
                target_session_id = sessions[0].id
        activity_store: Any = storage
        if target_session_id is not None:
            events = await activity_store.list_activity(session_id=target_session_id, limit=500)
        else:
            events = await activity_store.list_activity(limit=500)
        ledger = build_ledger(events)
        if ledger.is_empty():
            return
        console.print(f"\n[dim]{format_ledger(ledger)}[/dim]")
    except Exception as exc:
        console.print(f"[dim]defense ledger unavailable: {exc!s}[/dim]")


def _render_session_diff(activity: list[ActivityEvent], con: Console) -> None:
    file_events = [
        e
        for e in activity
        if e.kind == "tool_call.completed"
        and e.data.get("name") in ("write_file", "edit_file")
        and not e.data.get("is_error")
    ]
    shell_events = [
        e
        for e in activity
        if e.kind == "tool_call.completed"
        and e.data.get("name") == "shell"
        and not e.data.get("is_error")
    ]

    if not file_events and not shell_events:
        con.print("[dim]No file changes in this session.[/dim]")
        return

    for e in file_events:
        meta = e.data.get("metadata") or {}
        path = meta.get("path", "?")
        before = (meta.get("content_before") or "").splitlines(keepends=True)
        after = (meta.get("content_after") or "").splitlines(keepends=True)
        diff = list(difflib.unified_diff(before, after, fromfile=f"a/{path}", tofile=f"b/{path}"))
        con.rule(f"[bold]{e.data.get('name')}  {path}[/bold]")
        if diff:
            for line in diff:
                style = (
                    "green" if line.startswith("+") else "red" if line.startswith("-") else "dim"
                )
                con.print(line.rstrip(), style=style, highlight=False)
        else:
            con.print("[dim](no diff — content not captured)[/dim]")

    if shell_events:
        con.rule("[bold]shell[/bold]")
        for e in shell_events:
            meta = e.data.get("metadata") or {}
            cmd = (e.data.get("arguments") or {}).get("command", "?")
            con.print(f"  [dim]{cmd}[/dim]  exit_code={meta.get('exit_code', '?')}")


def _render_session(session: Session) -> None:
    header = (
        f"[bold]{session.id}[/bold]  "
        f"{_status_style(session.status)}  "
        f"{session.provider}/{session.model}\n"
        f"[dim]created {_ago(session.created_at)}, updated {_ago(session.updated_at)}[/dim]\n"
        f"[dim]cwd: {session.cwd}[/dim]"
    )
    console.print(Panel(header, title="session", expand=False))

    for msg in session.messages:
        if msg.role == "user":
            console.print(Panel(msg.content or "", title="[cyan]user[/cyan]", expand=False))
        elif msg.role == "system":
            console.print(Panel(msg.content or "", title="[grey]system[/grey]", expand=False))
        elif msg.role == "assistant":
            parts: list[str] = []
            if msg.content:
                parts.append(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append(
                        f"→ {tc.name}({_args_preview(tc.arguments)})  [dim]({tc.id})[/dim]"
                    )
            console.print(
                Panel(
                    "\n".join(parts) or "[dim](empty turn)[/dim]",
                    title="[green]assistant[/green]",
                    expand=False,
                )
            )
        elif msg.role == "tool":
            console.print(
                Panel(
                    msg.content or "",
                    title=f"[yellow]tool: {msg.name}[/yellow]  [dim]({msg.tool_call_id})[/dim]",
                    expand=False,
                )
            )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


_STATUS_STYLES = {
    "pending": "white",
    "running": "blue",
    "paused": "yellow",
    "done": "green",
    "failed": "red",
    "cancelled": "magenta",
}


def _status_style(status: str) -> str:
    color = _STATUS_STYLES.get(status, "white")
    return f"[{color}]{status}[/{color}]"


_APPROVAL_STATUS_STYLES = {
    "pending": "yellow",
    "granted": "green",
    "denied": "red",
}


def _approval_status_style(status: str) -> str:
    color = _APPROVAL_STATUS_STYLES.get(status, "white")
    return f"[{color}]{status}[/{color}]"


def _render_approval(approval: PendingApproval) -> None:
    lines = [
        f"[bold]{approval.id}[/bold]  {_approval_status_style(approval.status)}  "
        f"{approval.tool_name}",
        f"[dim]session: {approval.session_id}[/dim]",
    ]
    if approval.task_id:
        lines.append(f"[dim]task: {approval.task_id}[/dim]")
    lines.append(f"[dim]tool_call_id: {approval.tool_call_id}[/dim]")
    lines.append(f"[dim]requested {_ago(approval.requested_at)}[/dim]")
    if approval.resolved_at:
        lines.append(
            f"[dim]resolved {_ago(approval.resolved_at)} by {approval.resolved_by or '—'}[/dim]"
        )
    if approval.replayed_at:
        lines.append(f"[dim]replayed {_ago(approval.replayed_at)}[/dim]")
    console.print(Panel("\n".join(lines), title="approval", expand=False))

    if approval.arguments:
        import json as _json

        console.print(
            Panel(
                _json.dumps(approval.arguments, indent=2),
                title="[blue]arguments[/blue]",
                expand=False,
            )
        )


_TASK_STATUS_STYLES = {
    "backlog": "white",
    "todo": "cyan",
    "in_progress": "blue",
    "waiting": "yellow",
    "done": "green",
    "cancelled": "magenta",
}


def _task_status_style(status: str) -> str:
    color = _TASK_STATUS_STYLES.get(status, "white")
    return f"[{color}]{status}[/{color}]"


def _render_task(task: Task, events: list[ActivityEvent]) -> None:
    """Render a task header + body + activity timeline."""
    header_lines = [
        f"[bold]{task.ref}[/bold]  {_task_status_style(task.status)}  {task.title}",
        f"[dim]created {_ago(task.created_at)}, updated {_ago(task.updated_at)}[/dim]",
        f"[dim]cwd: {task.cwd}[/dim]",
    ]
    if task.priority:
        header_lines.append(f"[dim]priority: {task.priority}[/dim]")
    if task.labels:
        header_lines.append(f"[dim]labels: {', '.join(task.labels)}[/dim]")
    if task.parent_id:
        header_lines.append(f"[dim]parent: {task.parent_id}[/dim]")
    console.print(Panel("\n".join(header_lines), title="task", expand=False))

    if task.description:
        console.print(Panel(task.description, title="[cyan]description[/cyan]", expand=False))

    if task.links:
        link_lines = [f"{link.relation:<12} → {link.target_ref}" for link in task.links]
        console.print(Panel("\n".join(link_lines), title="[blue]links[/blue]", expand=False))

    if task.session_ids:
        console.print(
            Panel(
                "\n".join(task.session_ids),
                title=f"[magenta]sessions ({len(task.session_ids)})[/magenta]",
                expand=False,
            )
        )

    if events:
        lines = [
            f"[dim]{e.timestamp.isoformat(timespec='seconds')}[/dim]  "
            f"[bold]{e.kind}[/bold]  {_compact_event_data(e.data)}"
            for e in events
        ]
        console.print(
            Panel(
                "\n".join(lines),
                title=f"[yellow]activity ({len(events)})[/yellow]",
                expand=False,
            )
        )


def _compact_event_data(data: dict) -> str:
    if not data:
        return ""
    parts = []
    for k, v in data.items():
        if isinstance(v, str) and len(v) > 60:
            v = v[:57] + "…"
        parts.append(f"{k}={v}")
    return " ".join(parts)


def _ago(dt: datetime) -> str:
    now = datetime.now(UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _args_preview(args: dict) -> str:
    if not args:
        return ""
    parts = [f"{k}={_truncate(repr(v), 40)}" for k, v in args.items()]
    return ", ".join(parts)


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


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
    """Run a multi-step goal: the LLM plans first, then executes each step."""
    configure_logging(level="DEBUG" if verbose else "INFO")

    cfg = _load_cli_config(config_path)
    chain = _resolve_chain(failover_flag=None, provider_flag=provider, config=cfg)
    effective_model = model or cfg.default_model or "llama3.2"

    working_dir = (cwd or Path.cwd()).resolve()
    if not working_dir.exists() or not working_dir.is_dir():
        console.print(f"[red]--cwd does not exist or is not a directory: {working_dir}[/red]")
        raise typer.Exit(2)

    try:
        asyncio.run(
            _run_once(
                prompt=prompt,
                model=effective_model,
                chain=chain,
                base_url=base_url,
                cwd=working_dir,
                max_steps=max_steps,
                session_id=None,
                task_ref=None,
                db=db,
                in_memory=in_memory,
                yes=yes,
                inbox=False,
                verify=None,
                goal=True,
                max_context_tokens=None,
                config=cfg,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user.[/yellow]")
        raise typer.Exit(130) from None


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
    """Initialise a workspace-local Harness database in .harness/harness.db."""
    working_dir = (cwd or Path.cwd()).resolve()
    harness_dir = working_dir / ".harness"
    db_path = harness_dir / "harness.db"

    if db_path.exists():
        console.print(f"[dim]Already initialised at [/dim]{db_path}[dim] — nothing to do.[/dim]")
        return

    harness_dir.mkdir(parents=True, exist_ok=True)
    # Touch the db so SQLiteStorage picks it up next time.
    db_path.touch()
    console.print(
        f"[green]Initialized[/green] harness workspace at {harness_dir}"
        "\n[dim]Future commands run from this directory will use .harness/harness.db[/dim]"
    )

    gitignore = working_dir / ".gitignore"
    if gitignore.exists():
        content = gitignore.read_text()
        if ".harness/" not in content:
            console.print(
                "\n[dim]Tip: add [/dim].harness/[dim] to .gitignore to keep the db local.[/dim]"
            )


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
    """Save a new memory entry."""
    _VALID_KINDS = {"user_preference", "user_fact", "project_fact", "project_context"}
    if kind not in _VALID_KINDS:
        console.print(
            f"[red]Invalid --kind:[/red] {kind!r}. Choose from: {', '.join(sorted(_VALID_KINDS))}"
        )
        raise typer.Exit(1)

    async def _go() -> None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            entry = MemoryEntry(kind=kind, text=text)  # type: ignore[arg-type]
            saved = await storage.save_memory(entry)  # type: ignore[attr-defined]
            console.print(f"[green]Saved[/green] {saved.id}  ({saved.kind})  {saved.text}")
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    asyncio.run(_go())


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
    """List stored memory entries."""
    _VALID_KINDS = {"user_preference", "user_fact", "project_fact", "project_context"}
    if kind is not None and kind not in _VALID_KINDS:
        console.print(f"[red]Invalid --kind:[/red] {kind!r}")
        raise typer.Exit(1)

    async def _go() -> None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            entries = await storage.list_memory(  # type: ignore[attr-defined]
                kind=kind,
                limit=limit,  # type: ignore[arg-type]
            )
            if not entries:
                console.print("[dim]No memories stored.[/dim]")
                return
            table = Table(title="Memories", show_header=True)
            table.add_column("ID", no_wrap=True)
            table.add_column("Kind")
            table.add_column("Text")
            table.add_column("Created")
            for e in entries:
                table.add_row(e.id, e.kind, e.text, _ago(e.created_at))
            console.print(table)
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    asyncio.run(_go())


@memory_app.command("search")
def memory_search(
    query: Annotated[str, typer.Argument(help="Search query (substring match).")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    """Search memory entries by text."""

    async def _go() -> None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            entries = await storage.search_memory(query, limit=limit)  # type: ignore[attr-defined]
            if not entries:
                console.print("[dim]No matches.[/dim]")
                return
            table = Table(title=f"Memory search: {query!r}", show_header=True)
            table.add_column("ID", no_wrap=True)
            table.add_column("Kind")
            table.add_column("Text")
            table.add_column("Created")
            for e in entries:
                table.add_row(e.id, e.kind, e.text, _ago(e.created_at))
            console.print(table)
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    asyncio.run(_go())


@memory_app.command("rm")
def memory_rm(
    entry_id: Annotated[str, typer.Argument(help="Memory entry ID to delete.")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Delete a memory entry by ID."""

    async def _go() -> None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            existing = await storage.list_memory(limit=1000)  # type: ignore[attr-defined]
            match = next((e for e in existing if e.id == entry_id), None)
            if match is None:
                console.print(f"[red]Memory not found:[/red] {entry_id}")
                raise typer.Exit(1)
            if not yes:
                confirmed = Confirm.ask(f"Delete memory {entry_id!r}?")
                if not confirmed:
                    raise typer.Abort()
            await storage.delete_memory(entry_id)  # type: ignore[attr-defined]
            console.print(f"[green]Deleted[/green] {entry_id}")
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    asyncio.run(_go())


# ---------------------------------------------------------------------------
# lab subcommands — multi-agent orchestration
# ---------------------------------------------------------------------------


_ROLE_COLORS = {
    "planner": "blue",
    "reporter": "green",
}


def _role_color(role: str) -> str:
    if role.startswith("worker"):
        return "cyan"
    return _ROLE_COLORS.get(role, "white")


class LabRenderer:
    """Renders OrchestratorEvent stream with role-color coding."""

    def __init__(self, con: Console) -> None:
        self._console = con
        self._text_bufs: dict[str, str] = {}

    def render(self, event: object) -> None:
        if isinstance(event, AgentStartedEvent):
            color = _role_color(event.role)
            self._console.print(f"[{color}]▶ {event.role}[/{color}]")
        elif isinstance(event, AgentDoneEvent):
            color = _role_color(event.role)
            # Flush any buffered text for this role
            buf = self._text_bufs.pop(event.role, "").strip()
            if buf:
                self._console.print(f"  [{color}][{event.role}][/{color}] {buf}")
            self._console.print(
                f"[{color}]✓ {event.role} done ({event.turn_count} turns)[/{color}]"
            )
        elif isinstance(event, WorkItemCreatedEvent):
            self._console.print(f"[dim]  + {event.task_ref} {event.title}[/dim]")
        elif isinstance(event, WorkItemClaimedEvent):
            self._console.print(f"[cyan]  → claimed {event.task_ref}[/cyan]")
        elif isinstance(event, WorkItemCompletedEvent):
            self._console.print(f"[cyan]  ✓ completed {event.task_ref}[/cyan]")
        elif isinstance(event, WorkItemVerifiedEvent):
            conf_str = f" ({event.confidence:.0%})" if event.confidence is not None else ""
            self._console.print(f"[green]  ✓ {event.task_ref} verified{conf_str}[/green]")
        elif isinstance(event, WorkItemRejectedEvent):
            self._console.print(
                f"[yellow]  ✗ {event.task_ref} rejected"
                f" (attempt {event.attempt}): {event.reason}[/yellow]"
            )
        elif isinstance(event, WorkItemOrphanedEvent):
            self._console.print(
                f"[yellow]  ~ {event.task_ref} orphaned"
                f" (attempt {event.attempt}) — re-queued[/yellow]"
            )
        elif isinstance(event, PlanRejectedEvent):
            self._console.print(
                f"[red]  ✗ plan rejected (attempt {event.attempt}): {event.reason}[/red]"
            )
        elif isinstance(event, AgentEventWrapper):
            self._render_wrapped(event.role, event.event)

    def _render_wrapped(self, role: str, event: object) -> None:
        color = _role_color(role)
        prefix = f"  [{color}][{role}][/{color}]"
        if isinstance(event, TextDelta):
            self._text_bufs.setdefault(role, "")
            self._text_bufs[role] += event.text
        elif isinstance(event, Done):
            buf = self._text_bufs.pop(role, "").strip()
            if buf:
                # Print each line with role prefix
                for line in buf.splitlines():
                    if line.strip():
                        self._console.print(f"{prefix} {line}")
            if event.usage:
                u = event.usage
                self._console.print(
                    f"{prefix} [dim]tokens: {u.prompt_tokens:,}in / {u.completion_tokens:,}out[/dim]"
                )
        elif isinstance(event, ToolCallEvent):
            self._console.print(
                f"{prefix} [dim]→ [bold]{event.call.name}[/bold]"
                f"({_args_preview(event.call.arguments)})[/dim]"
            )
        elif isinstance(event, ToolResultEvent):
            marker = "[red]✗[/red]" if event.result.is_error else "[green]✓[/green]"
            preview = _truncate(event.result.content, 120)
            self._console.print(f"{prefix} [dim]{marker} {event.result.name}: {preview}[/dim]")
        elif isinstance(event, ErrorEvent):
            self._console.print(f"{prefix} [red]error ({event.kind}):[/red] {event.error}")


@lab_app.command("run")
def lab_run(
    prompt: Annotated[str, typer.Argument(help="Top-level task prompt for the planner.")],
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="LLM provider (ollama, openrouter, …)."),
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

    async def _run() -> None:
        cfg = _load_cli_config(config_path)
        working_dir = (cwd or Path.cwd()).resolve()
        resolved_provider = provider or cfg.default_provider or "ollama"
        resolved_model = model or cfg.default_model or "llama3.2"

        if db is not None:
            from harness.storage.sqlite import SQLiteStorage

            storage: InMemoryStorage = SQLiteStorage(path=db)  # type: ignore[assignment]
        else:
            storage = InMemoryStorage()
        worker_budget = (
            ContextBudget(max_tokens=max_context_tokens) if max_context_tokens is not None else None
        )
        renderer = LabRenderer(console)

        def agent_factory(role: AgentRole) -> Agent:
            job_id = role.job_id or "_job_"
            item_id = role.item_id or "_item_"

            tools = ToolRegistry()

            if role.name == "planner":
                # Planner only sees queue tools — no filesystem access to prevent drift
                tools.register(CreateWorkItemTool(storage, parent_id=job_id, cwd=working_dir))
                tools.register(ListWorkItemsTool(storage, job_id))
            elif role.name.startswith("worker"):
                tools.register(ReadFileTool(cwd=working_dir))
                tools.register(ListDirTool(cwd=working_dir))
                tools.register(GlobTool(cwd=working_dir))
                tools.register(WriteFileTool(cwd=working_dir))
                tools.register(EditFileTool(cwd=working_dir))
                tools.register(ShellTool(cwd=working_dir))
                tools.register(TavilySearchTool())
                tools.register(FetchUrlTool())
                tools.register(ListWorkItemsTool(storage, job_id))
                tools.register(CompleteWorkItemTool(storage, item_id))
            else:
                # reporter: read-only + queue listing
                tools.register(ReadFileTool(cwd=working_dir))
                tools.register(ListDirTool(cwd=working_dir))
                tools.register(GlobTool(cwd=working_dir))
                tools.register(ListWorkItemsTool(storage, job_id))

            adapters = {
                resolved_provider: _build_adapter(resolved_provider, base_url=None, config=cfg)
            }
            return Agent(
                adapters=adapters,
                tools=tools,
                storage=storage,
                failover=FailoverPolicy(chain=[resolved_provider]),
                approval_policy=ApprovalPolicy(default="auto"),
                approval_handler=AutoApprove(),
                activity_store=storage,  # type: ignore[arg-type]
                approval_store=storage,  # type: ignore[arg-type]
                memory_store=storage,  # type: ignore[arg-type]
                default_model=role.model or resolved_model,
                default_cwd=str(working_dir),
                system_prompt=role.system_prompt,
                predictor=ConsequencePredictor(),
                repair=RepairOrchestrator(),
                budget=worker_budget if role.name.startswith("worker") else None,
            )

        resolved_planner_model = planner_model or resolved_model
        resolved_worker_model = worker_model or resolved_model
        resolved_reporter_model = reporter_model or resolved_model

        planner_role = AgentRole(
            name="planner",
            model=resolved_planner_model,
            system_prompt=(
                "You are a Planner. Your ONLY job is to decompose the user's task into "
                "independent work items using create_work_item.\n\n"
                "Rules:\n"
                "1. Read the task carefully. Each work item must be completable on its own "
                "without depending on the output of another work item.\n"
                "2. Use as few work items as possible — prefer 1-3 self-contained items over "
                "4+ sequential steps. If the task can be done in one item, use one.\n"
                "3. Do NOT read files, run commands, or do any work yourself.\n"
                "4. Once you have called create_work_item for each sub-task, stop immediately."
            ),
        )
        worker_role = AgentRole(
            name="worker",
            model=resolved_worker_model,
            max_steps=max_steps,
            system_prompt=(
                "You are a Worker. Complete the assigned work item using tools.\n\n"
                "1. Read the work item title and description.\n"
                "2. Use the minimum tools needed to complete it.\n"
                "3. Call complete_work_item(summary=...) as soon as the work is done. "
                "The summary must describe what you actually did (file names, commands run, "
                "results computed) — not just 'task completed'.\n\n"
                "CRITICAL: Call complete_work_item as a tool call, not as plain text. "
                "Do NOT write 'complete_work_item(...)' in your response — call it as a tool. "
                "Do not loop or re-read files unnecessarily. Stay focused."
            ),
        )
        reporter_role = AgentRole(
            name="reporter",
            model=resolved_reporter_model,
            system_prompt=(
                "You are a Reporter. Synthesize the completed work items into a clear, "
                "concise final report for the user."
            ),
        )

        judge_adapter = _build_adapter(resolved_provider, base_url=None, config=cfg)
        work_item_judge: WorkItemJudge | None = None
        if not no_judge:
            work_item_judge = WorkItemJudge(
                adapter=judge_adapter,
                model=resolved_planner_model,
            )

        orchestrator = MultiAgentOrchestrator(
            agent_factory=agent_factory,
            store=storage,
            planner_role=planner_role,
            worker_role=worker_role,
            reporter_role=reporter_role,
            max_workers=workers,
            max_worker_steps=max_steps,
            job_cwd=working_dir,
            provider=resolved_provider,
            model=resolved_model,
            work_item_judge=work_item_judge,
            activity_store=storage,
        )

        console.print(f"[bold]harness lab run[/bold] — {workers} workers  max-steps={max_steps}")
        if resolved_planner_model == resolved_worker_model == resolved_reporter_model:
            console.print(
                f"[dim]provider=[/dim]{resolved_provider}  [dim]model=[/dim]{resolved_model}"
            )
        else:
            console.print(
                f"[dim]provider=[/dim]{resolved_provider}  "
                f"[dim]planner=[/dim]{resolved_planner_model}  "
                f"[dim]worker=[/dim]{resolved_worker_model}  "
                f"[dim]reporter=[/dim]{resolved_reporter_model}"
            )
        console.print()

        try:
            async for event in orchestrator.run(prompt):
                renderer.render(event)
        finally:
            if hasattr(storage, "close"):
                await storage.close()  # type: ignore[attr-defined]

    asyncio.run(_run())


@lab_app.command("status")
def lab_status(
    job_id: Annotated[str, typer.Argument(help="Job ID to inspect.")],
    db: Annotated[
        Path,
        typer.Option("--db", help="SQLite database path."),
    ] = Path("harness.db"),
) -> None:
    """Show work item status for a job stored in a SQLite database."""

    async def _run() -> None:
        from harness.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(path=db)
        try:
            items = await storage.list_tasks(parent_id=job_id)
            if not items:
                console.print(f"[yellow]No work items found for job {job_id!r}[/yellow]")
                return

            status_colors = {
                "todo": "white",
                "in_progress": "cyan",
                "done": "green",
                "cancelled": "red",
            }

            console.print(f"[bold]Job {job_id}[/bold] — {len(items)} work items\n")
            for item in sorted(items, key=lambda t: t.created_at):
                color = status_colors.get(item.status, "white")
                summary = item.metadata.get("result_summary", "")
                summary_str = f"  [dim]{summary[:80]}[/dim]" if summary else ""
                retries = item.metadata.get("_judge_retries", 0)
                retry_str = f" [yellow](retried {retries}x)[/yellow]" if retries else ""
                console.print(
                    f"  [{color}]{item.status:12}[/{color}] {item.ref or item.id[:8]}  {item.title}"
                    f"{retry_str}{summary_str}"
                )
        finally:
            await storage.close()

    asyncio.run(_run())


@lab_app.command("list")
def lab_list(
    db: Annotated[
        Path,
        typer.Option("--db", help="SQLite database path."),
    ] = Path("harness.db"),
) -> None:
    """List all jobs in a SQLite database."""

    async def _run() -> None:
        from harness.storage.sqlite import SQLiteStorage

        storage = SQLiteStorage(path=db)
        try:
            # Root tasks have no parent_id
            all_tasks = await storage.list_tasks(parent_id=None)
            jobs = [t for t in all_tasks if t.parent_id is None]
            if not jobs:
                console.print("[yellow]No jobs found.[/yellow]")
                return

            status_colors = {
                "todo": "white",
                "in_progress": "cyan",
                "done": "green",
                "cancelled": "red",
            }

            for job in sorted(jobs, key=lambda t: t.created_at, reverse=True):
                color = status_colors.get(job.status, "white")
                items = await storage.list_tasks(parent_id=job.id)
                done_count = sum(1 for t in items if t.status == "done")
                total_count = len(items)
                ts = job.created_at.strftime("%Y-%m-%d %H:%M")
                console.print(
                    f"[{color}]{job.status:12}[/{color}]  {job.id[:16]}  "
                    f"[dim]{ts}[/dim]  {done_count}/{total_count} items  {job.title[:60]}"
                )
        finally:
            await storage.close()

    asyncio.run(_run())


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

    async def _run() -> None:
        from harness.storage.sqlite import SQLiteStorage

        cfg = _load_cli_config(config_path)
        resolved_provider = provider or cfg.default_provider or "ollama"
        resolved_model = model or cfg.default_model or "llama3.2"
        resolved_worker_model = worker_model or resolved_model
        resolved_planner_model = planner_model or resolved_model

        storage = SQLiteStorage(path=db)
        renderer = LabRenderer(console)

        # Look up the job to get its cwd
        root = await storage.get_task(job_id)
        if root is None:
            console.print(f"[red]Job {job_id!r} not found in {db}[/red]")
            raise typer.Exit(1)

        working_dir = root.cwd

        worker_budget: ContextBudget | None = None

        def agent_factory(role: AgentRole) -> Agent:
            job = role.job_id or "_job_"
            item = role.item_id or "_item_"
            tools = ToolRegistry()

            if role.name.startswith("worker"):
                tools.register(ReadFileTool(cwd=working_dir))
                tools.register(ListDirTool(cwd=working_dir))
                tools.register(GlobTool(cwd=working_dir))
                tools.register(WriteFileTool(cwd=working_dir))
                tools.register(EditFileTool(cwd=working_dir))
                tools.register(ShellTool(cwd=working_dir))
                tools.register(TavilySearchTool())
                tools.register(FetchUrlTool())
                tools.register(ListWorkItemsTool(storage, job))
                tools.register(CompleteWorkItemTool(storage, item))
            else:
                tools.register(ReadFileTool(cwd=working_dir))
                tools.register(ListDirTool(cwd=working_dir))
                tools.register(GlobTool(cwd=working_dir))
                tools.register(ListWorkItemsTool(storage, job))

            adapters = {
                resolved_provider: _build_adapter(resolved_provider, base_url=None, config=cfg)
            }
            return Agent(
                adapters=adapters,
                tools=tools,
                storage=storage,
                failover=FailoverPolicy(chain=[resolved_provider]),
                approval_policy=ApprovalPolicy(default="auto"),
                approval_handler=AutoApprove(),
                activity_store=storage,  # type: ignore[arg-type]
                approval_store=storage,  # type: ignore[arg-type]
                memory_store=storage,  # type: ignore[arg-type]
                default_model=role.model or resolved_model,
                default_cwd=str(working_dir),
                system_prompt=role.system_prompt,
                predictor=ConsequencePredictor(),
                repair=RepairOrchestrator(),
                budget=worker_budget if role.name.startswith("worker") else None,
            )

        worker_role = AgentRole(
            name="worker",
            model=resolved_worker_model,
            max_steps=max_steps,
            system_prompt=(
                "You are a Worker. Complete the assigned work item using tools.\n\n"
                "1. Read the work item title and description.\n"
                "2. Use the minimum tools needed to complete it.\n"
                "3. Call complete_work_item(summary=...) as soon as the work is done. "
                "The summary must describe what you actually did.\n\n"
                "CRITICAL: Call complete_work_item as a tool call, not as plain text."
            ),
        )
        reporter_role = AgentRole(
            name="reporter",
            model=resolved_model,
            system_prompt=(
                "You are a Reporter. Synthesize the completed work items into a clear, "
                "concise final report for the user."
            ),
        )
        planner_role = AgentRole(
            name="planner",
            model=resolved_planner_model,
            system_prompt="",
        )

        judge_adapter = _build_adapter(resolved_provider, base_url=None, config=cfg)
        work_item_judge: WorkItemJudge | None = None
        if not no_judge:
            work_item_judge = WorkItemJudge(
                adapter=judge_adapter,
                model=resolved_planner_model,
            )

        orchestrator = MultiAgentOrchestrator(
            agent_factory=agent_factory,
            store=storage,
            planner_role=planner_role,
            worker_role=worker_role,
            reporter_role=reporter_role,
            max_workers=workers,
            max_worker_steps=max_steps,
            job_cwd=working_dir,
            provider=resolved_provider,
            model=resolved_model,
            work_item_judge=work_item_judge,
            activity_store=storage,
        )

        console.print(f"[bold]harness lab resume[/bold] {job_id[:16]}  — {workers} workers")
        console.print(
            f"[dim]provider=[/dim]{resolved_provider}  "
            f"[dim]worker-model=[/dim]{resolved_worker_model}"
        )
        console.print()

        try:
            async for event in orchestrator.resume(job_id):
                renderer.render(event)
        finally:
            await storage.close()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# eval subcommands
# ---------------------------------------------------------------------------


def _load_eval_module(name: str, evals_root: Path):
    """Load runner.py or judge.py from the evals/ directory at runtime."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, evals_root / f"{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {name}.py from {evals_root}")
    import sys as _sys

    mod = importlib.util.module_from_spec(spec)
    _sys.modules[name] = mod  # register before exec so @dataclass can resolve its module
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _find_evals_root() -> Path | None:
    """Walk CWD upward looking for evals/fixtures/."""
    current = Path.cwd().resolve()
    while True:
        if (current / "evals" / "fixtures").is_dir():
            return current / "evals"
        parent = current.parent
        if parent == current:
            return None
        current = parent


@eval_app.command("list")
def eval_list() -> None:
    """List all available eval fixtures."""
    evals_root = _find_evals_root()
    if evals_root is None:
        console.print("[red]No evals/fixtures/ directory found — run from the harness repo.[/red]")
        raise typer.Exit(1)

    runner = _load_eval_module("runner", evals_root)
    fixtures = runner.discover_fixtures(evals_root)
    if not fixtures:
        console.print("[dim]No fixtures found.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Fixture", no_wrap=True)
    table.add_column("Primary Dimension")
    table.add_column("Trap (summary)")
    for fx in fixtures:
        primary = ""
        trap = ""
        for line in fx.eval_md.splitlines():
            if line.startswith("primary_dimension:"):
                primary = line.split(":", 1)[1].strip()
            if line.strip().startswith("trap:") or (trap and line.startswith(" ")):
                trap += line.split(":", 1)[-1].strip() + " "
        table.add_row(fx.name, primary, _truncate(trap.strip(), 70))
    console.print(table)


@eval_app.command("run")
def eval_run(
    fixture_name: Annotated[
        str | None,
        typer.Argument(help="Fixture to run (e.g. 01-reproduce-before-repair). Omit to run all."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", "-p", help="Provider for the agent."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model for the agent."),
    ] = None,
    judge_model: Annotated[
        str | None,
        typer.Option("--judge-model", help="Model for the judge (defaults to --model)."),
    ] = None,
    judge_provider: Annotated[
        str | None,
        typer.Option(
            "--judge-provider",
            help="Provider for the judge (defaults to --provider, or ollama when provider=claude).",
        ),
    ] = None,
    agent_timeout: Annotated[
        int,
        typer.Option("--timeout", help="Agent timeout per fixture in seconds."),
    ] = 300,
    n_runs: Annotated[
        int,
        typer.Option(
            "--n-runs",
            help=(
                "Run each fixture N times to measure variance. Reports median "
                "score per dimension and (min..max) range in the final table. "
                "Use 3+ on non-deterministic local models. Default 1."
            ),
        ),
    ] = 1,
    ab: Annotated[
        bool,
        typer.Option(
            "--ab",
            help=(
                "A/B mode: run each fixture twice per rep — once with the "
                "full harness defense chain (defended), once with --bare "
                "(no structural verifiers, no critic). Reports both arms "
                "side-by-side so you can measure the harness's value-add."
            ),
        ),
    ] = False,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Run one or all eval fixtures and display scored results."""
    evals_root = _find_evals_root()
    if evals_root is None:
        console.print("[red]No evals/fixtures/ directory found — run from the harness repo.[/red]")
        raise typer.Exit(1)

    cfg = _load_cli_config(config_path)
    resolved_provider = provider or cfg.default_provider or "ollama"
    resolved_model = model or cfg.default_model or "llama3.2"
    resolved_judge_model = judge_model or resolved_model
    # claude -p has no adapter; fall back to ollama for judging unless overridden.
    resolved_judge_provider = judge_provider or (
        "ollama" if resolved_provider == "claude" else resolved_provider
    )

    runner = _load_eval_module("runner", evals_root)
    judge_mod = _load_eval_module("judge", evals_root)

    fixtures = runner.discover_fixtures(evals_root)
    if fixture_name:
        fixtures = [f for f in fixtures if f.name == fixture_name]
        if not fixtures:
            console.print(f"[red]Fixture not found:[/red] {fixture_name}")
            raise typer.Exit(1)

    if not fixtures:
        console.print("[dim]No fixtures to run.[/dim]")
        return

    judge_adapter = _build_adapter(resolved_judge_provider, base_url=None, config=cfg)

    def _score_cell(score: int) -> str:
        color = "green" if score >= 4 else ("yellow" if score == 3 else "red")
        return f"[{color}]{score}/5[/{color}]"

    _DIM_ORDER = (
        ("verification", "Verif."),
        ("scope", "Scope"),
        ("decomposition", "Decomp."),
        ("correctness", "Correct."),
        ("pushback", "Pushback"),
        ("epistemic", "Epist."),
        ("overall", "Overall"),
    )

    def _print_per_fixture(r: Any) -> None:
        """Print one fixture's scorecard + rationales immediately, so a
        killed batch still leaves partial results in the transcript."""
        row_table = Table(show_header=True, header_style="bold")
        row_table.add_column("Fixture", no_wrap=True)
        for _, col in _DIM_ORDER:
            row_table.add_column(col, justify="center")
        row_table.add_column("Pass?", justify="center")
        row_table.add_row(
            r.fixture_name,
            *(_score_cell(getattr(r, dim).score) for dim, _ in _DIM_ORDER),
            "[green]PASS[/green]" if r.passed else "[red]FAIL[/red]",
        )
        console.print(row_table)
        for dim_name, _ in _DIM_ORDER:
            dim = getattr(r, dim_name)
            color = "green" if dim.score >= 4 else ("yellow" if dim.score == 3 else "red")
            console.print(
                f"  [{color}]{dim.score}/5[/{color}] [dim]{dim_name}[/dim]  {dim.rationale}"
            )

    # Variant arms: when --ab is on, run each fixture twice per rep — once
    # defended (full structural chain + critic), once bare (model + tools
    # only). The judge stays blind to which arm produced the output.
    variants: tuple[str, ...] = ("defended", "bare") if ab else ("defended",)

    # Collect every per-run EvalResult, keyed by (fixture, variant).
    runs_by_pair: dict[tuple[str, str], list[Any]] = {}
    # Per-trial (defense ledger, passed, variant, fixture) for correlation.
    defense_trials: list[tuple[Any, bool, str, str]] = []
    for fx in fixtures:
        console.print(f"\n[bold blue]▶ {fx.name}[/bold blue]")
        agent_desc = (
            resolved_provider
            if resolved_provider == "claude"
            else f"{resolved_provider}/{resolved_model}"
        )
        for variant in variants:
            for run_idx in range(n_runs):
                pieces: list[str] = []
                if ab:
                    pieces.append(f"[{variant}]")
                if n_runs > 1:
                    pieces.append(f"(run {run_idx + 1}/{n_runs})")
                run_label = " " + " ".join(pieces) if pieces else ""
                with console.status(f"[dim]running agent ({agent_desc}){run_label}...[/dim]"):
                    try:
                        outcome = runner.run_fixture(
                            fx,
                            provider=resolved_provider,
                            model=resolved_model,
                            agent_timeout=agent_timeout,
                            variant=variant,
                        )
                    except Exception as exc:
                        console.print(f"  [red]run failed{run_label}:[/red] {exc}")
                        continue

                exit_icon = (
                    "[green]✓[/green]" if outcome.agent_exit_code == 0 else "[yellow]![/yellow]"
                )
                test_icon = "[green]✓[/green]" if outcome.test_exit_code == 0 else "[red]✗[/red]"
                console.print(
                    f"  agent {exit_icon} (exit {outcome.agent_exit_code})  "
                    f"tests {test_icon} (exit {outcome.test_exit_code}){run_label}"
                )

                with console.status(f"[dim]scoring{run_label}...[/dim]"):
                    try:
                        result = judge_mod.judge(
                            adapter=judge_adapter,
                            model=resolved_judge_model,
                            fixture_name=fx.name,
                            task_text=fx.task_text,
                            eval_md=fx.eval_md,
                            transcript=outcome.transcript,
                            git_diff=outcome.git_diff,
                            test_output=outcome.test_output,
                        )
                        runs_by_pair.setdefault((fx.name, variant), []).append(result)
                    except Exception as exc:
                        console.print(f"  [red]judge failed{run_label}:[/red] {exc}")
                        continue

                # Capture the defense ledger from this trial's transcript so we
                # can correlate which defenses fired with PASS/FAIL outcomes.
                # Bare-variant trials never emit a ledger (no structural chain
                # → nothing to log) but we still record (None, passed) so the
                # report distinguishes "defense was silent" from "no data."
                ledger = parse_ledger_text(outcome.transcript)
                defense_trials.append((ledger, result.passed, variant, fx.name))

                _print_per_fixture(result)

    if not runs_by_pair:
        return

    # End-of-batch rollup. With n_runs>1, each cell is "<median>/5 (min..max)".
    # With --ab, the fixture column carries the variant label too.
    def _cell_for_runs(runs: list[Any], dim_name: str) -> str:
        scores = sorted(getattr(r, dim_name).score for r in runs)
        # statistics.median averages the two middle values for even-N lists,
        # which is what we want — picking scores[N//2] silently biased high
        # on N=2 (e.g. median of [1, 5] became 5 instead of 3).
        import statistics

        median = statistics.median(scores)
        median_round = round(median)
        color = "green" if median_round >= 4 else ("yellow" if median_round == 3 else "red")
        # Render as int when whole, single decimal otherwise.
        median_str = f"{median:g}" if median != int(median) else str(int(median))
        if len(scores) == 1:
            return f"[{color}]{median_str}/5[/{color}]"
        return f"[{color}]{median_str}/5[/{color}] [dim]({scores[0]}..{scores[-1]})[/dim]"

    console.print()
    title_pieces = ["Eval Results"]
    if n_runs > 1:
        title_pieces.append(f"{n_runs} runs each")
    if ab:
        title_pieces.append("A/B: defended vs bare")
    title = " — ".join(title_pieces)
    table = Table(show_header=True, header_style="bold", title=title)
    table.add_column("Fixture / variant", no_wrap=True)
    for _, col in _DIM_ORDER:
        table.add_column(col, justify="center")
    table.add_column("Pass rate", justify="center")
    for (fx_name, variant), runs in runs_by_pair.items():
        n_passed = sum(1 for r in runs if r.passed)
        pass_color = "green" if n_passed == len(runs) else ("yellow" if n_passed > 0 else "red")
        label = f"{fx_name} [dim]({variant})[/dim]" if ab else fx_name
        table.add_row(
            label,
            *(_cell_for_runs(runs, dim) for dim, _ in _DIM_ORDER),
            f"[{pass_color}]{n_passed}/{len(runs)}[/{pass_color}]",
        )
    console.print(table)

    # Defense correlation report: which defenses fired correlate with PASS or
    # FAIL? Only meaningful when we have multiple defended trials — bare
    # trials always have empty ledgers (no structural chain), so a pure-bare
    # run produces no signal here.
    defended_trials = [
        (ledger, passed) for ledger, passed, variant, _ in defense_trials if variant == "defended"
    ]
    if len(defended_trials) >= 3:
        stats = correlate_defenses(defended_trials)
        console.print()
        defense_table = Table(
            show_header=True,
            header_style="bold",
            title=f"Defense correlation ({len(defended_trials)} defended trials)",
        )
        defense_table.add_column("Defense", no_wrap=True)
        defense_table.add_column("block→pass", justify="center")
        defense_table.add_column("block→fail", justify="center")
        defense_table.add_column("silent→pass", justify="center")
        defense_table.add_column("silent→fail", justify="center")
        defense_table.add_column("Verdict", justify="left")
        verdict_color = {
            "helps": "green",
            "neutral": "yellow",
            "hurts": "red",
            "n/a": "dim",
            "n/a (small N)": "dim",
        }
        for s in stats:
            color = verdict_color.get(s.verdict(), "white")
            defense_table.add_row(
                s.name,
                str(s.block_pass),
                str(s.block_fail),
                str(s.silent_pass),
                str(s.silent_fail),
                f"[{color}]{s.verdict()}[/{color}]",
            )
        console.print(defense_table)
        console.print(
            "[dim]Read: a defense that 'hurts' fires when a trial fails more "
            "often than when it passes. Manually consider disabling such "
            "defenses; this report is diagnostic only.[/dim]"
        )


if __name__ == "__main__":
    app()
