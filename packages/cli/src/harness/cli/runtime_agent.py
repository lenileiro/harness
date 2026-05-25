from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from harness.cli.approval import RichApprovalHandler
from harness.cli.config import HarnessConfig
from harness.core import (
    Adapter,
    Agent,
    AgentRole,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalPolicy,
    ApprovalStore,
    AutoApprove,
    BugfixCommentRewriteVerifier,
    ChainedVerifier,
    CheckMessagesTool,
    CompleteWorkItemTool,
    ConsequencePredictor,
    ContextBudget,
    CreateWorkItemTool,
    Critic,
    DiagnosisAlignmentVerifier,
    FailoverPolicy,
    FileScopeVerifier,
    InboxApprovalHandler,
    ListWorkItemsTool,
    MinimalFixVerifier,
    MisdirectedSuggestionVerifier,
    MultiAgentOrchestrator,
    NegativeConstraintVerifier,
    NotifyTool,
    PhaseGateVerifier,
    PhaseTool,
    Planner,
    PromptSurfaceRevertVerifier,
    RepairOrchestrator,
    RequestCritiqueTool,
    ResearchPromotionFlowVerifier,
    Storage,
    TestsBeforeEditVerifier,
    ToolCall,
    ToolRegistry,
    ToolResult,
    Verifier,
    VerifyBeforeDoneVerifier,
    VerifyWorkTool,
)
from harness.storage.memory import InMemoryStorage

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
        build_adapter: Any,
        build_tools: Any,
        build_search_fn: Any,
        max_workers: int = 3,
        approval_policy: ApprovalPolicy | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        self._provider = provider
        self._model = model
        self._cwd = cwd
        self._config = config
        self._build_adapter = build_adapter
        self._build_tools = build_tools
        self._build_search_fn = build_search_fn
        self._max_workers = max_workers
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
            sub_tools.register(NotifyTool(role=role.name, task_id=job_id, activity_store=store))
            sub_tools.register(
                CheckMessagesTool(role=role.name, task_id=job_id, activity_store=store)
            )

            if role.name == "planner":
                sub_tools.register(ListWorkItemsTool(store, job_id))
                sub_tools.register(CreateWorkItemTool(store, parent_id=job_id, cwd=self._cwd))
                sub_tools.register(self._build_tools(self._cwd).get("list_dir"))  # type: ignore[arg-type]
            elif role.name.startswith("worker"):
                built = self._build_tools(self._cwd)
                for name in ("read_file", "list_dir", "glob", "shell"):
                    sub_tools.register(built.get(name))  # type: ignore[arg-type]
                sub_tools.register(ListWorkItemsTool(store, job_id))
                sub_tools.register(CompleteWorkItemTool(store, item_id))
            else:
                built = self._build_tools(self._cwd)
                for name in ("read_file", "list_dir", "glob"):
                    sub_tools.register(built.get(name))  # type: ignore[arg-type]
                sub_tools.register(ListWorkItemsTool(store, job_id))

            adapter = self._build_adapter(self._provider, base_url=None, config=self._config)
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
            from harness.core import AgentEventWrapper, TextDelta

            if (
                isinstance(event, AgentEventWrapper)
                and event.role == "reporter"
                and isinstance(event.event, TextDelta)
            ):
                reporter_text.append(event.event.text)

        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content="".join(reporter_text).strip() or "No output from agents.",
        )


def load_project_context(cwd: Path) -> str:
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


def build_agent(
    *,
    chain: list[str],
    base_url: str | None,
    model: str,
    storage: Storage,
    cwd: Path,
    config: HarnessConfig,
    yes: bool,
    build_adapter: Any,
    build_tools: Any,
    build_search_fn: Any,
    console: Any,
    inbox: bool = False,
    activity_store: Any = None,
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
) -> Agent:
    if not chain:
        raise typer.BadParameter("provider chain is empty")
    if inbox and approval_store is None:
        raise typer.BadParameter("--inbox requires an approval_store (passed by _build_agent)")

    adapters: dict[str, Adapter] = {}
    for index, provider in enumerate(chain):
        provider_base_url = base_url if index == 0 else None
        adapters[provider] = build_adapter(provider, base_url=provider_base_url, config=config)

    project_ctx = load_project_context(cwd)
    if project_ctx and system_prompt:
        system_prompt = f"{system_prompt}\n\n{project_ctx}"
    elif project_ctx:
        system_prompt = project_ctx

    tools = build_tools(cwd)
    tools.register(VerifyWorkTool(cwd=cwd))
    if phases_enabled:
        tools.register(PhaseTool(activity_store=activity_store))
    primary_adapter = adapters[chain[0]]
    tools.register(
        RequestCritiqueTool(
            adapter=primary_adapter,
            model=model,
            search_fn=build_search_fn(),
        )
    )

    if profile == "strict":
        structural = ChainedVerifier(
            FileScopeVerifier(),
            ResearchPromotionFlowVerifier(),
            MinimalFixVerifier(),
            TestsBeforeEditVerifier(),
            VerifyBeforeDoneVerifier(),
            DiagnosisAlignmentVerifier(),
            MisdirectedSuggestionVerifier(),
            NegativeConstraintVerifier(),
            BugfixCommentRewriteVerifier(),
            PromptSurfaceRevertVerifier(),
            PhaseGateVerifier(),
        )
        verifier = ChainedVerifier(structural, verifier) if verifier is not None else structural
    elif profile == "diagnostic":
        structural = ChainedVerifier(
            FileScopeVerifier(),
            ResearchPromotionFlowVerifier(),
            VerifyBeforeDoneVerifier(),
            DiagnosisAlignmentVerifier(),
            MisdirectedSuggestionVerifier(),
            NegativeConstraintVerifier(),
            BugfixCommentRewriteVerifier(),
            PromptSurfaceRevertVerifier(),
        )
        verifier = ChainedVerifier(structural, verifier) if verifier is not None else structural
    elif profile == "minimal":
        verify_only = ChainedVerifier(ResearchPromotionFlowVerifier(), VerifyBeforeDoneVerifier())
        verifier = ChainedVerifier(verify_only, verifier) if verifier is not None else verify_only

    approval_policy = ApprovalPolicy(default="prompt", per_tool=dict(config.approval))
    if yes:
        approval_handler: ApprovalHandler = AutoApprove()
    elif inbox:
        assert approval_store is not None
        approval_handler = InboxApprovalHandler(approval_store=approval_store)
    else:
        approval_handler = RichApprovalHandler(console=console, session_overrides=session_overrides)

    tools.register(
        SpawnAgentsTool(
            provider=chain[0],
            model=model,
            cwd=cwd,
            config=config,
            build_adapter=build_adapter,
            build_tools=build_tools,
            build_search_fn=build_search_fn,
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
        loop_detector=loop_detector,
        contracts=contracts,
        tips_provider=tips_provider,
        resume=resume,
        memory_tools_enabled=True,
    )


__all__ = ["_SPAWN_SCHEMA", "SpawnAgentsTool", "build_agent", "load_project_context"]
