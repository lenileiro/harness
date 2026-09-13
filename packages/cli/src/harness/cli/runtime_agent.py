from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import AsyncExitStack, aclosing, asynccontextmanager
from pathlib import Path
from typing import Any, cast

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
    PublicSourceEvidenceVerifier,
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
from harness.core.clarification import QuestionStore
from harness.core.memory import MemoryScope
from harness.core.skills import SkillLibrary, default_skill_paths
from harness.core.tools import Tool
from harness.storage.memory import InMemoryStorage
from harness.tools.mcp import MCPToolset

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
        inherit_from: Callable[[], Agent] | None = None,
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
        self._inherit_from = inherit_from
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

        inherited_tools: ToolRegistry | None = None
        approval_policy = self._approval_policy
        approval_handler = self._approval_handler
        if self._inherit_from is not None:
            try:
                parent = self._inherit_from()
            except RuntimeError as exc:
                return ToolResult(
                    tool_call_id=call.id, name=self.name, content=str(exc), is_error=True
                )
            inherited_tools = parent.tools
            approval_policy = parent.approval_policy
            approval_handler = parent.approval_handler
        elif self._config.execution is not None:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="Delegation with managed execution requires an active parent tool context.",
                is_error=True,
            )

        store = InMemoryStorage()
        cleanup = AsyncExitStack()

        def agent_factory(role: AgentRole) -> Agent:
            from harness.cli.runtime_helpers import AUTONOMOUS_CONTEXT_POLICY

            job_id = role.job_id or "_job_"
            item_id = role.item_id or "_item_"
            sub_tools = ToolRegistry()
            sub_tools.register(NotifyTool(role=role.name, task_id=job_id, activity_store=store))
            sub_tools.register(
                CheckMessagesTool(role=role.name, task_id=job_id, activity_store=store)
            )

            # Borrow the effective parent tools, including backend replacements,
            # wrappers, and exposure removals. Children never own these contexts.
            # A standalone local tool retains its supplied builder for compatibility.
            built = inherited_tools if inherited_tools is not None else self._build_tools(self._cwd)
            discovery_names = ("read_file", "list_dir", "glob", "web_search", "fetch_url")
            for name in discovery_names:
                if (
                    built.has(name)
                    and getattr(built.get(name), "effect_scope", None) == "read_only"
                ):
                    sub_tools.register(built.get(name))

            if role.name == "planner":
                sub_tools.register(ListWorkItemsTool(store, job_id))
                sub_tools.register(CreateWorkItemTool(store, parent_id=job_id, cwd=self._cwd))
            elif role.name.startswith("worker"):
                if built.has("shell"):
                    sub_tools.register(built.get("shell"))
                sub_tools.register(ListWorkItemsTool(store, job_id))
                sub_tools.register(CompleteWorkItemTool(store, item_id))
            else:
                sub_tools.register(ListWorkItemsTool(store, job_id))

            unavailable = [name for name in discovery_names if not sub_tools.has(name)]
            discovery_context = (
                "Use only the tools supplied for this role and the assigned workspace. "
                "Inspect relevant repository instructions and local evidence before researching "
                "remaining gaps. Tool errors are not evidence; use a supported alternative "
                "or report the exact unresolved dependency."
            )
            if unavailable:
                discovery_context += (
                    " These discovery tools are unavailable in the inherited tool configuration: "
                    + ", ".join(unavailable)
                    + ". Only tools declared read-only qualify for discovery. "
                    "Do not bypass their absence with an unapproved capability."
                )
            adapter = self._build_adapter(self._provider, base_url=None, config=self._config)
            adapter = _align_adapter_cwd(self._provider, adapter, self._cwd)
            agent = Agent(
                adapters={self._provider: adapter},
                tools=sub_tools,
                storage=store,
                failover=FailoverPolicy(chain=[self._provider]),
                approval_policy=approval_policy,
                approval_handler=approval_handler,
                default_model=role.model or self._model,
                default_cwd=str(self._cwd),
                system_prompt="\n\n".join(
                    (AUTONOMOUS_CONTEXT_POLICY, role.system_prompt, discovery_context)
                ),
            )
            cleanup.push_async_callback(agent.aclose)
            return agent

        planner_role = AgentRole(
            name="planner",
            system_prompt=(
                "You are a Planner. Read the goal carefully and decompose it into "
                "independent work items — one per distinct area the goal explicitly "
                "asks about. Inspect enough local context and authoritative sources to "
                "make the assignments concrete; keep discovery scoped to the goal. "
                "Create as few items as needed. Stop immediately after "
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
        async with cleanup, aclosing(orchestrator.run(goal)) as events:
            async for event in events:
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


def _align_adapter_cwd(provider: str, adapter: Adapter, cwd: Path) -> Adapter:
    if provider in {"codex", "claude"} and hasattr(adapter, "cwd"):
        adapter_with_cwd = cast(Any, adapter)
        adapter_with_cwd.cwd = cwd.resolve()
    return adapter


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
    pause_on_approval: bool = False,
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
    verify_command: str | None = None,
    phases_enabled: bool = False,
    loop_detector: Any | None = None,
    contracts: Any | None = None,
    tips_provider: Any | None = None,
    resume: Any | None = None,
    memory_tools_enabled: bool = True,
    auxiliary_tools_enabled: bool = True,
    project_context_enabled: bool = True,
    skip_builtin_verify_before_done: bool = False,
) -> Agent:
    from harness.cli.runtime_helpers import AUTONOMOUS_CONTEXT_POLICY

    system_prompt = "\n\n".join(part for part in (AUTONOMOUS_CONTEXT_POLICY, system_prompt) if part)
    if not chain:
        raise typer.BadParameter("provider chain is empty")
    if inbox and approval_store is None:
        raise typer.BadParameter("--inbox requires an approval_store (passed by _build_agent)")
    if config.delegation_enabled and not auxiliary_tools_enabled:
        raise typer.BadParameter(
            "Standalone delegation requires a local agent; API children use the server's scoped delegation service"
        )
    if config.computer.enabled and not auxiliary_tools_enabled:
        raise typer.BadParameter(
            "Computer control is available only to an explicitly configured local agent"
        )

    adapters: dict[str, Adapter] = {}
    for index, provider in enumerate(chain):
        provider_base_url = base_url if index == 0 else None
        adapter = build_adapter(provider, base_url=provider_base_url, config=config)
        adapters[provider] = _align_adapter_cwd(provider, adapter, cwd)

    from harness.core.paths import read_regular_file, user_home

    persona_path = user_home() / "SOUL.md"
    if project_context_enabled and persona_path.is_file():
        persona = read_regular_file(persona_path, max_bytes=64_000).decode("utf-8")
        system_prompt = "\n\n".join(part for part in (system_prompt, persona) if part)
    project_ctx = load_project_context(cwd) if project_context_enabled else ""
    if project_ctx and system_prompt:
        system_prompt = f"{system_prompt}\n\n{project_ctx}"
    elif project_ctx:
        system_prompt = project_ctx

    tools = build_tools(cwd)
    execution_allowed = {tool.name for tool in tools.all()}
    portal_names = config.portal.tool_names() if auxiliary_tools_enabled else set()
    for name in portal_names:
        if tools.has(name):
            tools.unregister(name)
    if "shell" in execution_allowed:
        execution_allowed.update({"process", "verify_work"})
    if config.execution is not None:
        if not auxiliary_tools_enabled:
            raise typer.BadParameter(
                "Managed execution requires a local trusted agent; remote tool exposure must be configured separately"
            )
        for name in ("read_file", "write_file", "edit_file", "list_dir", "glob", "shell"):
            if tools.has(name):
                tools.unregister(name)
    skill_library = None
    if config.skills_enabled and project_context_enabled:
        skill_roots = [(cwd / Path(p).expanduser()).resolve() for p in config.skill_paths]
        skill_library = SkillLibrary.load(skill_roots + default_skill_paths(cwd.resolve()))
    mcp_servers = tuple(server for server in config.mcp_servers if server.enabled)
    if (
        mcp_servers
        and "codex" in chain
        and config.provider("codex").get("mode", "exec") != "app-server"
    ):
        raise typer.BadParameter(
            "Codex uses native tools and cannot dispatch configured Harness MCP tools. Choose an API provider."
        )

    managed_active = False

    @asynccontextmanager
    async def managed_tools() -> AsyncIterator[Sequence[Tool]]:
        nonlocal managed_active

        from harness.tools.browser import BrowserToolset
        from harness.tools.execution import ExecutionToolset
        from harness.tools.media import MediaConfig, MediaToolset

        async with AsyncExitStack() as stack:
            agent.session_tool_factory = None
            managed: list[Tool] = []
            if config.execution is not None:
                execution = await stack.enter_async_context(
                    ExecutionToolset(config.execution, cwd=cwd, verify_command=verify_command)
                )
                managed.extend(tool for tool in execution.tools if tool.name in execution_allowed)
            if mcp_servers:
                mcp = await stack.enter_async_context(MCPToolset(mcp_servers, cwd=cwd))
                managed.extend(mcp.tools)
            if config.browser is not None and "browser" not in portal_names:
                browser = await stack.enter_async_context(BrowserToolset(config.browser, cwd=cwd))
                managed.extend(browser.tools)
            if config.computer.enabled:
                from harness.tools.computer import ComputerToolset

                computer = await stack.enter_async_context(
                    ComputerToolset(config.computer, cwd=cwd)
                )
                managed.extend(computer.tools)
            if config.media.enabled:
                effective_media = config.media.model_copy(
                    update={
                        key: None
                        for name, key in [
                            ("image_generate", "image_model"),
                            ("speech_generate", "speech_model"),
                            ("audio_transcribe", "transcription_model"),
                        ]
                        if name in portal_names
                    }
                )
                media = await stack.enter_async_context(MediaToolset(effective_media, cwd=cwd))
                managed.extend(tool for tool in media.tools if tool.name not in portal_names)
            if portal_names:
                from harness.cli.account_auth import AccountAuth, OAuthAccountConfig
                from harness.cli.portal_tools import PortalToolset

                settings = config.provider(config.portal.provider)
                account = AccountAuth(
                    config.portal.provider,
                    OAuthAccountConfig.model_validate(settings.get("oauth")),
                    resource=settings.get("base_url", ""),
                )
                portal = await stack.enter_async_context(
                    PortalToolset(config.portal, account, cwd=cwd)
                )
                managed.extend(portal.tools)
            if config.delegation_enabled:
                import hashlib
                from dataclasses import replace

                from harness.cli.gateway_tool_boundary import install_gateway_tool_boundary
                from harness.server.delegation import LocalDelegationToolset
                from harness.tools.computer import ComputerConfig

                child_config = replace(
                    config,
                    execution=None,
                    browser=None,
                    computer=ComputerConfig(),
                    media=MediaConfig(),
                    mcp_servers=(),
                    delegation_enabled=False,
                    skills_enabled=False,
                )
                exposed = {
                    "read_file",
                    "write_file",
                    "edit_file",
                    "list_dir",
                    "glob",
                    "web_search",
                    "fetch_url",
                    "recall_memory",
                    "search_sessions",
                    "conversation_window",
                }
                if config.clarification_enabled:
                    exposed.add("clarify")

                def child_builder(context):
                    child = build_agent(
                        chain=chain,
                        base_url=base_url,
                        model=context.model or model,
                        storage=context.storage,
                        cwd=context.workspace,
                        config=child_config,
                        yes=False,
                        build_adapter=build_adapter,
                        build_tools=lambda path: _child_tools(path),
                        build_search_fn=build_search_fn,
                        console=console,
                        inbox=True,
                        pause_on_approval=True,
                        approval_store=context.storage,
                        activity_store=context.storage,
                        memory_store=context.storage,
                        auxiliary_tools_enabled=False,
                        project_context_enabled=False,
                        skip_builtin_verify_before_done=True,
                    )
                    install_gateway_tool_boundary(child, context.workspace)
                    return child

                def _child_tools(path):
                    child_tools = build_tools(path)
                    for tool in child_tools.all():
                        if tool.name not in exposed:
                            child_tools.unregister(tool.name)
                    return child_tools

                children = {}

                async def bind_children(session):
                    if session.id not in children:
                        key = hashlib.sha256(session.id.encode()).hexdigest()[:24]
                        children[session.id] = await stack.enter_async_context(
                            LocalDelegationToolset(
                                cwd / ".harness/delegation" / f"{key}.db",
                                cwd,
                                child_builder,
                                limits=config.delegation_limits,
                                exposed_tools=sorted(exposed),
                            )
                        )
                    bound = await children[session.id].bind(session)
                    return [
                        tool
                        for tool in bound
                        if "write_file" in execution_allowed
                        or "shell" in execution_allowed
                        or getattr(tool, "effect_scope", None) in {"read_only", "session_ephemeral"}
                    ]

                agent.session_tool_factory = bind_children
            if config.honcho.enabled and auxiliary_tools_enabled:
                from harness.cli.honcho_tools import HonchoToolset

                honcho = await stack.enter_async_context(HonchoToolset(config.honcho))
                previous_factory = agent.session_tool_factory

                async def bind_honcho(session):
                    previous = await previous_factory(session) if previous_factory else []
                    selected = [
                        tool
                        for tool in honcho.bind(session)
                        if "write_file" in execution_allowed
                        or "shell" in execution_allowed
                        or getattr(tool, "effect_scope", None) in {"read_only", "session_ephemeral"}
                    ]
                    return [*previous, *selected]

                agent.session_tool_factory = bind_honcho
            if config.a2a.enabled and auxiliary_tools_enabled:
                from harness.cli.a2a_tools import A2AToolset

                peers = await stack.enter_async_context(A2AToolset(config.a2a))
                previous_a2a_factory = agent.session_tool_factory

                async def bind_peers(session):
                    previous = await previous_a2a_factory(session) if previous_a2a_factory else []
                    selected = [
                        tool
                        for tool in peers.bind(session)
                        if "write_file" in execution_allowed
                        or "shell" in execution_allowed
                        or getattr(tool, "effect_scope", None) in {"read_only", "session_ephemeral"}
                    ]
                    return [*previous, *selected]

                agent.session_tool_factory = bind_peers
            if config.homeassistant.enabled and auxiliary_tools_enabled:
                from harness.tools.homeassistant import HomeAssistantToolset

                homeassistant = await stack.enter_async_context(
                    HomeAssistantToolset(config.homeassistant)
                )
                managed.extend(homeassistant.tools)
            if "write_file" not in execution_allowed and "shell" not in execution_allowed:
                managed = [
                    tool
                    for tool in managed
                    if getattr(tool, "effect_scope", None) in {"read_only", "session_ephemeral"}
                ]
            managed_active = True
            try:
                yield managed
            finally:
                managed_active = False

    if auxiliary_tools_enabled and config.execution is None:
        tools.register(VerifyWorkTool(cwd=cwd, default_command=verify_command))
        if phases_enabled:
            tools.register(PhaseTool(activity_store=activity_store))
    primary_adapter = adapters[chain[0]]
    if auxiliary_tools_enabled:
        tools.register(
            RequestCritiqueTool(
                adapter=primary_adapter,
                model=model,
                search_fn=build_search_fn(),
            )
        )

    if profile == "strict":
        verify_before_done = VerifyBeforeDoneVerifier(
            default_verify_command_available=bool(verify_command)
        )
        structural = ChainedVerifier(
            FileScopeVerifier(),
            PublicSourceEvidenceVerifier(),
            ResearchPromotionFlowVerifier(),
            MinimalFixVerifier(),
            TestsBeforeEditVerifier(),
            verify_before_done,
            DiagnosisAlignmentVerifier(),
            MisdirectedSuggestionVerifier(),
            NegativeConstraintVerifier(),
            BugfixCommentRewriteVerifier(),
            PromptSurfaceRevertVerifier(),
            PhaseGateVerifier(),
            fail_fast=False,
        )
        verifier = (
            ChainedVerifier(structural, verifier, fail_fast=False)
            if verifier is not None
            else structural
        )
    elif profile == "diagnostic":
        verify_before_done = VerifyBeforeDoneVerifier(
            default_verify_command_available=bool(verify_command)
        )
        structural = ChainedVerifier(
            FileScopeVerifier(),
            PublicSourceEvidenceVerifier(),
            ResearchPromotionFlowVerifier(),
            verify_before_done,
            DiagnosisAlignmentVerifier(),
            MisdirectedSuggestionVerifier(),
            NegativeConstraintVerifier(),
            BugfixCommentRewriteVerifier(),
            PromptSurfaceRevertVerifier(),
            fail_fast=False,
        )
        verifier = (
            ChainedVerifier(structural, verifier, fail_fast=False)
            if verifier is not None
            else structural
        )
    elif profile == "minimal":
        verify_only = (
            ChainedVerifier(
                PublicSourceEvidenceVerifier(),
                ResearchPromotionFlowVerifier(),
            )
            if skip_builtin_verify_before_done
            else ChainedVerifier(
                PublicSourceEvidenceVerifier(),
                ResearchPromotionFlowVerifier(),
                VerifyBeforeDoneVerifier(default_verify_command_available=bool(verify_command)),
            )
        )
        verifier = (
            ChainedVerifier(verify_only, verifier, fail_fast=False)
            if verifier is not None
            else verify_only
        )

    approval_policy = ApprovalPolicy(default="prompt", per_tool=dict(config.approval))
    if yes:
        approval_handler: ApprovalHandler = AutoApprove()
    elif inbox:
        assert approval_store is not None
        approval_handler = InboxApprovalHandler(approval_store=approval_store)
    else:
        approval_handler = RichApprovalHandler(console=console, session_overrides=session_overrides)

    if auxiliary_tools_enabled and not config.delegation_enabled:

        def spawn_parent() -> Agent:
            if config.execution is not None and not managed_active:
                raise RuntimeError(
                    "Delegation with managed execution requires an active parent tool context."
                )
            return agent

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
                inherit_from=spawn_parent,
            )
        )

    multi = len(chain) > 1
    agent = Agent(
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
        pause_on_approval=pause_on_approval,
        question_store_factory=(lambda: QuestionStore(getattr(storage, "path", ":memory:")))
        if config.clarification_enabled
        else None,
        verifier=verifier,
        critic=critic,
        budget=budget,
        default_model=model,
        default_cwd=str(cwd),
        memory_store=memory_store,
        memory_scope=MemoryScope(workspace=str(cwd)),
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
        memory_tools_enabled=memory_tools_enabled,
        skill_library=skill_library,
        toolset_factory=managed_tools
        if mcp_servers
        or config.execution is not None
        or config.browser is not None
        or config.media.enabled
        or config.computer.enabled
        or config.delegation_enabled
        or portal_names
        or (
            (config.honcho.enabled or config.homeassistant.enabled or config.a2a.enabled)
            and auxiliary_tools_enabled
        )
        else None,
        provider_models={
            name: config.provider(name)["model"]
            for name in chain
            if isinstance(config.provider(name).get("model"), str)
        },
    )
    return agent


__all__ = [
    "_SPAWN_SCHEMA",
    "SpawnAgentsTool",
    "_align_adapter_cwd",
    "build_agent",
    "load_project_context",
]
