from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from harness.cli.builtin_tools import BuiltinToolProvider
from harness.core import (
    Agent,
    AgentDoneEvent,
    AgentEventWrapper,
    AgentRole,
    AgentStartedEvent,
    ApprovalPolicy,
    AutoApprove,
    CompleteWorkItemTool,
    ConsequencePredictor,
    ContextBudget,
    CreateWorkItemTool,
    Done,
    ErrorEvent,
    FailoverPolicy,
    ListWorkItemsTool,
    MultiAgentOrchestrator,
    RepairOrchestrator,
    TextDelta,
    ToolCallEvent,
    ToolRegistry,
    ToolResultEvent,
    WorkItemClaimedEvent,
    WorkItemCompletedEvent,
    WorkItemCreatedEvent,
    WorkItemJudge,
    WorkItemOrphanedEvent,
    WorkItemRejectedEvent,
    WorkItemVerifiedEvent,
)
from harness.storage.memory import InMemoryStorage
from harness.storage.sqlite import SQLiteStorage


def role_color(role: str) -> str:
    if role == "planner":
        return "magenta"
    if role.startswith("worker"):
        return "cyan"
    if role == "reporter":
        return "green"
    return "white"


class LabRenderer:
    def __init__(self, con: Console, *, args_preview: Any, truncate: Any) -> None:
        self._console = con
        self._text_bufs: dict[str, str] = {}
        self._args_preview = args_preview
        self._truncate = truncate

    def render(self, event: object) -> None:
        if isinstance(event, AgentStartedEvent):
            color = role_color(event.role)
            self._console.print(f"[{color}]▶ {event.role}[/{color}]")
        elif isinstance(event, AgentDoneEvent):
            color = role_color(event.role)
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
        elif isinstance(event, AgentEventWrapper):
            self._render_wrapped(event.role, event.event)

    def _render_wrapped(self, role: str, event: object) -> None:
        color = role_color(role)
        prefix = f"  [{color}][{role}][/{color}]"
        if isinstance(event, TextDelta):
            self._text_bufs.setdefault(role, "")
            self._text_bufs[role] += event.text
        elif isinstance(event, Done):
            buf = self._text_bufs.pop(role, "").strip()
            if buf:
                for line in buf.splitlines():
                    if line.strip():
                        self._console.print(f"{prefix} {line}")
            if event.usage:
                usage = event.usage
                self._console.print(
                    f"{prefix} [dim]tokens: {usage.prompt_tokens:,}in / "
                    f"{usage.completion_tokens:,}out[/dim]"
                )
        elif isinstance(event, ToolCallEvent):
            self._console.print(
                f"{prefix} [dim]→ [bold]{event.call.name}[/bold]"
                f"({self._args_preview(event.call.arguments)})[/dim]"
            )
        elif isinstance(event, ToolResultEvent):
            marker = "[red]✗[/red]" if event.result.is_error else "[green]✓[/green]"
            preview = self._truncate(event.result.content, 120)
            self._console.print(f"{prefix} [dim]{marker} {event.result.name}: {preview}[/dim]")
        elif isinstance(event, ErrorEvent):
            self._console.print(f"{prefix} [red]error ({event.kind}):[/red] {event.error}")


def lab_run_command(
    *,
    prompt: str,
    provider: str | None,
    model: str | None,
    workers: int,
    yes: bool,
    cwd: Path | None,
    config_path: Path | None,
    no_judge: bool,
    db: Path | None,
    max_context_tokens: int | None,
    max_steps: int,
    planner_model: str | None,
    worker_model: str | None,
    reporter_model: str | None,
    console: Console,
    load_cli_config: Any,
    build_adapter: Any,
    run_async: Any,
    args_preview: Any,
    truncate: Any,
    orchestrator_cls: type[MultiAgentOrchestrator] = MultiAgentOrchestrator,
    work_item_judge_cls: type[WorkItemJudge] = WorkItemJudge,
) -> None:
    async def run() -> None:
        cfg = load_cli_config(config_path)
        working_dir = (cwd or Path.cwd()).resolve()
        resolved_provider = provider or cfg.default_provider or "ollama"
        resolved_model = model or cfg.default_model or "llama3.2"

        if db is not None:
            storage: InMemoryStorage = SQLiteStorage(path=db)  # type: ignore[assignment]
        else:
            storage = InMemoryStorage()
        worker_budget = (
            ContextBudget(max_tokens=max_context_tokens) if max_context_tokens is not None else None
        )
        renderer = LabRenderer(console, args_preview=args_preview, truncate=truncate)

        def agent_factory(role: AgentRole) -> Agent:
            job_id = role.job_id or "_job_"
            item_id = role.item_id or "_item_"
            provider = BuiltinToolProvider()

            if role.name == "planner":
                tools = ToolRegistry()
                tools.register(CreateWorkItemTool(storage, parent_id=job_id, cwd=working_dir))
                tools.register(ListWorkItemsTool(storage, job_id))
            elif role.name.startswith("worker"):
                tools = provider.build_registry(
                    cwd=working_dir,
                    include={
                        "read_file",
                        "list_dir",
                        "glob",
                        "write_file",
                        "edit_file",
                        "shell",
                        "web_search",
                        "fetch_url",
                    },
                )
                tools.register(ListWorkItemsTool(storage, job_id))
                tools.register(CompleteWorkItemTool(storage, item_id))
            else:
                tools = provider.build_registry(
                    cwd=working_dir,
                    include={"read_file", "list_dir", "glob"},
                )
                tools.register(ListWorkItemsTool(storage, job_id))

            adapters = {
                resolved_provider: build_adapter(resolved_provider, base_url=None, config=cfg)
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

        judge_adapter = build_adapter(resolved_provider, base_url=None, config=cfg)
        work_item_judge: WorkItemJudge | None = None
        if not no_judge:
            work_item_judge = work_item_judge_cls(
                adapter=judge_adapter,
                model=resolved_planner_model,
            )

        orchestrator = orchestrator_cls(
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

    run_async(run())


def lab_status_command(*, job_id: str, db: Path, console: Console, run_async: Any) -> None:
    async def run() -> None:
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

    run_async(run())


def lab_list_command(*, db: Path, console: Console, run_async: Any) -> None:
    async def run() -> None:
        storage = SQLiteStorage(path=db)
        try:
            all_tasks = await storage.list_tasks(parent_id=None)
            jobs = [task for task in all_tasks if task.parent_id is None]
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
                done_count = sum(1 for task in items if task.status == "done")
                total_count = len(items)
                ts = job.created_at.strftime("%Y-%m-%d %H:%M")
                console.print(
                    f"[{color}]{job.status:12}[/{color}]  {job.id[:16]}  "
                    f"[dim]{ts}[/dim]  {done_count}/{total_count} items  {job.title[:60]}"
                )
        finally:
            await storage.close()

    run_async(run())


def lab_resume_command(
    *,
    job_id: str,
    provider: str | None,
    model: str | None,
    workers: int,
    db: Path,
    config_path: Path | None,
    no_judge: bool,
    max_steps: int,
    planner_model: str | None,
    worker_model: str | None,
    console: Console,
    load_cli_config: Any,
    build_adapter: Any,
    run_async: Any,
    args_preview: Any,
    truncate: Any,
    orchestrator_cls: type[MultiAgentOrchestrator] = MultiAgentOrchestrator,
    work_item_judge_cls: type[WorkItemJudge] = WorkItemJudge,
) -> None:
    async def run() -> None:
        cfg = load_cli_config(config_path)
        resolved_provider = provider or cfg.default_provider or "ollama"
        resolved_model = model or cfg.default_model or "llama3.2"
        resolved_worker_model = worker_model or resolved_model
        resolved_planner_model = planner_model or resolved_model

        storage = SQLiteStorage(path=db)
        renderer = LabRenderer(console, args_preview=args_preview, truncate=truncate)

        root = await storage.get_task(job_id)
        if root is None:
            console.print(f"[red]Job {job_id!r} not found in {db}[/red]")
            raise typer.Exit(1)

        working_dir = root.cwd
        worker_budget: ContextBudget | None = None

        def agent_factory(role: AgentRole) -> Agent:
            job = role.job_id or "_job_"
            item = role.item_id or "_item_"
            provider = BuiltinToolProvider()

            if role.name.startswith("worker"):
                tools = provider.build_registry(
                    cwd=working_dir,
                    include={
                        "read_file",
                        "list_dir",
                        "glob",
                        "write_file",
                        "edit_file",
                        "shell",
                        "web_search",
                        "fetch_url",
                    },
                )
                tools.register(ListWorkItemsTool(storage, job))
                tools.register(CompleteWorkItemTool(storage, item))
            else:
                tools = provider.build_registry(
                    cwd=working_dir,
                    include={"read_file", "list_dir", "glob"},
                )
                tools.register(ListWorkItemsTool(storage, job))

            adapters = {
                resolved_provider: build_adapter(resolved_provider, base_url=None, config=cfg)
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

        judge_adapter = build_adapter(resolved_provider, base_url=None, config=cfg)
        work_item_judge: WorkItemJudge | None = None
        if not no_judge:
            work_item_judge = work_item_judge_cls(
                adapter=judge_adapter,
                model=resolved_planner_model,
            )

        orchestrator = orchestrator_cls(
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

    run_async(run())
