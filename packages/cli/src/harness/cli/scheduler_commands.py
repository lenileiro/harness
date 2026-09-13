from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from harness.cli.common import _run_async
from harness.cli.config import default_config_path, load_config
from harness.cli.gateway_runtime import _default_gateway_model, _default_gateway_provider
from harness.cli.mission_commands import run_agent_mission
from harness.cli.plugins import load_cli_hook_providers
from harness.cli.scheduled_prompt import prompt_session_id, run_scheduled_prompt
from harness.core import (
    compute_next_run_at,
    create_scheduler_job,
    default_scheduler_root,
    parse_schedule_spec,
    run_scheduler_job,
    run_scheduler_loop,
)
from harness.core.extensions import LifecycleHook
from harness.core.mission_runtime import write_mission_scheduled_run_record
from harness.core.mission_store import MissionStore, default_mission_root
from harness.core.scheduler_models import SchedulerExecutionResult, SchedulerJob
from harness.core.scheduler_runtime import retry_scheduler_deliveries
from harness.core.scheduler_store import SchedulerStore

console = Console()

scheduler_app = typer.Typer(
    name="scheduler",
    help="Manage scheduled prompts, missions and research work.",
    no_args_is_help=True,
)


def _emit_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2))


def _load_store(cwd: Path | None) -> tuple[Path, SchedulerStore]:
    working_dir = (cwd or Path.cwd()).resolve()
    return working_dir, SchedulerStore(root=default_scheduler_root(working_dir))


def _load_hooks(cwd: Path) -> tuple[LifecycleHook, ...]:
    hooks: list[LifecycleHook] = []
    for provider in load_cli_hook_providers(cwd):
        hooks.extend(provider.hooks())
    return tuple(hooks)


def _resolve_schedule(
    *, at: str | None, every: str | None, cron: str | None, timezone: str = "UTC"
):
    try:
        return parse_schedule_spec(at=at, every=every, cron=cron, timezone=timezone)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _utcnow_text() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _execute_agent_mission_job(job: SchedulerJob) -> tuple[str, str, str]:
    payload = job.payload
    cwd = Path(job.cwd)
    config_path = str(payload.get("config_path") or "")
    result = _run_async(
        run_agent_mission(
            mission_id=str(payload.get("mission_id") or ""),
            cwd=cwd,
            provider=str(payload.get("provider") or "") or None,
            model=str(payload.get("model") or "") or None,
            base_url=str(payload.get("base_url") or "") or None,
            max_features=int(payload.get("max_steps", 20)),
            max_worker_steps=int(payload.get("max_worker_steps", 20)),
            timeout_seconds=float(payload.get("timeout_seconds", 300)),
            yes=bool(payload.get("yes", False)),
            inbox=not bool(payload.get("yes", False)),
            config=load_config(Path(config_path) if config_path else None),
        )
    )
    record_dir = write_mission_scheduled_run_record(
        store=MissionStore(root=default_mission_root(cwd)),
        cwd=cwd,
        result=result,
    )
    return result.status, result.stop_reason, str(record_dir)


def _execute_prompt_job(job: SchedulerJob) -> SchedulerExecutionResult:
    return _run_async(run_scheduled_prompt(job))


@scheduler_app.command("add-prompt")
def scheduler_add_prompt_command(
    *,
    prompt: str = typer.Option(..., "--prompt"),
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    at: str | None = typer.Option(None, "--at"),
    every: str | None = typer.Option(None, "--every"),
    cron: str | None = typer.Option(
        None, "--cron", help="Five-field cron expression in the selected timezone."
    ),
    timezone: str = typer.Option(
        "UTC", "--timezone", help="IANA timezone, for example Europe/Tallinn."
    ),
    transport: str | None = typer.Option(
        None,
        "--transport",
        help="Notification channel; omit all target flags for a local approval inbox.",
    ),
    user_id: str | None = typer.Option(
        None, "--user", help="User authorized to approve this job's actions."
    ),
    thread_id: str | None = typer.Option(
        None, "--thread", help="Conversation receiving results and approval requests."
    ),
    max_steps: int = typer.Option(20, "--max-steps", min=1),
    timeout_seconds: float = typer.Option(300, "--timeout", min=0.1),
    config_path: Path | None = typer.Option(None, "--config"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Schedule a real prompt. Mutations queue for approval; no blanket auto-approval."""
    working_dir, store = _load_store(cwd)
    if not working_dir.is_dir():
        raise typer.BadParameter("--cwd must be an existing directory")
    if not prompt.strip():
        raise typer.BadParameter("--prompt must not be empty")
    if not math.isfinite(timeout_seconds):
        raise typer.BadParameter("--timeout must be finite")
    target = (transport, user_id, thread_id)
    if any(item is not None for item in target) and not all(
        item and item.strip() for item in target
    ):
        raise typer.BadParameter(
            "Provide --transport, --user and --thread together, or omit all three"
        )
    cfg = load_config(config_path)
    selected_provider = (provider or cfg.default_provider or _default_gateway_provider()).strip()
    selected_model = (
        model or cfg.default_model or _default_gateway_model(selected_provider)
    ).strip()
    if selected_provider == "codex":
        raise typer.BadParameter(
            "Codex native tools cannot use queued approvals; choose an API provider"
        )
    if not selected_provider or not selected_model:
        raise typer.BadParameter("A provider and model are required")
    job = create_scheduler_job(
        store=store,
        kind="prompt.run",
        cwd=working_dir,
        schedule=_resolve_schedule(at=at, every=every, cron=cron, timezone=timezone),
        title="scheduled-prompt",
        payload={
            "prompt": prompt,
            "provider": selected_provider,
            "model": selected_model,
            "max_steps": max_steps,
            "timeout_seconds": timeout_seconds,
            "transport": transport or "local",
            "user_id": user_id or "cli",
            "thread_id": thread_id or "",
            "local_only": transport is None,
            "config_path": str((config_path or default_config_path()).resolve()),
        },
    )
    store.add_job(job)
    output = {**job.to_dict(), "session_id": prompt_session_id(job.id)}
    if json_output:
        _emit_json(output)
        return
    console.print(f"[green]Scheduled prompt[/green] {job.id}")
    console.print(f"session_id={output['session_id']}")
    console.print(f"next_run_at={job.next_run_at}")
    if transport is None:
        console.print(
            "Actions use the local inbox: harness approvals list; grant an approval, then harness sessions resume <session_id> from this workspace."
        )


@scheduler_app.command("add-mission")
def scheduler_add_mission_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    at: str | None = typer.Option(None, "--at"),
    every: str | None = typer.Option(None, "--every"),
    cron: str | None = typer.Option(None, "--cron"),
    timezone: str = typer.Option("UTC", "--timezone"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    auto_complete: bool | None = typer.Option(None, "--auto-complete/--no-auto-complete"),
    execute: bool = typer.Option(
        False, "--execute", help="Run real Agent workers with independent assertion commands."
    ),
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    base_url: str | None = typer.Option(None, "--base-url"),
    max_worker_steps: int = typer.Option(20, "--max-worker-steps"),
    timeout_seconds: float = typer.Option(300.0, "--timeout"),
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Automatically approve worker tools; otherwise queue approvals in the inbox.",
    ),
    config_path: Path | None = typer.Option(
        None, "--config", help=f"Override config path (default: {default_config_path()})."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir, store = _load_store(cwd)
    cfg = load_config(config_path)
    scheduler = cfg.mission_scheduler
    resolved_max_steps = max_steps if max_steps is not None else (scheduler.max_steps or 20)
    resolved_auto_complete = (
        auto_complete if auto_complete is not None else bool(scheduler.auto_complete or False)
    )
    if resolved_max_steps < 1:
        raise typer.BadParameter("--max-steps must be at least 1")
    if execute and resolved_auto_complete:
        raise typer.BadParameter("--execute cannot be combined with --auto-complete")
    if max_worker_steps < 1 or timeout_seconds <= 0:
        raise typer.BadParameter("worker limits must be positive")
    schedule = _resolve_schedule(at=at, every=every, cron=cron, timezone=timezone)
    job = create_scheduler_job(
        store=store,
        kind="mission.schedule_once",
        cwd=working_dir,
        schedule=schedule,
        payload={
            "mission_id": mission_id,
            "max_steps": resolved_max_steps,
            "auto_complete": resolved_auto_complete,
            "execution_mode": "agent"
            if execute
            else "simulation"
            if resolved_auto_complete
            else "handoff",
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "max_worker_steps": max_worker_steps,
            "timeout_seconds": timeout_seconds,
            "yes": yes,
            "config_path": str(config_path.resolve()) if config_path else "",
        },
        title=mission_id,
    )
    store.add_job(job)
    if json_output:
        _emit_json(job.to_dict())
        return
    console.print(f"[green]Added scheduler job[/green] {job.id}")
    console.print(f"kind={job.kind} next_run_at={job.next_run_at}")


@scheduler_app.command("add-research")
def scheduler_add_research_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    at: str | None = typer.Option(None, "--at"),
    every: str | None = typer.Option(None, "--every"),
    cron: str | None = typer.Option(None, "--cron"),
    timezone: str = typer.Option("UTC", "--timezone"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    max_risk: str | None = typer.Option(None, "--max-risk"),
    base_branch: str | None = typer.Option(None, "--base-branch"),
    create_branch: bool | None = typer.Option(None, "--create-branch/--no-create-branch"),
    commit: bool | None = typer.Option(None, "--commit/--no-commit"),
    push: bool | None = typer.Option(None, "--push/--no-push"),
    open_pr: bool | None = typer.Option(None, "--open/--no-open"),
    draft_pr: bool | None = typer.Option(None, "--draft/--ready"),
    config_path: Path | None = typer.Option(
        None, "--config", help=f"Override config path (default: {default_config_path()})."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir, store = _load_store(cwd)
    cfg = load_config(config_path)
    scheduler = cfg.research_scheduler
    resolved_max_steps = max_steps if max_steps is not None else (scheduler.max_steps or 5)
    resolved_max_risk = max_risk if max_risk is not None else (scheduler.max_risk or "medium")
    resolved_base_branch = (
        base_branch if base_branch is not None else (scheduler.base_branch or "main")
    )
    resolved_create_branch = (
        create_branch if create_branch is not None else bool(scheduler.create_branch or False)
    )
    resolved_commit = commit if commit is not None else bool(scheduler.commit or False)
    resolved_push = push if push is not None else bool(scheduler.push or False)
    resolved_open_pr = open_pr if open_pr is not None else bool(scheduler.open_pr or False)
    resolved_draft_pr = (
        draft_pr
        if draft_pr is not None
        else (True if scheduler.draft_pr is None else scheduler.draft_pr)
    )
    if resolved_max_steps < 1:
        raise typer.BadParameter("--max-steps must be at least 1")
    if resolved_open_pr and not resolved_push:
        raise typer.BadParameter("--open requires --push so the branch exists remotely")
    if resolved_push and not resolved_create_branch:
        raise typer.BadParameter("--push requires --create-branch")
    schedule = _resolve_schedule(at=at, every=every, cron=cron, timezone=timezone)
    job = create_scheduler_job(
        store=store,
        kind="research.schedule_once",
        cwd=working_dir,
        schedule=schedule,
        payload={
            "max_steps": resolved_max_steps,
            "max_risk": resolved_max_risk,
            "base_branch": resolved_base_branch,
            "create_branch": resolved_create_branch,
            "commit": resolved_commit,
            "push": resolved_push,
            "open_pr": resolved_open_pr,
            "draft_pr": resolved_draft_pr,
        },
        title=f"research-{working_dir.name}",
    )
    store.add_job(job)
    if json_output:
        _emit_json(job.to_dict())
        return
    console.print(f"[green]Added scheduler job[/green] {job.id}")
    console.print(f"kind={job.kind} next_run_at={job.next_run_at}")


@scheduler_app.command("list")
def scheduler_list_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    _, store = _load_store(cwd)
    jobs = store.list_jobs()
    if json_output:
        _emit_json([job.to_dict() for job in jobs])
        return
    if not jobs:
        console.print("[dim]No scheduler jobs registered.[/dim]")
        return
    table = Table("id", "kind", "status", "next_run_at", "last_status")
    for job in jobs:
        table.add_row(job.id, job.kind, job.status, job.next_run_at or "-", job.last_status or "-")
    console.print(table)


@scheduler_app.command("list-runs")
def scheduler_list_runs_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    job_id: str | None = typer.Option(None, "--job"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    _, store = _load_store(cwd)
    runs = store.list_run_records(job_id=job_id)
    if json_output:
        _emit_json([record.to_dict() for record in runs])
        return
    if not runs:
        console.print("[dim]No scheduler runs recorded.[/dim]")
        return
    table = Table("id", "job_id", "status", "result_status", "finished_at")
    for record in runs:
        table.add_row(
            record.id,
            record.job_id,
            record.status,
            record.result_status,
            record.finished_at,
        )
    console.print(table)


@scheduler_app.command("pause")
def scheduler_pause_command(
    *,
    job_id: str = typer.Argument(...),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    _, store = _load_store(cwd)
    try:
        job = store.pause_job(job_id, updated_at=_utcnow_text())
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown scheduler job: {job_id!r}") from exc
    console.print(f"[yellow]Paused[/yellow] {job.id}")


@scheduler_app.command("list-deliveries")
def scheduler_list_deliveries_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    status: str | None = typer.Option(None, "--status"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Inspect notification outcomes independently of job execution."""
    _, store = _load_store(cwd)
    deliveries = store.list_deliveries(status=status)
    if json_output:
        _emit_json([item.to_dict() for item in deliveries])
        return
    table = Table("id", "run", "status", "attempts", "next_attempt", "error")
    for item in deliveries:
        table.add_row(
            item.id,
            item.run_id,
            item.status,
            str(item.attempts),
            item.next_attempt_at,
            item.last_error,
        )
    console.print(table)


@scheduler_app.command("retry-deliveries")
def scheduler_retry_deliveries_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    force: bool = typer.Option(False, "--force", help="Retry before the backoff interval expires."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Retry pending notifications without executing their jobs again."""
    working_dir, store = _load_store(cwd)
    sent = retry_scheduler_deliveries(store=store, hooks=_load_hooks(working_dir), force=force)
    payload = {"sent": sent, "pending": len(store.list_deliveries(status="pending"))}
    if json_output:
        _emit_json(payload)
    else:
        console.print(f"sent={sent} pending={payload['pending']}")


@scheduler_app.command("resume")
def scheduler_resume_command(
    *,
    job_id: str = typer.Argument(...),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    _, store = _load_store(cwd)
    try:
        job = store.load_job(job_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown scheduler job: {job_id!r}") from exc
    next_run_at = job.next_run_at
    if not next_run_at:
        if job.schedule.kind == "at":
            next_run_at = job.schedule.value
        else:
            next_run_at = compute_next_run_at(schedule=job.schedule)
    resumed = store.resume_job(job_id, next_run_at=next_run_at, updated_at=_utcnow_text())
    console.print(f"[green]Resumed[/green] {resumed.id}")


@scheduler_app.command("run-now")
def scheduler_run_now_command(
    *,
    job_id: str = typer.Argument(...),
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir, store = _load_store(cwd)
    hooks = _load_hooks(working_dir)
    try:
        record = run_scheduler_job(
            store=store,
            job_id=job_id,
            trigger="manual",
            hooks=hooks,
            mission_executor=_execute_agent_mission_job,
            prompt_executor=_execute_prompt_job,
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown scheduler job: {job_id!r}") from exc
    if json_output:
        _emit_json(record.to_dict())
        return
    color = "green" if record.status == "completed" else "red"
    console.print(f"[{color}]{record.status}[/{color}] {record.summary}")
    console.print(f"run_id={record.id}")


@scheduler_app.command("start")
def scheduler_start_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    once: bool = typer.Option(False, "--once", help="Process due jobs once and exit."),
    poll_interval: float = typer.Option(30.0, "--poll-interval"),
    max_ticks: int | None = typer.Option(None, "--max-ticks"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    if poll_interval <= 0:
        raise typer.BadParameter("--poll-interval must be positive")
    working_dir, store = _load_store(cwd)
    hooks = _load_hooks(working_dir)
    result = run_scheduler_loop(
        store=store,
        poll_interval_seconds=poll_interval,
        once=once,
        max_ticks=max_ticks,
        hooks=hooks,
        mission_executor=_execute_agent_mission_job,
        prompt_executor=_execute_prompt_job,
    )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(
        f"[green]scheduler tick complete[/green] jobs_seen={result.jobs_seen} "
        f"jobs_executed={result.jobs_executed}"
    )


__all__ = ["scheduler_app"]
