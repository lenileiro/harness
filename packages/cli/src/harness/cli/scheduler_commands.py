from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from harness.cli.config import default_config_path, load_config
from harness.cli.plugins import load_cli_hook_providers
from harness.core import (
    compute_next_run_at,
    create_scheduler_job,
    default_scheduler_root,
    parse_schedule_spec,
    run_scheduler_job,
    run_scheduler_loop,
)
from harness.core.extensions import LifecycleHook
from harness.core.scheduler_store import SchedulerStore

console = Console()

scheduler_app = typer.Typer(
    name="scheduler",
    help="Manage long-lived scheduled mission and research work.",
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


def _resolve_schedule(*, at: str | None, every: str | None, cron: str | None):
    try:
        return parse_schedule_spec(at=at, every=every, cron=cron)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _utcnow_text() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@scheduler_app.command("add-mission")
def scheduler_add_mission_command(
    *,
    mission_id: str = typer.Option(..., "--mission"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    at: str | None = typer.Option(None, "--at"),
    every: str | None = typer.Option(None, "--every"),
    cron: str | None = typer.Option(None, "--cron"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    auto_complete: bool | None = typer.Option(None, "--auto-complete/--no-auto-complete"),
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
    schedule = _resolve_schedule(at=at, every=every, cron=cron)
    job = create_scheduler_job(
        store=store,
        kind="mission.schedule_once",
        cwd=working_dir,
        schedule=schedule,
        payload={
            "mission_id": mission_id,
            "max_steps": resolved_max_steps,
            "auto_complete": resolved_auto_complete,
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
    schedule = _resolve_schedule(at=at, every=every, cron=cron)
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
        record = run_scheduler_job(store=store, job_id=job_id, trigger="manual", hooks=hooks)
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
    )
    if json_output:
        _emit_json(result.to_dict())
        return
    console.print(
        f"[green]scheduler tick complete[/green] jobs_seen={result.jobs_seen} "
        f"jobs_executed={result.jobs_executed}"
    )


__all__ = ["scheduler_app"]
