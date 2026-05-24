from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from harness.core import PendingApproval
from harness.tasks import ActivityEvent, ActivityStore


def approvals_list_command(
    *,
    pending_only: bool,
    task: str | None,
    session_id: str | None,
    limit: int,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    close_if_sqlite: Any,
    run_async: Any,
    approval_status_style: Any,
    truncate: Any,
    ago: Any,
) -> None:
    async def _go() -> list[PendingApproval]:
        storage = build_storage(db=db, in_memory=in_memory)
        try:
            store = storage
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
            if close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    items = run_async(_go())
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
    for approval in items:
        table.add_row(
            approval.id,
            approval_status_style(approval.status),
            approval.tool_name,
            truncate(repr(approval.arguments), 40),
            approval.session_id,
            ago(approval.requested_at),
        )
    console.print(table)


def approvals_show_command(
    *,
    approval_id: str,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    close_if_sqlite: Any,
    run_async: Any,
    render_approval: Any,
) -> None:
    async def _go() -> PendingApproval | None:
        storage = build_storage(db=db, in_memory=in_memory)
        try:
            return await storage.get_approval(approval_id)  # type: ignore[union-attr]
        finally:
            if close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    approval = run_async(_go())
    if approval is None:
        console.print(f"[red]Approval not found:[/red] {approval_id}")
        raise typer.Exit(1)
    render_approval(approval)


def approvals_grant_command(
    *,
    approval_id: str,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    close_if_sqlite: Any,
    run_async: Any,
) -> None:
    async def _go() -> PendingApproval | None:
        storage = build_storage(db=db, in_memory=in_memory)
        try:
            return await storage.resolve_approval(  # type: ignore[union-attr]
                approval_id, status="granted", resolved_by="cli"
            )
        finally:
            if close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    updated = run_async(_go())
    if updated is None:
        console.print(f"[red]Approval not found:[/red] {approval_id}")
        raise typer.Exit(1)
    console.print(
        f"[green]Granted[/green] {updated.id}  "
        f"[dim]({updated.tool_name})[/dim]  — "
        f"resume the session to dispatch."
    )


def approvals_deny_command(
    *,
    approval_id: str,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    close_if_sqlite: Any,
    run_async: Any,
) -> None:
    async def _go() -> PendingApproval | None:
        storage = build_storage(db=db, in_memory=in_memory)
        try:
            return await storage.resolve_approval(  # type: ignore[union-attr]
                approval_id, status="denied", resolved_by="cli"
            )
        finally:
            if close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    updated = run_async(_go())
    if updated is None:
        console.print(f"[red]Approval not found:[/red] {approval_id}")
        raise typer.Exit(1)
    console.print(f"[yellow]Denied[/yellow] {updated.id}  [dim]({updated.tool_name})[/dim]")


def evidence_list_command(
    *,
    task: str | None,
    session_id: str | None,
    tool_name: str | None,
    errors_only: bool,
    limit: int,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    close_if_sqlite: Any,
    run_async: Any,
    truncate: Any,
    ago: Any,
) -> None:
    async def _go() -> list[ActivityEvent]:
        storage = build_storage(db=db, in_memory=in_memory)
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
            if close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

        if tool_name is not None:
            events = [event for event in events if event.data.get("name") == tool_name]
        if errors_only:
            events = [event for event in events if event.data.get("is_error") is True]
        return events

    items = run_async(_go())
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
    for event in items:
        is_error = bool(event.data.get("is_error"))
        status = "[red]error[/red]" if is_error else "[green]ok[/green]"
        duration = event.data.get("duration_ms")
        duration_str = "—" if duration is None else str(duration)
        meta = event.data.get("metadata") or {}
        meta_str = truncate(
            " ".join(f"{k}={v}" for k, v in meta.items()) or "—",
            60,
        )
        table.add_row(
            ago(event.timestamp),
            str(event.data.get("name", "?")),
            truncate(repr(event.data.get("arguments", {})), 30),
            status,
            duration_str,
            meta_str,
        )
    console.print(table)


__all__ = [
    "approvals_deny_command",
    "approvals_grant_command",
    "approvals_list_command",
    "approvals_show_command",
    "evidence_list_command",
]
