from __future__ import annotations

import difflib
from typing import Any

from rich.panel import Panel

from harness.cli.common import _ago, _args_preview, console
from harness.core import PendingApproval, Session
from harness.tasks import ActivityEvent, Task

_STATUS_STYLES = {
    "pending": "white",
    "running": "blue",
    "paused": "yellow",
    "done": "green",
    "failed": "red",
    "cancelled": "magenta",
}

_APPROVAL_STATUS_STYLES = {
    "pending": "yellow",
    "granted": "green",
    "denied": "red",
}

_TASK_STATUS_STYLES = {
    "backlog": "white",
    "todo": "cyan",
    "in_progress": "blue",
    "waiting": "yellow",
    "done": "green",
    "cancelled": "magenta",
}


def _render_session_diff(activity: list[ActivityEvent], con: Any) -> None:
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


def _status_style(status: str) -> str:
    color = _STATUS_STYLES.get(status, "white")
    return f"[{color}]{status}[/{color}]"


def _approval_status_style(status: str) -> str:
    color = _APPROVAL_STATUS_STYLES.get(status, "white")
    return f"[{color}]{status}[/{color}]"


def _task_status_style(status: str) -> str:
    color = _TASK_STATUS_STYLES.get(status, "white")
    return f"[{color}]{status}[/{color}]"


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


def _compact_event_data(data: dict) -> str:
    if not data:
        return ""
    parts = []
    for k, v in data.items():
        if isinstance(v, str) and len(v) > 60:
            v = v[:57] + "…"
        parts.append(f"{k}={v}")
    return " ".join(parts)


def _render_task(task: Task, events: list[ActivityEvent]) -> None:
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
