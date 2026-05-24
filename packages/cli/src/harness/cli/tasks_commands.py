from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from harness.storage.sqlite import SQLiteStorage
from harness.tasks import ActivityEvent, ActivityStore, Task, TaskLink, TaskStore
from harness.tasks import activity as task_activity


async def append_task_activity(
    storage: ActivityStore, *, task_id: str, kind: str, data: dict
) -> None:
    event = ActivityEvent(task_id=task_id, kind=kind, data=data)
    await storage.append_activity(event)


def close_if_sqlite(storage: object) -> bool:
    return isinstance(storage, SQLiteStorage)


def tasks_new_command(
    *,
    title: str,
    description: str | None,
    priority: str | None,
    labels: str | None,
    parent: str | None,
    cwd: Path | None,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    run_async: Any,
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    label_list: list[str] = [s.strip() for s in labels.split(",") if s.strip()] if labels else []

    async def _go() -> Task:
        storage = build_storage(db=db, in_memory=in_memory)
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
                ref="",
                title=title,
                description=description,
                priority=priority,  # type: ignore[arg-type]
                labels=label_list,
                parent_id=parent_id,
                cwd=working_dir,
            )
            saved = await store.create_task(draft)
            await append_task_activity(
                storage,  # type: ignore[arg-type]
                task_id=saved.id,
                kind=task_activity.TASK_CREATED,
                data={"ref": saved.ref, "title": saved.title},
            )
            return saved
        finally:
            if close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    task = run_async(_go())
    console.print(f"[green]Created[/green] {task.ref}  {task.title}")


def tasks_list_command(
    *,
    status: str | None,
    limit: int,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    run_async: Any,
    task_status_style: Any,
    truncate: Any,
    ago: Any,
) -> None:
    async def _go() -> list[Task]:
        storage = build_storage(db=db, in_memory=in_memory)
        try:
            store: TaskStore = storage  # type: ignore[assignment]
            return await store.list_tasks(limit=limit, status=status)  # type: ignore[arg-type]
        finally:
            if close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    tasks = run_async(_go())
    if not tasks:
        console.print("[dim]No tasks.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Ref")
    table.add_column("Status")
    table.add_column("Title")
    table.add_column("Labels")
    table.add_column("Updated")
    for task in tasks:
        table.add_row(
            task.ref,
            task_status_style(task.status),
            truncate(task.title, 60),
            ", ".join(task.labels) if task.labels else "—",
            ago(task.updated_at),
        )
    console.print(table)


def tasks_show_command(
    *,
    ref: str,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    run_async: Any,
    render_task: Any,
) -> None:
    async def _go() -> tuple[Task | None, list[ActivityEvent]]:
        storage = build_storage(db=db, in_memory=in_memory)
        try:
            store: TaskStore = storage  # type: ignore[assignment]
            task = await store.get_task_by_ref(ref)
            if task is None:
                return None, []
            activity_store: ActivityStore = storage  # type: ignore[assignment]
            events = await activity_store.list_activity(task_id=task.id, limit=200)
            return task, events
        finally:
            if close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    task, events = run_async(_go())
    if task is None:
        console.print(f"[red]Task not found:[/red] {ref}")
        raise typer.Exit(1)
    render_task(task, events)


def tasks_update_command(
    *,
    ref: str,
    status: str | None,
    title: str | None,
    description: str | None,
    priority: str | None,
    labels: str | None,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    run_async: Any,
) -> None:
    async def _go() -> Task | None:
        storage = build_storage(db=db, in_memory=in_memory)
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
            await append_task_activity(storage, task_id=saved.id, kind=kind, data=data)  # type: ignore[arg-type]
            return saved
        finally:
            if close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    task = run_async(_go())
    if task is None:
        console.print(f"[red]Task not found:[/red] {ref}")
        raise typer.Exit(1)
    console.print(f"[green]Updated[/green] {task.ref}")


def tasks_link_command(
    *,
    ref: str,
    target: str,
    relation: str,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    run_async: Any,
) -> None:
    async def _go() -> Task | None:
        storage = build_storage(db=db, in_memory=in_memory)
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
            await append_task_activity(
                storage,  # type: ignore[arg-type]
                task_id=saved.id,
                kind=task_activity.TASK_LINKED,
                data={"ref": saved.ref, "target_ref": target, "relation": relation},
            )
            return saved
        finally:
            if close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    task = run_async(_go())
    if task is None:
        console.print(f"[red]Task not found:[/red] {ref}")
        raise typer.Exit(1)
    console.print(f"[green]Linked[/green] {task.ref} --{relation}--> {target}")


def tasks_rm_command(
    *,
    ref: str,
    db: Path | None,
    in_memory: bool,
    yes: bool,
    console: Console,
    build_storage: Any,
    run_async: Any,
) -> None:
    if not yes and not Confirm.ask(f"Delete task [bold]{ref}[/bold]?", default=False):
        console.print("[yellow]Aborted.[/yellow]")
        raise typer.Exit(1)

    async def _go() -> bool:
        storage = build_storage(db=db, in_memory=in_memory)
        try:
            store: TaskStore = storage  # type: ignore[assignment]
            task = await store.get_task_by_ref(ref)
            if task is None:
                return False
            await store.delete_task(task.id)
            await append_task_activity(
                storage,  # type: ignore[arg-type]
                task_id=task.id,
                kind=task_activity.TASK_DELETED,
                data={"ref": ref},
            )
            return True
        finally:
            if close_if_sqlite(storage):
                await storage.close()  # type: ignore[union-attr]

    if not run_async(_go()):
        console.print(f"[red]Task not found:[/red] {ref}")
        raise typer.Exit(1)
    console.print(f"[green]Deleted[/green] {ref}")


__all__ = [
    "append_task_activity",
    "close_if_sqlite",
    "tasks_link_command",
    "tasks_list_command",
    "tasks_new_command",
    "tasks_rm_command",
    "tasks_show_command",
    "tasks_update_command",
]
