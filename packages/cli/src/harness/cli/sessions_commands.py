from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from harness.core import ContextBudget, configure_logging
from harness.storage.sqlite import SQLiteStorage


def sessions_list_command(
    *,
    db: Path | None,
    in_memory: bool,
    limit: int,
    status: str | None,
    console: Console,
    build_storage: Any,
    run_async: Any,
    ago: Any,
    status_style: Any,
) -> None:
    async def _go() -> None:
        storage = build_storage(db=db, in_memory=in_memory)
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
        for session in sessions:
            table.add_row(
                session.id,
                status_style(session.status),
                session.provider,
                session.model,
                ago(session.updated_at),
                str(len(session.messages)),
            )
        console.print(table)

    run_async(_go())


def sessions_show_command(
    *,
    session_id: str,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    run_async: Any,
    render_session: Any,
) -> None:
    async def _go() -> None:
        storage = build_storage(db=db, in_memory=in_memory)
        try:
            session = await storage.get(session_id)
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

        if session is None:
            console.print(f"[red]Session not found:[/red] {session_id}")
            raise typer.Exit(1)
        render_session(session)

    run_async(_go())


def sessions_resume_command(
    *,
    session_id: str,
    prompt: str | None,
    db: Path | None,
    in_memory: bool,
    cwd: Path | None,
    base_url: str | None,
    max_steps: int,
    failover: str | None,
    yes: bool,
    inbox: bool,
    verify: str | None,
    max_context_tokens: int | None,
    config_path: Path | None,
    verbose: bool,
    console: Console,
    load_cli_config: Any,
    resolve_chain: Any,
    build_storage: Any,
    build_verifier: Any,
    build_adapter: Any,
    build_agent: Any,
    render: Any,
    run_async: Any,
) -> None:
    configure_logging(level="DEBUG" if verbose else "INFO")
    cfg = load_cli_config(config_path)
    working_dir_hint = cwd.resolve() if cwd else None

    async def _go() -> None:
        storage = build_storage(db=db, in_memory=in_memory, cwd=working_dir_hint)
        try:
            session = await storage.get(session_id)
            if session is None:
                console.print(f"[red]Session not found:[/red] {session_id}")
                raise typer.Exit(1)

            working_dir = (cwd or session.cwd).resolve()
            chain = resolve_chain(
                failover_flag=failover, provider_flag=session.provider, config=cfg
            )
            verifier = build_verifier(
                verify,
                chain=chain,
                model=session.model,
                config=cfg,
                build_adapter=build_adapter,
            )
            budget = (
                ContextBudget(max_tokens=max_context_tokens)
                if max_context_tokens is not None
                else None
            )
            agent = build_agent(
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
                    render(event)
            except Exception as exc:
                console.print(f"\n[red]Unhandled error:[/red] {exc!s}")
                raise typer.Exit(1) from None
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    run_async(_go())


def sessions_rm_command(
    *,
    session_id: str,
    db: Path | None,
    in_memory: bool,
    yes: bool,
    console: Console,
    build_storage: Any,
    run_async: Any,
) -> None:
    if not yes and not Confirm.ask(f"Delete session [bold]{session_id}[/bold]?", default=False):
        console.print("[yellow]Aborted.[/yellow]")
        raise typer.Exit(1)

    async def _go() -> None:
        storage = build_storage(db=db, in_memory=in_memory)
        try:
            await storage.delete(session_id)
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    run_async(_go())
    console.print(f"[green]Deleted[/green] {session_id}")


def sessions_fork_command(
    *,
    session_id: str,
    prompt: str | None,
    new_id: str | None,
    db: Path | None,
    in_memory: bool,
    yes: bool,
    config_path: Path | None,
    verbose: bool,
    console: Console,
    load_cli_config: Any,
    build_storage: Any,
    build_agent: Any,
    render: Any,
    run_async: Any,
    fork_session_fn: Any,
) -> None:
    configure_logging(level="DEBUG" if verbose else "INFO")
    cfg = load_cli_config(config_path)

    async def _go() -> None:
        from harness.core.errors import ConfigurationError as _CE

        storage = build_storage(db=db, in_memory=in_memory)
        try:
            try:
                forked = await fork_session_fn(storage, session_id, new_session_id=new_id)
            except _CE as exc:
                console.print(f"[red]Error:[/red] {exc}")
                raise typer.Exit(1) from None
            console.print(f"[green]Forked[/green] {session_id} → {forked.id}")
            if not prompt:
                console.print(
                    f'[dim]Resume with:[/dim] harness sessions resume {forked.id} "<prompt>"'
                )
                return
            chain = [forked.provider]
            agent = build_agent(
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
                render(event)
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    try:
        run_async(_go())
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user.[/yellow]")
        raise typer.Exit(130) from None


def sessions_diff_command(
    *,
    session_id: str,
    db: Path | None,
    in_memory: bool,
    build_storage: Any,
    render_session_diff: Any,
    console: Console,
    run_async: Any,
) -> None:
    async def _go() -> None:
        storage = build_storage(db=db, in_memory=in_memory)
        try:
            activity = await storage.list_activity(session_id=session_id)  # type: ignore[attr-defined]
            render_session_diff(activity, console)
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    run_async(_go())


__all__ = [
    "sessions_diff_command",
    "sessions_fork_command",
    "sessions_list_command",
    "sessions_resume_command",
    "sessions_rm_command",
    "sessions_show_command",
]
