from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from harness.core import MemoryEntry, configure_logging
from harness.storage.sqlite import SQLiteStorage, default_db_path


def run_goal_command(
    *,
    prompt: str,
    model: str | None,
    provider: str | None,
    base_url: str | None,
    cwd: Path | None,
    max_steps: int,
    db: Path | None,
    in_memory: bool,
    yes: bool,
    config_path: Path | None,
    verbose: bool,
    console: Console,
    load_cli_config: Any,
    resolve_chain: Any,
    run_async: Any,
    run_once: Any,
) -> None:
    """Run a multi-step goal: the LLM plans first, then executes each step."""
    configure_logging(level="DEBUG" if verbose else "INFO")

    cfg = load_cli_config(config_path)
    chain = resolve_chain(failover_flag=None, provider_flag=provider, config=cfg)
    effective_model = model or cfg.default_model or "llama3.2"

    working_dir = (cwd or Path.cwd()).resolve()
    if not working_dir.exists() or not working_dir.is_dir():
        console.print(f"[red]--cwd does not exist or is not a directory: {working_dir}[/red]")
        raise typer.Exit(2)

    try:
        run_async(
            run_once(
                prompt=prompt,
                model=effective_model,
                chain=chain,
                base_url=base_url,
                cwd=working_dir,
                max_steps=max_steps,
                max_output_tokens=None,
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


def init_workspace(
    *,
    cwd: Path | None,
    console: Console,
) -> None:
    """Initialise a workspace-local Harness database in .harness/harness.db."""
    working_dir = (cwd or Path.cwd()).resolve()
    harness_dir = working_dir / ".harness"
    db_path = harness_dir / "harness.db"

    if db_path.exists():
        console.print(f"[dim]Already initialised at [/dim]{db_path}[dim] — nothing to do.[/dim]")
        return

    harness_dir.mkdir(parents=True, exist_ok=True)
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


def memory_save_command(
    *,
    text: str,
    kind: str,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    run_async: Any,
) -> None:
    valid_kinds = {"user_preference", "user_fact", "project_fact", "project_context"}
    if kind not in valid_kinds:
        console.print(
            f"[red]Invalid --kind:[/red] {kind!r}. Choose from: {', '.join(sorted(valid_kinds))}"
        )
        raise typer.Exit(1)

    async def _go() -> None:
        storage = build_storage(db=db, in_memory=in_memory)
        try:
            entry = MemoryEntry(kind=kind, text=text)  # type: ignore[arg-type]
            saved = await storage.save_memory(entry)  # type: ignore[attr-defined]
            console.print(f"[green]Saved[/green] {saved.id}  ({saved.kind})  {saved.text}")
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    run_async(_go())


def memory_list_command(
    *,
    kind: str | None,
    limit: int,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    run_async: Any,
    ago: Any,
) -> None:
    valid_kinds = {"user_preference", "user_fact", "project_fact", "project_context"}
    if kind is not None and kind not in valid_kinds:
        console.print(f"[red]Invalid --kind:[/red] {kind!r}")
        raise typer.Exit(1)

    async def _go() -> None:
        storage = build_storage(db=db, in_memory=in_memory)
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
            for entry in entries:
                table.add_row(entry.id, entry.kind, entry.text, ago(entry.created_at))
            console.print(table)
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    run_async(_go())


def memory_search_command(
    *,
    query: str,
    limit: int,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    run_async: Any,
    ago: Any,
) -> None:
    async def _go() -> None:
        storage = build_storage(db=db, in_memory=in_memory)
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
            for entry in entries:
                table.add_row(entry.id, entry.kind, entry.text, ago(entry.created_at))
            console.print(table)
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    run_async(_go())


def memory_rm_command(
    *,
    entry_id: str,
    db: Path | None,
    in_memory: bool,
    yes: bool,
    console: Console,
    build_storage: Any,
    run_async: Any,
) -> None:
    async def _go() -> None:
        storage = build_storage(db=db, in_memory=in_memory)
        try:
            existing = await storage.list_memory(limit=1000)  # type: ignore[attr-defined]
            match = next((entry for entry in existing if entry.id == entry_id), None)
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

    run_async(_go())


__all__ = [
    "default_db_path",
    "init_workspace",
    "memory_list_command",
    "memory_rm_command",
    "memory_save_command",
    "memory_search_command",
    "run_goal_command",
]
