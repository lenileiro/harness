from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from harness.core import Procedure, ProcedureLibrary, curate_procedures

console = Console()

experience_app = typer.Typer(
    name="experience",
    help="Manage reusable procedures and curate accumulated experience.",
    no_args_is_help=True,
)
procedures_app = typer.Typer(
    name="procedures",
    help="List and add writable procedure artifacts.",
    no_args_is_help=True,
)
experience_app.add_typer(procedures_app, name="procedures")


def _procedure_paths(cwd: Path, scope: str) -> list[Path]:
    repo = cwd / ".harness" / "procedures"
    user = Path.home() / ".harness" / "procedures"
    if scope == "repo":
        return [repo]
    if scope == "user":
        return [user]
    if scope != "both":
        raise typer.BadParameter("--scope must be repo, user, or both")
    return [repo, user]


@procedures_app.command("list")
def procedures_list_cmd(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    scope: str = typer.Option("both", "--scope"),
) -> None:
    working = (cwd or Path.cwd()).resolve()
    library = ProcedureLibrary.load(_procedure_paths(working, scope))
    if not library:
        console.print("[dim]No procedures loaded.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Name")
    table.add_column("Domain", no_wrap=True)
    table.add_column("Confidence", justify="right")
    table.add_column("Triggers")
    table.add_column("Source", no_wrap=True)
    for procedure in sorted(
        library.procedures, key=lambda item: (item.confidence, item.created_at), reverse=True
    ):
        table.add_row(
            procedure.id,
            procedure.name,
            procedure.domain,
            f"{procedure.confidence:.1f}",
            ", ".join(procedure.triggers) or "[dim](always)[/dim]",
            procedure.source,
        )
    console.print(table)


@procedures_app.command("add")
def procedures_add_cmd(
    *,
    name: str = typer.Option(..., "--name"),
    body: str = typer.Option(..., "--body"),
    triggers: str | None = typer.Option(None, "--triggers"),
    domain: str = typer.Option("general", "--domain"),
    source: str = typer.Option("human", "--source"),
    confidence: float = typer.Option(1.0, "--confidence"),
    scope: str = typer.Option("repo", "--scope"),
    created_from: str | None = typer.Option(None, "--created-from"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working = (cwd or Path.cwd()).resolve()
    paths = _procedure_paths(working, scope)
    target_root = paths[0]
    library = ProcedureLibrary.load([target_root])
    library.root = target_root
    procedure = Procedure(
        name=name.strip(),
        body=body.strip(),
        triggers=tuple(
            trigger.strip() for trigger in (triggers or "").split(",") if trigger.strip()
        ),
        domain=domain.strip() or "general",
        source=source.strip() or "human",
        confidence=confidence,
        created_from=created_from.strip() if created_from else None,
    )
    target = library.add(procedure)
    console.print(f"[green]Added procedure {procedure.id}[/green] at {target}")


@experience_app.command("curate")
def experience_curate_cmd(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    scope: str = typer.Option("both", "--scope"),
    stale_days: int = typer.Option(30, "--stale-days"),
    low_confidence: float = typer.Option(1.0, "--low-confidence"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    working = (cwd or Path.cwd()).resolve()
    report = curate_procedures(
        _procedure_paths(working, scope),
        stale_days=stale_days,
        low_confidence_threshold=low_confidence,
        dry_run=dry_run,
    )
    console.print(
        f"scanned={report.scanned} archived={report.archived} dry_run={'yes' if dry_run else 'no'}"
    )
    if not report.actions:
        console.print("[dim]No curator actions.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Kind", no_wrap=True)
    table.add_column("Procedure")
    table.add_column("Reason", overflow="fold")
    table.add_column("Archive path", overflow="fold")
    for action in report.actions:
        table.add_row(
            action.kind,
            action.name,
            action.reason,
            str(action.archive_path) if action.archive_path else "—",
        )
    console.print(table)


__all__ = ["experience_app"]
