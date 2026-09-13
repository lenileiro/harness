"""Local inspection and operator reconciliation of ambiguous Honcho exports."""

from __future__ import annotations

import json
import sqlite3
from typing import Literal

import typer

from harness.core.paths import user_home

honcho_app = typer.Typer(
    help="Inspect and reconcile this identity's Honcho export ledger.", no_args_is_help=True
)


@honcho_app.command("exports")
def exports() -> None:
    database = user_home() / "integrations/honcho.sqlite3"
    if not database.exists():
        typer.echo("[]")
        return
    with sqlite3.connect(database) as db:
        db.row_factory = sqlite3.Row
        rows = [
            dict(row)
            for row in db.execute(
                "SELECT workspace,reference,state FROM exports ORDER BY workspace,reference LIMIT 1000"
            )
        ]
    typer.echo(json.dumps(rows, indent=2))


@honcho_app.command("reconcile")
def reconcile(
    workspace: str,
    reference: str,
    outcome: Literal["exported", "not-exported"] = typer.Option(
        ...,
        "--outcome",
        help="Use only after checking the remote service for this ambiguous export.",
    ),
) -> None:
    database = user_home() / "integrations/honcho.sqlite3"
    if not database.exists():
        raise typer.BadParameter("No local Honcho export ledger exists")
    with sqlite3.connect(database) as db:
        db.execute("BEGIN IMMEDIATE")
        found = db.execute(
            "SELECT state FROM exports WHERE workspace=? AND reference=?", (workspace, reference)
        ).fetchone()
        if found != ("uncertain",):
            raise typer.BadParameter("Only an uncertain export can be reconciled")
        if outcome == "exported":
            db.execute(
                'UPDATE exports SET state="exported" WHERE workspace=? AND reference=?',
                (workspace, reference),
            )
        else:
            db.execute(
                "DELETE FROM exports WHERE workspace=? AND reference=?", (workspace, reference)
            )
    typer.echo("Recorded the operator's reconciliation; no remote request was sent")
