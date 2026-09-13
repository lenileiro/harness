"""Explicit offline migration commands; no source installation is modified."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer

from harness.cli.openclaw_migration import (
    MigrationError,
    MigrationPlan,
    apply_openclaw,
    inspect_openclaw,
)
from harness.core.paths import read_regular_file

migration_app = typer.Typer(
    help="Review and import another agent's local data into a new Harness profile."
)
openclaw_app = typer.Typer(
    help="Offline OpenClaw settings, memory, skills and selected API-key migration."
)
migration_app.add_typer(openclaw_app, name="openclaw")


def _credentials(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        source, separator, variable = value.partition("=")
        if not separator or not source or not variable or source in result:
            raise MigrationError(
                "each --credential must be a unique SOURCE_ID=DESTINATION_ENV_NAME"
            )
        result[source] = variable
    return result


@openclaw_app.command("inspect")
def inspect_command(
    source: Annotated[
        Path,
        typer.Option(
            "--source",
            help="Explicit OpenClaw state directory (never inferred from this machine's credentials).",
        ),
    ],
    profile: Annotated[str, typer.Option("--profile", help="New Harness profile name.")],
    output: Annotated[
        Path, typer.Option("--output", help="New secret-free JSON plan file; refuses overwrite.")
    ],
    agent: Annotated[str, typer.Option("--agent")] = "main",
    workspace: Annotated[
        Path | None,
        typer.Option(
            "--workspace", help="Explicitly authorize reading an external agent workspace."
        ),
    ] = None,
    credential: Annotated[
        list[str] | None,
        typer.Option(
            "--credential",
            help="Select SOURCE_ID=ENV_NAME from available_credentials; repeat for each static API key.",
        ),
    ] = None,
) -> None:
    """Write a hashed, reviewable plan. Credential values are never written to it."""
    try:
        plan = inspect_openclaw(
            source,
            profile,
            agent=agent,
            workspace=workspace,
            credentials=_credentials(credential or []),
        )
        import os

        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(plan.model_dump_json(indent=2) + "\n")
    except (ValueError, OSError) as exc:
        detail = (
            str(exc)
            if isinstance(exc, MigrationError)
            else "invalid arguments, unreadable source or existing output; inspect paths and syntax"
        )
        typer.echo(f"Migration inspection failed: {detail}", err=True)
        raise typer.Exit(1) from None
    typer.echo(
        f"Saved plan to {output}: {len(plan.memory_files)} memory files, {len(plan.installed_skills)} portable skills, {len(plan.archived_skills)} archived skills, {len(plan.credentials)} selected credentials."
    )
    typer.echo(
        "Review settings, warnings and available_credentials in the plan; apply only after reviewing it."
    )


@openclaw_app.command("apply")
def apply_command(plan_path: Annotated[Path, typer.Option("--plan")]) -> None:
    """Apply an unchanged plan into a new profile without selecting it."""
    try:
        plan = MigrationPlan.model_validate_json(
            read_regular_file(plan_path, max_bytes=4 * 1024 * 1024)
        )
        profile = asyncio.run(apply_openclaw(plan))
    except (ValueError, OSError) as exc:
        detail = (
            str(exc)
            if isinstance(exc, MigrationError)
            else "invalid plan or unavailable target; no existing profile was overwritten"
        )
        typer.echo(f"Migration failed: {detail}", err=True)
        raise typer.Exit(1) from None
    typer.echo(
        f"Created profile {profile.name} at {profile.workspace.parent}; it has not been selected."
    )
    typer.echo(
        f"Check it with harness --profile {profile.name} doctor, then select it explicitly when ready."
    )
