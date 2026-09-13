from __future__ import annotations

import json
from pathlib import Path

import typer

from harness.cli.profiles import (
    PROFILE_RESERVED_ENV,
    active_profile,
    change_credential,
    create_profile,
    load_profile,
    profile_root,
    profiles_root,
    read_credentials,
)

profiles_app = typer.Typer(
    help="Manage isolated identities, workspaces, and personas.", no_args_is_help=True
)
auth_app = typer.Typer(
    help="Manage credentials for a named profile; values are never displayed.", no_args_is_help=True
)


@profiles_app.command("create")
def create(
    name: str,
    workspace: Path | None = typer.Option(None, "--workspace"),
    description: str = typer.Option("", "--description"),
) -> None:
    try:
        profile = create_profile(name, workspace=workspace, description=description)
        typer.echo(profile.model_dump_json(indent=2))
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@profiles_app.command("list")
def list_profiles() -> None:
    records = []
    for path in sorted(profiles_root().glob("*/profile.json")):
        try:
            records.append(load_profile(path.parent.name).model_dump(mode="json"))
        except (OSError, ValueError):
            records.append({"name": path.parent.name, "error": "invalid profile"})
    typer.echo(json.dumps({"active": active_profile(), "profiles": records}, indent=2))


@profiles_app.command("select")
def select(name: str) -> None:
    try:
        load_profile(name)
        path = profiles_root() / "active.json"
        temp = path.with_suffix(".tmp")
        temp.write_text(json.dumps({"name": name}), encoding="utf-8")
        temp.replace(path)
        typer.echo(f"Selected profile {name}")
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@profiles_app.command("persona")
def persona(name: str, text: str = typer.Option(..., "--text")) -> None:
    load_profile(name)
    if len(text) > 16000:
        raise typer.BadParameter("persona must contain at most 16000 characters")
    (profile_root(name) / "SOUL.md").write_text(text, encoding="utf-8")
    typer.echo(f"Updated persona for {name}")


def _credentials(profile: str | None) -> Path:
    name = profile or active_profile()
    if not name:
        raise typer.BadParameter("Select a profile with --profile NAME or profiles select NAME")
    load_profile(name)
    return profile_root(name) / "credentials.env"


@auth_app.command("set")
def auth_set(
    name: str,
    value: str | None = typer.Option(None, "--value", help="Omit to enter securely."),
    profile: str | None = typer.Option(None, "--profile"),
) -> None:
    import re

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name in PROFILE_RESERVED_ENV:
        raise typer.BadParameter("Provide a credential variable name, not a runtime setting")
    target = _credentials(profile)
    secret = value if value is not None else typer.prompt(f"Value for {name}", hide_input=True)
    if not secret or "\x00" in secret:
        raise typer.BadParameter("credential must be nonempty and contain no NUL")
    change_credential(target, name, secret)
    typer.echo(f"Stored {name}")


@auth_app.command("status")
def auth_status(profile: str | None = typer.Option(None, "--profile")) -> None:
    typer.echo(json.dumps({"configured": sorted(read_credentials(_credentials(profile)))}))


@auth_app.command("remove")
def auth_remove(name: str, profile: str | None = typer.Option(None, "--profile")) -> None:
    path = _credentials(profile)
    change_credential(path, name, None)
    typer.echo(f"Removed {name}")
