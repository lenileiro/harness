from __future__ import annotations

import json
from pathlib import Path

import typer
import yaml

from harness.cli.common import _load_cli_config
from harness.cli.skill_lifecycle_commands import register_skill_lifecycle_commands
from harness.core.paths import user_home
from harness.core.skill_lifecycle import SkillLifecycle
from harness.core.skills import (
    SkillError,
    SkillLibrary,
    default_skill_paths,
    validate_skill_name,
)

skills_app = typer.Typer(
    help="Discover, inspect, and install portable Agent Skills.", no_args_is_help=True
)
register_skill_lifecycle_commands(skills_app)


def _library(cwd: Path | None, config_path: Path | None) -> SkillLibrary:
    config = _load_cli_config(config_path)
    working = (cwd or Path.cwd()).resolve()
    if not config.skills_enabled:
        return SkillLibrary()
    roots = [(working / Path(p).expanduser()).resolve() for p in config.skill_paths]
    return SkillLibrary.load(roots + default_skill_paths(working))


@skills_app.command("list")
def skills_list(
    cwd: Path | None = typer.Option(None, "--cwd"),
    config_path: Path | None = typer.Option(None, "--config"),
) -> None:
    library = _library(cwd, config_path)
    typer.echo(
        json.dumps(
            {
                "skills": [
                    {"name": s.name, "description": s.description, "path": str(s.directory)}
                    for s in library.skills.values()
                ],
                "errors": library.errors,
                "shadowed": library.shadowed,
            },
            indent=2,
        )
    )


@skills_app.command("show")
def skills_show(
    name: str,
    cwd: Path | None = typer.Option(None, "--cwd"),
    config_path: Path | None = typer.Option(None, "--config"),
) -> None:
    try:
        typer.echo(_library(cwd, config_path).get(name).read())
    except SkillError as exc:
        raise typer.BadParameter(str(exc)) from exc


@skills_app.command("validate")
def skills_validate(
    cwd: Path | None = typer.Option(None, "--cwd"),
    config_path: Path | None = typer.Option(None, "--config"),
) -> None:
    library = _library(cwd, config_path)
    for path, error in library.errors.items():
        typer.echo(f"{path}: {error}", err=True)
    typer.echo(
        f"{len(library.skills)} valid skills; {len(library.errors)} invalid; "
        f"{len(library.shadowed)} shadowed"
    )
    if library.errors:
        raise typer.Exit(1)


@skills_app.command("install")
def skills_install(
    source: Path,
    cwd: Path | None = typer.Option(None, "--cwd"),
    user: bool = typer.Option(False, "--user"),
) -> None:
    root = user_home() / "skills" if user else (cwd or Path.cwd()).resolve() / ".harness/skills"
    try:
        typer.echo(SkillLifecycle(root).install(source)["path"])
    except (SkillError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc


@skills_app.command("create")
def skills_create(
    name: str,
    description: str = typer.Option(..., "--description"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    try:
        validate_skill_name(name)
        if not description.strip() or len(description) > 1024:
            raise SkillError("description must contain 1-1024 characters")
        destination = (cwd or Path.cwd()).resolve() / ".harness/skills" / name
        destination.mkdir(parents=True, exist_ok=False)
        text = "---\n" + yaml.safe_dump({"name": name, "description": description}, sort_keys=False)
        (destination / "SKILL.md").write_text(
            text + "---\n\n# Instructions\n\nDescribe when and how to perform this task.\n",
            encoding="utf-8",
        )
        typer.echo(str(destination / "SKILL.md"))
    except (SkillError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
