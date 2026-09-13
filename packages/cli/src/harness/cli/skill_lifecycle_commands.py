"""Operator commands for pinned packages, history, and reviewed evolution."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from harness.core.paths import user_home
from harness.core.skill_lifecycle import SkillLifecycle
from harness.core.skills import SkillError


def _manager(cwd: Path | None, user: bool, root: Path | None) -> SkillLifecycle:
    if root is not None and user:
        raise typer.BadParameter("choose either --root or --user")
    target = root or (
        user_home() / "skills" if user else (cwd or Path.cwd()).resolve() / ".harness/skills"
    )
    return SkillLifecycle(target)


def _emit(action: Callable[[], Any]) -> None:
    try:
        typer.echo(json.dumps(action(), indent=2, ensure_ascii=False))
    except (SkillError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc


def install_git(
    name: str,
    repository: str,
    commit: str = typer.Option(
        ..., "--commit", help="Full 40-character commit ID; branches and tags are rejected."
    ),
    path: str = typer.Option(
        ..., "--path", help="Skill package directory inside the repository; use . for its root."
    ),
    cwd: Path | None = typer.Option(None, "--cwd"),
    user: bool = typer.Option(False, "--user"),
    root: Path | None = typer.Option(None, "--root"),
) -> None:
    """Install a bounded public GitHub/GitLab archive with commit provenance."""
    _emit(lambda: _manager(cwd, user, root).install_git(name, repository, commit, path))


def inspect_skill(
    name: str,
    cwd: Path | None = typer.Option(None, "--cwd"),
    user: bool = typer.Option(False, "--user"),
    root: Path | None = typer.Option(None, "--root"),
) -> None:
    """Show installed revision, modifications, and retained provenance/history."""
    _emit(lambda: _manager(cwd, user, root).inspect(name))


def update_skill(
    name: str,
    expected: str = typer.Option(..., "--expected", help="Current revision from skills inspect."),
    source: Path | None = typer.Option(None, "--source", help="Local replacement package."),
    repository: str | None = typer.Option(None, "--repository"),
    commit: str | None = typer.Option(None, "--commit"),
    path: str | None = typer.Option(None, "--path"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    user: bool = typer.Option(False, "--user"),
    root: Path | None = typer.Option(None, "--root"),
) -> None:
    """Replace a skill from a local package or a new explicitly pinned commit."""
    if source is not None:
        if any(value is not None for value in (repository, commit, path)):
            raise typer.BadParameter("choose --source or --repository/--commit/--path")
        _emit(lambda: _manager(cwd, user, root).update(name, source, expected=expected))
    else:
        if repository is None or commit is None or path is None:
            raise typer.BadParameter("provide --source or all of --repository, --commit, --path")
        _emit(
            lambda: _manager(cwd, user, root).install_git(
                name, repository, commit, path, expected=expected
            )
        )


def remove_skill(
    name: str,
    expected: str = typer.Option(..., "--expected"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    user: bool = typer.Option(False, "--user"),
    root: Path | None = typer.Option(None, "--root"),
) -> None:
    """Remove an installed skill, retaining history for rollback."""
    _emit(lambda: _manager(cwd, user, root).remove(name, expected=expected))


def rollback_skill(
    name: str,
    revision: str,
    expected: str = typer.Option(
        ..., "--expected", help="Current revision, or absent after removal."
    ),
    cwd: Path | None = typer.Option(None, "--cwd"),
    user: bool = typer.Option(False, "--user"),
    root: Path | None = typer.Option(None, "--root"),
) -> None:
    """Restore a retained revision without overwriting intervening edits."""
    _emit(
        lambda: _manager(cwd, user, root).rollback(
            name, revision, expected=None if expected == "absent" else expected
        )
    )


def evidence(
    name: str | None = typer.Argument(
        None, help="Omit to list generic completed-work evidence for new skills."
    ),
    cwd: Path | None = typer.Option(None, "--cwd"),
    user: bool = typer.Option(False, "--user"),
    root: Path | None = typer.Option(None, "--root"),
) -> None:
    """List actual local run evidence; failed runs remain explicitly marked."""
    _emit(lambda: _manager(cwd, user, root).evidence(name))


def proposal(
    proposal_id: str,
    cwd: Path | None = typer.Option(None, "--cwd"),
    user: bool = typer.Option(False, "--user"),
    root: Path | None = typer.Option(None, "--root"),
) -> None:
    """Inspect a persisted proposed change, its full diff, and source evidence."""
    _emit(lambda: _manager(cwd, user, root).proposal(proposal_id))


def register_skill_lifecycle_commands(app: typer.Typer) -> None:
    for name, function in (
        ("install-git", install_git),
        ("inspect", inspect_skill),
        ("update", update_skill),
        ("remove", remove_skill),
        ("rollback", rollback_skill),
        ("evidence", evidence),
        ("proposal", proposal),
    ):
        app.command(name)(function)
