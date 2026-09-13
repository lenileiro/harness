"""Explicit Photon device authorization and private project credential setup."""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path

import httpx
import typer
from filelock import FileLock

from harness.cli.channels.photon_setup import PhotonSetup
from harness.cli.channels.transports import ChannelError
from harness.core.paths import read_regular_file


def photon_account_path(cwd: Path) -> Path:
    root = cwd.resolve()
    current = root
    for component in (".harness", "channels", "photon-account.json"):
        current /= component
        if current.is_symlink() or (
            current.exists() and getattr(current.lstat(), "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("Photon account path must not contain symlinks or reparse points")
    return current


def save_photon_account(cwd: Path, account: dict[str, str], *, replace: bool = False) -> Path:
    target = photon_account_path(cwd)
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if target.exists():
        if not replace:
            raise ValueError("Photon account exists; use --replace explicitly")
        read_regular_file(target, max_bytes=16 * 1024)
    descriptor, name = tempfile.mkstemp(prefix=".photon-", dir=target.parent)
    staged = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            os.chmod(staged, 0o600)
            json.dump(account, stream)
            stream.flush()
            os.fsync(stream.fileno())
        photon_account_path(cwd)
        if replace:
            os.replace(staged, target)
        else:
            os.link(staged, target)
        return target
    finally:
        staged.unlink(missing_ok=True)


def register_photon_commands(app: typer.Typer) -> None:
    @app.command("photon-setup")
    def photon_setup(
        phone: str = typer.Option(..., "--phone"),
        project: str = typer.Option("", "--project"),
        create_project: str = typer.Option("", "--create-project"),
        cwd: Path | None = typer.Option(None, "--cwd"),
        replace: bool = typer.Option(False, "--replace"),
        timeout: int = typer.Option(600, "--timeout", min=1, max=1800),
    ) -> None:
        """Authorize Photon, select/create a project and register one phone without sending an invite."""
        root = (cwd or Path.cwd()).resolve()
        if not root.is_dir() or not re.fullmatch(r"\+[1-9][0-9]{6,14}", phone):
            raise typer.BadParameter("Provide an existing --cwd and an E.164 --phone")
        if bool(project) == bool(create_project):
            raise typer.BadParameter("Choose exactly one of --project or --create-project")
        try:
            target = photon_account_path(root)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with FileLock(target.with_suffix(".lock"), timeout=0, mode=0o600):
                if target.exists() and not replace:
                    raise ValueError("Photon account exists; use --replace explicitly")

                async def run():
                    async def display(uri: str, code: str):
                        typer.echo(f"Open {uri} and enter code {code} to authorize Photon.")

                    async with httpx.AsyncClient(
                        timeout=30, follow_redirects=False, trust_env=False
                    ) as client:
                        setup = PhotonSetup(client)
                        await setup.login(display, timeout_seconds=timeout)
                        return await setup.setup(
                            phone=phone, project_id=project, create_project=create_project
                        )

                account = asyncio.run(run())
                target = save_photon_account(root, account, replace=replace)
        except KeyboardInterrupt:
            return
        except (ValueError, ChannelError, TimeoutError) as exc:
            typer.echo(str(exc) or "Photon authorization timed out", err=True)
            raise typer.Exit(1) from None
        except Exception:
            typer.echo("Photon setup failed; credentials were not printed", err=True)
            raise typer.Exit(1) from None
        typer.echo("Credentials saved. Add to the selected profile configuration:")
        typer.echo("[channels.photon]")
        typer.echo("app_id = " + json.dumps(account["project_id"]))
        typer.echo("account_file = " + json.dumps(str(target)))
        typer.echo("allowed_users = " + json.dumps([account["user_id"]]))
        typer.echo(
            "Assigned iMessage line: "
            + (account["assigned_phone"] or "pending; inspect the Photon dashboard")
        )
