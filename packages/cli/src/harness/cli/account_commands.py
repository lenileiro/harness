"""Explicit browser/device login for configured provider accounts."""

from __future__ import annotations

import asyncio
import json
import webbrowser
from pathlib import Path

import typer

from harness.cli.account_auth import AccountAuth, OAuthAccountConfig
from harness.cli.config import load_config
from harness.core import ConfigurationError

accounts_app = typer.Typer(
    help="Authorize, inspect, and remove configured provider accounts.", no_args_is_help=True
)


def account(name: str, config: Path | None = None) -> AccountAuth:
    settings = load_config(config).provider(name)
    if not settings.get("oauth") or not settings.get("base_url"):
        raise typer.BadParameter(
            "Configure [provider.NAME.oauth] and provider base_url before login"
        )
    return AccountAuth(
        name, OAuthAccountConfig.model_validate(settings["oauth"]), resource=settings["base_url"]
    )


@accounts_app.command("login")
def login(
    name: str,
    config: Path | None = typer.Option(None, "--config"),
    open_browser: bool = typer.Option(True, "--browser/--no-browser"),
) -> None:
    async def display(url: str, code: str) -> None:
        typer.echo(f"Open {url}\nAuthorization code: {code}")
        if open_browser:
            await asyncio.to_thread(webbrowser.open, url)

    try:
        asyncio.run(account(name, config).login(display))
    except (ConfigurationError, ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from None
    typer.echo(f"Authorized {name}; credentials are saved for this identity")


@accounts_app.command("status")
def status(name: str, config: Path | None = typer.Option(None, "--config")) -> None:
    typer.echo(json.dumps(account(name, config).status(), indent=2))


@accounts_app.command("logout")
def logout(name: str, config: Path | None = typer.Option(None, "--config")) -> None:
    account(name, config).logout()
    typer.echo(f"Removed local credentials for {name}")
