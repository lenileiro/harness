"""Inspectable single-account provider and tool bundle configuration."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import tomlkit
import typer

from harness.cli.account_commands import account, login
from harness.cli.config import default_config_path, load_config
from harness.cli.portal_tools import PortalConfig

portal_app = typer.Typer(
    help="Configure the Nous-compatible model and tool service bundle.", no_args_is_help=True
)


@portal_app.command("configure")
def configure(
    client_id: str = typer.Option(
        ..., "--client-id", help="Public OAuth client ID registered with your portal."
    ),
    model: str = typer.Option(..., "--model"),
    config: Path | None = typer.Option(None, "--config"),
    routes: list[str] = typer.Option(
        ["web", "images", "speech", "transcription", "browser"], "--route"
    ),
    provider: str = typer.Option("nous", "--provider"),
    replace_existing: bool = typer.Option(False, "--replace-existing"),
) -> None:
    """Write inspectable routing and OAuth settings; no account or service is contacted."""
    target = (config or default_config_path()).expanduser().absolute()
    if not client_id.strip() or not model.strip():
        raise typer.BadParameter("client-id and model must be nonempty")
    if target.is_symlink():
        raise typer.BadParameter("Config must not be a symlink")
    document = tomlkit.parse(target.read_text()) if target.exists() else tomlkit.document()
    providers = document.setdefault("provider", tomlkit.table())
    if (provider in providers or "portal" in document) and not replace_existing:
        raise typer.BadParameter(
            "Provider or portal settings exist; use --replace-existing to replace these sections"
        )
    bundle = PortalConfig.model_validate({"enabled": True, "provider": provider, "routes": routes})
    providers[provider] = {
        "driver": "openai-compatible",
        "base_url": "https://inference-api.nousresearch.com/v1",
        "input_media": ["image", "audio", "file"],
        "oauth": {
            "device_authorization_endpoint": "https://portal.nousresearch.com/api/oauth/device/code",
            "token_endpoint": "https://portal.nousresearch.com/api/oauth/token",
            "client_id": client_id,
            "scopes": "inference:invoke",
            "refresh_token_header": "x-nous-refresh-token",
        },
    }
    defaults = document.setdefault("default", tomlkit.table())
    defaults["provider"], defaults["model"] = provider, model
    document["portal"] = bundle.model_dump(mode="json", exclude_defaults=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".portal-config-", suffix=".toml", dir=target.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(tomlkit.dumps(document))
        load_config(temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    typer.echo(f"Saved {target}. Authorize with harness accounts login {provider}.")


@portal_app.command("login")
def portal_login(
    config: Path | None = typer.Option(None, "--config"),
    open_browser: bool = typer.Option(True, "--browser/--no-browser"),
) -> None:
    login(load_config(config).portal.provider, config, open_browser)


@portal_app.command("info")
def info(config: Path | None = typer.Option(None, "--config")) -> None:
    settings = load_config(config)
    result = {
        "enabled": settings.portal.enabled,
        "provider": settings.portal.provider,
        "routes": settings.portal.routes,
        "tools": sorted(settings.portal.tool_names()),
    }
    try:
        result["account"] = account(settings.portal.provider, config).status()
    except (ValueError, OSError):
        result["account"] = {"configured": False}
    typer.echo(json.dumps(result, indent=2))
