from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path

import typer

from harness.cli.channel_photon_commands import register_photon_commands
from harness.cli.channel_publish_commands import register_publish_commands
from harness.cli.channel_weixin_commands import register_weixin_commands
from harness.cli.channels.runtime import build_transport, run_transport
from harness.cli.channels.transports import ChannelError
from harness.cli.config import load_config
from harness.core.gateway_channels import ChannelConfig, ChannelStore

channel_app = typer.Typer(
    name="channel",
    help="Run authenticated messaging channels with scoped approvals and durable delivery.",
    no_args_is_help=True,
)
register_publish_commands(channel_app)
register_photon_commands(channel_app)
register_weixin_commands(channel_app)


@channel_app.command("run")
def channel_run(
    name: str,
    cwd: Path | None = typer.Option(None, "--cwd"),
    config_path: Path | None = typer.Option(None, "--config"),
    allow_user: list[str] | None = typer.Option(None, "--allow-user"),
    token_env: str | None = typer.Option(None, "--token-env"),
    app_token_env: str | None = typer.Option(None, "--app-token-env"),
) -> None:
    """Run a foreground bot; credentials are read only from environment references."""
    cfg = load_config(config_path)
    configured = getattr(cfg, "channels", {}).get(name)
    selected = configured or ChannelConfig(
        token_env=f"{name.upper()}_BOT_TOKEN",
        app_token_env="SLACK_APP_TOKEN" if name == "slack" else "",
    )
    if allow_user is not None:
        selected = replace(selected, allowed_users=allow_user)
    if token_env is not None:
        selected = replace(selected, token_env=token_env)
    if app_token_env is not None:
        selected = replace(selected, app_token_env=app_token_env)
    if selected.profile and selected.profile != os.environ.get("HARNESS_PROFILE"):
        raise typer.BadParameter(
            "Channel profile must match the process profile; start with harness --profile NAME channel run"
        )
    working_dir = (cwd or Path.cwd()).resolve()
    if not working_dir.is_dir():
        raise typer.BadParameter("--cwd must be an existing directory")
    try:
        asyncio.run(run_transport(cwd=working_dir, transport=build_transport(name, selected)))
    except ChannelError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    except KeyboardInterrupt:
        return
    except Exception:
        # HTTP and socket exception reprs may contain bot tokens/session tickets.
        typer.echo(
            "Channel stopped; inspect channel status and credential configuration.", err=True
        )
        raise typer.Exit(1) from None


@channel_app.command("status")
def channel_status(name: str, cwd: Path | None = typer.Option(None, "--cwd")) -> None:
    store = ChannelStore(cwd=(cwd or Path.cwd()).resolve(), transport=name)
    try:
        typer.echo(json.dumps(store.status(include_private=True), indent=2))
    finally:
        store.close()


@channel_app.command("retry")
def channel_retry(
    name: str,
    item_id: str,
    inbound: bool = typer.Option(
        False,
        "--inbound",
        help="Retry an uncertain model dispatch after inspecting its original session.",
    ),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    """Retry a failed/uncertain delivery after inspecting its destination for duplicates."""
    store = ChannelStore(cwd=(cwd or Path.cwd()).resolve(), transport=name)
    try:
        if not store.retry(item_id, inbound=inbound):
            raise typer.BadParameter("Item not found or not in a retryable state")
        typer.echo("Queued for the next running channel worker.")
    finally:
        store.close()


@channel_app.command("run-all")
def channel_run_all(
    cwd: Path | None = typer.Option(None, "--cwd"),
    config_path: Path | None = typer.Option(None, "--config"),
) -> None:
    """Own all configured channel connections in one foreground process."""
    config = load_config(config_path)
    if not config.channels:
        raise typer.BadParameter("Configure at least one [channels.NAME] table")
    working_dir = (cwd or Path.cwd()).resolve()
    if not working_dir.is_dir():
        raise typer.BadParameter("--cwd must be an existing directory")
    # Validate every configuration before opening any connection. Profiles have
    # separate process credentials; one daemon never switches identity globally.
    for channel in config.channels.values():
        if channel.profile and channel.profile != os.environ.get("HARNESS_PROFILE"):
            raise typer.BadParameter("Every channel must match the selected process profile")

    async def serve_all():
        transports = []
        try:
            for name, selected in config.channels.items():
                transports.append(build_transport(name, selected))
            async with asyncio.TaskGroup() as group:
                for transport in transports:
                    group.create_task(
                        run_transport(cwd=working_dir, transport=transport),
                        name=f"channel-{transport.name}",
                    )
        finally:
            # Also close transports built before a later configuration fails,
            # and tasks cancelled before their runner entered its own finally.
            await asyncio.gather(
                *(transport.close() for transport in transports), return_exceptions=True
            )

    try:
        asyncio.run(serve_all())
    except KeyboardInterrupt:
        return
    except Exception:
        typer.echo(
            "A channel stopped; all owned connections were closed. Inspect channel status and configuration.",
            err=True,
        )
        raise typer.Exit(1) from None
