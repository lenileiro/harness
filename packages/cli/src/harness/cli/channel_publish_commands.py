"""Explicit publisher for ntfy's signed single-owner command envelope."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from uuid import uuid4

import typer

from harness.cli.channels.ntfy import NtfyTransport, sign_ntfy_message
from harness.cli.channels.runtime import build_transport
from harness.cli.channels.transports import ChannelError
from harness.cli.config import load_config


def register_publish_commands(app: typer.Typer) -> None:
    app.command("publish-ntfy")(publish_ntfy)


def publish_ntfy(prompt: str, config_path: Path | None = typer.Option(None, "--config")) -> None:
    """Publish an authenticated command to the configured ntfy input topic."""
    config = load_config(config_path).channels.get("ntfy")
    if config is None:
        raise typer.BadParameter("Configure [channels.ntfy] before publishing")
    if config.profile and config.profile != os.environ.get("HARNESS_PROFILE"):
        raise typer.BadParameter("Use the configured process profile before publishing")
    identifier = uuid4().hex

    async def publish() -> None:
        transport = build_transport("ntfy", config)
        assert isinstance(transport, NtfyTransport)
        try:
            await transport.authenticate()
            message = sign_ntfy_message(
                text=prompt,
                topic=config.topic,
                owner=config.username,
                secret=transport.app_token,
                message_id=identifier,
            )
            await transport.api(
                "POST",
                transport.base + "/",
                data={"topic": config.topic, "message": message},
                auth="Bearer " + transport.token,
            )
        finally:
            await transport.close()

    try:
        asyncio.run(publish())
    except ChannelError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from None
    except Exception:
        typer.echo("ntfy publish failed; inspect the input topic before retrying.", err=True)
        raise typer.Exit(1) from None
    typer.echo(json.dumps({"message_id": identifier, "status": "published"}))
