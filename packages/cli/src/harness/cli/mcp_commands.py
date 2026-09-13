"""Inspect explicit MCP configuration and check actual SDK discovery."""

from __future__ import annotations

import asyncio
import json
import tomllib
from pathlib import Path
from typing import Annotated

import typer

from harness.cli.config import default_config_path
from harness.tools.mcp import MCPServerConfig, MCPToolset, parse_mcp_servers

mcp_app = typer.Typer(
    name="mcp", help="Inspect and connect configured MCP servers.", no_args_is_help=True
)


def _servers(path: Path | None) -> tuple[MCPServerConfig, ...]:
    target = path or default_config_path()
    if not target.exists():
        if path is not None:
            raise ValueError(f"MCP configuration file does not exist: {target}")
        return ()
    with target.open("rb") as handle:
        raw = tomllib.load(handle)
    section = raw.get("mcp", {})
    if not isinstance(section, dict):
        raise ValueError("[mcp] must be a table")
    return parse_mcp_servers(section.get("servers"))


@mcp_app.command("list")
def mcp_list(
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List configured servers without connecting or resolving credential values."""
    try:
        servers = _servers(config_path)
    except (ValueError, OSError) as exc:
        typer.echo(f"MCP configuration error: {exc}", err=True)
        raise typer.Exit(1) from None
    rows = [
        {
            "name": server.name,
            "transport": server.transport,
            "enabled": server.enabled,
            "approval": server.approval,
            "resources": server.expose_resources,
            "prompts": server.expose_prompts,
        }
        for server in servers
    ]
    if json_output:
        typer.echo(json.dumps(rows))
    elif not rows:
        typer.echo(
            "No MCP servers configured. Add [mcp.servers.NAME] to your Harness configuration."
        )
    else:
        for row in rows:
            typer.echo(
                f"{row['name']}  {row['transport']}  {'enabled' if row['enabled'] else 'disabled'}  approval={row['approval']}"
            )


@mcp_app.command("check")
def mcp_check(
    server: Annotated[
        str | None, typer.Argument(help="Check one named server, or all enabled servers.")
    ] = None,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
    cwd: Annotated[Path | None, typer.Option("--cwd")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Initialize servers and discover available tools, then close all connections."""

    async def check(servers: tuple[MCPServerConfig, ...]) -> list[dict[str, str]]:
        async with MCPToolset(servers, cwd=cwd or Path.cwd()) as toolset:
            return [
                {"name": tool.name, "description": tool.description, "approval": tool.approval}
                for tool in toolset.tools
            ]

    try:
        servers = _servers(config_path)
        if server is not None:
            servers = tuple(item for item in servers if item.name == server)
            if not servers:
                raise ValueError(f"MCP server {server!r} is not configured")
        if not any(item.enabled for item in servers):
            raise ValueError("no enabled MCP servers selected")
        rows = asyncio.run(check(servers))
    except (ValueError, OSError, RuntimeError, TimeoutError) as exc:
        typer.echo(f"MCP check failed: {exc}", err=True)
        raise typer.Exit(1) from None
    if json_output:
        typer.echo(
            json.dumps({"servers": [item.name for item in servers if item.enabled], "tools": rows})
        )
    else:
        typer.echo(f"Connected successfully; discovered {len(rows)} tool(s).")
        for row in rows:
            typer.echo(f"{row['name']}  approval={row['approval']}")


@mcp_app.command("login")
def mcp_login(
    name: str, config_path: Annotated[Path | None, typer.Option("--config")] = None
) -> None:
    """Authorize an OAuth MCP server in your browser; refresh tokens stay private."""
    import webbrowser

    import httpx

    from harness.tools.mcp.oauth import authorization_callback, oauth_provider

    async def login(server: MCPServerConfig) -> None:
        async with authorization_callback(server.oauth_redirect_port) as callback:

            async def redirect(url: str) -> None:
                opened = await asyncio.to_thread(webbrowser.open, url)
                if not opened:
                    typer.echo(f"Open this authorization URL in your browser: {url}")

            auth = oauth_provider(server, redirect_handler=redirect, callback_handler=callback)

            def factory(config, headers):
                return httpx.AsyncClient(
                    headers=headers, auth=auth, timeout=300, follow_redirects=False, trust_env=False
                )

            async with MCPToolset(
                (server.model_copy(update={"timeout": 300}),),
                cwd=Path.cwd(),
                http_client_factory=factory,
            ):
                typer.echo(f"Authorized MCP server {server.name}")

    try:
        server = next((item for item in _servers(config_path) if item.name == name), None)
        if server is None or not server.oauth:
            raise ValueError("Configure a named streamable-http server with oauth=true first")
        asyncio.run(login(server))
    except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
        typer.echo(
            f"MCP authorization failed: {type(exc).__name__}. Check server configuration and retry.",
            err=True,
        )
        raise typer.Exit(1) from None


@mcp_app.command("logout")
def mcp_logout(
    name: str, config_path: Annotated[Path | None, typer.Option("--config")] = None
) -> None:
    from harness.tools.mcp.oauth import FileTokenStorage

    server = next((item for item in _servers(config_path) if item.name == name), None)
    if server is None:
        raise typer.BadParameter("Unknown MCP server")
    FileTokenStorage(server).path.unlink(missing_ok=True)
    typer.echo(f"Removed local OAuth credentials for {name}")
