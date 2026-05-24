from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from harness.cli.common import (
    KNOWN_PROVIDERS,
    _build_adapter,
    _build_tools,
    _load_cli_config,
    _run_async,
    _truncate,
    console,
)
from harness.cli.plugins import discover_cli_plugins
from harness.core import ApprovalPolicy
from harness.core.plugin_loader import validate_provider_plugin

providers_app = typer.Typer(
    name="providers", help="Inspect available providers.", no_args_is_help=True
)

tools_app = typer.Typer(name="tools", help="Inspect the built-in tools.", no_args_is_help=True)
plugins_app = typer.Typer(name="plugins", help="Inspect discovered plugins.", no_args_is_help=True)


@providers_app.command("list")
def providers_list_cmd(
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """List known providers and their configuration status."""
    cfg = _load_cli_config(config_path)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Provider")
    table.add_column("Status")
    table.add_column("Notes")

    settings = cfg.provider("ollama")
    ollama_base = settings.get("base_url") or os.environ.get(
        "OLLAMA_HOST", "http://localhost:11434"
    )
    table.add_row(
        "ollama",
        "[green]ready[/green]",
        f"base_url: {ollama_base}",
    )

    has_or_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    or_settings = cfg.provider("openrouter")
    or_status = "[green]ready[/green]" if has_or_key else "[red]missing OPENROUTER_API_KEY[/red]"
    or_notes_parts = []
    if has_or_key:
        or_notes_parts.append("env: OPENROUTER_API_KEY set")
    if "http_referer" in or_settings:
        or_notes_parts.append(f"http_referer: {or_settings['http_referer']}")
    if "x_title" in or_settings:
        or_notes_parts.append(f"x_title: {or_settings['x_title']}")
    table.add_row("openrouter", or_status, ", ".join(or_notes_parts) or "—")

    console.print(table)


@providers_app.command("capabilities")
def providers_capabilities_cmd(
    name: Annotated[str, typer.Argument(help="Provider name (ollama or openrouter).")],
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Print a provider's reported Capabilities."""
    cfg = _load_cli_config(config_path)
    if name not in KNOWN_PROVIDERS:
        console.print(f"[red]Unknown provider:[/red] {name}")
        raise typer.Exit(2)

    async def _go() -> None:
        try:
            adapter = _build_adapter(name, base_url=None, config=cfg)
        except Exception as exc:
            console.print(f"[red]Could not construct adapter:[/red] {exc}")
            raise typer.Exit(2) from None
        caps = await adapter.capabilities()
        table = Table(show_header=False)
        table.add_column("Field", style="bold")
        table.add_column("Value")
        table.add_row("streaming", str(caps.streaming))
        table.add_row("tool_use", str(caps.tool_use))
        table.add_row("structured_output", str(caps.structured_output))
        table.add_row(
            "max_context_tokens",
            "—" if caps.max_context_tokens is None else str(caps.max_context_tokens),
        )
        table.add_row(
            "models",
            "—" if caps.models is None else ", ".join(caps.models),
        )
        console.print(table)

    _run_async(_go())


@tools_app.command("list")
def tools_list_cmd(
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Working directory used to construct fs/shell tools."),
    ] = None,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """List available tools with their effective approval levels."""
    cfg = _load_cli_config(config_path)
    working_dir = (cwd or Path.cwd()).resolve()
    registry = _build_tools(working_dir, config=cfg)
    policy = ApprovalPolicy(default="prompt", per_tool=dict(cfg.approval))

    table = Table(show_header=True, header_style="bold")
    table.add_column("Tool")
    table.add_column("Approval")
    table.add_column("Description")
    for tool in registry.all():
        effective = policy.decide(tool)
        color = {"auto": "green", "prompt": "yellow", "deny": "red"}.get(effective, "white")
        table.add_row(
            tool.name,
            f"[{color}]{effective}[/{color}]",
            _truncate(tool.description, 80),
        )
    console.print(table)


@plugins_app.command("list")
def plugins_list_cmd(
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Workspace used to resolve .harness/plugins."),
    ] = None,
    kind: Annotated[
        str,
        typer.Option(
            "--kind",
            help="Plugin kind to list: all, tool, experience, domain-profile, verifier, or critic.",
        ),
    ] = "all",
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """List discovered provider plugins and their precedence-resolved sources."""
    cfg = _load_cli_config(config_path)
    working_dir = (cwd or Path.cwd()).resolve()
    kind_map = {
        "all": None,
        "tool": "tool",
        "experience": "experience",
        "domain-profile": "domain_profile",
        "verifier": "verifier",
        "critic": "critic",
    }
    resolved_kind = kind_map.get(kind)
    if kind not in kind_map:
        console.print(
            "[red]Invalid --kind; expected all, tool, experience, domain-profile, verifier, or critic.[/red]"
        )
        raise typer.Exit(2)
    plugins = discover_cli_plugins(working_dir, config=cfg, kind=resolved_kind)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Plugin", no_wrap=True)
    table.add_column("Kind", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Provider", overflow="fold")
    table.add_column("Path")
    table.add_column("Description")
    for plugin in plugins:
        table.add_row(
            plugin.name,
            plugin.kind,
            plugin.source,
            plugin.provider_ref,
            str(plugin.path) if plugin.path else "—",
            plugin.description or "—",
        )
    console.print(table)


@plugins_app.command("validate")
def plugins_validate_cmd(
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Workspace used to resolve .harness/plugins."),
    ] = None,
    kind: Annotated[
        str,
        typer.Option(
            "--kind",
            help="Plugin kind to validate: all, tool, experience, domain-profile, verifier, or critic.",
        ),
    ] = "all",
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
) -> None:
    """Attempt to load discovered plugins and report readiness."""
    cfg = _load_cli_config(config_path)
    working_dir = (cwd or Path.cwd()).resolve()
    kind_map = {
        "all": None,
        "tool": "tool",
        "experience": "experience",
        "domain-profile": "domain_profile",
        "verifier": "verifier",
        "critic": "critic",
    }
    resolved_kind = kind_map.get(kind)
    if kind not in kind_map:
        console.print(
            "[red]Invalid --kind; expected all, tool, experience, domain-profile, verifier, or critic.[/red]"
        )
        raise typer.Exit(2)
    plugins = discover_cli_plugins(working_dir, config=cfg, kind=resolved_kind)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Plugin", no_wrap=True)
    table.add_column("Kind", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail")
    failed = False
    for plugin in plugins:
        ok, detail = validate_provider_plugin(plugin)
        failed = failed or not ok
        table.add_row(
            plugin.name,
            plugin.kind,
            plugin.source,
            "[green]ok[/green]" if ok else "[red]error[/red]",
            _truncate(detail, 120),
        )
    console.print(table)
    if failed:
        raise typer.Exit(1)
