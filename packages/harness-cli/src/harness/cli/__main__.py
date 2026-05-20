"""Harness CLI entry point.

Phase 4 surface:
- `harness run "prompt"`         — one-shot prompt with the full tool set
- `harness sessions list`        — list saved sessions
- `harness sessions show <id>`   — print full transcript
- `harness sessions resume <id>` — continue an existing session
- `harness sessions rm <id>`     — delete a session
- `harness version`              — print the installed CLI version

Providers: ollama, openrouter.
Tools: read_file, write_file, edit_file, list_dir, glob, shell, fetch_url.

Config: `$XDG_CONFIG_HOME/harness/config.toml` (or ~/.config/harness/config.toml)
provides defaults for provider, model, per-provider settings, and per-tool
approval levels. CLI flags override the config.

Tool approvals default to `prompt` for any tool that mutates state or makes
network calls; the CLI shows a Rich prompt. Pass `--yes` to auto-approve
everything (handy for non-interactive use), or set approvals in config.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from harness.adapters.ollama import OllamaAdapter
from harness.adapters.openrouter import OpenRouterAdapter
from harness.cli.approval import RichApprovalHandler
from harness.cli.config import HarnessConfig, default_config_path, load_config
from harness.core import (
    Adapter,
    Agent,
    ApprovalHandler,
    ApprovalPolicy,
    AutoApprove,
    Done,
    ErrorEvent,
    FailoverPolicy,
    RunRequest,
    Session,
    StepCompleted,
    StepStarted,
    Storage,
    TextDelta,
    ToolCallEvent,
    ToolRegistry,
    ToolResultEvent,
    configure_logging,
)
from harness.storage.memory import InMemoryStorage
from harness.storage.sqlite import SQLiteStorage, default_db_path
from harness.tools.fs import (
    EditFileTool,
    GlobTool,
    ListDirTool,
    ReadFileTool,
    WriteFileTool,
)
from harness.tools.shell import ShellTool
from harness.tools.web import FetchUrlTool

app = typer.Typer(
    name="harness",
    help="Harness — Python agent runtime over OpenRouter and Ollama.",
    no_args_is_help=True,
    add_completion=False,
)

sessions_app = typer.Typer(
    name="sessions", help="Inspect, resume, and remove saved sessions.", no_args_is_help=True
)
app.add_typer(sessions_app, name="sessions")

console = Console()


# ---------------------------------------------------------------------------
# Construction helpers
# ---------------------------------------------------------------------------


def _build_storage(*, db: Path | None, in_memory: bool) -> Storage:
    if in_memory:
        return InMemoryStorage()
    return SQLiteStorage(path=db or default_db_path())


def _build_adapter(provider: str, *, base_url: str | None, config: HarnessConfig) -> Adapter:
    settings = config.provider(provider)
    effective_base_url = base_url or settings.get("base_url")
    if provider == "ollama":
        return OllamaAdapter(base_url=effective_base_url) if effective_base_url else OllamaAdapter()
    if provider == "openrouter":
        return OpenRouterAdapter(
            base_url=effective_base_url,
            http_referer=settings.get("http_referer"),
            x_title=settings.get("x_title"),
        )
    raise typer.BadParameter(f"unknown provider: {provider!r}")


def _build_tools(cwd: Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadFileTool(cwd=cwd))
    registry.register(WriteFileTool(cwd=cwd))
    registry.register(EditFileTool(cwd=cwd))
    registry.register(ListDirTool(cwd=cwd))
    registry.register(GlobTool(cwd=cwd))
    registry.register(ShellTool(cwd=cwd))
    registry.register(FetchUrlTool())
    return registry


def _build_agent(
    *,
    provider: str,
    base_url: str | None,
    model: str,
    storage: Storage,
    cwd: Path,
    config: HarnessConfig,
    yes: bool,
) -> Agent:
    adapter = _build_adapter(provider, base_url=base_url, config=config)
    tools = _build_tools(cwd)

    approval_policy = ApprovalPolicy(default="prompt", per_tool=dict(config.approval))
    approval_handler: ApprovalHandler = (
        AutoApprove() if yes else RichApprovalHandler(console=console)
    )

    return Agent(
        adapters={provider: adapter},
        tools=tools,
        storage=storage,
        failover=FailoverPolicy(
            chain=[provider], max_attempts=1, backoff_base=0.0, backoff_jitter=0.0
        ),
        approval_policy=approval_policy,
        approval_handler=approval_handler,
        default_model=model,
        default_cwd=str(cwd),
    )


def _load_cli_config(config_path: Path | None) -> HarnessConfig:
    try:
        return load_config(config_path)
    except Exception as exc:
        console.print(f"[red]Bad config:[/red] {exc}")
        raise typer.Exit(2) from None


# ---------------------------------------------------------------------------
# version
# ---------------------------------------------------------------------------


@app.command()
def version() -> None:
    """Print the installed harness-cli version."""
    from harness.cli import __version__

    typer.echo(__version__)


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


@app.command()
def run(
    prompt: Annotated[str, typer.Argument(help="The user prompt for the agent.")],
    model: Annotated[
        str | None,
        typer.Option("--model", "-m", help="Model identifier (overrides config)."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option(
            "--provider", "-p", help="Provider: 'ollama' or 'openrouter' (overrides config)."
        ),
    ] = None,
    base_url: Annotated[
        str | None, typer.Option("--base-url", help="Override the provider's base URL.")
    ] = None,
    cwd: Annotated[
        Path | None,
        typer.Option(
            "--cwd", help="Working directory for filesystem tools (default: current dir)."
        ),
    ] = None,
    max_steps: Annotated[
        int, typer.Option("--max-steps", help="Maximum ReAct turns before giving up.")
    ] = 25,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session", help="Reuse / create a session with this id. Required for resume later."
        ),
    ] = None,
    db: Annotated[
        Path | None,
        typer.Option("--db", help=f"SQLite session db path. Default: {default_db_path()}."),
    ] = None,
    in_memory: Annotated[
        bool, typer.Option("--in-memory", help="Use in-memory storage (session lost on exit).")
    ] = False,
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Auto-approve all tool calls (non-interactive).")
    ] = False,
    config_path: Annotated[
        Path | None,
        typer.Option("--config", help=f"Override config path (default: {default_config_path()})."),
    ] = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Enable DEBUG logging to stderr.")
    ] = False,
) -> None:
    """Run a single prompt through the agent and stream the result to stdout."""
    configure_logging(level="DEBUG" if verbose else "INFO")

    cfg = _load_cli_config(config_path)
    effective_provider = provider or cfg.default_provider or "ollama"
    effective_model = model or cfg.default_model or "llama3.2"

    working_dir = (cwd or Path.cwd()).resolve()
    if not working_dir.exists() or not working_dir.is_dir():
        console.print(f"[red]--cwd does not exist or is not a directory: {working_dir}[/red]")
        raise typer.Exit(2)

    try:
        asyncio.run(
            _run_once(
                prompt=prompt,
                model=effective_model,
                provider=effective_provider,
                base_url=base_url,
                cwd=working_dir,
                max_steps=max_steps,
                session_id=session_id,
                db=db,
                in_memory=in_memory,
                yes=yes,
                config=cfg,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user.[/yellow]")
        raise typer.Exit(130) from None


async def _run_once(
    *,
    prompt: str,
    model: str,
    provider: str,
    base_url: str | None,
    cwd: Path,
    max_steps: int,
    session_id: str | None,
    db: Path | None,
    in_memory: bool,
    yes: bool,
    config: HarnessConfig,
) -> None:
    storage = _build_storage(db=db, in_memory=in_memory)
    try:
        agent = _build_agent(
            provider=provider,
            base_url=base_url,
            model=model,
            storage=storage,
            cwd=cwd,
            config=config,
            yes=yes,
        )

        request_kwargs: dict[str, object] = {
            "prompt": prompt,
            "model": model,
            "max_steps": max_steps,
        }
        if session_id:
            request_kwargs["session_id"] = session_id
        request = RunRequest(**request_kwargs)  # type: ignore[arg-type]

        try:
            async for event in agent.run(request):
                _render(event)
        except Exception as exc:
            console.print(f"\n[red]Unhandled error:[/red] {exc!s}")
            raise typer.Exit(1) from None
    finally:
        if isinstance(storage, SQLiteStorage):
            await storage.close()


# ---------------------------------------------------------------------------
# sessions subcommands
# ---------------------------------------------------------------------------


@sessions_app.command("list")
def sessions_list(
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
    limit: Annotated[int, typer.Option("--limit", "-n", help="Max sessions to show.")] = 25,
    status: Annotated[
        str | None,
        typer.Option(
            "--status",
            help="Filter: pending | running | paused | done | failed | cancelled.",
        ),
    ] = None,
) -> None:
    """List saved sessions, newest first."""

    async def _go() -> None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            sessions = await storage.list(limit=limit, status=status)  # type: ignore[arg-type]
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

        if not sessions:
            console.print("[dim]No sessions.[/dim]")
            return

        table = Table(show_header=True, header_style="bold")
        table.add_column("ID")
        table.add_column("Status")
        table.add_column("Provider")
        table.add_column("Model")
        table.add_column("Updated")
        table.add_column("Turns", justify="right")
        for s in sessions:
            table.add_row(
                s.id,
                _status_style(s.status),
                s.provider,
                s.model,
                _ago(s.updated_at),
                str(len(s.messages)),
            )
        console.print(table)

    asyncio.run(_go())


@sessions_app.command("show")
def sessions_show(
    session_id: Annotated[str, typer.Argument(help="Session id to display.")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
) -> None:
    """Print a session's full transcript."""

    async def _go() -> None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            session = await storage.get(session_id)
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

        if session is None:
            console.print(f"[red]Session not found:[/red] {session_id}")
            raise typer.Exit(1)
        _render_session(session)

    asyncio.run(_go())


@sessions_app.command("resume")
def sessions_resume(
    session_id: Annotated[str, typer.Argument(help="Session id to continue.")],
    prompt: Annotated[
        str | None,
        typer.Argument(help="New user prompt. Omit to continue without new input."),
    ] = None,
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
    cwd: Annotated[
        Path | None,
        typer.Option("--cwd", help="Working directory for filesystem tools."),
    ] = None,
    base_url: Annotated[str | None, typer.Option("--base-url")] = None,
    max_steps: Annotated[int, typer.Option("--max-steps")] = 25,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Auto-approve all tool calls.")] = False,
    config_path: Annotated[Path | None, typer.Option("--config")] = None,
    verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """Continue a saved session, optionally with a new user prompt."""
    configure_logging(level="DEBUG" if verbose else "INFO")

    cfg = _load_cli_config(config_path)

    async def _go() -> None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            session = await storage.get(session_id)
            if session is None:
                console.print(f"[red]Session not found:[/red] {session_id}")
                raise typer.Exit(1)

            working_dir = (cwd or session.cwd).resolve()
            agent = _build_agent(
                provider=session.provider,
                base_url=base_url,
                model=session.model,
                storage=storage,
                cwd=working_dir,
                config=cfg,
                yes=yes,
            )
            try:
                async for event in agent.resume(session_id, prompt=prompt, max_steps=max_steps):
                    _render(event)
            except Exception as exc:
                console.print(f"\n[red]Unhandled error:[/red] {exc!s}")
                raise typer.Exit(1) from None
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    asyncio.run(_go())


@sessions_app.command("rm")
def sessions_rm(
    session_id: Annotated[str, typer.Argument(help="Session id to delete.")],
    db: Annotated[Path | None, typer.Option("--db")] = None,
    in_memory: Annotated[bool, typer.Option("--in-memory")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation.")] = False,
) -> None:
    """Delete a saved session."""
    if not yes and not Confirm.ask(f"Delete session [bold]{session_id}[/bold]?", default=False):
        console.print("[yellow]Aborted.[/yellow]")
        raise typer.Exit(1)

    async def _go() -> None:
        storage = _build_storage(db=db, in_memory=in_memory)
        try:
            await storage.delete(session_id)
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    asyncio.run(_go())
    console.print(f"[green]Deleted[/green] {session_id}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render(event: Any) -> None:
    if isinstance(event, TextDelta):
        console.out(event.text, end="", style=None, highlight=False)
    elif isinstance(event, ToolCallEvent):
        console.print()
        console.print(
            f"[blue]→[/blue] [bold]{event.call.name}[/bold]({_args_preview(event.call.arguments)})",
            style="dim",
        )
    elif isinstance(event, ToolResultEvent):
        marker = "[red]✗[/red]" if event.result.is_error else "[green]✓[/green]"
        preview = _truncate(event.result.content, 200)
        console.print(f"{marker} {event.result.name}: {preview}", style="dim")
    elif isinstance(event, ErrorEvent):
        console.print()
        console.print(f"[red]Error ({event.kind}):[/red] {event.error}")
    elif isinstance(event, Done):
        console.print()
    elif isinstance(event, StepStarted | StepCompleted):
        pass


def _render_session(session: Session) -> None:
    header = (
        f"[bold]{session.id}[/bold]  "
        f"{_status_style(session.status)}  "
        f"{session.provider}/{session.model}\n"
        f"[dim]created {_ago(session.created_at)}, updated {_ago(session.updated_at)}[/dim]\n"
        f"[dim]cwd: {session.cwd}[/dim]"
    )
    console.print(Panel(header, title="session", expand=False))

    for msg in session.messages:
        if msg.role == "user":
            console.print(Panel(msg.content or "", title="[cyan]user[/cyan]", expand=False))
        elif msg.role == "system":
            console.print(Panel(msg.content or "", title="[grey]system[/grey]", expand=False))
        elif msg.role == "assistant":
            parts: list[str] = []
            if msg.content:
                parts.append(msg.content)
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    parts.append(
                        f"→ {tc.name}({_args_preview(tc.arguments)})  [dim]({tc.id})[/dim]"
                    )
            console.print(
                Panel(
                    "\n".join(parts) or "[dim](empty turn)[/dim]",
                    title="[green]assistant[/green]",
                    expand=False,
                )
            )
        elif msg.role == "tool":
            console.print(
                Panel(
                    msg.content or "",
                    title=f"[yellow]tool: {msg.name}[/yellow]  [dim]({msg.tool_call_id})[/dim]",
                    expand=False,
                )
            )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


_STATUS_STYLES = {
    "pending": "white",
    "running": "blue",
    "paused": "yellow",
    "done": "green",
    "failed": "red",
    "cancelled": "magenta",
}


def _status_style(status: str) -> str:
    color = _STATUS_STYLES.get(status, "white")
    return f"[{color}]{status}[/{color}]"


def _ago(dt: datetime) -> str:
    now = datetime.now(UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"


def _args_preview(args: dict) -> str:
    if not args:
        return ""
    parts = [f"{k}={_truncate(repr(v), 40)}" for k, v in args.items()]
    return ", ".join(parts)


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


if __name__ == "__main__":
    app()
