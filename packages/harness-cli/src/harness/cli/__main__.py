"""Harness CLI entry point.

Phase 2 surface: `harness run "prompt"` drives a real Ollama-backed agent
with the read_file tool registered. Sessions, REPL, providers, tools list
commands arrive in later phases.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from harness.adapters.ollama import OllamaAdapter
from harness.core import (
    Agent,
    ApprovalPolicy,
    AutoApprove,
    Done,
    ErrorEvent,
    FailoverPolicy,
    RunRequest,
    StepCompleted,
    StepStarted,
    TextDelta,
    ToolCallEvent,
    ToolRegistry,
    ToolResultEvent,
    configure_logging,
)
from harness.storage.memory import InMemoryStorage
from harness.tools.fs import ReadFileTool

app = typer.Typer(
    name="harness",
    help="Harness — Python agent runtime over OpenRouter and Ollama.",
    no_args_is_help=True,
    add_completion=False,
)

console = Console()


@app.command()
def version() -> None:
    """Print the installed harness-cli version."""
    from harness.cli import __version__

    typer.echo(__version__)


@app.command()
def run(
    prompt: str = typer.Argument(..., help="The user prompt for the agent."),
    model: str = typer.Option(
        "llama3.2", "--model", "-m", help="Model name to send to the provider."
    ),
    provider: str = typer.Option(
        "ollama", "--provider", "-p", help="Provider name (only 'ollama' is wired up in Phase 2)."
    ),
    base_url: str | None = typer.Option(
        None,
        "--base-url",
        help="Override the Ollama base URL (default: $OLLAMA_HOST or http://localhost:11434).",
    ),
    cwd: Path | None = typer.Option(
        None, "--cwd", help="Working directory for filesystem tools (default: current dir)."
    ),
    max_steps: int = typer.Option(25, "--max-steps", help="Maximum ReAct turns before giving up."),
    session_id: str | None = typer.Option(
        None, "--session", help="Reuse an existing session id (in-memory only in Phase 2)."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG logging to stderr."),
) -> None:
    """Run a single prompt through the agent and stream the result to stdout.

    In Phase 2 the only provider is `ollama` (talks to a local daemon) and
    the only tool is `read_file`. SQLite persistence + OpenRouter + more
    tools arrive in Phase 3 and later.
    """
    if verbose:
        configure_logging(level="DEBUG")
    else:
        configure_logging(level="INFO")

    if provider != "ollama":
        console.print(f"[red]Provider {provider!r} is not wired up in this phase.[/red]")
        raise typer.Exit(2)

    working_dir = (cwd or Path.cwd()).resolve()
    if not working_dir.exists() or not working_dir.is_dir():
        console.print(f"[red]--cwd does not exist or is not a directory: {working_dir}[/red]")
        raise typer.Exit(2)

    try:
        asyncio.run(
            _run_main(
                prompt=prompt,
                model=model,
                base_url=base_url,
                cwd=working_dir,
                max_steps=max_steps,
                session_id=session_id,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user.[/yellow]")
        raise typer.Exit(130) from None


async def _run_main(
    *,
    prompt: str,
    model: str,
    base_url: str | None,
    cwd: Path,
    max_steps: int,
    session_id: str | None,
) -> None:
    adapter = OllamaAdapter(base_url=base_url) if base_url else OllamaAdapter()
    storage = InMemoryStorage()
    registry = ToolRegistry()
    registry.register(ReadFileTool(cwd=cwd))

    agent = Agent(
        adapters={"ollama": adapter},
        tools=registry,
        storage=storage,
        failover=FailoverPolicy(
            chain=["ollama"], max_attempts=1, backoff_base=0.0, backoff_jitter=0.0
        ),
        approval_policy=ApprovalPolicy(default="auto"),
        approval_handler=AutoApprove(),
        default_model=model,
        default_cwd=str(cwd),
    )

    request_kwargs: dict[str, object] = {"prompt": prompt, "model": model, "max_steps": max_steps}
    if session_id:
        request_kwargs["session_id"] = session_id
    request = RunRequest(**request_kwargs)  # type: ignore[arg-type]

    exit_code = 0
    try:
        async for event in agent.run(request):
            _render(event)
    except Exception as exc:
        console.print(f"\n[red]Unhandled error:[/red] {exc!s}")
        exit_code = 1

    if exit_code:
        raise typer.Exit(exit_code)


def _render(event: object) -> None:
    """Render a single Event onto the console."""
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
        console.print()  # newline after the final streamed text
    elif isinstance(event, StepStarted | StepCompleted):
        pass  # Silent in v1; only the inner stream is interesting to humans.


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
