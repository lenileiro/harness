"""Harness CLI entry point.

This is a Phase 0 stub. Real commands land in Phase 2+.
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="harness",
    help="Harness — Python agent runtime over OpenRouter and Ollama.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def version() -> None:
    """Print the installed harness-cli version."""
    from harness.cli import __version__

    typer.echo(__version__)


@app.command()
def info() -> None:
    """Print scaffolding status. Real commands land in Phase 2+."""
    typer.echo("harness: scaffolding phase. Run `harness version` for the build version.")


if __name__ == "__main__":
    app()
