"""Rich-based approval handler for the CLI.

Shown whenever a tool's effective approval is `prompt`. Choices:

  y  approve this call
  n  deny this call
  a  approve from now on (writes "auto" into session.approval_overrides)
  d  deny from now on (writes "deny" into session.approval_overrides)

The runtime saves the session after every turn, so `a` / `d` persist across
resumes of the same session.

When stdin is not a TTY (scripts, CI), the handler returns False rather than
hanging. Pass `--yes` to the CLI to skip prompts entirely (AutoApprove).
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from harness.core import Session, Tool, ToolCall


def _args_preview(args: dict, limit: int = 200) -> str:
    if not args:
        return ""
    pretty = ", ".join(f"{k}={v!r}" for k, v in args.items())
    if len(pretty) <= limit:
        return pretty
    return pretty[: limit - 1] + "…"


class RichApprovalHandler:
    """Interactive y/n/a/d approval prompt rendered with Rich."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    async def __call__(self, tool: Tool, call: ToolCall, session: Session) -> bool:
        # Non-interactive: refuse rather than hang.
        if not sys.stdin.isatty():
            self.console.print(
                f"[red]Tool {tool.name!r} requires approval but stdin is not a TTY. "
                "Pass --yes to auto-approve.[/red]"
            )
            return False

        panel = Panel(
            f"[bold]{tool.name}[/bold]({_args_preview(call.arguments)})\n\n"
            f"[dim]{tool.description}[/dim]",
            title="[yellow]Tool approval needed[/yellow]",
            expand=False,
        )
        self.console.print(panel)

        choice = Prompt.ask(
            "Approve?  [y]es  [n]o  [a]lways  [d]eny-always",
            choices=["y", "n", "a", "d"],
            default="n",
        )

        if choice == "a":
            session.approval_overrides[tool.name] = "auto"
            return True
        if choice == "d":
            session.approval_overrides[tool.name] = "deny"
            return False
        return choice == "y"


__all__ = ["RichApprovalHandler"]
