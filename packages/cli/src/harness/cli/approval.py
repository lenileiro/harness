"""Rich-based approval handler for the CLI.

Shown whenever a tool's effective approval is `prompt`. Choices:

  y  approve this call
  n  deny this call
  s  trust this tool for the rest of the session
  a  approve from now on (writes "auto" into session.approval_overrides)
  d  deny from now on (writes "deny" into session.approval_overrides)

The runtime saves the session after every turn, so `a` / `d` persist across
resumes of the same session. `s` writes into a transient overrides dict that
lasts only for the current process.

When stdin is not a TTY (scripts, CI), the handler returns False rather than
hanging. Pass `--yes` to the CLI to skip prompts entirely (AutoApprove).
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from harness.core import ApprovalDecision, Session, Tool, ToolCall

_HIGH_RISK = {"shell", "write_file", "edit_file"}


def _tool_risk(name: str) -> tuple[str, str]:
    """Return (border_style, icon) based on tool risk level."""
    if name in _HIGH_RISK:
        return ("yellow", "⚠")
    return ("blue", "○")


def _format_args_full(args: dict) -> str:
    if not args:
        return "[dim](no arguments)[/dim]"
    return "\n".join(f"[dim]{k}:[/dim]  {v}" for k, v in args.items())


class RichApprovalHandler:
    """Interactive approval prompt rendered with Rich.

    Pass ``session_overrides`` to support the ``s`` (trust-session) choice.
    The dict is mutated in-place so the caller can observe which tools were
    trusted without needing a callback.
    """

    def __init__(
        self,
        console: Console | None = None,
        session_overrides: dict[str, ApprovalDecision] | None = None,
    ) -> None:
        self.console = console or Console()
        self._session_overrides = session_overrides if session_overrides is not None else {}

    async def __call__(self, tool: Tool, call: ToolCall, session: Session) -> bool:
        # Non-interactive: refuse rather than hang.
        if not sys.stdin.isatty():
            self.console.print(
                f"[red]Tool {tool.name!r} requires approval but stdin is not a TTY. "
                "Pass --yes to auto-approve.[/red]"
            )
            return False

        # Check transient session overrides first.
        if self._session_overrides.get(tool.name) == "auto":
            return True

        border_style, icon = _tool_risk(tool.name)
        body = (
            f"[bold]{icon} {tool.name}[/bold]\n\n"
            + _format_args_full(call.arguments)
            + f"\n\n[dim]{tool.description}[/dim]"
        )
        self.console.print(
            Panel(body, title=f"[{border_style}]approve?[/{border_style}]", expand=False)
        )

        choice = Prompt.ask(
            "[y]es  [n]o  [s]ession-trust  [a]lways  [d]eny-always",
            choices=["y", "n", "s", "a", "d"],
            default="n",
        )

        if choice == "s":
            self._session_overrides[tool.name] = "auto"
            return True
        if choice == "a":
            session.approval_overrides[tool.name] = "auto"
            return True
        if choice == "d":
            session.approval_overrides[tool.name] = "deny"
            return False
        return choice == "y"


__all__ = ["RichApprovalHandler"]
