from __future__ import annotations

import re
import time
from typing import Any

import unicodeitplus as _unicodeit
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.spinner import Spinner

from harness.cli.common import _args_preview, _truncate
from harness.core import (
    Critique,
    Done,
    ErrorEvent,
    PhaseCompletedEvent,
    PhaseStartedEvent,
    PredictionEvent,
    PredictionMismatchEvent,
    StepCompleted,
    StepStarted,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    Verification,
)

_DOLLAR = re.escape("$")
_MATH_DISPLAY = re.compile(_DOLLAR * 2 + r"(.+?)" + _DOLLAR * 2, re.DOTALL)
_MATH_INLINE = re.compile(_DOLLAR + r"([^\n]+?)" + _DOLLAR)
_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_MERMAID_FENCE = re.compile(r"```mermaid\n(.*?)\n```", re.DOTALL)

_mermaid_render_cache: dict[str, str] = {}


def _render_mermaid(source: str) -> str:
    """Convert mermaid source to an ASCII-art fenced block."""
    if source in _mermaid_render_cache:
        return _mermaid_render_cache[source]

    ascii_art: str | None = None
    try:
        from mermaid_ascii import mermaid_to_ascii  # type: ignore[import-untyped]

        result = mermaid_to_ascii(source)
        if result and result.strip():
            ascii_art = result.strip()
    except ImportError:
        import shutil
        import subprocess

        if shutil.which("mermaid-ascii"):
            try:
                proc = subprocess.run(
                    ["mermaid-ascii", "-i", "-"],
                    input=source,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if proc.returncode == 0 and proc.stdout.strip():
                    ascii_art = proc.stdout.strip()
            except Exception:
                pass
    except Exception:
        pass

    rendered = f"```\n{ascii_art}\n```" if ascii_art else f"```\n{source}\n```"
    _mermaid_render_cache[source] = rendered
    return rendered


def _convert_math(inner: str) -> str:
    return _unicodeit.replace(inner)


def _preprocess_markdown(text: str) -> str:
    """Prepare LLM output for Rich Markdown rendering."""
    text = _MERMAID_FENCE.sub(lambda m: _render_mermaid(m.group(1)), text)
    text = _MATH_DISPLAY.sub(lambda m: f"`{_convert_math(m.group(1))}`", text)
    text = _MATH_INLINE.sub(lambda m: _convert_math(m.group(1)), text)
    text = _THINK_BLOCK.sub(
        lambda m: "> *thinking: " + m.group(1).strip().replace("\n", " ")[:200] + "…*\n",
        text,
    )
    return text


class Renderer:
    """Stateful event renderer for streaming CLI output."""

    def __init__(self, con: Console) -> None:
        self._console = con
        self._live: Live | None = None
        self._live_kind: str = ""
        self._pending_start: float = 0.0
        self._text_buf: str = ""

    def render(self, event: Any) -> None:
        if isinstance(event, TextDelta):
            self._stop_spinner()
            self._text_buf += event.text
            rendered = Markdown(_preprocess_markdown(self._text_buf))
            if self._live is None:
                self._live = Live(
                    rendered,
                    console=self._console,
                    refresh_per_second=12,
                    vertical_overflow="visible",
                )
                self._live_kind = "text"
                self._live.start()
            else:
                self._live.update(rendered)
        elif isinstance(event, ToolCallEvent):
            self._flush_text()
            self._console.print()
            self._console.print(
                f"[blue]→[/blue] [bold]{event.call.name}[/bold]({_args_preview(event.call.arguments)})",
                style="dim",
            )
            self._start_spinner(event.call.name)
        elif isinstance(event, ToolResultEvent):
            elapsed = time.monotonic() - self._pending_start if self._pending_start else 0.0
            self._stop_spinner()
            partial = bool(event.result.metadata and event.result.metadata.get("partial"))
            marker = (
                "[cyan]…[/cyan]"
                if partial
                else "[red]✗[/red]"
                if event.result.is_error
                else "[green]✓[/green]"
            )
            full_len = len(event.result.content)
            preview = _truncate(event.result.content, 200)
            suffix = f"  [dim]… {full_len:,} bytes[/dim]" if full_len > 200 else ""
            self._console.print(
                f"{marker} {event.result.name}: {preview}{suffix}  [dim]({elapsed:.1f}s)[/dim]",
                style="dim",
            )
        elif isinstance(event, StepStarted):
            self._flush_text()
            if event.total_steps > 1:
                label = f"Step {event.step + 1}/{event.total_steps}"
                if event.description:
                    label += f": {event.description}"
                self._console.print(f"\n[bold blue]●[/bold blue] {label}")
        elif isinstance(event, StepCompleted):
            pass
        elif isinstance(event, ErrorEvent):
            self._stop_spinner()
            self._flush_text()
            self._console.print()
            self._console.print(f"[red]Error ({event.kind}):[/red] {event.error}")
        elif isinstance(event, Verification):
            self._flush_text()
            r = event.result
            marker = "[green]✓[/green]" if r.can_finish else "[red]✗[/red]"
            conf = (
                f"  [dim](confidence {r.confidence:.2f})[/dim]" if r.confidence is not None else ""
            )
            self._console.print()
            self._console.print(
                f"{marker} [bold]verify[/bold] ({r.verifier_name})  {r.reason}{conf}"
            )
        elif isinstance(event, Critique):
            self._flush_text()
            self._console.print()
            self._console.print(
                f"[yellow bold]critic[/yellow bold] [dim](attempt {event.attempt})[/dim]"
            )
            for line in event.text.splitlines():
                self._console.print(f"  [yellow]{line}[/yellow]")
            self._console.print()
        elif isinstance(event, PhaseStartedEvent):
            self._flush_text()
            position = f" {event.index + 1}/{event.total}" if event.total > 1 else ""
            note = f"  [dim]{event.notes}[/dim]" if event.notes else ""
            self._console.print(f"[cyan]▶ phase{position}: {event.name}[/cyan]{note}")
        elif isinstance(event, PhaseCompletedEvent):
            self._flush_text()
            position = f" {event.index + 1}/{event.total}" if event.total > 1 else ""
            note = f"  [dim]{event.notes}[/dim]" if event.notes else ""
            self._console.print(f"[green]✓ phase{position}: {event.name}[/green]{note}")
        elif isinstance(event, PredictionEvent):
            p = event.prediction
            scope = p.effect_scope or "unknown"
            self._console.print(
                f"[dim]  ⟳ predict scope={scope} confidence={p.confidence:.2f} "
                f"expected={p.expected_status} reversibility={p.reversibility}[/dim]"
            )
        elif isinstance(event, PredictionMismatchEvent):
            o = event.outcome
            self._console.print(
                f"[yellow]  ⚠ mismatch severity={o.severity} actual={o.actual_status} "
                f"lesson={o.lesson}[/yellow]"
            )
        elif isinstance(event, Done):
            self._stop_spinner()
            self._flush_text()
            self._console.print()
            if event.usage:
                u = event.usage
                self._console.print(
                    f"[dim]tokens: {u.prompt_tokens:,} in / {u.completion_tokens:,} out[/dim]"
                )

    def _flush_text(self) -> None:
        if self._live is not None and self._live_kind == "text":
            self._live.stop()
            self._live = None
            self._live_kind = ""
        self._text_buf = ""

    def _start_spinner(self, name: str) -> None:
        self._pending_start = time.monotonic()
        self._live = Live(
            Spinner("dots", text=f"[dim]{name}[/dim]"),
            console=self._console,
            refresh_per_second=10,
            transient=True,
        )
        self._live_kind = "spinner"
        self._live.start()

    def _stop_spinner(self) -> None:
        if self._live is not None and self._live_kind == "spinner":
            self._live.stop()
            self._live = None
            self._live_kind = ""
        self._pending_start = 0.0


__all__ = ["Renderer", "_preprocess_markdown", "_render_mermaid"]
