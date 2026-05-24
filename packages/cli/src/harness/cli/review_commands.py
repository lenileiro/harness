from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from harness.cli.config import HarnessConfig
from harness.core import ReviewReport, parse_review_report


def build_review_prompt(*, base: str, diff: str) -> str:
    return (
        f"Review the current git diff against base {base}.\n\n"
        "Focus on correctness bugs, regressions, unsafe assumptions, and missing tests. "
        "Ignore style-only nits.\n\n"
        "Use repository tools to inspect changed files when needed.\n\n"
        "--- BEGIN DIFF ---\n"
        f"{diff.strip()}\n"
        "--- END DIFF ---\n"
    )


def _git_diff(cwd: Path, *, base: str) -> str:
    result = subprocess.run(
        ["git", "diff", f"{base}...HEAD"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git diff failed")
    return result.stdout


def _render_review_report(report: ReviewReport, *, console: Console) -> None:
    if report.summary:
        console.print(report.summary)
    if not report.findings:
        console.print("[green]No findings.[/green]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Severity", no_wrap=True)
    table.add_column("File", no_wrap=True)
    table.add_column("Line", no_wrap=True)
    table.add_column("Issue", overflow="fold")
    table.add_column("Rationale", overflow="fold")
    for finding in report.findings:
        table.add_row(
            finding.severity,
            finding.file,
            str(finding.line) if finding.line is not None else "—",
            finding.issue,
            finding.rationale,
        )
    console.print(table)


def review_command(
    *,
    base: str,
    model: str | None,
    provider: str | None,
    failover: str | None,
    base_url: str | None,
    cwd: Path | None,
    max_steps: int,
    max_output_tokens: int | None,
    db: Path | None,
    in_memory: bool,
    yes: bool,
    verbose: bool,
    json_output: bool,
    config_path: Path | None,
    console: Console,
    load_cli_config: Any,
    resolve_chain: Any,
    run_async: Any,
    run_once: Any,
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    if not working_dir.exists() or not working_dir.is_dir():
        console.print(f"[red]--cwd does not exist or is not a directory: {working_dir}[/red]")
        raise typer.Exit(2)

    cfg: HarnessConfig = load_cli_config(config_path)
    chain = resolve_chain(failover_flag=failover, provider_flag=provider, config=cfg)
    effective_model = model or cfg.default_model or "llama3.2"

    try:
        diff = _git_diff(working_dir, base=base)
    except Exception as exc:
        console.print(f"[red]Could not load git diff:[/red] {exc}")
        raise typer.Exit(2) from None
    if not diff.strip():
        console.print("[yellow]No diff to review.[/yellow]")
        raise typer.Exit(1)

    prompt = build_review_prompt(base=base, diff=diff)
    final_text = run_async(
        run_once(
            prompt=prompt,
            model=effective_model,
            chain=chain,
            base_url=base_url,
            cwd=working_dir,
            max_steps=max_steps,
            max_output_tokens=max_output_tokens,
            session_id=None,
            task_ref=None,
            db=db,
            in_memory=in_memory,
            yes=yes,
            inbox=False,
            verify="none",
            verify_command=None,
            critic=None,
            require_tools=True,
            goal=False,
            max_context_tokens=None,
            predict=False,
            auto_compact=False,
            max_repair=1,
            profile="bare",
            domain="code-review",
            phases=None,
            loop_detect=False,
            contracts=False,
            tips=True,
            silent=json_output,
            config=cfg,
        )
    )

    parsed = parse_review_report(final_text or "")
    if json_output:
        if parsed is not None:
            console.print(json.dumps(parsed.to_dict(), indent=2))
        else:
            console.print(final_text or "")
        return
    if parsed is not None:
        _render_review_report(parsed, console=console)
    else:
        console.print(final_text or "")


__all__ = ["build_review_prompt", "review_command"]
