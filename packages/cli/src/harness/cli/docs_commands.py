from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from harness.cli.config import HarnessConfig
from harness.core import DocsAuditReport, parse_docs_audit_report


def build_docs_audit_prompt(*, focus: str | None = None) -> str:
    base = (
        "Audit the repository documentation.\n\n"
        "Focus on stale, missing, or unclear docs that would materially hurt onboarding, safe use, "
        "or maintenance. Use repository files and web tools when they help.\n\n"
        "Return JSON only in the requested docs-audit report shape.\n"
    )
    if focus and focus.strip():
        return base + f"\nAdditional focus: {focus.strip()}\n"
    return base


def _render_docs_audit(report: DocsAuditReport, *, console: Console) -> None:
    if report.summary:
        console.print(report.summary)
    if report.findings:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Severity", no_wrap=True)
        table.add_column("Path", no_wrap=True)
        table.add_column("Issue", overflow="fold")
        table.add_column("Rationale", overflow="fold")
        for finding in report.findings:
            table.add_row(
                finding.severity,
                finding.path or "—",
                finding.issue,
                finding.rationale,
            )
        console.print("\n[bold]Findings[/bold]")
        console.print(table)
    if report.missing_topics:
        console.print("\n[bold]Missing Topics[/bold]")
        for topic in report.missing_topics:
            console.print(f"- {topic}")


def docs_audit_command(
    *,
    focus: str | None,
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

    prompt = build_docs_audit_prompt(focus=focus)
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
            domain="docs-audit",
            phases=None,
            loop_detect=False,
            contracts=False,
            tips=True,
            silent=json_output,
            config=cfg,
        )
    )

    parsed = parse_docs_audit_report(final_text or "")
    if json_output:
        if parsed is not None:
            console.print(json.dumps(parsed.to_dict(), indent=2))
        else:
            console.print(final_text or "")
        return
    if parsed is not None:
        _render_docs_audit(parsed, console=console)
    else:
        console.print(final_text or "")


__all__ = ["build_docs_audit_prompt", "docs_audit_command"]
