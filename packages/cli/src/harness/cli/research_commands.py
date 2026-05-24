from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from harness.cli.config import HarnessConfig
from harness.core import ResearchMemo, parse_research_memo


def build_research_prompt(*, topic: str) -> str:
    return (
        "Research the following topic.\n\n"
        f"Topic: {topic.strip()}\n\n"
        "Use repository files first and only gather the minimum evidence needed. "
        "If the repository already contains enough evidence, do not browse the web. "
        "Once you have enough evidence, stop and return the final memo.\n\n"
        "Focus on the most decision-useful findings, key tradeoffs, and open questions.\n\n"
        "Return JSON only in the requested research memo shape.\n"
    )


def _render_research_memo(memo: ResearchMemo, *, console: Console) -> None:
    if memo.summary:
        console.print(memo.summary)
    if memo.findings:
        console.print("\n[bold]Findings[/bold]")
        for finding in memo.findings:
            console.print(f"- {finding}")
    if memo.open_questions:
        console.print("\n[bold]Open Questions[/bold]")
        for question in memo.open_questions:
            console.print(f"- {question}")
    if memo.sources:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Title")
        table.add_column("URL")
        table.add_column("Excerpt", overflow="fold")
        for source in memo.sources:
            table.add_row(source.title, source.url, source.excerpt or "—")
        console.print("\n[bold]Sources[/bold]")
        console.print(table)


def research_command(
    *,
    topic: str,
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

    prompt = build_research_prompt(topic=topic)
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
            require_tools=False,
            goal=False,
            max_context_tokens=None,
            predict=False,
            auto_compact=False,
            max_repair=1,
            profile="minimal",
            domain="research",
            phases=None,
            loop_detect=True,
            contracts=False,
            tips=True,
            silent=json_output,
            config=cfg,
        )
    )

    parsed = parse_research_memo(final_text or "")
    if json_output:
        if parsed is not None:
            console.print(json.dumps(parsed.to_dict(), indent=2))
        else:
            console.print(final_text or "")
        return
    if parsed is not None:
        _render_research_memo(parsed, console=console)
    else:
        console.print(final_text or "")


__all__ = ["build_research_prompt", "research_command"]
