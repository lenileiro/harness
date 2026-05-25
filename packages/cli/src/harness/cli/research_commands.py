from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from harness.cli.common import _load_cli_config, _resolve_chain, _run_async
from harness.cli.config import HarnessConfig, default_config_path
from harness.cli.promotion_commands import (
    pr_command as _pr_command,
)
from harness.cli.promotion_commands import (
    promote_command as _promote_command,
)
from harness.cli.promotion_commands import (
    show_candidate_command as _show_candidate_command,
)
from harness.core import ResearchMemo, parse_research_memo
from harness.core.autonomy import (
    execute_next_research_item,
    execute_research_burst,
    run_scheduled_research_burst,
)
from harness.core.citations import Citation
from harness.core.experiment_plans import ExperimentPlan
from harness.core.experiment_runner import compare_experiment_results, run_experiment_plan
from harness.core.hypotheses import Hypothesis
from harness.core.inspiration import ExternalSource, InspirationNote
from harness.core.observations import Observation
from harness.core.opportunities import Opportunity
from harness.core.portfolio import build_portfolio_snapshot
from harness.core.promotion_candidates import PromotionCandidate
from harness.core.publications import summarize_publication
from harness.core.research_models import Publication, RabbitHole, Theme, Unknown, Vision
from harness.core.research_roles import BUILTIN_RESEARCH_ROLES
from harness.core.research_scheduler import build_research_queue, rebalance_research_queue
from harness.core.research_store import ResearchStore, _split_csv, default_research_root
from harness.core.section_maps import SectionMap

console = Console()

vision_app = typer.Typer(name="vision", help="Manage the current research vision.")
research_app = typer.Typer(
    name="research",
    help="Run structured research or manage durable research artifacts.",
    no_args_is_help=True,
)
experiment_app = typer.Typer(name="experiment", help="Run and compare experiment plans.")
candidate_app = typer.Typer(name="candidate", help="Inspect promotion candidates.")
research_app.add_typer(experiment_app, name="experiment")
research_app.add_typer(candidate_app, name="candidate")


@vision_app.command("show")
def vision_show_command(
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        vision = store.load_vision()
    except FileNotFoundError:
        console.print("[dim]No current vision defined.[/dim]")
        return
    console.print(f"[bold]{vision.title}[/bold]")
    console.print(vision.summary)
    if vision.themes:
        console.print("\n[bold]Themes[/bold]")
        for item in vision.themes:
            console.print(f"- {item}")
    if vision.success_metrics:
        console.print("\n[bold]Success Metrics[/bold]")
        for item in vision.success_metrics:
            console.print(f"- {item}")


@vision_app.command("update")
def vision_update_command(
    *,
    title: str = typer.Option(..., "--title"),
    summary: str = typer.Option(..., "--summary"),
    theme: list[str] = typer.Option([], "--theme"),
    success_metric: list[str] = typer.Option([], "--success-metric"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        previous = store.load_vision()
        vision = Vision(
            id=previous.id,
            title=title.strip(),
            summary=summary.strip(),
            themes=tuple(item.strip() for item in theme if item.strip()),
            success_metrics=tuple(item.strip() for item in success_metric if item.strip()),
            created_at=previous.created_at,
        )
    except FileNotFoundError:
        vision = Vision(
            id="current",
            title=title.strip(),
            summary=summary.strip(),
            themes=tuple(item.strip() for item in theme if item.strip()),
            success_metrics=tuple(item.strip() for item in success_metric if item.strip()),
        )
    target = store.update_vision(vision)
    console.print(f"[green]Updated vision[/green] at {target}")


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


@research_app.command("run")
def research_run_command(
    topic: str = typer.Argument(..., help="Research topic or question."),
    model: str | None = typer.Option(None, "--model", "-m"),
    provider: str | None = typer.Option(None, "--provider", "-p"),
    failover: str | None = typer.Option(None, "--failover"),
    base_url: str | None = typer.Option(None, "--base-url"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    max_steps: int = typer.Option(20, "--max-steps"),
    max_output_tokens: int | None = typer.Option(None, "--max-output-tokens"),
    db: Path | None = typer.Option(None, "--db"),
    in_memory: bool = typer.Option(False, "--in-memory"),
    yes: bool = typer.Option(False, "--yes", "-y"),
    json_output: bool = typer.Option(False, "--json"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    config_path: Path | None = typer.Option(None, "--config"),
) -> None:
    from harness.cli.run_commands import run_once as _run_once

    research_command(
        topic=topic,
        model=model,
        provider=provider,
        failover=failover,
        base_url=base_url,
        cwd=cwd,
        max_steps=max_steps,
        max_output_tokens=max_output_tokens,
        db=db,
        in_memory=in_memory,
        yes=yes,
        verbose=verbose,
        json_output=json_output,
        config_path=config_path,
        console=console,
        load_cli_config=_load_cli_config,
        resolve_chain=_resolve_chain,
        run_async=_run_async,
        run_once=_run_once,
    )


@research_app.command("open")
def research_open_command(
    *,
    title: str = typer.Option(..., "--title"),
    question: str = typer.Option(..., "--question"),
    scope: str = typer.Option(..., "--scope"),
    theme: str = typer.Option(..., "--theme"),
    related_sections: str | None = typer.Option(None, "--related-sections"),
    tags: str | None = typer.Option(None, "--tags"),
    opened_by: str = typer.Option("human", "--opened-by"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    mode: str | None = typer.Option(None, "--mode"),
    subsystem: str | None = typer.Option(None, "--subsystem"),
    rationale: str | None = typer.Option(None, "--rationale"),
    expected_outcome: str | None = typer.Option(None, "--expected-outcome"),
    risk: str | None = typer.Option(None, "--risk"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        change_intent = store.parse_change_intent(
            mode=mode,
            subsystem=subsystem,
            rationale=rationale,
            expected_outcome=expected_outcome,
            risk=risk,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    rabbit_hole = RabbitHole(
        id=store.new_id("rh", title),
        title=title.strip(),
        question=question.strip(),
        scope=scope.strip(),
        theme=theme.strip(),
        related_sections=_split_csv(related_sections),
        tags=_split_csv(tags),
        opened_by=opened_by.strip() or "human",
        change_intent=change_intent,
    )
    target = store.add_rabbit_hole(rabbit_hole)
    console.print(f"[green]Opened rabbit hole {rabbit_hole.id}[/green] at {target}")


@research_app.command("publish")
def research_publish_command(
    *,
    rabbit_hole_id: str = typer.Option(..., "--rabbit-hole"),
    title: str = typer.Option(..., "--title"),
    summary: str = typer.Option(..., "--summary"),
    claim: list[str] = typer.Option([], "--claim"),
    evidence: list[str] = typer.Option([], "--evidence"),
    counterevidence: list[str] = typer.Option([], "--counterevidence"),
    recommendation: list[str] = typer.Option([], "--recommendation"),
    open_question: list[str] = typer.Option([], "--open-question"),
    source: list[str] = typer.Option([], "--source"),
    artifact: list[str] = typer.Option([], "--artifact"),
    citation: list[str] = typer.Option([], "--citation"),
    confidence: float = typer.Option(1.0, "--confidence"),
    status: str = typer.Option("exploratory", "--status"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    mode: str | None = typer.Option(None, "--mode"),
    subsystem: str | None = typer.Option(None, "--subsystem"),
    rationale: str | None = typer.Option(None, "--rationale"),
    expected_outcome: str | None = typer.Option(None, "--expected-outcome"),
    risk: str | None = typer.Option(None, "--risk"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        store.load_rabbit_hole(rabbit_hole_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown rabbit hole: {rabbit_hole_id!r}") from exc
    try:
        change_intent = store.parse_change_intent(
            mode=mode,
            subsystem=subsystem,
            rationale=rationale,
            expected_outcome=expected_outcome,
            risk=risk,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    publication = Publication(
        id=store.new_id("pub", title),
        rabbit_hole_id=rabbit_hole_id.strip(),
        title=title.strip(),
        summary=summary.strip(),
        claims=tuple(item.strip() for item in claim if item.strip()),
        supporting_evidence=tuple(item.strip() for item in evidence if item.strip()),
        counterevidence=tuple(item.strip() for item in counterevidence if item.strip()),
        recommendations=tuple(item.strip() for item in recommendation if item.strip()),
        open_questions=tuple(item.strip() for item in open_question if item.strip()),
        sources=tuple(item.strip() for item in source if item.strip()),
        artifacts=tuple(item.strip() for item in artifact if item.strip()),
        citations=tuple(item.strip() for item in citation if item.strip()),
        confidence=confidence,
        status=status,  # type: ignore[arg-type]
        change_intent=change_intent,
    )
    target = store.add_publication(publication)
    console.print(f"[green]Published {publication.id}[/green] at {target}")


@research_app.command("search")
def research_search_command(
    query: str = typer.Argument(..., help="Search text across rabbit holes and publications."),
    *,
    kind: str = typer.Option("all", "--kind"),
    limit: int = typer.Option(10, "--limit"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    if kind == "all":
        kinds = (
            "vision",
            "theme",
            "unknown",
            "rabbit_hole",
            "publication",
            "citation",
            "inspiration",
            "section_map",
            "observation",
            "opportunity",
            "hypothesis",
            "experiment_plan",
        )
    elif kind == "rabbit-hole":
        kinds = ("rabbit_hole",)
    elif kind == "vision":
        kinds = ("vision",)
    elif kind == "theme":
        kinds = ("theme",)
    elif kind == "unknown":
        kinds = ("unknown",)
    elif kind == "publication":
        kinds = ("publication",)
    elif kind == "citation":
        kinds = ("citation",)
    elif kind == "inspiration":
        kinds = ("inspiration",)
    elif kind == "section-map":
        kinds = ("section_map",)
    elif kind == "observation":
        kinds = ("observation",)
    elif kind == "opportunity":
        kinds = ("opportunity",)
    elif kind == "hypothesis":
        kinds = ("hypothesis",)
    elif kind == "experiment-plan":
        kinds = ("experiment_plan",)
    else:
        raise typer.BadParameter(
            "--kind must be all, vision, theme, unknown, rabbit-hole, publication, citation, "
            "inspiration, section-map, observation, opportunity, hypothesis, or experiment-plan"
        )
    hits = store.search(query, kinds=kinds, limit=limit)
    if not hits:
        console.print("[dim]No research results found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Kind", no_wrap=True)
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Title")
    table.add_column("Summary", overflow="fold")
    table.add_column("Path", overflow="fold")
    for hit in hits:
        table.add_row(hit.kind, hit.id, hit.title, hit.summary, str(hit.path))
    console.print(table)


@research_app.command("show-publication")
def research_show_publication_command(
    publication_id: str = typer.Argument(..., help="Publication id."),
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        publication = store.load_publication(publication_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown publication: {publication_id!r}") from exc
    for line in summarize_publication(publication):
        console.print(line)


@research_app.command("cite")
def research_cite_command(
    *,
    source_publication: str = typer.Option(..., "--source-publication"),
    target_publication: str = typer.Option(..., "--target-publication"),
    relationship: str = typer.Option(..., "--relationship"),
    note: str = typer.Option("", "--note"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        store.load_publication(source_publication)
        store.load_publication(target_publication)
    except FileNotFoundError as exc:
        raise typer.BadParameter(str(exc)) from exc
    citation = Citation(
        id=store.new_id("cite", f"{source_publication}-{target_publication}"),
        source_publication_id=source_publication,
        target_publication_id=target_publication,
        relationship=relationship,  # type: ignore[arg-type]
        note=note.strip(),
    )
    target = store.add_citation(citation)
    console.print(f"[green]Recorded citation {citation.id}[/green] at {target}")


@research_app.command("ingest-web")
def research_ingest_web_command(
    *,
    title: str = typer.Option(..., "--title"),
    url: str = typer.Option(..., "--url"),
    summary: str = typer.Option(..., "--summary"),
    excerpt: str = typer.Option("", "--excerpt"),
    themes: str | None = typer.Option(None, "--themes"),
    related_sections: str | None = typer.Option(None, "--related-sections"),
    created_by: str = typer.Option("human", "--created-by"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    _ingest_inspiration(
        source_kind="web",
        source_ref=url,
        source_title=title,
        title=title,
        summary=summary,
        excerpt=excerpt,
        themes=themes,
        related_sections=related_sections,
        created_by=created_by,
        cwd=cwd,
    )


@research_app.command("ingest-paper")
def research_ingest_paper_command(
    *,
    title: str = typer.Option(..., "--title"),
    citation: str = typer.Option(..., "--citation"),
    summary: str = typer.Option(..., "--summary"),
    excerpt: str = typer.Option("", "--excerpt"),
    themes: str | None = typer.Option(None, "--themes"),
    related_sections: str | None = typer.Option(None, "--related-sections"),
    created_by: str = typer.Option("human", "--created-by"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    _ingest_inspiration(
        source_kind="paper",
        source_ref=citation,
        source_title=title,
        title=title,
        summary=summary,
        excerpt=excerpt,
        themes=themes,
        related_sections=related_sections,
        created_by=created_by,
        cwd=cwd,
    )


@research_app.command("ingest-repo")
def research_ingest_repo_command(
    *,
    title: str = typer.Option(..., "--title"),
    repo: str = typer.Option(..., "--repo"),
    summary: str = typer.Option(..., "--summary"),
    excerpt: str = typer.Option("", "--excerpt"),
    themes: str | None = typer.Option(None, "--themes"),
    related_sections: str | None = typer.Option(None, "--related-sections"),
    created_by: str = typer.Option("human", "--created-by"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    _ingest_inspiration(
        source_kind="repo",
        source_ref=repo,
        source_title=title,
        title=title,
        summary=summary,
        excerpt=excerpt,
        themes=themes,
        related_sections=related_sections,
        created_by=created_by,
        cwd=cwd,
    )


def _ingest_inspiration(
    *,
    source_kind: str,
    source_ref: str,
    source_title: str,
    title: str,
    summary: str,
    excerpt: str,
    themes: str | None,
    related_sections: str | None,
    created_by: str,
    cwd: Path | None,
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    note = InspirationNote(
        id=store.new_id("insp", title),
        title=title.strip(),
        summary=summary.strip(),
        source=ExternalSource(
            kind=source_kind,
            ref=source_ref.strip(),
            title=source_title.strip(),
            excerpt=excerpt.strip(),
        ),
        related_themes=_split_csv(themes),
        related_sections=_split_csv(related_sections),
        created_by=created_by.strip() or "human",
    )
    target = store.add_inspiration_note(note)
    console.print(f"[green]Recorded inspiration {note.id}[/green] at {target}")


@research_app.command("scout")
def research_scout_command(
    *,
    source_kind: str | None = typer.Option(None, "--source-kind"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    notes = store.list_inspiration_notes(source_kind=source_kind)
    if not notes:
        console.print("[dim]No inspiration notes found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Kind", no_wrap=True)
    table.add_column("Title")
    table.add_column("Summary", overflow="fold")
    for note in notes:
        table.add_row(note.id, note.source.kind, note.title, note.summary)
    console.print(table)


@research_app.command("roles")
def research_roles_command() -> None:
    table = Table(show_header=True, header_style="bold")
    table.add_column("Role", no_wrap=True)
    table.add_column("Description", overflow="fold")
    for role in BUILTIN_RESEARCH_ROLES:
        table.add_row(role.name, role.description)
    console.print(table)


@research_app.command("portfolio")
def research_portfolio_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    snapshot = build_portfolio_snapshot(store)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Count", no_wrap=True)
    for label, value in (
        ("themes", snapshot.themes),
        ("unknowns", snapshot.unknowns),
        ("rabbit_holes", snapshot.rabbit_holes),
        ("publications", snapshot.publications),
        ("opportunities", snapshot.opportunities),
        ("hypotheses", snapshot.hypotheses),
        ("experiment_plans", snapshot.experiment_plans),
        ("experiments", snapshot.experiments),
        ("promotion_candidates", snapshot.promotion_candidates),
        ("archived_items", snapshot.archived_items),
        ("inspiration_notes", snapshot.inspiration_notes),
    ):
        table.add_row(label, str(value))
    console.print(table)


@research_app.command("queue")
def research_queue_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    items = build_research_queue(store)
    if not items:
        console.print("[dim]Research queue is empty.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Kind", no_wrap=True)
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Priority", no_wrap=True)
    table.add_column("Summary", overflow="fold")
    for item in items:
        table.add_row(item.kind, item.id, str(item.priority), item.summary)
    console.print(table)


@research_app.command("execute-next")
def research_execute_next_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    max_risk: str = typer.Option("medium", "--max-risk"),
    base_branch: str = typer.Option("main", "--base-branch"),
    create_branch: bool = typer.Option(False, "--create-branch/--no-create-branch"),
    commit: bool = typer.Option(False, "--commit/--no-commit"),
    push: bool = typer.Option(False, "--push/--no-push"),
    open_pr: bool = typer.Option(False, "--open/--no-open"),
    draft_pr: bool = typer.Option(True, "--draft/--ready"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    if open_pr and not push:
        raise typer.BadParameter("--open requires --push so the branch exists remotely")
    if push and not create_branch:
        raise typer.BadParameter("--push requires --create-branch")
    store = ResearchStore(root=default_research_root(working_dir))
    result = execute_next_research_item(
        store=store,
        cwd=working_dir,
        max_risk=max_risk,
        base_branch=base_branch,
        create_branch=create_branch,
        commit=commit,
        push=push,
        open_pr=open_pr,
        draft_pr=draft_pr,
    )
    if json_output:
        console.print(json.dumps(result.to_dict(), indent=2))
        return
    status_color = {
        "executed": "green",
        "deferred": "yellow",
        "no_work": "dim",
    }.get(result.status, "white")
    console.print(f"[{status_color}]{result.status}[/{status_color}] {result.message}")
    if result.queue_item_kind and result.queue_item_id:
        console.print(f"queue_item={result.queue_item_kind}:{result.queue_item_id}")
    if result.branch_name:
        console.print(f"branch={result.branch_name}")
    if result.draft_json:
        console.print(f"draft_json={result.draft_json}")
    if result.pr_body:
        console.print(f"pr_body={result.pr_body}")


@research_app.command("execute-burst")
def research_execute_burst_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    max_steps: int = typer.Option(5, "--max-steps"),
    max_risk: str = typer.Option("medium", "--max-risk"),
    base_branch: str = typer.Option("main", "--base-branch"),
    create_branch: bool = typer.Option(False, "--create-branch/--no-create-branch"),
    commit: bool = typer.Option(False, "--commit/--no-commit"),
    push: bool = typer.Option(False, "--push/--no-push"),
    open_pr: bool = typer.Option(False, "--open/--no-open"),
    draft_pr: bool = typer.Option(True, "--draft/--ready"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    if max_steps < 1:
        raise typer.BadParameter("--max-steps must be at least 1")
    if open_pr and not push:
        raise typer.BadParameter("--open requires --push so the branch exists remotely")
    if push and not create_branch:
        raise typer.BadParameter("--push requires --create-branch")
    store = ResearchStore(root=default_research_root(working_dir))
    result = execute_research_burst(
        store=store,
        cwd=working_dir,
        max_steps=max_steps,
        max_risk=max_risk,
        base_branch=base_branch,
        create_branch=create_branch,
        commit=commit,
        push=push,
        open_pr=open_pr,
        draft_pr=draft_pr,
    )
    if json_output:
        console.print(json.dumps(result.to_dict(), indent=2))
        return
    status_color = {
        "completed": "green",
        "paused": "yellow",
    }.get(result.status, "white")
    console.print(
        f"[{status_color}]{result.status}[/{status_color}] "
        f"steps={result.steps_run} stop_reason={result.stop_reason}"
    )
    for index, step in enumerate(result.results, start=1):
        console.print(f"{index}. {step.status} {step.message}")
        if step.queue_item_kind and step.queue_item_id:
            console.print(f"   queue_item={step.queue_item_kind}:{step.queue_item_id}")
        if step.branch_name:
            console.print(f"   branch={step.branch_name}")
        if step.draft_json:
            console.print(f"   draft_json={step.draft_json}")
        if step.pr_body:
            console.print(f"   pr_body={step.pr_body}")


@research_app.command("schedule-once")
def research_schedule_once_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    max_steps: int | None = typer.Option(None, "--max-steps"),
    max_risk: str | None = typer.Option(None, "--max-risk"),
    base_branch: str | None = typer.Option(None, "--base-branch"),
    create_branch: bool | None = typer.Option(None, "--create-branch/--no-create-branch"),
    commit: bool | None = typer.Option(None, "--commit/--no-commit"),
    push: bool | None = typer.Option(None, "--push/--no-push"),
    open_pr: bool | None = typer.Option(None, "--open/--no-open"),
    draft_pr: bool | None = typer.Option(None, "--draft/--ready"),
    config_path: Path | None = typer.Option(
        None, "--config", help=f"Override config path (default: {default_config_path()})."
    ),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    cfg = _load_cli_config(config_path)
    scheduler = cfg.research_scheduler
    resolved_max_steps = max_steps if max_steps is not None else (scheduler.max_steps or 5)
    resolved_max_risk = max_risk if max_risk is not None else (scheduler.max_risk or "medium")
    resolved_base_branch = (
        base_branch if base_branch is not None else (scheduler.base_branch or "main")
    )
    resolved_create_branch = (
        create_branch if create_branch is not None else bool(scheduler.create_branch or False)
    )
    resolved_commit = commit if commit is not None else bool(scheduler.commit or False)
    resolved_push = push if push is not None else bool(scheduler.push or False)
    resolved_open_pr = open_pr if open_pr is not None else bool(scheduler.open_pr or False)
    resolved_draft_pr = (
        draft_pr
        if draft_pr is not None
        else (True if scheduler.draft_pr is None else scheduler.draft_pr)
    )
    if resolved_max_steps < 1:
        raise typer.BadParameter("--max-steps must be at least 1")
    if resolved_open_pr and not resolved_push:
        raise typer.BadParameter("--open requires --push so the branch exists remotely")
    if resolved_push and not resolved_create_branch:
        raise typer.BadParameter("--push requires --create-branch")
    store = ResearchStore(root=default_research_root(working_dir))
    result, record_dir = run_scheduled_research_burst(
        store=store,
        cwd=working_dir,
        max_steps=resolved_max_steps,
        max_risk=resolved_max_risk,
        base_branch=resolved_base_branch,
        create_branch=resolved_create_branch,
        commit=resolved_commit,
        push=resolved_push,
        open_pr=resolved_open_pr,
        draft_pr=resolved_draft_pr,
    )
    payload = {
        "result": result.to_dict(),
        "record_dir": str(record_dir),
    }
    if json_output:
        console.print(json.dumps(payload, indent=2))
        return
    status_color = {
        "completed": "green",
        "paused": "yellow",
    }.get(result.status, "white")
    console.print(
        f"[{status_color}]{result.status}[/{status_color}] "
        f"steps={result.steps_run} stop_reason={result.stop_reason}"
    )
    console.print(f"record_dir={record_dir}")


@research_app.command("list-runs")
def research_list_runs_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    limit: int = typer.Option(10, "--limit"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    runs_dir = default_research_root(working_dir) / "autonomy-runs"
    if not runs_dir.exists():
        console.print("[dim]No autonomy runs found.[/dim]")
        return
    entries = sorted(
        (entry for entry in runs_dir.iterdir() if (entry / "run.json").is_file()),
        reverse=True,
    )
    if limit > 0:
        entries = entries[:limit]
    if not entries:
        console.print("[dim]No autonomy runs found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Mode", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Stop", no_wrap=True)
    table.add_column("Steps", no_wrap=True)
    for entry in entries:
        payload = json.loads((entry / "run.json").read_text(encoding="utf-8"))
        table.add_row(
            str(payload.get("id") or entry.name),
            str(payload.get("mode") or "burst"),
            str(payload.get("status") or "unknown"),
            str(payload.get("stop_reason") or ""),
            str(payload.get("steps_run") or 0),
        )
    console.print(table)


@research_app.command("show-run")
def research_show_run_command(
    run_id: str,
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    run_json = default_research_root(working_dir) / "autonomy-runs" / run_id / "run.json"
    if not run_json.is_file():
        raise typer.BadParameter(f"unknown autonomy run: {run_id!r}")
    payload = json.loads(run_json.read_text(encoding="utf-8"))
    if json_output:
        console.print(json.dumps(payload, indent=2))
        return
    console.print(f"[bold]{payload.get('id') or run_id}[/bold]")
    console.print(
        f"mode={payload.get('mode') or 'burst'} "
        f"status={payload.get('status') or 'unknown'} "
        f"stop_reason={payload.get('stop_reason') or ''} "
        f"steps={payload.get('steps_run') or 0}"
    )
    results = payload.get("results") or []
    for index, result in enumerate(results, start=1):
        if not isinstance(result, dict):
            continue
        console.print(f"{index}. {result.get('status') or 'unknown'} {result.get('message') or ''}")
        queue_kind = result.get("queue_item_kind")
        queue_id = result.get("queue_item_id")
        if queue_kind and queue_id:
            console.print(f"   queue_item={queue_kind}:{queue_id}")
        if result.get("branch_name"):
            console.print(f"   branch={result['branch_name']}")
        if result.get("draft_json"):
            console.print(f"   draft_json={result['draft_json']}")
        if result.get("pr_body"):
            console.print(f"   pr_body={result['pr_body']}")


@research_app.command("rebalance")
def research_rebalance_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    stats = rebalance_research_queue(store)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Count", no_wrap=True)
    for key, value in stats.items():
        table.add_row(key, str(value))
    console.print(table)


@research_app.command("add-theme")
def research_add_theme_command(
    *,
    title: str = typer.Option(..., "--title"),
    description: str = typer.Option(..., "--description"),
    vision_id: str = typer.Option("current", "--vision-id"),
    priority: str = typer.Option("medium", "--priority"),
    status: str = typer.Option("active", "--status"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    theme = Theme(
        id=store.new_id("theme", title),
        vision_id=vision_id.strip() or "current",
        title=title.strip(),
        description=description.strip(),
        priority=priority.strip() or "medium",
        status=status,  # type: ignore[arg-type]
    )
    target = store.add_theme(theme)
    console.print(f"[green]Added theme {theme.id}[/green] at {target}")


@research_app.command("list-themes")
def research_list_themes_command(
    *,
    vision_id: str | None = typer.Option(None, "--vision-id"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    themes = store.list_themes(vision_id=vision_id)
    if not themes:
        console.print("[dim]No themes found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Title")
    table.add_column("Priority", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Vision", style="dim", no_wrap=True)
    for item in themes:
        table.add_row(item.id, item.title, item.priority, item.status, item.vision_id)
    console.print(table)


@research_app.command("create-unknown")
def research_create_unknown_command(
    *,
    theme_id: str = typer.Option(..., "--theme-id"),
    question: str = typer.Option(..., "--question"),
    why_it_matters: str = typer.Option(..., "--why-it-matters"),
    current_belief: str = typer.Option("", "--current-belief"),
    confidence: float = typer.Option(0.0, "--confidence"),
    status: str = typer.Option("open", "--status"),
    related_sections: str | None = typer.Option(None, "--related-sections"),
    created_by: str = typer.Option("human", "--created-by"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        store.load_theme(theme_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown theme: {theme_id!r}") from exc
    unknown = Unknown(
        id=store.new_id("unknown", question),
        theme_id=theme_id.strip(),
        question=question.strip(),
        why_it_matters=why_it_matters.strip(),
        current_belief=current_belief.strip(),
        confidence=confidence,
        status=status,  # type: ignore[arg-type]
        related_sections=_split_csv(related_sections),
        created_by=created_by.strip() or "human",
    )
    target = store.add_unknown(unknown)
    console.print(f"[green]Created unknown {unknown.id}[/green] at {target}")


@research_app.command("list-unknowns")
def research_list_unknowns_command(
    *,
    theme_id: str | None = typer.Option(None, "--theme-id"),
    status: str | None = typer.Option(None, "--status"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    unknowns = store.list_unknowns(theme_id=theme_id, status=status)
    if not unknowns:
        console.print("[dim]No unknowns found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Theme", style="dim", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Confidence", no_wrap=True)
    table.add_column("Question", overflow="fold")
    for item in unknowns:
        table.add_row(item.id, item.theme_id, item.status, f"{item.confidence:.2f}", item.question)
    console.print(table)


@research_app.command("map-section")
def research_map_section_command(
    *,
    section: str = typer.Option(..., "--section"),
    files: str | None = typer.Option(None, "--files"),
    interfaces: str | None = typer.Option(None, "--interfaces"),
    constraints: str | None = typer.Option(None, "--constraints"),
    weaknesses: str | None = typer.Option(None, "--weaknesses"),
    opportunities: str | None = typer.Option(None, "--opportunities"),
    created_by: str = typer.Option("human", "--created-by"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    section_map = SectionMap(
        id=store.new_id("section", section),
        section=section.strip(),
        files=_split_csv(files),
        interfaces=_split_csv(interfaces),
        constraints=_split_csv(constraints),
        weaknesses=_split_csv(weaknesses),
        opportunities=_split_csv(opportunities),
        created_by=created_by.strip() or "human",
    )
    target = store.add_section_map(section_map)
    console.print(f"[green]Mapped section {section_map.section}[/green] at {target}")


@research_app.command("add-observation")
def research_add_observation_command(
    *,
    title: str = typer.Option(..., "--title"),
    summary: str = typer.Option(..., "--summary"),
    source_type: str = typer.Option(..., "--source-type"),
    source_ref: str | None = typer.Option(None, "--source-ref"),
    related_sections: str | None = typer.Option(None, "--related-sections"),
    theme: str | None = typer.Option(None, "--theme"),
    created_by: str = typer.Option("human", "--created-by"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    observation = Observation(
        id=store.new_id("obs", title),
        title=title.strip(),
        summary=summary.strip(),
        source_type=source_type.strip(),
        source_ref=(source_ref or "").strip(),
        related_sections=_split_csv(related_sections),
        theme=(theme or "").strip(),
        created_by=created_by.strip() or "human",
    )
    target = store.add_observation(observation)
    console.print(f"[green]Recorded observation {observation.id}[/green] at {target}")


@research_app.command("show-section")
def research_show_section_command(
    section_or_id: str = typer.Argument(..., help="Section name or section-map id."),
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    section_map = store.find_section_map(section_or_id)
    if section_map is None:
        raise typer.BadParameter(f"unknown section map: {section_or_id!r}")
    console.print(f"[bold]{section_map.section}[/bold] ({section_map.id})")
    for label, values in (
        ("Files", section_map.files),
        ("Interfaces", section_map.interfaces),
        ("Constraints", section_map.constraints),
        ("Weaknesses", section_map.weaknesses),
        ("Opportunities", section_map.opportunities),
    ):
        if values:
            console.print(f"\n[bold]{label}[/bold]")
            for value in values:
                console.print(f"- {value}")


@research_app.command("create-opportunity")
def research_create_opportunity_command(
    *,
    title: str = typer.Option(..., "--title"),
    summary: str = typer.Option(..., "--summary"),
    related_sections: str | None = typer.Option(None, "--related-sections"),
    origin_observations: str | None = typer.Option(None, "--origin-observations"),
    change_modes: str | None = typer.Option(None, "--change-modes"),
    theme: str | None = typer.Option(None, "--theme"),
    priority: str = typer.Option("medium", "--priority"),
    created_by: str = typer.Option("human", "--created-by"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    opportunity = Opportunity(
        id=store.new_id("opp", title),
        title=title.strip(),
        summary=summary.strip(),
        related_sections=_split_csv(related_sections),
        origin_observations=_split_csv(origin_observations),
        change_modes=_split_csv(change_modes),
        theme=(theme or "").strip(),
        priority=priority.strip() or "medium",
        created_by=created_by.strip() or "human",
    )
    target = store.add_opportunity(opportunity)
    console.print(f"[green]Created opportunity {opportunity.id}[/green] at {target}")


@research_app.command("list-opportunities")
def research_list_opportunities_command(
    *,
    theme: str | None = typer.Option(None, "--theme"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    opportunities = store.list_opportunities(theme=theme)
    if not opportunities:
        console.print("[dim]No opportunities found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Title")
    table.add_column("Theme", no_wrap=True)
    table.add_column("Priority", no_wrap=True)
    table.add_column("Sections", overflow="fold")
    for opportunity in opportunities:
        table.add_row(
            opportunity.id,
            opportunity.title,
            opportunity.theme or "—",
            opportunity.priority,
            ", ".join(opportunity.related_sections) or "—",
        )
    console.print(table)


@research_app.command("related")
def research_related_command(
    target: str = typer.Argument(..., help="Section, observation id, or keyword to match."),
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    opportunities = store.related_opportunities(target)
    if not opportunities:
        console.print("[dim]No related opportunities found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Title")
    table.add_column("Summary", overflow="fold")
    for opportunity in opportunities:
        table.add_row(opportunity.id, opportunity.title, opportunity.summary)
    console.print(table)


@research_app.command("hypothesize")
def research_hypothesize_command(
    *,
    opportunity_id: str = typer.Option(..., "--opportunity"),
    claim: str = typer.Option(..., "--claim"),
    expected_win: str = typer.Option(..., "--expected-win"),
    risk_level: str = typer.Option(..., "--risk-level"),
    change_mode: str = typer.Option(..., "--change-mode"),
    created_by: str = typer.Option("human", "--created-by"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        store.load_opportunity(opportunity_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown opportunity: {opportunity_id!r}") from exc
    hypothesis = Hypothesis(
        id=store.new_id("hyp", claim),
        opportunity_id=opportunity_id.strip(),
        claim=claim.strip(),
        expected_win=expected_win.strip(),
        risk_level=risk_level.strip(),
        change_mode=change_mode.strip(),
        created_by=created_by.strip() or "human",
    )
    target = store.add_hypothesis(hypothesis)
    console.print(f"[green]Created hypothesis {hypothesis.id}[/green] at {target}")


@research_app.command("plan-experiment")
def research_plan_experiment_command(
    *,
    hypothesis_id: str = typer.Option(..., "--hypothesis"),
    plan: str = typer.Option(..., "--plan"),
    target_files: str | None = typer.Option(None, "--target-files"),
    checks: str | None = typer.Option(None, "--checks"),
    eval_slices: str | None = typer.Option(None, "--eval-slices"),
    created_by: str = typer.Option("human", "--created-by"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        store.load_hypothesis(hypothesis_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown hypothesis: {hypothesis_id!r}") from exc
    experiment_plan = ExperimentPlan(
        id=store.new_id("plan", hypothesis_id),
        hypothesis_id=hypothesis_id.strip(),
        plan=plan.strip(),
        target_files=_split_csv(target_files),
        checks=_split_csv(checks),
        eval_slices=_split_csv(eval_slices),
        created_by=created_by.strip() or "human",
    )
    target = store.add_experiment_plan(experiment_plan)
    console.print(f"[green]Planned experiment {experiment_plan.id}[/green] at {target}")


@research_app.command("refine")
def research_refine_command(
    *,
    title: str = typer.Option(..., "--title"),
    summary: str = typer.Option(..., "--summary"),
    source_publication: list[str] = typer.Option([], "--source-publication"),
    source_hypothesis: list[str] = typer.Option([], "--source-hypothesis"),
    target_files: str | None = typer.Option(None, "--target-files"),
    expected_metric: str | None = typer.Option(None, "--expected-metric"),
    validation_plan: str | None = typer.Option(None, "--validation-plan"),
    risk_level: str = typer.Option("medium", "--risk-level"),
    created_by: str = typer.Option("human", "--created-by"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    mode: str | None = typer.Option(None, "--mode"),
    subsystem: str | None = typer.Option(None, "--subsystem"),
    rationale: str | None = typer.Option(None, "--rationale"),
    expected_outcome: str | None = typer.Option(None, "--expected-outcome"),
    risk: str | None = typer.Option(None, "--risk"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    for publication_id in source_publication:
        try:
            store.load_publication(publication_id)
        except FileNotFoundError as exc:
            raise typer.BadParameter(f"unknown publication: {publication_id!r}") from exc
    for hypothesis_id in source_hypothesis:
        try:
            store.load_hypothesis(hypothesis_id)
        except FileNotFoundError as exc:
            raise typer.BadParameter(f"unknown hypothesis: {hypothesis_id!r}") from exc
    try:
        change_intent = store.parse_change_intent(
            mode=mode,
            subsystem=subsystem,
            rationale=rationale,
            expected_outcome=expected_outcome,
            risk=risk,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    candidate = PromotionCandidate(
        id=store.new_id("promo", title),
        title=title.strip(),
        summary=summary.strip(),
        source_publications=tuple(item.strip() for item in source_publication if item.strip()),
        source_hypotheses=tuple(item.strip() for item in source_hypothesis if item.strip()),
        target_files=_split_csv(target_files),
        expected_metric=(expected_metric or "").strip(),
        validation_plan=(validation_plan or "").strip(),
        risk_level=risk_level.strip() or "medium",
        created_by=created_by.strip() or "human",
        change_intent=change_intent,
    )
    target = store.add_promotion_candidate(candidate)
    console.print(f"[green]Created promotion candidate {candidate.id}[/green] at {target}")


@research_app.command("list-candidates")
def research_list_candidates_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    candidates = store.list_promotion_candidates()
    if not candidates:
        console.print("[dim]No promotion candidates found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Title")
    table.add_column("Risk", no_wrap=True)
    table.add_column("Metric", overflow="fold")
    for candidate in candidates:
        table.add_row(
            candidate.id,
            candidate.title,
            candidate.risk_level,
            candidate.expected_metric or "—",
        )
    console.print(table)


@research_app.command("show-candidate")
@candidate_app.command("show")
def research_show_candidate_command(
    candidate_id: str = typer.Argument(..., help="Promotion candidate id."),
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    _show_candidate_command(candidate_id=candidate_id, cwd=cwd, console=console)


@research_app.command("promote")
def research_promote_command(
    *,
    candidate_id: str = typer.Option(..., "--candidate"),
    base_branch: str = typer.Option("main", "--base-branch"),
    create_branch: bool = typer.Option(True, "--create-branch/--no-create-branch"),
    commit: bool = typer.Option(False, "--commit/--no-commit"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    _promote_command(
        candidate_id=candidate_id,
        base_branch=base_branch,
        create_branch=create_branch,
        commit=commit,
        cwd=cwd,
        console=console,
    )


@research_app.command("pr")
def research_pr_command(
    *,
    candidate_id: str = typer.Option(..., "--candidate"),
    base_branch: str = typer.Option("main", "--base-branch"),
    push: bool = typer.Option(False, "--push/--no-push"),
    open_pr: bool = typer.Option(False, "--open/--no-open"),
    draft_pr: bool = typer.Option(True, "--draft/--ready"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    _pr_command(
        candidate_id=candidate_id,
        base_branch=base_branch,
        push=push,
        open_pr=open_pr,
        draft_pr=draft_pr,
        cwd=cwd,
        console=console,
    )


@research_app.command("archive")
def research_archive_command(
    *,
    kind: str = typer.Option(..., "--kind"),
    item_id: str = typer.Option(..., "--id"),
    reason: str = typer.Option(..., "--reason"),
    note: str = typer.Option("", "--note"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        target = store.archive_item(kind=kind, item_id=item_id, reason=reason, note=note)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Archived {kind}:{item_id}[/green] at {target}")


@research_app.command("reject")
def research_reject_command(
    *,
    kind: str = typer.Option(..., "--kind"),
    item_id: str = typer.Option(..., "--id"),
    reason: str = typer.Option(..., "--reason"),
    note: str = typer.Option("", "--note"),
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    full_reason = reason.strip()
    if note.strip():
        full_reason = f"{full_reason} ({note.strip()})"
    try:
        target = store.archive_item(kind=kind, item_id=item_id, reason=full_reason)
    except (FileNotFoundError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Rejected and archived {kind}:{item_id}[/green] at {target}")


@research_app.command("list-archive")
def research_list_archive_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    items = store.list_archive_items()
    if not items:
        console.print("[dim]No archived research items found.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Archive ID", style="dim", no_wrap=True)
    table.add_column("Kind", no_wrap=True)
    table.add_column("Original ID", style="dim", no_wrap=True)
    table.add_column("Reason", overflow="fold")
    for item in items:
        table.add_row(item.archive_id, item.kind, item.original_id, item.reason)
    console.print(table)


@research_app.command("resurrect")
def research_resurrect_command(
    archive_id: str = typer.Argument(..., help="Archive id to restore."),
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        target = store.resurrect_archive_item(archive_id)
    except (FileNotFoundError, FileExistsError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    console.print(f"[green]Resurrected {archive_id}[/green] to {target}")


@experiment_app.command("run")
def research_experiment_run_command(
    *,
    plan_id: str = typer.Option(..., "--plan"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    created_by: str = typer.Option("human", "--created-by"),
    timeout: int = typer.Option(600, "--timeout"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        plan = store.load_experiment_plan(plan_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown experiment plan: {plan_id!r}") from exc
    experiment, result = run_experiment_plan(
        store=store,
        plan=plan,
        cwd=working_dir,
        created_by=created_by,
        timeout=timeout,
    )
    store.add_experiment(experiment, result)
    console.print(
        f"[green]Experiment {experiment.id}[/green] {result.status} "
        f"commands={len(result.command_results)} duration={result.duration_seconds:.2f}s"
    )


@experiment_app.command("show")
def research_experiment_show_command(
    experiment_id: str = typer.Argument(..., help="Experiment id."),
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        experiment = store.load_experiment(experiment_id)
        result = store.load_experiment_result(experiment_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown experiment: {experiment_id!r}") from exc
    console.print(f"[bold]{experiment.id}[/bold] plan={experiment.plan_id} status={result.status}")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Kind", no_wrap=True)
    table.add_column("Exit", no_wrap=True)
    table.add_column("Seconds", no_wrap=True)
    table.add_column("Command", overflow="fold")
    for item in result.command_results:
        table.add_row(item.kind, str(item.exit_code), f"{item.duration_seconds:.2f}", item.command)
    console.print(table)


@experiment_app.command("compare")
def research_experiment_compare_command(
    left: str = typer.Argument(..., help="Left experiment id."),
    right: str = typer.Argument(..., help="Right experiment id."),
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        left_result = store.load_experiment_result(left)
        right_result = store.load_experiment_result(right)
    except FileNotFoundError as exc:
        raise typer.BadParameter("unknown experiment id in compare") from exc
    comparison = compare_experiment_results(left_result, right_result)
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Left")
    table.add_column("Right")
    table.add_row("status", str(comparison["left_status"]), str(comparison["right_status"]))
    table.add_row(
        "duration_seconds",
        f"{comparison['left_duration_seconds']:.2f}",
        f"{comparison['right_duration_seconds']:.2f}",
    )
    table.add_row("commands", str(comparison["left_commands"]), str(comparison["right_commands"]))
    console.print(table)


__all__ = [
    "build_research_prompt",
    "research_add_observation_command",
    "research_add_theme_command",
    "research_app",
    "research_archive_command",
    "research_command",
    "research_create_opportunity_command",
    "research_create_unknown_command",
    "research_experiment_compare_command",
    "research_experiment_run_command",
    "research_experiment_show_command",
    "research_hypothesize_command",
    "research_list_archive_command",
    "research_list_candidates_command",
    "research_list_opportunities_command",
    "research_list_themes_command",
    "research_list_unknowns_command",
    "research_map_section_command",
    "research_plan_experiment_command",
    "research_pr_command",
    "research_promote_command",
    "research_refine_command",
    "research_reject_command",
    "research_related_command",
    "research_resurrect_command",
    "research_run_command",
    "research_search_command",
    "research_show_section_command",
    "vision_app",
    "vision_show_command",
    "vision_update_command",
]
