from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from harness.core.pr_generation import (
    build_promotion_draft,
    commit_paths,
    create_pull_request,
    ensure_branch,
    push_branch,
    write_promotion_draft,
)
from harness.core.research_store import ResearchStore, default_research_root


def show_candidate_command(*, candidate_id: str, cwd: Path | None, console: Console) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        candidate = store.load_promotion_candidate(candidate_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown promotion candidate: {candidate_id!r}") from exc
    console.print(f"[bold]{candidate.title}[/bold]")
    console.print(candidate.summary)
    if candidate.target_files:
        console.print("\n[bold]Target Files[/bold]")
        for item in candidate.target_files:
            console.print(f"- {item}")


def promote_command(
    *,
    candidate_id: str,
    base_branch: str,
    create_branch: bool,
    commit: bool,
    cwd: Path | None,
    console: Console,
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        candidate = store.load_promotion_candidate(candidate_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown promotion candidate: {candidate_id!r}") from exc
    if commit and not candidate.target_files:
        raise typer.BadParameter("candidate must declare target_files before --commit can be used")
    draft = build_promotion_draft(candidate, base_branch=base_branch)
    candidate_dir = store.promotion_candidates_dir / candidate.id
    json_path, body_path = write_promotion_draft(draft=draft, target_dir=candidate_dir)
    if create_branch:
        ensure_branch(cwd=working_dir, branch_name=draft.branch_name, base_branch=base_branch)
    if commit:
        commit_paths(cwd=working_dir, message=draft.commit_message, paths=candidate.target_files)
    console.print(f"[green]Prepared promotion draft for {candidate.id}[/green]")
    console.print(f"branch={draft.branch_name}")
    console.print(f"commit_message={draft.commit_message}")
    console.print(f"draft_json={json_path}")
    console.print(f"pr_body={body_path}")


def pr_command(
    *,
    candidate_id: str,
    base_branch: str,
    push: bool,
    open_pr: bool,
    draft_pr: bool,
    cwd: Path | None,
    console: Console,
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = ResearchStore(root=default_research_root(working_dir))
    try:
        candidate = store.load_promotion_candidate(candidate_id)
    except FileNotFoundError as exc:
        raise typer.BadParameter(f"unknown promotion candidate: {candidate_id!r}") from exc
    if open_pr and not push:
        raise typer.BadParameter("--open requires --push so the branch exists on the remote")
    draft = build_promotion_draft(candidate, base_branch=base_branch)
    candidate_dir = store.promotion_candidates_dir / candidate.id
    _json_path, body_path = write_promotion_draft(draft=draft, target_dir=candidate_dir)
    if push:
        push_branch(cwd=working_dir, branch_name=draft.branch_name)
    if open_pr:
        create_pull_request(
            cwd=working_dir,
            title=draft.pr_title,
            body_path=body_path,
            base_branch=base_branch,
            head_branch=draft.branch_name,
            draft=draft_pr,
        )
    console.print(f"[green]Prepared PR payload for {candidate.id}[/green]")
    console.print(f"title={draft.pr_title}")
    console.print(f"branch={draft.branch_name}")
    console.print(f"body={body_path}")


__all__ = ["pr_command", "promote_command", "show_candidate_command"]
