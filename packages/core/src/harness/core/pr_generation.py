from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.core.promotion_candidates import PromotionCandidate


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "candidate"


@dataclass(frozen=True, slots=True)
class PromotionDraft:
    candidate_id: str
    branch_name: str
    commit_message: str
    pr_title: str
    pr_body: str
    base_branch: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "branch_name": self.branch_name,
            "commit_message": self.commit_message,
            "pr_title": self.pr_title,
            "pr_body": self.pr_body,
            "base_branch": self.base_branch,
        }


def branch_name_for_candidate(candidate: PromotionCandidate) -> str:
    return f"research/{_slugify(candidate.id)[:48]}"


def commit_message_for_candidate(candidate: PromotionCandidate) -> str:
    return f"research: promote {candidate.title.strip() or candidate.id}"


def pr_title_for_candidate(candidate: PromotionCandidate) -> str:
    if candidate.change_intent is not None:
        return f"{candidate.change_intent.mode}: {candidate.title.strip() or candidate.id}"
    return f"research: {candidate.title.strip() or candidate.id}"


def pr_body_for_candidate(
    candidate: PromotionCandidate, *, branch_name: str, base_branch: str
) -> str:
    lines = [
        "## Promotion Candidate",
        f"- candidate: `{candidate.id}`",
        f"- branch: `{branch_name}`",
        f"- base: `{base_branch}`",
        f"- risk: `{candidate.risk_level}`",
        "",
        "## Summary",
        candidate.summary or "—",
    ]
    if candidate.change_intent is not None:
        lines += [
            "",
            "## Change Intent",
            f"- mode: `{candidate.change_intent.mode}`",
            f"- subsystem: `{candidate.change_intent.subsystem}`",
            f"- risk: `{candidate.change_intent.risk}`",
            "",
            candidate.change_intent.rationale,
            "",
            f"Expected outcome: {candidate.change_intent.expected_outcome}",
        ]
    if candidate.target_files:
        lines += ["", "## Target Files", *[f"- `{item}`" for item in candidate.target_files]]
    if candidate.source_publications:
        lines += [
            "",
            "## Source Publications",
            *[f"- `{item}`" for item in candidate.source_publications],
        ]
    if candidate.source_hypotheses:
        lines += [
            "",
            "## Source Hypotheses",
            *[f"- `{item}`" for item in candidate.source_hypotheses],
        ]
    if candidate.expected_metric:
        lines += ["", "## Expected Metric", candidate.expected_metric]
    if candidate.validation_plan:
        lines += ["", "## Validation Plan", candidate.validation_plan]
    lines += [
        "",
        "## Evidence Checklist",
        "- [ ] Targeted tests passed",
        "- [ ] Relevant eval slices passed",
        "- [ ] No unintended file-scope expansion",
        "- [ ] Risk remains acceptable for review",
    ]
    return "\n".join(lines).strip() + "\n"


def build_promotion_draft(
    candidate: PromotionCandidate, *, base_branch: str = "main"
) -> PromotionDraft:
    branch_name = branch_name_for_candidate(candidate)
    return PromotionDraft(
        candidate_id=candidate.id,
        branch_name=branch_name,
        commit_message=commit_message_for_candidate(candidate),
        pr_title=pr_title_for_candidate(candidate),
        pr_body=pr_body_for_candidate(candidate, branch_name=branch_name, base_branch=base_branch),
        base_branch=base_branch,
    )


def write_promotion_draft(*, draft: PromotionDraft, target_dir: Path) -> tuple[Path, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    json_path = target_dir / "promotion_draft.json"
    body_path = target_dir / "PR_BODY.md"
    json_path.write_text(json.dumps(draft.to_dict(), indent=2), encoding="utf-8")
    body_path.write_text(draft.pr_body, encoding="utf-8")
    return json_path, body_path


def _git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=True)


def ensure_branch(*, cwd: Path, branch_name: str, base_branch: str) -> None:
    exists = subprocess.run(
        ["git", "rev-parse", "--verify", branch_name],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if exists.returncode == 0:
        _git(["git", "switch", branch_name], cwd=cwd)
        return
    _git(["git", "switch", "-c", branch_name, base_branch], cwd=cwd)


def paths_have_changes(*, cwd: Path, paths: tuple[str, ...]) -> bool:
    if not paths:
        return False
    result = subprocess.run(
        ["git", "status", "--short", "--", *paths],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def commit_paths(*, cwd: Path, message: str, paths: tuple[str, ...]) -> bool:
    if not paths:
        raise ValueError("commit requires at least one target file")
    _git(["git", "add", "--", *paths], cwd=cwd)
    if not paths_have_changes(cwd=cwd, paths=paths):
        return False
    _git(["git", "commit", "-m", message, "--", *paths], cwd=cwd)
    return True


def push_branch(*, cwd: Path, branch_name: str, remote: str = "origin") -> None:
    _git(["git", "push", "-u", remote, branch_name], cwd=cwd)


def create_pull_request(
    *,
    cwd: Path,
    title: str,
    body_path: Path,
    base_branch: str,
    head_branch: str,
    draft: bool,
) -> None:
    args = [
        "gh",
        "pr",
        "create",
        "--base",
        base_branch,
        "--head",
        head_branch,
        "--title",
        title,
        "--body-file",
        str(body_path),
    ]
    if draft:
        args.append("--draft")
    _git(args, cwd=cwd)


__all__ = [
    "PromotionDraft",
    "branch_name_for_candidate",
    "build_promotion_draft",
    "commit_message_for_candidate",
    "commit_paths",
    "create_pull_request",
    "ensure_branch",
    "paths_have_changes",
    "pr_body_for_candidate",
    "pr_title_for_candidate",
    "push_branch",
    "write_promotion_draft",
]
