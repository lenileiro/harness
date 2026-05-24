"""Post-run artifact analysis for eval failures and reusable harness lessons."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path
from typing import Any

from evals.types import HarnessAdjustment

_STOPWORDS = {
    "this",
    "that",
    "with",
    "from",
    "into",
    "task",
    "file",
    "files",
    "only",
    "when",
    "then",
    "they",
    "them",
    "your",
    "have",
    "must",
    "should",
    "would",
    "could",
    "after",
    "before",
    "there",
    "their",
    "while",
    "tests",
    "test",
    "code",
    "change",
    "changes",
    "added",
    "fix",
}


def _new_adjustment_id() -> str:
    return f"adj_{uuid.uuid4().hex[:10]}"


def _keywords_from_task_text(text: str, *, max_keywords: int = 8) -> list[str]:
    identifiers = re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", text)
    tokens = identifiers + re.findall(r"[A-Za-z][A-Za-z0-9_]{3,}", text)
    seen: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in _STOPWORDS or lowered in seen:
            continue
        seen.append(lowered)
        if len(seen) >= max_keywords:
            break
    return seen


def _added_comment_lines(git_diff: str) -> list[str]:
    return [
        line[1:].strip()
        for line in git_diff.splitlines()
        if line.startswith("+") and not line.startswith("+++") and line[1:].lstrip().startswith("#")
    ]


def _load_artifact_bundle(artifact_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    outcome = json.loads((artifact_dir / "outcome.json").read_text(encoding="utf-8"))
    trace_events = [
        json.loads(line)
        for line in (artifact_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    git_diff = (artifact_dir / "git_diff.patch").read_text(encoding="utf-8")
    return outcome, trace_events, git_diff


def _append_adjustment(
    adjustments: list[HarnessAdjustment],
    *,
    text: str,
    kind: str,
    triggers: list[str],
    weight: float,
    fixture_name: str,
    variant: str,
    artifact_dir: Path,
    rationale: str,
    evidence: dict[str, Any],
) -> None:
    if any(existing.text == text for existing in adjustments):
        return
    adjustments.append(
        HarnessAdjustment(
            id=_new_adjustment_id(),
            kind=kind,
            text=text,
            triggers=triggers,
            weight=weight,
            source_fixture_name=fixture_name,
            source_variant=variant,
            source_artifact_dir=artifact_dir,
            rationale=rationale,
            evidence=evidence,
        )
    )


def analyze_artifact_dir(artifact_dir: Path) -> list[HarnessAdjustment]:
    outcome, trace_events, git_diff = _load_artifact_bundle(artifact_dir)
    fixture = outcome.get("fixture") or {}
    hard_metrics = outcome.get("hard_metrics") or {}
    task_text = str(fixture.get("task_text") or "")
    family = str(fixture.get("family") or "").strip().lower()
    fixture_name = str(fixture.get("name") or "")
    variant = str(outcome.get("variant") or "defended")
    verify_passed = bool(hard_metrics.get("verify_passed", outcome.get("test_exit_code") == 0))
    triggers = _keywords_from_task_text(task_text)

    adjustments: list[HarnessAdjustment] = []

    if hard_metrics.get("edit_before_repro"):
        _append_adjustment(
            adjustments,
            text="Run the narrowest failing test or verify command before editing; do not patch from the prompt alone.",
            kind="tests_first",
            triggers=triggers,
            weight=2.4 if not verify_passed else 1.6,
            fixture_name=fixture_name,
            variant=variant,
            artifact_dir=artifact_dir,
            rationale="Artifact trace showed edits happened before the first verification step.",
            evidence={
                "edit_before_repro": True,
                "time_to_first_verification_seconds": hard_metrics.get(
                    "time_to_first_verification_seconds"
                ),
            },
        )

    if hard_metrics.get("verification_after_failure"):
        _append_adjustment(
            adjustments,
            text="After a failed verification, rerun the targeted check after each repair and do not declare completion until it passes.",
            kind="verify_loop",
            triggers=triggers,
            weight=1.8,
            fixture_name=fixture_name,
            variant=variant,
            artifact_dir=artifact_dir,
            rationale="The trace required at least one verification retry loop.",
            evidence={
                "verification_after_failure": True,
                "retry_loops": hard_metrics.get("retry_loops", 0),
            },
        )

    added_comments = _added_comment_lines(git_diff)
    if added_comments:
        _append_adjustment(
            adjustments,
            text="Do not rewrite nearby comments or add banner-style test scaffolding unless the task explicitly asks for it; keep the patch in executable lines.",
            kind="scope_comments",
            triggers=triggers,
            weight=2.1,
            fixture_name=fixture_name,
            variant=variant,
            artifact_dir=artifact_dir,
            rationale="The final diff added comment-only lines, which has been a repeated source of overscoped patches.",
            evidence={"added_comment_lines": added_comments[:5]},
        )

    if family == "reproduce-before-repair" and verify_passed:
        _append_adjustment(
            adjustments,
            text="Keep reproduce-before-repair fixes in the production lookup path; do not broaden normalization or patch adjacent validation code.",
            kind="family_pattern",
            triggers=triggers,
            weight=2.2,
            fixture_name=fixture_name,
            variant=variant,
            artifact_dir=artifact_dir,
            rationale="Successful run stayed in the target production path without extra normalization.",
            evidence={"family": family, "verify_passed": True},
        )

    if family == "scope-discipline":
        _append_adjustment(
            adjustments,
            text="For null-guard bugfixes, make the smallest possible change in the named formatter and avoid touching sibling helpers or unrelated formatting.",
            kind="family_pattern",
            triggers=triggers,
            weight=2.0 if verify_passed else 2.3,
            fixture_name=fixture_name,
            variant=variant,
            artifact_dir=artifact_dir,
            rationale="Scope-discipline tasks reward minimal one-function fixes rather than broader cleanup.",
            evidence={"family": family, "verify_passed": verify_passed},
        )

    if family == "wrong-diagnosis" and ("TIMEOUT_SECONDS" in task_text or "constant" in task_text):
        _append_adjustment(
            adjustments,
            text="Treat prompt-suggested symptom edits as provisional. If you change a named constant to test a theory, revert it once the real root cause is identified.",
            kind="prompt_surface_revert",
            triggers=triggers,
            weight=2.5 if not verify_passed else 2.1,
            fixture_name=fixture_name,
            variant=variant,
            artifact_dir=artifact_dir,
            rationale="Wrong-diagnosis fixtures often leave a prompt-surface constant changed after the real bug is fixed.",
            evidence={"family": family, "verify_passed": verify_passed},
        )

    if family == "sustained-coherence":
        _append_adjustment(
            adjustments,
            text="Ignore cleanup bait in sustained-coherence tasks. Add the requested method, docs entry, and plain regression test without opportunistic comment or typo cleanup.",
            kind="family_pattern",
            triggers=triggers,
            weight=2.4,
            fixture_name=fixture_name,
            variant=variant,
            artifact_dir=artifact_dir,
            rationale="These tasks reward resisting adjacent cleanup and holding scope over longer trajectories.",
            evidence={
                "family": family,
                "tool_calls": hard_metrics.get("tool_calls"),
                "trace_events": len(trace_events),
            },
        )

    if not adjustments:
        _append_adjustment(
            adjustments,
            text="Use the smallest evidence-backed patch that satisfies the failing verification path before expanding scope.",
            kind="general",
            triggers=triggers,
            weight=1.0,
            fixture_name=fixture_name,
            variant=variant,
            artifact_dir=artifact_dir,
            rationale="Fallback adjustment when no narrower failure pattern was detected.",
            evidence={"verify_passed": verify_passed},
        )

    return adjustments


def persist_harness_adjustments(artifact_dir: Path) -> list[HarnessAdjustment]:
    adjustments = analyze_artifact_dir(artifact_dir)
    (artifact_dir / "harness_adjustments.json").write_text(
        json.dumps([adjustment.to_dict() for adjustment in adjustments], indent=2),
        encoding="utf-8",
    )
    return adjustments
