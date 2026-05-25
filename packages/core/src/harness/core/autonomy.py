from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from harness.core.experiment_plans import ExperimentPlan
from harness.core.experiment_runner import run_experiment_plan
from harness.core.hypotheses import Hypothesis
from harness.core.opportunities import Opportunity
from harness.core.pr_generation import (
    build_promotion_draft,
    commit_paths,
    create_pull_request,
    ensure_branch,
    push_branch,
    write_promotion_draft,
)
from harness.core.promotion_candidates import PromotionCandidate
from harness.core.research_models import Publication, RabbitHole
from harness.core.research_scheduler import build_research_queue
from harness.core.research_store import ResearchStore

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def _risk_value(level: str) -> int:
    return _RISK_ORDER.get(level.strip().lower(), _RISK_ORDER["high"])


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class AutonomyExecutionResult:
    status: str
    queue_item_kind: str | None
    queue_item_id: str | None
    message: str
    branch_name: str | None = None
    draft_json: Path | None = None
    pr_body: Path | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "status": self.status,
            "queue_item_kind": self.queue_item_kind,
            "queue_item_id": self.queue_item_id,
            "message": self.message,
            "branch_name": self.branch_name,
            "draft_json": str(self.draft_json) if self.draft_json is not None else None,
            "pr_body": str(self.pr_body) if self.pr_body is not None else None,
        }


@dataclass(frozen=True, slots=True)
class AutonomyBurstResult:
    status: str
    steps_run: int
    stop_reason: str
    results: tuple[AutonomyExecutionResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "steps_run": self.steps_run,
            "stop_reason": self.stop_reason,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True, slots=True)
class AutonomyRunRecord:
    id: str
    mode: str
    status: str
    stop_reason: str
    steps_run: int
    cwd: str
    created_at: str
    results: tuple[AutonomyExecutionResult, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "mode": self.mode,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "steps_run": self.steps_run,
            "cwd": self.cwd,
            "created_at": self.created_at,
            "results": [result.to_dict() for result in self.results],
        }


def _review_promotion_artifacts(*, candidate: PromotionCandidate, draft: object) -> list[str]:
    issues: list[str] = []
    if not candidate.summary.strip():
        issues.append("candidate summary is empty")
    if not candidate.target_files:
        issues.append("candidate has no target files")
    if not candidate.expected_metric.strip():
        issues.append("candidate is missing an expected metric")
    if not candidate.validation_plan.strip():
        issues.append("candidate is missing a validation plan")
    if candidate.change_intent is None:
        issues.append("candidate is missing change intent")

    pr_body = getattr(draft, "pr_body", "")
    required_sections = (
        "## Promotion Candidate",
        "## Summary",
        "## Target Files",
        "## Expected Metric",
        "## Validation Plan",
        "## Evidence Checklist",
    )
    for section in required_sections:
        if section not in pr_body:
            issues.append(f"PR body is missing required section: {section}")
    return issues


def _write_promotion_review(
    *, target_dir: Path, candidate: PromotionCandidate, issues: list[str]
) -> tuple[Path, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    review_json = target_dir / "promotion_review.json"
    review_md = target_dir / "PROMOTION_REVIEW.md"
    payload = {
        "candidate_id": candidate.id,
        "status": "passed" if not issues else "failed",
        "issues": issues,
    }
    review_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        f"# Promotion Review: {candidate.id}",
        "",
        f"Status: {'passed' if not issues else 'failed'}",
    ]
    if issues:
        lines += ["", "## Issues", *[f"- {issue}" for issue in issues]]
    else:
        lines += ["", "No blocking issues detected."]
    review_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return review_json, review_md


def _create_revision_opportunity(
    *, store: ResearchStore, candidate: PromotionCandidate, issues: list[str]
) -> Opportunity:
    title = f"Revise {candidate.title}"
    return Opportunity(
        id=store.new_id("opp", title),
        title=title,
        summary="Resolve autonomous promotion review findings: " + "; ".join(issues),
        related_sections=candidate.target_files,
        change_modes=(
            (candidate.change_intent.mode,) if candidate.change_intent is not None else ("improve",)
        ),
        theme="autonomous-improvement",
        priority="high",
        created_by="autonomy",
    )


def execute_next_research_item(
    *,
    store: ResearchStore,
    cwd: Path,
    max_risk: str = "medium",
    base_branch: str = "main",
    create_branch: bool = False,
    commit: bool = False,
    push: bool = False,
    open_pr: bool = False,
    draft_pr: bool = True,
) -> AutonomyExecutionResult:
    queue = build_research_queue(store)
    if not queue:
        return AutonomyExecutionResult(
            status="no_work",
            queue_item_kind=None,
            queue_item_id=None,
            message="Research queue is empty.",
        )

    item = queue[0]
    if item.kind == "unknown":
        unknown = store.load_unknown(item.id)
        theme = store.load_theme(unknown.theme_id)
        rabbit_hole = RabbitHole(
            id=store.new_id("rh", unknown.question),
            title=f"Investigate {unknown.question}",
            question=unknown.question,
            scope=unknown.why_it_matters,
            theme=theme.title,
            related_sections=unknown.related_sections,
            tags=("autonomy", "continuation"),
            opened_by="autonomy",
        )
        store.add_rabbit_hole(rabbit_hole)
        publication = Publication(
            id=store.new_id("pub", f"{unknown.question} continuation"),
            rabbit_hole_id=rabbit_hole.id,
            title=f"Continuation plan for {unknown.question}",
            summary=(
                "This unknown was selected for bounded continuation. "
                "Investigate it through the linked rabbit hole and preserve findings for the next agent."
            ),
            claims=((unknown.current_belief,) if unknown.current_belief else ()),
            recommendations=(
                "Continue with the linked rabbit hole before attempting implementation.",
            ),
            open_questions=(unknown.question,),
        )
        store.add_publication(publication)
        return AutonomyExecutionResult(
            status="executed",
            queue_item_kind=item.kind,
            queue_item_id=item.id,
            message="Opened a bounded rabbit hole and continuation publication for the top unknown.",
        )

    if item.kind == "opportunity":
        opportunity = store.load_opportunity(item.id)
        change_mode = (
            opportunity.change_modes[0]
            if opportunity.change_modes
            else ("improve" if opportunity.priority in {"high", "medium"} else "build_on")
        )
        hypothesis = Hypothesis(
            id=store.new_id("hyp", opportunity.title),
            opportunity_id=opportunity.id,
            claim=f"Bounded follow-up on {opportunity.title} will improve the current research trajectory.",
            expected_win=opportunity.summary
            or f"Narrow {opportunity.title} into a testable next step.",
            risk_level="low" if opportunity.priority != "high" else "medium",
            change_mode=change_mode,
            created_by="autonomy",
        )
        store.add_hypothesis(hypothesis)
        return AutonomyExecutionResult(
            status="executed",
            queue_item_kind=item.kind,
            queue_item_id=item.id,
            message="Created a bounded hypothesis for the top opportunity.",
        )

    if item.kind == "hypothesis":
        hypothesis = store.load_hypothesis(item.id)
        plan = ExperimentPlan(
            id=store.new_id("plan", hypothesis.id),
            hypothesis_id=hypothesis.id,
            plan=f"Test the hypothesis: {hypothesis.claim}",
            checks=("pytest",),
            eval_slices=("workflow-smoke",),
            created_by="autonomy",
        )
        store.add_experiment_plan(plan)
        return AutonomyExecutionResult(
            status="executed",
            queue_item_kind=item.kind,
            queue_item_id=item.id,
            message="Created a bounded experiment plan for the top hypothesis.",
        )

    if item.kind == "experiment_plan":
        plan = store.load_experiment_plan(item.id)
        experiment, result = run_experiment_plan(
            store=store,
            plan=plan,
            cwd=cwd,
            created_by="autonomy",
        )
        store.add_experiment(experiment, result)
        return AutonomyExecutionResult(
            status="executed",
            queue_item_kind=item.kind,
            queue_item_id=item.id,
            message=f"Ran the top experiment plan and recorded a {result.status} result.",
        )

    if item.kind == "experiment_result":
        experiment = store.load_experiment(item.id)
        result = store.load_experiment_result(item.id)
        plan = store.load_experiment_plan(experiment.plan_id)
        hypothesis = store.load_hypothesis(plan.hypothesis_id)
        opportunity = store.load_opportunity(hypothesis.opportunity_id)
        subsystem = (
            opportunity.related_sections[0]
            if opportunity.related_sections
            else opportunity.title.strip().lower().replace(" ", "-")
        )
        candidate = PromotionCandidate(
            id=store.new_id("promo", opportunity.title),
            title=f"{opportunity.title} promotion candidate",
            summary=(
                f"Promote the passed experiment for {opportunity.title}. "
                f"Evidence: {hypothesis.expected_win or hypothesis.claim}"
            ),
            source_hypotheses=(hypothesis.id,),
            target_files=plan.target_files,
            expected_metric=hypothesis.expected_win or "Promotable experiment result recorded.",
            validation_plan=(
                "Re-run the bounded experiment checks and eval slices before opening a PR."
            ),
            risk_level=hypothesis.risk_level or "medium",
            created_by="autonomy",
            change_intent=store.parse_change_intent(
                mode=hypothesis.change_mode or "improve",
                subsystem=subsystem,
                rationale=hypothesis.claim,
                expected_outcome=hypothesis.expected_win or opportunity.summary or hypothesis.claim,
                risk=hypothesis.risk_level or "medium",
            ),
        )
        store.add_promotion_candidate(candidate)
        return AutonomyExecutionResult(
            status="executed",
            queue_item_kind=item.kind,
            queue_item_id=item.id,
            message=(
                f"Created a promotion candidate from the passed experiment result {result.experiment_id}."
            ),
        )

    if item.kind != "promotion_candidate":
        return AutonomyExecutionResult(
            status="deferred",
            queue_item_kind=item.kind,
            queue_item_id=item.id,
            message=(
                "Top queue item is exploratory work. "
                "Autonomous execution currently only advances promotion candidates."
            ),
        )

    candidate = store.load_promotion_candidate(item.id)
    if _risk_value(candidate.risk_level) > _risk_value(max_risk):
        return AutonomyExecutionResult(
            status="deferred",
            queue_item_kind=item.kind,
            queue_item_id=item.id,
            message=(
                f"Candidate risk {candidate.risk_level!r} exceeds max allowed risk {max_risk!r}."
            ),
        )

    draft = build_promotion_draft(candidate, base_branch=base_branch)
    candidate_dir = store.promotion_candidates_dir / candidate.id
    json_path, body_path = write_promotion_draft(draft=draft, target_dir=candidate_dir)
    issues = _review_promotion_artifacts(candidate=candidate, draft=draft)
    _write_promotion_review(target_dir=candidate_dir, candidate=candidate, issues=issues)
    if issues:
        opportunity = _create_revision_opportunity(store=store, candidate=candidate, issues=issues)
        store.add_opportunity(opportunity)
        return AutonomyExecutionResult(
            status="deferred",
            queue_item_kind=item.kind,
            queue_item_id=item.id,
            message=(
                "Promotion artifacts were generated but failed autonomous review: "
                + "; ".join(issues)
                + f". Created follow-up opportunity {opportunity.id}."
            ),
            branch_name=draft.branch_name,
            draft_json=json_path,
            pr_body=body_path,
        )

    if create_branch:
        ensure_branch(cwd=cwd, branch_name=draft.branch_name, base_branch=base_branch)
    if commit:
        commit_paths(cwd=cwd, message=draft.commit_message, paths=candidate.target_files)
    if push:
        push_branch(cwd=cwd, branch_name=draft.branch_name)
    if open_pr:
        create_pull_request(
            cwd=cwd,
            title=draft.pr_title,
            body_path=body_path,
            base_branch=base_branch,
            head_branch=draft.branch_name,
            draft=draft_pr,
        )

    return AutonomyExecutionResult(
        status="executed",
        queue_item_kind=item.kind,
        queue_item_id=item.id,
        message="Prepared bounded promotion artifacts for the top queue item.",
        branch_name=draft.branch_name,
        draft_json=json_path,
        pr_body=body_path,
    )


def execute_research_burst(
    *,
    store: ResearchStore,
    cwd: Path,
    max_steps: int = 5,
    max_risk: str = "medium",
    base_branch: str = "main",
    create_branch: bool = False,
    commit: bool = False,
    push: bool = False,
    open_pr: bool = False,
    draft_pr: bool = True,
) -> AutonomyBurstResult:
    results: list[AutonomyExecutionResult] = []
    for _ in range(max_steps):
        result = execute_next_research_item(
            store=store,
            cwd=cwd,
            max_risk=max_risk,
            base_branch=base_branch,
            create_branch=create_branch,
            commit=commit,
            push=push,
            open_pr=open_pr,
            draft_pr=draft_pr,
        )
        results.append(result)
        if result.status == "no_work":
            return AutonomyBurstResult(
                status="completed",
                steps_run=len(results),
                stop_reason="no_work",
                results=tuple(results),
            )
        if result.status == "deferred":
            return AutonomyBurstResult(
                status="paused",
                steps_run=len(results),
                stop_reason="deferred",
                results=tuple(results),
            )
    return AutonomyBurstResult(
        status="paused",
        steps_run=len(results),
        stop_reason="max_steps",
        results=tuple(results),
    )


def write_autonomy_run_record(
    *,
    store: ResearchStore,
    cwd: Path,
    mode: str,
    result: AutonomyBurstResult,
) -> Path:
    run_id = store.new_id("autorun", mode)
    target = store.root / "autonomy-runs" / run_id
    target.mkdir(parents=True, exist_ok=True)
    record = AutonomyRunRecord(
        id=run_id,
        mode=mode,
        status=result.status,
        stop_reason=result.stop_reason,
        steps_run=result.steps_run,
        cwd=str(cwd),
        created_at=_utcnow(),
        results=result.results,
    )
    (target / "run.json").write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    lines = [
        f"# Autonomy Run {record.id}",
        "",
        f"- mode: `{record.mode}`",
        f"- status: `{record.status}`",
        f"- stop_reason: `{record.stop_reason}`",
        f"- steps_run: `{record.steps_run}`",
        f"- cwd: `{record.cwd}`",
        f"- created_at: `{record.created_at}`",
        "",
        "## Steps",
    ]
    for index, step in enumerate(record.results, start=1):
        lines.append(f"{index}. `{step.status}` {step.message}")
        if step.queue_item_kind and step.queue_item_id:
            lines.append(f"   - queue_item: `{step.queue_item_kind}:{step.queue_item_id}`")
    (target / "RUN.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return target


def run_scheduled_research_burst(
    *,
    store: ResearchStore,
    cwd: Path,
    max_steps: int = 5,
    max_risk: str = "medium",
    base_branch: str = "main",
    create_branch: bool = False,
    commit: bool = False,
    push: bool = False,
    open_pr: bool = False,
    draft_pr: bool = True,
) -> tuple[AutonomyBurstResult, Path]:
    result = execute_research_burst(
        store=store,
        cwd=cwd,
        max_steps=max_steps,
        max_risk=max_risk,
        base_branch=base_branch,
        create_branch=create_branch,
        commit=commit,
        push=push,
        open_pr=open_pr,
        draft_pr=draft_pr,
    )
    record_dir = write_autonomy_run_record(
        store=store,
        cwd=cwd,
        mode="burst",
        result=result,
    )
    return result, record_dir


__all__ = [
    "AutonomyBurstResult",
    "AutonomyExecutionResult",
    "AutonomyRunRecord",
    "execute_next_research_item",
    "execute_research_burst",
    "run_scheduled_research_burst",
    "write_autonomy_run_record",
]
