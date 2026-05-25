from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from harness.core.experiment_plans import ExperimentPlan
from harness.core.experiment_runner import run_experiment_plan
from harness.core.hypotheses import Hypothesis
from harness.core.pr_generation import (
    build_promotion_draft,
    commit_paths,
    create_pull_request,
    ensure_branch,
    push_branch,
    write_promotion_draft,
)
from harness.core.research_models import Publication, RabbitHole
from harness.core.research_scheduler import build_research_queue
from harness.core.research_store import ResearchStore

_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


def _risk_value(level: str) -> int:
    return _RISK_ORDER.get(level.strip().lower(), _RISK_ORDER["high"])


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


__all__ = ["AutonomyExecutionResult", "execute_next_research_item"]
