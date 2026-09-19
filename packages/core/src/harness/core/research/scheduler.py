from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime

from harness.core.experiments import Experiment
from harness.core.research.store import ResearchStore


@dataclass(frozen=True, slots=True)
class ResearchQueueItem:
    kind: str
    id: str
    priority: int
    summary: str


@dataclass(frozen=True, slots=True)
class PatternFinding:
    label: str
    count: int


def promotion_completed_stages(store: ResearchStore, candidate_id: str) -> dict[str, bool]:
    """Read durable completed actions, including the former prepared-only format."""
    path = store.promotion_candidates_dir / candidate_id / "promotion_execution.json"
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    stages = payload.get("stages")
    if isinstance(stages, dict):
        return {str(name): value is True for name, value in stages.items()}
    completed = {
        name: payload.get(name) is True for name in ("branch", "commit", "push", "open_pr")
    }
    completed["prepare"] = payload.get("status") == "prepared"
    return completed


def _latest_unambiguous_experiments(experiments: list[Experiment]) -> list[Experiment]:
    latest: dict[str, tuple[datetime, Experiment]] = {}
    tied: set[str] = set()
    invalid: set[str] = set()
    for experiment in experiments:
        try:
            timestamp = datetime.fromisoformat(experiment.created_at)
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=UTC)
        except ValueError:
            invalid.add(experiment.plan_id)
            continue
        previous = latest.get(experiment.plan_id)
        if previous is None or timestamp > previous[0]:
            latest[experiment.plan_id] = (timestamp, experiment)
            tied.discard(experiment.plan_id)
        elif timestamp == previous[0]:
            tied.add(experiment.plan_id)
    return [item for plan_id, (_, item) in latest.items() if plan_id not in tied | invalid]


def build_research_queue(
    store: ResearchStore, *, promotion_actions: tuple[str, ...] = ("prepare",)
) -> list[ResearchQueueItem]:
    items: list[ResearchQueueItem] = []
    hypotheses = store.list_hypotheses()
    plans = store.list_experiment_plans()
    continued_opportunities = {item.opportunity_id for item in hypotheses}
    planned_hypotheses = {item.hypothesis_id for item in plans}
    continued_unknowns = {item.source_unknown_id for item in store.list_rabbit_holes()}
    candidates = store.list_promotion_candidates()
    active_candidates = []
    for candidate in candidates:
        completed = promotion_completed_stages(store, candidate.id)
        if all(completed.get(action, False) for action in promotion_actions):
            continue
        review_path = store.promotion_candidates_dir / candidate.id / "promotion_review.json"
        if review_path.is_file():
            payload = json.loads(review_path.read_text(encoding="utf-8"))
            if str(payload.get("status") or "").strip() == "failed":
                continue
        active_candidates.append(candidate)
    promoted_hypotheses = {
        hypothesis_id for candidate in candidates for hypothesis_id in candidate.source_hypotheses
    }
    for candidate in active_candidates:
        items.append(
            ResearchQueueItem(
                kind="promotion_candidate",
                id=candidate.id,
                priority=100,
                summary=candidate.summary,
            )
        )
    for unknown in store.list_unknowns(status="open"):
        if unknown.id in continued_unknowns:
            continue
        items.append(
            ResearchQueueItem(
                kind="unknown",
                id=unknown.id,
                priority=80,
                summary=unknown.question,
            )
        )
    for opportunity in store.list_opportunities():
        if opportunity.id in continued_opportunities:
            continue
        items.append(
            ResearchQueueItem(
                kind="opportunity",
                id=opportunity.id,
                priority=60 if opportunity.priority == "high" else 40,
                summary=opportunity.title,
            )
        )
    for hypothesis in hypotheses:
        if hypothesis.id in promoted_hypotheses or hypothesis.id in planned_hypotheses:
            continue
        items.append(
            ResearchQueueItem(
                kind="hypothesis",
                id=hypothesis.id,
                priority=70 if hypothesis.risk_level == "low" else 50,
                summary=hypothesis.claim,
            )
        )
    experiments = store.list_experiments()
    executed_plan_ids = {experiment.plan_id for experiment in experiments}
    for plan in plans:
        if plan.id in executed_plan_ids:
            continue
        items.append(
            ResearchQueueItem(
                kind="experiment_plan",
                id=plan.id,
                priority=75,
                summary=plan.plan,
            )
        )
    plans_by_id = {plan.id: plan for plan in plans}
    for experiment in _latest_unambiguous_experiments(experiments):
        try:
            result = store.load_experiment_result(experiment.id)
        except (OSError, ValueError):
            # A partial/missing latest record cannot authorize an older pass.
            continue
        if result.status != "passed":
            continue
        plan = plans_by_id.get(experiment.plan_id)
        if plan is None or plan.hypothesis_id in promoted_hypotheses:
            continue
        items.append(
            ResearchQueueItem(
                kind="experiment_result",
                id=experiment.id,
                priority=85,
                summary=result.status,
            )
        )
    return sorted(items, key=lambda item: (-item.priority, item.id))


def rebalance_research_queue(store: ResearchStore) -> dict[str, int]:
    return {
        "themes": len(store.list_themes()),
        "open_unknowns": len(store.list_unknowns(status="open")),
        "opportunities": len(store.list_opportunities()),
        "promotion_candidates": len(store.list_promotion_candidates()),
        "archived_items": len(store.list_archive_items()),
    }


def mine_new_failures(store: ResearchStore) -> list[str]:
    return [item.reason for item in store.list_archive_items() if item.reason]


def mine_new_successes(store: ResearchStore) -> list[str]:
    return [publication.summary for publication in store.list_publications() if publication.summary]


def discover_repeated_patterns(store: ResearchStore) -> list[PatternFinding]:
    counts = Counter(item.reason for item in store.list_archive_items() if item.reason)
    counts.update(
        unknown.question for unknown in store.list_unknowns(status="open") if unknown.question
    )
    return [
        PatternFinding(label=label, count=count)
        for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count > 1
    ]


def suggest_unknowns(store: ResearchStore) -> list[str]:
    suggestions: list[str] = []
    theme_ids = {theme.id for theme in store.list_themes()}
    existing_questions = {unknown.question for unknown in store.list_unknowns()}
    for note in store.list_inspiration_notes():
        for theme in note.related_themes:
            if theme in theme_ids:
                question = f"How should Harness react to inspiration from {note.source.kind}: {note.title}?"
                if question not in existing_questions:
                    suggestions.append(question)
    return suggestions


def suggest_opportunities(store: ResearchStore) -> list[str]:
    existing_titles = {opportunity.title for opportunity in store.list_opportunities()}
    suggestions: list[str] = []
    for note in store.list_inspiration_notes():
        title = f"Incorporate {note.source.kind} inspiration: {note.title}"
        if title not in existing_titles:
            suggestions.append(title)
    return suggestions


def surface_stale_publications(store: ResearchStore) -> list[str]:
    return [
        publication.id
        for publication in store.list_publications()
        if publication.status == "exploratory"
    ]


def rank_promotion_candidates(store: ResearchStore) -> list[str]:
    ranked = sorted(
        store.list_promotion_candidates(),
        key=lambda item: (
            0 if item.risk_level == "low" else 1,
            -(len(item.source_publications) + len(item.source_hypotheses)),
            item.id,
        ),
    )
    return [item.id for item in ranked]


__all__ = [
    "PatternFinding",
    "ResearchQueueItem",
    "build_research_queue",
    "discover_repeated_patterns",
    "mine_new_failures",
    "mine_new_successes",
    "promotion_completed_stages",
    "rank_promotion_candidates",
    "rebalance_research_queue",
    "suggest_opportunities",
    "suggest_unknowns",
    "surface_stale_publications",
]
