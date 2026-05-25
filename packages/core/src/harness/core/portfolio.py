from __future__ import annotations

from dataclasses import dataclass

from harness.core.research_store import ResearchStore


@dataclass(frozen=True, slots=True)
class PortfolioSnapshot:
    themes: int
    unknowns: int
    rabbit_holes: int
    publications: int
    opportunities: int
    hypotheses: int
    experiment_plans: int
    experiments: int
    promotion_candidates: int
    archived_items: int
    inspiration_notes: int


def build_portfolio_snapshot(store: ResearchStore) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        themes=len(store.list_themes()),
        unknowns=len(store.list_unknowns()),
        rabbit_holes=len(store.list_rabbit_holes()),
        publications=len(store.list_publications()),
        opportunities=len(store.list_opportunities()),
        hypotheses=len(store.list_hypotheses()),
        experiment_plans=len(store.list_experiment_plans()),
        experiments=len(store.list_experiments()),
        promotion_candidates=len(store.list_promotion_candidates()),
        archived_items=len(store.list_archive_items()),
        inspiration_notes=len(store.list_inspiration_notes()),
    )


__all__ = ["PortfolioSnapshot", "build_portfolio_snapshot"]
