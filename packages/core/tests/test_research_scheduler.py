from __future__ import annotations

from harness.core.inspiration import ExternalSource, InspirationNote
from harness.core.promotion_candidates import PromotionCandidate
from harness.core.research_models import Publication, Theme, Unknown
from harness.core.research_scheduler import (
    build_research_queue,
    discover_repeated_patterns,
    mine_new_failures,
    mine_new_successes,
    rank_promotion_candidates,
    suggest_opportunities,
    suggest_unknowns,
    surface_stale_publications,
)
from harness.core.research_store import ResearchStore, default_research_root


def test_scheduler_helpers_mine_and_rank(tmp_path) -> None:
    store = ResearchStore(root=default_research_root(tmp_path))
    store.add_theme(
        Theme(
            id="theme-auto",
            vision_id="current",
            title="Autonomous improvement",
            description="Study safe self-improvement loops.",
        )
    )
    store.add_unknown(
        Unknown(
            id="unknown-1",
            theme_id="theme-auto",
            question="What change classes are safe?",
            why_it_matters="Promotion needs a bounded first lane.",
        )
    )
    store.add_publication(
        Publication(
            id="pub-1",
            rabbit_hole_id="rh-1",
            title="Research completion findings",
            summary="Repo-first research plus loop detection improves completion.",
            status="exploratory",
        )
    )
    store.add_promotion_candidate(
        PromotionCandidate(
            id="promo-1",
            title="Research completion fix",
            summary="Promote the repo-first research and loop-detection change.",
            source_publications=("pub-1",),
            risk_level="low",
        )
    )
    store.add_inspiration_note(
        InspirationNote(
            id="insp-1",
            title="Trend note",
            summary="Tool registries and research memories are trending.",
            source=ExternalSource(kind="web", ref="https://example.com/trend"),
            related_themes=("theme-auto",),
        )
    )
    store.archive_item(kind="unknown", item_id="unknown-1", reason="duplicate unknown")
    store.add_unknown(
        Unknown(
            id="unknown-2",
            theme_id="theme-auto",
            question="duplicate unknown",
            why_it_matters="Repeated concern worth tracking.",
        )
    )

    assert build_research_queue(store)[0].kind == "promotion_candidate"
    assert mine_new_failures(store) == ["duplicate unknown"]
    assert mine_new_successes(store)[0].startswith("Repo-first research")
    assert discover_repeated_patterns(store)[0].label == "duplicate unknown"
    assert suggest_unknowns(store)
    assert suggest_opportunities(store)
    assert surface_stale_publications(store) == ["pub-1"]
    assert rank_promotion_candidates(store) == ["promo-1"]
