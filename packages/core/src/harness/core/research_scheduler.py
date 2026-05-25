from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from harness.core.research_store import ResearchStore


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


def build_research_queue(store: ResearchStore) -> list[ResearchQueueItem]:
    items: list[ResearchQueueItem] = []
    for candidate in store.list_promotion_candidates():
        items.append(
            ResearchQueueItem(
                kind="promotion_candidate",
                id=candidate.id,
                priority=100,
                summary=candidate.summary,
            )
        )
    for unknown in store.list_unknowns(status="open"):
        items.append(
            ResearchQueueItem(
                kind="unknown",
                id=unknown.id,
                priority=80,
                summary=unknown.question,
            )
        )
    for opportunity in store.list_opportunities():
        items.append(
            ResearchQueueItem(
                kind="opportunity",
                id=opportunity.id,
                priority=60 if opportunity.priority == "high" else 40,
                summary=opportunity.title,
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
    "rank_promotion_candidates",
    "rebalance_research_queue",
    "suggest_opportunities",
    "suggest_unknowns",
    "surface_stale_publications",
]
