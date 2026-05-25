from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResearchRole:
    name: str
    description: str


BUILTIN_RESEARCH_ROLES: tuple[ResearchRole, ...] = (
    ResearchRole(
        "section-investigator", "Map one subsystem deeply and surface local leverage points."
    ),
    ResearchRole(
        "scout", "Gather inspiration from repo artifacts, peers, papers, trends, and the web."
    ),
    ResearchRole(
        "synthesizer", "Link multiple sections and observations into cross-cutting opportunities."
    ),
    ResearchRole(
        "hypothesis-agent", "Propose multiple competing improvement angles for one opportunity."
    ),
    ResearchRole("experiment-agent", "Run one bounded experiment and capture concrete evidence."),
    ResearchRole("publisher", "Turn deep investigations into structured publications."),
    ResearchRole("challenger", "Retest or disprove prior claims and stale assumptions."),
    ResearchRole("refiner", "Simplify promising experiments into promotion candidates."),
    ResearchRole(
        "promotion-agent", "Prepare bounded branches, commits, and PR payloads with evidence."
    ),
    ResearchRole("archivist", "Archive weak or superseded work so it is not retried blindly."),
)


__all__ = ["BUILTIN_RESEARCH_ROLES", "ResearchRole"]
