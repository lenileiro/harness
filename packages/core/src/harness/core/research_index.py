from __future__ import annotations

from dataclasses import dataclass

from harness.core.research_store import ResearchSearchHit, ResearchStore


@dataclass(frozen=True, slots=True)
class ResearchIndex:
    store: ResearchStore

    def search(self, query: str, *, limit: int = 10) -> list[ResearchSearchHit]:
        return self.store.search(query, limit=limit)


__all__ = ["ResearchIndex"]
