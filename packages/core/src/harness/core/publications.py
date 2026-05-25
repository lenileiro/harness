from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.core.research_models import Publication


@dataclass(frozen=True, slots=True)
class ResearchAsset:
    id: str
    kind: str
    path: str
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "path": self.path,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResearchAsset:
        return cls(
            id=str(data["id"]),
            kind=str(data.get("kind") or "").strip(),
            path=str(data.get("path") or "").strip(),
            description=str(data.get("description") or "").strip(),
        )


def summarize_publication(publication: Publication) -> list[str]:
    lines = [publication.title, publication.summary]
    if publication.claims:
        lines.extend(f"claim: {item}" for item in publication.claims)
    if publication.recommendations:
        lines.extend(f"recommendation: {item}" for item in publication.recommendations)
    return lines


__all__ = ["ResearchAsset", "summarize_publication"]
