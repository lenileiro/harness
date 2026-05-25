from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

CitationRelationship = Literal["builds_on", "contradicts", "refines", "reuses", "supersedes"]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class Citation:
    id: str
    source_publication_id: str
    target_publication_id: str
    relationship: CitationRelationship
    note: str = ""
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_publication_id": self.source_publication_id,
            "target_publication_id": self.target_publication_id,
            "relationship": self.relationship,
            "note": self.note,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Citation:
        return cls(
            id=str(data["id"]),
            source_publication_id=str(data.get("source_publication_id") or "").strip(),
            target_publication_id=str(data.get("target_publication_id") or "").strip(),
            relationship=str(data.get("relationship") or "builds_on"),  # type: ignore[arg-type]
            note=str(data.get("note") or "").strip(),
            created_at=str(data.get("created_at") or _utcnow()),
        )


__all__ = ["Citation", "CitationRelationship"]
