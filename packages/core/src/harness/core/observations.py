from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class Observation:
    id: str
    title: str
    summary: str
    source_type: str
    source_ref: str = ""
    related_sections: tuple[str, ...] = ()
    theme: str = ""
    created_by: str = "human"
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "related_sections": list(self.related_sections),
            "theme": self.theme,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation:
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or "").strip(),
            summary=str(data.get("summary") or "").strip(),
            source_type=str(data.get("source_type") or "").strip(),
            source_ref=str(data.get("source_ref") or "").strip(),
            related_sections=tuple(
                str(item).strip() for item in data.get("related_sections") or []
            ),
            theme=str(data.get("theme") or "").strip(),
            created_by=str(data.get("created_by") or "human").strip() or "human",
            created_at=str(data.get("created_at") or _utcnow()),
        )


__all__ = ["Observation"]
