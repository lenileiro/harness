from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class Opportunity:
    id: str
    title: str
    summary: str
    mission_id: str = ""
    mission_feature_id: str = ""
    related_sections: tuple[str, ...] = ()
    origin_observations: tuple[str, ...] = ()
    change_modes: tuple[str, ...] = ()
    theme: str = ""
    priority: str = "medium"
    created_by: str = "human"
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "mission_id": self.mission_id,
            "mission_feature_id": self.mission_feature_id,
            "related_sections": list(self.related_sections),
            "origin_observations": list(self.origin_observations),
            "change_modes": list(self.change_modes),
            "theme": self.theme,
            "priority": self.priority,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Opportunity:
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or "").strip(),
            summary=str(data.get("summary") or "").strip(),
            mission_id=str(data.get("mission_id") or "").strip(),
            mission_feature_id=str(data.get("mission_feature_id") or "").strip(),
            related_sections=tuple(
                str(item).strip() for item in data.get("related_sections") or []
            ),
            origin_observations=tuple(
                str(item).strip() for item in data.get("origin_observations") or []
            ),
            change_modes=tuple(str(item).strip() for item in data.get("change_modes") or []),
            theme=str(data.get("theme") or "").strip(),
            priority=str(data.get("priority") or "medium").strip() or "medium",
            created_by=str(data.get("created_by") or "human").strip() or "human",
            created_at=str(data.get("created_at") or _utcnow()),
        )


__all__ = ["Opportunity"]
