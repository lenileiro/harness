from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from harness.core.research_models import ChangeIntent


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class PromotionCandidate:
    id: str
    title: str
    summary: str
    mission_id: str = ""
    mission_feature_ids: tuple[str, ...] = ()
    source_publications: tuple[str, ...] = ()
    source_hypotheses: tuple[str, ...] = ()
    target_files: tuple[str, ...] = ()
    expected_metric: str = ""
    validation_plan: str = ""
    risk_level: str = "medium"
    created_by: str = "human"
    created_at: str = field(default_factory=_utcnow)
    change_intent: ChangeIntent | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "mission_id": self.mission_id,
            "mission_feature_ids": list(self.mission_feature_ids),
            "source_publications": list(self.source_publications),
            "source_hypotheses": list(self.source_hypotheses),
            "target_files": list(self.target_files),
            "expected_metric": self.expected_metric,
            "validation_plan": self.validation_plan,
            "risk_level": self.risk_level,
            "created_by": self.created_by,
            "created_at": self.created_at,
            "change_intent": self.change_intent.to_dict() if self.change_intent else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PromotionCandidate:
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or "").strip(),
            summary=str(data.get("summary") or "").strip(),
            mission_id=str(data.get("mission_id") or "").strip(),
            mission_feature_ids=tuple(
                str(item).strip() for item in data.get("mission_feature_ids") or []
            ),
            source_publications=tuple(
                str(item).strip() for item in data.get("source_publications") or []
            ),
            source_hypotheses=tuple(
                str(item).strip() for item in data.get("source_hypotheses") or []
            ),
            target_files=tuple(str(item).strip() for item in data.get("target_files") or []),
            expected_metric=str(data.get("expected_metric") or "").strip(),
            validation_plan=str(data.get("validation_plan") or "").strip(),
            risk_level=str(data.get("risk_level") or "medium").strip() or "medium",
            created_by=str(data.get("created_by") or "human").strip() or "human",
            created_at=str(data.get("created_at") or _utcnow()),
            change_intent=ChangeIntent.from_dict(data.get("change_intent")),
        )


__all__ = ["PromotionCandidate"]
