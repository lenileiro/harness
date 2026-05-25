from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class Hypothesis:
    id: str
    opportunity_id: str
    claim: str
    expected_win: str
    risk_level: str
    change_mode: str
    created_by: str = "human"
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "opportunity_id": self.opportunity_id,
            "claim": self.claim,
            "expected_win": self.expected_win,
            "risk_level": self.risk_level,
            "change_mode": self.change_mode,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hypothesis:
        return cls(
            id=str(data["id"]),
            opportunity_id=str(data.get("opportunity_id") or "").strip(),
            claim=str(data.get("claim") or "").strip(),
            expected_win=str(data.get("expected_win") or "").strip(),
            risk_level=str(data.get("risk_level") or "").strip(),
            change_mode=str(data.get("change_mode") or "").strip(),
            created_by=str(data.get("created_by") or "human").strip() or "human",
            created_at=str(data.get("created_at") or _utcnow()),
        )


__all__ = ["Hypothesis"]
