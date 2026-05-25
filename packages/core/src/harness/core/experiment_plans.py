from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    id: str
    hypothesis_id: str
    plan: str
    target_files: tuple[str, ...] = ()
    checks: tuple[str, ...] = ()
    eval_slices: tuple[str, ...] = ()
    created_by: str = "human"
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "hypothesis_id": self.hypothesis_id,
            "plan": self.plan,
            "target_files": list(self.target_files),
            "checks": list(self.checks),
            "eval_slices": list(self.eval_slices),
            "created_by": self.created_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentPlan:
        return cls(
            id=str(data["id"]),
            hypothesis_id=str(data.get("hypothesis_id") or "").strip(),
            plan=str(data.get("plan") or "").strip(),
            target_files=tuple(str(item).strip() for item in data.get("target_files") or []),
            checks=tuple(str(item).strip() for item in data.get("checks") or []),
            eval_slices=tuple(str(item).strip() for item in data.get("eval_slices") or []),
            created_by=str(data.get("created_by") or "human").strip() or "human",
            created_at=str(data.get("created_at") or _utcnow()),
        )


__all__ = ["ExperimentPlan"]
