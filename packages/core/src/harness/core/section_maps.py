from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class SectionMap:
    id: str
    section: str
    files: tuple[str, ...] = ()
    interfaces: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    weaknesses: tuple[str, ...] = ()
    opportunities: tuple[str, ...] = ()
    created_by: str = "human"
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "section": self.section,
            "files": list(self.files),
            "interfaces": list(self.interfaces),
            "constraints": list(self.constraints),
            "weaknesses": list(self.weaknesses),
            "opportunities": list(self.opportunities),
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SectionMap:
        return cls(
            id=str(data["id"]),
            section=str(data.get("section") or "").strip(),
            files=tuple(str(item).strip() for item in data.get("files") or []),
            interfaces=tuple(str(item).strip() for item in data.get("interfaces") or []),
            constraints=tuple(str(item).strip() for item in data.get("constraints") or []),
            weaknesses=tuple(str(item).strip() for item in data.get("weaknesses") or []),
            opportunities=tuple(str(item).strip() for item in data.get("opportunities") or []),
            created_by=str(data.get("created_by") or "human").strip() or "human",
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
        )


__all__ = ["SectionMap"]
