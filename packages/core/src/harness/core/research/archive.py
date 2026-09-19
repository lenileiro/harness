from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class ArchivedResearchItem:
    archive_id: str
    kind: str
    original_id: str
    original_relpath: str
    reason: str
    note: str = ""
    archived_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "archive_id": self.archive_id,
            "kind": self.kind,
            "original_id": self.original_id,
            "original_relpath": self.original_relpath,
            "reason": self.reason,
            "note": self.note,
            "archived_at": self.archived_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArchivedResearchItem:
        return cls(
            archive_id=str(data["archive_id"]),
            kind=str(data.get("kind") or "").strip(),
            original_id=str(data.get("original_id") or "").strip(),
            original_relpath=str(data.get("original_relpath") or "").strip(),
            reason=str(data.get("reason") or "").strip(),
            note=str(data.get("note") or "").strip(),
            archived_at=str(data.get("archived_at") or _utcnow()),
        )


@dataclass(frozen=True, slots=True)
class RejectedIdea:
    item: ArchivedResearchItem


@dataclass(frozen=True, slots=True)
class SupersededPublication:
    item: ArchivedResearchItem


__all__ = ["ArchivedResearchItem", "RejectedIdea", "SupersededPublication"]
