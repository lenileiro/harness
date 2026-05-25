from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class ExternalSource:
    kind: str
    ref: str
    title: str = ""
    excerpt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "ref": self.ref,
            "title": self.title,
            "excerpt": self.excerpt,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExternalSource:
        return cls(
            kind=str(data.get("kind") or "").strip(),
            ref=str(data.get("ref") or "").strip(),
            title=str(data.get("title") or "").strip(),
            excerpt=str(data.get("excerpt") or "").strip(),
        )


@dataclass(frozen=True, slots=True)
class InspirationNote:
    id: str
    title: str
    summary: str
    source: ExternalSource
    related_themes: tuple[str, ...] = ()
    related_sections: tuple[str, ...] = ()
    created_by: str = "human"
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "source": self.source.to_dict(),
            "related_themes": list(self.related_themes),
            "related_sections": list(self.related_sections),
            "created_by": self.created_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InspirationNote:
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or "").strip(),
            summary=str(data.get("summary") or "").strip(),
            source=ExternalSource.from_dict(data.get("source") or {}),
            related_themes=tuple(str(item).strip() for item in data.get("related_themes") or []),
            related_sections=tuple(
                str(item).strip() for item in data.get("related_sections") or []
            ),
            created_by=str(data.get("created_by") or "human").strip() or "human",
            created_at=str(data.get("created_at") or _utcnow()),
        )


__all__ = ["ExternalSource", "InspirationNote"]
