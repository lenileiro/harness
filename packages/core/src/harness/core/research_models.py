from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

ChangeMode = Literal["build_on", "improve", "extend", "rework"]
ResearchStatus = Literal["open", "paused", "published", "abandoned"]
PublicationStatus = Literal["exploratory", "promising", "superseded", "rejected", "promoted"]
ThemeStatus = Literal["active", "paused", "archived"]
UnknownStatus = Literal["open", "narrowing", "resolved", "rejected"]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class ChangeIntent:
    mode: ChangeMode
    subsystem: str
    rationale: str
    expected_outcome: str
    risk: str = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "subsystem": self.subsystem,
            "rationale": self.rationale,
            "expected_outcome": self.expected_outcome,
            "risk": self.risk,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ChangeIntent | None:
        if not data:
            return None
        return cls(
            mode=str(data["mode"]),  # type: ignore[arg-type]
            subsystem=str(data.get("subsystem") or "").strip(),
            rationale=str(data.get("rationale") or "").strip(),
            expected_outcome=str(data.get("expected_outcome") or "").strip(),
            risk=str(data.get("risk") or "medium").strip() or "medium",
        )


@dataclass(frozen=True, slots=True)
class Vision:
    id: str
    title: str
    summary: str
    themes: tuple[str, ...] = ()
    success_metrics: tuple[str, ...] = ()
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "themes": list(self.themes),
            "success_metrics": list(self.success_metrics),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Vision:
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or "").strip(),
            summary=str(data.get("summary") or "").strip(),
            themes=tuple(str(item).strip() for item in data.get("themes") or []),
            success_metrics=tuple(str(item).strip() for item in data.get("success_metrics") or []),
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
        )


@dataclass(frozen=True, slots=True)
class Theme:
    id: str
    vision_id: str
    title: str
    description: str
    priority: str = "medium"
    status: ThemeStatus = "active"
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "vision_id": self.vision_id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Theme:
        return cls(
            id=str(data["id"]),
            vision_id=str(data.get("vision_id") or "").strip(),
            title=str(data.get("title") or "").strip(),
            description=str(data.get("description") or "").strip(),
            priority=str(data.get("priority") or "medium").strip() or "medium",
            status=str(data.get("status") or "active"),  # type: ignore[arg-type]
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
        )


@dataclass(frozen=True, slots=True)
class Unknown:
    id: str
    theme_id: str
    question: str
    why_it_matters: str
    current_belief: str = ""
    confidence: float = 0.0
    status: UnknownStatus = "open"
    related_sections: tuple[str, ...] = ()
    created_by: str = "human"
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "theme_id": self.theme_id,
            "question": self.question,
            "why_it_matters": self.why_it_matters,
            "current_belief": self.current_belief,
            "confidence": self.confidence,
            "status": self.status,
            "related_sections": list(self.related_sections),
            "created_by": self.created_by,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Unknown:
        return cls(
            id=str(data["id"]),
            theme_id=str(data.get("theme_id") or "").strip(),
            question=str(data.get("question") or "").strip(),
            why_it_matters=str(data.get("why_it_matters") or "").strip(),
            current_belief=str(data.get("current_belief") or "").strip(),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            status=str(data.get("status") or "open"),  # type: ignore[arg-type]
            related_sections=tuple(
                str(item).strip() for item in data.get("related_sections") or []
            ),
            created_by=str(data.get("created_by") or "human").strip() or "human",
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
        )


@dataclass(frozen=True, slots=True)
class RabbitHole:
    id: str
    title: str
    question: str
    scope: str
    theme: str
    related_sections: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    opened_by: str = "human"
    status: ResearchStatus = "open"
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    change_intent: ChangeIntent | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "question": self.question,
            "scope": self.scope,
            "theme": self.theme,
            "related_sections": list(self.related_sections),
            "tags": list(self.tags),
            "opened_by": self.opened_by,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "change_intent": self.change_intent.to_dict() if self.change_intent else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RabbitHole:
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or "").strip(),
            question=str(data.get("question") or "").strip(),
            scope=str(data.get("scope") or "").strip(),
            theme=str(data.get("theme") or "").strip(),
            related_sections=tuple(
                str(item).strip() for item in data.get("related_sections") or []
            ),
            tags=tuple(str(item).strip() for item in data.get("tags") or []),
            opened_by=str(data.get("opened_by") or "human").strip() or "human",
            status=str(data.get("status") or "open"),  # type: ignore[arg-type]
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
            change_intent=ChangeIntent.from_dict(data.get("change_intent")),
        )


@dataclass(frozen=True, slots=True)
class Publication:
    id: str
    rabbit_hole_id: str
    title: str
    summary: str
    claims: tuple[str, ...] = ()
    supporting_evidence: tuple[str, ...] = ()
    counterevidence: tuple[str, ...] = ()
    recommendations: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    confidence: float = 1.0
    status: PublicationStatus = "exploratory"
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    change_intent: ChangeIntent | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "rabbit_hole_id": self.rabbit_hole_id,
            "title": self.title,
            "summary": self.summary,
            "claims": list(self.claims),
            "supporting_evidence": list(self.supporting_evidence),
            "counterevidence": list(self.counterevidence),
            "recommendations": list(self.recommendations),
            "open_questions": list(self.open_questions),
            "sources": list(self.sources),
            "artifacts": list(self.artifacts),
            "citations": list(self.citations),
            "confidence": self.confidence,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "change_intent": self.change_intent.to_dict() if self.change_intent else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Publication:
        return cls(
            id=str(data["id"]),
            rabbit_hole_id=str(data.get("rabbit_hole_id") or "").strip(),
            title=str(data.get("title") or "").strip(),
            summary=str(data.get("summary") or "").strip(),
            claims=tuple(str(item).strip() for item in data.get("claims") or []),
            supporting_evidence=tuple(
                str(item).strip() for item in data.get("supporting_evidence") or []
            ),
            counterevidence=tuple(str(item).strip() for item in data.get("counterevidence") or []),
            recommendations=tuple(str(item).strip() for item in data.get("recommendations") or []),
            open_questions=tuple(str(item).strip() for item in data.get("open_questions") or []),
            sources=tuple(str(item).strip() for item in data.get("sources") or []),
            artifacts=tuple(str(item).strip() for item in data.get("artifacts") or []),
            citations=tuple(str(item).strip() for item in data.get("citations") or []),
            confidence=float(data.get("confidence", 1.0) or 1.0),
            status=str(data.get("status") or "exploratory"),  # type: ignore[arg-type]
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
            change_intent=ChangeIntent.from_dict(data.get("change_intent")),
        )


__all__ = [
    "ChangeIntent",
    "ChangeMode",
    "Publication",
    "PublicationStatus",
    "RabbitHole",
    "ResearchStatus",
    "Theme",
    "ThemeStatus",
    "Unknown",
    "UnknownStatus",
    "Vision",
]
