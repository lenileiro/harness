from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

MissionStatus = Literal[
    "draft", "planned", "approved", "running", "blocked", "completed", "abandoned"
]
MilestoneStatus = Literal["pending", "active", "validated", "blocked", "completed"]
FeatureStatus = Literal["pending", "active", "handoff", "validated", "blocked", "completed"]
AssertionKind = Literal["behavior", "test", "review", "contract", "manual"]
FindingSeverity = Literal["info", "warning", "error"]
RunStatus = Literal["pending", "running", "completed", "failed"]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _clean_tuple(values: list[Any] | tuple[Any, ...] | None) -> tuple[str, ...]:
    return tuple(str(item).strip() for item in values or [] if str(item).strip())


@dataclass(frozen=True, slots=True)
class ValidationAssertion:
    id: str
    contract_id: str
    title: str
    description: str
    kind: AssertionKind
    verification_method: str
    covered_by_features: tuple[str, ...] = ()
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "contract_id": self.contract_id,
            "title": self.title,
            "description": self.description,
            "kind": self.kind,
            "verification_method": self.verification_method,
            "covered_by_features": list(self.covered_by_features),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationAssertion:
        return cls(
            id=str(data["id"]),
            contract_id=str(data.get("contract_id") or "").strip(),
            title=str(data.get("title") or "").strip(),
            description=str(data.get("description") or "").strip(),
            kind=str(data.get("kind") or "contract"),  # type: ignore[arg-type]
            verification_method=str(data.get("verification_method") or "").strip(),
            covered_by_features=_clean_tuple(data.get("covered_by_features")),
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
        )


@dataclass(frozen=True, slots=True)
class ValidationContract:
    id: str
    mission_id: str
    summary: str
    assertions: tuple[ValidationAssertion, ...] = ()
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mission_id": self.mission_id,
            "summary": self.summary,
            "assertions": [item.to_dict() for item in self.assertions],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationContract:
        return cls(
            id=str(data["id"]),
            mission_id=str(data.get("mission_id") or "").strip(),
            summary=str(data.get("summary") or "").strip(),
            assertions=tuple(
                ValidationAssertion.from_dict(item)
                for item in data.get("assertions") or []
                if isinstance(item, dict)
            ),
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
        )


@dataclass(frozen=True, slots=True)
class Mission:
    id: str
    title: str
    goal: str
    status: MissionStatus = "draft"
    created_by: str = "human"
    planner_model: str = ""
    worker_model: str = ""
    validator_model: str = ""
    reporter_model: str = ""
    planner_brief: str = ""
    worker_brief: str = ""
    validator_brief: str = ""
    reporter_brief: str = ""
    budget_tokens: int | None = None
    budget_runtime_minutes: int | None = None
    current_milestone_id: str = ""
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "goal": self.goal,
            "status": self.status,
            "created_by": self.created_by,
            "planner_model": self.planner_model,
            "worker_model": self.worker_model,
            "validator_model": self.validator_model,
            "reporter_model": self.reporter_model,
            "planner_brief": self.planner_brief,
            "worker_brief": self.worker_brief,
            "validator_brief": self.validator_brief,
            "reporter_brief": self.reporter_brief,
            "budget_tokens": self.budget_tokens,
            "budget_runtime_minutes": self.budget_runtime_minutes,
            "current_milestone_id": self.current_milestone_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Mission:
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or "").strip(),
            goal=str(data.get("goal") or "").strip(),
            status=str(data.get("status") or "draft"),  # type: ignore[arg-type]
            created_by=str(data.get("created_by") or "human").strip() or "human",
            planner_model=str(data.get("planner_model") or "").strip(),
            worker_model=str(data.get("worker_model") or "").strip(),
            validator_model=str(data.get("validator_model") or "").strip(),
            reporter_model=str(data.get("reporter_model") or "").strip(),
            planner_brief=str(data.get("planner_brief") or "").strip(),
            worker_brief=str(data.get("worker_brief") or "").strip(),
            validator_brief=str(data.get("validator_brief") or "").strip(),
            reporter_brief=str(data.get("reporter_brief") or "").strip(),
            budget_tokens=(
                int(data["budget_tokens"]) if data.get("budget_tokens") is not None else None
            ),
            budget_runtime_minutes=(
                int(data["budget_runtime_minutes"])
                if data.get("budget_runtime_minutes") is not None
                else None
            ),
            current_milestone_id=str(data.get("current_milestone_id") or "").strip(),
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
        )


@dataclass(frozen=True, slots=True)
class Milestone:
    id: str
    mission_id: str
    title: str
    summary: str
    status: MilestoneStatus = "pending"
    order: int = 1
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mission_id": self.mission_id,
            "title": self.title,
            "summary": self.summary,
            "status": self.status,
            "order": self.order,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Milestone:
        return cls(
            id=str(data["id"]),
            mission_id=str(data.get("mission_id") or "").strip(),
            title=str(data.get("title") or "").strip(),
            summary=str(data.get("summary") or "").strip(),
            status=str(data.get("status") or "pending"),  # type: ignore[arg-type]
            order=int(data.get("order") or 1),
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
        )


@dataclass(frozen=True, slots=True)
class MissionFeature:
    id: str
    mission_id: str
    milestone_id: str
    title: str
    summary: str
    status: FeatureStatus = "pending"
    depends_on: tuple[str, ...] = ()
    assigned_role: str = "worker"
    target_files: tuple[str, ...] = ()
    research_refs: tuple[str, ...] = ()
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mission_id": self.mission_id,
            "milestone_id": self.milestone_id,
            "title": self.title,
            "summary": self.summary,
            "status": self.status,
            "depends_on": list(self.depends_on),
            "assigned_role": self.assigned_role,
            "target_files": list(self.target_files),
            "research_refs": list(self.research_refs),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionFeature:
        return cls(
            id=str(data["id"]),
            mission_id=str(data.get("mission_id") or "").strip(),
            milestone_id=str(data.get("milestone_id") or "").strip(),
            title=str(data.get("title") or "").strip(),
            summary=str(data.get("summary") or "").strip(),
            status=str(data.get("status") or "pending"),  # type: ignore[arg-type]
            depends_on=_clean_tuple(data.get("depends_on")),
            assigned_role=str(data.get("assigned_role") or "worker").strip() or "worker",
            target_files=_clean_tuple(data.get("target_files")),
            research_refs=_clean_tuple(data.get("research_refs")),
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
        )


@dataclass(frozen=True, slots=True)
class MissionHandoff:
    id: str
    mission_id: str
    feature_id: str
    role: str
    completed_work: str
    role_model: str = ""
    remaining_work: str = ""
    commands_run: tuple[str, ...] = ()
    exit_codes: tuple[str, ...] = ()
    known_issues: tuple[str, ...] = ()
    next_recommendation: str = ""
    confidence: float = 0.0
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mission_id": self.mission_id,
            "feature_id": self.feature_id,
            "role": self.role,
            "role_model": self.role_model,
            "completed_work": self.completed_work,
            "remaining_work": self.remaining_work,
            "commands_run": list(self.commands_run),
            "exit_codes": list(self.exit_codes),
            "known_issues": list(self.known_issues),
            "next_recommendation": self.next_recommendation,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionHandoff:
        return cls(
            id=str(data["id"]),
            mission_id=str(data.get("mission_id") or "").strip(),
            feature_id=str(data.get("feature_id") or "").strip(),
            role=str(data.get("role") or "").strip(),
            role_model=str(data.get("role_model") or "").strip(),
            completed_work=str(data.get("completed_work") or "").strip(),
            remaining_work=str(data.get("remaining_work") or "").strip(),
            commands_run=_clean_tuple(data.get("commands_run")),
            exit_codes=_clean_tuple(data.get("exit_codes")),
            known_issues=_clean_tuple(data.get("known_issues")),
            next_recommendation=str(data.get("next_recommendation") or "").strip(),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
        )


@dataclass(frozen=True, slots=True)
class MissionFinding:
    id: str
    mission_id: str
    milestone_id: str
    source: str
    severity: FindingSeverity
    summary: str
    recommended_fix: str = ""
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mission_id": self.mission_id,
            "milestone_id": self.milestone_id,
            "source": self.source,
            "severity": self.severity,
            "summary": self.summary,
            "recommended_fix": self.recommended_fix,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionFinding:
        return cls(
            id=str(data["id"]),
            mission_id=str(data.get("mission_id") or "").strip(),
            milestone_id=str(data.get("milestone_id") or "").strip(),
            source=str(data.get("source") or "").strip(),
            severity=str(data.get("severity") or "info"),  # type: ignore[arg-type]
            summary=str(data.get("summary") or "").strip(),
            recommended_fix=str(data.get("recommended_fix") or "").strip(),
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
        )


@dataclass(frozen=True, slots=True)
class MissionRun:
    id: str
    mission_id: str
    role: str
    status: RunStatus
    role_model: str = ""
    summary: str = ""
    related_feature_id: str = ""
    related_milestone_id: str = ""
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "mission_id": self.mission_id,
            "role": self.role,
            "role_model": self.role_model,
            "status": self.status,
            "summary": self.summary,
            "related_feature_id": self.related_feature_id,
            "related_milestone_id": self.related_milestone_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionRun:
        return cls(
            id=str(data["id"]),
            mission_id=str(data.get("mission_id") or "").strip(),
            role=str(data.get("role") or "").strip(),
            role_model=str(data.get("role_model") or "").strip(),
            status=str(data.get("status") or "pending"),  # type: ignore[arg-type]
            summary=str(data.get("summary") or "").strip(),
            related_feature_id=str(data.get("related_feature_id") or "").strip(),
            related_milestone_id=str(data.get("related_milestone_id") or "").strip(),
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
        )


__all__ = [
    "AssertionKind",
    "FeatureStatus",
    "FindingSeverity",
    "Milestone",
    "MilestoneStatus",
    "Mission",
    "MissionFeature",
    "MissionFinding",
    "MissionHandoff",
    "MissionRun",
    "MissionStatus",
    "RunStatus",
    "ValidationAssertion",
    "ValidationContract",
]
