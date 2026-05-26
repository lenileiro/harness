from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def default_scheduler_root(cwd: Path | None = None) -> Path:
    return (cwd or Path.cwd()).resolve() / ".harness" / "scheduler"


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class ScheduleSpec:
    kind: str
    value: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind, "value": self.value}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ScheduleSpec:
        return cls(
            kind=str(payload.get("kind", "")).strip(),
            value=str(payload.get("value", "")).strip(),
        )


@dataclass(frozen=True, slots=True)
class SchedulerJob:
    id: str
    kind: str
    cwd: str
    status: str
    schedule: ScheduleSpec
    next_run_at: str
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    last_run_at: str = ""
    last_status: str = ""
    last_error: str = ""
    last_record_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "cwd": self.cwd,
            "status": self.status,
            "schedule": self.schedule.to_dict(),
            "next_run_at": self.next_run_at,
            "payload": self.payload,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_run_at": self.last_run_at,
            "last_status": self.last_status,
            "last_error": self.last_error,
            "last_record_dir": self.last_record_dir,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SchedulerJob:
        raw_payload = payload.get("payload", {})
        if not isinstance(raw_payload, dict):
            raise ValueError("scheduler job payload must be a JSON object")
        return cls(
            id=str(payload.get("id", "")).strip(),
            kind=str(payload.get("kind", "")).strip(),
            cwd=str(payload.get("cwd", "")).strip(),
            status=str(payload.get("status", "")).strip(),
            schedule=ScheduleSpec.from_dict(payload.get("schedule", {})),
            next_run_at=str(payload.get("next_run_at", "")).strip(),
            payload=dict(raw_payload),
            created_at=str(payload.get("created_at", "")).strip() or _utcnow(),
            updated_at=str(payload.get("updated_at", "")).strip() or _utcnow(),
            last_run_at=str(payload.get("last_run_at", "")).strip(),
            last_status=str(payload.get("last_status", "")).strip(),
            last_error=str(payload.get("last_error", "")).strip(),
            last_record_dir=str(payload.get("last_record_dir", "")).strip(),
        )


@dataclass(frozen=True, slots=True)
class SchedulerRunRecord:
    id: str
    job_id: str
    kind: str
    cwd: str
    trigger: str
    status: str
    result_status: str
    result_stop_reason: str
    started_at: str
    finished_at: str
    record_dir: str
    summary: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "kind": self.kind,
            "cwd": self.cwd,
            "trigger": self.trigger,
            "status": self.status,
            "result_status": self.result_status,
            "result_stop_reason": self.result_stop_reason,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "record_dir": self.record_dir,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SchedulerRunRecord:
        return cls(
            id=str(payload.get("id", "")).strip(),
            job_id=str(payload.get("job_id", "")).strip(),
            kind=str(payload.get("kind", "")).strip(),
            cwd=str(payload.get("cwd", "")).strip(),
            trigger=str(payload.get("trigger", "")).strip(),
            status=str(payload.get("status", "")).strip(),
            result_status=str(payload.get("result_status", "")).strip(),
            result_stop_reason=str(payload.get("result_stop_reason", "")).strip(),
            started_at=str(payload.get("started_at", "")).strip(),
            finished_at=str(payload.get("finished_at", "")).strip(),
            record_dir=str(payload.get("record_dir", "")).strip(),
            summary=str(payload.get("summary", "")).strip(),
        )


__all__ = [
    "ScheduleSpec",
    "SchedulerJob",
    "SchedulerRunRecord",
    "default_scheduler_root",
]
