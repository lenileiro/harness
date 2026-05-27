from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def default_gateway_root(cwd: Path | None = None) -> Path:
    return (cwd or Path.cwd()).resolve() / ".harness" / "gateway"


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class GatewayMessage:
    id: str
    transport: str
    user_id: str
    thread_id: str
    text: str
    received_at: str = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "transport": self.transport,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "text": self.text,
            "received_at": self.received_at,
            "metadata": self.metadata,
        }


@dataclass(frozen=True, slots=True)
class GatewayReply:
    session_id: str
    command: str
    status: str
    text: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "command": self.command,
            "status": self.status,
            "text": self.text,
            "data": self.data,
        }


@dataclass(frozen=True, slots=True)
class GatewaySessionBinding:
    id: str
    transport: str
    user_id: str
    thread_id: str
    current_mission_id: str = ""
    last_job_id: str = ""
    last_run_id: str = ""
    last_command: str = ""
    updated_at: str = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "transport": self.transport,
            "user_id": self.user_id,
            "thread_id": self.thread_id,
            "current_mission_id": self.current_mission_id,
            "last_job_id": self.last_job_id,
            "last_run_id": self.last_run_id,
            "last_command": self.last_command,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GatewaySessionBinding:
        raw_metadata = payload.get("metadata", {})
        if not isinstance(raw_metadata, dict):
            raise ValueError("gateway session metadata must be a JSON object")
        return cls(
            id=str(payload.get("id", "")).strip(),
            transport=str(payload.get("transport", "")).strip(),
            user_id=str(payload.get("user_id", "")).strip(),
            thread_id=str(payload.get("thread_id", "")).strip(),
            current_mission_id=str(payload.get("current_mission_id", "")).strip(),
            last_job_id=str(payload.get("last_job_id", "")).strip(),
            last_run_id=str(payload.get("last_run_id", "")).strip(),
            last_command=str(payload.get("last_command", "")).strip(),
            updated_at=str(payload.get("updated_at", "")).strip() or _utcnow(),
            metadata=dict(raw_metadata),
        )


@dataclass(frozen=True, slots=True)
class GatewayWorkRef:
    ref: str
    kind: str
    title: str
    summary: str
    status: str = "active"
    source_thread_id: str = ""
    updated_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "kind": self.kind,
            "title": self.title,
            "summary": self.summary,
            "status": self.status,
            "source_thread_id": self.source_thread_id,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GatewayWorkRef:
        return cls(
            ref=str(payload.get("ref", "")).strip(),
            kind=str(payload.get("kind", "")).strip(),
            title=str(payload.get("title", "")).strip(),
            summary=str(payload.get("summary", "")).strip(),
            status=str(payload.get("status", "")).strip() or "active",
            source_thread_id=str(payload.get("source_thread_id", "")).strip(),
            updated_at=str(payload.get("updated_at", "")).strip() or _utcnow(),
        )


@dataclass(frozen=True, slots=True)
class GatewayUserProfile:
    id: str
    transport: str
    user_id: str
    active_work: list[GatewayWorkRef] = field(default_factory=list)
    recent_threads: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=_utcnow)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "transport": self.transport,
            "user_id": self.user_id,
            "active_work": [item.to_dict() for item in self.active_work],
            "recent_threads": list(self.recent_threads),
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> GatewayUserProfile:
        raw_active_work = payload.get("active_work", [])
        raw_recent_threads = payload.get("recent_threads", [])
        raw_metadata = payload.get("metadata", {})
        return cls(
            id=str(payload.get("id", "")).strip(),
            transport=str(payload.get("transport", "")).strip(),
            user_id=str(payload.get("user_id", "")).strip(),
            active_work=[
                GatewayWorkRef.from_dict(item) for item in raw_active_work if isinstance(item, dict)
            ],
            recent_threads=[str(item).strip() for item in raw_recent_threads if str(item).strip()]
            if isinstance(raw_recent_threads, list)
            else [],
            updated_at=str(payload.get("updated_at", "")).strip() or _utcnow(),
            metadata=dict(raw_metadata) if isinstance(raw_metadata, dict) else {},
        )


__all__ = [
    "GatewayMessage",
    "GatewayReply",
    "GatewaySessionBinding",
    "GatewayUserProfile",
    "GatewayWorkRef",
    "default_gateway_root",
]
