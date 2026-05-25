from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


ExperimentStatus = Literal["passed", "failed"]
CommandKind = Literal["check", "eval"]


@dataclass(frozen=True, slots=True)
class CommandResult:
    kind: CommandKind
    command: str
    exit_code: int
    duration_seconds: float
    stdout_path: str = ""
    stderr_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "command": self.command,
            "exit_code": self.exit_code,
            "duration_seconds": self.duration_seconds,
            "stdout_path": self.stdout_path,
            "stderr_path": self.stderr_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommandResult:
        return cls(
            kind=str(data.get("kind") or "check"),  # type: ignore[arg-type]
            command=str(data.get("command") or "").strip(),
            exit_code=int(data.get("exit_code", 1)),
            duration_seconds=float(data.get("duration_seconds", 0.0) or 0.0),
            stdout_path=str(data.get("stdout_path") or "").strip(),
            stderr_path=str(data.get("stderr_path") or "").strip(),
        )


@dataclass(frozen=True, slots=True)
class Experiment:
    id: str
    plan_id: str
    branch: str = ""
    worktree: str = ""
    created_by: str = "human"
    created_at: str = field(default_factory=_utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "plan_id": self.plan_id,
            "branch": self.branch,
            "worktree": self.worktree,
            "created_by": self.created_by,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Experiment:
        return cls(
            id=str(data["id"]),
            plan_id=str(data.get("plan_id") or "").strip(),
            branch=str(data.get("branch") or "").strip(),
            worktree=str(data.get("worktree") or "").strip(),
            created_by=str(data.get("created_by") or "human").strip() or "human",
            created_at=str(data.get("created_at") or _utcnow()),
        )


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    experiment_id: str
    status: ExperimentStatus
    command_results: tuple[CommandResult, ...]
    started_at: str
    finished_at: str
    duration_seconds: float
    artifact_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "status": self.status,
            "command_results": [result.to_dict() for result in self.command_results],
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "artifact_dir": self.artifact_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentResult:
        return cls(
            experiment_id=str(data.get("experiment_id") or "").strip(),
            status=str(data.get("status") or "failed"),  # type: ignore[arg-type]
            command_results=tuple(
                CommandResult.from_dict(item) for item in data.get("command_results") or []
            ),
            started_at=str(data.get("started_at") or _utcnow()),
            finished_at=str(data.get("finished_at") or _utcnow()),
            duration_seconds=float(data.get("duration_seconds", 0.0) or 0.0),
            artifact_dir=str(data.get("artifact_dir") or "").strip(),
        )


__all__ = ["CommandResult", "Experiment", "ExperimentResult"]
