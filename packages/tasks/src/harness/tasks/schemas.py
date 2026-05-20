"""Pydantic schemas for tasks.

A `Task` is the durable container for agent work — one task lasts across
many sessions / agent runs. The activity ledger primitives
(`ActivityEvent`, `ActivityStore`) live in `harness.core.activity` so the
runtime can emit events without depending on this package.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

TaskStatus = Literal["backlog", "todo", "in_progress", "waiting", "done", "cancelled"]
Priority = Literal["low", "medium", "high"]
Relation = Literal["blocks", "depends_on", "duplicates", "fixes", "tests", "relates_to"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class TaskLink(BaseModel):
    """A typed edge from this task to another task (referenced by ref)."""

    model_config = ConfigDict(extra="forbid")

    target_ref: str
    """Ref of the other task, e.g. `T-002`."""
    relation: Relation


class Task(BaseModel):
    """A durable unit of agent work.

    `ref` is the human-facing short id (`T-001`, `T-002`, ...). It is
    assigned by the store on creation and is immutable thereafter. `id` is
    the internal uuid-based identifier.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: _new_id("task"))
    ref: str
    title: str
    description: str | None = None
    status: TaskStatus = "backlog"
    priority: Priority | None = None
    labels: list[str] = Field(default_factory=list)
    parent_id: str | None = None
    links: list[TaskLink] = Field(default_factory=list)
    session_ids: list[str] = Field(default_factory=list)
    cwd: Path
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def touch(self) -> None:
        """Bump `updated_at` to now."""
        self.updated_at = _utcnow()


__all__ = [
    "Priority",
    "Relation",
    "Task",
    "TaskLink",
    "TaskStatus",
]
