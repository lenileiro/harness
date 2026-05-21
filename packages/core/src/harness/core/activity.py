"""Activity event ledger primitives.

`ActivityEvent` is an immutable record of something that happened during an
agent run. `ActivityStore` is the append-only storage Protocol. Both live
in `core` so the runtime (`harness.core.runtime.Agent`) can emit events
without taking a dependency on the `tasks` package.

`ActivityEvent.kind` is an open string. Built-in kinds emitted by the
runtime are listed below; ecosystem packages (e.g. `harness.tasks`) define
their own under their own namespace.

Naming convention: `<domain>.<verb>`. Past tense for outcomes, present
tense for intent.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return f"act_{uuid.uuid4().hex[:12]}"


class ActivityEvent(BaseModel):
    """A single immutable entry in the activity ledger.

    `kind` is an open string (e.g. `"agent_run.started"`,
    `"tool_call.dispatched"`). Consumers can extend the vocabulary by
    coining new strings under their own namespace.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=_new_id)
    task_id: str | None = None
    session_id: str | None = None
    timestamp: datetime = Field(default_factory=_utcnow)
    kind: str
    data: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class ActivityStore(Protocol):
    """Append-only activity event ledger."""

    async def append_activity(self, event: ActivityEvent) -> None:
        """Persist a single event. Implementations are idempotent on `event.id`."""
        ...

    async def list_activity(
        self,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        kinds: tuple[str, ...] | None = None,
        limit: int = 200,
    ) -> list[ActivityEvent]:
        """Load events ordered by timestamp ascending; AND filters."""
        ...


# ---------------------------------------------------------------------------
# Built-in kind vocabulary emitted by the runtime
# ---------------------------------------------------------------------------

# Agent run lifecycle
AGENT_RUN_STARTED = "agent_run.started"
AGENT_RUN_COMPLETED = "agent_run.completed"
AGENT_RUN_FAILED = "agent_run.failed"
AGENT_RUN_CANCELLED = "agent_run.cancelled"
STEP_STARTED = "step.started"
STEP_COMPLETED = "step.completed"

# Tool execution
TOOL_CALL_DISPATCHED = "tool_call.dispatched"
TOOL_CALL_COMPLETED = "tool_call.completed"

# Approval flow
APPROVAL_REQUESTED = "approval.requested"
APPROVAL_GRANTED = "approval.granted"
APPROVAL_DENIED = "approval.denied"
APPROVAL_QUEUED = "approval.queued"
"""Emitted when an inbox handler defers a tool call for later review."""
APPROVAL_REPLAYED = "approval.replayed"
"""Emitted when the runtime re-dispatches a previously-granted, queued call."""


__all__ = [
    "AGENT_RUN_CANCELLED",
    "AGENT_RUN_COMPLETED",
    "AGENT_RUN_FAILED",
    "AGENT_RUN_STARTED",
    "APPROVAL_DENIED",
    "APPROVAL_GRANTED",
    "APPROVAL_QUEUED",
    "APPROVAL_REPLAYED",
    "APPROVAL_REQUESTED",
    "STEP_COMPLETED",
    "STEP_STARTED",
    "TOOL_CALL_COMPLETED",
    "TOOL_CALL_DISPATCHED",
    "ActivityEvent",
    "ActivityStore",
]
