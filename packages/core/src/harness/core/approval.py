"""Durable approval inbox primitives.

`PendingApproval` is the wire record for a tool call that's awaiting human
review. `ApprovalStore` is the storage Protocol. Both live in `core` so the
runtime can read/write them without depending on the `tasks` package.

Lifecycle:

    pending → granted (by user, via CLI or programmatic API)
                  ↓
            replayed (after Agent re-dispatches the original call)

    pending → denied (terminal — no replay)

`replayed_at` tracks the replay step separately from `status` so a granted
approval that hasn't been replayed yet is distinguishable from one that has.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

ApprovalStatus = Literal["pending", "granted", "denied"]

ApprovalOutcome = Literal["approved", "denied", "queued"]
"""Return type for `ApprovalHandler.__call__`. Older handlers may return
`bool` (True ↔ "approved", False ↔ "denied"); the runtime normalizes."""


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return f"appr_{uuid.uuid4().hex[:12]}"


class PendingApproval(BaseModel):
    """A queued tool call awaiting human review.

    `tool_call_id` is the id of the original ToolCall in the session
    transcript. When the user grants the approval, the runtime uses this to
    locate the corresponding `role=tool` message in `session.messages` and
    overwrite its content with the real tool result.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=_new_id)
    task_id: str | None = None
    session_id: str
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: ApprovalStatus = "pending"
    requested_at: datetime = Field(default_factory=_utcnow)
    resolved_at: datetime | None = None
    resolved_by: str | None = None
    replayed_at: datetime | None = None


@runtime_checkable
class ApprovalStore(Protocol):
    """Durable storage for `PendingApproval` records."""

    async def create_approval(self, approval: PendingApproval) -> PendingApproval:
        """Insert and return the persisted copy."""
        ...

    async def get_approval(self, approval_id: str) -> PendingApproval | None:
        """Load by id, or None if missing."""
        ...

    async def list_approvals(
        self,
        *,
        session_id: str | None = None,
        task_id: str | None = None,
        status: ApprovalStatus | None = None,
        limit: int = 100,
    ) -> list[PendingApproval]:
        """List approvals, newest-requested first; filters are AND'd."""
        ...

    async def resolve_approval(
        self,
        approval_id: str,
        *,
        status: ApprovalStatus,
        resolved_by: str | None = None,
    ) -> PendingApproval | None:
        """Mark `pending → granted | denied`. Returns the updated row, or None
        if the id doesn't exist."""
        ...

    async def mark_replayed(self, approval_id: str) -> None:
        """Set `replayed_at = now`. Idempotent."""
        ...

    async def list_unreplayed_granted(self, *, session_id: str) -> list[PendingApproval]:
        """Convenience: every granted-but-not-yet-replayed approval for a
        session, in `requested_at` order. Used by the runtime's resume path."""
        ...


__all__ = [
    "ApprovalOutcome",
    "ApprovalStatus",
    "ApprovalStore",
    "PendingApproval",
]
