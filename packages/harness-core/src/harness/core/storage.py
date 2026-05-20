"""Session storage Protocol.

Implementations: harness-storage-memory, harness-storage-sqlite.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from harness.core.schemas import Session, SessionStatus


@runtime_checkable
class Storage(Protocol):
    """Durable session storage. All methods are async to support I/O backends."""

    async def get(self, session_id: str) -> Session | None:
        """Load a session by id, or None if missing."""
        ...

    async def save(self, session: Session) -> None:
        """Insert or update. Implementations bump `updated_at` if not already current."""
        ...

    async def list(
        self,
        *,
        limit: int = 50,
        before: datetime | None = None,
        status: SessionStatus | None = None,
    ) -> list[Session]:
        """List sessions, newest first.

        `before` filters to sessions whose `updated_at` < before — used for
        pagination. `status` filters by lifecycle state.
        """
        ...

    async def delete(self, session_id: str) -> None:
        """Delete a session. No error if it doesn't exist."""
        ...


__all__ = ["Storage"]
