"""In-memory Storage implementation for Harness.

Sessions live only for the lifetime of the process. All mutations defensively
deep-copy sessions so callers cannot mutate the store by retaining references.
"""

from __future__ import annotations

from datetime import datetime

from harness.core import Session, SessionStatus, Storage

__version__ = "0.0.0"


class InMemoryStorage(Storage):
    """Storage that keeps sessions in a process-local dict.

    Implements the harness.core.Storage protocol. Reads and writes are O(1) by
    id; `list` is O(N log N) on session count (small N expected).
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    async def get(self, session_id: str) -> Session | None:
        stored = self._sessions.get(session_id)
        return stored.model_copy(deep=True) if stored else None

    async def save(self, session: Session) -> None:
        self._sessions[session.id] = session.model_copy(deep=True)

    async def list(
        self,
        *,
        limit: int = 50,
        before: datetime | None = None,
        status: SessionStatus | None = None,
    ) -> list[Session]:
        items = sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)
        if before is not None:
            items = [s for s in items if s.updated_at < before]
        if status is not None:
            items = [s for s in items if s.status == status]
        return [s.model_copy(deep=True) for s in items[:limit]]

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


__all__ = ["InMemoryStorage", "__version__"]
