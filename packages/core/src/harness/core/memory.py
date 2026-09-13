"""Memory primitives for persistent, cross-session workspace facts.

`MemoryEntry` is an immutable record (user preference, project fact, etc.).
`MemoryStore` is the Protocol that storage backends implement.

Memories are injected as a synthetic system message at the start of each
agent run so the LLM has them in context without explicit retrieval.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

MemoryKind = Literal["user_preference", "user_fact", "project_fact", "project_context"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return f"mem_{uuid.uuid4().hex[:12]}"


class MemoryScope(BaseModel):
    """Trusted workspace/user identity; never accepted from model tool arguments."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workspace: str = Field(min_length=1)
    user_id: str | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("workspace")
    @classmethod
    def canonical_workspace(cls, value: str) -> str:
        return str(Path(value).expanduser().resolve())


class MemoryEntry(BaseModel):
    """A single persistent workspace memory."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=_new_id)
    kind: MemoryKind
    text: str
    session_id: str | None = None
    task_id: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    scope: MemoryScope | None = None

    def model_dump_row(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "text": self.text,
            "session_id": self.session_id,
            "task_id": self.task_id,
            "created_at": self.created_at.isoformat(),
            "scope": self.scope.model_dump(mode="json") if self.scope else None,
        }


@runtime_checkable
class MemoryStore(Protocol):
    """Persistent memory store for cross-session workspace facts."""

    async def save_memory(self, entry: MemoryEntry) -> MemoryEntry: ...

    async def list_memory(
        self, *, kind: MemoryKind | None = None, limit: int = 50
    ) -> list[MemoryEntry]: ...

    async def search_memory(self, query: str, *, limit: int = 20) -> list[MemoryEntry]: ...

    async def delete_memory(self, entry_id: str) -> None: ...


@runtime_checkable
class ScopedMemoryStore(Protocol):
    """Optional scope-enforcing CRUD interface, separate from legacy MemoryStore."""

    async def save_scoped_memory(
        self, entry: MemoryEntry, *, scope: MemoryScope
    ) -> MemoryEntry: ...

    async def get_scoped_memory(
        self, entry_id: str, *, scope: MemoryScope
    ) -> MemoryEntry | None: ...

    async def list_scoped_memory(
        self, *, scope: MemoryScope, kind: MemoryKind | None = None, limit: int = 50
    ) -> list[MemoryEntry]: ...

    async def search_scoped_memory(
        self, query: str, *, scope: MemoryScope, limit: int = 20
    ) -> list[MemoryEntry]: ...

    async def delete_scoped_memory(self, entry_id: str, *, scope: MemoryScope) -> bool: ...


__all__ = ["MemoryEntry", "MemoryKind", "MemoryScope", "MemoryStore", "ScopedMemoryStore"]
