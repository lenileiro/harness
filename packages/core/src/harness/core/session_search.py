"""Optional scoped conversation retrieval; plain terms rather than query-language input."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from harness.core.memory import MemoryScope
from harness.core.schemas import Message, Session


class ConversationMatch(BaseModel):
    reference: str
    index: int
    role: str
    excerpt: str


def message_references(session_id: str, messages: list[Message]) -> list[str]:
    """Content references stay stable across appends and unrelated insertions."""
    occurrences: dict[str, int] = {}
    result = []
    for message in messages:
        payload = message.model_dump(mode="json", exclude={"cache_breakpoint"})
        digest = hashlib.sha256(
            json.dumps([session_id, payload], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
        occurrences[digest] = occurrences.get(digest, 0) + 1
        result.append(f"msg_{digest}_{occurrences[digest]}")
    return result


def matching_messages(
    session_id: str, messages: list[Message], terms: list[str]
) -> list[ConversationMatch]:
    refs = message_references(session_id, messages)
    matches = []
    for index, message in enumerate(messages):
        text = message.content or ""
        if message.role == "system" or not any(term in text.casefold() for term in terms):
            continue
        offset = max(
            0, min(text.casefold().index(term) for term in terms if term in text.casefold()) - 100
        )
        matches.append(
            ConversationMatch(
                reference=refs[index],
                index=index,
                role=message.role,
                excerpt=text[offset : offset + 1200],
            )
        )
        if len(matches) >= 10:
            break
    return matches


class SessionSearchResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    session_id: str
    excerpt: str
    updated_at: datetime
    matches: list[ConversationMatch] = Field(default_factory=list)


@runtime_checkable
class ConversationSearchStore(Protocol):
    async def search_sessions(
        self, query: str, *, scope: MemoryScope, limit: int = 10
    ) -> list[SessionSearchResult]: ...


def session_scope(session: Session) -> MemoryScope | None:
    """Missing identity is local-only; malformed explicit identity fails closed."""
    if "memory_scope" not in session.metadata:
        return MemoryScope(workspace=str(session.cwd))
    try:
        return MemoryScope.model_validate(session.metadata["memory_scope"])
    except (ValidationError, TypeError):
        return None


def search_terms(query: str) -> list[str]:
    if len(query) > 500:
        raise ValueError("search query must contain at most 500 characters")
    return list(dict.fromkeys(re.findall(r"\w+", query.casefold())))[:32]


def search_limit(limit: int) -> int:
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    return limit


def session_text(session: Session) -> str:
    return "\n".join(
        f"[{message.role}] {message.content}"
        for message in session.messages
        if message.role != "system" and message.content
    )


__all__ = ["ConversationSearchStore", "SessionSearchResult"]
