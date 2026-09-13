"""Transactional FTS5 maintenance and conservative legacy-memory scoping."""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite
from pydantic import ValidationError

from harness.core.memory import MemoryScope
from harness.core.schemas import Message, Session
from harness.core.session_search import session_scope, session_text


async def index_session(db: aiosqlite.Connection, session: Session) -> None:
    await db.execute("DELETE FROM session_search WHERE session_id = ?", (session.id,))
    scope = session_scope(session)
    if scope is None:
        return
    await db.execute(
        "INSERT INTO session_search (session_id, scope, content, updated_at) VALUES (?, ?, ?, ?)",
        (
            session.id,
            scope.model_dump_json(),
            session_text(session),
            session.updated_at.isoformat(),
        ),
    )


async def infer_memory_scope(
    db: aiosqlite.Connection, *, session_id: str | None, path: Path | str
) -> MemoryScope | None:
    if session_id:
        async with db.execute(
            "SELECT cwd, metadata FROM sessions WHERE id = ?", (session_id,)
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        try:
            session = Session(
                provider="legacy",
                model="legacy",
                cwd=row["cwd"],
                metadata=json.loads(row["metadata"]),
            )
            return session_scope(session)
        except (ValueError, TypeError):
            return None
    if isinstance(path, Path) and path.parent.name == ".harness" and path.name == "harness.db":
        return MemoryScope(workspace=str(path.parent.parent.resolve()))
    return None


async def initialize_search(db: aiosqlite.Connection, *, path: Path | str) -> None:
    """Called inside the schema transaction; no implicit commit or second connection."""
    await db.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS session_search USING fts5("
        "session_id UNINDEXED, scope UNINDEXED, content, updated_at UNINDEXED)"
    )
    await db.execute(
        "CREATE TABLE IF NOT EXISTS harness_search_migrations (version INTEGER PRIMARY KEY)"
    )
    async with db.execute("SELECT 1 FROM harness_search_migrations WHERE version = 1") as cursor:
        migrated = await cursor.fetchone()
    if migrated:
        return
    async with db.execute("SELECT * FROM sessions") as cursor:
        rows = await cursor.fetchall()
    for row in rows:
        try:
            session = Session(
                id=row["id"],
                provider=row["provider"],
                model=row["model"],
                cwd=row["cwd"],
                updated_at=row["updated_at"],
                metadata=json.loads(row["metadata"]),
                messages=[
                    Message.model_validate(message) for message in json.loads(row["messages"])
                ],
            )
        except (ValidationError, ValueError, TypeError):
            # Corrupt historical rows remain accessible to repair tooling, not search.
            continue
        await index_session(db, session)
    async with db.execute("SELECT id, session_id FROM memory WHERE scope IS NULL") as cursor:
        memories = await cursor.fetchall()
    for memory in memories:
        scope = await infer_memory_scope(db, session_id=memory["session_id"], path=path)
        if scope is not None:
            await db.execute(
                "UPDATE memory SET scope = ? WHERE id = ?", (scope.model_dump_json(), memory["id"])
            )
    await db.execute("INSERT INTO harness_search_migrations (version) VALUES (1)")
