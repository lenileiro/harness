"""SQLite-backed Storage for Harness.

v1 schema: a single `sessions` table where the conversation history,
approval overrides, and metadata are stored as JSON columns. Tool calls
and individual messages aren't normalized into separate tables yet —
that's deliberate for a CLI tool where every read fetches a whole session.

Connection model: one long-lived aiosqlite connection per Storage
instance, lazily initialized on first use. Use `await storage.close()`
to release it explicitly. The default location follows XDG:
`$XDG_STATE_HOME/harness/sessions.db` (or `~/.local/state/harness/`).
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import aiosqlite

from harness.core import Message, Session, SessionStatus, Storage

__version__ = "0.0.0"


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id                 TEXT PRIMARY KEY,
    provider           TEXT NOT NULL,
    model              TEXT NOT NULL,
    cwd                TEXT NOT NULL,
    status             TEXT NOT NULL,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    messages           TEXT NOT NULL,
    approval_overrides TEXT NOT NULL,
    metadata           TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
"""


def default_db_path() -> Path:
    """`$XDG_STATE_HOME/harness/sessions.db` or `~/.local/state/harness/sessions.db`."""
    base = os.environ.get("XDG_STATE_HOME")
    state_home = Path(base) if base else Path.home() / ".local" / "state"
    return state_home / "harness" / "sessions.db"


class SQLiteStorage(Storage):
    """Storage protocol implementation backed by aiosqlite.

    Args:
        path: SQLite database file path. Defaults to `default_db_path()`.
              Use `:memory:` for an ephemeral in-memory database (handy in tests).
    """

    def __init__(self, *, path: Path | str | None = None) -> None:
        if path == ":memory:":
            self.path: Path | str = ":memory:"
        else:
            self.path = Path(path) if path else default_db_path()
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def _ensure(self) -> aiosqlite.Connection:
        async with self._lock:
            if self._db is None:
                if isinstance(self.path, Path):
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                self._db = await aiosqlite.connect(str(self.path))
                self._db.row_factory = aiosqlite.Row
                await self._db.executescript(_SCHEMA)
                await self._db.commit()
        return self._db

    async def close(self) -> None:
        """Close the underlying connection. Idempotent."""
        async with self._lock:
            if self._db is not None:
                await self._db.close()
                self._db = None

    # ------------------------------------------------------------------ #
    # Storage Protocol                                                    #
    # ------------------------------------------------------------------ #

    async def get(self, session_id: str) -> Session | None:
        db = await self._ensure()
        async with db.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)) as cursor:
            row = await cursor.fetchone()
        return _row_to_session(row) if row else None

    async def save(self, session: Session) -> None:
        db = await self._ensure()
        await db.execute(
            """
            INSERT OR REPLACE INTO sessions
              (id, provider, model, cwd, status, created_at, updated_at,
               messages, approval_overrides, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session.id,
                session.provider,
                session.model,
                str(session.cwd),
                session.status,
                session.created_at.isoformat(),
                session.updated_at.isoformat(),
                json.dumps([m.model_dump(mode="json") for m in session.messages]),
                json.dumps(session.approval_overrides),
                json.dumps(session.metadata),
            ),
        )
        await db.commit()

    async def list(
        self,
        *,
        limit: int = 50,
        before: datetime | None = None,
        status: SessionStatus | None = None,
    ) -> list[Session]:
        db = await self._ensure()
        sql = "SELECT * FROM sessions"
        clauses: list[str] = []
        params: list[object] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if before is not None:
            clauses.append("updated_at < ?")
            params.append(before.isoformat())
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_session(r) for r in rows]

    async def delete(self, session_id: str) -> None:
        db = await self._ensure()
        await db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await db.commit()


def _row_to_session(row: aiosqlite.Row) -> Session:
    return Session(
        id=row["id"],
        provider=row["provider"],
        model=row["model"],
        cwd=Path(row["cwd"]),
        status=row["status"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        messages=[Message.model_validate(m) for m in json.loads(row["messages"])],
        approval_overrides=json.loads(row["approval_overrides"]),
        metadata=json.loads(row["metadata"]),
    )


__all__ = ["SQLiteStorage", "__version__", "default_db_path"]
