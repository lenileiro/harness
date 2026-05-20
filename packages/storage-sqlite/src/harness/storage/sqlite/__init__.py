"""SQLite-backed Storage for Harness.

Implements three Protocols against one aiosqlite database:

- `harness.core.Storage`         — sessions
- `harness.tasks.TaskStore`      — tasks (with auto-incrementing ref)
- `harness.tasks.ActivityStore`  — append-only activity ledger

Schema is JSON-column-heavy: messages, links, labels, etc. are stored as
serialized JSON inside a TEXT column. Per-message normalization isn't
worth the extra joins for a CLI tool that always reads whole rows.

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
from harness.tasks import (
    ActivityEvent,
    ActivityStore,
    Task,
    TaskLink,
    TaskStatus,
    TaskStore,
)

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
    metadata           TEXT NOT NULL,
    task_id            TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
CREATE INDEX IF NOT EXISTS idx_sessions_task_id ON sessions(task_id);

CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    ref         TEXT NOT NULL UNIQUE,
    title       TEXT NOT NULL,
    description TEXT,
    status      TEXT NOT NULL,
    priority    TEXT,
    labels      TEXT NOT NULL,        -- JSON list
    parent_id   TEXT,
    links       TEXT NOT NULL,        -- JSON list of {target_ref, relation}
    session_ids TEXT NOT NULL,        -- JSON list
    cwd         TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    metadata    TEXT NOT NULL         -- JSON
);
CREATE INDEX IF NOT EXISTS idx_tasks_updated_at ON tasks(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_parent ON tasks(parent_id);

CREATE TABLE IF NOT EXISTS task_activity (
    id         TEXT PRIMARY KEY,
    task_id    TEXT,
    session_id TEXT,
    timestamp  TEXT NOT NULL,
    kind       TEXT NOT NULL,
    data       TEXT NOT NULL          -- JSON
);
CREATE INDEX IF NOT EXISTS idx_activity_task_ts    ON task_activity(task_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_activity_session_ts ON task_activity(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_activity_kind       ON task_activity(kind);
"""

# Migration: older databases may not have task_id on sessions; add it.
_MIGRATIONS = ("ALTER TABLE sessions ADD COLUMN task_id TEXT",)


def default_db_path() -> Path:
    """`$XDG_STATE_HOME/harness/sessions.db` or `~/.local/state/harness/sessions.db`."""
    base = os.environ.get("XDG_STATE_HOME")
    state_home = Path(base) if base else Path.home() / ".local" / "state"
    return state_home / "harness" / "sessions.db"


class SQLiteStorage(Storage, TaskStore, ActivityStore):
    """SQLite backend covering sessions, tasks, and activity.

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
                # ALTER TABLE ADD COLUMN raises if the column exists already;
                # idempotent migrations swallow that one specific error.
                for stmt in _MIGRATIONS:
                    try:
                        await self._db.execute(stmt)
                    except aiosqlite.OperationalError as exc:
                        if "duplicate column" not in str(exc).lower():
                            raise
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
               messages, approval_overrides, metadata, task_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                session.task_id,
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

    # ------------------------------------------------------------------ #
    # TaskStore                                                           #
    # ------------------------------------------------------------------ #

    async def create_task(self, task: Task) -> Task:
        db = await self._ensure()
        # Compute the next ref under a write lock so concurrent creates can't
        # collide on the same number.
        async with db.execute("BEGIN IMMEDIATE"):
            pass
        try:
            async with db.execute(
                "SELECT COALESCE(MAX(CAST(SUBSTR(ref, 3) AS INTEGER)), 0) FROM tasks"
            ) as cursor:
                row = await cursor.fetchone()
            next_num = int(row[0] if row else 0) + 1
            ref = f"T-{next_num:03d}"
            updated = task.model_copy(update={"ref": ref})
            await db.execute(
                """
                INSERT INTO tasks
                  (id, ref, title, description, status, priority, labels,
                   parent_id, links, session_ids, cwd, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    updated.id,
                    updated.ref,
                    updated.title,
                    updated.description,
                    updated.status,
                    updated.priority,
                    json.dumps(updated.labels),
                    updated.parent_id,
                    json.dumps([link.model_dump(mode="json") for link in updated.links]),
                    json.dumps(updated.session_ids),
                    str(updated.cwd),
                    updated.created_at.isoformat(),
                    updated.updated_at.isoformat(),
                    json.dumps(updated.metadata),
                ),
            )
            await db.commit()
            return updated
        except Exception:
            await db.rollback()
            raise

    async def get_task(self, task_id: str) -> Task | None:
        db = await self._ensure()
        async with db.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)) as cursor:
            row = await cursor.fetchone()
        return _row_to_task(row) if row else None

    async def get_task_by_ref(self, ref: str) -> Task | None:
        db = await self._ensure()
        async with db.execute("SELECT * FROM tasks WHERE ref = ?", (ref,)) as cursor:
            row = await cursor.fetchone()
        return _row_to_task(row) if row else None

    async def list_tasks(
        self,
        *,
        limit: int = 50,
        status: TaskStatus | None = None,
        parent_id: str | None = None,
    ) -> list[Task]:
        db = await self._ensure()
        sql = "SELECT * FROM tasks"
        clauses: list[str] = []
        params: list[object] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if parent_id is not None:
            clauses.append("parent_id = ?")
            params.append(parent_id)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_task(r) for r in rows]

    async def update_task(self, task: Task) -> Task:
        db = await self._ensure()
        async with db.execute("SELECT 1 FROM tasks WHERE id = ?", (task.id,)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise KeyError(f"task {task.id!r} not found")
        await db.execute(
            """
            UPDATE tasks SET
              ref = ?, title = ?, description = ?, status = ?, priority = ?,
              labels = ?, parent_id = ?, links = ?, session_ids = ?, cwd = ?,
              created_at = ?, updated_at = ?, metadata = ?
            WHERE id = ?
            """,
            (
                task.ref,
                task.title,
                task.description,
                task.status,
                task.priority,
                json.dumps(task.labels),
                task.parent_id,
                json.dumps([link.model_dump(mode="json") for link in task.links]),
                json.dumps(task.session_ids),
                str(task.cwd),
                task.created_at.isoformat(),
                task.updated_at.isoformat(),
                json.dumps(task.metadata),
                task.id,
            ),
        )
        await db.commit()
        return task

    async def delete_task(self, task_id: str) -> None:
        db = await self._ensure()
        await db.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        await db.commit()

    # ------------------------------------------------------------------ #
    # ActivityStore                                                       #
    # ------------------------------------------------------------------ #

    async def append_activity(self, event: ActivityEvent) -> None:
        db = await self._ensure()
        await db.execute(
            """
            INSERT OR IGNORE INTO task_activity
              (id, task_id, session_id, timestamp, kind, data)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.task_id,
                event.session_id,
                event.timestamp.isoformat(),
                event.kind,
                json.dumps(event.data),
            ),
        )
        await db.commit()

    async def list_activity(
        self,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        kinds: tuple[str, ...] | None = None,
        limit: int = 200,
    ) -> list[ActivityEvent]:
        db = await self._ensure()
        sql = "SELECT * FROM task_activity"
        clauses: list[str] = []
        params: list[object] = []
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if kinds is not None:
            placeholders = ",".join("?" * len(kinds))
            clauses.append(f"kind IN ({placeholders})")
            params.extend(kinds)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp ASC LIMIT ?"
        params.append(limit)
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_activity(r) for r in rows]


def _row_to_session(row: aiosqlite.Row) -> Session:
    keys = set(row.keys())
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
        task_id=row["task_id"] if "task_id" in keys else None,
    )


def _row_to_task(row: aiosqlite.Row) -> Task:
    return Task(
        id=row["id"],
        ref=row["ref"],
        title=row["title"],
        description=row["description"],
        status=row["status"],
        priority=row["priority"],
        labels=json.loads(row["labels"]),
        parent_id=row["parent_id"],
        links=[TaskLink.model_validate(link) for link in json.loads(row["links"])],
        session_ids=json.loads(row["session_ids"]),
        cwd=Path(row["cwd"]),
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
        metadata=json.loads(row["metadata"]),
    )


def _row_to_activity(row: aiosqlite.Row) -> ActivityEvent:
    return ActivityEvent(
        id=row["id"],
        task_id=row["task_id"],
        session_id=row["session_id"],
        timestamp=datetime.fromisoformat(row["timestamp"]),
        kind=row["kind"],
        data=json.loads(row["data"]),
    )


__all__ = ["SQLiteStorage", "__version__", "default_db_path"]
