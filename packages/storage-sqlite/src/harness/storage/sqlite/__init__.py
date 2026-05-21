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
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from harness.core import Message, Session, SessionStatus, Storage
from harness.core.memory import MemoryEntry, MemoryKind, MemoryStore
from harness.tasks import (
    ActivityEvent,
    ActivityStore,
    ApprovalStatus,
    ApprovalStore,
    PendingApproval,
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

CREATE TABLE IF NOT EXISTS approvals (
    id           TEXT PRIMARY KEY,
    task_id      TEXT,
    session_id   TEXT NOT NULL,
    tool_call_id TEXT NOT NULL,
    tool_name    TEXT NOT NULL,
    arguments    TEXT NOT NULL,        -- JSON
    status       TEXT NOT NULL,
    requested_at TEXT NOT NULL,
    resolved_at  TEXT,
    resolved_by  TEXT,
    replayed_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_approvals_session  ON approvals(session_id, status);
CREATE INDEX IF NOT EXISTS idx_approvals_task     ON approvals(task_id);
CREATE INDEX IF NOT EXISTS idx_approvals_status   ON approvals(status);

CREATE TABLE IF NOT EXISTS memory (
    id         TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    text       TEXT NOT NULL,
    session_id TEXT,
    task_id    TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memory_kind       ON memory(kind);
CREATE INDEX IF NOT EXISTS idx_memory_created_at ON memory(created_at DESC);
"""

# Migration: older databases may not have these columns; add them idempotently.
_MIGRATIONS = (
    "ALTER TABLE sessions ADD COLUMN task_id TEXT",
    "ALTER TABLE sessions ADD COLUMN forked_from TEXT",
)


def default_db_path() -> Path:
    """`$XDG_STATE_HOME/harness/sessions.db` or `~/.local/state/harness/sessions.db`."""
    base = os.environ.get("XDG_STATE_HOME")
    state_home = Path(base) if base else Path.home() / ".local" / "state"
    return state_home / "harness" / "sessions.db"


class SQLiteStorage(Storage, TaskStore, ActivityStore, ApprovalStore, MemoryStore):
    """SQLite backend covering sessions, tasks, activity, and approvals.

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
               messages, approval_overrides, metadata, task_id, forked_from)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                session.forked_from,
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

    # ------------------------------------------------------------------ #
    # ApprovalStore                                                       #
    # ------------------------------------------------------------------ #

    async def create_approval(self, approval: PendingApproval) -> PendingApproval:
        db = await self._ensure()
        await db.execute(
            """
            INSERT INTO approvals
              (id, task_id, session_id, tool_call_id, tool_name, arguments,
               status, requested_at, resolved_at, resolved_by, replayed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                approval.id,
                approval.task_id,
                approval.session_id,
                approval.tool_call_id,
                approval.tool_name,
                json.dumps(approval.arguments),
                approval.status,
                approval.requested_at.isoformat(),
                approval.resolved_at.isoformat() if approval.resolved_at else None,
                approval.resolved_by,
                approval.replayed_at.isoformat() if approval.replayed_at else None,
            ),
        )
        await db.commit()
        return approval

    async def get_approval(self, approval_id: str) -> PendingApproval | None:
        db = await self._ensure()
        async with db.execute("SELECT * FROM approvals WHERE id = ?", (approval_id,)) as cursor:
            row = await cursor.fetchone()
        return _row_to_approval(row) if row else None

    async def list_approvals(
        self,
        *,
        session_id: str | None = None,
        task_id: str | None = None,
        status: ApprovalStatus | None = None,
        limit: int = 100,
    ) -> list[PendingApproval]:
        db = await self._ensure()
        sql = "SELECT * FROM approvals"
        clauses: list[str] = []
        params: list[object] = []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY requested_at DESC LIMIT ?"
        params.append(limit)
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_approval(r) for r in rows]

    async def resolve_approval(
        self,
        approval_id: str,
        *,
        status: ApprovalStatus,
        resolved_by: str | None = None,
    ) -> PendingApproval | None:
        db = await self._ensure()
        resolved_at = datetime.now(UTC).isoformat()
        async with db.execute(
            """
            UPDATE approvals SET status = ?, resolved_at = ?, resolved_by = ?
            WHERE id = ?
            """,
            (status, resolved_at, resolved_by, approval_id),
        ) as cursor:
            if cursor.rowcount == 0:
                return None
        await db.commit()
        return await self.get_approval(approval_id)

    async def mark_replayed(self, approval_id: str) -> None:
        db = await self._ensure()
        replayed_at = datetime.now(UTC).isoformat()
        await db.execute(
            "UPDATE approvals SET replayed_at = ? WHERE id = ? AND replayed_at IS NULL",
            (replayed_at, approval_id),
        )
        await db.commit()

    async def list_unreplayed_granted(self, *, session_id: str) -> list[PendingApproval]:
        db = await self._ensure()
        async with db.execute(
            """
            SELECT * FROM approvals
            WHERE session_id = ? AND status = 'granted' AND replayed_at IS NULL
            ORDER BY requested_at ASC
            """,
            (session_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_approval(r) for r in rows]

    # ------------------------------------------------------------------ #
    # MemoryStore                                                         #
    # ------------------------------------------------------------------ #

    async def save_memory(self, entry: MemoryEntry) -> MemoryEntry:
        db = await self._ensure()
        await db.execute(
            """
            INSERT OR REPLACE INTO memory (id, kind, text, session_id, task_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.kind,
                entry.text,
                entry.session_id,
                entry.task_id,
                entry.created_at.isoformat(),
            ),
        )
        await db.commit()
        return entry

    async def list_memory(
        self, *, kind: MemoryKind | None = None, limit: int = 50
    ) -> list[MemoryEntry]:
        db = await self._ensure()
        sql = "SELECT * FROM memory"
        params: list[object] = []
        if kind is not None:
            sql += " WHERE kind = ?"
            params.append(kind)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        async with db.execute(sql, params) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_memory(r) for r in rows]

    async def search_memory(self, query: str, *, limit: int = 20) -> list[MemoryEntry]:
        db = await self._ensure()
        async with db.execute(
            "SELECT * FROM memory WHERE text LIKE ? ORDER BY created_at DESC LIMIT ?",
            (f"%{query}%", limit),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_memory(r) for r in rows]

    async def delete_memory(self, entry_id: str) -> None:
        db = await self._ensure()
        await db.execute("DELETE FROM memory WHERE id = ?", (entry_id,))
        await db.commit()


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
        forked_from=row["forked_from"] if "forked_from" in keys else None,
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


def _row_to_approval(row: aiosqlite.Row) -> PendingApproval:
    return PendingApproval(
        id=row["id"],
        task_id=row["task_id"],
        session_id=row["session_id"],
        tool_call_id=row["tool_call_id"],
        tool_name=row["tool_name"],
        arguments=json.loads(row["arguments"]),
        status=row["status"],
        requested_at=datetime.fromisoformat(row["requested_at"]),
        resolved_at=datetime.fromisoformat(row["resolved_at"]) if row["resolved_at"] else None,
        resolved_by=row["resolved_by"],
        replayed_at=datetime.fromisoformat(row["replayed_at"]) if row["replayed_at"] else None,
    )


def _row_to_memory(row: aiosqlite.Row) -> MemoryEntry:
    return MemoryEntry(
        id=row["id"],
        kind=row["kind"],
        text=row["text"],
        session_id=row["session_id"],
        task_id=row["task_id"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )


__all__ = ["SQLiteStorage", "__version__", "default_db_path"]
