"""Durable queue and replayable events, separate from Agent session ownership."""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from harness.server.models import ServiceError


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


class ServiceStore:
    def __init__(self, path: Path):
        self.path = path
        self.db: aiosqlite.Connection | None = None
        self.lock = asyncio.Lock()
        self._lease: int | None = None

    async def start(self, *, acquire_lease: bool = True) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if acquire_lease:
            descriptor = os.open(
                self.path.with_suffix(self.path.suffix + ".server.lock"),
                os.O_CREAT | os.O_RDWR,
                0o600,
            )
            try:
                if os.name == "nt":
                    import msvcrt

                    os.write(descriptor, b"0")
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                os.close(descriptor)
                raise RuntimeError("Another Harness server owns this database") from None
            self._lease = descriptor
        try:
            self.db = await aiosqlite.connect(self.path)
            self.db.row_factory = aiosqlite.Row
            await self.db.executescript("""
                CREATE TABLE IF NOT EXISTS api_sessions (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS api_sessions_owner ON api_sessions(owner, created_at);
                CREATE TABLE IF NOT EXISTS api_session_workspaces (
                    session_id TEXT PRIMARY KEY, workspace TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS api_delegation_parents (
                    session_id TEXT PRIMARY KEY, owner TEXT NOT NULL, workspace TEXT NOT NULL,
                    root_id TEXT NOT NULL, depth INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS api_delegation_budgets (
                    root_id TEXT PRIMARY KEY, limits TEXT NOT NULL, children INTEGER NOT NULL DEFAULT 0,
                    steps INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
                    seconds REAL NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS api_delegations (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, parent_session_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, root_id TEXT NOT NULL, workspace TEXT NOT NULL,
                    request TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS api_delegation_attempts (
                    run_id TEXT PRIMARY KEY, delegation_id TEXT NOT NULL,
                    max_steps INTEGER NOT NULL, max_tokens INTEGER NOT NULL, timeout_seconds REAL NOT NULL);
                CREATE INDEX IF NOT EXISTS api_delegation_parent ON api_delegations(parent_session_id,owner);
                CREATE TABLE IF NOT EXISTS api_preferences (
                    owner TEXT PRIMARY KEY, preferences TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS api_a2a_tasks (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, session_id TEXT NOT NULL,
                    run_id TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS api_a2a_owner ON api_a2a_tasks(owner,created_at);
                CREATE TABLE IF NOT EXISTS api_a2a_messages (
                    owner TEXT NOT NULL, message_id TEXT NOT NULL, digest TEXT NOT NULL,
                    task_id TEXT NOT NULL, run_id TEXT NOT NULL,
                    PRIMARY KEY(owner,message_id));
                CREATE TABLE IF NOT EXISTS api_a2a_answer_receipts (
                    owner TEXT NOT NULL, message_id TEXT NOT NULL, digest TEXT NOT NULL,
                    task_id TEXT NOT NULL, run_id TEXT NOT NULL, question_id TEXT NOT NULL,
                    ready INTEGER NOT NULL, PRIMARY KEY(owner,message_id));
                CREATE TABLE IF NOT EXISTS api_a2a_task_runs (
                    run_id TEXT PRIMARY KEY, task_id TEXT NOT NULL);
                INSERT OR IGNORE INTO api_a2a_task_runs SELECT run_id,task_id FROM api_a2a_messages;
                INSERT OR IGNORE INTO api_a2a_task_runs SELECT run_id,id FROM api_a2a_tasks;
                CREATE TABLE IF NOT EXISTS api_a2a_callbacks (
                    owner TEXT NOT NULL, task_id TEXT NOT NULL, id TEXT NOT NULL,
                    config TEXT NOT NULL, last_seq INTEGER NOT NULL DEFAULT 0,
                    active INTEGER NOT NULL DEFAULT 1, protocol TEXT NOT NULL DEFAULT '1.0',
                    PRIMARY KEY(owner,task_id,id));
                CREATE TABLE IF NOT EXISTS api_a2a_callback_deliveries (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, task_id TEXT NOT NULL,
                    config_id TEXT NOT NULL, event_seq INTEGER NOT NULL,
                    payload TEXT NOT NULL, state TEXT NOT NULL, attempts INTEGER NOT NULL,
                    next_attempt REAL NOT NULL, error TEXT,
                    UNIQUE(owner,task_id,config_id,event_seq));
                CREATE TABLE IF NOT EXISTS api_schedules (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, title TEXT NOT NULL,
                    request TEXT NOT NULL, schedule TEXT NOT NULL, next_run_at TEXT,
                    state TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    error TEXT);
                CREATE INDEX IF NOT EXISTS api_schedules_due ON api_schedules(state,next_run_at);
                CREATE TABLE IF NOT EXISTS api_schedule_runs (
                    schedule_id TEXT NOT NULL, due_at TEXT NOT NULL, run_id TEXT NOT NULL,
                    PRIMARY KEY(schedule_id,due_at), UNIQUE(run_id));
                CREATE TABLE IF NOT EXISTS api_batches (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS api_runs (
                    id TEXT PRIMARY KEY, owner TEXT NOT NULL, session_id TEXT NOT NULL,
                    state TEXT NOT NULL, kind TEXT NOT NULL, request TEXT NOT NULL,
                    batch_id TEXT, resumed_from TEXT, created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL, error TEXT);
                CREATE UNIQUE INDEX IF NOT EXISTS api_session_active_run
                    ON api_runs(session_id) WHERE state IN ('queued','running');
                CREATE INDEX IF NOT EXISTS api_runs_queue ON api_runs(state, created_at);
                CREATE TABLE IF NOT EXISTS api_run_controls (run_id TEXT PRIMARY KEY, cancel_requested INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS api_events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL,
                    payload TEXT NOT NULL, created_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS api_events_run ON api_events(run_id, seq);
            """)
            async with self.db.execute("PRAGMA table_info(api_a2a_callbacks)") as cursor:
                callback_columns = {row["name"] for row in await cursor.fetchall()}
            if "protocol" not in callback_columns:
                await self.db.execute(
                    "ALTER TABLE api_a2a_callbacks ADD COLUMN protocol TEXT NOT NULL DEFAULT '1.0'"
                )
            await self.db.commit()
        except BaseException:
            await self.close()
            raise

    @asynccontextmanager
    async def connection(self):
        async with self.lock:
            if self.db is None:
                raise RuntimeError("Harness service is not running")
            try:
                yield self.db
            finally:
                await self.db.rollback()

    async def rows(self, sql: str, values: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        async with self.connection() as db, db.execute(sql, values) as cursor:
            return [dict(row) for row in await cursor.fetchall()]

    async def run(self, owner: str, run_id: str) -> dict[str, Any]:
        rows = await self.rows("SELECT * FROM api_runs WHERE id=? AND owner=?", (run_id, owner))
        if not rows:
            raise ServiceError(404, "Run not found")
        result = rows[0]
        result.pop("request")
        controls = await self.rows(
            "SELECT cancel_requested FROM api_run_controls WHERE run_id=?", (run_id,)
        )
        result["cancellation_requested"] = bool(controls and controls[0]["cancel_requested"])
        child = await self.rows(
            "SELECT d.id,d.parent_session_id FROM api_delegations d JOIN api_delegation_attempts a ON a.delegation_id=d.id WHERE a.run_id=?",
            (run_id,),
        )
        if child:
            result["delegation_handle"] = child[0]["id"]
            result["parent_session_id"] = child[0]["parent_session_id"]
        return result

    async def event(self, run_id: str, payload: dict[str, Any]) -> None:
        async with self.connection() as db:
            await db.execute(
                "INSERT INTO api_events(run_id,payload,created_at) VALUES (?,?,?)",
                (run_id, json.dumps(payload), now()),
            )
            await db.commit()

    async def finish(
        self, run_id: str, state: str, error: str | None = None, *, queued_only: bool = False
    ) -> None:
        async with self.connection() as db:
            stamp = now()
            cursor = await db.execute(
                "UPDATE api_runs SET state=?,error=?,updated_at=? WHERE id=? AND (state='queued' OR (state='running' AND ?=0))",
                (state, error, stamp, run_id, int(queued_only)),
            )
            if cursor.rowcount:
                await db.execute(
                    "INSERT INTO api_events(run_id,payload,created_at) VALUES (?,?,?)",
                    (
                        run_id,
                        json.dumps({"type": "run_status", "state": state, "error": error}),
                        stamp,
                    ),
                )
            await db.commit()

    async def close(self) -> None:
        async with self.lock:
            if self.db is not None:
                await self.db.close()
                self.db = None
            if self._lease is not None:
                os.close(self._lease)
                self._lease = None
