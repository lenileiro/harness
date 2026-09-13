"""Explicit, expiring transcript transfers between exact gateway destinations."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path
from uuid import uuid4


class HandoffStore:
    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        path = root / "handoffs.sqlite3"
        if path.is_symlink():
            raise ValueError("Handoff storage must not be a symlink")
        fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        os.close(fd)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS handoffs (key TEXT PRIMARY KEY, source TEXT NOT NULL, target TEXT NOT NULL, payload TEXT NOT NULL, expires REAL NOT NULL, imported_id TEXT NOT NULL, state TEXT NOT NULL)"
        )

    @staticmethod
    def identity(owner: dict[str, str]) -> str:
        if set(owner) != {"transport", "user_id", "thread_id"} or any(
            not isinstance(value, str) or not value or len(value) > 4096 for value in owner.values()
        ):
            raise ValueError("Destination needs exact transport, user_id and thread_id strings")
        return json.dumps(owner, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def key(token: str) -> str:
        if len(token) != 43 or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in token
        ):
            raise ValueError("Invalid or expired handoff code")
        return hashlib.sha256(token.encode()).hexdigest()

    def issue(self, source: dict[str, str], target: dict[str, str], payload: dict) -> str:
        source_id, target_id = self.identity(source), self.identity(target)
        if source_id == target_id:
            raise ValueError("Choose a different destination conversation")
        content = json.dumps(payload, separators=(",", ":"))
        if len(content.encode()) > 32 * 1024 * 1024:
            raise ValueError("Conversation exceeds the 32 MiB handoff limit")
        token = secrets.token_urlsafe(32)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute(
                "DELETE FROM handoffs WHERE expires < ? AND state != ?", (time.time(), "claimed")
            )
            count = self.db.execute(
                "SELECT COUNT(*) FROM handoffs WHERE source=? AND state='pending'", (source_id,)
            ).fetchone()[0]
            if count >= 10:
                raise ValueError("Ten handoffs are pending; revoke one or wait for expiry")
            self.db.execute(
                "INSERT INTO handoffs VALUES (?,?,?,?,?,?,?)",
                (
                    self.key(token),
                    source_id,
                    target_id,
                    content,
                    time.time() + 600,
                    "sess_" + uuid4().hex,
                    "pending",
                ),
            )
        return token

    def claim(self, token: str, target: dict[str, str]) -> dict:
        key = self.key(token)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT * FROM handoffs WHERE key=? AND target=?", (key, self.identity(target))
            ).fetchone()
            if (
                row is None
                or row["state"] == "revoked"
                or (row["expires"] < time.time() and row["state"] == "pending")
            ):
                raise ValueError("Invalid, expired or differently addressed handoff code")
            if row["state"] == "pending":
                self.db.execute("UPDATE handoffs SET state='claimed' WHERE key=?", (key,))
            return {
                "payload": json.loads(row["payload"]),
                "session_id": row["imported_id"],
                "complete": row["state"] == "complete",
            }

    def complete(self, token: str, target: dict[str, str]) -> None:
        with self.db:
            self.db.execute(
                "UPDATE handoffs SET state='complete', payload='{}' WHERE key=? AND target=? AND state='claimed'",
                (self.key(token), self.identity(target)),
            )

    def revoke(self, token: str, source: dict[str, str]) -> bool:
        with self.db:
            return (
                self.db.execute(
                    "UPDATE handoffs SET state='revoked', payload='{}' WHERE key=? AND source=? AND state='pending'",
                    (self.key(token), self.identity(source)),
                ).rowcount
                == 1
            )

    def close(self):
        self.db.close()
