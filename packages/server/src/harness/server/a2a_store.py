"""Bind A2A message identities in the same transaction as their executable runs."""

from __future__ import annotations

from dataclasses import dataclass

import aiosqlite

from harness.server.models import ServiceError
from harness.server.store import now


@dataclass(frozen=True)
class A2ABinding:
    task_id: str
    message_id: str
    digest: str
    expected_run_id: str | None = None
    notification_json: str | None = None
    notification_protocol: str = "1.0"

    async def existing(self, db: aiosqlite.Connection, owner: str) -> str | None:
        async with db.execute(
            "SELECT digest,run_id FROM api_a2a_messages WHERE owner=? AND message_id=?",
            (owner, self.message_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            async with db.execute(
                "SELECT digest,ready FROM api_a2a_answer_receipts WHERE owner=? AND message_id=?",
                (owner, self.message_id),
            ) as cursor:
                receipt = await cursor.fetchone()
            if receipt is not None:
                if receipt["digest"] != self.digest:
                    raise ServiceError(409, "A2A message ID was already used for different content")
                if not receipt["ready"]:
                    raise ServiceError(409, "This A2A message already supplied a partial answer")
            return None
        if row["digest"] != self.digest:
            raise ServiceError(409, "A2A message ID was already used for different content")
        return row["run_id"]

    async def save(
        self, db: aiosqlite.Connection, owner: str, session_id: str, run_id: str
    ) -> None:
        if self.expected_run_id:
            changed = await db.execute(
                "UPDATE api_a2a_tasks SET run_id=? WHERE id=? AND owner=? AND run_id=? AND session_id=?",
                (run_id, self.task_id, owner, self.expected_run_id, session_id),
            )
            if changed.rowcount != 1:
                raise ServiceError(409, "A2A task has already continued; fetch its current state")
        else:
            await db.execute(
                "INSERT INTO api_a2a_tasks VALUES (?,?,?,?,?)",
                (self.task_id, owner, session_id, run_id, now()),
            )
        await db.execute(
            "INSERT INTO api_a2a_messages VALUES (?,?,?,?,?)",
            (owner, self.message_id, self.digest, self.task_id, run_id),
        )
        await db.execute("INSERT INTO api_a2a_task_runs VALUES (?,?)", (run_id, self.task_id))
        if self.notification_json:
            import json

            config = json.loads(self.notification_json)
            async with db.execute(
                "SELECT COUNT(*) FROM api_a2a_callbacks WHERE owner=? AND active=1", (owner,)
            ) as cursor:
                count = await cursor.fetchone()
            if count is not None and count[0] >= 100:
                raise ServiceError(409, "Caller callback limit reached")
            await db.execute(
                "INSERT INTO api_a2a_callbacks (owner,task_id,id,config,last_seq,active,protocol) VALUES (?,?,?,?,0,1,?)",
                (
                    owner,
                    self.task_id,
                    config["id"],
                    self.notification_json,
                    self.notification_protocol,
                ),
            )
