"""Owned native progress receipts. Final responses still belong to the outbox."""

from __future__ import annotations

import json
import time
from typing import Any

from harness.core.gateway_channels import ChannelMessage, ChannelStore


def initialize(store: ChannelStore) -> None:
    store.db.execute("""CREATE TABLE IF NOT EXISTS channel_progress (
        source TEXT PRIMARY KEY, owner TEXT NOT NULL, native_id TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL, updated REAL NOT NULL)""")


def get(store: ChannelStore, source: str) -> dict[str, Any] | None:
    initialize(store)
    row = store.db.execute("SELECT * FROM channel_progress WHERE source=?", (source,)).fetchone()
    return dict(row) if row else None


def reserve(store: ChannelStore, message: ChannelMessage) -> bool:
    initialize(store)
    with store.db:
        return bool(
            store.db.execute(
                "INSERT OR IGNORE INTO channel_progress(source,owner,status,updated) VALUES (?,?,'creating',?)",
                (
                    "reply:" + message.id,
                    json.dumps(
                        {
                            "user_id": message.user_id,
                            "thread_id": message.thread_id,
                            "channel_id": message.channel_id,
                            "group": message.group,
                            "mentioned": message.mentioned,
                        }
                    ),
                    time.time(),
                ),
            ).rowcount
        )


def update(store: ChannelStore, source: str, status: str, native_id: str | None = None) -> None:
    with store.db:
        store.db.execute(
            "UPDATE channel_progress SET status=?,updated=?,native_id=COALESCE(?,native_id) WHERE source=?",
            (status, time.time(), native_id, source),
        )
