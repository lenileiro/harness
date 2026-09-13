"""Durable owner-bound native approval controls; never grants tool permission."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness.core.gateway_channels import ChannelMessage, ChannelStore


def initialize(store: ChannelStore) -> None:
    store.db.executescript("""
        CREATE TABLE IF NOT EXISTS approval_interactions (
            id TEXT PRIMARY KEY, approval_id TEXT NOT NULL, session_id TEXT NOT NULL,
            user_id TEXT NOT NULL, thread_id TEXT NOT NULL, channel_id TEXT NOT NULL,
            group_chat INTEGER NOT NULL, expires REAL NOT NULL,
            approve_token TEXT UNIQUE NOT NULL, deny_token TEXT UNIQUE NOT NULL,
            native_message_id TEXT NOT NULL DEFAULT '', consumed INTEGER NOT NULL DEFAULT 0
        );
    """)


def queue_cards(
    store: ChannelStore,
    message: ChannelMessage,
    cards: list[dict[str, Any]],
    *,
    start: int,
    source: str | None = None,
) -> None:
    """Caller owns the transaction that completes the triggering inbox message."""
    for offset, card in enumerate(cards[:10], start=start):
        source = source or "reply:" + message.id
        key = hashlib.sha256(json.dumps([source, offset]).encode()).hexdigest()
        existing = store.db.execute(
            "SELECT * FROM approval_interactions WHERE id=?", (key,)
        ).fetchone()
        if existing:
            continue
        if store.db.execute(
            """SELECT 1 FROM approval_interactions WHERE approval_id=? AND session_id=?
            AND user_id=? AND thread_id=? AND consumed=0 AND expires>?""",
            (
                card["approval_id"],
                card["session_id"],
                message.user_id,
                message.thread_id,
                time.time(),
            ),
        ).fetchone():
            continue
        approve, deny = "ha_" + secrets.token_urlsafe(24), "ha_" + secrets.token_urlsafe(24)
        store.db.execute(
            """INSERT INTO approval_interactions
            (id,approval_id,session_id,user_id,thread_id,channel_id,group_chat,expires,approve_token,deny_token)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (
                key,
                card["approval_id"],
                card["session_id"],
                message.user_id,
                message.thread_id,
                message.channel_id,
                int(message.group),
                card["expires"],
                approve,
                deny,
            ),
        )
        descriptor = {"approve": approve, "deny": deny, "approval_id": card["approval_id"]}
        store.db.execute(
            """INSERT INTO outbox (id,source,user_id,thread_id,text,part,interaction)
            VALUES (?,?,?,?,?,?,?)""",
            (
                key,
                source,
                message.user_id,
                message.thread_id,
                f"Decision for {card['approval_id']} ({card['tool_name']}). Review the action above.",
                offset,
                json.dumps(descriptor),
            ),
        )


def bind_message(store: ChannelStore, delivery: str, message_id: str) -> None:
    if not message_id or len(message_id) > 512:
        raise ValueError("Native approval response is missing a bounded message ID")
    with store.db:
        store.db.execute(
            "UPDATE approval_interactions SET native_message_id=? WHERE id=? AND consumed=0",
            (message_id, delivery),
        )


def lookup(store: ChannelStore, token: str) -> dict[str, Any] | None:
    if not isinstance(token, str) or not token.startswith("ha_") or len(token) != 35:
        return None
    row = store.db.execute(
        "SELECT * FROM approval_interactions WHERE approve_token=? OR deny_token=?", (token, token)
    ).fetchone()
    return dict(row) if row else None


def consume(store: ChannelStore, row: dict[str, Any], token: str, message: ChannelMessage) -> bool:
    """Consume the paired choice and persist its command in a single transaction."""
    with store.db:
        store.db.execute("BEGIN IMMEDIATE")
        updated = store.db.execute(
            """UPDATE approval_interactions SET consumed=1 WHERE id=? AND consumed=0
            AND expires>? AND (approve_token=? OR deny_token=?)""",
            (row["id"], time.time(), token, token),
        )
        if not updated.rowcount:
            return False
        # Another notification may have rendered this same pending action.
        # Invalidate every copy while atomically queueing the one decision.
        store.db.execute(
            "UPDATE approval_interactions SET consumed=1 WHERE approval_id=? AND session_id=?",
            (row["approval_id"], row["session_id"]),
        )
        store.db.execute(
            "INSERT INTO inbox (id,message,created) VALUES (?,?,?)",
            (message.id, json.dumps(asdict(message)), time.time()),
        )
    return True
