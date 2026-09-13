"""Native question choices and text-capture state, separate from tool approvals."""

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
        CREATE TABLE IF NOT EXISTS question_interactions (
            id TEXT PRIMARY KEY, question_id TEXT NOT NULL, session_id TEXT NOT NULL,
            question_key TEXT NOT NULL, user_id TEXT NOT NULL, thread_id TEXT NOT NULL,
            channel_id TEXT NOT NULL, group_chat INTEGER NOT NULL, expires REAL NOT NULL,
            spec TEXT NOT NULL, buttons TEXT NOT NULL, native_message_id TEXT NOT NULL DEFAULT '',
            mode TEXT NOT NULL DEFAULT 'choices', selected TEXT NOT NULL DEFAULT '[]',
            capture_message TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS question_interaction_tokens (
            token TEXT PRIMARY KEY, interaction_id TEXT NOT NULL, action TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS question_interaction_events (
            interaction_id TEXT NOT NULL, event_id TEXT NOT NULL,
            PRIMARY KEY(interaction_id,event_id)
        );
    """)


def queue_card(
    store: ChannelStore, message: ChannelMessage, card: dict[str, Any], *, start: int
) -> None:
    if store.db.execute(
        """SELECT 1 FROM question_interactions WHERE question_id=? AND question_key=?
        AND session_id=? AND mode IN ('choices','text','capturing') AND expires>?""",
        (card["question_id"], card["question_key"], card["session_id"], time.time()),
    ).fetchone():
        return
    source = "reply:" + message.id
    key = hashlib.sha256(json.dumps([source, start]).encode()).hexdigest()
    spec = card["spec"]
    buttons, tokens = [], []
    for index, option in enumerate(spec.get("choices") or []):
        token = "hq_" + secrets.token_urlsafe(24)
        tokens.append((token, key, str(index)))
        buttons.append({"label": f"{index + 1}. {option}", "token": token})
    for action, label in [
        ("other", "Other (type answer)"),
        *([("submit", "Submit selections")] if spec.get("multi_select") else []),
    ]:
        token = "hq_" + secrets.token_urlsafe(24)
        tokens.append((token, key, action))
        buttons.append({"label": label, "token": token})
    store.db.execute(
        """INSERT INTO question_interactions
        (id,question_id,session_id,question_key,user_id,thread_id,channel_id,group_chat,expires,spec,buttons)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            key,
            card["question_id"],
            card["session_id"],
            card["question_key"],
            message.user_id,
            message.thread_id,
            message.channel_id,
            int(message.group),
            card["expires"],
            json.dumps(spec),
            json.dumps(buttons),
        ),
    )
    store.db.executemany("INSERT INTO question_interaction_tokens VALUES (?,?,?)", tokens)
    text = spec["question"]
    if spec.get("multi_select"):
        text += "\nSelect any choices, then Submit selections; use Other to type an answer."
    else:
        text += "\nChoose one option or Other to type an answer."
    store.db.execute(
        """INSERT INTO outbox (id,source,user_id,thread_id,text,part,interaction)
        VALUES (?,?,?,?,?,?,?)""",
        (
            key,
            source,
            message.user_id,
            message.thread_id,
            text,
            start,
            json.dumps({"kind": "question", "buttons": buttons}),
        ),
    )


def lookup(store: ChannelStore, token: str) -> dict[str, Any] | None:
    if not isinstance(token, str) or len(token) != 35 or not token.startswith("hq_"):
        return None
    row = store.db.execute(
        """SELECT q.*,t.action FROM question_interactions AS q
        JOIN question_interaction_tokens AS t ON q.id=t.interaction_id WHERE t.token=?""",
        (token,),
    ).fetchone()
    return dict(row) if row else None


def bind_message(store: ChannelStore, delivery_id: str, native_message_id: str) -> None:
    if not native_message_id or len(native_message_id) > 1024:
        raise ValueError("Native question response is missing a bounded message ID")
    with store.db:
        store.db.execute(
            "UPDATE question_interactions SET native_message_id=? WHERE id=? AND mode='choices'",
            (native_message_id, delivery_id),
        )


def decide(store: ChannelStore, row: dict[str, Any], event_id: str) -> str:
    from harness.core.gateway_channels import ChannelMessage

    with store.db:
        store.db.execute("BEGIN IMMEDIATE")
        current = store.db.execute(
            "SELECT * FROM question_interactions WHERE id=?", (row["id"],)
        ).fetchone()
        if not current or current["mode"] != "choices" or current["expires"] <= time.time():
            return "This question control is no longer available."
        if not store.db.execute(
            "INSERT OR IGNORE INTO question_interaction_events VALUES (?,?)", (row["id"], event_id)
        ).rowcount:
            return "This interaction was already received."
        spec, selected = json.loads(current["spec"]), json.loads(current["selected"])
        action = row["action"]
        if action == "other":
            store.db.execute(
                "UPDATE question_interactions SET mode='used' WHERE user_id=? AND thread_id=? AND mode='text'",
                (row["user_id"], row["thread_id"]),
            )
            store.db.execute(
                "UPDATE question_interactions SET mode='text' WHERE id=?", (row["id"],)
            )
            store._queue(
                source="question-text:" + row["id"],
                user_id=row["user_id"],
                thread_id=row["thread_id"],
                text="Type your answer to: "
                + spec["question"]
                + "\nYour next ordinary text message in this conversation answers this question. Commands remain available.",
                limit=2000,
            )
            return "Type your answer in the same conversation."
        if action != "submit":
            option = (spec.get("choices") or [])[int(action)]
            if spec.get("multi_select"):
                selected = (
                    [x for x in selected if x != option]
                    if option in selected
                    else [*selected, option]
                )
                store.db.execute(
                    "UPDATE question_interactions SET selected=? WHERE id=?",
                    (json.dumps(selected), row["id"]),
                )
                return (
                    "Selected: "
                    + (", ".join(selected) or "(none)")
                    + ". Use Submit selections when ready."
                )
            answer: str | list[str] = option
        else:
            answer = selected
        message = ChannelMessage(
            id="question-choice:" + row["id"],
            user_id=row["user_id"],
            thread_id=row["thread_id"],
            channel_id=row["channel_id"],
            group=bool(row["group_chat"]),
            mentioned=True,
            text="/answer "
            + row["question_id"]
            + " "
            + json.dumps({row["question_key"]: answer}, ensure_ascii=False),
        )
        store.db.execute("UPDATE question_interactions SET mode='used' WHERE id=?", (row["id"],))
        store.db.execute(
            "INSERT INTO inbox (id,message,created) VALUES (?,?,?)",
            (message.id, json.dumps(asdict(message)), time.time()),
        )
        return "Answer queued."
