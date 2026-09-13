"""Native question UI reuses the persisted question/gateway answer path."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from harness.core.clarification import PendingQuestion, QuestionStore
from harness.core.gateway_channels import ChannelMessage, ChannelStore, split_channel_text
from harness.core.gateway_models import default_gateway_root
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.question_interactions import decide, lookup

if TYPE_CHECKING:
    from harness.cli.channels.transports import Transport

SUPPORTED = {"telegram", "discord", "slack", "google_chat"}


async def _record(
    store: ChannelStore, transport: Transport, row: dict[str, Any]
) -> PendingQuestion | None:
    from harness.cli.gateway_clarification import gateway_question_scope

    cwd = store.path.parents[2]
    binding = GatewaySessionStore(root=default_gateway_root(cwd)).load_runtime_binding(
        row["session_id"]
    )
    if (
        not binding
        or binding.local_only
        or not binding.belongs_to(
            transport=transport.name, user_id=row["user_id"], thread_id=row["thread_id"]
        )
    ):
        return None
    ledger = await asyncio.to_thread(QuestionStore, cwd / ".harness" / "harness.db")
    try:
        record = await asyncio.to_thread(
            ledger.get,
            row["question_id"],
            session_id=binding.session_id,
            scope=gateway_question_scope(cwd, transport.name, row["user_id"]),
        )
        if not record or record.status != "pending" or row["question_key"] in record.answers:
            return None
        index = int(row["question_key"][1:])
        if index < 0 or index >= len(record.questions):
            return None
        if row.get("spec") and record.questions[index].model_dump(mode="json") != json.loads(
            row["spec"]
        ):
            return None
        return record
    finally:
        await asyncio.to_thread(ledger.close)


async def pending_card(
    store: ChannelStore, transport: Transport, message: ChannelMessage, question_id: str
) -> dict[str, Any] | None:
    if transport.name not in SUPPORTED or not question_id:
        return None
    from harness.cli.gateway_clarification import gateway_questions

    cwd = store.path.parents[2]
    records = await gateway_questions(
        cwd=cwd,
        session_store=GatewaySessionStore(root=default_gateway_root(cwd)),
        transport=transport.name,
        user_id=message.user_id,
        thread_id=message.thread_id,
    )
    for record, binding in records:
        if record.id != question_id or record.status != "pending" or binding.local_only:
            continue
        for index, spec in enumerate(record.questions):
            if f"q{index}" not in record.answers:
                return {
                    "question_id": record.id,
                    "session_id": record.session_id,
                    "question_key": f"q{index}",
                    "spec": spec.model_dump(mode="json"),
                    "expires": record.expires_at.timestamp(),
                }
    return None


async def accept_question_choice(
    transport: Transport,
    store: ChannelStore,
    *,
    token: str,
    user_id: str,
    channel_id: str,
    native_message_id: str,
    event_id: str,
    thread_id: str | None = None,
) -> str:
    row = lookup(store, token)
    if (
        not row
        or row["mode"] != "choices"
        or row["expires"] <= time.time()
        or row["user_id"] != user_id
        or row["channel_id"] != channel_id
        or not native_message_id
        or row["native_message_id"] != native_message_id
        or (thread_id is not None and row["thread_id"] != thread_id)
        or not event_id
        or len(event_id) > 1024
        or not transport.config.permits(
            user_id=user_id, channel_id=channel_id, group=bool(row["group_chat"]), mentioned=True
        )
        or await _record(store, transport, row) is None
    ):
        return "Unavailable, expired, or already answered."
    return decide(store, row, event_id)


async def interaction_result(transport: Transport, store: ChannelStore, **kwargs: Any) -> str:
    token = kwargs.get("token", "")
    if isinstance(token, str) and token.startswith("hq_"):
        return await accept_question_choice(transport, store, **kwargs)
    from harness.cli.channels.approval_interactions import accept_choice

    return (
        "Decision queued."
        if await accept_choice(transport, store, **kwargs)
        else "Unavailable, expired, or already used."
    )


def display_text(text: str, limit: int) -> str:
    parts = split_channel_text(text, limit)
    return parts[0] if len(parts) == 1 else split_channel_text(text, limit - 1)[0] + "…"


def _ordinary_text(message: ChannelMessage, text: str) -> bool:
    from harness.cli.gateway_conversation_commands import is_conversation_command
    from harness.core.gateway_router import is_gateway_control_message

    return bool(
        text
        and not text.startswith("/")
        and not message.attachments
        and not is_conversation_command(text)
        and not is_gateway_control_message(text)
    )


def admit_text_answer(store: ChannelStore, message: ChannelMessage) -> ChannelMessage:
    if not _ordinary_text(message, message.text.strip()):
        return message
    row = store.db.execute(
        """SELECT id FROM question_interactions WHERE user_id=? AND thread_id=?
        AND channel_id=? AND mode='text' AND expires>? ORDER BY rowid DESC LIMIT 1""",
        (message.user_id, message.thread_id, message.channel_id, time.time()),
    ).fetchone()
    # The owner's explicit Other click authorizes one ordinary answer in this
    # conversation. Preserve that origin so a stale answer cannot become a new run.
    return replace(message, mentioned=True, question_capture_id=row["id"]) if row else message


async def capture_answer(
    transport: Transport, store: ChannelStore, message: ChannelMessage, text: str
) -> str:
    if not _ordinary_text(message, text):
        return text
    raw = store.db.execute(
        """SELECT * FROM question_interactions WHERE user_id=? AND thread_id=?
        AND (mode='text' OR (mode='capturing' AND capture_message=?)) ORDER BY rowid DESC LIMIT 1""",
        (message.user_id, message.thread_id, message.id),
    ).fetchone()
    if not raw:
        return "/questions" if message.question_capture_id else text
    row = dict(raw)
    if (message.question_capture_id and row["id"] != message.question_capture_id) or await _record(
        store, transport, row
    ) is None:
        return "/questions" if message.question_capture_id else text
    with store.db:
        store.db.execute(
            "UPDATE question_interactions SET mode='capturing',capture_message=? WHERE id=?",
            (message.id, row["id"]),
        )
    return (
        "/answer "
        + row["question_id"]
        + " "
        + json.dumps({row["question_key"]: text}, ensure_ascii=False)
    )
