"""Translate authenticated native controls into existing scoped approval commands."""

from __future__ import annotations

import time
from inspect import isawaitable
from typing import TYPE_CHECKING, Any, cast

from harness.cli.runtime_helpers import build_storage
from harness.core.approval import ApprovalStore
from harness.core.channel_interactions import consume, lookup
from harness.core.gateway_channels import ChannelMessage, ChannelStore
from harness.core.gateway_evidence import approval_expires_at
from harness.core.gateway_models import default_gateway_root
from harness.core.gateway_router import gateway_pending_approvals
from harness.core.gateway_sessions import GatewaySessionStore

if TYPE_CHECKING:
    from harness.cli.channels.transports import Transport

SUPPORTED = {"telegram", "discord", "slack", "feishu", "lark", "teams"}


async def pending_cards(
    store: ChannelStore, transport: Transport, message: ChannelMessage, ids: list[str]
) -> list[dict[str, Any]]:
    if transport.name not in SUPPORTED or not ids:
        return []
    cwd = store.path.parents[2]
    storage = build_storage(db=cwd / ".harness" / "harness.db", in_memory=False, cwd=cwd)
    try:
        records = await gateway_pending_approvals(
            approval_store=cast(ApprovalStore, storage),
            session_store=GatewaySessionStore(root=default_gateway_root(cwd)),
            transport=transport.name,
            user_id=message.user_id,
            thread_id=message.thread_id,
        )
        return [
            {
                "approval_id": item.id,
                "session_id": item.session_id,
                "tool_name": item.tool_name,
                "expires": approval_expires_at(item).timestamp(),
            }
            for item in records
            if item.id in ids
        ][:10]
    finally:
        close = getattr(storage, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                await result


async def accept_choice(
    transport: Transport,
    store: ChannelStore,
    *,
    token: str,
    user_id: str,
    channel_id: str,
    native_message_id: str,
    event_id: str,
    thread_id: str | None = None,
) -> bool:
    row = lookup(store, token)
    if (
        not row
        or row["consumed"]
        or row["expires"] <= time.time()
        or row["user_id"] != user_id
        or row["channel_id"] != channel_id
        or not native_message_id
        or row["native_message_id"] != native_message_id
        or (thread_id is not None and row["thread_id"] != thread_id)
        or not transport.config.permits(
            user_id=user_id, channel_id=channel_id, group=bool(row["group_chat"]), mentioned=True
        )
    ):
        return False
    # Re-read the authoritative record and binding after native authentication.
    # Native payloads carry no session, tool arguments, scope, or arbitrary command.
    command = "approve" if token == row["approve_token"] else "deny"
    message = ChannelMessage(
        id="approval-choice:" + row["id"],
        user_id=user_id,
        channel_id=channel_id,
        thread_id=row["thread_id"],
        text=command + " " + row["approval_id"],
        group=bool(row["group_chat"]),
        mentioned=True,
    )
    cards = await pending_cards(store, transport, message, [row["approval_id"]])
    if not cards or cards[0]["session_id"] != row["session_id"] or not event_id:
        return False
    return consume(store, row, token, message)
