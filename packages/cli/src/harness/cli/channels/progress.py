"""Best-effort native run status, with durable final-message receipt ownership.

Only fixed status strings leave the event observer. Raw token deltas, thinking,
tool arguments and outputs never become a preview, including split secrets.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from contextlib import suppress
from contextvars import ContextVar, Token
from typing import Any

from harness.cli.channels.progress_transport import (
    SUPPORTED,
    edit_preview,
    send_preview,
    send_typing,
)
from harness.cli.channels.transports import (
    DeliveryDeferred,
    DeliveryRejected,
    RateLimited,
    Transport,
)
from harness.core import (
    Done,
    ErrorEvent,
    Event,
    StepStarted,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
    Verification,
)
from harness.core import channel_progress as ledger
from harness.core.gateway_channels import ChannelMessage, ChannelStore

_CURRENT: ContextVar[NativeProgress | None] = ContextVar("native_channel_progress", default=None)


def observe_gateway_event(event: Event) -> None:
    current = _CURRENT.get()
    if current is not None:
        current.observe(event)


def permitted(transport: Transport, owner: dict[str, Any]) -> bool:
    return transport.config.permits(
        user_id=owner["user_id"],
        channel_id=owner["channel_id"],
        group=owner["group"],
        mentioned=owner["mentioned"],
    )


class NativeProgress:
    """One dispatch owns one fixed-status preview, completed by the outbox."""

    interval = 3.0
    initial_delay = 1.0
    operation_timeout = 5.0

    def __init__(self, transport: Transport, store: ChannelStore, message: ChannelMessage):
        self.transport, self.store, self.message = transport, store, message
        self.source = "reply:" + message.id
        self.state = "Working…"
        self.active = False
        self.paused = False
        self.stopped = asyncio.Event()
        self.task: asyncio.Task[None] | None = None
        self.token: Token[NativeProgress | None] | None = None
        self.owner = {
            "user_id": message.user_id,
            "thread_id": message.thread_id,
            "channel_id": message.channel_id,
            "group": message.group,
            "mentioned": message.mentioned,
        }
        self.last_text = ""

    async def __aenter__(self):
        if self.transport.name in SUPPORTED:
            self.token = _CURRENT.set(self)
            self.task = asyncio.create_task(self._run())
        return self

    def observe(self, event: Event) -> None:
        if isinstance(event, Done):
            self.paused = bool(
                (event.structured_result or {}).get("status")
                in {"waiting_for_input", "waiting_for_approval"}
            )
            self.stopped.set()
        elif isinstance(event, ErrorEvent):
            self.stopped.set()
        elif isinstance(
            event, (StepStarted, TextDelta, ToolCallEvent, ToolResultEvent, Verification)
        ):
            self.active = True
            self.state = (
                "Using a tool…"
                if isinstance(event, ToolCallEvent)
                else "Checking the result…"
                if isinstance(event, (ToolResultEvent, Verification))
                else "Writing the response…"
                if isinstance(event, TextDelta)
                else "Working…"
            )

    async def _tick(self) -> None:
        if not self.active or self.stopped.is_set() or not permitted(self.transport, self.owner):
            return
        row = ledger.get(self.store, self.source)
        if row is None:
            if not ledger.reserve(self.store, self.message):
                return
            try:
                native = await send_preview(
                    self.transport,
                    self.message.thread_id,
                    self.state,
                    hashlib.sha256(self.source.encode()).hexdigest(),
                )
                ledger.update(self.store, self.source, "active", native)
                self.last_text = self.state
            except BaseException:
                ledger.update(self.store, self.source, "uncertain")
                raise
            row = ledger.get(self.store, self.source)
        if not row or row["status"] != "active" or not row["native_id"]:
            return
        if self.stopped.is_set() or not permitted(self.transport, self.owner):
            return
        if self.state != self.last_text:
            await edit_preview(self.transport, self.message.thread_id, row["native_id"], self.state)
            self.last_text = self.state
        if not self.stopped.is_set() and permitted(self.transport, self.owner):
            await send_typing(self.transport, self.message.thread_id)

    async def _run(self) -> None:
        delay = self.initial_delay
        while not self.stopped.is_set():
            try:
                await asyncio.wait_for(self.stopped.wait(), timeout=delay)
                break
            except TimeoutError:
                pass
            try:
                await asyncio.wait_for(self._tick(), timeout=self.operation_timeout)
                delay = self.interval
            except RateLimited as exc:
                delay = max(self.interval, exc.delay)
            except Exception:
                # Progress cannot turn a useful model run into a failed run.
                delay = max(self.interval, 10.0)

    async def __aexit__(self, exc_type, exc, tb):
        if self.token is not None:
            _CURRENT.reset(self.token)
        self.stopped.set()
        if self.task is None:
            return
        cancelled_during_exit = False
        # Do not cancel an in-flight create before it can return its receipt.
        try:
            await asyncio.wait_for(asyncio.shield(self.task), timeout=self.operation_timeout + 1)
        except asyncio.CancelledError:
            cancelled_during_exit = True
            exc_type = asyncio.CancelledError
            with suppress(Exception):
                await asyncio.wait_for(
                    asyncio.shield(self.task), timeout=self.operation_timeout + 1
                )
            if not self.task.done():
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
        except Exception:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        row = ledger.get(self.store, self.source)
        if not row or row["status"] != "active":
            if cancelled_during_exit:
                raise asyncio.CancelledError
            return
        if exc_type is None:
            ledger.update(self.store, self.source, "ready")
            return
        if permitted(self.transport, self.owner):
            with suppress(Exception):
                await asyncio.wait_for(
                    edit_preview(
                        self.transport,
                        self.message.thread_id,
                        row["native_id"],
                        "Run interrupted. Inspect the session before retrying.",
                    ),
                    timeout=self.operation_timeout,
                )
        ledger.update(self.store, self.source, "interrupted")
        if cancelled_during_exit:
            raise asyncio.CancelledError


async def deliver_final_preview(
    transport: Transport, store: ChannelStore, delivery: dict[str, Any]
) -> bool:
    """Edit only the owned first text chunk; callers mark outbox sent afterward."""
    if delivery["part"] != 0 or transport.name not in SUPPORTED:
        return False
    row = ledger.get(store, delivery["source"])
    if not row or not row["native_id"]:
        return False
    owner = json.loads(row["owner"])
    if (
        owner["user_id"] != delivery["user_id"]
        or owner["thread_id"] != delivery["thread_id"]
        or not permitted(transport, owner)
    ):
        raise DeliveryRejected("Progress receipt no longer belongs to an allowed destination")
    if row["status"] == "finalized":
        return True
    try:
        await asyncio.wait_for(
            edit_preview(transport, delivery["thread_id"], row["native_id"], delivery["text"]),
            timeout=10,
        )
    except RateLimited:
        raise
    except Exception:
        # Repeating an edit is safe; creating a second final message is not.
        raise DeliveryDeferred(
            "Final progress edit will retry the same owned message", delay=10
        ) from None
    ledger.update(store, delivery["source"], "finalized")
    return True


async def recover_progress(transport: Transport, store: ChannelStore) -> None:
    ledger.initialize(store)
    rows = store.db.execute(
        "SELECT * FROM channel_progress WHERE status IN ('active','creating','ready')"
    ).fetchall()
    for raw in rows:
        row = dict(raw)
        # A queued final response will replace this receipt through the outbox.
        if store.db.execute("SELECT 1 FROM outbox WHERE source=?", (row["source"],)).fetchone():
            ledger.update(store, row["source"], "ready")
            continue
        if row["native_id"] and permitted(transport, json.loads(row["owner"])):
            with suppress(Exception):
                await asyncio.wait_for(
                    edit_preview(
                        transport,
                        json.loads(row["owner"])["thread_id"],
                        row["native_id"],
                        "Run interrupted. Inspect the session before retrying.",
                    ),
                    timeout=5,
                )
        ledger.update(store, row["source"], "interrupted" if row["native_id"] else "uncertain")
