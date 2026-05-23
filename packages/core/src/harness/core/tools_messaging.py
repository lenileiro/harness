"""Inter-agent messaging tools.

Within a ``MultiAgentOrchestrator`` job, sub-agents (planner / workers /
reporter) used to be fire-and-forget: each one ran to completion and the
parent only saw the final result. That's "fire and forget" coordination,
not real coordination — peers can't share progress, the parent can't see
that a worker has flagged a problem until everything's done.

These tools add a broadcast message channel scoped to a job (via task_id
on the activity events). Any agent in the job can:

- ``notify(text)``: append an INTER_AGENT_MESSAGE event with this agent's
  role identifier so other agents see who said what.
- ``check_messages(since=...)``: read pending messages from peers, filtered
  by sender if requested.

This is deliberately a coarse broadcast — point-to-point routing, message
acks, and durable subscriptions are future work. For the patterns
``MultiAgentOrchestrator`` supports today (planner emits items, workers
process them, reporter synthesizes), a shared channel is enough.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from harness.core import activity as activity_kinds
from harness.core.activity import ActivityEvent, ActivityStore
from harness.core.schemas import ApprovalDecision, ToolCall, ToolResult

_NOTIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "Message body to broadcast to other agents in this job.",
        },
    },
    "required": ["text"],
}


_CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "from_role": {
            "type": "string",
            "description": (
                "Optional sender filter. When set, only messages from agents "
                "with this role (e.g. 'planner', 'worker-1') are returned. "
                "Omit to see all messages."
            ),
        },
        "limit": {
            "type": "integer",
            "description": "Max messages to return (default 20, max 100).",
        },
    },
    "required": [],
}


class NotifyTool:
    """Broadcast a message to other agents in the same job.

    Args:
        role: This agent's identifier (e.g. 'planner', 'worker-1'). Sent
              alongside the message so receivers can tell who said what.
        task_id: The job-level task ID. Scopes the message to a single
                 job — other jobs won't see it.
        activity_store: Where to append the INTER_AGENT_MESSAGE event.
    """

    name = "notify"
    description = (
        "Broadcast a status message to other agents in this multi-agent job. "
        "Use this when you've made progress that other agents need to know "
        "about (e.g. 'finished cataloging the src/ tree', 'found a blocker "
        "in module X'). Other agents call `check_messages` to read these."
    )
    approval: ApprovalDecision = "auto"
    effect_scope = "session_ephemeral"
    phases: tuple[str, ...] = ("*",)

    def __init__(
        self,
        *,
        role: str,
        task_id: str | None = None,
        activity_store: ActivityStore | None = None,
    ) -> None:
        self._role = role
        self._task_id = task_id
        self._activity_store = activity_store
        self.parameters_schema = _NOTIFY_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        args = call.arguments if isinstance(call.arguments, dict) else {}
        text = (args.get("text") or "").strip()
        if not text:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="text is required",
                is_error=True,
            )
        if self._activity_store is None:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="(no activity store wired; message dropped)",
            )
        event = ActivityEvent(
            task_id=self._task_id,
            kind=activity_kinds.INTER_AGENT_MESSAGE,
            data={"from_role": self._role, "text": text},
        )
        await self._activity_store.append_activity(event)
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=f"notified peers as {self._role!r}: {text[:80]!r}",
            metadata={"from_role": self._role, "text": text},
        )


class CheckMessagesTool:
    """Read inter-agent messages broadcast within this job.

    Args:
        role: This agent's identifier. By default the agent does NOT see
              its own messages back — that's noise. Pass ``include_own=True``
              when constructing to override.
        task_id: Job-level task ID to scope reads.
        activity_store: Source of INTER_AGENT_MESSAGE events.
        include_own: Whether to return messages this agent sent. Default
                     False — the agent doesn't need to read its own
                     broadcasts as 'news.'
    """

    name = "check_messages"
    description = (
        "Read messages broadcast by other agents in this multi-agent job. "
        "Returns recent messages with the sender's role label. Use this "
        "between work units to pick up progress signals or blockers from "
        "peers. Filter by sender via the from_role argument."
    )
    approval: ApprovalDecision = "auto"
    effect_scope = "read_only"
    phases: tuple[str, ...] = ("*",)

    def __init__(
        self,
        *,
        role: str,
        task_id: str | None = None,
        activity_store: ActivityStore | None = None,
        include_own: bool = False,
    ) -> None:
        self._role = role
        self._task_id = task_id
        self._activity_store = activity_store
        self._include_own = include_own
        self._last_seen: datetime | None = None
        self.parameters_schema = _CHECK_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        args = call.arguments if isinstance(call.arguments, dict) else {}
        from_filter = (args.get("from_role") or "").strip()
        try:
            limit = max(1, min(int(args.get("limit", 20)), 100))
        except (TypeError, ValueError):
            limit = 20

        if self._activity_store is None or self._task_id is None:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="(no message channel configured)",
            )

        events = await self._activity_store.list_activity(
            task_id=self._task_id,
            kinds=(activity_kinds.INTER_AGENT_MESSAGE,),
            limit=limit,
        )
        msgs: list[dict[str, Any]] = []
        for ev in events:
            data = ev.data or {}
            from_role = str(data.get("from_role", ""))
            text = str(data.get("text", ""))
            if not self._include_own and from_role == self._role:
                continue
            if from_filter and from_role != from_filter:
                continue
            msgs.append(
                {
                    "from_role": from_role,
                    "text": text,
                    "timestamp": ev.timestamp.replace(tzinfo=UTC).isoformat()
                    if ev.timestamp.tzinfo is None
                    else ev.timestamp.isoformat(),
                }
            )

        if not msgs:
            content = (
                "(no messages from peers"
                + (f" matching from_role={from_filter!r}" if from_filter else "")
                + ")"
            )
        else:
            lines = [f"{m['timestamp']}  [{m['from_role']}] {m['text']}" for m in msgs]
            content = "\n".join(lines)
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=content,
            metadata={"count": len(msgs)},
        )


__all__ = ["CheckMessagesTool", "NotifyTool"]
