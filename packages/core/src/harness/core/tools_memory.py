"""Memory-as-Action tools.

Pattern borrowed from *Memory as Action: Autonomous Context Curation for
Long-Horizon Agentic Tasks* (arXiv 2510.12635). Their thesis: context
surgery (delete spans, summarize spans, append notes) should be
first-class actions in the agent's action space, not an implicit
runtime auto-compaction. The agent decides what to keep and what to
drop based on the task at hand.

We ship the *additive* half cleanly and the *subtractive* half
conservatively:

  NotesTool          — append, list, delete the agent's scratchpad.
                       The notes are durable across turns and the runtime
                       injects them as a system block, so the agent can
                       summarize "what matters" without paying for the
                       raw transcript bytes again.

  PruneLedgerTool    — drops *paired* (assistant tool_calls, tool result)
                       exchanges older than the last K turns. Conservative
                       on purpose: we never break tool-call ↔ tool-result
                       pairing, never drop user messages, never drop
                       system messages. Net effect is "compact the
                       transcript, keep the spine."

The agent typically uses these together: write a note summarizing what
N tool calls discovered, then prune those calls. The note replaces the
calls as the agent's working memory of that span.
"""

from __future__ import annotations

from typing import Any

from harness.core import activity as activity_kinds
from harness.core.activity import ActivityEvent, ActivityStore
from harness.core.schemas import (
    ApprovalDecision,
    Note,
    Session,
    ToolCall,
    ToolResult,
)

_NOTES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["add", "list", "delete"],
            "description": (
                "What to do: 'add' a new note, 'list' all notes, " "'delete' a note by its id."
            ),
        },
        "text": {
            "type": "string",
            "description": "The note body. Required for action='add'.",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional tags for filtering / grouping.",
        },
        "id": {
            "type": "string",
            "description": "The note id. Required for action='delete'.",
        },
    },
    "required": ["action"],
}


class NotesTool:
    """Read/write the session's note scratchpad.

    The tool mutates ``session.notes`` directly because notes are
    session-scoped state, not durable cross-session memory. The runtime
    picks up the changes on the next ``storage.save(session)`` call.

    Approval is ``"auto"`` because notes are purely additive workspace
    state — no filesystem mutation, no external side effect.
    """

    name = "notes"
    description = (
        "Persistent scratchpad attached to this session. Use 'add' to write "
        "a one-paragraph observation you want to keep around across turns. "
        "Use 'list' to see all notes. Use 'delete' with an id to remove one. "
        "Notes are automatically injected into context on every turn — they "
        "are how you carry memory across pruning."
    )
    approval: ApprovalDecision = "auto"
    effect_scope = "session_ephemeral"
    phases: tuple[str, ...] = ("*",)

    def __init__(
        self,
        *,
        session: Session,
        activity_store: ActivityStore | None = None,
    ) -> None:
        # Hold a reference to the session so mutations are visible to the
        # runtime. The runtime is single-threaded per session so this is safe.
        self._session = session
        self._activity_store = activity_store
        self.parameters_schema = _NOTES_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        args = call.arguments if isinstance(call.arguments, dict) else {}
        action = str(args.get("action", "")).strip().lower()

        if action not in ("add", "list", "delete"):
            return self._error(call, "action must be one of: add, list, delete")

        if action == "add":
            text = str(args.get("text") or "").strip()
            if not text:
                return self._error(call, "text is required for add")
            if len(text) > 1500:
                return self._error(
                    call,
                    f"note text too long ({len(text)} chars; max 1500). "
                    "Split into smaller notes or summarize.",
                )
            tags_raw = args.get("tags") or []
            tags = [str(t).strip() for t in tags_raw if str(t).strip()]
            note = Note(text=text, tags=tags)
            self._session.notes.append(note)
            await self._emit(
                activity_kinds.NOTE_WRITTEN,
                {"id": note.id, "tags": tags, "text_preview": text[:120]},
            )
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"note {note.id} added",
                metadata={"id": note.id},
            )

        if action == "list":
            if not self._session.notes:
                return ToolResult(
                    tool_call_id=call.id,
                    name=self.name,
                    content="(no notes yet)",
                )
            lines = []
            for note in self._session.notes:
                tag_part = f"  [{', '.join(note.tags)}]" if note.tags else ""
                lines.append(f"- {note.id}{tag_part}: {note.text}")
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="\n".join(lines),
            )

        # delete
        note_id = str(args.get("id") or "").strip()
        if not note_id:
            return self._error(call, "id is required for delete")
        before = len(self._session.notes)
        self._session.notes = [n for n in self._session.notes if n.id != note_id]
        if len(self._session.notes) == before:
            return self._error(call, f"no note with id={note_id!r}")
        await self._emit(activity_kinds.NOTE_DELETED, {"id": note_id})
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=f"note {note_id} deleted",
        )

    async def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self._activity_store is None:
            return
        ev = ActivityEvent(session_id=self._session.id, kind=kind, data=data)
        await self._activity_store.append_activity(ev)

    @staticmethod
    def _error(call: ToolCall, message: str) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name="notes",
            content=message,
            is_error=True,
        )


# ---------------------------------------------------------------------------
# PruneLedgerTool
# ---------------------------------------------------------------------------


_PRUNE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "keep_recent_turns": {
            "type": "integer",
            "description": (
                "Number of recent (assistant→tool) exchanges to keep. "
                "Older paired tool exchanges are dropped. User messages "
                "and system messages are never dropped. Default 4."
            ),
            "minimum": 1,
            "maximum": 50,
        },
    },
    "required": [],
}


class PruneLedgerTool:
    """Drop paired old tool exchanges to reclaim context budget.

    Conservative rules — the tool ONLY removes (assistant with
    tool_calls, tool_result) pairs older than the keep-recent window.
    User and system messages stay. The most recent N assistant/tool
    turns also stay so the model retains its working state.

    Returns a one-line summary of how many messages were dropped.
    """

    name = "prune_ledger"
    description = (
        "Drop older paired tool-call/tool-result exchanges from this "
        "session's transcript to reclaim context budget. Pass "
        "'keep_recent_turns' (default 4) to control how much working "
        "state stays. Useful after a long investigation phase: write a "
        "note summarizing what you learned, then call prune_ledger to "
        "drop the raw tool exchanges that informed the note."
    )
    approval: ApprovalDecision = "auto"
    effect_scope = "session_ephemeral"
    phases: tuple[str, ...] = ("*",)

    def __init__(
        self,
        *,
        session: Session,
        activity_store: ActivityStore | None = None,
    ) -> None:
        self._session = session
        self._activity_store = activity_store
        self.parameters_schema = _PRUNE_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        args = call.arguments if isinstance(call.arguments, dict) else {}
        keep = int(args.get("keep_recent_turns") or 4)
        if keep < 1:
            keep = 1
        if keep > 50:
            keep = 50

        # Find tool-call/tool-result pairs in the transcript and keep
        # only the most recent ``keep`` of them. We walk from the end so
        # the index math is simpler.
        messages = self._session.messages
        # Pair indexes: an assistant message with tool_calls plus all
        # following tool messages until the next non-tool message.
        pairs: list[tuple[int, int]] = []  # (start_idx_inclusive, end_idx_exclusive)
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg.role == "assistant" and msg.tool_calls:
                start = i
                j = i + 1
                while j < len(messages) and messages[j].role == "tool":
                    j += 1
                pairs.append((start, j))
                i = j
            else:
                i += 1

        if len(pairs) <= keep:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=(
                    f"nothing to prune: {len(pairs)} tool exchange(s) on record, "
                    f"keep_recent_turns={keep}"
                ),
                metadata={"dropped_pairs": 0, "pairs_remaining": len(pairs)},
            )

        # Drop all pairs except the last `keep`.
        drop_pairs = pairs[: len(pairs) - keep]
        drop_ranges: set[int] = set()
        for start, end in drop_pairs:
            drop_ranges.update(range(start, end))

        before_len = len(messages)
        self._session.messages = [m for idx, m in enumerate(messages) if idx not in drop_ranges]
        dropped = before_len - len(self._session.messages)

        await self._emit(
            activity_kinds.LEDGER_PRUNED,
            {
                "messages_dropped": dropped,
                "pairs_dropped": len(drop_pairs),
                "pairs_remaining": keep,
            },
        )
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=(
                f"pruned {dropped} message(s) across {len(drop_pairs)} old "
                f"tool exchange(s); {keep} kept."
            ),
            metadata={"dropped_pairs": len(drop_pairs), "pairs_remaining": keep},
        )

    async def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self._activity_store is None:
            return
        ev = ActivityEvent(session_id=self._session.id, kind=kind, data=data)
        await self._activity_store.append_activity(ev)


__all__ = ["NotesTool", "PruneLedgerTool"]
