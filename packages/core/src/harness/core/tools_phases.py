"""Phase declaration tool — coordination primitive for multi-step work.

Long-running tasks have an internal SDLC: implement → test → document →
verify, for instance. Without an explicit phase tracker, an agent skips
steps to please the user (one of the most common sycophancy failure modes:
"I implemented and tested it" while never actually running the tests).

The `phase` tool gives the agent (or an external caller via the CLI) an
explicit place to record "I'm starting / finishing phase X." A companion
verifier reads the activity log and refuses Done when the agent declared
phases out of order or skipped completing one.

This module is the in-process tool. The external CLI counterpart
(``harness phase ...``) writes the same activity events directly, so an
external coordinator (Claude Code, Cursor, an MCP-served harness) sees
the same state.
"""

from __future__ import annotations

from typing import Any, Literal

from harness.core import activity as activity_kinds
from harness.core.activity import ActivityEvent, ActivityStore
from harness.core.schemas import ApprovalDecision, ToolCall, ToolResult

_PHASE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["declare", "complete", "status"],
            "description": (
                "What to do: 'declare' starts a phase, 'complete' marks it "
                "done, 'status' returns the current declared/completed phases."
            ),
        },
        "name": {
            "type": "string",
            "description": (
                "Name of the phase. Required for declare and complete; "
                "ignored for status. Convention: short single-word "
                "lowercase identifiers like 'implement', 'test', 'document'."
            ),
        },
        "notes": {
            "type": "string",
            "description": "Optional one-line note about this phase transition.",
        },
    },
    "required": ["action"],
}


PhaseAction = Literal["declare", "complete", "status"]


class PhaseTool:
    """Declare and complete phases of a multi-step task.

    Args:
        session_id: The session whose activity log records phase events.
                   When `activity_store` is None this is informational only.
        activity_store: Where to append PHASE_DECLARED / PHASE_COMPLETED
                       events. When None, the tool still returns useful
                       text but the verifier won't see anything.

    The tool is read_only from the file-system perspective (it only writes
    to the activity ledger), so default approval is auto.
    """

    name = "phase"
    description = (
        "Declare or complete phases of a multi-step task. Use this when the "
        "user's prompt enumerates several sequential steps (e.g. 'implement "
        "X, then add tests, then update the README'). Call with "
        "action='declare', name='<phase>' before starting work on a phase; "
        "call with action='complete', name='<phase>' after the phase's "
        "evidence (test pass, file written, etc.) is in. Call action='status' "
        "to see what you've declared so far. The harness enforces ordering "
        "via the PhaseGateVerifier — skipping phases will block Done."
    )
    approval: ApprovalDecision = "auto"
    effect_scope = "session_ephemeral"
    phases: tuple[str, ...] = ("*",)

    def __init__(
        self,
        *,
        session_id: str | None = None,
        activity_store: ActivityStore | None = None,
    ) -> None:
        self._session_id = session_id
        self._activity_store = activity_store
        self.parameters_schema = _PHASE_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        args = call.arguments if isinstance(call.arguments, dict) else {}
        action = args.get("action", "").strip().lower()
        name = (args.get("name") or "").strip().lower()
        notes = (args.get("notes") or "").strip()

        if action not in ("declare", "complete", "status"):
            return self._error(call, "action must be one of: declare, complete, status")

        if action != "status" and not name:
            return self._error(call, "name is required for declare/complete")

        if action == "status":
            return await self._status(call)

        # Read current phase state from the activity log so the gate sees the
        # same source of truth the runtime uses to reconstruct session.phases.
        declared_order, completed_names = await self._derive_state()
        in_flight = [p for p in declared_order if p not in completed_names]

        # ── Gates ────────────────────────────────────────────────────────
        # The LLM drives phase creation, but each transition has to pass
        # through here. Out-of-order or skipped phases are refused with a
        # concrete error the agent can act on.
        if action == "declare":
            if in_flight:
                current = in_flight[-1]
                return self._error(
                    call,
                    f"cannot declare phase {name!r}: phase {current!r} is still "
                    f"in flight. Call phase(action='complete', name={current!r}) "
                    f"with evidence first, then declare the next phase.",
                )
            if name in declared_order:
                return self._error(
                    call,
                    f"phase {name!r} was already declared (status: "
                    f"{'completed' if name in completed_names else 'unknown'}). "
                    f"Phase names must be unique within a session.",
                )
        else:  # complete
            if not in_flight:
                return self._error(
                    call,
                    f"cannot complete phase {name!r}: no phase is currently in "
                    f"flight. Use phase(action='declare', name={name!r}) to "
                    f"start it first.",
                )
            expected = in_flight[-1]
            if name != expected:
                return self._error(
                    call,
                    f"cannot complete phase {name!r}: the currently in-flight "
                    f"phase is {expected!r}. Complete that one first, or use "
                    f"phase(action='status') to see the current plan.",
                )

        # Gate passed — record the transition.
        kind = (
            activity_kinds.PHASE_DECLARED if action == "declare" else activity_kinds.PHASE_COMPLETED
        )
        if self._activity_store is not None:
            event = ActivityEvent(
                session_id=self._session_id,
                kind=kind,
                data={"phase": name, "notes": notes} if notes else {"phase": name},
            )
            await self._activity_store.append_activity(event)

        verb = "declared" if action == "declare" else "completed"
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=f"phase {name!r} {verb}" + (f": {notes}" if notes else ""),
            metadata={"phase": name, "action": action},
        )

    async def _derive_state(self) -> tuple[list[str], set[str]]:
        """Return (declared_order, completed_names) from this session's activity log."""
        if self._activity_store is None or self._session_id is None:
            return [], set()
        events = await self._activity_store.list_activity(
            session_id=self._session_id,
            kinds=(activity_kinds.PHASE_DECLARED, activity_kinds.PHASE_COMPLETED),
            limit=500,
        )
        declared_order: list[str] = []
        completed: set[str] = set()
        for ev in events:
            pname = str((ev.data or {}).get("phase", "")).strip().lower()
            if not pname:
                continue
            if ev.kind == activity_kinds.PHASE_DECLARED and pname not in declared_order:
                declared_order.append(pname)
            elif ev.kind == activity_kinds.PHASE_COMPLETED:
                completed.add(pname)
        return declared_order, completed

    async def _status(self, call: ToolCall) -> ToolResult:
        if self._activity_store is None or self._session_id is None:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="no activity store wired; status unavailable",
            )
        events = await self._activity_store.list_activity(
            session_id=self._session_id,
            kinds=(activity_kinds.PHASE_DECLARED, activity_kinds.PHASE_COMPLETED),
            limit=200,
        )
        declared: list[str] = []
        completed: set[str] = set()
        for ev in events:
            name = str((ev.data or {}).get("phase", "")).strip()
            if not name:
                continue
            if ev.kind == activity_kinds.PHASE_DECLARED and name not in declared:
                declared.append(name)
            elif ev.kind == activity_kinds.PHASE_COMPLETED:
                completed.add(name)
        if not declared:
            content = "no phases declared yet"
        else:
            lines = [f"phases declared (in order): {', '.join(declared)}"]
            if completed:
                lines.append(f"completed: {', '.join(sorted(completed))}")
            outstanding = [p for p in declared if p not in completed]
            if outstanding:
                lines.append(f"outstanding: {', '.join(outstanding)}")
            content = "\n".join(lines)
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=content,
            metadata={"declared": declared, "completed": sorted(completed)},
        )

    @staticmethod
    def _error(call: ToolCall, message: str) -> ToolResult:
        return ToolResult(
            tool_call_id=call.id,
            name="phase",
            content=message,
            is_error=True,
        )


__all__ = ["PhaseAction", "PhaseTool"]
