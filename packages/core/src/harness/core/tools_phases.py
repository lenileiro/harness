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

        # Read current phase state from the activity log so the advisory
        # check sees the same source of truth the runtime uses to
        # reconstruct session.phases.
        declared_order, completed_names = await self._derive_state()
        in_flight = [p for p in declared_order if p not in completed_names]

        # ── Advisory gates ───────────────────────────────────────────────
        # We deliberately don't refuse here — early experiments showed
        # weak-instruction-following models get stuck in refusal loops
        # they can't escape. Instead the call always succeeds and a
        # warning is surfaced when the transition looks malformed. The
        # *hard* gate stays at Done time via PhaseGateVerifier: any
        # declared phase that's never marked complete blocks completion.
        warnings: list[str] = []
        side_effects: list[tuple[str, str]] = []  # extra (kind, phase) events to write
        skip_primary_event = False

        if action == "declare":
            if name in declared_order:
                # Idempotent. No state change, just an INFO line so the
                # agent knows its redeclare was a no-op.
                completion_state = "completed" if name in completed_names else "in flight"
                warnings.append(
                    f"[INFO] phase {name!r} was already declared "
                    f"(currently {completion_state}); re-declare is a no-op."
                )
                skip_primary_event = True
            elif in_flight:
                current = in_flight[-1]
                warnings.append(
                    f"[WARNING] declaring phase {name!r} while phase {current!r} "
                    f"is still in flight. If {current!r} is actually done, complete "
                    f"it via phase(action='complete', name={current!r}). "
                    f"PhaseGateVerifier will refuse Done while any phase is "
                    f"outstanding."
                )
        else:  # complete
            if not in_flight:
                if name in completed_names:
                    warnings.append(
                        f"[INFO] phase {name!r} was already completed; " f"re-complete is a no-op."
                    )
                    skip_primary_event = True
                else:
                    warnings.append(
                        f"[WARNING] no phase was in flight when you marked "
                        f"{name!r} complete. Recording {name!r} as both "
                        f"declared and completed in a single step."
                    )
                    # Synthesize the missing declare so state stays consistent.
                    side_effects.append((activity_kinds.PHASE_DECLARED, name))
            else:
                expected = in_flight[-1]
                if name != expected:
                    if name in declared_order:
                        # Completing an older still-pending phase out of order.
                        warnings.append(
                            f"[WARNING] completing phase {name!r} while {expected!r} "
                            f"is the most recent in-flight phase. Phases are "
                            f"usually completed in declaration order; mark "
                            f"{expected!r} too if it's done."
                        )
                    else:
                        # Completing a name that was never declared.
                        warnings.append(
                            f"[WARNING] phase {name!r} was never declared. "
                            f"Recording it as declared+completed in a single step. "
                            f"The currently in-flight phase {expected!r} is still "
                            f"outstanding."
                        )
                        side_effects.append((activity_kinds.PHASE_DECLARED, name))

        # Write side-effect events (synthetic declares) before the primary one.
        if self._activity_store is not None:
            for side_kind, side_name in side_effects:
                ev = ActivityEvent(
                    session_id=self._session_id,
                    kind=side_kind,
                    data={"phase": side_name, "auto_declared": True},
                )
                await self._activity_store.append_activity(ev)
            if not skip_primary_event:
                primary_kind = (
                    activity_kinds.PHASE_DECLARED
                    if action == "declare"
                    else activity_kinds.PHASE_COMPLETED
                )
                primary_event = ActivityEvent(
                    session_id=self._session_id,
                    kind=primary_kind,
                    data={"phase": name, "notes": notes} if notes else {"phase": name},
                )
                await self._activity_store.append_activity(primary_event)

        verb = "declared" if action == "declare" else "completed"
        body = f"phase {name!r} {verb}" + (f": {notes}" if notes else "")
        content = "\n".join([*warnings, body]) if warnings else body
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=content,
            metadata={"phase": name, "action": action, "warnings": len(warnings)},
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
