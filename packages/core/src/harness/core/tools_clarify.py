"""Model-facing questions; the runtime pauses and resumes their durable results."""

from __future__ import annotations

import asyncio
import json

from pydantic import ValidationError

from harness.core.clarification import ClarifyArguments, QuestionStore
from harness.core.memory import MemoryScope
from harness.core.schemas import ApprovalDecision, ToolCall, ToolResult


class ClarifyTool:
    name = "clarify"
    description = (
        "Use only when explicitly enabled and essential information remains unavailable after "
        "inspecting relevant context and using available research tools, or when the user "
        "explicitly requested an interactive decision. Prefer reasonable reversible assumptions "
        "for ordinary implementation choices; do not ask for discoverable files, facts or commands. "
        "Pass 1-5 independent questions together; each supports up to four choices "
        "(recommended first), multiple selection, or open text. The run pauses until "
        "the owner answers or the question expires. Never use this tool for permission "
        "to run dangerous commands; tool approvals are separate."
    )
    approval: ApprovalDecision = "auto"
    effect_scope = "session_ephemeral"
    phases = ("*",)

    def __init__(self, store: QuestionStore, *, session_id: str, scope: MemoryScope):
        self.store, self.session_id, self.scope = store, session_id, scope
        self.parameters_schema = ClarifyArguments.model_json_schema()

    async def __call__(self, call: ToolCall) -> ToolResult:
        try:
            args = ClarifyArguments.model_validate(call.arguments)
            pending = asyncio.create_task(
                asyncio.to_thread(
                    self.store.create,
                    session_id=self.session_id,
                    scope=self.scope,
                    tool_call_id=call.id,
                    questions=args.questions,
                )
            )
            try:
                record = await asyncio.shield(pending)
            except asyncio.CancelledError:
                # A SQLite worker cannot be cancelled while waiting for a write
                # lock. Settle it before cleanup so it cannot create a live
                # question after the owning run has already been cancelled.
                try:
                    record = await pending
                    await asyncio.to_thread(
                        self.store.cancel, record.id, scope=self.scope, session_id=self.session_id
                    )
                except Exception:
                    pass
                raise
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=json.dumps({"status": "waiting_for_input", "question_id": record.id}),
                metadata={"pending_question_id": record.id},
            )
        except (ValueError, ValidationError) as exc:
            return ToolResult(tool_call_id=call.id, name=self.name, content=str(exc), is_error=True)
