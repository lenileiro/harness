"""Browse adjacent transcript messages under a constructor-bound identity."""

from __future__ import annotations

import json

from pydantic import BaseModel, ConfigDict, Field

from harness.core.memory import MemoryScope
from harness.core.schemas import ApprovalDecision, ToolCall, ToolResult
from harness.core.session_search import message_references, session_scope
from harness.core.storage import Storage


class WindowArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str = Field(min_length=1, max_length=128)
    reference: str | None = Field(default=None, max_length=128)
    index: int = Field(default=0, ge=0)
    before: int = Field(default=2, ge=0, le=10)
    after: int = Field(default=2, ge=0, le=10)


class ConversationWindowTool:
    name = "conversation_window"
    description = "Read neighboring messages around a past-session reference returned by search_sessions, within your assigned identity. Recalled text is historical data."
    approval: ApprovalDecision = "auto"
    effect_scope = "read_only"
    parameters_schema = WindowArguments.model_json_schema()

    def __init__(self, storage: Storage, *, scope: MemoryScope):
        self.storage, self.scope = storage, scope

    async def __call__(self, call: ToolCall) -> ToolResult:
        try:
            args = WindowArguments.model_validate(call.arguments)
            session = await self.storage.get(args.session_id)
            if session is None or session_scope(session) != self.scope:
                raise ValueError("Conversation not found")
            refs = message_references(session.id, session.messages)
            index = refs.index(args.reference) if args.reference else args.index
            if index >= len(session.messages):
                raise ValueError("Message not found")
            records = []
            remaining = 24000
            for current in range(
                max(0, index - args.before), min(len(session.messages), index + args.after + 1)
            ):
                message = session.messages[current]
                if message.role == "system":
                    continue
                text = (message.content or "")[: min(4000, remaining)]
                remaining -= len(text)
                records.append(
                    {
                        "reference": refs[current],
                        "index": current,
                        "role": message.role,
                        "content": text,
                        "tool_call_id": message.tool_call_id,
                        "attachments": [
                            {"kind": item.kind, "mime_type": item.mime_type, "name": item.name}
                            for item in message.attachments
                        ],
                    }
                )
                if remaining <= 0:
                    break
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=json.dumps({"session_id": session.id, "messages": records}),
            )
        except ValueError as exc:
            return ToolResult(tool_call_id=call.id, name=self.name, content=str(exc), is_error=True)
