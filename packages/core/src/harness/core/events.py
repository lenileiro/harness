"""Event stream produced by `Agent.run`.

Events are normalized across providers. Adapters convert provider-specific
streaming chunks into these types; the runtime consumes and re-emits them
(plus injects its own `step_*` and `done` events).

Discriminated by the `type` field so they can be serialized/persisted/replayed
losslessly.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from harness.core.schemas import Message, ToolCall, ToolResult, Usage, VerificationResult


class _EventBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TextDelta(_EventBase):
    """An incremental chunk of assistant text. May arrive many times per turn."""

    type: Literal["text_delta"] = "text_delta"
    text: str


class ToolCallEvent(_EventBase):
    """The model has decided to call a tool. Emitted once per call."""

    type: Literal["tool_call"] = "tool_call"
    call: ToolCall


class ToolResultEvent(_EventBase):
    """The runtime has invoked a tool and is feeding the result back."""

    type: Literal["tool_result"] = "tool_result"
    result: ToolResult


class StepStarted(_EventBase):
    """A new step in the agent plan has begun."""

    type: Literal["step_started"] = "step_started"
    step: int
    description: str | None = None
    total_steps: int = 0  # 0 means unknown / single-step plan


class StepCompleted(_EventBase):
    """A step in the agent plan has completed (successfully or not)."""

    type: Literal["step_completed"] = "step_completed"
    step: int


class Done(_EventBase):
    """Terminal event — the run finished successfully.

    `final_message` is the assistant's last Message (the answer).
    """

    type: Literal["done"] = "done"
    final_message: Message | None = None
    usage: Usage | None = None


class ErrorEvent(_EventBase):
    """Terminal event — the run failed.

    `error` is a short human-readable string. `kind` mirrors the error class
    name so consumers can react programmatically without re-raising.
    """

    type: Literal["error"] = "error"
    error: str
    kind: str
    recoverable: bool = False


class Verification(_EventBase):
    """Post-run verdict from the configured `Verifier`.

    Emitted by the agent (after `Done`) when a verifier is configured.
    Consumers can use this to gate "actually done" vs "needs review".
    """

    type: Literal["verification"] = "verification"
    result: VerificationResult


Event = Annotated[
    TextDelta
    | ToolCallEvent
    | ToolResultEvent
    | StepStarted
    | StepCompleted
    | Done
    | ErrorEvent
    | Verification,
    Field(discriminator="type"),
]


__all__ = [
    "Done",
    "ErrorEvent",
    "Event",
    "StepCompleted",
    "StepStarted",
    "TextDelta",
    "ToolCallEvent",
    "ToolResultEvent",
    "Verification",
]
