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

from harness.core.prediction import PredictionOutcome, ToolPrediction
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
    When `result_type` was set on the :class:`~harness.core.schemas.RunRequest`,
    `structured_result` holds the validated model as a plain dict.
    """

    type: Literal["done"] = "done"
    final_message: Message | None = None
    usage: Usage | None = None
    structured_result: dict | None = None


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


class Critique(_EventBase):
    """Critic's challenge to the agent after a failed repair attempt.

    Emitted by the repair loop after a Critic produces its review, before the
    repair directive is appended to the session. Consumers can render this
    distinctively so users can see what hypothesis is being challenged.
    """

    type: Literal["critique"] = "critique"
    attempt: int
    text: str


class PhaseStartedEvent(_EventBase):
    """Emitted by the runtime when a phase is declared (started).

    Fires both when the runtime pre-declares phases from
    ``RunRequest.phases`` and when the agent advances via the ``phase``
    tool. Consumers render this as ``Phase 2/4: test...`` and similar.
    """

    type: Literal["phase_started"] = "phase_started"
    name: str
    notes: str = ""
    index: int = 0
    """Zero-based index in the session's phase list at the time of declaration."""
    total: int = 0
    """Total declared phases at the time of declaration."""


class PhaseCompletedEvent(_EventBase):
    """Emitted by the runtime when a phase is marked complete."""

    type: Literal["phase_completed"] = "phase_completed"
    name: str
    notes: str = ""
    index: int = 0
    total: int = 0


class PredictionEvent(_EventBase):
    """Pre-execution prediction committed by ConsequencePredictor.

    Emitted before a tool executes (after approval, before the actual call).
    Consumers can use this to audit what the runtime expected vs. what happened.
    """

    type: Literal["prediction"] = "prediction"
    prediction: ToolPrediction


class PredictionMismatchEvent(_EventBase):
    """Post-execution outcome where prediction did not match reality.

    Only emitted when `outcome.matched` is False. The `outcome.severity` field
    indicates how serious the mismatch is for this effect_scope.
    """

    type: Literal["prediction_mismatch"] = "prediction_mismatch"
    outcome: PredictionOutcome


class ModelRequestEvent(_EventBase):
    """Emitted immediately before the runtime calls the adapter.

    Useful for agent.iter() consumers and middleware that needs to inspect
    or modify the exact messages sent to the LLM each turn.
    """

    type: Literal["model_request"] = "model_request"
    messages: list[Message]


class GuardrailTrippedEvent(_EventBase):
    """A guardrail fired and the model response was cancelled.

    The run terminates after this event without a Done event.
    """

    type: Literal["guardrail_tripped"] = "guardrail_tripped"
    guardrail_name: str
    reason: str


class HandoffEvent(_EventBase):
    """Control is being handed off to another agent.

    Emitted by the runtime when a tool raises
    :class:`~harness.core.errors.Handoff`. The target agent then takes over
    the session from the same conversation state.
    """

    type: Literal["handoff"] = "handoff"
    target_name: str
    reason: str


Event = Annotated[
    TextDelta
    | ToolCallEvent
    | ToolResultEvent
    | StepStarted
    | StepCompleted
    | Done
    | ErrorEvent
    | Verification
    | Critique
    | PhaseStartedEvent
    | PhaseCompletedEvent
    | PredictionEvent
    | PredictionMismatchEvent
    | ModelRequestEvent
    | GuardrailTrippedEvent
    | HandoffEvent,
    Field(discriminator="type"),
]


__all__ = [
    "Critique",
    "Done",
    "ErrorEvent",
    "Event",
    "GuardrailTrippedEvent",
    "HandoffEvent",
    "ModelRequestEvent",
    "PhaseCompletedEvent",
    "PhaseStartedEvent",
    "PredictionEvent",
    "PredictionMismatchEvent",
    "StepCompleted",
    "StepStarted",
    "TextDelta",
    "ToolCallEvent",
    "ToolResultEvent",
    "Verification",
]
