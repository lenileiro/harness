"""Pydantic schemas for the Harness runtime.

These mirror the OpenAI chat-completions wire format closely because both
supported providers (OpenRouter, Ollama) speak OpenAI-compatible APIs.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant", "tool"]
ApprovalDecision = Literal["auto", "prompt", "deny"]
SessionStatus = Literal["pending", "running", "paused", "done", "failed", "cancelled"]
EffectScope = Literal[
    "read_only",  # read_file, list_dir — never approval-required
    "session_ephemeral",  # in-process scratch state, disappears with session
    "task_durable",  # writes to harness task/session storage
    "agent_orchestration",  # spawns child agents
    "workspace_durable",  # write_file, edit_file, shell — mutates workspace
    "external_side_effect",  # HTTP calls, external APIs — irreversible
    "routed",  # goes through an action router
]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Tool call wire types
# ---------------------------------------------------------------------------


class ToolCall(BaseModel):
    """A request from the model to invoke a tool.

    `arguments` holds the already-parsed JSON object. Adapters are responsible
    for parsing the raw JSON string the LLM emits.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """The outcome of executing a tool call.

    `content` is always a string — tools serialize their result themselves
    (JSON, text, whatever the model can read).

    `metadata` carries structured evidence for the activity ledger — e.g.
    `{"exit_code": 0, "duration_ms": 42}` for a shell call, `{"bytes": 1024}`
    for a file read. The model never sees `metadata`; only the runtime does,
    and only to attach it to the emitted `tool_call.completed` event.
    """

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False
    metadata: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Conversation messages
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single turn in the conversation, OpenAI-compatible.

    Field validity by role:
      - system / user: `content` required, no tool_calls / tool_call_id
      - assistant:     `content` optional, `tool_calls` optional
      - tool:          `content` required, `tool_call_id` required, `name` required

    `cache_breakpoint=True` is a hint to cache-aware adapters: place a
    provider-specific cache anchor on this message so the prefix up to
    and including it is cached for subsequent turns. The runtime sets
    this on the last stable system block when ordering messages for a
    cache-friendly layout (`Don't Break the Cache`, arXiv 2601.06007).
    Adapters that don't support explicit cache markers ignore it; the
    ordering itself still produces prefix-cache hits via byte match.
    """

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    cache_breakpoint: bool = False


# ---------------------------------------------------------------------------
# Provider capabilities
# ---------------------------------------------------------------------------


class Capabilities(BaseModel):
    """What an adapter can do. Used by the runtime to gate features."""

    model_config = ConfigDict(extra="forbid")

    streaming: bool = True
    tool_use: bool = False
    structured_output: bool = False
    max_context_tokens: int | None = None
    models: list[str] | None = None


# ---------------------------------------------------------------------------
# Usage / cost
# ---------------------------------------------------------------------------


class Note(BaseModel):
    """An agent-authored scratchpad entry attached to a session.

    Notes are the *additive* half of Memory-as-Action (arXiv 2510.12635):
    the agent uses a tool to write a durable observation it wants to
    keep around even when older transcript messages are pruned. The
    runtime injects the current notes list as a system block on every
    turn, so the agent can refer back to them without paying for the
    raw transcript bytes.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: _new_id("note"))
    text: str
    created_at: datetime = Field(default_factory=_utcnow)
    tags: list[str] = Field(default_factory=list)


class Usage(BaseModel):
    """Token accounting for a single adapter turn.

    The ``cache_*`` fields are populated by providers that surface prompt-
    cache statistics (Anthropic via ``cache_creation_input_tokens`` /
    ``cache_read_input_tokens``). Adapters that don't expose them leave
    the fields at 0; the defense ledger uses non-zero values to track
    cache hit ratios over a run.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cache_creation_input_tokens: int = 0
    """Tokens written into the provider's prompt cache on this turn."""
    cache_read_input_tokens: int = 0
    """Tokens served from the provider's prompt cache on this turn."""


# ---------------------------------------------------------------------------
# Phase status (native runtime coordination state)
# ---------------------------------------------------------------------------


class PhaseStatus(BaseModel):
    """A single phase in a multi-step task — declared, in-flight, or done.

    Phases are a first-class runtime concept: callers can declare an
    ordered list via ``RunRequest.phases`` (or omit them entirely), the
    agent advances them via the ``phase`` tool, and the runtime emits
    ``PhaseStartedEvent`` / ``PhaseCompletedEvent`` so consumers (CLI,
    eval, external coordinators) see transitions live.

    Status transitions are append-only on declared_at and completed_at:
    once declared, a phase can either stay declared (in flight) or move
    to completed. Reopening a completed phase is not modeled.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    """Lowercase identifier, e.g. 'implement', 'test', 'document'."""
    declared_at: datetime = Field(default_factory=_utcnow)
    completed_at: datetime | None = None
    notes: list[str] = Field(default_factory=list)
    """Optional notes captured at declare- and complete-time, oldest first."""

    @property
    def is_complete(self) -> bool:
        return self.completed_at is not None


# ---------------------------------------------------------------------------
# Run request (input to Agent.run)
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """Input to `Agent.run`.

    `prompt` is the user's new turn — the runtime appends it as a user
    Message before invoking the adapter. If `session_id` matches an existing
    session, history is loaded; otherwise a new session is created.

    Set `result_type` to a Pydantic :class:`~pydantic.BaseModel` subclass to
    enable structured output: the runtime injects a JSON-schema system message
    and validates the final assistant response against it, retrying on failure.
    """

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    prompt: str
    session_id: str = Field(default_factory=lambda: _new_id("sess"))
    provider: str | None = None
    model: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    max_steps: int = 25
    stream: bool = True
    task_id: str | None = None
    """Optional id of the parent Task. When set on a new session, the session
    inherits this `task_id`."""
    result_type: type | None = Field(default=None, exclude=True)
    """When set to a Pydantic model class, the runtime validates the final
    response as JSON and populates ``Done.structured_result``."""
    require_tool_use: bool = False
    """When True, the runtime forces the model to call at least one tool before
    emitting a final answer. Prevents models from answering from memory when
    tool evidence is required."""
    phases: list[str] | None = None
    """Optional ordered list of phase names the runtime should track. When
    set, the runtime pre-populates :attr:`Session.phases` with declared
    entries and ``PhaseGateVerifier`` will refuse Done until each one has
    been completed. When None, phases are still trackable via the
    ``phase`` tool but the runtime doesn't pre-declare any."""


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session(BaseModel):
    """Durable state for a single conversation.

    Owned by the runtime; the adapter is stateless per call.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: _new_id("sess"))
    provider: str
    model: str
    cwd: Path
    messages: list[Message] = Field(default_factory=list)
    status: SessionStatus = "pending"
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)
    approval_overrides: dict[str, ApprovalDecision] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    task_id: str | None = None
    """Optional id of the parent Task (`harness.tasks.Task`). None for legacy
    / standalone sessions; set when a session was created under `--task T-NNN`."""
    forked_from: str | None = None
    """Parent session id when this session was created via `harness sessions fork`."""
    phases: list[PhaseStatus] = Field(default_factory=list)
    """Ordered list of declared phases. Populated from ``RunRequest.phases``
    at run start (if set) and updated as the agent advances. Defaults to
    empty for backward compatibility with existing serialized sessions."""
    notes: list[Note] = Field(default_factory=list)
    """Agent-authored scratchpad. Written via the ``notes`` tool; injected
    as a system block on each turn so the agent can carry observations
    across context-budget prunes. Memory-as-Action (arXiv 2510.12635)."""

    def touch(self) -> None:
        """Bump `updated_at` to now."""
        self.updated_at = _utcnow()

    def phase_by_name(self, name: str) -> PhaseStatus | None:
        """Return the phase entry with this name, or None."""
        lookup = name.strip().lower()
        for phase in self.phases:
            if phase.name == lookup:
                return phase
        return None

    def outstanding_phases(self) -> list[str]:
        """Names of declared phases that haven't been completed yet."""
        return [p.name for p in self.phases if not p.is_complete]


# ---------------------------------------------------------------------------
# Verification result
# ---------------------------------------------------------------------------


class VerificationResult(BaseModel):
    """A verifier's verdict on whether the run can finish.

    Schema only — the `Verifier` Protocol + implementations live in
    `harness.core.verification`. We keep the schema here so the
    `Verification` event in `events.py` can reference it without a
    circular import.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: _new_id("ver"))
    can_finish: bool
    reason: str
    confidence: float | None = None
    """0.0 to 1.0 if the verifier reports a confidence; None otherwise."""
    evidence_event_ids: list[str] = Field(default_factory=list)
    """Activity event ids the verifier relied on (e.g. failing tool calls)."""
    verifier_name: str
    verified_at: datetime = Field(default_factory=_utcnow)


__all__ = [
    "ApprovalDecision",
    "Capabilities",
    "EffectScope",
    "Message",
    "Note",
    "Role",
    "RunRequest",
    "Session",
    "SessionStatus",
    "ToolCall",
    "ToolResult",
    "Usage",
    "VerificationResult",
]
