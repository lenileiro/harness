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
    """

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    name: str
    content: str
    is_error: bool = False


# ---------------------------------------------------------------------------
# Conversation messages
# ---------------------------------------------------------------------------


class Message(BaseModel):
    """A single turn in the conversation, OpenAI-compatible.

    Field validity by role:
      - system / user: `content` required, no tool_calls / tool_call_id
      - assistant:     `content` optional, `tool_calls` optional
      - tool:          `content` required, `tool_call_id` required, `name` required
    """

    model_config = ConfigDict(extra="forbid")

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


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


class Usage(BaseModel):
    """Token accounting for a single adapter turn."""

    model_config = ConfigDict(extra="forbid")

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


# ---------------------------------------------------------------------------
# Run request (input to Agent.run)
# ---------------------------------------------------------------------------


class RunRequest(BaseModel):
    """Input to `Agent.run`.

    `prompt` is the user's new turn — the runtime appends it as a user
    Message before invoking the adapter. If `session_id` matches an existing
    session, history is loaded; otherwise a new session is created.
    """

    model_config = ConfigDict(extra="forbid")

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

    def touch(self) -> None:
        """Bump `updated_at` to now."""
        self.updated_at = _utcnow()


__all__ = [
    "ApprovalDecision",
    "Capabilities",
    "Message",
    "Role",
    "RunRequest",
    "Session",
    "SessionStatus",
    "ToolCall",
    "ToolResult",
    "Usage",
]
