from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from harness.core import Agent
from harness.core.schemas import MediaAttachment
from harness.storage.sqlite import SQLiteStorage

RunState = Literal["queued", "running", "completed", "paused", "failed", "cancelled", "interrupted"]
TERMINAL_STATES = frozenset({"completed", "paused", "failed", "cancelled", "interrupted"})


class RunSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    prompt: str = Field(default="", max_length=100_000)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)
    model: str | None = Field(default=None, min_length=1, max_length=256)
    provider: str | None = Field(default=None, min_length=1, max_length=128)
    max_steps: int = Field(default=25, ge=1, le=100)
    attachments: list[MediaAttachment] = Field(default_factory=list, max_length=16)


class ToolSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    name: str = Field(min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class RunContext:
    owner: str
    workspace: Path
    storage: SQLiteStorage
    run_id: str
    session_id: str
    model: str | None
    provider: str | None = None


AgentBuilder = Callable[[RunContext], Agent | Awaitable[Agent]]


class ServiceError(ValueError):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


class ProviderOption(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128)
    default_model: str | None = Field(default=None, min_length=1, max_length=256)


class ToolPresentation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1, max_length=128)
    description: str = Field(default="", max_length=10000)
    parameters_schema: dict[str, Any] = Field(default_factory=dict)
    approval: Literal["auto", "prompt", "deny"] = "prompt"
    effect_scope: str = "unknown"


class ServerPresentation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    providers: list[ProviderOption] = Field(default_factory=list, max_length=100)
    default_provider: str | None = None
    default_model: str | None = None
    tools: list[ToolPresentation] = Field(default_factory=list, max_length=1000)


class UserPreferences(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    provider: str | None = Field(default=None, min_length=1, max_length=128)
    model: str | None = Field(default=None, min_length=1, max_length=256)
    timezone: str = Field(default="UTC", min_length=1, max_length=128)
