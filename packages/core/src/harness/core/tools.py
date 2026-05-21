"""Tool Protocol, registry, and approval policy.

A Tool is any async callable with a name, description, JSON Schema, and a
default approval level. The runtime resolves approval through ApprovalPolicy
(which may override the tool's default) and delegates to an ApprovalHandler
when "prompt" is the decision.

## Phase scoping

Tools may declare which agent phases they're available in via the optional
`phases: tuple[str, ...]` attribute. Common values: `"research"`, `"act"`,
`"verify"`, `"repair"`. The wildcard `"*"` means "any phase including no
phase set". When the Agent's `current_phase` is `None`, all tools are
available (backward compatible).

The attribute is optional — tools that don't declare it default to
`("*",)` via the registry's `getattr` fallback.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from harness.core.approval import ApprovalOutcome, ApprovalStore, PendingApproval
from harness.core.schemas import ApprovalDecision, EffectScope, Session, ToolCall, ToolResult

WILDCARD_PHASE = "*"
_DEFAULT_PHASES: tuple[str, ...] = (WILDCARD_PHASE,)


def _tool_phases(tool: Tool) -> tuple[str, ...]:
    """Return the tool's declared phases, defaulting to `("*",)`."""
    return tuple(getattr(tool, "phases", _DEFAULT_PHASES))


def tool_matches_phase(tool: Tool, phase: str | None) -> bool:
    """True if the tool is available in the given phase.

    `phase=None` (no phase set) → all tools available (backward compat).
    Otherwise the tool's `phases` must contain `phase` or the wildcard `"*"`.
    """
    if phase is None:
        return True
    phases = _tool_phases(tool)
    return WILDCARD_PHASE in phases or phase in phases


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@runtime_checkable
class Tool(Protocol):
    """Async callable the agent can invoke.

    Receives the full `ToolCall` (so the tool has access to `call.id`,
    `call.name`, and `call.arguments`) and returns a `ToolResult`. This lets
    tools construct rich results — set `is_error=True` for failures, or
    customize the content shape — while keeping the wire format authoritative.

    The runtime still catches exceptions raised inside `__call__` and wraps
    them as `ToolResult(is_error=True)` so a tool crash never breaks the loop.

    The optional `phases` attribute scopes tool visibility to specific agent
    phases (`"research"`, `"act"`, `"verify"`, ...). The wildcard `"*"` means
    "any phase". The registry treats tools without the attribute as `("*",)`.
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]
    """JSON Schema for the tool's parameter object (an `object` schema)."""
    approval: ApprovalDecision
    """Default approval level. Overridable per-session or globally via ApprovalPolicy."""

    # Note: `phases` and `effect_scope` are intentionally not listed as
    # required attributes here so existing Tool implementations don't break.
    # Access `phases` through `tool_matches_phase` / `_tool_phases`.
    # Access `effect_scope` via `getattr(tool, "effect_scope", None)`.

    async def __call__(self, call: ToolCall) -> ToolResult: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Lookup + OpenAI-format schema export for a set of Tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            raise KeyError(f"tool {name!r} not registered") from exc

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return list(self._tools.values())

    def for_phase(self, phase: str | None) -> list[Tool]:
        """Tools available in the given phase. `None` returns everything."""
        return [t for t in self._tools.values() if tool_matches_phase(t, phase)]

    def openai_schemas(self, phase: str | None = None) -> list[dict[str, Any]]:
        """Render tools in OpenAI's `tools` request format.

        When `phase` is set, only tools whose `phases` allow that phase are
        included. `phase=None` keeps the legacy behavior of returning every
        registered tool.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema,
                },
            }
            for t in self.for_phase(phase)
        ]


# ---------------------------------------------------------------------------
# Approval policy + handler
# ---------------------------------------------------------------------------


class ApprovalPolicy(BaseModel):
    """Maps tool name → approval level.

    `default` is the fallback when a tool isn't in `per_tool` and the Tool's
    own `approval` isn't being respected. The order of precedence at runtime:

      1. Session-level overrides (set via `Session.approval_overrides`)
      2. ApprovalPolicy.per_tool
      3. Tool.approval (the tool's own default)
      4. ApprovalPolicy.default
    """

    model_config = ConfigDict(extra="forbid")

    default: ApprovalDecision = "prompt"
    per_tool: dict[str, ApprovalDecision] = Field(default_factory=dict)

    def decide(
        self,
        tool: Tool,
        *,
        session_overrides: dict[str, ApprovalDecision] | None = None,
    ) -> ApprovalDecision:
        if session_overrides and tool.name in session_overrides:
            return session_overrides[tool.name]
        if tool.name in self.per_tool:
            return self.per_tool[tool.name]
        # Derive approval from effect_scope when the tool doesn't override it.
        scope: EffectScope | None = getattr(tool, "effect_scope", None)
        if scope in ("workspace_durable", "external_side_effect"):
            return "prompt"
        if scope == "read_only":
            return "auto"
        return tool.approval or self.default


@runtime_checkable
class ApprovalHandler(Protocol):
    """Called by the runtime whenever a tool's effective approval is `prompt`.

    Implementations: CLI shows a Rich prompt; tests use auto-approve/deny mocks;
    `InboxApprovalHandler` queues for later human review.

    Receives the current `session` so handlers may persist "always" decisions
    by mutating `session.approval_overrides` (the runtime saves the session
    after each turn).

    Return value semantics:
      - `True` / `"approved"`  → execute the tool
      - `False` / `"denied"`   → synth `ToolResult(content="user denied approval", is_error=True)`
      - `"queued"`             → synth `ToolResult(content="queued for approval", is_error=True)`

    Returning `"queued"` is meant for the inbox flow (`InboxApprovalHandler`).
    The handler is expected to have already persisted a `PendingApproval` row.
    """

    async def __call__(
        self, tool: Tool, call: ToolCall, session: Session
    ) -> bool | ApprovalOutcome: ...


class AutoApprove:
    """An ApprovalHandler that approves everything. Useful in tests and `--yes` mode."""

    async def __call__(self, tool: Tool, call: ToolCall, session: Session) -> bool:
        return True


class AutoDeny:
    """An ApprovalHandler that denies everything. Useful as a safety stop."""

    async def __call__(self, tool: Tool, call: ToolCall, session: Session) -> bool:
        return False


class InboxApprovalHandler:
    """Queue tool calls for asynchronous human review.

    Writes a `PendingApproval` to the configured store and returns
    `"queued"`. The runtime then surfaces `ToolResult(is_error=True,
    content="queued for approval ...")` to the agent so the session can
    continue cleanly. When the user later resolves the approval (via
    `harness approvals grant`), the next `agent.resume()` re-dispatches
    the original tool call and overwrites the queued result in history.

    Pass the same storage that satisfies `ApprovalStore` (in practice the
    main `Storage` instance) at construction.
    """

    def __init__(self, *, approval_store: ApprovalStore) -> None:
        self.approval_store = approval_store

    async def __call__(self, tool: Tool, call: ToolCall, session: Session) -> ApprovalOutcome:
        approval = PendingApproval(
            task_id=session.task_id,
            session_id=session.id,
            tool_call_id=call.id,
            tool_name=tool.name,
            arguments=call.arguments,
        )
        await self.approval_store.create_approval(approval)
        return "queued"


__all__ = [
    "WILDCARD_PHASE",
    "ApprovalHandler",
    "ApprovalPolicy",
    "AutoApprove",
    "AutoDeny",
    "EffectScope",
    "InboxApprovalHandler",
    "Tool",
    "ToolRegistry",
    "tool_matches_phase",
]
