"""Tool Protocol, registry, and approval policy.

A Tool is any async callable with a name, description, JSON Schema, and a
default approval level. The runtime resolves approval through ApprovalPolicy
(which may override the tool's default) and delegates to an ApprovalHandler
when "prompt" is the decision.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from harness.core.schemas import ApprovalDecision, ToolCall

# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


@runtime_checkable
class Tool(Protocol):
    """Async callable the agent can invoke.

    Tools return a `str` (the content the model will see). They raise to
    signal errors — the runtime catches and wraps any exception in a
    `ToolResult` with `is_error=True`. This keeps tool implementations free
    of runtime concerns (they never construct ToolResults or see tool_call_ids).
    """

    name: str
    description: str
    parameters_schema: dict[str, Any]
    """JSON Schema for the tool's parameter object (an `object` schema)."""
    approval: ApprovalDecision
    """Default approval level. Overridable per-session or globally via ApprovalPolicy."""

    async def __call__(self, **kwargs: Any) -> str: ...


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

    def openai_schemas(self) -> list[dict[str, Any]]:
        """Render every registered tool in OpenAI's `tools` request format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters_schema,
                },
            }
            for t in self._tools.values()
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
        return tool.approval or self.default


@runtime_checkable
class ApprovalHandler(Protocol):
    """Called by the runtime whenever a tool's effective approval is `prompt`.

    Implementations: CLI shows a Rich prompt; tests use auto-approve/deny mocks.
    Return value semantics:
      - True  → execute the tool
      - False → skip the tool; runtime synthesizes a tool_result with is_error=True
        and `content` = "user denied approval"
    """

    async def __call__(self, tool: Tool, call: ToolCall) -> bool: ...


class AutoApprove:
    """An ApprovalHandler that approves everything. Useful in tests and `--yes` mode."""

    async def __call__(self, tool: Tool, call: ToolCall) -> bool:
        return True


class AutoDeny:
    """An ApprovalHandler that denies everything. Useful as a safety stop."""

    async def __call__(self, tool: Tool, call: ToolCall) -> bool:
        return False


__all__ = [
    "ApprovalHandler",
    "ApprovalPolicy",
    "AutoApprove",
    "AutoDeny",
    "Tool",
    "ToolRegistry",
]
