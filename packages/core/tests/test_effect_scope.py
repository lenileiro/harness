"""Tests for EffectScope taxonomy and ApprovalPolicy integration."""

from __future__ import annotations

from harness.core import EffectScope
from harness.core.schemas import ApprovalDecision, ToolCall, ToolResult
from harness.core.tools import ApprovalPolicy


class _ScopedTool:
    """Minimal Tool implementation with configurable scope and approval."""

    def __init__(
        self, name: str, *, effect_scope: str | None, approval: ApprovalDecision = "auto"
    ) -> None:
        self.name = name
        self.description = "test"
        self.parameters_schema: dict = {"type": "object", "properties": {}, "required": []}
        self.effect_scope = effect_scope
        self.approval: ApprovalDecision = approval

    async def __call__(self, call: ToolCall) -> ToolResult:
        return ToolResult(tool_call_id=call.id, name=self.name, content="ok")


class TestEffectScopeApprovalPolicy:
    def test_workspace_durable_auto_prompts(self) -> None:
        tool = _ScopedTool("write_file", effect_scope="workspace_durable", approval="auto")
        policy = ApprovalPolicy()
        assert policy.decide(tool) == "prompt"

    def test_external_side_effect_auto_prompts(self) -> None:
        tool = _ScopedTool("http_call", effect_scope="external_side_effect", approval="auto")
        policy = ApprovalPolicy()
        assert policy.decide(tool) == "prompt"

    def test_read_only_auto_approves(self) -> None:
        tool = _ScopedTool("read_file", effect_scope="read_only", approval="prompt")
        policy = ApprovalPolicy()
        # scope=read_only overrides the tool's own approval="prompt"
        assert policy.decide(tool) == "auto"

    def test_per_tool_override_wins_over_scope(self) -> None:
        # Even workspace_durable can be forced to auto via per_tool
        tool = _ScopedTool("write_file", effect_scope="workspace_durable")
        policy = ApprovalPolicy(per_tool={"write_file": "auto"})
        assert policy.decide(tool) == "auto"

    def test_session_override_wins_over_scope(self) -> None:
        tool = _ScopedTool("write_file", effect_scope="workspace_durable")
        policy = ApprovalPolicy()
        # Session override → "deny" wins over scope-derived "prompt"
        assert policy.decide(tool, session_overrides={"write_file": "deny"}) == "deny"

    def test_missing_scope_falls_through_to_tool_approval(self) -> None:
        tool = _ScopedTool("custom_tool", effect_scope=None, approval="prompt")
        policy = ApprovalPolicy()
        assert policy.decide(tool) == "prompt"

    def test_task_durable_uses_tool_approval(self) -> None:
        # Non-workspace_durable, non-read_only → no scope-based override → tool's own "auto"
        tool = _ScopedTool("write_session", effect_scope="task_durable", approval="auto")
        policy = ApprovalPolicy()
        assert policy.decide(tool) == "auto"

    def test_effect_scope_literal_values(self) -> None:
        from typing import get_args

        scopes = get_args(EffectScope)
        assert "read_only" in scopes
        assert "workspace_durable" in scopes
        assert "external_side_effect" in scopes
        assert len(scopes) == 7
