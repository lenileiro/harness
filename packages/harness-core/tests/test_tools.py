from __future__ import annotations

import pytest

from harness.core import (
    ApprovalPolicy,
    AutoApprove,
    AutoDeny,
    ToolCall,
    ToolRegistry,
)

from .conftest import MockTool


class TestToolRegistry:
    def test_register_and_get(self) -> None:
        r = ToolRegistry()
        t = MockTool(name="echo")
        r.register(t)
        assert r.get("echo") is t
        assert r.has("echo")
        assert r.names() == ["echo"]

    def test_duplicate_register_rejected(self) -> None:
        r = ToolRegistry()
        r.register(MockTool(name="echo"))
        with pytest.raises(ValueError, match="already registered"):
            r.register(MockTool(name="echo"))

    def test_unknown_get_raises(self) -> None:
        r = ToolRegistry()
        with pytest.raises(KeyError):
            r.get("missing")

    def test_names_sorted(self) -> None:
        r = ToolRegistry()
        r.register(MockTool(name="bravo"))
        r.register(MockTool(name="alpha"))
        r.register(MockTool(name="charlie"))
        assert r.names() == ["alpha", "bravo", "charlie"]

    def test_openai_schemas_shape(self) -> None:
        r = ToolRegistry()
        r.register(MockTool(name="echo", description="Echo input."))
        [schema] = r.openai_schemas()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "echo"
        assert schema["function"]["description"] == "Echo input."
        assert schema["function"]["parameters"]["type"] == "object"


class TestApprovalPolicyDecide:
    def test_session_overrides_win(self) -> None:
        policy = ApprovalPolicy(default="prompt", per_tool={"shell": "deny"})
        tool = MockTool(name="shell", approval="auto")
        decision = policy.decide(tool, session_overrides={"shell": "auto"})
        assert decision == "auto"

    def test_per_tool_beats_tool_default(self) -> None:
        policy = ApprovalPolicy(default="prompt", per_tool={"shell": "deny"})
        tool = MockTool(name="shell", approval="auto")
        assert policy.decide(tool) == "deny"

    def test_tool_default_beats_policy_default(self) -> None:
        policy = ApprovalPolicy(default="prompt")
        tool = MockTool(name="read", approval="auto")
        assert policy.decide(tool) == "auto"

    def test_falls_through_to_policy_default(self) -> None:
        # A tool whose own approval is falsy (empty string) lets the policy
        # default kick in. Normal Tools set a non-empty default — this exercises
        # the fallback safely.
        from typing import ClassVar

        class Bare:
            name = "bare"
            description = ""
            parameters_schema: ClassVar[dict] = {"type": "object", "properties": {}}
            approval = ""  # falsy

            async def __call__(self, **_kw: object) -> str:  # pragma: no cover - unused
                return ""

        policy = ApprovalPolicy(default="deny")
        assert policy.decide(Bare()) == "deny"  # type: ignore[arg-type]


@pytest.mark.asyncio
class TestApprovalHandlers:
    async def test_auto_approve_returns_true(self) -> None:
        handler = AutoApprove()
        tool = MockTool()
        call = ToolCall(id="c1", name="echo", arguments={})
        assert await handler(tool, call) is True

    async def test_auto_deny_returns_false(self) -> None:
        handler = AutoDeny()
        tool = MockTool()
        call = ToolCall(id="c1", name="echo", arguments={})
        assert await handler(tool, call) is False
