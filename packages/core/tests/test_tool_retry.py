"""Tests for ToolRetry — tool raises to re-feed feedback to the model."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest

from harness.core import (
    RunRequest,
    ToolResultEvent,
    ToolRetry,
)
from harness.core.schemas import ToolCall, ToolResult

from .conftest import MockAdapter, MockStorage, MockTool, text_turn, tool_call_turn


async def collect(it):
    out = []
    async for e in it:
        out.append(e)
    return out


def make_agent(adapters, tools=None, default_cwd="/tmp"):
    from harness.core import Agent, FailoverPolicy, ToolRegistry

    storage = MockStorage()
    registry = ToolRegistry()
    for t in tools or []:
        registry.register(t)
    failover = FailoverPolicy(chain=list(adapters), max_attempts=1)
    agent = Agent(
        adapters=adapters,
        tools=registry,
        storage=storage,
        failover=failover,
        default_model="test-model",
        default_cwd=default_cwd,
    )
    return agent


@pytest.mark.asyncio
class TestToolRetry:
    async def test_retry_feeds_error_back_to_model(self, tmp_path: Path) -> None:
        """ToolRetry should produce an error ToolResult without raising to the runtime."""
        call_count = 0

        class RetryOnceTool:
            name = "flaky"
            description = "Flaky tool"
            parameters_schema: ClassVar[dict] = {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            }
            approval = "auto"
            calls: ClassVar[list] = []

            async def __call__(self, call: ToolCall) -> ToolResult:
                nonlocal call_count
                call_count += 1
                self.calls.append(call.arguments)
                if call_count == 1:
                    raise ToolRetry("bad input, try again with different text")
                return ToolResult(tool_call_id=call.id, name=self.name, content="success")

        tool = RetryOnceTool()
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="flaky", arguments={"text": "first"}),
                tool_call_turn(call_id="c2", name="flaky", arguments={"text": "second"}),
                text_turn("done"),
            ],
        )
        agent = make_agent({"mock": adapter}, tools=[tool], default_cwd=str(tmp_path))
        events = await collect(agent.run(RunRequest(prompt="run flaky")))

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(result_events) >= 1
        first_result = result_events[0]
        assert first_result.result.is_error is True
        assert "ToolRetry" in first_result.result.content
        assert "bad input" in first_result.result.content

    async def test_retry_exhausted_after_max(self, tmp_path: Path) -> None:
        """After max_retries for a tool name, further ToolRetry returns an exhausted error."""

        class AlwaysRetryTool:
            name = "always_retry"
            description = "Always retries"
            parameters_schema: ClassVar[dict] = {
                "type": "object",
                "properties": {"text": {"type": "string"}},
            }
            approval = "auto"

            async def __call__(self, call: ToolCall) -> ToolResult:
                raise ToolRetry("keep retrying")

        tool = AlwaysRetryTool()
        # Script: 5 tool calls then a final text answer
        scripts = [
            tool_call_turn(call_id=f"c{i}", name="always_retry", arguments={"text": "x"})
            for i in range(5)
        ]
        scripts.append(text_turn("gave up"))

        adapter = MockAdapter("mock", scripts=scripts)
        agent = make_agent({"mock": adapter}, tools=[tool], default_cwd=str(tmp_path))
        events = await collect(agent.run(RunRequest(prompt="run")))

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert result_events, "expected tool result events"
        # Eventually we should see an exhausted error
        exhausted = [e for e in result_events if "exhausted" in e.result.content]
        assert exhausted, "expected at least one 'exhausted' result after max retries"
        for e in exhausted:
            assert e.result.is_error is True

    async def test_normal_exception_is_not_retry(self, tmp_path: Path) -> None:
        """A regular RuntimeError from a tool is NOT a ToolRetry — it's an error result."""
        tool = MockTool(
            name="crasher",
            approval="auto",
            responder=lambda **_: RuntimeError("crash"),
        )
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="crasher", arguments={"text": "x"}),
                text_turn("recovered"),
            ],
        )
        agent = make_agent({"mock": adapter}, tools=[tool], default_cwd=str(tmp_path))
        events = await collect(agent.run(RunRequest(prompt="crash")))

        result_events = [e for e in events if isinstance(e, ToolResultEvent)]
        assert result_events[0].result.is_error is True
        assert "crash" in result_events[0].result.content
        # Should NOT say "ToolRetry"
        assert "ToolRetry" not in result_events[0].result.content
