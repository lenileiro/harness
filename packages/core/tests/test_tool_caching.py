"""Tests for tool result caching (cache = True opt-in)."""

from __future__ import annotations

from typing import ClassVar

import pytest

from harness.core import RunRequest, ToolResultEvent
from harness.core.schemas import ToolCall, ToolResult

from .conftest import MockAdapter, MockStorage, text_turn, tool_call_turn


def make_agent(adapters, tools=None, default_cwd="/tmp"):
    from harness.core import Agent, FailoverPolicy, ToolRegistry

    storage = MockStorage()
    registry = ToolRegistry()
    for t in tools or []:
        registry.register(t)
    failover = FailoverPolicy(chain=list(adapters), max_attempts=1)
    return Agent(
        adapters=adapters,
        tools=registry,
        storage=storage,
        failover=failover,
        default_model="test-model",
        default_cwd=default_cwd,
    )


async def collect(it):
    out = []
    async for e in it:
        out.append(e)
    return out


@pytest.mark.asyncio
class TestToolCaching:
    async def test_cached_tool_only_executes_once(self, tmp_path) -> None:
        """When cache=True, the second call with the same args skips the tool body."""
        call_count = 0

        class ExpensiveTool:
            name = "expensive"
            description = "Expensive computation"
            parameters_schema: ClassVar[dict] = {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            }
            approval = "auto"
            cache = True

            async def __call__(self, call: ToolCall) -> ToolResult:
                nonlocal call_count
                call_count += 1
                return ToolResult(
                    tool_call_id=call.id, name=self.name, content=f"result:{call.arguments['x']}"
                )

        tool = ExpensiveTool()
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="expensive", arguments={"x": 42}),
                tool_call_turn(call_id="c2", name="expensive", arguments={"x": 42}),
                text_turn("done"),
            ],
        )
        agent = make_agent({"mock": adapter}, tools=[tool], default_cwd=str(tmp_path))
        events = await collect(agent.run(RunRequest(prompt="run")))

        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert len(results) == 2
        # Both results return the same content
        assert results[0].result.content == "result:42"
        assert results[1].result.content == "result:42"
        # But the tool body ran only once
        assert call_count == 1

    async def test_different_args_not_cached(self, tmp_path) -> None:
        """Different arguments produce separate cache entries."""
        call_count = 0

        class CacheTool:
            name = "lookup"
            description = "Lookup by key"
            parameters_schema: ClassVar[dict] = {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            }
            approval = "auto"
            cache = True

            async def __call__(self, call: ToolCall) -> ToolResult:
                nonlocal call_count
                call_count += 1
                return ToolResult(
                    tool_call_id=call.id,
                    name=self.name,
                    content=f"value:{call.arguments['key']}",
                )

        tool = CacheTool()
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="lookup", arguments={"key": "a"}),
                tool_call_turn(call_id="c2", name="lookup", arguments={"key": "b"}),
                text_turn("done"),
            ],
        )
        agent = make_agent({"mock": adapter}, tools=[tool], default_cwd=str(tmp_path))
        await collect(agent.run(RunRequest(prompt="run")))

        # Different keys → two separate executions
        assert call_count == 2

    async def test_no_cache_attribute_does_not_cache(self, tmp_path) -> None:
        """Tools without cache=True are called every time."""
        call_count = 0

        class NormalTool:
            name = "normal"
            description = "Normal tool"
            parameters_schema: ClassVar[dict] = {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            }
            approval = "auto"
            # no cache attribute

            async def __call__(self, call: ToolCall) -> ToolResult:
                nonlocal call_count
                call_count += 1
                return ToolResult(tool_call_id=call.id, name=self.name, content="ok")

        tool = NormalTool()
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="normal", arguments={"x": 1}),
                tool_call_turn(call_id="c2", name="normal", arguments={"x": 1}),
                text_turn("done"),
            ],
        )
        agent = make_agent({"mock": adapter}, tools=[tool], default_cwd=str(tmp_path))
        await collect(agent.run(RunRequest(prompt="run")))

        # Called both times
        assert call_count == 2

    async def test_error_result_not_cached(self, tmp_path) -> None:
        """Error results are not stored in the cache."""
        call_count = 0

        class FlakyTool:
            name = "flaky"
            description = "Flaky"
            parameters_schema: ClassVar[dict] = {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
            }
            approval = "auto"
            cache = True

            async def __call__(self, call: ToolCall) -> ToolResult:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return ToolResult(
                        tool_call_id=call.id, name=self.name, content="fail", is_error=True
                    )
                return ToolResult(tool_call_id=call.id, name=self.name, content="ok")

        tool = FlakyTool()
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="flaky", arguments={"x": 1}),
                tool_call_turn(call_id="c2", name="flaky", arguments={"x": 1}),
                text_turn("done"),
            ],
        )
        agent = make_agent({"mock": adapter}, tools=[tool], default_cwd=str(tmp_path))
        events = await collect(agent.run(RunRequest(prompt="run")))

        results = [e for e in events if isinstance(e, ToolResultEvent)]
        # First call: error (not cached); second call: success
        assert results[0].result.is_error is True
        assert results[1].result.content == "ok"
        assert call_count == 2
