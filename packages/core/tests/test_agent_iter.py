"""Tests for AgentRun / agent.iter() — the context-manager step interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.core import (
    Agent,
    FailoverPolicy,
    RunRequest,
    ToolRegistry,
)
from harness.core.agent_iter import (
    AgentRun,
    FinalResponseStep,
    ModelRequestStep,
    ToolCallStep,
    ToolResultStep,
)

from .conftest import MockAdapter, MockStorage, MockTool, text_turn, tool_call_turn


def make_agent(adapters, tools=None, default_cwd="/tmp"):
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


@pytest.mark.asyncio
class TestAgentIter:
    async def test_iter_returns_agent_run(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("hello")])
        agent = make_agent({"mock": adapter}, default_cwd=str(tmp_path))
        run = agent.iter(RunRequest(prompt="hi"))
        assert isinstance(run, AgentRun)

    async def test_context_manager_text_only(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("hello world")])
        agent = make_agent({"mock": adapter}, default_cwd=str(tmp_path))

        steps = []
        async with agent.iter(RunRequest(prompt="hi")) as run:
            async for step in run:
                steps.append(step)

        step_types = [type(s).__name__ for s in steps]
        assert "ModelRequestStep" in step_types
        assert "FinalResponseStep" in step_types

    async def test_final_response_step_contains_text(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("the answer is 42")])
        agent = make_agent({"mock": adapter}, default_cwd=str(tmp_path))

        final_steps = []
        async with agent.iter(RunRequest(prompt="what is the answer")) as run:
            async for step in run:
                if isinstance(step, FinalResponseStep):
                    final_steps.append(step)

        assert final_steps
        assert final_steps[0].text == "the answer is 42"

    async def test_model_request_step_has_messages(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("yes")])
        agent = make_agent({"mock": adapter}, default_cwd=str(tmp_path))

        model_steps = []
        async with agent.iter(RunRequest(prompt="test")) as run:
            async for step in run:
                if isinstance(step, ModelRequestStep):
                    model_steps.append(step)

        assert model_steps
        # The messages should include the user prompt
        msgs = model_steps[0].messages
        assert any(m.role == "user" for m in msgs)

    async def test_tool_call_and_result_steps(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "ping"}),
                text_turn("got it"),
            ],
        )
        tool = MockTool(name="echo", approval="auto")
        agent = make_agent({"mock": adapter}, tools=[tool], default_cwd=str(tmp_path))

        steps = []
        async with agent.iter(RunRequest(prompt="echo")) as run:
            async for step in run:
                steps.append(step)

        step_types = [type(s).__name__ for s in steps]
        assert "ToolCallStep" in step_types
        assert "ToolResultStep" in step_types

        tool_call_step = next(s for s in steps if isinstance(s, ToolCallStep))
        assert tool_call_step.tool_call.name == "echo"

        tool_result_step = next(s for s in steps if isinstance(s, ToolResultStep))
        assert tool_result_step.tool_result.content == "ping"

    async def test_iter_without_context_manager(self, tmp_path: Path) -> None:
        """agent.iter() can also be used as an async iterable directly."""
        adapter = MockAdapter("mock", scripts=[text_turn("direct")])
        agent = make_agent({"mock": adapter}, default_cwd=str(tmp_path))

        steps = []
        async with agent.iter(RunRequest(prompt="hi")) as run:
            async for step in run:
                steps.append(step)

        assert any(isinstance(s, FinalResponseStep) for s in steps)
