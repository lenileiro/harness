from contextlib import asynccontextmanager

import pytest

from harness.core import Agent, Capabilities, FailoverPolicy, RunRequest, ToolRegistry
from harness.core.errors import ConfigurationError

from .conftest import MockAdapter, MockStorage, MockTool, text_turn, tool_call_turn


async def test_managed_tools_live_for_dispatch_and_close_after_each_run():
    opened = []
    closed = []
    tool = MockTool(name="managed")

    @asynccontextmanager
    async def factory():
        opened.append(True)
        try:
            yield [tool]
        finally:
            closed.append(True)

    registry = ToolRegistry()
    adapter = MockAdapter(
        "mock",
        scripts=[
            tool_call_turn(call_id="c", name="managed", arguments={"text": "hello"}),
            text_turn("done"),
            text_turn("again"),
        ],
    )
    agent = Agent(
        adapters={"mock": adapter},
        tools=registry,
        storage=MockStorage(),
        failover=FailoverPolicy(chain=["mock"]),
        toolset_factory=factory,
    )
    async for _ in agent.run(RunRequest(prompt="work", model="m", session_id="s")):
        pass
    assert opened == closed == [True]
    assert not registry.has("managed")
    async for _ in agent.resume("s", prompt="continue"):
        pass
    assert len(opened) == len(closed) == 2


async def test_collision_closes_toolset_and_preserves_original():
    original = MockTool(name="collision")
    registry = ToolRegistry()
    registry.register(original)
    closed = []

    @asynccontextmanager
    async def factory():
        try:
            yield [MockTool(name="new"), MockTool(name="collision")]
        finally:
            closed.append(True)

    agent = Agent(
        adapters={"mock": MockAdapter("mock")},
        tools=registry,
        storage=MockStorage(),
        failover=FailoverPolicy(chain=["mock"]),
        toolset_factory=factory,
    )
    with pytest.raises(ValueError):
        async for _ in agent.run(RunRequest(prompt="work", model="m")):
            pass
    assert registry.get("collision") is original
    assert not registry.has("new") and closed


async def test_native_tool_adapter_rejected_before_managed_server_starts():
    opened = []

    @asynccontextmanager
    async def factory():
        opened.append(True)
        yield [MockTool(name="external")]

    adapter = MockAdapter("mock", capabilities=Capabilities(tool_use=True, external_tools=False))
    agent = Agent(
        adapters={"mock": adapter},
        tools=ToolRegistry(),
        storage=MockStorage(),
        failover=FailoverPolicy(chain=["mock"]),
        toolset_factory=factory,
    )
    with pytest.raises(ConfigurationError, match="external tool bridge"):
        async for _ in agent.run(RunRequest(prompt="act", model="m")):
            pass
    assert not opened and not adapter.calls
