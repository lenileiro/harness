from contextlib import asynccontextmanager

from harness.core import Agent, FailoverPolicy, NetworkError, RunRequest, ToolRegistry

from .conftest import MockAdapter, MockStorage, MockTool, text_turn


async def test_retained_tool_scope_and_async_parent_binding():
    opened, closed, bound = [], [], []
    tool = MockTool(name="process")

    @asynccontextmanager
    async def toolset():
        opened.append(True)
        try:
            yield [tool]
        finally:
            closed.append(True)

    async def bind(session):
        bound.append(session.id)
        return [MockTool(name="delegate")]

    agent = Agent(
        adapters={"mock": MockAdapter("mock", scripts=[text_turn("one"), text_turn("two")])},
        tools=ToolRegistry(),
        storage=MockStorage(),
        failover=FailoverPolicy(chain=["mock"]),
        toolset_factory=toolset,
        session_tool_factory=bind,
    )
    async with agent:
        for sid in ["one", "two"]:
            async for _ in agent.run(RunRequest(prompt="work", session_id=sid, model="m")):
                pass
        assert opened == [True] and closed == []
        assert agent.tools.get("process") is tool
    assert closed == [True] and bound == ["one", "two"]


async def test_failover_model_mapping_survives_resume():
    primary = MockAdapter("a", error=NetworkError("offline"))
    fallback = MockAdapter("b", scripts=[text_turn("one"), text_turn("two")])
    storage = MockStorage()
    agent = Agent(
        adapters={"a": primary, "b": fallback},
        tools=ToolRegistry(),
        storage=storage,
        failover=FailoverPolicy(chain=["a", "b"], max_attempts=2, backoff_base=0),
        provider_models={"a": "model-a", "b": "model-b"},
    )
    async for _ in agent.run(RunRequest(prompt="work", session_id="s", model="explicit-a")):
        pass
    saved = await storage.get("s")
    assert saved and saved.provider == "b" and saved.model == "model-b"
    async for _ in agent.resume("s", prompt="continue"):
        pass
    assert [call["model"] for call in primary.calls] == ["explicit-a", "model-a"]
    assert [call["model"] for call in fallback.calls] == ["model-b", "model-b"]
