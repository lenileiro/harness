"""Tests for phase-scoped tool registration + dispatch enforcement."""

from __future__ import annotations

import pytest

from harness.core import (
    Agent,
    AutoApprove,
    FailoverPolicy,
    RunRequest,
    ToolCall,
    ToolRegistry,
    tool_matches_phase,
)
from harness.core.tools import WILDCARD_PHASE

from .conftest import MockAdapter, MockStorage, MockTool, text_turn, tool_call_turn

# ---------------------------------------------------------------------------
# tool_matches_phase + ToolRegistry.for_phase
# ---------------------------------------------------------------------------


class TestToolMatchesPhase:
    def test_none_phase_matches_everything(self) -> None:
        tool = MockTool(name="t")
        # MockTool doesn't declare `phases` — falls back to ("*",).
        assert tool_matches_phase(tool, None) is True
        # An arbitrary phase also matches via the wildcard default.
        assert tool_matches_phase(tool, "research") is True

    def test_explicit_phases_filter(self) -> None:
        tool = MockTool(name="writer")
        tool.phases = ("act",)  # type: ignore[attr-defined]
        assert tool_matches_phase(tool, "act") is True
        assert tool_matches_phase(tool, "research") is False
        assert tool_matches_phase(tool, "verify") is False

    def test_wildcard_in_phases(self) -> None:
        tool = MockTool(name="anywhere")
        tool.phases = (WILDCARD_PHASE,)  # type: ignore[attr-defined]
        assert tool_matches_phase(tool, "research") is True
        assert tool_matches_phase(tool, "act") is True

    def test_multiple_explicit_phases(self) -> None:
        tool = MockTool(name="reader")
        tool.phases = ("research", "verify")  # type: ignore[attr-defined]
        assert tool_matches_phase(tool, "research") is True
        assert tool_matches_phase(tool, "verify") is True
        assert tool_matches_phase(tool, "act") is False


class TestRegistryForPhase:
    def test_none_returns_everything(self) -> None:
        r = ToolRegistry()
        r.register(MockTool(name="a"))
        r.register(MockTool(name="b"))
        assert len(r.for_phase(None)) == 2

    def test_filters_by_declared_phases(self) -> None:
        r = ToolRegistry()
        read = MockTool(name="reader")
        read.phases = ("research", "act", "verify")  # type: ignore[attr-defined]
        write = MockTool(name="writer")
        write.phases = ("act",)  # type: ignore[attr-defined]
        r.register(read)
        r.register(write)

        research = {t.name for t in r.for_phase("research")}
        assert research == {"reader"}
        act = {t.name for t in r.for_phase("act")}
        assert act == {"reader", "writer"}

    def test_openai_schemas_respects_phase(self) -> None:
        r = ToolRegistry()
        read = MockTool(name="reader")
        read.phases = ("research", "*")  # type: ignore[attr-defined]
        write = MockTool(name="writer")
        write.phases = ("act",)  # type: ignore[attr-defined]
        r.register(read)
        r.register(write)

        schemas = r.openai_schemas(phase="research")
        names = {s["function"]["name"] for s in schemas}
        assert names == {"reader"}

        schemas = r.openai_schemas(phase=None)  # legacy: everything
        names = {s["function"]["name"] for s in schemas}
        assert names == {"reader", "writer"}


# ---------------------------------------------------------------------------
# Agent: current_phase wiring
# ---------------------------------------------------------------------------


def _make_agent(
    *,
    adapter: MockAdapter,
    tools: list[MockTool],
    current_phase: str | None = None,
) -> Agent:
    registry = ToolRegistry()
    for t in tools:
        registry.register(t)
    return Agent(
        adapters={"mock": adapter},  # type: ignore[arg-type]
        tools=registry,
        storage=MockStorage(),
        failover=FailoverPolicy(chain=["mock"], max_attempts=1),
        approval_handler=AutoApprove(),
        current_phase=current_phase,
        default_model="m",
    )


@pytest.mark.asyncio
class TestAgentPhaseSends:
    async def test_only_phase_matching_tools_sent_to_adapter(self) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("ok")])
        read = MockTool(name="reader")
        read.phases = ("research", "act", "*")  # type: ignore[attr-defined]
        write = MockTool(name="writer")
        write.phases = ("act",)  # type: ignore[attr-defined]
        agent = _make_agent(adapter=adapter, tools=[read, write], current_phase="research")

        # Drain a run; we only care about the call captured by the adapter.
        async for _ in agent.run(RunRequest(prompt="hi", model="m")):
            pass

        sent_tools = adapter.calls[0]["tools"]
        assert sent_tools is not None
        sent_names = {t["function"]["name"] for t in sent_tools}
        assert sent_names == {"reader"}

    async def test_none_phase_sends_every_tool(self) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("ok")])
        read = MockTool(name="reader")
        read.phases = ("research",)  # type: ignore[attr-defined]
        write = MockTool(name="writer")
        write.phases = ("act",)  # type: ignore[attr-defined]
        agent = _make_agent(adapter=adapter, tools=[read, write], current_phase=None)

        async for _ in agent.run(RunRequest(prompt="hi", model="m")):
            pass

        sent_names = {t["function"]["name"] for t in adapter.calls[0]["tools"]}
        assert sent_names == {"reader", "writer"}


@pytest.mark.asyncio
class TestAgentPhaseDispatchEnforcement:
    async def test_refuses_out_of_phase_call(self) -> None:
        # Adapter "hallucinates" a write tool call even though we're in
        # research phase. Runtime must refuse to dispatch it.
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="writer", arguments={"text": "x"}),
                text_turn("done"),
            ],
        )
        writer = MockTool(name="writer")
        writer.phases = ("act",)  # type: ignore[attr-defined]
        writer.responder = lambda **_: "should not run"  # type: ignore[assignment]
        agent = _make_agent(adapter=adapter, tools=[writer], current_phase="research")

        events = []
        async for e in agent.run(RunRequest(prompt="x", model="m")):
            events.append(e)

        # The writer should never have been called.
        assert writer.calls == []
        # And the tool_result event should carry an is_error message.
        from harness.core import ToolResultEvent

        results = [e for e in events if isinstance(e, ToolResultEvent)]
        assert results
        assert results[0].result.is_error is True
        assert "not available in phase" in results[0].result.content
        assert "'research'" in results[0].result.content

    async def test_allows_in_phase_call(self) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="reader", arguments={"text": "ok"}),
                text_turn("done"),
            ],
        )
        reader = MockTool(name="reader", approval="auto")
        reader.phases = ("research",)  # type: ignore[attr-defined]
        agent = _make_agent(adapter=adapter, tools=[reader], current_phase="research")

        async for _ in agent.run(RunRequest(prompt="x", model="m")):
            pass

        # The reader IS called when in-phase.
        assert reader.calls == [{"text": "ok"}]


class TestBackwardCompat:
    """Tools without `phases` attribute behave like wildcard."""

    def test_tool_without_phases_attribute(self) -> None:
        # Plain object with the minimum Tool surface — no `phases`.
        from typing import ClassVar

        class Bare:
            name = "bare"
            description = ""
            parameters_schema: ClassVar[dict] = {"type": "object", "properties": {}}
            approval = "auto"

            async def __call__(self, call: ToolCall):  # pragma: no cover
                from harness.core import ToolResult

                return ToolResult(tool_call_id=call.id, name=self.name, content="")

        bare = Bare()
        # Treated as ("*",) so it matches every phase, including None.
        assert tool_matches_phase(bare, None) is True  # type: ignore[arg-type]
        assert tool_matches_phase(bare, "research") is True  # type: ignore[arg-type]
        assert tool_matches_phase(bare, "act") is True  # type: ignore[arg-type]


class TestBuiltInPhases:
    """The bundled tools declare the expected phase scoping."""

    def test_read_tools_visible_in_research(self) -> None:
        from pathlib import Path

        from harness.tools.fs import GlobTool, ListDirTool, ReadFileTool
        from harness.tools.web import FetchUrlTool

        for cls in (ReadFileTool, ListDirTool, GlobTool):
            tool = cls(cwd=Path.cwd())
            assert tool_matches_phase(tool, "research") is True
            assert tool_matches_phase(tool, "verify") is True

        fetch = FetchUrlTool()
        assert tool_matches_phase(fetch, "research") is True

    def test_write_tools_hidden_in_research(self) -> None:
        from pathlib import Path

        from harness.tools.fs import EditFileTool, WriteFileTool
        from harness.tools.shell import ShellTool

        for cls in (WriteFileTool, EditFileTool):
            tool = cls(cwd=Path.cwd())
            assert tool_matches_phase(tool, "research") is False
            assert tool_matches_phase(tool, "verify") is False
            assert tool_matches_phase(tool, "act") is True

        shell = ShellTool(cwd=Path.cwd())
        assert tool_matches_phase(shell, "research") is False
        assert tool_matches_phase(shell, "act") is True
