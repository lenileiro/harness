"""Tests for the predictor/repair/calibration pipeline wired into Agent._invoke_tool,
plus queued approval, system_prompt injection, and runtime edge cases.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.core import (
    Agent,
    ApprovalPolicy,
    AutoApprove,
    ConfigurationError,
    ConsequencePredictor,
    Done,
    Event,
    FailoverPolicy,
    InboxApprovalHandler,
    OutcomeCalibration,
    PredictionEvent,
    PredictionMismatchEvent,
    RepairOrchestrator,
    RunRequest,
    ToolResultEvent,
)
from harness.storage.memory import InMemoryStorage

from .conftest import MockAdapter, MockStorage, MockTool, text_turn, tool_call_turn


async def collect(it) -> list[Event]:
    out: list[Event] = []
    async for e in it:
        out.append(e)
    return out


def make_agent(
    *,
    adapters: dict[str, MockAdapter],
    tools: list[MockTool] | None = None,
    storage: MockStorage | None = None,
    predictor: ConsequencePredictor | None = None,
    repair: RepairOrchestrator | None = None,
    calibration: OutcomeCalibration | None = None,
    approval_policy: ApprovalPolicy | None = None,
    approval_handler=None,
    system_prompt: str | None = None,
    default_model: str = "test-model",
    default_cwd: str | None = None,
) -> Agent:
    from harness.core import ToolRegistry

    storage = storage or MockStorage()
    registry = ToolRegistry()
    for t in tools or []:
        registry.register(t)
    failover = FailoverPolicy(chain=list(adapters), max_attempts=1)
    return Agent(
        adapters=adapters,  # type: ignore[arg-type]
        tools=registry,
        storage=storage,
        failover=failover,
        approval_policy=approval_policy,
        approval_handler=approval_handler,
        default_model=default_model,
        default_cwd=default_cwd,
        predictor=predictor,
        repair=repair,
        system_prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# ConsequencePredictor pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPredictorPipeline:
    async def test_prediction_event_fires_before_tool_result(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "hi"}),
                text_turn("done"),
            ],
        )
        tool = MockTool(name="echo")
        agent = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            predictor=ConsequencePredictor(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="echo hi")))
        types = [type(e).__name__ for e in events]
        pred_idx = types.index("PredictionEvent")
        result_idx = types.index("ToolResultEvent")
        assert pred_idx < result_idx

    async def test_prediction_event_contains_tool_name(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "x"}),
                text_turn("ok"),
            ],
        )
        tool = MockTool(name="echo")
        agent = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            predictor=ConsequencePredictor(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="go")))
        pred = next(e for e in events if isinstance(e, PredictionEvent))
        assert pred.prediction.tool_name == "echo"

    async def test_no_prediction_event_without_predictor(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "x"}),
                text_turn("ok"),
            ],
        )
        tool = MockTool(name="echo")
        agent = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="go")))
        assert not any(isinstance(e, PredictionEvent) for e in events)

    async def test_mismatch_event_on_read_only_tool_error(self, tmp_path: Path) -> None:
        """read_only scope predicts 'ok'; if tool errors, that's a mismatch."""
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="boom", arguments={"text": "x"}),
                text_turn("ok"),
            ],
        )
        # effect_scope="read_only" → predictor expects "ok" status
        tool = MockTool(name="boom", responder=lambda **_: ValueError("file missing"))
        tool.effect_scope = "read_only"  # type: ignore[attr-defined]
        agent = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            predictor=ConsequencePredictor(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="go")))
        mismatch_events = [e for e in events if isinstance(e, PredictionMismatchEvent)]
        assert len(mismatch_events) == 1
        assert not mismatch_events[0].outcome.matched


# ---------------------------------------------------------------------------
# RepairOrchestrator pipeline
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRepairPipeline:
    async def test_repair_does_not_block_successful_tool(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "hi"}),
                text_turn("done"),
            ],
        )
        tool = MockTool(name="echo")
        agent = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            repair=RepairOrchestrator(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="echo hi")))
        result_event = next(e for e in events if isinstance(e, ToolResultEvent))
        assert not result_event.result.is_error

    async def test_predictor_and_repair_together(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "hi"}),
                text_turn("done"),
            ],
        )
        tool = MockTool(name="echo")
        agent = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            predictor=ConsequencePredictor(),
            repair=RepairOrchestrator(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="go")))
        assert any(isinstance(e, PredictionEvent) for e in events)
        result_event = next(e for e in events if isinstance(e, ToolResultEvent))
        assert not result_event.result.is_error


# ---------------------------------------------------------------------------
# Queued approval
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestQueuedApproval:
    async def test_inbox_handler_queues_tool(self, tmp_path: Path) -> None:
        store = InMemoryStorage()
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "x"}),
                text_turn("ok"),
            ],
        )
        tool = MockTool(name="echo", approval="prompt")
        agent = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            storage=store,  # type: ignore[arg-type]
            approval_policy=ApprovalPolicy(default="prompt"),
            approval_handler=InboxApprovalHandler(approval_store=store),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="echo")))
        result_event = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result_event.result.is_error
        assert "queued" in result_event.result.content
        assert tool.calls == []  # tool never actually ran

    async def test_inbox_stores_pending_approval(self, tmp_path: Path) -> None:
        store = InMemoryStorage()
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "x"}),
                text_turn("ok"),
            ],
        )
        tool = MockTool(name="echo", approval="prompt")
        agent = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            storage=store,  # type: ignore[arg-type]
            approval_policy=ApprovalPolicy(default="prompt"),
            approval_handler=InboxApprovalHandler(approval_store=store),
            default_cwd=str(tmp_path),
        )

        await collect(agent.run(RunRequest(prompt="echo")))
        approvals = await store.list_approvals(status="pending")
        assert len(approvals) == 1
        assert approvals[0].tool_name == "echo"


# ---------------------------------------------------------------------------
# system_prompt injection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSystemPromptInjection:
    async def test_system_prompt_appears_in_adapter_messages(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("hello")])
        agent = make_agent(
            adapters={"mock": adapter},
            system_prompt="You are a helpful assistant.",
            default_cwd=str(tmp_path),
        )

        await collect(agent.run(RunRequest(prompt="hi")))
        messages = adapter.calls[0]["messages"]
        system_messages = [m for m in messages if m.role == "system"]
        assert any("You are a helpful assistant." in (m.content or "") for m in system_messages)

    async def test_no_extra_system_message_without_system_prompt(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("hello")])
        agent = make_agent(
            adapters={"mock": adapter},
            default_cwd=str(tmp_path),
        )

        await collect(agent.run(RunRequest(prompt="hi")))
        messages = adapter.calls[0]["messages"]
        system_messages = [m for m in messages if m.role == "system"]
        assert len(system_messages) == 0


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestConfigurationErrors:
    async def test_resume_unknown_session_raises(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[])
        agent = make_agent(
            adapters={"mock": adapter},
            default_cwd=str(tmp_path),
        )
        with pytest.raises(ConfigurationError, match="unknown session"):
            async for _ in agent.resume("nonexistent_session_id"):
                pass

    async def test_no_model_raises_configuration_error(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("hi")])
        from harness.core import ToolRegistry
        storage = MockStorage()
        agent = Agent(
            adapters={"mock": adapter},  # type: ignore[arg-type]
            tools=ToolRegistry(),
            storage=storage,
            failover=FailoverPolicy(chain=["mock"], max_attempts=1),
            default_cwd=str(tmp_path),
            # No default_model and no model in request
        )
        with pytest.raises(ConfigurationError, match="no model"):
            await collect(agent.run(RunRequest(prompt="hi")))
