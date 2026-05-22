"""Tests for Guardrail — blocking and parallel guardrail checks."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.core import (
    Done,
    GuardrailTrippedEvent,
    Message,
    RunRequest,
)
from harness.core.guardrails import GuardrailMode, GuardrailResult

from .conftest import MockAdapter, MockStorage, text_turn


async def collect(it):
    out = []
    async for e in it:
        out.append(e)
    return out


class AllowGuardrail:
    name = "allow_all"
    mode: GuardrailMode = "blocking"

    async def __call__(self, messages: list[Message]) -> GuardrailResult:
        return GuardrailResult(tripped=False)


class DenyGuardrail:
    name = "deny_all"
    mode: GuardrailMode = "blocking"

    async def __call__(self, messages: list[Message]) -> GuardrailResult:
        return GuardrailResult(tripped=True, reason="blocked by deny guardrail")


class ParallelDenyGuardrail:
    name = "parallel_deny"
    mode: GuardrailMode = "parallel"

    async def __call__(self, messages: list[Message]) -> GuardrailResult:
        return GuardrailResult(tripped=True, reason="parallel block")


class ParallelAllowGuardrail:
    name = "parallel_allow"
    mode: GuardrailMode = "parallel"

    async def __call__(self, messages: list[Message]) -> GuardrailResult:
        return GuardrailResult(tripped=False)


def make_agent(adapters, guardrails=None, default_cwd="/tmp"):
    from harness.core import Agent, FailoverPolicy, ToolRegistry

    storage = MockStorage()
    registry = ToolRegistry()
    failover = FailoverPolicy(chain=list(adapters), max_attempts=1)
    agent = Agent(
        adapters=adapters,
        tools=registry,
        storage=storage,
        failover=failover,
        guardrails=guardrails,
        default_model="test-model",
        default_cwd=default_cwd,
    )
    return agent


@pytest.mark.asyncio
class TestBlockingGuardrail:
    async def test_allow_guardrail_passes_through(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("hello")])
        agent = make_agent(
            {"mock": adapter},
            guardrails=[AllowGuardrail()],
            default_cwd=str(tmp_path),
        )
        events = await collect(agent.run(RunRequest(prompt="hi")))
        done_events = [e for e in events if isinstance(e, Done)]
        assert done_events, "expected Done event"
        tripped = [e for e in events if isinstance(e, GuardrailTrippedEvent)]
        assert not tripped

    async def test_blocking_guardrail_stops_run(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("never seen")])
        agent = make_agent(
            {"mock": adapter},
            guardrails=[DenyGuardrail()],
            default_cwd=str(tmp_path),
        )
        events = await collect(agent.run(RunRequest(prompt="blocked")))
        tripped = [e for e in events if isinstance(e, GuardrailTrippedEvent)]
        assert tripped, "expected GuardrailTrippedEvent"
        assert tripped[0].guardrail_name == "deny_all"
        assert "blocked by deny guardrail" in tripped[0].reason
        # No Done event — run was aborted
        done_events = [e for e in events if isinstance(e, Done)]
        assert not done_events

    async def test_blocking_guardrail_name_and_reason(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("never")])
        agent = make_agent(
            {"mock": adapter},
            guardrails=[DenyGuardrail()],
            default_cwd=str(tmp_path),
        )
        events = await collect(agent.run(RunRequest(prompt="x")))
        tripped = next(e for e in events if isinstance(e, GuardrailTrippedEvent))
        assert tripped.guardrail_name == "deny_all"
        assert tripped.reason == "blocked by deny guardrail"


@pytest.mark.asyncio
class TestParallelGuardrail:
    async def test_parallel_allow_does_not_stop_run(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("hello")])
        agent = make_agent(
            {"mock": adapter},
            guardrails=[ParallelAllowGuardrail()],
            default_cwd=str(tmp_path),
        )
        events = await collect(agent.run(RunRequest(prompt="hi")))
        done_events = [e for e in events if isinstance(e, Done)]
        assert done_events

    async def test_parallel_deny_stops_run(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("never")])
        agent = make_agent(
            {"mock": adapter},
            guardrails=[ParallelDenyGuardrail()],
            default_cwd=str(tmp_path),
        )
        events = await collect(agent.run(RunRequest(prompt="blocked")))
        tripped = [e for e in events if isinstance(e, GuardrailTrippedEvent)]
        assert tripped
        assert tripped[0].guardrail_name == "parallel_deny"


@pytest.mark.asyncio
class TestNoGuardrails:
    async def test_no_guardrails_runs_normally(self, tmp_path: Path) -> None:
        """Sanity check: no guardrails configured → normal run, no GuardrailTrippedEvent."""
        adapter = MockAdapter("mock", scripts=[text_turn("works fine")])
        agent = make_agent({"mock": adapter}, default_cwd=str(tmp_path))
        events = await collect(agent.run(RunRequest(prompt="hi")))
        assert not any(isinstance(e, GuardrailTrippedEvent) for e in events)
        assert any(isinstance(e, Done) for e in events)
