"""Tests for Guardrail — blocking and parallel guardrail checks."""

from __future__ import annotations

import asyncio
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


class DelayedParallelDenyGuardrail:
    name = "delayed_parallel_deny"
    mode: GuardrailMode = "parallel"

    async def __call__(self, messages: list[Message]) -> GuardrailResult:
        await asyncio.sleep(0.01)
        return GuardrailResult(tripped=True, reason="delayed parallel block")


class ClosableStreamAdapter(MockAdapter):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.closed = False

    def stream(self, **kwargs):  # type: ignore[override]
        self.calls.append(kwargs)
        return self._stream()

    async def _stream(self):
        try:
            await asyncio.sleep(0.05)
            yield Done(final_message=Message(role="assistant", content="too late"))
        finally:
            self.closed = True


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

    async def test_parallel_guardrail_closes_stream_on_trip(self, tmp_path: Path) -> None:
        adapter = ClosableStreamAdapter("mock")
        agent = make_agent(
            {"mock": adapter},
            guardrails=[DelayedParallelDenyGuardrail()],
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="blocked")))

        tripped = [e for e in events if isinstance(e, GuardrailTrippedEvent)]
        assert tripped
        assert tripped[0].guardrail_name == "delayed_parallel_deny"
        assert adapter.closed is True
        assert not any(isinstance(e, Done) for e in events)


@pytest.mark.asyncio
class TestNoGuardrails:
    async def test_no_guardrails_runs_normally(self, tmp_path: Path) -> None:
        """Sanity check: no guardrails configured → normal run, no GuardrailTrippedEvent."""
        adapter = MockAdapter("mock", scripts=[text_turn("works fine")])
        agent = make_agent({"mock": adapter}, default_cwd=str(tmp_path))
        events = await collect(agent.run(RunRequest(prompt="hi")))
        assert not any(isinstance(e, GuardrailTrippedEvent) for e in events)
        assert any(isinstance(e, Done) for e in events)


@pytest.mark.asyncio
class TestGuardrailLeak:
    async def test_parallel_guardrail_cancellation_leak(self, tmp_path: Path) -> None:
        """
        Verify that when a parallel guardrail trips, it doesn't leave un-cancelled
        tasks running in the background.
        """

        class LeakyGuardrail:
            name = "leaky_guardrail"
            mode: GuardrailMode = "parallel"

            def __init__(self):
                self.was_cancelled = False

            async def __call__(self, messages: list[Message]) -> GuardrailResult:
                try:
                    await asyncio.sleep(1.0)
                    return GuardrailResult(tripped=False)
                except asyncio.CancelledError:
                    self.was_cancelled = True
                    raise

        adapter = MockAdapter("mock", scripts=[text_turn("hello")])
        agent = make_agent(
            {"mock": adapter},
            guardrails=[LeakyGuardrail(), ParallelDenyGuardrail()],
            default_cwd=str(tmp_path),
        )

        # We expect this to return quickly because ParallelDenyGuardrail
        # should trip (it doesn't sleep) and trigger cancellation.
        events = await collect(agent.run(RunRequest(prompt="blocked")))

        tripped = [e for e in events if isinstance(e, GuardrailTrippedEvent)]
        assert len(tripped) > 0
        assert tripped[0].guardrail_name == "parallel_deny"

        # Wait a bit to see if the leaky guardrail actually gets cancelled
        await asyncio.sleep(0.1)

        # Check if the leaky task was cancelled.
        # Note: In a real system, we'd inspect the event loop or task registry,
        # but here we rely on the side effect we injected.
        # However, we need to access the specific instance of the guardrail.
        # Let's refactor the test to capture the guardrail.
        pass


@pytest.mark.asyncio
class TestGuardrailLeakCorrected:
    async def test_parallel_guardrail_cancellation_leak(self, tmp_path: Path) -> None:
        class LeakyGuardrail:
            name = "leaky_guardrail"
            mode: GuardrailMode = "parallel"

            def __init__(self):
                self.was_cancelled = False

            async def __call__(self, messages: list[Message]) -> GuardrailResult:
                try:
                    # This task will be running in parallel with the stream
                    await asyncio.sleep(2.0)
                    return GuardrailResult(tripped=False)
                except asyncio.CancelledError:
                    self.was_cancelled = True
                    raise

        leaky = LeakyGuardrail()
        adapter = MockAdapter("mock", scripts=[text_turn("hello")])
        agent = make_agent(
            {"mock": adapter},
            guardrails=[leaky, ParallelDenyGuardrail()],
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="blocked")))

        tripped = [e for e in events if isinstance(e, GuardrailTrippedEvent)]
        assert len(tripped) > 0
        assert tripped[0].guardrail_name == "parallel_deny"

        # Give the loop a chance to process the cancellation
        await asyncio.sleep(0.1)

        assert leaky.was_cancelled, "Leaky guardrail task was not cancelled!"
