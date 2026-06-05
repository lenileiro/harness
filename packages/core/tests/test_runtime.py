"""End-to-end tests for the Agent ReAct loop with MockAdapter + MockTool."""

from __future__ import annotations

# pyright: reportReturnType=false, reportAttributeAccessIssue=false, reportArgumentType=false
import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from harness.core import (
    ActivityEvent,
    ActivityStore,
    Agent,
    ApprovalPolicy,
    AutoApprove,
    AutoDeny,
    ConfigurationError,
    Done,
    ErrorEvent,
    Event,
    FailoverPolicy,
    Message,
    NetworkError,
    Plan,
    PlanStep,
    RunRequest,
    StepCompleted,
    StepStarted,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
    Verification,
    VerificationResult,
    VerifyBeforeDoneVerifier,
    VerifyWorkTool,
)
from harness.core import (
    TimeoutError as HarnessTimeoutError,
)

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
    failover: FailoverPolicy | None = None,
    approval_policy: ApprovalPolicy | None = None,
    approval_handler=None,
    verifier=None,
    activity_store: ActivityStore | None = None,
    planner=None,
    default_model: str = "test-model",
    default_cwd: str | None = None,
    max_repair_attempts: int = 3,
) -> tuple[Agent, MockStorage]:
    from harness.core import ToolRegistry

    storage = storage or MockStorage()
    registry = ToolRegistry()
    for t in tools or []:
        registry.register(t)
    failover = failover or FailoverPolicy(chain=list(adapters), max_attempts=2)
    agent = Agent(
        adapters=adapters,  # type: ignore[arg-type]
        tools=registry,
        storage=storage,
        failover=failover,
        approval_policy=approval_policy,
        approval_handler=approval_handler,
        activity_store=activity_store,
        verifier=verifier,
        planner=planner,
        default_model=default_model,
        default_cwd=default_cwd,
        max_repair_attempts=max_repair_attempts,
    )
    return agent, storage


class AlwaysPassVerifier:
    async def verify(self, *, session, activity) -> VerificationResult:
        return VerificationResult(can_finish=True, reason="verified", verifier_name="test")


class InMemoryActivitySink(ActivityStore):
    def __init__(self) -> None:
        self.events: list[ActivityEvent] = []

    async def append_activity(self, event: ActivityEvent) -> None:
        self.events.append(event)

    async def list_activity(
        self,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        kinds: tuple[str, ...] | None = None,
        limit: int = 200,
    ) -> list[ActivityEvent]:
        items = list(self.events)
        if session_id is not None:
            items = [event for event in items if event.session_id == session_id]
        if kinds is not None:
            items = [event for event in items if event.kind in kinds]
        if limit <= 0:
            return []
        return items[-limit:]


class FailThenPassVerifier:
    def __init__(self) -> None:
        self.calls = 0

    async def verify(self, *, session, activity) -> VerificationResult:
        self.calls += 1
        if self.calls == 1:
            return VerificationResult(
                can_finish=False,
                reason="compile failed after initial edit",
                verifier_name="test",
            )
        return VerificationResult(can_finish=True, reason="verified", verifier_name="test")


class StaticPlanner:
    def __init__(self, *steps: str) -> None:
        self._steps = list(steps)

    async def plan(self, goal, context) -> Plan:
        return Plan(steps=[PlanStep(description=step) for step in self._steps])


class ExhaustionProbeAdapter(MockAdapter):
    def __init__(self, name: str) -> None:
        super().__init__(name, scripts=[])
        self.exhausted_after_done = False

    def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Event]:
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "kwargs": kwargs,
            }
        )
        return self._stream()

    async def _stream(self) -> AsyncIterator[Event]:
        yield TextDelta(text="hello")
        yield Done(final_message=Message(role="assistant", content="hello"))
        self.exhausted_after_done = True


class SequencedAdapter(MockAdapter):
    def __init__(self, name: str, entries: list[list[Event] | BaseException]) -> None:
        super().__init__(name, scripts=[])
        self.entries = list(entries)

    def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Event]:
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "kwargs": kwargs,
            }
        )
        return self._stream_next()

    async def _stream_next(self) -> AsyncIterator[Event]:
        if not self.entries:
            raise AssertionError("SequencedAdapter: out of entries")
        entry = self.entries.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        for event in entry:
            yield event


class ModelToolSupportAdapter(MockAdapter):
    def __init__(self, name: str, *, supports_tools: bool | None, scripts=None) -> None:
        super().__init__(name, scripts=scripts)
        self.supports_tools = supports_tools
        self.tool_support_checks: list[str] = []

    async def model_supports_tools(self, model: str) -> bool | None:
        self.tool_support_checks.append(model)
        return self.supports_tools


class HangingOnceToolSupportAdapter(MockAdapter):
    def __init__(self, name: str, *, scripts=None) -> None:
        super().__init__(name, scripts=scripts)
        self.tool_support_checks = 0

    async def model_supports_tools(self, model: str) -> bool | None:
        self.tool_support_checks += 1
        if self.tool_support_checks == 1:
            await asyncio.Event().wait()
        return True


class HangingAdapter(MockAdapter):
    def __init__(self, name: str, entries: list[list[Event] | str]) -> None:
        super().__init__(name, scripts=[])
        self.entries = list(entries)

    def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Event]:
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "kwargs": kwargs,
            }
        )
        return self._stream_next()

    async def _stream_next(self) -> AsyncIterator[Event]:
        if not self.entries:
            raise AssertionError("HangingAdapter: out of entries")
        entry = self.entries.pop(0)
        if entry == "hang":
            await asyncio.Event().wait()
            return
        if entry == "whitespace":
            while True:
                await asyncio.sleep(0.005)
                yield TextDelta(text=" ")
        if entry == "text_forever":
            while True:
                await asyncio.sleep(0.005)
                yield TextDelta(text="still working ")
        for event in entry:
            yield event


# ---------------------------------------------------------------------------
# Happy path: plain text answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHappyPath:
    async def test_text_only_turn(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("hello world")])
        agent, _storage = make_agent(adapters={"mock": adapter}, default_cwd=str(tmp_path))

        events = await collect(agent.run(RunRequest(prompt="hi there")))

        assert isinstance(events[0], StepStarted)
        text_event = next(e for e in events if isinstance(e, TextDelta))
        assert text_event.text == "hello world"
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert done.final_message.content == "hello world"
        assert isinstance(events[-1], StepCompleted)

    async def test_session_is_persisted(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("ok")])
        agent, storage = make_agent(adapters={"mock": adapter}, default_cwd=str(tmp_path))
        req = RunRequest(prompt="hi", session_id="sess_fixed")
        await collect(agent.run(req))

        stored = await storage.get("sess_fixed")
        assert stored is not None
        assert stored.status == "done"
        # User message and assistant message are both in history.
        assert len(stored.messages) == 2
        assert stored.messages[0].role == "user"
        assert stored.messages[0].content == "hi"
        assert stored.messages[1].role == "assistant"
        assert stored.messages[1].content == "ok"

    async def test_adapter_stream_is_exhausted_after_done(self, tmp_path: Path) -> None:
        adapter = ExhaustionProbeAdapter("mock")
        agent, _storage = make_agent(adapters={"mock": adapter}, default_cwd=str(tmp_path))

        events = await collect(agent.run(RunRequest(prompt="hi there")))

        assert any(isinstance(e, Done) for e in events)
        assert adapter.exhausted_after_done is True


# ---------------------------------------------------------------------------
# Tool-call loop
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestToolCallLoop:
    async def test_one_tool_then_answer(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "ping"}),
                text_turn("got: ping"),
            ],
        )
        tool = MockTool(name="echo", approval="auto")
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="echo ping")))

        # We should see the tool_call event, then tool_result, then final Done.
        kinds = [type(e).__name__ for e in events]
        assert "ToolCallEvent" in kinds
        assert "ToolResultEvent" in kinds
        result_event = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result_event.result.content == "ping"
        assert result_event.result.is_error is False

        # Tool was called once with the parsed args.
        assert tool.calls == [{"text": "ping"}]

        # Session has: user, assistant(tool_call), tool, assistant(final)
        [sess] = await storage.list()
        roles = [m.role for m in sess.messages]
        assert roles == ["user", "assistant", "tool", "assistant"]

    async def test_require_tool_use_retries_text_only_response(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                text_turn("I am ready to assist you."),
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "Tokyo weather"}),
                text_turn("Tokyo weather: rain"),
            ],
        )
        tool = MockTool(name="echo", approval="auto")
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            default_cwd=str(tmp_path),
        )

        events = await collect(
            agent.run(
                RunRequest(
                    prompt="What is the weather in Tokyo?",
                    require_tool_use=True,
                    max_steps=3,
                )
            )
        )

        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert done.final_message.content == "Tokyo weather: rain"
        assert tool.calls == [{"text": "Tokyo weather"}]
        assert adapter.calls[0]["kwargs"]["tool_choice"] == "required"
        assert adapter.calls[1]["kwargs"]["tool_choice"] == "required"
        assert adapter.calls[2]["kwargs"].get("tool_choice") is None
        [sess] = await storage.list()
        assert any(
            m.role == "user" and "No tool evidence was produced" in (m.content or "")
            for m in sess.messages
        )

    async def test_tool_model_preflight_rejects_unsupported_model(self, tmp_path: Path) -> None:
        adapter = ModelToolSupportAdapter("mock", supports_tools=False)
        tool = MockTool(name="echo", approval="auto")
        agent, _ = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            failover=FailoverPolicy(chain=["mock"], max_attempts=1),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="Use a tool.")))

        error_event = next(e for e in events if isinstance(e, ErrorEvent))
        assert error_event.kind == "model_unavailable"
        assert "does not support tool use" in error_event.error
        assert adapter.tool_support_checks == ["test-model"]
        assert adapter.calls == []

    async def test_tool_model_preflight_can_fail_over_before_streaming(
        self, tmp_path: Path
    ) -> None:
        primary = ModelToolSupportAdapter("primary", supports_tools=False)
        secondary = MockAdapter("secondary", scripts=[text_turn("from backup")])
        tool = MockTool(name="echo", approval="auto")
        agent, _ = make_agent(
            adapters={"primary": primary, "secondary": secondary},
            tools=[tool],
            failover=FailoverPolicy(
                chain=["primary", "secondary"],
                max_attempts=2,
                backoff_base=0.0,
                backoff_jitter=0.0,
            ),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="Use a tool if needed.")))

        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert done.final_message.content == "from backup"
        assert primary.tool_support_checks == ["test-model"]
        assert primary.calls == []
        assert len(secondary.calls) == 1

    async def test_tool_exception_becomes_error_result(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="boom", arguments={"text": "x"}),
                text_turn("recovered"),
            ],
        )
        tool = MockTool(
            name="boom",
            approval="auto",
            responder=lambda **_: RuntimeError("kaboom"),
        )
        agent, _ = make_agent(adapters={"mock": adapter}, tools=[tool], default_cwd=str(tmp_path))

        events = await collect(agent.run(RunRequest(prompt="run boom")))
        result_event = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result_event.result.is_error is True
        assert "kaboom" in result_event.result.content

    async def test_empty_final_after_tool_error_is_rejected(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="boom", arguments={"text": "x"}),
                text_turn(""),
                tool_call_turn(call_id="c2", name="echo", arguments={"text": "recovered"}),
                text_turn("recovered"),
            ],
        )
        boom_tool = MockTool(
            name="boom",
            approval="auto",
            responder=lambda **_: RuntimeError("kaboom"),
        )
        echo_tool = MockTool(name="echo", approval="auto")
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[boom_tool, echo_tool],
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="recover from tool error", max_steps=4)))

        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert done.final_message.content == "recovered"
        assert boom_tool.calls == [{"text": "x"}]
        assert echo_tool.calls == [{"text": "recovered"}]
        [sess] = await storage.list()
        assert any(
            m.role == "user" and "previous tool call failed" in (m.content or "")
            for m in sess.messages
        )

    async def test_post_tool_timeout_gets_bounded_retry(self, tmp_path: Path) -> None:
        adapter = SequencedAdapter(
            "mock",
            [
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "context"}),
                HarnessTimeoutError("provider timed out"),
                text_turn("continued after timeout"),
            ],
        )
        tool = MockTool(name="echo", approval="auto")
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="use context", max_steps=4)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert done.final_message.content == "continued after timeout"
        assert len(adapter.calls) == 3
        [sess] = await storage.list()
        assert any(
            m.role == "user" and "previous model turn timed out" in (m.content or "")
            for m in sess.messages
        )

    async def test_post_tool_timeout_can_fail_over_from_durable_tool_evidence(
        self, tmp_path: Path
    ) -> None:
        primary = SequencedAdapter(
            "primary",
            [
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "context"}),
                HarnessTimeoutError("provider timed out"),
            ],
        )
        secondary = MockAdapter("secondary", scripts=[text_turn("continued from backup")])
        tool = MockTool(name="echo", approval="auto")
        agent, storage = make_agent(
            adapters={"primary": primary, "secondary": secondary},
            tools=[tool],
            failover=FailoverPolicy(
                chain=["primary", "secondary"],
                max_attempts=2,
                backoff_base=0.0,
                backoff_jitter=0.0,
            ),
            default_cwd=str(tmp_path),
            max_repair_attempts=0,
        )

        events = await collect(agent.run(RunRequest(prompt="use context", max_steps=4)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert done.final_message.content == "continued from backup"
        assert len(primary.calls) == 2
        assert len(secondary.calls) == 1
        assert tool.calls == [{"text": "context"}]
        [sess] = await storage.list()
        assert any(m.role == "tool" and m.content == "context" for m in sess.messages)

    async def test_initial_idle_stream_timeout_gets_bounded_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARNESS_MODEL_STREAM_IDLE_TIMEOUT", "0.01")
        adapter = HangingAdapter(
            "mock",
            [
                "hang",
                text_turn("started after retry"),
            ],
        )
        agent, storage = make_agent(
            adapters={"mock": adapter},
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="start work", max_steps=3)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert done.final_message.content == "started after retry"
        assert len(adapter.calls) == 2
        [sess] = await storage.list()
        assert any(
            m.role == "user" and "produced no output before any tool evidence" in (m.content or "")
            for m in sess.messages
        )

    async def test_initial_idle_stream_timeout_uses_repair_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARNESS_MODEL_STREAM_IDLE_TIMEOUT", "0.01")
        adapter = HangingAdapter(
            "mock",
            [
                "hang",
                "hang",
                "hang",
                text_turn("started after repeated idle timeouts"),
            ],
        )
        agent, _storage = make_agent(
            adapters={"mock": adapter},
            default_cwd=str(tmp_path),
            max_repair_attempts=3,
        )

        events = await collect(agent.run(RunRequest(prompt="start work", max_steps=5)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert done.final_message.content == "started after repeated idle timeouts"
        assert len(adapter.calls) == 4

    async def test_empty_final_after_read_only_tool_evidence_gets_repair_turn(
        self, tmp_path: Path
    ) -> None:
        adapter = SequencedAdapter(
            "mock",
            [
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "context"}),
                [Done(final_message=Message(role="assistant", content=None))],
                text_turn("continued after empty final"),
            ],
        )
        tool = MockTool(name="echo", approval="auto")
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="use context", max_steps=5)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert done.final_message.content == "continued after empty final"
        assert len(adapter.calls) == 3
        [sess] = await storage.list()
        assert any(
            m.role == "user" and "empty final response after tool evidence" in (m.content or "")
            for m in sess.messages
        )

    async def test_tool_support_preflight_timeout_gets_bounded_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARNESS_MODEL_TURN_TIMEOUT", "0.01")
        adapter = HangingOnceToolSupportAdapter(
            "mock",
            scripts=[text_turn("continued after preflight timeout")],
        )
        tool = MockTool(name="echo", approval="auto")
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="start work", max_steps=3)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert done.final_message.content == "continued after preflight timeout"
        assert adapter.tool_support_checks == 2
        [sess] = await storage.list()
        assert any(
            m.role == "user" and "produced no output before any tool evidence" in (m.content or "")
            for m in sess.messages
        )

    async def test_post_tool_idle_stream_timeout_gets_bounded_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARNESS_MODEL_STREAM_IDLE_TIMEOUT", "0.01")
        adapter = HangingAdapter(
            "mock",
            [
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "context"}),
                "hang",
                text_turn("continued after idle timeout"),
            ],
        )
        tool = MockTool(name="echo", approval="auto")
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="use context", max_steps=4)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert done.final_message.content == "continued after idle timeout"
        assert len(adapter.calls) == 3
        [sess] = await storage.list()
        assert any(
            m.role == "user" and "previous model turn timed out" in (m.content or "")
            for m in sess.messages
        )

    async def test_post_tool_whitespace_turn_timeout_gets_bounded_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARNESS_MODEL_STREAM_IDLE_TIMEOUT", "1")
        monkeypatch.setenv("HARNESS_MODEL_TURN_TIMEOUT", "0.02")
        adapter = HangingAdapter(
            "mock",
            [
                tool_call_turn(call_id="c1", name="echo", arguments={"text": "context"}),
                "whitespace",
                text_turn("continued after whitespace timeout"),
            ],
        )
        tool = MockTool(name="echo", approval="auto")
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="use context", max_steps=4)))

        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert done.final_message.content == "continued after whitespace timeout"
        assert len(adapter.calls) == 3
        [sess] = await storage.list()
        assert any(
            m.role == "user" and "previous model turn timed out" in (m.content or "")
            for m in sess.messages
        )

    async def test_mutating_tool_timeout_hands_current_state_to_verifier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARNESS_MODEL_STREAM_IDLE_TIMEOUT", "0.01")
        adapter = HangingAdapter(
            "mock",
            [
                tool_call_turn(
                    call_id="c1",
                    name="edit_file",
                    arguments={"path": "src/app.py", "old": "x", "new": "y"},
                ),
                "hang",
            ],
        )
        tool = MockTool(name="edit_file", approval="auto")
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            verifier=AlwaysPassVerifier(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="edit then verify", max_steps=4)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert "Handing the current state to verification" in (done.final_message.content or "")
        assert len(adapter.calls) == 2
        [sess] = await storage.list()
        assert any(
            m.role == "assistant" and "model timed out after changing" in (m.content or "")
            for m in sess.messages
        )

    async def test_mutating_tool_partial_text_timeout_hands_current_state_to_verifier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARNESS_MODEL_STREAM_IDLE_TIMEOUT", "1")
        monkeypatch.setenv("HARNESS_MODEL_TURN_TIMEOUT", "0.02")
        adapter = HangingAdapter(
            "mock",
            [
                tool_call_turn(
                    call_id="c1",
                    name="edit_file",
                    arguments={"path": "src/app.py", "old": "x", "new": "y"},
                ),
                "text_forever",
            ],
        )
        tool = MockTool(name="edit_file", approval="auto")
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            verifier=AlwaysPassVerifier(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="edit then verify", max_steps=4)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert "Handing the current state to verification" in (done.final_message.content or "")
        assert len(adapter.calls) == 2
        [sess] = await storage.list()
        assert any(
            m.role == "assistant" and "model timed out after changing" in (m.content or "")
            for m in sess.messages
        )

    async def test_mutating_tool_timeout_across_plan_steps_hands_current_state_to_verifier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARNESS_MODEL_STREAM_IDLE_TIMEOUT", "0.01")
        adapter = HangingAdapter(
            "mock",
            [
                tool_call_turn(
                    call_id="c1",
                    name="write_file",
                    arguments={"path": "src/app.py", "content": "print('changed')"},
                ),
                text_turn("first planner step done"),
                "hang",
                text_turn("third planner step should not run"),
            ],
        )
        tool = MockTool(name="write_file", approval="auto")
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            verifier=AlwaysPassVerifier(),
            planner=StaticPlanner("change workspace", "continue work", "must not start"),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="edit then continue", max_steps=4)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = [e for e in events if isinstance(e, Done)][-1]
        assert done.final_message is not None
        assert "Handing the current state to verification" in (done.final_message.content or "")
        assert done.structured_result == {
            "harness_runtime": {
                "action": "verify_current_state",
                "reason": "model_timeout",
            }
        }
        assert len(adapter.calls) == 3
        assert [e.step for e in events if isinstance(e, StepStarted)] == [0, 1]
        assert [e.step for e in events if isinstance(e, StepCompleted)] == [0]
        assert any(isinstance(e, Verification) and e.result.can_finish for e in events)
        [sess] = await storage.list()
        assert any(
            m.role == "assistant" and "model timed out after changing" in (m.content or "")
            for m in sess.messages
        )

    async def test_mutating_tool_max_steps_hands_current_state_to_verifier(
        self, tmp_path: Path
    ) -> None:
        adapter = HangingAdapter(
            "mock",
            [
                tool_call_turn(
                    call_id="c1",
                    name="edit_file",
                    arguments={"path": "src/app.py", "old": "x", "new": "y"},
                ),
            ],
        )
        tool = MockTool(name="edit_file", approval="auto")
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            verifier=AlwaysPassVerifier(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="edit then verify", max_steps=1)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert "reached max_steps after changing" in (done.final_message.content or "")
        assert len(adapter.calls) == 1
        [sess] = await storage.list()
        assert any(
            m.role == "assistant" and "reached max_steps after changing" in (m.content or "")
            for m in sess.messages
        )

    async def test_mutating_shell_max_steps_hands_current_state_to_verifier(
        self, tmp_path: Path
    ) -> None:
        adapter = HangingAdapter(
            "mock",
            [
                tool_call_turn(
                    call_id="c1",
                    name="shell",
                    arguments={"command": "touch generated.txt"},
                ),
            ],
        )
        tool = MockTool(name="shell", approval="auto", responder=lambda **_: "ok")
        agent, _storage = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            verifier=AlwaysPassVerifier(),
            approval_handler=AutoApprove(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="generate file", max_steps=1)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert "reached max_steps after changing" in (done.final_message.content or "")

    async def test_read_only_shell_max_steps_still_fails_without_final_answer(
        self, tmp_path: Path
    ) -> None:
        adapter = HangingAdapter(
            "mock",
            [
                tool_call_turn(
                    call_id="c1",
                    name="shell",
                    arguments={"command": "ls"},
                ),
            ],
        )
        tool = MockTool(name="shell", approval="auto", responder=lambda **_: "ok")
        agent, _storage = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            verifier=AlwaysPassVerifier(),
            approval_handler=AutoApprove(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="inspect files", max_steps=1)))

        error = next(e for e in events if isinstance(e, ErrorEvent))
        assert error.kind == "internal"
        assert "exceeded max_steps" in error.error

    async def test_custom_workspace_tool_max_steps_hands_current_state_to_verifier(
        self, tmp_path: Path
    ) -> None:
        adapter = HangingAdapter(
            "mock",
            [
                tool_call_turn(
                    call_id="c1",
                    name="generate_file",
                    arguments={"text": "payload"},
                ),
            ],
        )
        tool = MockTool(name="generate_file", approval="auto", responder=lambda **_: "ok")
        tool.effect_scope = "workspace_durable"
        agent, _storage = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            verifier=AlwaysPassVerifier(),
            approval_handler=AutoApprove(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="generate file", max_steps=1)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert "reached max_steps after changing" in (done.final_message.content or "")

    async def test_custom_read_only_tool_max_steps_still_fails_without_final_answer(
        self, tmp_path: Path
    ) -> None:
        adapter = HangingAdapter(
            "mock",
            [
                tool_call_turn(
                    call_id="c1",
                    name="inspect_metadata",
                    arguments={"text": "payload"},
                ),
            ],
        )
        tool = MockTool(name="inspect_metadata", approval="auto", responder=lambda **_: "ok")
        tool.effect_scope = "read_only"
        agent, _storage = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            verifier=AlwaysPassVerifier(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="inspect metadata", max_steps=1)))

        error = next(e for e in events if isinstance(e, ErrorEvent))
        assert error.kind == "internal"
        assert "exceeded max_steps" in error.error

    async def test_configured_verify_work_auto_runs_before_structural_verifier(
        self, tmp_path: Path
    ) -> None:
        sink = InMemoryActivitySink()
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(
                    call_id="c1",
                    name="edit_file",
                    arguments={"path": "src/app.py", "old": "x", "new": "y"},
                ),
                text_turn("done"),
            ],
        )
        edit_tool = MockTool(name="edit_file", approval="auto")
        verify_tool = VerifyWorkTool(cwd=tmp_path, default_command="printf ok")
        agent, _storage = make_agent(
            adapters={"mock": adapter},
            tools=[edit_tool, verify_tool],
            verifier=VerifyBeforeDoneVerifier(default_verify_command_available=True),
            activity_store=sink,
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="edit then finish", max_steps=4)))

        verify_result = next(event.result for event in events if isinstance(event, Verification))
        assert verify_result.can_finish is True
        auto_verify_results = [
            event.result
            for event in events
            if isinstance(event, ToolResultEvent) and event.result.name == "verify_work"
        ]
        assert auto_verify_results
        assert auto_verify_results[0].is_error is False
        assert "PASSED" in auto_verify_results[0].content
        completed_verify_events = [
            event
            for event in sink.events
            if event.kind == "tool_call.completed" and event.data.get("name") == "verify_work"
        ]
        assert completed_verify_events
        assert completed_verify_events[0].data.get("arguments") == {}

    async def test_configured_verify_work_auto_runs_without_activity_store(
        self, tmp_path: Path
    ) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(
                    call_id="c1",
                    name="edit_file",
                    arguments={"path": "src/app.py", "old": "x", "new": "y"},
                ),
                text_turn("done"),
            ],
        )
        edit_tool = MockTool(name="edit_file", approval="auto")
        verify_tool = VerifyWorkTool(cwd=tmp_path, default_command="printf ok")
        agent, _storage = make_agent(
            adapters={"mock": adapter},
            tools=[edit_tool, verify_tool],
            verifier=VerifyBeforeDoneVerifier(default_verify_command_available=True),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="edit then finish", max_steps=4)))

        verify_result = next(event.result for event in events if isinstance(event, Verification))
        assert verify_result.can_finish is True
        auto_verify_results = [
            event.result
            for event in events
            if isinstance(event, ToolResultEvent) and event.result.name == "verify_work"
        ]
        assert auto_verify_results
        assert auto_verify_results[0].is_error is False
        assert "PASSED" in auto_verify_results[0].content

    async def test_structural_verifier_sees_mutation_without_activity_store(
        self, tmp_path: Path
    ) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(
                    call_id="c1",
                    name="edit_file",
                    arguments={"path": "src/app.py", "old": "x", "new": "y"},
                ),
                text_turn("done"),
            ],
        )
        edit_tool = MockTool(name="edit_file", approval="auto")
        agent, _storage = make_agent(
            adapters={"mock": adapter},
            tools=[edit_tool],
            verifier=VerifyBeforeDoneVerifier(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="edit then finish", max_steps=4)))

        verify_result = next(event.result for event in events if isinstance(event, Verification))
        assert verify_result.can_finish is False
        assert "never ran verify_work" in verify_result.reason

    async def test_adapter_native_tool_results_are_persisted_for_verification(
        self, tmp_path: Path
    ) -> None:
        sink = InMemoryActivitySink()
        call = ToolCall(
            id="native-file-change",
            name="apply_diff",
            arguments={"path": "answer.txt", "paths": ["answer.txt"]},
        )
        result = ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content="Codex file change completed:\n- add: answer.txt",
            metadata={
                "backend": "codex",
                "workspace_changed": True,
                "paths": ["answer.txt"],
                "path": "answer.txt",
            },
        )

        class CapturingVerifier:
            name = "capture"

            def __init__(self) -> None:
                self.activity: list[ActivityEvent] = []

            async def verify(self, *, session, activity) -> VerificationResult:
                del session
                self.activity = activity
                return VerificationResult(can_finish=True, reason="verified", verifier_name="test")

        verifier = CapturingVerifier()
        adapter = MockAdapter(
            "mock",
            scripts=[
                [
                    ToolCallEvent(call=call),
                    ToolResultEvent(result=result),
                    Done(final_message=Message(role="assistant", content="done")),
                ],
            ],
        )
        agent, _storage = make_agent(
            adapters={"mock": adapter},
            verifier=verifier,
            activity_store=sink,
            default_cwd=str(tmp_path),
            max_repair_attempts=0,
        )

        events = await collect(agent.run(RunRequest(prompt="write file", max_steps=2)))

        assert any(isinstance(event, Verification) for event in events)
        completed = [
            event
            for event in verifier.activity
            if event.kind == "tool_call.completed" and event.data.get("name") == "apply_diff"
        ]
        assert len(completed) == 1
        assert completed[0].data["arguments"]["path"] == "answer.txt"
        assert completed[0].data["metadata"]["workspace_changed"] is True
        dispatched = [
            event
            for event in sink.events
            if event.kind == "tool_call.dispatched"
            and event.data.get("tool_call_id") == "native-file-change"
        ]
        assert len(dispatched) == 1
        assert dispatched[0].data["source"] == "adapter"

    async def test_adapter_native_tool_results_satisfy_required_tool_use(
        self, tmp_path: Path
    ) -> None:
        call = ToolCall(id="native-shell", name="shell", arguments={"command": "printf ok"})
        result = ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content="ok",
            metadata={"backend": "codex", "command": "printf ok"},
        )
        adapter = MockAdapter(
            "mock",
            scripts=[
                [
                    ToolCallEvent(call=call),
                    ToolResultEvent(result=result),
                    Done(final_message=Message(role="assistant", content="verified")),
                ],
            ],
        )
        shell_tool = MockTool(name="shell", approval="auto", responder=lambda **_: "ok")
        agent, _storage = make_agent(
            adapters={"mock": adapter},
            tools=[shell_tool],
            default_cwd=str(tmp_path),
            max_repair_attempts=0,
        )

        events = await collect(
            agent.run(RunRequest(prompt="do real work", max_steps=1, require_tool_use=True))
        )

        assert not any(isinstance(event, ErrorEvent) for event in events)
        done = next(event for event in events if isinstance(event, Done))
        assert done.final_message is not None
        assert done.final_message.content == "verified"

    async def test_adapter_partial_tool_results_are_not_completed_activity(
        self, tmp_path: Path
    ) -> None:
        sink = InMemoryActivitySink()
        call = ToolCall(id="native-shell", name="shell", arguments={"command": "touch a.txt"})
        partial = ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content="running",
            metadata={"backend": "codex", "partial": True, "command": "touch a.txt"},
        )
        adapter = MockAdapter(
            "mock",
            scripts=[
                [
                    ToolCallEvent(call=call),
                    ToolResultEvent(result=partial),
                    Done(final_message=Message(role="assistant", content="still running")),
                ],
            ],
        )
        agent, _storage = make_agent(
            adapters={"mock": adapter},
            verifier=AlwaysPassVerifier(),
            activity_store=sink,
            default_cwd=str(tmp_path),
            max_repair_attempts=0,
        )

        events = await collect(agent.run(RunRequest(prompt="run shell", max_steps=2)))

        assert any(isinstance(event, Verification) for event in events)
        assert not [
            event
            for event in sink.events
            if event.kind == "tool_call.completed" and event.data.get("name") == "shell"
        ]

    async def test_plain_adapter_tool_results_are_not_persisted_as_native_activity(
        self, tmp_path: Path
    ) -> None:
        sink = InMemoryActivitySink()
        call = ToolCall(id="fake-edit", name="edit_file", arguments={"path": "src/app.py"})
        result = ToolResult(
            tool_call_id=call.id,
            name=call.name,
            content="edited",
        )
        adapter = MockAdapter(
            "mock",
            scripts=[
                [
                    ToolCallEvent(call=call),
                    ToolResultEvent(result=result),
                    Done(final_message=Message(role="assistant", content="done")),
                ],
            ],
        )
        agent, _storage = make_agent(
            adapters={"mock": adapter},
            verifier=AlwaysPassVerifier(),
            activity_store=sink,
            default_cwd=str(tmp_path),
            max_repair_attempts=0,
        )

        events = await collect(agent.run(RunRequest(prompt="edit", max_steps=2)))

        assert any(isinstance(event, Verification) for event in events)
        assert not [
            event
            for event in sink.events
            if event.kind == "tool_call.completed" and event.data.get("name") == "edit_file"
        ]

    async def test_verification_repair_directive_keeps_agent_autonomous(
        self, tmp_path: Path
    ) -> None:
        adapter = SequencedAdapter(
            "mock",
            [
                text_turn("ready for verification"),
                text_turn("continued after repair directive"),
            ],
        )
        verifier = FailThenPassVerifier()
        agent, storage = make_agent(
            adapters={"mock": adapter},
            verifier=verifier,
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="do verified work", max_steps=4)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = [e for e in events if isinstance(e, Done)][-1]
        assert done.final_message is not None
        assert done.final_message.content == "continued after repair directive"
        assert verifier.calls == 2
        [sess] = await storage.list()
        repair_messages = [
            m.content or ""
            for m in sess.messages
            if m.role == "user" and "Verification failed" in (m.content or "")
        ]
        assert repair_messages
        assert "compile failed after initial edit" in repair_messages[-1]
        assert "Continue autonomously from this evidence" in repair_messages[-1]
        assert "Do not ask the user to choose" in repair_messages[-1]
        assert "tool or dependency appears missing" in repair_messages[-1]
        assert "web_search" in repair_messages[-1]

    async def test_repair_partial_text_timeout_gets_bounded_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARNESS_MODEL_STREAM_IDLE_TIMEOUT", "1")
        monkeypatch.setenv("HARNESS_MODEL_TURN_TIMEOUT", "0.02")
        adapter = HangingAdapter(
            "mock",
            [
                tool_call_turn(
                    call_id="c1",
                    name="edit_file",
                    arguments={"path": "src/app.py", "old": "x", "new": "y"},
                ),
                text_turn("ready for verification"),
                tool_call_turn(
                    call_id="c2",
                    name="read_file",
                    arguments={"text": "src/app.py"},
                ),
                "text_forever",
                text_turn("continued after repair timeout"),
            ],
        )
        edit_tool = MockTool(name="edit_file", approval="auto")
        read_tool = MockTool(name="read_file", approval="auto")
        verifier = FailThenPassVerifier()
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[edit_tool, read_tool],
            verifier=verifier,
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="edit then repair", max_steps=6)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = [e for e in events if isinstance(e, Done)][-1]
        assert done.final_message is not None
        assert done.final_message.content == "continued after repair timeout"
        assert len(adapter.calls) == 5
        assert verifier.calls == 2
        [sess] = await storage.list()
        assert sess.status == "done"
        assert any(
            m.role == "user" and "previous model turn timed out" in (m.content or "")
            for m in sess.messages
        )

    async def test_repair_no_event_timeout_gets_bounded_retry(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARNESS_MODEL_STREAM_IDLE_TIMEOUT", "0.02")
        monkeypatch.setenv("HARNESS_MODEL_TURN_TIMEOUT", "1")
        adapter = HangingAdapter(
            "mock",
            [
                tool_call_turn(
                    call_id="c1",
                    name="edit_file",
                    arguments={"path": "src/app.py", "old": "x", "new": "y"},
                ),
                text_turn("ready for verification"),
                "hang",
                text_turn("continued after no-event repair timeout"),
            ],
        )
        edit_tool = MockTool(name="edit_file", approval="auto")
        verifier = FailThenPassVerifier()
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[edit_tool],
            verifier=verifier,
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="edit then repair", max_steps=6)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = [e for e in events if isinstance(e, Done)][-1]
        assert done.final_message is not None
        assert done.final_message.content == "continued after no-event repair timeout"
        assert len(adapter.calls) == 4
        assert verifier.calls == 2
        [sess] = await storage.list()
        assert sess.status == "done"
        assert any(
            m.role == "user" and "previous model turn timed out" in (m.content or "")
            for m in sess.messages
        )

    async def test_verify_work_then_final_timeout_hands_current_state_to_verifier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARNESS_MODEL_STREAM_IDLE_TIMEOUT", "1")
        monkeypatch.setenv("HARNESS_MODEL_TURN_TIMEOUT", "0.02")
        adapter = HangingAdapter(
            "mock",
            [
                tool_call_turn(call_id="c1", name="verify_work", arguments={}),
                "text_forever",
            ],
        )
        verify_tool = MockTool(name="verify_work", approval="auto")
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[verify_tool],
            verifier=AlwaysPassVerifier(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="verify then finish", max_steps=4)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert "timed out after passing verify_work" in (done.final_message.content or "")
        assert len(adapter.calls) == 2
        [sess] = await storage.list()
        assert any(
            m.role == "assistant" and "timed out after passing verify_work" in (m.content or "")
            for m in sess.messages
        )

    async def test_verify_work_then_empty_final_hands_current_state_to_verifier(
        self, tmp_path: Path
    ) -> None:
        adapter = SequencedAdapter(
            "mock",
            [
                tool_call_turn(call_id="c1", name="verify_work", arguments={}),
                [Done(final_message=Message(role="assistant", content=None))],
                text_turn("should not be needed"),
            ],
        )
        verify_tool = MockTool(name="verify_work", approval="auto")
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[verify_tool],
            verifier=AlwaysPassVerifier(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="verify then finish", max_steps=4)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert "empty final response after passing verify_work" in (
            done.final_message.content or ""
        )
        assert done.structured_result == {
            "harness_runtime": {
                "action": "verify_current_state",
                "reason": "empty_final",
            }
        }
        assert len(adapter.calls) == 2
        assert any(isinstance(e, Verification) and e.result.can_finish for e in events)
        [sess] = await storage.list()
        assert any(
            m.role == "assistant"
            and "empty final response after passing verify_work" in (m.content or "")
            for m in sess.messages
        )

    async def test_mutating_tool_then_empty_final_hands_current_state_to_verifier(
        self, tmp_path: Path
    ) -> None:
        adapter = SequencedAdapter(
            "mock",
            [
                tool_call_turn(
                    call_id="c1",
                    name="write_file",
                    arguments={"path": "src/app.py", "content": "print('changed')"},
                ),
                [Done(final_message=Message(role="assistant", content=None))],
                text_turn("should not be needed"),
            ],
        )
        tool = MockTool(name="write_file", approval="auto")
        agent, storage = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            verifier=AlwaysPassVerifier(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="edit then finish", max_steps=4)))

        assert not any(isinstance(e, ErrorEvent) for e in events)
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert "empty final response after changing the workspace" in (
            done.final_message.content or ""
        )
        assert done.structured_result == {
            "harness_runtime": {
                "action": "verify_current_state",
                "reason": "empty_final",
            }
        }
        assert len(adapter.calls) == 2
        assert any(isinstance(e, Verification) and e.result.can_finish for e in events)
        [sess] = await storage.list()
        assert any(
            m.role == "assistant" and "empty final response after changing" in (m.content or "")
            for m in sess.messages
        )


# ---------------------------------------------------------------------------
# Approval policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestApproval:
    async def test_policy_deny_short_circuits(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="shell", arguments={"text": "rm -rf"}),
                text_turn("done"),
            ],
        )
        tool = MockTool(name="shell", approval="auto")
        policy = ApprovalPolicy(per_tool={"shell": "deny"})
        agent, _ = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            approval_policy=policy,
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="run shell")))

        result_event = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result_event.result.is_error is True
        assert "denied by policy" in result_event.result.content
        # The tool should NOT have been called.
        assert tool.calls == []

    async def test_prompt_handler_denial(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="web", arguments={"text": "x"}),
                text_turn("ok"),
            ],
        )
        tool = MockTool(name="web", approval="prompt")
        agent, _ = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            approval_handler=AutoDeny(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="fetch")))
        result_event = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result_event.result.is_error is True
        assert "denied approval" in result_event.result.content
        assert tool.calls == []

    async def test_prompt_handler_approval(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="web", arguments={"text": "x"}),
                text_turn("ok"),
            ],
        )
        tool = MockTool(name="web", approval="prompt")
        agent, _ = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            approval_handler=AutoApprove(),
            default_cwd=str(tmp_path),
        )

        events = await collect(agent.run(RunRequest(prompt="fetch")))
        result_event = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result_event.result.is_error is False
        assert tool.calls == [{"text": "x"}]

    async def test_prompt_without_handler_errors(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="web", arguments={"text": "x"}),
                text_turn("ok"),
            ],
        )
        tool = MockTool(name="web", approval="prompt")
        agent, _ = make_agent(
            adapters={"mock": adapter},
            tools=[tool],
            default_cwd=str(tmp_path),
        )
        events = await collect(agent.run(RunRequest(prompt="fetch")))
        result_event = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result_event.result.is_error is True
        assert "no handler" in result_event.result.content


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestUnknownTool:
    async def test_unknown_tool_becomes_error_result(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="ghost", arguments={"text": "x"}),
                text_turn("ok"),
            ],
        )
        agent, _ = make_agent(adapters={"mock": adapter}, default_cwd=str(tmp_path))
        events = await collect(agent.run(RunRequest(prompt="x")))
        result_event = next(e for e in events if isinstance(e, ToolResultEvent))
        assert result_event.result.is_error is True
        assert "unknown tool" in result_event.result.content


# ---------------------------------------------------------------------------
# Failover
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFailover:
    async def test_retryable_error_falls_through_to_next(self, tmp_path: Path) -> None:
        primary = MockAdapter("primary", error=NetworkError("connection refused"))
        secondary = MockAdapter("secondary", scripts=[text_turn("from backup")])
        agent, _ = make_agent(
            adapters={"primary": primary, "secondary": secondary},
            failover=FailoverPolicy(
                chain=["primary", "secondary"],
                max_attempts=2,
                backoff_base=0.0,
                backoff_jitter=0.0,
            ),
            default_cwd=str(tmp_path),
        )
        events = await collect(agent.run(RunRequest(prompt="hi")))
        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert done.final_message.content == "from backup"
        # Both adapters were tried.
        assert len(primary.calls) == 1
        assert len(secondary.calls) == 1

    async def test_non_retryable_error_terminates(self, tmp_path: Path) -> None:
        # ConfigurationError is not in default retry_on list.
        primary = MockAdapter("primary", error=ConfigurationError("bad key"))
        secondary = MockAdapter("secondary", scripts=[text_turn("never")])
        agent, _ = make_agent(
            adapters={"primary": primary, "secondary": secondary},
            failover=FailoverPolicy(
                chain=["primary", "secondary"],
                max_attempts=2,
                backoff_base=0.0,
                backoff_jitter=0.0,
            ),
            default_cwd=str(tmp_path),
        )
        events = await collect(agent.run(RunRequest(prompt="hi")))
        error_event = next(e for e in events if isinstance(e, ErrorEvent))
        assert error_event.kind == "configuration"
        # Secondary was NOT tried.
        assert len(secondary.calls) == 0


# ---------------------------------------------------------------------------
# Construction / configuration errors
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_rejects_empty_adapters(self) -> None:
        with pytest.raises(ConfigurationError):
            Agent(
                adapters={},
                tools=__import__("harness.core").core.ToolRegistry(),
                storage=MockStorage(),
                failover=FailoverPolicy(chain=["x"]),
            )

    def test_rejects_chain_with_unknown_provider(self) -> None:
        with pytest.raises(ConfigurationError):
            Agent(
                adapters={"a": MockAdapter("a")},  # type: ignore[dict-item]
                tools=__import__("harness.core").core.ToolRegistry(),
                storage=MockStorage(),
                failover=FailoverPolicy(chain=["a", "missing"]),
            )


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestResume:
    async def test_resume_appends_to_existing_history(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[text_turn("first"), text_turn("second")],
        )
        agent, storage = make_agent(adapters={"mock": adapter}, default_cwd=str(tmp_path))

        # First turn
        req = RunRequest(prompt="hello", session_id="sess_keep")
        await collect(agent.run(req))

        # Resume with a new prompt
        await collect(agent.resume("sess_keep", prompt="more"))

        stored = await storage.get("sess_keep")
        assert stored is not None
        roles = [m.role for m in stored.messages]
        assert roles == ["user", "assistant", "user", "assistant"]
        assert stored.messages[0].content == "hello"
        assert stored.messages[2].content == "more"


# ---------------------------------------------------------------------------
# Stall detection
# ---------------------------------------------------------------------------


def _stall_script(total_chars: int) -> list[Event]:
    """Build a script that emits total_chars of TextDelta with no Done."""
    chunk = "x" * 500  # 500 chars per delta
    events: list[Event] = []
    emitted = 0
    while emitted < total_chars:
        events.append(TextDelta(text=chunk))
        emitted += len(chunk)
    # No Done event — the runtime should abort before reaching end anyway.
    return events


@pytest.mark.asyncio
class TestStallDetection:
    async def test_stall_yields_error_event(self, tmp_path: Path) -> None:
        from harness.core.runtime import Agent as _Agent

        limit = _Agent._STALL_CHAR_LIMIT
        adapter = MockAdapter("mock", scripts=[_stall_script(limit + 1000)])
        agent, _ = make_agent(adapters={"mock": adapter}, default_cwd=str(tmp_path))

        events = await collect(agent.run(RunRequest(prompt="deep dive on the code")))

        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert error_events, "expected an ErrorEvent when stall is detected"
        assert error_events[0].kind == "stall"
        assert "stall" in error_events[0].error.lower()

    async def test_normal_response_under_limit_succeeds(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("short answer")])
        agent, _ = make_agent(adapters={"mock": adapter}, default_cwd=str(tmp_path))

        events = await collect(agent.run(RunRequest(prompt="hi")))
        error_events = [e for e in events if isinstance(e, ErrorEvent)]
        assert not error_events
