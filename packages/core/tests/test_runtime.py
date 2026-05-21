"""End-to-end tests for the Agent ReAct loop with MockAdapter + MockTool."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.core import (
    Agent,
    ApprovalPolicy,
    AutoApprove,
    AutoDeny,
    ConfigurationError,
    Done,
    ErrorEvent,
    Event,
    FailoverPolicy,
    NetworkError,
    RunRequest,
    StallError,
    StepCompleted,
    StepStarted,
    TextDelta,
    ToolResultEvent,
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
    default_model: str = "test-model",
    default_cwd: str | None = None,
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
        default_model=default_model,
        default_cwd=default_cwd,
    )
    return agent, storage


# ---------------------------------------------------------------------------
# Happy path: plain text answer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestHappyPath:
    async def test_text_only_turn(self, tmp_path: Path) -> None:
        adapter = MockAdapter("mock", scripts=[text_turn("hello world")])
        agent, _storage = make_agent(adapters={"mock": adapter}, default_cwd=str(tmp_path))

        events = await collect(agent.run(RunRequest(prompt="hi there")))

        # Stream shape: StepStarted, TextDelta, Done, StepCompleted
        assert isinstance(events[0], StepStarted)
        assert isinstance(events[1], TextDelta)
        assert events[1].text == "hello world"
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
