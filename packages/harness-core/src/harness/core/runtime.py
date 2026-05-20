"""Agent runtime.

The Agent ties adapters, tools, storage, planner, approval, and failover into
a single ReAct loop. Inputs are `RunRequest`; outputs are streams of `Event`.

Failover semantics in v1:
- The failover chain is consulted on the *first* call to an adapter.
- Once any event has been yielded to the consumer we stop failing over —
  partial state can't be cleanly retried against a different provider.
- Adapter-level errors before any yield trigger backoff + retry per
  `FailoverPolicy.should_retry`.

Tool dispatch:
- The adapter emits each ToolCallEvent during streaming AND echoes the
  aggregated tool_calls inside `Done.final_message`. The authoritative source
  for dispatch is `final_message.tool_calls`.
- For each tool call, the runtime resolves approval, invokes (or denies), and
  appends a `role=tool` Message to the session. Then it loops back to the
  adapter for the model's next turn.

This module has no I/O of its own — it's all async orchestration over
injected dependencies.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

from harness.core.adapter import Adapter
from harness.core.errors import (
    CancelledError,
    ConfigurationError,
    HarnessError,
    InternalError,
)
from harness.core.events import (
    Done,
    ErrorEvent,
    Event,
    StepCompleted,
    StepStarted,
    ToolResultEvent,
)
from harness.core.failover import FailoverPolicy, classify
from harness.core.planner import NoOpPlanner, PlanContext, Planner
from harness.core.schemas import Message, RunRequest, Session, ToolCall, ToolResult
from harness.core.storage import Storage
from harness.core.telemetry import get_logger, span
from harness.core.tools import ApprovalHandler, ApprovalPolicy, ToolRegistry

logger = get_logger("harness.runtime")


class Agent:
    """The Harness ReAct runtime.

    Construct once with all dependencies, then call `run(request)` per turn.
    The agent is stateless across calls; durable state lives in `Storage`.
    """

    def __init__(
        self,
        *,
        adapters: dict[str, Adapter],
        tools: ToolRegistry,
        storage: Storage,
        failover: FailoverPolicy,
        approval_policy: ApprovalPolicy | None = None,
        approval_handler: ApprovalHandler | None = None,
        planner: Planner | None = None,
        default_provider: str | None = None,
        default_model: str | None = None,
        default_cwd: str | None = None,
    ) -> None:
        if not adapters:
            raise ConfigurationError("at least one adapter is required")
        for provider in failover.chain:
            if provider not in adapters:
                raise ConfigurationError(f"failover chain references unknown provider {provider!r}")

        self.adapters = adapters
        self.tools = tools
        self.storage = storage
        self.failover = failover
        self.approval_policy = approval_policy or ApprovalPolicy()
        self.approval_handler = approval_handler
        self.planner: Planner = planner or NoOpPlanner()
        self.default_provider = default_provider or failover.chain[0]
        self.default_model = default_model
        self.default_cwd = default_cwd

    # ------------------------------------------------------------------ #
    # Public API                                                          #
    # ------------------------------------------------------------------ #

    async def run(self, request: RunRequest) -> AsyncIterator[Event]:
        async for event in self._run(request):
            yield event

    async def _run(self, request: RunRequest) -> AsyncIterator[Event]:
        session = await self._get_or_create_session(request)
        session.messages.append(Message(role="user", content=request.prompt))
        session.status = "running"
        session.touch()
        await self.storage.save(session)

        plan = await self.planner.plan(
            request.prompt,
            PlanContext(
                session_id=session.id,
                messages=session.messages,
                available_tools=self.tools.names(),
            ),
        )

        any_yielded = False
        try:
            for step_idx, step in enumerate(plan.steps):
                yield StepStarted(step=step_idx, description=step.description)
                any_yielded = True

                async for event in self._step_with_failover(
                    request=request,
                    session=session,
                    initial_yield_flag=any_yielded,
                ):
                    any_yielded = True
                    yield event

                yield StepCompleted(step=step_idx)
        except asyncio.CancelledError:
            session.status = "cancelled"
            session.touch()
            await self.storage.save(session)
            raise
        except CancelledError:
            session.status = "cancelled"
            session.touch()
            await self.storage.save(session)
            raise
        except HarnessError as exc:
            kind = classify(exc)
            logger.error("agent.run.failed", error=str(exc), kind=kind)
            session.status = "failed"
            session.touch()
            await self.storage.save(session)
            yield ErrorEvent(error=str(exc), kind=kind, recoverable=False)
            return

        session.status = "done"
        session.touch()
        await self.storage.save(session)

    async def resume(
        self,
        session_id: str,
        prompt: str | None = None,
        **overrides: Any,
    ) -> AsyncIterator[Event]:
        """Resume a prior session, optionally with a new user prompt."""
        stored = await self.storage.get(session_id)
        if stored is None:
            raise ConfigurationError(f"unknown session {session_id!r}")

        request = RunRequest(
            session_id=session_id,
            prompt=prompt or "",
            provider=overrides.get("provider", stored.provider),
            model=overrides.get("model", stored.model),
            temperature=overrides.get("temperature"),
            max_tokens=overrides.get("max_tokens"),
            max_steps=overrides.get("max_steps", 25),
        )
        async for event in self._run(request):
            yield event

    # ------------------------------------------------------------------ #
    # Session bootstrap                                                   #
    # ------------------------------------------------------------------ #

    async def _get_or_create_session(self, request: RunRequest) -> Session:
        existing = await self.storage.get(request.session_id)
        if existing is not None:
            return existing

        provider = request.provider or self.default_provider
        model = request.model or self.default_model
        if model is None:
            raise ConfigurationError(
                "no model specified: pass `model=` in RunRequest or set `default_model` on Agent"
            )

        from pathlib import Path

        cwd = Path(self.default_cwd) if self.default_cwd else Path.cwd()
        session = Session(
            id=request.session_id,
            provider=provider,
            model=model,
            cwd=cwd,
        )
        return session

    # ------------------------------------------------------------------ #
    # Step / failover                                                     #
    # ------------------------------------------------------------------ #

    async def _step_with_failover(
        self,
        *,
        request: RunRequest,
        session: Session,
        initial_yield_flag: bool,
    ):
        """Run one plan step with bounded failover.

        Once we've yielded any token through (`yielded_any` becomes True),
        further failover is suppressed — we can't unsay events.
        """
        yielded_any = False
        last_exc: BaseException | None = None

        for attempt in range(self.failover.max_attempts):
            provider_name = self.failover.next_provider(attempt=attempt)
            adapter = self.adapters[provider_name]

            try:
                with span(
                    "agent.step", provider=provider_name, attempt=attempt, session=session.id
                ):
                    async for event in self._react_with(adapter, request, session):
                        yielded_any = True
                        yield event
                return
            except asyncio.CancelledError:
                raise
            except CancelledError:
                raise
            except HarnessError as exc:
                last_exc = exc
                logger.warning(
                    "agent.step.error",
                    provider=provider_name,
                    attempt=attempt,
                    kind=classify(exc),
                    error=str(exc),
                )
                if yielded_any:
                    raise
                if not self.failover.should_retry(exc, attempt=attempt):
                    raise
                await asyncio.sleep(self.failover.backoff(attempt=attempt))
                continue

        # Exhausted the chain
        if last_exc is not None:
            raise last_exc
        raise InternalError("failover exhausted with no recorded error")

    # ------------------------------------------------------------------ #
    # ReAct loop (one step, single adapter)                               #
    # ------------------------------------------------------------------ #

    async def _react_with(
        self,
        adapter: Adapter,
        request: RunRequest,
        session: Session,
    ):
        for _turn in range(request.max_steps):
            final: Message | None = None
            usage = None

            stream = adapter.stream(
                model=request.model or session.model,
                messages=session.messages,
                tools=self.tools.openai_schemas() or None,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            async for event in stream:
                if isinstance(event, Done):
                    final = event.final_message
                    usage = event.usage
                    break
                yield event

            if final is None:
                raise InternalError("adapter ended stream without a Done event")

            session.messages.append(final)
            session.touch()

            if not final.tool_calls:
                yield Done(final_message=final, usage=usage)
                return

            for tool_call in final.tool_calls:
                result = await self._invoke_tool(tool_call, session)
                session.messages.append(
                    Message(
                        role="tool",
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        content=result.content,
                    )
                )
                session.touch()
                yield ToolResultEvent(result=result)

        raise InternalError(f"exceeded max_steps={request.max_steps} without final answer")

    # ------------------------------------------------------------------ #
    # Tool dispatch                                                       #
    # ------------------------------------------------------------------ #

    async def _invoke_tool(self, call: ToolCall, session: Session) -> ToolResult:
        if not self.tools.has(call.name):
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"unknown tool: {call.name!r}",
                is_error=True,
            )

        tool = self.tools.get(call.name)
        decision = self.approval_policy.decide(tool, session_overrides=session.approval_overrides)

        if decision == "deny":
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content="tool denied by policy",
                is_error=True,
            )
        if decision == "prompt":
            if self.approval_handler is None:
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content="approval required but no handler configured",
                    is_error=True,
                )
            approved = await self.approval_handler(tool, call, session)
            if not approved:
                return ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content="user denied approval",
                    is_error=True,
                )

        try:
            with span("agent.tool", tool=call.name, call_id=call.id):
                return await tool(call)
        except Exception as exc:
            logger.warning("agent.tool.error", tool=call.name, error=str(exc))
            return ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"tool error: {exc!s}",
                is_error=True,
            )


__all__ = ["Agent"]
