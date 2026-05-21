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

from harness.core import activity as activity_kinds
from harness.core.activity import ActivityEvent, ActivityStore
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
from harness.core.tools import (
    ApprovalHandler,
    ApprovalPolicy,
    ToolRegistry,
    tool_matches_phase,
)

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
        activity_store: ActivityStore | None = None,
        current_phase: str | None = None,
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
        self.activity_store = activity_store
        self.current_phase = current_phase
        """When set, only tools whose `phases` allow it are sent to the model
        and dispatched. When None, every registered tool is available
        (backward compatible)."""
        self.default_provider = default_provider or failover.chain[0]
        self.default_model = default_model
        self.default_cwd = default_cwd

    # ------------------------------------------------------------------ #
    # Activity log emission                                               #
    # ------------------------------------------------------------------ #

    async def _emit(self, session: Session, kind: str, data: dict[str, Any] | None = None) -> None:
        """Append an ActivityEvent if an activity_store is configured.

        Swallows storage errors (logged, not raised) so a flaky ledger never
        breaks the agent loop.
        """
        if self.activity_store is None:
            return
        try:
            event = ActivityEvent(
                task_id=session.task_id,
                session_id=session.id,
                kind=kind,
                data=data or {},
            )
            await self.activity_store.append_activity(event)
        except Exception as exc:
            logger.warning("agent.activity.emit_failed", kind=kind, error=str(exc))

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
        await self._emit(
            session,
            activity_kinds.AGENT_RUN_STARTED,
            {"provider": session.provider, "model": session.model, "prompt": request.prompt},
        )

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
                await self._emit(
                    session,
                    activity_kinds.STEP_STARTED,
                    {"step": step_idx, "description": step.description},
                )

                async for event in self._step_with_failover(
                    request=request,
                    session=session,
                    initial_yield_flag=any_yielded,
                ):
                    any_yielded = True
                    yield event

                yield StepCompleted(step=step_idx)
                await self._emit(session, activity_kinds.STEP_COMPLETED, {"step": step_idx})
        except asyncio.CancelledError:
            session.status = "cancelled"
            session.touch()
            await self.storage.save(session)
            await self._emit(session, activity_kinds.AGENT_RUN_CANCELLED)
            raise
        except CancelledError:
            session.status = "cancelled"
            session.touch()
            await self.storage.save(session)
            await self._emit(session, activity_kinds.AGENT_RUN_CANCELLED)
            raise
        except HarnessError as exc:
            kind = classify(exc)
            logger.error("agent.run.failed", error=str(exc), kind=kind)
            session.status = "failed"
            session.touch()
            await self.storage.save(session)
            await self._emit(
                session,
                activity_kinds.AGENT_RUN_FAILED,
                {"error": str(exc), "kind": kind},
            )
            yield ErrorEvent(error=str(exc), kind=kind, recoverable=False)
            return

        session.status = "done"
        session.touch()
        await self.storage.save(session)
        await self._emit(session, activity_kinds.AGENT_RUN_COMPLETED)

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
            task_id=request.task_id,
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
                tools=self.tools.openai_schemas(phase=self.current_phase) or None,
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
        await self._emit(
            session,
            activity_kinds.TOOL_CALL_DISPATCHED,
            {"tool_call_id": call.id, "name": call.name, "arguments": call.arguments},
        )

        if not self.tools.has(call.name):
            result = ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"unknown tool: {call.name!r}",
                is_error=True,
            )
            await self._emit_tool_completed(session, call, result)
            return result

        tool = self.tools.get(call.name)

        # Defence in depth: filter both the schemas sent to the model AND
        # what we dispatch. If a model hallucinates an out-of-phase call,
        # refuse rather than execute.
        if not tool_matches_phase(tool, self.current_phase):
            result = ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=(f"tool {call.name!r} is not available in phase {self.current_phase!r}"),
                is_error=True,
            )
            await self._emit_tool_completed(session, call, result)
            return result

        decision = self.approval_policy.decide(tool, session_overrides=session.approval_overrides)

        if decision == "deny":
            result = ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content="tool denied by policy",
                is_error=True,
            )
            await self._emit_tool_completed(session, call, result)
            return result

        if decision == "prompt":
            if self.approval_handler is None:
                result = ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content="approval required but no handler configured",
                    is_error=True,
                )
                await self._emit_tool_completed(session, call, result)
                return result
            await self._emit(
                session,
                activity_kinds.APPROVAL_REQUESTED,
                {"tool_call_id": call.id, "name": call.name, "arguments": call.arguments},
            )
            approved = await self.approval_handler(tool, call, session)
            await self._emit(
                session,
                activity_kinds.APPROVAL_GRANTED if approved else activity_kinds.APPROVAL_DENIED,
                {"tool_call_id": call.id, "name": call.name},
            )
            if not approved:
                result = ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content="user denied approval",
                    is_error=True,
                )
                await self._emit_tool_completed(session, call, result)
                return result

        try:
            with span("agent.tool", tool=call.name, call_id=call.id):
                result = await tool(call)
        except Exception as exc:
            logger.warning("agent.tool.error", tool=call.name, error=str(exc))
            result = ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=f"tool error: {exc!s}",
                is_error=True,
            )
        await self._emit_tool_completed(session, call, result)
        return result

    async def _emit_tool_completed(
        self, session: Session, call: ToolCall, result: ToolResult
    ) -> None:
        await self._emit(
            session,
            activity_kinds.TOOL_CALL_COMPLETED,
            {
                "tool_call_id": call.id,
                "name": call.name,
                "is_error": result.is_error,
                "content_preview": result.content[:200],
            },
        )


__all__ = ["Agent"]
