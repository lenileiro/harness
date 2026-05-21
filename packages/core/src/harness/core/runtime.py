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
import time
from collections.abc import AsyncIterator
from typing import Any

from harness.core import activity as activity_kinds
from harness.core.activity import ActivityEvent, ActivityStore
from harness.core.adapter import Adapter
from harness.core.approval import ApprovalOutcome, ApprovalStore
from harness.core.budget import ContextBudget, count_tokens, prune
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
    Verification,
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
from harness.core.verification import Verifier

logger = get_logger("harness.runtime")


def _normalize_outcome(raw: bool | ApprovalOutcome) -> ApprovalOutcome:
    """Coerce legacy `bool` return values into the open `ApprovalOutcome` set."""
    if isinstance(raw, bool):
        return "approved" if raw else "denied"
    if raw in ("approved", "denied", "queued"):
        return raw  # type: ignore[return-value]
    # Unknown string — treat as denied, the safe fallback.
    logger.warning("agent.approval.unknown_outcome", outcome=raw)
    return "denied"


_OUTCOME_TO_ACTIVITY = {
    "approved": activity_kinds.APPROVAL_GRANTED,
    "denied": activity_kinds.APPROVAL_DENIED,
    "queued": activity_kinds.APPROVAL_QUEUED,
}


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
        approval_store: ApprovalStore | None = None,
        verifier: Verifier | None = None,
        budget: ContextBudget | None = None,
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
        self.approval_store = approval_store
        """When set, the runtime checks for granted approvals at the top of
        every run and re-dispatches them — see `_replay_granted_approvals`."""
        self.verifier = verifier
        """When set, the runtime calls verifier.verify(...) after the terminal
        Done event and yields a Verification(result=...) event. The verdict is
        also recorded as a `verification.completed` activity entry."""
        self.budget = budget
        """When set, the runtime prunes session.messages with a token-aware
        sliding window before each adapter call. The full session history is
        never mutated — only the view passed to the adapter shrinks."""
        self.current_phase = current_phase
        """When set, only tools whose `phases` allow it are sent to the model
        and dispatched. When None, every registered tool is available
        (backward compatible)."""
        self.default_provider = default_provider or failover.chain[0]
        self.default_model = default_model
        self.default_cwd = default_cwd

    # ------------------------------------------------------------------ #
    # Approval replay                                                     #
    # ------------------------------------------------------------------ #

    async def _replay_granted_approvals(self, session: Session) -> None:
        """Re-dispatch queued tool calls the user has granted out-of-band.

        For each granted-but-unreplayed PendingApproval on this session, we:

          1. Find the corresponding `role=tool` message in `session.messages`
             (matched by `tool_call_id`).
          2. Invoke the tool with the original arguments.
          3. Overwrite that message's content with the real result (and
             update `is_error` semantics in the queued placeholder).
          4. Mark the approval as replayed.

        Errors are best-effort: a missing tool or message logs a warning and
        leaves the queued placeholder unchanged. The user can then re-deny
        or recreate the call.
        """
        if self.approval_store is None:
            return
        granted = await self.approval_store.list_unreplayed_granted(session_id=session.id)
        if not granted:
            return

        for approval in granted:
            # Locate the tool message in transcript.
            tool_msg = next(
                (
                    m
                    for m in session.messages
                    if m.role == "tool" and m.tool_call_id == approval.tool_call_id
                ),
                None,
            )
            if tool_msg is None:
                logger.warning(
                    "agent.approval.replay.no_tool_message",
                    approval=approval.id,
                    tool_call_id=approval.tool_call_id,
                )
                await self.approval_store.mark_replayed(approval.id)
                continue

            if not self.tools.has(approval.tool_name):
                logger.warning(
                    "agent.approval.replay.unknown_tool",
                    approval=approval.id,
                    tool=approval.tool_name,
                )
                tool_msg.content = f"replay failed: unknown tool {approval.tool_name!r}"
                await self.approval_store.mark_replayed(approval.id)
                continue

            tool = self.tools.get(approval.tool_name)
            call = ToolCall(
                id=approval.tool_call_id,
                name=approval.tool_name,
                arguments=approval.arguments,
            )
            try:
                with span("agent.tool.replay", tool=tool.name, call_id=call.id):
                    result = await tool(call)
            except Exception as exc:
                logger.warning(
                    "agent.approval.replay.tool_error",
                    approval=approval.id,
                    error=str(exc),
                )
                result = ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content=f"tool error during replay: {exc!s}",
                    is_error=True,
                )

            # Overwrite the queued placeholder in transcript.
            tool_msg.content = result.content
            await self.approval_store.mark_replayed(approval.id)
            await self._emit(
                session,
                activity_kinds.APPROVAL_REPLAYED,
                {
                    "approval_id": approval.id,
                    "tool_call_id": approval.tool_call_id,
                    "name": approval.tool_name,
                    "is_error": result.is_error,
                },
            )

        # Persist the mutated transcript so subsequent loads see the real
        # results, not the queued placeholders.
        session.touch()
        await self.storage.save(session)

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

        # Before appending the new user turn, replay any approvals the user
        # has granted out-of-band. This mutates the queued-for-approval tool
        # results in-place so the model sees real outcomes when it resumes.
        await self._replay_granted_approvals(session)

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

        # Run the optional verifier before marking the session done. The
        # verifier may decide can_finish=False, but the session is still
        # closed for this turn — the verdict is advisory in v1, with the
        # consumer choosing what to do (resume, escalate, ignore).
        if self.verifier is not None:
            async for ev in self._run_verification(session):
                yield ev

        session.status = "done"
        session.touch()
        await self.storage.save(session)
        await self._emit(session, activity_kinds.AGENT_RUN_COMPLETED)

    async def _run_verification(self, session: Session) -> AsyncIterator[Event]:
        """Call the configured verifier, emit Verification event + activity."""
        assert self.verifier is not None  # guarded by caller
        activity_events: list[ActivityEvent] = []
        if self.activity_store is not None:
            activity_events = await self.activity_store.list_activity(
                session_id=session.id, limit=500
            )
        try:
            result = await self.verifier.verify(session=session, activity=activity_events)
        except Exception as exc:
            logger.warning("agent.verifier.error", verifier=self.verifier.name, error=str(exc))
            # Don't swallow silently — synthesize a failure verdict the same
            # shape consumers expect.
            from harness.core.schemas import VerificationResult as _VR

            result = _VR(
                can_finish=False,
                reason=f"verifier {self.verifier.name!r} raised: {exc!s}",
                confidence=0.0,
                verifier_name=self.verifier.name,
            )
        await self._emit(
            session,
            activity_kinds.VERIFICATION_COMPLETED,
            {
                "verifier_name": result.verifier_name,
                "can_finish": result.can_finish,
                "reason": result.reason,
                "confidence": result.confidence,
                "evidence_event_ids": list(result.evidence_event_ids),
            },
        )
        yield Verification(result=result)

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

            messages_for_turn = await self._apply_budget(session, request)
            stream = adapter.stream(
                model=request.model or session.model,
                messages=messages_for_turn,
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
            raw_outcome = await self.approval_handler(tool, call, session)
            outcome = _normalize_outcome(raw_outcome)
            await self._emit(
                session,
                _OUTCOME_TO_ACTIVITY[outcome],
                {"tool_call_id": call.id, "name": call.name},
            )
            if outcome == "denied":
                result = ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content="user denied approval",
                    is_error=True,
                )
                await self._emit_tool_completed(session, call, result)
                return result
            if outcome == "queued":
                result = ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content=("queued for approval — review with `harness approvals list`"),
                    is_error=True,
                )
                await self._emit_tool_completed(session, call, result)
                return result
            # outcome == "approved" — fall through to execute

        started = time.perf_counter()
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
        duration_ms = int((time.perf_counter() - started) * 1000)
        await self._emit_tool_completed(session, call, result, duration_ms=duration_ms)
        return result

    async def _apply_budget(self, session: Session, request: RunRequest) -> list[Message]:
        """Return the message list to actually send the adapter.

        When `self.budget` is set, runs the pruner. Emits a `context.pruned`
        activity event when the pruned message list is shorter than the
        full transcript (so the user can see when truncation kicked in).

        Returns `session.messages` unchanged if no budget is configured —
        full backward compatibility.
        """
        if self.budget is None:
            return session.messages
        model = request.model or session.model
        before = len(session.messages)
        before_tokens = count_tokens(session.messages, model)
        pruned = prune(session.messages, budget=self.budget, model=model)
        if len(pruned) < before:
            after_tokens = count_tokens(pruned, model)
            await self._emit(
                session,
                activity_kinds.CONTEXT_PRUNED,
                {
                    "model": model,
                    "max_tokens": self.budget.max_tokens,
                    "messages_before": before,
                    "messages_after": len(pruned),
                    "tokens_before": before_tokens,
                    "tokens_after": after_tokens,
                },
            )
        return pruned

    async def _emit_tool_completed(
        self,
        session: Session,
        call: ToolCall,
        result: ToolResult,
        *,
        duration_ms: int | None = None,
    ) -> None:
        """Emit the evidence record for a tool call.

        `tool_call.completed` data shape (the evidence ledger entry):

          tool_call_id, name, is_error, content_preview, content_size,
          arguments, duration_ms (None when the call short-circuited before
          execution — e.g. denied / queued / out-of-phase), metadata (the
          tool's own structured fields).
        """
        await self._emit(
            session,
            activity_kinds.TOOL_CALL_COMPLETED,
            {
                "tool_call_id": call.id,
                "name": call.name,
                "is_error": result.is_error,
                "content_preview": result.content[:200],
                "content_size": len(result.content),
                "arguments": call.arguments,
                "duration_ms": duration_ms,
                "metadata": result.metadata or {},
            },
        )


__all__ = ["Agent"]
