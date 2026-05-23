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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from harness.core.agent_iter import AgentRun

from harness.core import activity as activity_kinds
from harness.core.activity import ActivityEvent, ActivityStore
from harness.core.adapter import Adapter
from harness.core.approval import ApprovalOutcome, ApprovalStore
from harness.core.budget import ContextBudget, count_tokens, prune
from harness.core.calibration import OutcomeCalibration
from harness.core.compactor import ContextCompactor
from harness.core.critic import Critic
from harness.core.errors import (
    CancelledError,
    ConfigurationError,
    Handoff,
    HarnessError,
    InternalError,
    StallError,
    ToolRetry,
)
from harness.core.events import (
    Critique,
    Done,
    ErrorEvent,
    Event,
    GuardrailTrippedEvent,
    HandoffEvent,
    ModelRequestEvent,
    PredictionEvent,
    PredictionMismatchEvent,
    StepCompleted,
    StepStarted,
    TextDelta,
    ToolResultEvent,
    Verification,
)
from harness.core.failover import FailoverPolicy, classify
from harness.core.guardrails import Guardrail
from harness.core.memory import MemoryStore
from harness.core.planner import NoOpPlanner, PlanContext, Planner
from harness.core.prediction import ConsequencePredictor, ToolPrediction, compare_prediction
from harness.core.repair import RepairOrchestrator
from harness.core.schemas import Message, RunRequest, Session, ToolCall, ToolResult
from harness.core.storage import Storage
from harness.core.telemetry import get_logger, span
from harness.core.test_signals import extract_failing_test_names as _extract_failing_test_names
from harness.core.tools import (
    ApprovalHandler,
    ApprovalPolicy,
    ToolRegistry,
    tool_matches_phase,
)
from harness.core.verification import EvidenceContract, VerificationGateway, Verifier

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
        critic: Critic | None = None,
        max_repair_attempts: int = 3,
        budget: ContextBudget | None = None,
        current_phase: str | None = None,
        default_provider: str | None = None,
        default_model: str | None = None,
        default_cwd: str | None = None,
        memory_store: MemoryStore | None = None,
        predictor: ConsequencePredictor | None = None,
        calibration: OutcomeCalibration | None = None,
        repair: RepairOrchestrator | None = None,
        evidence_contract: EvidenceContract | None = None,
        system_prompt: str | None = None,
        compactor: ContextCompactor | None = None,
        guardrails: list[Guardrail] | None = None,
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
        self.critic = critic
        """When set, the repair loop calls critic.critique(...) after each
        failed verification and prepends the critique to the repair directive.
        This forces the agent to address a specific hypothesis challenge rather
        than just re-reading raw failure output."""
        self._max_repair_attempts = max_repair_attempts
        """How many times to re-run the agent when the verifier returns
        can_finish=False. Each retry appends the failure output as a user
        message so the agent has concrete feedback to act on."""
        self.budget = budget
        """When set, the runtime prunes session.messages with a token-aware
        sliding window before each adapter call. The full session history is
        never mutated — only the view passed to the adapter shrinks."""
        self.memory_store = memory_store
        self.current_phase = current_phase
        """When set, only tools whose `phases` allow it are sent to the model
        and dispatched. When None, every registered tool is available
        (backward compatible)."""
        self.default_provider = default_provider or failover.chain[0]
        self.default_model = default_model
        self.default_cwd = default_cwd
        self._predictor = predictor
        self._calibration = calibration
        self._repair = repair
        self._evidence_contract = evidence_contract
        self.system_prompt = system_prompt
        self._compactor = compactor
        self.guardrails: list[Guardrail] = guardrails or []
        self._tool_cache: dict[str, ToolResult] = {}

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

    def iter(self, request: RunRequest) -> AgentRun:
        """Return an :class:`~harness.core.agent_iter.AgentRun` context manager.

        Exposes the run loop as typed :data:`~harness.core.agent_iter.AgentRunStep`
        objects rather than raw events::

            async with agent.iter(request) as run:
                async for step in run:
                    match step:
                        case ToolCallStep(tool_call=call):
                            print(f"calling {call.name}")
                        case FinalResponseStep(text=text):
                            print(text)
        """
        from harness.core.agent_iter import AgentRun

        return AgentRun(self, request)

    async def _run(self, request: RunRequest) -> AsyncIterator[Event]:
        session = await self._get_or_create_session(request)

        # Before appending the new user turn, replay any approvals the user
        # has granted out-of-band. This mutates the queued-for-approval tool
        # results in-place so the model sees real outcomes when it resumes.
        await self._replay_granted_approvals(session)

        # Load memories once per run so they're injected into every adapter turn.
        memory_prefix: list[Message] = []
        if self.memory_store is not None:
            entries = await self.memory_store.list_memory(limit=20)
            if entries:
                text = "\n".join(f"[{e.kind}] {e.text}" for e in entries)
                memory_prefix = [Message(role="system", content=f"Remembered context:\n{text}")]
        if self.system_prompt:
            memory_prefix.append(Message(role="system", content=self.system_prompt))

        if request.result_type is not None:
            import json as _json_schema

            schema = request.result_type.model_json_schema()  # type: ignore[attr-defined]
            memory_prefix.append(
                Message(
                    role="system",
                    content=(
                        "Respond ONLY with a valid JSON object (no markdown, no prose) "
                        "matching this schema:\n" + _json_schema.dumps(schema, indent=2)
                    ),
                )
            )

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
                yield StepStarted(
                    step=step_idx, description=step.description, total_steps=len(plan.steps)
                )
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
                    memory_prefix=memory_prefix,
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

        # Verification + repair loop.
        # If the verifier returns can_finish=False we append the failure as a
        # user message and run another agent turn, up to max_repair_attempts.
        # This is what gives weaker models the feedback loop they need.
        if self.verifier is not None:
            for _repair_attempt in range(self._max_repair_attempts + 1):
                last_verification: Verification | None = None
                async for ev in self._run_verification(session):
                    yield ev
                    if isinstance(ev, Verification):
                        last_verification = ev

                if (
                    last_verification is None
                    or last_verification.result.can_finish
                    or _repair_attempt >= self._max_repair_attempts
                ):
                    break

                # Optionally call the critic to challenge the agent's hypothesis
                # before assembling the repair directive.
                #
                # Defer the critic to attempt >= 2. Attempt 1 is just "re-read
                # the failure and try again" — adding a devil's-advocate
                # critique here causes small models to flip-flop on their own
                # diagnosis before they've even attempted to act on the new
                # information. Only invoke the critic once the model has
                # already failed at least once after seeing the test output.
                critique_text = ""
                if self.critic is not None and _repair_attempt >= 1:
                    activity_for_critic: list[ActivityEvent] = []
                    if self.activity_store is not None:
                        activity_for_critic = await self.activity_store.list_activity(
                            session_id=session.id, limit=500
                        )
                    critique_text = await self.critic.critique(
                        session=session,
                        verification_result=last_verification.result,
                        activity=activity_for_critic,
                    )
                    if critique_text:
                        yield Critique(attempt=_repair_attempt + 1, text=critique_text)

                # Build the repair directive: critique (if any) + raw failure output.
                attempt_label = (
                    f"Verification failed (attempt {_repair_attempt + 1} of "
                    f"{self._max_repair_attempts}).\n\n"
                )
                failing_tests = _extract_failing_test_names(last_verification.result.reason)
                failing_header = (
                    f"**Failing tests:** {', '.join(failing_tests)}\n\n" if failing_tests else ""
                )
                if critique_text:
                    repair_msg = (
                        attempt_label + failing_header + f"**Code Review:**\n{critique_text}\n\n"
                        f"**Test Output:**\n{last_verification.result.reason}\n\n"
                        "Address the code review and fix the remaining failures."
                    )
                else:
                    repair_msg = (
                        attempt_label + failing_header + f"{last_verification.result.reason}\n\n"
                        "Fix the remaining failures and try again."
                    )
                session.messages.append(Message(role="user", content=repair_msg))
                await self._emit(
                    session,
                    activity_kinds.REPAIR_DIRECTIVE_ISSUED,
                    {
                        "attempt": _repair_attempt + 1,
                        "verifier": last_verification.result.verifier_name,
                        "critic": bool(critique_text),
                        "reason_preview": last_verification.result.reason[:300],
                    },
                )
                async for ev in self._step_with_failover(
                    request=request,
                    session=session,
                    initial_yield_flag=True,
                    memory_prefix=memory_prefix,
                ):
                    yield ev

        session.status = "done"
        session.touch()
        await self._maybe_compact(session)
        await self.storage.save(session)
        await self._emit(session, activity_kinds.AGENT_RUN_COMPLETED)

    async def _stream_with_guardrails(
        self,
        stream: AsyncIterator[Event],
        messages: list[Message],
    ) -> AsyncIterator[Event]:
        """Wrap an adapter stream with guardrail checking.

        Blocking guardrails run before the stream starts. Parallel guardrails
        run as background tasks while the stream is consumed; if one trips, the
        stream is abandoned and a :class:`~harness.core.events.GuardrailTrippedEvent`
        is yielded instead of the remaining stream events.
        """
        if not self.guardrails:
            async for event in stream:
                yield event
            return

        blocking = [g for g in self.guardrails if g.mode == "blocking"]
        parallel = [g for g in self.guardrails if g.mode == "parallel"]

        for g in blocking:
            result = await g(messages)
            if result.tripped:
                yield GuardrailTrippedEvent(guardrail_name=g.name, reason=result.reason)
                return

        if not parallel:
            async for event in stream:
                yield event
            return

        # Launch parallel guardrails as background tasks
        guard_tasks = [asyncio.ensure_future(g(messages)) for g in parallel]
        guard_names = [g.name for g in parallel]

        async for event in stream:
            # Yield a tick so scheduled guardrail tasks can run.
            await asyncio.sleep(0)
            # Check completed guardrail tasks after each streamed event
            for i, task in enumerate(guard_tasks):
                if task.done() and not task.cancelled():
                    try:
                        gr = task.result()
                        if gr.tripped:
                            for t in guard_tasks:
                                if not t.done():
                                    t.cancel()
                            yield GuardrailTrippedEvent(
                                guardrail_name=guard_names[i], reason=gr.reason
                            )
                            return
                    except Exception:
                        pass
            yield event

        # Stream finished — await any remaining guardrail tasks before declaring clean.
        for i, task in enumerate(guard_tasks):
            try:
                gr = await task
                if gr.tripped:
                    for t in guard_tasks:
                        if not t.done():
                            t.cancel()
                    yield GuardrailTrippedEvent(guardrail_name=guard_names[i], reason=gr.reason)
                    return
            except (asyncio.CancelledError, Exception):
                pass

    async def _maybe_compact(self, session: Session) -> None:
        if self._compactor is None:
            return
        if not self._compactor.should_compact(session.messages):
            return
        session.messages = await self._compactor.compact(session.messages)

    async def _run_verification(self, session: Session) -> AsyncIterator[Event]:
        """Call the configured verifier, emit Verification event + activity."""
        assert self.verifier is not None  # guarded by caller
        activity_events: list[ActivityEvent] = []
        if self.activity_store is not None:
            activity_events = await self.activity_store.list_activity(
                session_id=session.id, limit=500
            )
        # Phase 4 — wrap verifier in VerificationGateway when evidence_contract is set.
        verifier = self.verifier
        if self._evidence_contract is not None:
            verifier = VerificationGateway(verifier, self._evidence_contract)
        try:
            result = await verifier.verify(session=session, activity=activity_events)
        except Exception as exc:
            logger.warning("agent.verifier.error", verifier=verifier.name, error=str(exc))
            # Don't swallow silently — synthesize a failure verdict the same
            # shape consumers expect.
            from harness.core.schemas import VerificationResult as _VR

            result = _VR(
                can_finish=False,
                reason=f"verifier {verifier.name!r} raised: {exc!s}",
                confidence=0.0,
                verifier_name=verifier.name,
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
        memory_prefix: list[Message] | None = None,
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
                    async for event in self._react_with(
                        adapter, request, session, memory_prefix=memory_prefix
                    ):
                        if not isinstance(event, ModelRequestEvent):
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

    # Per-turn character limit. Models should not output more than ~12 500
    # tokens (~50 000 chars) in a single response. Exceeding this is almost
    # always a generation loop. Not configurable — the runtime enforces it.
    _STALL_CHAR_LIMIT: int = 50_000

    async def _react_with(
        self,
        adapter: Adapter,
        request: RunRequest,
        session: Session,
        memory_prefix: list[Message] | None = None,
    ):
        import json as _json

        from pydantic import ValidationError as _ValidationError

        _retry_counts: dict[str, int] = {}
        for _turn in range(request.max_steps):
            final: Message | None = None
            usage = None
            char_count = 0

            messages_for_turn = await self._apply_budget(session, request)
            if memory_prefix:
                messages_for_turn = memory_prefix + messages_for_turn
            yield ModelRequestEvent(messages=messages_for_turn)
            _any_tool_called = any(m.role == "tool" for m in session.messages)
            _tool_choice: str | None = (
                "required"
                if request.require_tool_use
                and not _any_tool_called
                and self.tools.openai_schemas(phase=self.current_phase)
                else None
            )
            stream = adapter.stream(
                model=request.model or session.model,
                messages=messages_for_turn,
                tools=self.tools.openai_schemas(phase=self.current_phase) or None,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                tool_choice=_tool_choice,
            )
            _stream_source = (
                self._stream_with_guardrails(stream, messages_for_turn)
                if self.guardrails
                else stream
            )
            async for event in _stream_source:
                if isinstance(event, GuardrailTrippedEvent):
                    yield event
                    return
                if isinstance(event, Done):
                    final = event.final_message
                    usage = event.usage
                    break
                if isinstance(event, TextDelta):
                    char_count += len(event.text)
                    if char_count > self._STALL_CHAR_LIMIT:
                        await self._emit(
                            session,
                            activity_kinds.AGENT_RUN_STALLED,
                            {"chars_before_abort": char_count},
                        )
                        raise StallError(
                            f"model stall: output exceeded {self._STALL_CHAR_LIMIT:,} chars "
                            "in a single turn — possible generation loop. "
                            "Try a smaller/different model."
                        )
                yield event

            if final is None:
                raise InternalError("adapter ended stream without a Done event")

            session.messages.append(final)
            session.touch()

            if not final.tool_calls:
                # Structured output: validate assistant response against result_type.
                if request.result_type is not None:
                    raw = (final.content or "").strip()
                    # Strip markdown code fences if present.
                    if raw.startswith("```"):
                        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                    try:
                        parsed = request.result_type.model_validate_json(raw)  # type: ignore[attr-defined]
                        yield Done(
                            final_message=final,
                            usage=usage,
                            structured_result=parsed.model_dump(),
                        )
                        return
                    except (_ValidationError, _json.JSONDecodeError, ValueError) as exc:
                        session.messages.append(
                            Message(
                                role="user",
                                content=(
                                    f"Your response did not match the required JSON schema: {exc}. "
                                    "Please respond with valid JSON only."
                                ),
                            )
                        )
                        continue  # retry the adapter turn

                yield Done(final_message=final, usage=usage)
                return

            for tool_call in final.tool_calls:
                try:
                    result, extra_events = await self._invoke_tool(
                        tool_call, session, _retry_counts=_retry_counts
                    )
                except Handoff as handoff:
                    target_agent = handoff.target
                    target_name = getattr(target_agent, "name", type(target_agent).__name__)
                    yield HandoffEvent(target_name=target_name, reason=handoff.reason)
                    provider_name = target_agent.default_provider
                    target_adapter = target_agent.adapters[provider_name]
                    async for ev in target_agent._react_with(
                        target_adapter, request, session, memory_prefix=memory_prefix
                    ):
                        yield ev
                    return
                for ev in extra_events:
                    yield ev
                # Prompt-injection probe: scan tool output for hijack patterns
                # before it enters the next agent turn's context. If the output
                # contains text like "ignore previous instructions" or fake
                # SYSTEM:/[INST] markers, prepend a notice telling the model
                # to treat the content as data, not as instructions.
                # Local import: prompt_injection_probe imports re-only, so
                # this is effectively free at module-load time.
                from harness.core.prompt_injection_probe import annotate_if_suspicious

                annotated_content = annotate_if_suspicious(result.content or "")
                session.messages.append(
                    Message(
                        role="tool",
                        tool_call_id=tool_call.id,
                        name=tool_call.name,
                        content=annotated_content,
                    )
                )
                session.touch()
                yield ToolResultEvent(result=result)

        raise InternalError(f"exceeded max_steps={request.max_steps} without final answer")

    # ------------------------------------------------------------------ #
    # Tool dispatch                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cache_key(tool_name: str, arguments: dict) -> str:
        import hashlib
        import json as _json

        payload = tool_name + _json.dumps(arguments, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    async def _invoke_tool(
        self,
        call: ToolCall,
        session: Session,
        _retry_counts: dict[str, int] | None = None,
    ) -> tuple[ToolResult, list[Event]]:
        """Dispatch a tool call through the full gate pipeline.

        Returns (result, extra_events) where extra_events contains any
        PredictionEvent / PredictionMismatchEvent to yield before ToolResultEvent.
        """
        extra_events: list[Event] = []

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
            return result, extra_events

        tool = self.tools.get(call.name)

        # Phase filter: defence-in-depth — refuse out-of-phase calls.
        if not tool_matches_phase(tool, self.current_phase):
            result = ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content=(f"tool {call.name!r} is not available in phase {self.current_phase!r}"),
                is_error=True,
            )
            await self._emit_tool_completed(session, call, result)
            return result, extra_events

        # Phase 3 — Verifier isolation: block mutating tools in verify phase.
        if self.current_phase == "verify":
            scope = getattr(tool, "effect_scope", None)
            if scope not in (None, "read_only", "session_ephemeral"):
                result = ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content=(
                        f"verifier isolation: tool {call.name!r} (scope={scope!r}) "
                        "is blocked in verify phase — verifiers are read-only"
                    ),
                    is_error=True,
                )
                await self._emit_tool_completed(session, call, result)
                return result, extra_events

        decision = self.approval_policy.decide(tool, session_overrides=session.approval_overrides)

        if decision == "deny":
            result = ToolResult(
                tool_call_id=call.id,
                name=call.name,
                content="tool denied by policy",
                is_error=True,
            )
            await self._emit_tool_completed(session, call, result)
            return result, extra_events

        if decision == "prompt":
            if self.approval_handler is None:
                result = ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content="approval required but no handler configured",
                    is_error=True,
                )
                await self._emit_tool_completed(session, call, result)
                return result, extra_events
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
                return result, extra_events
            if outcome == "queued":
                result = ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content=("queued for approval — review with `harness approvals list`"),
                    is_error=True,
                )
                await self._emit_tool_completed(session, call, result)
                return result, extra_events
            # outcome == "approved" — fall through to execute

        # Phase 2 — ConsequencePredictor: commit to prediction before execution.
        prediction: ToolPrediction | None = None
        if self._predictor is not None:
            effect_scope = getattr(tool, "effect_scope", None)
            prediction = self._predictor.predict(
                tool_name=call.name, call=call, effect_scope=effect_scope
            )
            await self._emit(
                session,
                activity_kinds.TOOL_CALL_PREDICTED,
                prediction.model_dump(mode="json"),
            )
            extra_events.append(PredictionEvent(prediction=prediction))

        # Tool result cache: opt-in via `cache = True` attribute on the tool.
        tool_cacheable = getattr(tool, "cache", False)
        cache_key: str = ""
        if tool_cacheable:
            cache_key = self._cache_key(call.name, call.arguments)
            cached = self._tool_cache.get(cache_key)
            if cached is not None:
                result = ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content=cached.content,
                    is_error=cached.is_error,
                    metadata=cached.metadata,
                )
                await self._emit_tool_completed(session, call, result)
                return result, extra_events

        max_retries: int = getattr(tool, "max_retries", 3)
        retry_counts = _retry_counts if _retry_counts is not None else {}
        prior_retries = retry_counts.get(call.name, 0)

        started = time.perf_counter()
        try:
            with span("agent.tool", tool=call.name, call_id=call.id):
                result = await tool(call)
        except ToolRetry as exc:
            duration_ms = int((time.perf_counter() - started) * 1000)
            if prior_retries >= max_retries:
                logger.warning(
                    "agent.tool.retry_exhausted",
                    tool=call.name,
                    attempts=prior_retries,
                    feedback=exc.message,
                )
                result = ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content=(
                        f"[ToolRetry exhausted after {prior_retries} attempt(s)] {exc.message}"
                    ),
                    is_error=True,
                )
            else:
                retry_counts[call.name] = prior_retries + 1
                logger.info(
                    "agent.tool.retry",
                    tool=call.name,
                    attempt=prior_retries + 1,
                    feedback=exc.message,
                )
                result = ToolResult(
                    tool_call_id=call.id,
                    name=call.name,
                    content=f"[ToolRetry] {exc.message} — please fix your input and try again.",
                    is_error=True,
                )
            await self._emit_tool_completed(session, call, result, duration_ms=duration_ms)
            return result, extra_events
        except Handoff:
            # Re-raise so _react_with can catch it and delegate to the target agent.
            raise
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

        # Store in cache if the tool opted in and the result is not an error.
        if tool_cacheable and not result.is_error:
            self._tool_cache[cache_key] = result

        # Phase 2 — PredictionError: compare prediction vs actual.
        pred_outcome = None
        if prediction is not None:
            pred_outcome = compare_prediction(prediction, result)
            await self._emit(
                session,
                activity_kinds.TOOL_CALL_PREDICTION_ERROR,
                pred_outcome.model_dump(mode="json"),
            )
            if not pred_outcome.matched:
                extra_events.append(PredictionMismatchEvent(outcome=pred_outcome))

            # Phase 5 — OutcomeCalibration: adjust confidence based on outcome.
            if self._calibration is not None:
                cal_record = self._calibration.record(
                    tool_name=call.name,
                    effect_scope=prediction.effect_scope,
                    base_confidence=prediction.confidence,
                    outcome=pred_outcome,
                )
                await self._emit(
                    session,
                    activity_kinds.CALIBRATION_UPDATED,
                    cal_record.model_dump(mode="json"),
                )

        # Phase 6 — RepairOrchestrator: track failure streaks, emit directive.
        if self._repair is not None:
            directive = self._repair.assess(
                tool_name=call.name,
                effect_scope=getattr(tool, "effect_scope", None),
                result=result,
                outcome=pred_outcome,
            )
            await self._emit(
                session,
                activity_kinds.REPAIR_DIRECTIVE_ISSUED,
                directive.model_dump(mode="json"),
            )

        return result, extra_events

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


async def fork_session(
    storage: Storage,
    parent_id: str,
    *,
    new_session_id: str | None = None,
) -> Session:
    """Branch from a parent session's message history into a new session.

    The fork copies messages and approval_overrides from the parent. Activity
    ledger, approval inbox, and task session_ids list are NOT copied — the fork
    starts its own audit trail.
    """
    from harness.core.schemas import _new_id  # avoid circular at module level

    parent = await storage.get(parent_id)
    if parent is None:
        raise ConfigurationError(f"session {parent_id!r} not found")

    forked = Session(
        id=new_session_id or _new_id("sess"),
        provider=parent.provider,
        model=parent.model,
        cwd=parent.cwd,
        task_id=parent.task_id,
        forked_from=parent.id,
        status="pending",
        messages=[m.model_copy(deep=True) for m in parent.messages],
        approval_overrides=dict(parent.approval_overrides),
    )
    await storage.save(forked)
    return forked


__all__ = ["Agent", "fork_session"]
