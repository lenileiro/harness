"""Bounded, resumable mission workers backed by the normal Agent runtime."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from contextlib import aclosing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from harness.core.adapter import Adapter
from harness.core.budget import count_tokens
from harness.core.events import Done, ErrorEvent, Event, Verification
from harness.core.mission.models import Mission, MissionFeature, MissionHandoff, MissionRun
from harness.core.mission.roles import resolve_mission_role_profile
from harness.core.mission.runtime import (
    MissionBurstResult,
    MissionLoopStep,
    complete_mission_feature,
    execute_next_mission_feature,
)
from harness.core.mission.store import MissionStore
from harness.core.mission.validator import validate_mission_milestone
from harness.core.runtime import Agent
from harness.core.scheduler.store import SchedulerStore
from harness.core.schemas import Capabilities, Message, RunRequest


@dataclass
class _TokenBudget:
    limit: int | None
    used: int


class _BudgetAdapter:
    """Account for every adapter turn, including tool-use turns and failovers."""

    def __init__(self, adapter: Adapter, budget: _TokenBudget):
        self.adapter, self.budget = adapter, budget
        self.name = adapter.name

    def __getattr__(self, name: str) -> Any:
        return getattr(self.adapter, name)

    async def capabilities(self) -> Capabilities:
        return await self.adapter.capabilities()

    async def cancel(self, session_id: str) -> None:
        await self.adapter.cancel(session_id)

    async def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Event]:
        prompt_estimate = count_tokens(messages, model)
        if self.budget.limit is not None:
            remaining = self.budget.limit - self.budget.used - prompt_estimate
            if remaining < 1:
                raise RuntimeError("mission token budget exhausted")
            max_tokens = min(max_tokens, remaining) if max_tokens is not None else remaining
        accounted = False
        async with aclosing(
            cast(
                AsyncGenerator[Event, None],
                self.adapter.stream(
                    model=model,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                ),
            )
        ) as stream:
            try:
                async for event in stream:
                    if isinstance(event, Done):
                        usage = event.usage
                        self.budget.used += (
                            usage.total_tokens or usage.prompt_tokens + usage.completion_tokens
                            if usage is not None
                            else prompt_estimate
                            + count_tokens(
                                [event.final_message] if event.final_message is not None else [],
                                model,
                            )
                        )
                        accounted = True
                    yield event
            finally:
                if not accounted:
                    # Interrupted providers may not emit usage; retain at least
                    # prompt accounting instead of granting a free resumed turn.
                    self.budget.used += prompt_estimate


async def execute_mission_agents(
    *,
    store: MissionStore,
    mission_id: str,
    cwd: Path,
    agent_factory: Callable[[Mission, MissionFeature], Agent],
    max_features: int = 5,
    max_worker_steps: int = 20,
    timeout_seconds: float = 300.0,
) -> MissionBurstResult:
    """Execute workers and independent assertion commands with persisted sessions.

    Token limits prevent further requests once exhausted. Provider usage and
    prompt estimates can differ, so a completed in-flight request may exceed the
    remaining token allowance. Wall time and per-worker step limits are enforced.
    """
    if (
        max_features < 1
        or max_worker_steps < 1
        or not math.isfinite(timeout_seconds)
        or timeout_seconds <= 0
    ):
        raise ValueError("mission execution limits must be positive")
    mission = store.load_mission(mission_id)
    if mission.status not in {"approved", "running", "blocked"}:
        raise ValueError("agent execution requires an approved, running, or blocked mission")
    if mission.execution_mode == "simulation":
        raise ValueError("replan simulated work before executing it with real workers")
    contract = store.load_contract_for_mission(mission_id)
    if not contract.assertions or any(not assertion.command for assertion in contract.assertions):
        raise ValueError("every assertion needs an explicit argv command before agent execution")
    features_by_id = {feature.id: feature for feature in store.list_features(mission_id=mission_id)}
    for assertion in contract.assertions:
        if not assertion.covered_by_features:
            raise ValueError(
                f"assertion {assertion.id!r} must reference at least one mission feature"
            )
        unknown = set(assertion.covered_by_features) - features_by_id.keys()
        if unknown:
            raise ValueError(
                f"assertion {assertion.id!r} references unknown mission feature IDs: "
                + ", ".join(sorted(unknown))
            )
        milestone_ids = {
            features_by_id[feature_id].milestone_id for feature_id in assertion.covered_by_features
        }
        if len(milestone_ids) > 1:
            raise ValueError(
                f"assertion {assertion.id!r} spans multiple milestones; "
                "split into milestone-local assertions before agent execution"
            )
    if mission.budget_tokens is not None and mission.budget_tokens < 1:
        raise ValueError("mission token budget must be positive")
    if mission.budget_runtime_minutes is not None:
        remaining_time = mission.budget_runtime_minutes * 60 - mission.runtime_seconds_used
        if remaining_time <= 0:
            return MissionBurstResult("paused", mission_id, 0, "runtime_budget", ())
        timeout_seconds = min(timeout_seconds, remaining_time)
    if mission.budget_tokens is not None and mission.tokens_used >= mission.budget_tokens:
        return MissionBurstResult("paused", mission_id, 0, "token_budget", ())

    # Reuse kernel-backed locks, in a separate namespace from scheduled jobs.
    locks = SchedulerStore(root=store.root / "execution-locks")
    if not locks.acquire_job_lock(mission_id):
        return MissionBurstResult("blocked", mission_id, 0, "already_running", ())
    started = time.monotonic()
    budget = _TokenBudget(mission.budget_tokens, mission.tokens_used)
    initial_runtime = mission.runtime_seconds_used
    steps: list[MissionLoopStep] = []
    workers_run = 0

    def result(status: str, reason: str) -> MissionBurstResult:
        return MissionBurstResult(status, mission_id, len(steps), reason, tuple(steps))

    try:
        store.update_mission(replace(mission, execution_mode="agent"))
        while True:
            mission = store.load_mission(mission_id)
            if mission.status == "completed":
                return result("completed", "mission_completed")
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                return result("paused", "runtime_budget")
            if budget.limit is not None and budget.used >= budget.limit:
                return result("paused", "token_budget")
            if workers_run >= max_features:
                # Verification may still close the milestone after the last worker.
                pending = [
                    f
                    for f in store.list_features(
                        mission_id=mission_id, milestone_id=mission.current_milestone_id
                    )
                    if f.status not in {"completed", "validated"}
                ]
                if pending:
                    return result("paused", "max_features")
            dispatch = execute_next_mission_feature(store=store, mission_id=mission_id)
            if dispatch.status == "ready_for_validation":
                validation = validate_mission_milestone(
                    store=store,
                    mission_id=mission_id,
                    milestone_id=dispatch.milestone_id,
                    cwd=cwd,
                    contract=contract,
                    timeout_seconds=remaining,
                )
                steps.append(
                    MissionLoopStep(
                        "validate",
                        validation.status,
                        validation.message,
                        milestone_id=validation.milestone_id,
                        run_id=validation.run_id,
                    )
                )
                if validation.status == "failed":
                    return result("blocked", "verification_failed")
                continue
            if not dispatch.feature_id:
                return result(dispatch.status, dispatch.status)
            feature = store.load_feature(dispatch.feature_id)
            if feature.status not in {"active", "handoff", "blocked"}:
                return result("blocked", "feature_not_ready")
            if workers_run >= max_features:
                return result("paused", "max_features")
            workers_run += 1
            role = resolve_mission_role_profile(mission=mission, role=feature.assigned_role)
            session_id = feature.worker_session_id or store.new_id("session", feature.title)
            store.update_feature(replace(feature, status="active", worker_session_id=session_id))
            run = MissionRun(
                id=store.new_id("run", feature.title),
                mission_id=mission_id,
                role=role.role,
                role_model=role.model,
                status="running",
                related_feature_id=feature.id,
                related_milestone_id=feature.milestone_id,
                summary=f"Worker session {session_id}",
            )
            store.add_run(run)
            run_dir = store.runs_dir / run.id
            history = store.list_handoffs(mission_id=mission_id, feature_id=feature.id)
            findings = store.list_findings(mission_id=mission_id, milestone_id=feature.milestone_id)
            prompt = (
                f"Mission: {mission.goal}\nFeature: {feature.title}\n{feature.summary}\n"
                f"Role: {role.brief}\nTarget files: {json.dumps(feature.target_files)}\n"
                "Inspect the workspace and required tools before implementing this feature. "
                "Run relevant public checks and summarize actual changes and remaining issues. "
                "The mission validator will run the approved assertion commands independently.\n"
                f"Assertions: {json.dumps([a.to_dict() for a in contract.assertions if feature.id in a.covered_by_features])}\n"
                f"Recent handoffs: {json.dumps([h.to_dict() for h in history[-3:]])}\n"
                f"Findings: {json.dumps([f.to_dict() for f in findings[-5:]])}"
            )
            summary, failure = "", ""
            done = False
            agent: Agent | None = None
            original_adapters: dict[str, Adapter] = {}
            try:
                agent = agent_factory(mission, feature)
                original_adapters = agent.adapters
                agent.adapters = {
                    name: cast(Adapter, _BudgetAdapter(adapter, budget))
                    for name, adapter in original_adapters.items()
                }
                async with asyncio.timeout(remaining):
                    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as log:
                        async with aclosing(
                            agent.run(
                                RunRequest(
                                    prompt=prompt,
                                    session_id=session_id,
                                    max_steps=max_worker_steps,
                                    require_tool_use=True,
                                )
                            )
                        ) as stream:
                            async for event in stream:
                                log.write(event.model_dump_json() + "\n")
                                log.flush()
                                if isinstance(event, ErrorEvent):
                                    failure = event.error
                                elif isinstance(event, Done):
                                    done = True
                                    failure = ""
                                    summary = (
                                        (event.final_message.content or "")
                                        if event.final_message
                                        else ""
                                    )
                                elif (
                                    isinstance(event, Verification) and not event.result.can_finish
                                ):
                                    failure = event.result.reason
                if not done or failure:
                    raise RuntimeError(failure or "Worker ended without a completion event")
            except (Exception, asyncio.CancelledError) as exc:
                failure = str(exc) or type(exc).__name__
                store.update_feature(replace(store.load_feature(feature.id), status="blocked"))
                store.add_handoff(
                    MissionHandoff(
                        id=store.new_id("handoff", feature.title),
                        mission_id=mission_id,
                        feature_id=feature.id,
                        role=role.role,
                        role_model=role.model,
                        completed_work=summary or "Worker interrupted before verified completion.",
                        remaining_work=feature.summary,
                        known_issues=(failure,),
                        next_recommendation=f"Resume worker session {session_id}; inspect {run_dir / 'events.jsonl'}.",
                    )
                )
                store.add_run(replace(run, status="failed", summary=failure))
                store.update_mission(replace(store.load_mission(mission_id), status="blocked"))
                steps.append(
                    MissionLoopStep(
                        "worker", "failed", failure, feature.milestone_id, feature.id, run.id
                    )
                )
                if isinstance(exc, asyncio.CancelledError):
                    raise
                return result(
                    "paused" if isinstance(exc, TimeoutError) else "blocked", "worker_interrupted"
                )
            finally:
                if agent is not None:
                    agent.adapters = original_adapters
                store.update_mission(
                    replace(
                        store.load_mission(mission_id),
                        tokens_used=budget.used,
                        runtime_seconds_used=initial_runtime + time.monotonic() - started,
                    )
                )
            complete_mission_feature(
                store=store,
                mission_id=mission_id,
                feature_id=feature.id,
                completed_work=summary or "Worker finished; independent validation is pending.",
            )
            store.update_mission(
                replace(
                    store.load_mission(mission_id),
                    status="running",
                    execution_mode="agent",
                    current_milestone_id=feature.milestone_id,
                )
            )
            store.add_run(replace(run, status="completed", summary=summary))
            steps.append(
                MissionLoopStep(
                    "worker", "completed", summary, feature.milestone_id, feature.id, run.id
                )
            )
    finally:
        try:
            store.update_mission(
                replace(
                    store.load_mission(mission_id),
                    tokens_used=budget.used,
                    runtime_seconds_used=initial_runtime + time.monotonic() - started,
                )
            )
        finally:
            locks.release_job_lock(mission_id)


__all__ = ["execute_mission_agents"]
