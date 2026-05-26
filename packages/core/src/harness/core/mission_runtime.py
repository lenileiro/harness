from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

from harness.core.mission_models import (
    Milestone,
    Mission,
    MissionFeature,
    MissionHandoff,
    MissionRun,
)
from harness.core.mission_roles import resolve_mission_role_profile
from harness.core.mission_store import MissionStore
from harness.core.mission_validator import validate_mission_milestone

_DONE_FEATURE_STATUSES = {"completed", "validated"}
_ACTIVE_FEATURE_STATUSES = {"active", "handoff", "blocked"}


@dataclass(frozen=True, slots=True)
class MissionExecutionResult:
    status: str
    mission_id: str
    message: str
    milestone_id: str = ""
    feature_id: str = ""
    run_id: str = ""
    handoff_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "status": self.status,
            "mission_id": self.mission_id,
            "message": self.message,
            "milestone_id": self.milestone_id,
            "feature_id": self.feature_id,
            "run_id": self.run_id,
            "handoff_id": self.handoff_id,
        }


@dataclass(frozen=True, slots=True)
class MissionLoopStep:
    kind: str
    status: str
    message: str
    milestone_id: str = ""
    feature_id: str = ""
    run_id: str = ""
    handoff_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "status": self.status,
            "message": self.message,
            "milestone_id": self.milestone_id,
            "feature_id": self.feature_id,
            "run_id": self.run_id,
            "handoff_id": self.handoff_id,
        }


@dataclass(frozen=True, slots=True)
class MissionMilestoneExecutionResult:
    status: str
    mission_id: str
    milestone_id: str
    steps_run: int
    stop_reason: str
    steps: tuple[MissionLoopStep, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "mission_id": self.mission_id,
            "milestone_id": self.milestone_id,
            "steps_run": self.steps_run,
            "stop_reason": self.stop_reason,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class MissionBurstResult:
    status: str
    mission_id: str
    steps_run: int
    stop_reason: str
    steps: tuple[MissionLoopStep, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "mission_id": self.mission_id,
            "steps_run": self.steps_run,
            "stop_reason": self.stop_reason,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True, slots=True)
class MissionScheduledRunRecord:
    id: str
    status: str
    stop_reason: str
    steps_run: int
    mission_id: str
    cwd: str
    created_at: str
    steps: tuple[MissionLoopStep, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "steps_run": self.steps_run,
            "mission_id": self.mission_id,
            "cwd": self.cwd,
            "created_at": self.created_at,
            "steps": [step.to_dict() for step in self.steps],
        }


def _sorted_milestones(store: MissionStore, mission_id: str) -> list[Milestone]:
    return sorted(
        store.list_milestones(mission_id=mission_id),
        key=lambda item: (item.order, item.created_at, item.id),
    )


def _sorted_features(
    store: MissionStore, mission_id: str, milestone_id: str
) -> list[MissionFeature]:
    return sorted(
        store.list_features(mission_id=mission_id, milestone_id=milestone_id),
        key=lambda item: (item.created_at, item.id),
    )


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _refresh_mission_and_milestone(
    *,
    store: MissionStore,
    mission: Mission,
    milestones: list[Milestone],
) -> Mission:
    current = mission
    pending_or_active = [item for item in milestones if item.status != "completed"]
    if not pending_or_active:
        completed = replace(current, status="completed", current_milestone_id="")
        store.update_mission(completed)
        return completed
    next_milestone = pending_or_active[0]
    if current.current_milestone_id != next_milestone.id or current.status != "running":
        current = replace(current, current_milestone_id=next_milestone.id, status="running")
        store.update_mission(current)
    return current


def execute_next_mission_feature(*, store: MissionStore, mission_id: str) -> MissionExecutionResult:
    mission = store.load_mission(mission_id)
    if mission.status not in {"approved", "running", "blocked"}:
        raise ValueError("mission execution requires an approved, running, or blocked mission")

    milestones = _sorted_milestones(store, mission_id)
    if not milestones:
        return MissionExecutionResult(
            status="no_work",
            mission_id=mission_id,
            message="Mission has no milestones to execute.",
        )

    mission = _refresh_mission_and_milestone(store=store, mission=mission, milestones=milestones)
    if mission.status == "completed":
        return MissionExecutionResult(
            status="completed",
            mission_id=mission_id,
            message="Mission is already complete.",
        )

    milestone = store.load_milestone(mission.current_milestone_id)
    features = _sorted_features(store, mission_id, milestone.id)
    if not features:
        completed_milestone = replace(milestone, status="completed")
        store.update_milestone(completed_milestone)
        mission = _refresh_mission_and_milestone(
            store=store, mission=mission, milestones=_sorted_milestones(store, mission_id)
        )
        return MissionExecutionResult(
            status=mission.status,
            mission_id=mission_id,
            message="Current milestone had no features and was completed automatically.",
            milestone_id=completed_milestone.id,
        )

    active = [item for item in features if item.status in _ACTIVE_FEATURE_STATUSES]
    if active:
        current = active[0]
        return MissionExecutionResult(
            status="blocked",
            mission_id=mission_id,
            message="A feature is already active and must be completed before dispatching another one.",
            milestone_id=milestone.id,
            feature_id=current.id,
        )

    features_by_id = {item.id: item for item in features}
    ready_feature: MissionFeature | None = None
    blocked_messages: list[str] = []
    for feature in features:
        if feature.status in _DONE_FEATURE_STATUSES:
            continue
        unmet = [
            dependency
            for dependency in feature.depends_on
            if features_by_id.get(dependency) is None
            or features_by_id[dependency].status not in _DONE_FEATURE_STATUSES
        ]
        if unmet:
            blocked_messages.append(f"{feature.id} waits on {', '.join(unmet)}")
            continue
        ready_feature = feature
        break

    if ready_feature is None:
        if features and all(item.status in _DONE_FEATURE_STATUSES for item in features):
            return MissionExecutionResult(
                status="ready_for_validation",
                mission_id=mission_id,
                message="All features in the current milestone are complete. Run mission validation before advancing.",
                milestone_id=milestone.id,
            )
        return MissionExecutionResult(
            status="blocked",
            mission_id=mission_id,
            message="No ready feature could be dispatched. " + "; ".join(blocked_messages),
            milestone_id=milestone.id,
        )

    if milestone.status == "pending":
        milestone = replace(milestone, status="active")
        store.update_milestone(milestone)

    dispatched_feature = replace(ready_feature, status="handoff")
    store.update_feature(dispatched_feature)
    worker_profile = resolve_mission_role_profile(
        mission=mission,
        role=dispatched_feature.assigned_role or "worker",
    )
    planner_profile = resolve_mission_role_profile(mission=mission, role="planner")
    run = MissionRun(
        id=store.new_id("run", dispatched_feature.title),
        mission_id=mission_id,
        role=worker_profile.role,
        role_model=worker_profile.model,
        status="completed",
        summary=(
            f"Dispatched feature '{dispatched_feature.title}' for mission execution. "
            "Use the persisted handoff as the bounded work brief."
        ),
        related_feature_id=dispatched_feature.id,
        related_milestone_id=milestone.id,
    )
    handoff = MissionHandoff(
        id=store.new_id("handoff", dispatched_feature.title),
        mission_id=mission_id,
        feature_id=dispatched_feature.id,
        role=planner_profile.role,
        role_model=planner_profile.model,
        completed_work=(
            f"Prepared the execution brief for '{dispatched_feature.title}' and activated it "
            f"under milestone '{milestone.title}'."
        ),
        remaining_work=dispatched_feature.summary,
        next_recommendation=(
            f"{worker_profile.brief} "
            "After implementation, complete the feature with a concrete handoff."
        ),
        confidence=0.8,
    )
    store.add_run(run)
    store.add_handoff(handoff)
    return MissionExecutionResult(
        status="dispatched",
        mission_id=mission_id,
        message=f"Dispatched feature '{dispatched_feature.title}'.",
        milestone_id=milestone.id,
        feature_id=dispatched_feature.id,
        run_id=run.id,
        handoff_id=handoff.id,
    )


def complete_mission_feature(
    *,
    store: MissionStore,
    mission_id: str,
    feature_id: str,
    completed_work: str,
    remaining_work: str = "",
    known_issues: tuple[str, ...] = (),
    next_recommendation: str = "",
    confidence: float = 0.9,
    role: str = "worker",
) -> MissionExecutionResult:
    feature = store.load_feature(feature_id)
    if feature.mission_id != mission_id:
        raise ValueError(f"feature {feature_id!r} does not belong to mission {mission_id!r}")
    if feature.status not in {"active", "handoff", "blocked"}:
        raise ValueError("feature completion requires an active or handed-off feature")

    completed_feature = replace(feature, status="completed")
    store.update_feature(completed_feature)
    mission = store.load_mission(mission_id)
    role_profile = resolve_mission_role_profile(mission=mission, role=role)
    handoff = MissionHandoff(
        id=store.new_id("handoff", completed_feature.title),
        mission_id=mission_id,
        feature_id=feature_id,
        role=role_profile.role,
        role_model=role_profile.model,
        completed_work=completed_work.strip(),
        remaining_work=remaining_work.strip(),
        known_issues=tuple(item.strip() for item in known_issues if item.strip()),
        next_recommendation=next_recommendation.strip() or role_profile.brief,
        confidence=confidence,
    )
    run = MissionRun(
        id=store.new_id("run", completed_feature.title),
        mission_id=mission_id,
        role=role_profile.role,
        role_model=role_profile.model,
        status="completed",
        summary=f"Completed feature '{completed_feature.title}'.",
        related_feature_id=feature_id,
        related_milestone_id=completed_feature.milestone_id,
    )
    store.add_handoff(handoff)
    store.add_run(run)
    return MissionExecutionResult(
        status="recorded",
        mission_id=mission_id,
        message=f"Recorded completion for feature '{completed_feature.title}'.",
        milestone_id=completed_feature.milestone_id,
        feature_id=feature_id,
        run_id=run.id,
        handoff_id=handoff.id,
    )


def _as_step(kind: str, result: MissionExecutionResult) -> MissionLoopStep:
    return MissionLoopStep(
        kind=kind,
        status=result.status,
        message=result.message,
        milestone_id=result.milestone_id,
        feature_id=result.feature_id,
        run_id=result.run_id,
        handoff_id=result.handoff_id,
    )


def execute_mission_milestone(
    *,
    store: MissionStore,
    mission_id: str,
    milestone_id: str | None = None,
    max_steps: int = 20,
    auto_complete: bool = False,
) -> MissionMilestoneExecutionResult:
    if max_steps < 1:
        raise ValueError("mission milestone execution requires max_steps >= 1")
    mission = store.load_mission(mission_id)
    target_milestone_id = milestone_id or mission.current_milestone_id
    if not target_milestone_id:
        raise ValueError(
            "mission milestone execution requires a current milestone or an explicit milestone"
        )
    milestone = store.load_milestone(target_milestone_id)
    if milestone.mission_id != mission_id:
        raise ValueError(
            f"milestone {target_milestone_id!r} does not belong to mission {mission_id!r}"
        )

    steps: list[MissionLoopStep] = []
    for _ in range(max_steps):
        dispatch = execute_next_mission_feature(store=store, mission_id=mission_id)
        if dispatch.milestone_id and dispatch.milestone_id != target_milestone_id:
            stop_reason = "advanced_to_next_milestone"
            return MissionMilestoneExecutionResult(
                status="completed",
                mission_id=mission_id,
                milestone_id=target_milestone_id,
                steps_run=len(steps),
                stop_reason=stop_reason,
                steps=tuple(steps),
            )
        if dispatch.status in {"no_work", "completed"}:
            steps.append(_as_step("dispatch", dispatch))
            return MissionMilestoneExecutionResult(
                status=dispatch.status,
                mission_id=mission_id,
                milestone_id=target_milestone_id,
                steps_run=len(steps),
                stop_reason=dispatch.status,
                steps=tuple(steps),
            )
        if dispatch.status == "dispatched":
            steps.append(_as_step("dispatch", dispatch))
            if not auto_complete:
                return MissionMilestoneExecutionResult(
                    status="paused",
                    mission_id=mission_id,
                    milestone_id=target_milestone_id,
                    steps_run=len(steps),
                    stop_reason="feature_dispatched",
                    steps=tuple(steps),
                )
            completion = complete_mission_feature(
                store=store,
                mission_id=mission_id,
                feature_id=dispatch.feature_id,
                completed_work=f"Auto-completed '{dispatch.feature_id}' for bounded mission execution.",
                next_recommendation="Proceed to the next feature or validation step.",
                role="worker",
            )
            steps.append(_as_step("complete", completion))
            continue
        if dispatch.status == "ready_for_validation":
            validation = validate_mission_milestone(
                store=store,
                mission_id=mission_id,
                milestone_id=target_milestone_id,
            )
            steps.append(
                MissionLoopStep(
                    kind="validate",
                    status=validation.status,
                    message=validation.message,
                    milestone_id=validation.milestone_id,
                    run_id=validation.run_id,
                )
            )
            if validation.status == "failed":
                return MissionMilestoneExecutionResult(
                    status="blocked",
                    mission_id=mission_id,
                    milestone_id=target_milestone_id,
                    steps_run=len(steps),
                    stop_reason="validation_failed",
                    steps=tuple(steps),
                )
            return MissionMilestoneExecutionResult(
                status="completed",
                mission_id=mission_id,
                milestone_id=target_milestone_id,
                steps_run=len(steps),
                stop_reason="validation_passed",
                steps=tuple(steps),
            )
        steps.append(_as_step("dispatch", dispatch))
        return MissionMilestoneExecutionResult(
            status="blocked",
            mission_id=mission_id,
            milestone_id=target_milestone_id,
            steps_run=len(steps),
            stop_reason=dispatch.status,
            steps=tuple(steps),
        )

    return MissionMilestoneExecutionResult(
        status="paused",
        mission_id=mission_id,
        milestone_id=target_milestone_id,
        steps_run=len(steps),
        stop_reason="max_steps",
        steps=tuple(steps),
    )


def execute_mission_burst(
    *,
    store: MissionStore,
    mission_id: str,
    max_steps: int = 50,
    auto_complete: bool = False,
) -> MissionBurstResult:
    if max_steps < 1:
        raise ValueError("mission burst execution requires max_steps >= 1")
    mission = store.load_mission(mission_id)
    steps: list[MissionLoopStep] = []
    while len(steps) < max_steps:
        mission = store.load_mission(mission_id)
        if mission.status == "completed":
            return MissionBurstResult(
                status="completed",
                mission_id=mission_id,
                steps_run=len(steps),
                stop_reason="mission_completed",
                steps=tuple(steps),
            )
        target_milestone_id = mission.current_milestone_id
        if not target_milestone_id:
            return MissionBurstResult(
                status="no_work",
                mission_id=mission_id,
                steps_run=len(steps),
                stop_reason="no_current_milestone",
                steps=tuple(steps),
            )
        remaining_steps = max_steps - len(steps)
        milestone_result = execute_mission_milestone(
            store=store,
            mission_id=mission_id,
            milestone_id=target_milestone_id,
            max_steps=remaining_steps,
            auto_complete=auto_complete,
        )
        steps.extend(milestone_result.steps)
        if milestone_result.status in {"paused", "blocked", "no_work"}:
            return MissionBurstResult(
                status=milestone_result.status,
                mission_id=mission_id,
                steps_run=len(steps),
                stop_reason=milestone_result.stop_reason,
                steps=tuple(steps),
            )
        mission = store.load_mission(mission_id)
        if mission.status == "completed":
            return MissionBurstResult(
                status="completed",
                mission_id=mission_id,
                steps_run=len(steps),
                stop_reason="mission_completed",
                steps=tuple(steps),
            )
    return MissionBurstResult(
        status="paused",
        mission_id=mission_id,
        steps_run=len(steps),
        stop_reason="max_steps",
        steps=tuple(steps),
    )


def write_mission_scheduled_run_record(
    *,
    store: MissionStore,
    cwd: Path,
    result: MissionBurstResult,
) -> Path:
    store.ensure_layout()
    run_id = store.new_id("mission-run", result.mission_id)
    target = store.runs_dir / run_id
    target.mkdir(parents=True, exist_ok=True)
    record = MissionScheduledRunRecord(
        id=run_id,
        status=result.status,
        stop_reason=result.stop_reason,
        steps_run=result.steps_run,
        mission_id=result.mission_id,
        cwd=str(cwd),
        created_at=_utcnow(),
        steps=result.steps,
    )
    (target / "run.json").write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    lines = [
        f"# Mission Scheduled Run {record.id}",
        "",
        f"- mission_id: `{record.mission_id}`",
        f"- status: `{record.status}`",
        f"- stop_reason: `{record.stop_reason}`",
        f"- steps_run: `{record.steps_run}`",
        f"- cwd: `{record.cwd}`",
        "",
        "## Steps",
    ]
    for index, step in enumerate(record.steps, start=1):
        lines.append(f"{index}. [{step.kind}] {step.status} {step.message}")
    (target / "RUN.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return target


def run_scheduled_mission_burst(
    *,
    store: MissionStore,
    cwd: Path,
    mission_id: str,
    max_steps: int = 20,
    auto_complete: bool = False,
) -> tuple[MissionBurstResult, Path]:
    result = execute_mission_burst(
        store=store,
        mission_id=mission_id,
        max_steps=max_steps,
        auto_complete=auto_complete,
    )
    record_dir = write_mission_scheduled_run_record(
        store=store,
        cwd=cwd,
        result=result,
    )
    return result, record_dir


__all__ = [
    "MissionBurstResult",
    "MissionExecutionResult",
    "MissionLoopStep",
    "MissionMilestoneExecutionResult",
    "MissionScheduledRunRecord",
    "complete_mission_feature",
    "execute_mission_burst",
    "execute_mission_milestone",
    "execute_next_mission_feature",
    "run_scheduled_mission_burst",
    "write_mission_scheduled_run_record",
]
