from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from contextlib import suppress
from dataclasses import dataclass, replace
from pathlib import Path

from harness.core.mission_models import (
    Mission,
    MissionFeature,
    MissionFinding,
    MissionRun,
    ValidationAssertion,
    ValidationContract,
)
from harness.core.mission_roles import resolve_mission_role_profile
from harness.core.mission_store import MissionStore

_DONE_FEATURE_STATUSES = {"completed", "validated"}


def run_assertion_command(
    *,
    assertion: ValidationAssertion,
    cwd: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    """Execute an approved argv contract outside the worker's claimed result."""
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output_path = evidence_dir / "output.txt"
    result: dict[str, object] = {
        "assertion_id": assertion.id,
        "command": list(assertion.command),
        "status": "blocked",
        "exit_code": None,
        "output_path": str(output_path),
    }
    try:
        if not assertion.command or not assertion.command[0].strip():
            result["reason"] = "No executable command configured for this assertion"
        elif assertion.timeout_seconds <= 0:
            result["reason"] = "Assertion timeout must be positive"
        else:
            try:
                with output_path.open("w", encoding="utf-8") as output:
                    process = subprocess.Popen(
                        list(assertion.command),
                        cwd=cwd,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,
                        start_new_session=os.name == "posix",
                    )
                    try:
                        code = process.wait(timeout=assertion.timeout_seconds)
                        result.update(status="passed" if code == 0 else "failed", exit_code=code)
                    except subprocess.TimeoutExpired:
                        result.update(status="failed", reason="Assertion timed out", timed_out=True)
                    except BaseException as exc:
                        result.update(
                            status="failed",
                            reason=f"Assertion interrupted: {type(exc).__name__}",
                            interrupted=True,
                        )
                        raise
                    finally:
                        # A successful parent may leave descendants running. Reap
                        # our command and stop its entire group on every exit path,
                        # including KeyboardInterrupt and timeout.
                        with suppress(ProcessLookupError, PermissionError):
                            if os.name == "posix":
                                os.killpg(process.pid, signal.SIGKILL)
                            elif process.poll() is None:
                                process.kill()
                        process.wait()
                        if result["exit_code"] is None:
                            result["exit_code"] = process.returncode
            except (OSError, ValueError) as exc:
                result.update(status="failed", reason=str(exc))
    finally:
        (evidence_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result


def _run_executable_assertions(
    *,
    store: MissionStore,
    mission_id: str,
    milestone_id: str,
    assertions: list[ValidationAssertion],
    cwd: Path,
    timeout_seconds: float | None = None,
) -> list[MissionFinding]:
    findings: list[MissionFinding] = []
    verification_id = store.new_id("verification", milestone_id)
    evidence_root = store.runs_dir / verification_id
    deadline = time.monotonic() + timeout_seconds if timeout_seconds is not None else None
    for index, assertion in enumerate(assertions):
        if deadline is not None:
            # Keep the per-command timeout valid after the enclosing budget ends.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                assertion = replace(assertion, command=())
            else:
                assertion = replace(
                    assertion, timeout_seconds=min(assertion.timeout_seconds, remaining)
                )
        result = run_assertion_command(
            assertion=assertion,
            cwd=cwd,
            evidence_dir=evidence_root / str(index),
        )
        if result["status"] == "passed":
            continue
        finding = MissionFinding(
            id=store.new_id("finding", assertion.title),
            mission_id=mission_id,
            milestone_id=milestone_id,
            source="executable-validator",
            severity="error",
            summary=f"Assertion '{assertion.title}' {result['status']}: {result.get('reason', 'command failed')}",
            recommended_fix=f"Resolve the assertion failure and rerun verification. Evidence: {evidence_root / str(index) / 'result.json'}",
        )
        store.add_finding(finding)
        findings.append(finding)
    if not assertions:
        finding = MissionFinding(
            id=store.new_id("finding", "missing assertions"),
            mission_id=mission_id,
            milestone_id=milestone_id,
            source="executable-validator",
            severity="error",
            summary="Milestone has no applicable executable assertions",
            recommended_fix="Configure an assertion command before running the milestone.",
        )
        store.add_finding(finding)
        findings.append(finding)
    return findings


@dataclass(frozen=True, slots=True)
class MissionValidationResult:
    status: str
    mission_id: str
    message: str
    milestone_id: str
    run_id: str
    scrutiny_run_id: str = ""
    behavior_run_id: str = ""
    findings_count: int = 0
    corrective_feature_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "mission_id": self.mission_id,
            "message": self.message,
            "milestone_id": self.milestone_id,
            "run_id": self.run_id,
            "scrutiny_run_id": self.scrutiny_run_id,
            "behavior_run_id": self.behavior_run_id,
            "findings_count": self.findings_count,
            "corrective_feature_ids": list(self.corrective_feature_ids),
        }


def _sorted_milestones(store: MissionStore, mission_id: str) -> list:
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


def _relevant_assertions(
    assertions: tuple[ValidationAssertion, ...], feature_ids: set[str]
) -> list[ValidationAssertion]:
    return [
        assertion
        for assertion in assertions
        if any(feature_id in feature_ids for feature_id in assertion.covered_by_features)
    ]


def _refresh_mission_after_validation(store: MissionStore, mission: Mission) -> Mission:
    milestones = _sorted_milestones(store, mission.id)
    remaining = [item for item in milestones if item.status != "completed"]
    if not remaining:
        completed = replace(mission, status="completed", current_milestone_id="")
        store.update_mission(completed)
        return completed
    next_milestone = remaining[0]
    updated = replace(mission, status="running", current_milestone_id=next_milestone.id)
    store.update_mission(updated)
    return updated


def _corrective_title(feature: MissionFeature) -> str:
    return f"Corrective: {feature.title}"


def _create_corrective_feature(
    *,
    store: MissionStore,
    mission_id: str,
    milestone_id: str,
    feature: MissionFeature,
    recommended_fix: str,
) -> str:
    existing = _sorted_features(store, mission_id, milestone_id)
    title = _corrective_title(feature)
    for candidate in existing:
        if candidate.title == title and candidate.status not in {"completed", "validated"}:
            return candidate.id
    corrective = MissionFeature(
        id=store.new_id("feature", title),
        mission_id=mission_id,
        milestone_id=milestone_id,
        title=title,
        summary=recommended_fix,
        status="pending",
        depends_on=(feature.id,),
        assigned_role="worker",
        target_files=feature.target_files,
        research_refs=("mission-validator:corrective", f"source-feature:{feature.id}"),
    )
    store.add_feature(corrective)
    return corrective.id


def _run_scrutiny_checks(
    *,
    store: MissionStore,
    mission_id: str,
    milestone_id: str,
    features: list[MissionFeature],
) -> tuple[list[MissionFinding], list[str]]:
    findings: list[MissionFinding] = []
    corrective_feature_ids: list[str] = []
    for feature in features:
        if feature.status in _DONE_FEATURE_STATUSES:
            continue
        recommended_fix = (
            f"Finish '{feature.title}' and update its worker handoff before milestone validation."
        )
        finding = MissionFinding(
            id=store.new_id("finding", feature.title),
            mission_id=mission_id,
            milestone_id=milestone_id,
            source="scrutiny-validator",
            severity="error",
            summary=f"Feature '{feature.title}' is not complete.",
            recommended_fix=recommended_fix,
        )
        store.add_finding(finding)
        findings.append(finding)
        corrective_feature_ids.append(
            _create_corrective_feature(
                store=store,
                mission_id=mission_id,
                milestone_id=milestone_id,
                feature=feature,
                recommended_fix=recommended_fix,
            )
        )
    return findings, corrective_feature_ids


def _run_behavior_checks(
    *,
    store: MissionStore,
    mission_id: str,
    milestone_id: str,
    assertions: list[ValidationAssertion],
    features_by_id: dict[str, MissionFeature],
) -> list[MissionFinding]:
    findings: list[MissionFinding] = []
    for assertion in assertions:
        related = [
            features_by_id[feature_id]
            for feature_id in assertion.covered_by_features
            if feature_id in features_by_id
        ]
        if not related:
            continue
        if any(feature.status not in _DONE_FEATURE_STATUSES for feature in related):
            recommended_fix = f"Complete the feature work linked to assertion '{assertion.title}' before validation can pass."
            finding = MissionFinding(
                id=store.new_id("finding", assertion.title),
                mission_id=mission_id,
                milestone_id=milestone_id,
                source="behavior-validator",
                severity="error",
                summary=f"Assertion '{assertion.title}' is not yet satisfied by completed feature work.",
                recommended_fix=recommended_fix,
            )
            store.add_finding(finding)
            findings.append(finding)
    return findings


def validate_mission_milestone(
    *,
    store: MissionStore,
    mission_id: str,
    milestone_id: str | None = None,
    cwd: Path | None = None,
    contract: ValidationContract | None = None,
    timeout_seconds: float | None = None,
) -> MissionValidationResult:
    mission = store.load_mission(mission_id)
    if mission.status not in {"approved", "running", "blocked"}:
        raise ValueError("mission validation requires an approved, running, or blocked mission")

    resolved_milestone_id = milestone_id or mission.current_milestone_id
    if not resolved_milestone_id:
        raise ValueError(
            "mission validation requires a current milestone or an explicit --milestone"
        )
    milestone = store.load_milestone(resolved_milestone_id)
    if milestone.mission_id != mission_id:
        raise ValueError(
            f"milestone {resolved_milestone_id!r} does not belong to mission {mission_id!r}"
        )

    features = _sorted_features(store, mission_id, milestone.id)
    contract = contract or store.load_contract_for_mission(mission_id)
    feature_ids = {feature.id for feature in features}
    relevant_assertions = _relevant_assertions(contract.assertions, feature_ids)
    features_by_id = {feature.id: feature for feature in features}
    scrutiny_findings, corrective_feature_ids = _run_scrutiny_checks(
        store=store,
        mission_id=mission_id,
        milestone_id=milestone.id,
        features=features,
    )
    behavior_findings = _run_behavior_checks(
        store=store,
        mission_id=mission_id,
        milestone_id=milestone.id,
        assertions=relevant_assertions,
        features_by_id=features_by_id,
    )
    if mission.execution_mode == "agent":
        behavior_findings.extend(
            _run_executable_assertions(
                store=store,
                mission_id=mission_id,
                milestone_id=milestone.id,
                assertions=relevant_assertions,
                cwd=(cwd or store.root.parent.parent).resolve(),
                timeout_seconds=timeout_seconds,
            )
        )
        if behavior_findings:
            # Keep the original bounded work resumable with concrete verifier evidence.
            for feature in features:
                if feature.status == "completed":
                    store.update_feature(replace(feature, status="blocked"))
    findings = [*scrutiny_findings, *behavior_findings]
    validator_profile = resolve_mission_role_profile(mission=mission, role="validator")
    scrutiny_run = MissionRun(
        id=store.new_id("run", f"scrutiny {milestone.title}"),
        mission_id=mission_id,
        role=validator_profile.role,
        role_model=validator_profile.model,
        status="failed" if scrutiny_findings else "completed",
        summary=(
            f"Scrutiny validation found {len(scrutiny_findings)} finding(s)."
            if scrutiny_findings
            else "Scrutiny validation passed."
        ),
        related_milestone_id=milestone.id,
    )
    behavior_run = MissionRun(
        id=store.new_id("run", f"behavior {milestone.title}"),
        mission_id=mission_id,
        role=validator_profile.role,
        role_model=validator_profile.model,
        status="failed" if behavior_findings else "completed",
        summary=(
            f"Behavior validation found {len(behavior_findings)} finding(s)."
            if behavior_findings
            else "Behavior validation passed."
        ),
        related_milestone_id=milestone.id,
    )
    store.add_run(scrutiny_run)
    store.add_run(behavior_run)

    if findings:
        blocked_milestone = replace(milestone, status="blocked")
        store.update_milestone(blocked_milestone)
        blocked_mission = replace(
            mission, status="blocked", current_milestone_id=blocked_milestone.id
        )
        store.update_mission(blocked_mission)
        return MissionValidationResult(
            status="failed",
            mission_id=mission_id,
            message="Mission validation found blocking issues and created corrective follow-up work.",
            milestone_id=blocked_milestone.id,
            run_id=scrutiny_run.id,
            scrutiny_run_id=scrutiny_run.id,
            behavior_run_id=behavior_run.id,
            findings_count=len(findings),
            corrective_feature_ids=tuple(corrective_feature_ids),
        )

    for feature in features:
        if feature.status == "completed":
            store.update_feature(replace(feature, status="validated"))
    completed_milestone = replace(milestone, status="completed")
    store.update_milestone(completed_milestone)
    refreshed_mission = _refresh_mission_after_validation(store, mission)
    return MissionValidationResult(
        status="completed" if refreshed_mission.status == "completed" else "passed",
        mission_id=mission_id,
        message="Mission validation passed and milestone assertions are satisfied.",
        milestone_id=completed_milestone.id,
        run_id=scrutiny_run.id,
        scrutiny_run_id=scrutiny_run.id,
        behavior_run_id=behavior_run.id,
        findings_count=0,
        corrective_feature_ids=(),
    )


__all__ = ["MissionValidationResult", "run_assertion_command", "validate_mission_milestone"]
