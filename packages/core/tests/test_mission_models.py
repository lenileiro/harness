from __future__ import annotations

from harness.core.mission_models import (
    Milestone,
    Mission,
    MissionFeature,
    MissionFinding,
    MissionHandoff,
    MissionRun,
    ValidationAssertion,
    ValidationContract,
)


def test_mission_round_trip_preserves_role_models_and_budget() -> None:
    mission = Mission(
        id="mission-demo",
        title="Mission demo",
        goal="Deliver a bounded multi-step feature safely.",
        status="planned",
        planner_model="gpt-planner",
        worker_model="gpt-worker",
        validator_model="gpt-validator",
        reporter_model="gpt-reporter",
        planner_brief="Plan before coding.",
        worker_brief="Implement the assigned slice.",
        validator_brief="Check the assertions independently.",
        reporter_brief="Summarize the current mission state.",
        budget_tokens=12345,
        budget_runtime_minutes=90,
        current_milestone_id="milestone-1",
    )

    loaded = Mission.from_dict(mission.to_dict())

    assert loaded.status == "planned"
    assert loaded.worker_model == "gpt-worker"
    assert loaded.worker_brief == "Implement the assigned slice."
    assert loaded.budget_tokens == 12345
    assert loaded.current_milestone_id == "milestone-1"


def test_validation_contract_round_trip_preserves_assertions() -> None:
    contract = ValidationContract(
        id="contract-demo",
        mission_id="mission-demo",
        summary="Assertions must exist before implementation.",
        assertions=(
            ValidationAssertion(
                id="assertion-1",
                contract_id="contract-demo",
                title="Login flow works",
                description="The primary sign-in flow completes end to end.",
                kind="behavior",
                verification_method="Run browser validation against /login.",
                covered_by_features=("feature-1",),
            ),
        ),
    )

    loaded = ValidationContract.from_dict(contract.to_dict())

    assert len(loaded.assertions) == 1
    assert loaded.assertions[0].kind == "behavior"
    assert loaded.assertions[0].covered_by_features == ("feature-1",)


def test_milestone_feature_handoff_finding_and_run_round_trip() -> None:
    milestone = Milestone(
        id="milestone-1",
        mission_id="mission-demo",
        title="Milestone 1",
        summary="Build the first validated slice.",
        status="active",
        order=1,
    )
    feature = MissionFeature(
        id="feature-1",
        mission_id="mission-demo",
        milestone_id="milestone-1",
        title="Implement login page",
        summary="Add a login screen and supporting behavior.",
        status="active",
        depends_on=("feature-0",),
        target_files=("app/login.py",),
    )
    handoff = MissionHandoff(
        id="handoff-1",
        mission_id="mission-demo",
        feature_id="feature-1",
        role="worker",
        role_model="gpt-worker",
        completed_work="Login form renders and submits.",
        remaining_work="Validator must confirm browser flow.",
        commands_run=("pytest tests/login",),
        exit_codes=("0",),
        known_issues=("Needs browser verification",),
        next_recommendation="Run milestone validation.",
        confidence=0.8,
    )
    finding = MissionFinding(
        id="finding-1",
        mission_id="mission-demo",
        milestone_id="milestone-1",
        source="behavior-validator",
        severity="warning",
        summary="The form succeeds, but missing empty-state copy.",
        recommended_fix="Add validation copy for empty submission.",
    )
    run = MissionRun(
        id="run-1",
        mission_id="mission-demo",
        role="validator",
        role_model="gpt-validator",
        status="completed",
        summary="Behavior checks completed with one warning.",
        related_feature_id="feature-1",
        related_milestone_id="milestone-1",
    )

    assert Milestone.from_dict(milestone.to_dict()).order == 1
    assert MissionFeature.from_dict(feature.to_dict()).depends_on == ("feature-0",)
    assert MissionHandoff.from_dict(handoff.to_dict()).known_issues == (
        "Needs browser verification",
    )
    assert MissionHandoff.from_dict(handoff.to_dict()).role_model == "gpt-worker"
    assert MissionFinding.from_dict(finding.to_dict()).severity == "warning"
    loaded_run = MissionRun.from_dict(run.to_dict())
    assert loaded_run.role == "validator"
    assert loaded_run.role_model == "gpt-validator"
