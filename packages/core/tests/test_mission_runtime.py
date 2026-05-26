from __future__ import annotations

from pathlib import Path

from harness.core.mission_models import Mission
from harness.core.mission_planner import (
    PlannedAssertionInput,
    PlannedFeatureInput,
    PlannedMilestoneInput,
    build_mission_plan,
)
from harness.core.mission_runtime import (
    complete_mission_feature,
    execute_mission_burst,
    execute_mission_milestone,
    execute_next_mission_feature,
    write_mission_scheduled_run_record,
)
from harness.core.mission_store import MissionStore, default_mission_root
from harness.core.mission_validator import validate_mission_milestone


def _seed_approved_mission(tmp_path: Path) -> tuple[MissionStore, str]:
    store = MissionStore(root=default_mission_root(tmp_path))
    mission = Mission(
        id="mission-runtime",
        title="Runtime mission",
        goal="Dispatch and complete features.",
        status="draft",
        planner_model="gpt-planner",
        worker_model="gpt-worker",
        validator_model="gpt-validator",
        planner_brief="Plan the next feature.",
        worker_brief="Implement the feature and leave a handoff.",
        validator_brief="Validate assertions independently.",
    )
    store.add_mission(mission)
    planned = build_mission_plan(
        store=store,
        mission=mission,
        contract_summary="Mission runtime assertions.",
        milestones=(PlannedMilestoneInput("m1", "Milestone 1", "First milestone."),),
        assertions=(
            PlannedAssertionInput(
                "a1",
                "Feature 1 works",
                "First feature is implemented.",
                "contract",
                "Inspect output.",
            ),
            PlannedAssertionInput(
                "a2",
                "Feature 2 works",
                "Second feature is implemented.",
                "contract",
                "Inspect output.",
            ),
        ),
        features=(
            PlannedFeatureInput(
                "f1",
                "m1",
                "Feature 1",
                "Implement the first feature.",
                "worker",
                ("app/one.py",),
                (),
                ("a1",),
            ),
            PlannedFeatureInput(
                "f2",
                "m1",
                "Feature 2",
                "Implement the second feature.",
                "worker",
                ("app/two.py",),
                ("f1",),
                ("a2",),
            ),
        ),
    )
    store.update_mission(planned.mission)
    for milestone in planned.milestones:
        store.add_milestone(milestone)
    for feature in planned.features:
        store.add_feature(feature)
    store.add_contract(planned.contract)
    store.update_mission(Mission.from_dict({**planned.mission.to_dict(), "status": "approved"}))
    return store, planned.mission.id


def _seed_two_milestone_mission(tmp_path: Path) -> tuple[MissionStore, str]:
    store = MissionStore(root=default_mission_root(tmp_path))
    mission = Mission(
        id="mission-burst",
        title="Burst mission",
        goal="Run through two milestones.",
        status="draft",
    )
    store.add_mission(mission)
    planned = build_mission_plan(
        store=store,
        mission=mission,
        contract_summary="Burst mission assertions.",
        milestones=(
            PlannedMilestoneInput("m1", "Milestone 1", "First milestone."),
            PlannedMilestoneInput("m2", "Milestone 2", "Second milestone."),
        ),
        assertions=(
            PlannedAssertionInput(
                "a1",
                "Feature 1 works",
                "First feature is implemented.",
                "contract",
                "Inspect output.",
            ),
            PlannedAssertionInput(
                "a2",
                "Feature 2 works",
                "Second feature is implemented.",
                "contract",
                "Inspect output.",
            ),
        ),
        features=(
            PlannedFeatureInput(
                "f1",
                "m1",
                "Feature 1",
                "Implement the first feature.",
                "worker",
                ("app/one.py",),
                (),
                ("a1",),
            ),
            PlannedFeatureInput(
                "f2",
                "m2",
                "Feature 2",
                "Implement the second feature.",
                "worker",
                ("app/two.py",),
                (),
                ("a2",),
            ),
        ),
    )
    store.update_mission(planned.mission)
    for milestone in planned.milestones:
        store.add_milestone(milestone)
    for feature in planned.features:
        store.add_feature(feature)
    store.add_contract(planned.contract)
    store.update_mission(Mission.from_dict({**planned.mission.to_dict(), "status": "approved"}))
    return store, planned.mission.id


def test_execute_next_dispatches_ready_feature_and_persists_artifacts(tmp_path: Path) -> None:
    store, mission_id = _seed_approved_mission(tmp_path)

    result = execute_next_mission_feature(store=store, mission_id=mission_id)

    assert result.status == "dispatched"
    feature = store.load_feature(result.feature_id)
    mission = store.load_mission(mission_id)
    milestone = store.load_milestone(result.milestone_id)
    run = store.load_run(result.run_id)
    handoff = store.load_handoff(result.handoff_id)

    assert feature.status == "handoff"
    assert mission.status == "running"
    assert mission.current_milestone_id == milestone.id
    assert milestone.status == "active"
    assert run.related_feature_id == feature.id
    assert run.role_model == "gpt-worker"
    assert handoff.feature_id == feature.id
    assert handoff.role_model == "gpt-planner"
    assert "Prepared the execution brief" in handoff.completed_work
    assert "Implement the feature and leave a handoff." in handoff.next_recommendation


def test_complete_feature_advances_dependency_chain_and_completes_mission(tmp_path: Path) -> None:
    store, mission_id = _seed_approved_mission(tmp_path)

    first_dispatch = execute_next_mission_feature(store=store, mission_id=mission_id)
    first_complete = complete_mission_feature(
        store=store,
        mission_id=mission_id,
        feature_id=first_dispatch.feature_id,
        completed_work="Implemented the first feature.",
        next_recommendation="Proceed to the dependent feature.",
    )
    assert first_complete.status == "recorded"

    second_dispatch = execute_next_mission_feature(store=store, mission_id=mission_id)
    assert second_dispatch.status == "dispatched"
    second_feature = store.load_feature(second_dispatch.feature_id)
    assert second_feature.title == "Feature 2"

    second_complete = complete_mission_feature(
        store=store,
        mission_id=mission_id,
        feature_id=second_dispatch.feature_id,
        completed_work="Implemented the second feature.",
    )
    assert second_complete.status == "recorded"
    waiting = execute_next_mission_feature(store=store, mission_id=mission_id)
    assert waiting.status == "ready_for_validation"

    validated = validate_mission_milestone(store=store, mission_id=mission_id)
    assert validated.status == "completed"
    mission = store.load_mission(mission_id)
    milestone = store.load_milestone(second_complete.milestone_id)
    assert mission.status == "completed"
    assert mission.current_milestone_id == ""
    assert milestone.status == "completed"


def test_validate_mission_milestone_creates_findings_and_corrective_features(
    tmp_path: Path,
) -> None:
    store, mission_id = _seed_approved_mission(tmp_path)

    first_dispatch = execute_next_mission_feature(store=store, mission_id=mission_id)
    assert first_dispatch.status == "dispatched"

    validation = validate_mission_milestone(store=store, mission_id=mission_id)

    assert validation.status == "failed"
    assert validation.findings_count >= 1
    assert validation.corrective_feature_ids
    assert validation.scrutiny_run_id
    assert validation.behavior_run_id
    mission = store.load_mission(mission_id)
    milestone = store.load_milestone(validation.milestone_id)
    corrective = store.load_feature(validation.corrective_feature_ids[0])
    findings = store.list_findings(mission_id=mission_id, milestone_id=validation.milestone_id)

    assert mission.status == "blocked"
    assert milestone.status == "blocked"
    assert corrective.title.startswith("Corrective:")
    assert findings
    scrutiny_run = store.load_run(validation.scrutiny_run_id)
    behavior_run = store.load_run(validation.behavior_run_id)
    assert scrutiny_run.role_model == "gpt-validator"
    assert behavior_run.role_model == "gpt-validator"
    assert {item.source for item in findings} >= {"scrutiny-validator", "behavior-validator"}

    completed = complete_mission_feature(
        store=store,
        mission_id=mission_id,
        feature_id=first_dispatch.feature_id,
        completed_work="Implemented the original feature after validator feedback.",
    )
    assert completed.status == "recorded"

    follow_up = execute_next_mission_feature(store=store, mission_id=mission_id)
    assert follow_up.status == "dispatched"
    assert follow_up.feature_id == corrective.id


def test_execute_mission_milestone_auto_complete_passes_validation(tmp_path: Path) -> None:
    store, mission_id = _seed_approved_mission(tmp_path)

    result = execute_mission_milestone(
        store=store,
        mission_id=mission_id,
        max_steps=10,
        auto_complete=True,
    )

    assert result.status == "completed"
    assert result.stop_reason == "validation_passed"
    assert any(step.kind == "dispatch" for step in result.steps)
    assert any(step.kind == "complete" for step in result.steps)
    assert any(step.kind == "validate" for step in result.steps)


def test_execute_mission_burst_advances_across_milestones(tmp_path: Path) -> None:
    store, mission_id = _seed_two_milestone_mission(tmp_path)

    result = execute_mission_burst(
        store=store,
        mission_id=mission_id,
        max_steps=20,
        auto_complete=True,
    )

    assert result.status == "completed"
    assert result.stop_reason == "mission_completed"
    mission = store.load_mission(mission_id)
    assert mission.status == "completed"
    assert sum(1 for step in result.steps if step.kind == "validate") == 2


def test_write_mission_scheduled_run_record_persists_artifacts(tmp_path: Path) -> None:
    store, mission_id = _seed_two_milestone_mission(tmp_path)
    result = execute_mission_burst(
        store=store,
        mission_id=mission_id,
        max_steps=20,
        auto_complete=True,
    )

    target = write_mission_scheduled_run_record(
        store=store,
        cwd=tmp_path,
        result=result,
    )

    assert (target / "run.json").is_file()
    assert (target / "RUN.md").is_file()
