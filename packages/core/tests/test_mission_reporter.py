from __future__ import annotations

import json
from pathlib import Path

from harness.core.mission_models import Mission
from harness.core.mission_planner import (
    PlannedAssertionInput,
    PlannedFeatureInput,
    PlannedMilestoneInput,
    build_mission_plan,
)
from harness.core.mission_reporter import (
    build_mission_summary_report,
    list_mission_reports,
    load_mission_report,
    write_mission_summary_report,
)
from harness.core.mission_runtime import execute_next_mission_feature
from harness.core.mission_store import MissionStore, default_mission_root
from harness.core.mission_validator import validate_mission_milestone


def _seed_blocked_mission(tmp_path: Path) -> tuple[MissionStore, str]:
    store = MissionStore(root=default_mission_root(tmp_path))
    mission = Mission(
        id="mission-report",
        title="Mission report",
        goal="Produce a readable mission summary.",
        status="draft",
        planner_model="gpt-planner",
        worker_model="gpt-worker",
        reporter_model="gpt-reporter",
        planner_brief="Break the work into bounded slices.",
        worker_brief="Implement the current feature and hand it off.",
        reporter_brief="Summarize the state and next actions.",
    )
    store.add_mission(mission)
    plan = build_mission_plan(
        store=store,
        mission=mission,
        contract_summary="Mission report assertions.",
        milestones=(PlannedMilestoneInput("m1", "Milestone 1", "First milestone."),),
        assertions=(
            PlannedAssertionInput(
                "a1",
                "Report works",
                "Mission reporter should explain blocked work.",
                "contract",
                "Inspect report artifact.",
            ),
        ),
        features=(
            PlannedFeatureInput(
                "f1",
                "m1",
                "Feature 1",
                "Implement the reported feature.",
                "worker",
                ("app/report.py",),
                (),
                ("a1",),
            ),
        ),
    )
    store.update_mission(plan.mission)
    for milestone in plan.milestones:
        store.add_milestone(milestone)
    for feature in plan.features:
        store.add_feature(feature)
    store.add_contract(plan.contract)
    store.update_mission(Mission.from_dict({**plan.mission.to_dict(), "status": "approved"}))
    execute_next_mission_feature(store=store, mission_id=mission.id)
    validate_mission_milestone(store=store, mission_id=mission.id)
    return store, mission.id


def test_build_and_write_mission_summary_report(tmp_path: Path) -> None:
    store, mission_id = _seed_blocked_mission(tmp_path)

    report = build_mission_summary_report(store=store, mission_id=mission_id)
    target = write_mission_summary_report(store=store, report=report)

    assert report.status == "blocked"
    assert report.findings
    assert report.next_actions
    assert report.role_profiles
    payload = json.loads((target / "report.json").read_text(encoding="utf-8"))
    assert payload["mission_id"] == mission_id
    assert payload["role_profiles"][0]["role"] == "planner"
    assert (target / "REPORT.md").is_file()


def test_list_and_load_mission_reports(tmp_path: Path) -> None:
    store, mission_id = _seed_blocked_mission(tmp_path)
    report = build_mission_summary_report(store=store, mission_id=mission_id)
    write_mission_summary_report(store=store, report=report)

    reports = list_mission_reports(store=store, mission_id=mission_id)
    assert len(reports) == 1
    loaded = load_mission_report(store=store, report_id=reports[0].id)
    assert loaded.id == reports[0].id
    assert loaded.status == "blocked"
