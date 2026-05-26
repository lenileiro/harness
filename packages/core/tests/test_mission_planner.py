from __future__ import annotations

import pytest

from harness.core.mission_models import Mission
from harness.core.mission_planner import (
    MissionPlanDraft,
    PlannedAssertionInput,
    PlannedFeatureInput,
    PlannedMilestoneInput,
    build_mission_plan,
    parse_mission_plan_draft,
)
from harness.core.mission_store import MissionStore, default_mission_root


def test_build_mission_plan_links_features_and_assertions(tmp_path) -> None:
    store = MissionStore(root=default_mission_root(tmp_path))
    mission = Mission(
        id="mission-demo",
        title="Mission demo",
        goal="Deliver a planned milestone.",
    )

    plan = build_mission_plan(
        store=store,
        mission=mission,
        contract_summary="Assertions define correctness before coding.",
        milestones=(
            PlannedMilestoneInput(label="m1", title="Milestone 1", summary="Ship the first slice."),
        ),
        assertions=(
            PlannedAssertionInput(
                label="a1",
                title="Login works",
                description="Primary login flow succeeds.",
                kind="behavior",
                verification_method="Run browser validation.",
            ),
        ),
        features=(
            PlannedFeatureInput(
                label="f1",
                milestone_label="m1",
                title="Implement login flow",
                summary="Add a login form and handler.",
                assigned_role="worker",
                target_files=("app/login.py",),
                assertion_labels=("a1",),
            ),
        ),
    )

    assert plan.mission.status == "planned"
    assert plan.mission.current_milestone_id == plan.milestones[0].id
    assert len(plan.contract.assertions) == 1
    assert plan.contract.assertions[0].covered_by_features == (plan.features[0].id,)
    assert plan.features[0].research_refs == ()


def test_build_mission_plan_preserves_research_refs(tmp_path) -> None:
    store = MissionStore(root=default_mission_root(tmp_path))
    mission = Mission(
        id="mission-demo",
        title="Mission demo",
        goal="Deliver a planned milestone.",
    )

    plan = build_mission_plan(
        store=store,
        mission=mission,
        contract_summary="Assertions define correctness before coding.",
        milestones=(
            PlannedMilestoneInput(label="m1", title="Milestone 1", summary="Ship the first slice."),
        ),
        assertions=(
            PlannedAssertionInput(
                label="a1",
                title="Login works",
                description="Primary login flow succeeds.",
                kind="behavior",
                verification_method="Run browser validation.",
            ),
        ),
        features=(
            PlannedFeatureInput(
                label="f1",
                milestone_label="m1",
                title="Implement login flow",
                summary="Add a login form and handler.",
                assigned_role="worker",
                target_files=("app/login.py",),
                assertion_labels=("a1",),
                research_refs=("publication-123", "hypothesis-abc"),
            ),
        ),
    )

    assert plan.features[0].research_refs == ("publication-123", "hypothesis-abc")


def test_parse_mission_plan_draft_reads_structured_json() -> None:
    parsed = parse_mission_plan_draft(
        """
        {
          "contract_summary": "Assertions exist before coding.",
          "milestones": [
            {"label": "m1", "title": "Milestone 1", "summary": "Ship the first slice."}
          ],
          "assertions": [
            {
              "label": "a1",
              "title": "Slice works",
              "description": "The slice validates cleanly.",
              "kind": "contract",
              "verification_method": "Inspect validation output."
            }
          ],
          "features": [
            {
              "label": "f1",
              "milestone_label": "m1",
              "title": "Implement slice",
              "summary": "Build the first slice.",
              "assigned_role": "worker",
              "target_files": ["app/slice.py"],
              "depends_on_labels": [],
              "assertion_labels": ["a1"],
              "research_refs": ["publication-seed"]
            }
          ]
        }
        """
    )
    assert isinstance(parsed, MissionPlanDraft)
    assert parsed.contract_summary == "Assertions exist before coding."
    assert parsed.milestones[0].label == "m1"
    assert parsed.assertions[0].verification_method == "Inspect validation output."
    assert parsed.features[0].research_refs == ("publication-seed",)


def test_build_mission_plan_rejects_unknown_assertion_references(tmp_path) -> None:
    store = MissionStore(root=default_mission_root(tmp_path))
    mission = Mission(
        id="mission-demo",
        title="Mission demo",
        goal="Deliver a planned milestone.",
    )

    with pytest.raises(ValueError, match="references unknown assertions"):
        build_mission_plan(
            store=store,
            mission=mission,
            contract_summary="Assertions define correctness before coding.",
            milestones=(
                PlannedMilestoneInput(
                    label="m1",
                    title="Milestone 1",
                    summary="Ship the first slice.",
                ),
            ),
            assertions=(
                PlannedAssertionInput(
                    label="a1",
                    title="Login works",
                    description="Primary login flow succeeds.",
                    kind="behavior",
                    verification_method="Run browser validation.",
                ),
            ),
            features=(
                PlannedFeatureInput(
                    label="f1",
                    milestone_label="m1",
                    title="Implement login flow",
                    summary="Add a login form and handler.",
                    assigned_role="worker",
                    target_files=("app/login.py",),
                    assertion_labels=("missing-assertion",),
                ),
            ),
        )


def test_build_mission_plan_rejects_uncovered_assertions(tmp_path) -> None:
    store = MissionStore(root=default_mission_root(tmp_path))
    mission = Mission(
        id="mission-demo",
        title="Mission demo",
        goal="Deliver a planned milestone.",
    )

    with pytest.raises(ValueError, match="covered by at least one feature"):
        build_mission_plan(
            store=store,
            mission=mission,
            contract_summary="Assertions define correctness before coding.",
            milestones=(
                PlannedMilestoneInput(
                    label="m1",
                    title="Milestone 1",
                    summary="Ship the first slice.",
                ),
            ),
            assertions=(
                PlannedAssertionInput(
                    label="a1",
                    title="Login works",
                    description="Primary login flow succeeds.",
                    kind="behavior",
                    verification_method="Run browser validation.",
                ),
                PlannedAssertionInput(
                    label="a2",
                    title="Logout works",
                    description="Primary logout flow succeeds.",
                    kind="behavior",
                    verification_method="Run browser validation.",
                ),
            ),
            features=(
                PlannedFeatureInput(
                    label="f1",
                    milestone_label="m1",
                    title="Implement login flow",
                    summary="Add a login form and handler.",
                    assigned_role="worker",
                    target_files=("app/login.py",),
                    assertion_labels=("a1",),
                ),
            ),
        )
