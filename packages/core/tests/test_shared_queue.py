from __future__ import annotations

from harness.core.mission_models import (
    Milestone,
    Mission,
    MissionFeature,
    ValidationAssertion,
    ValidationContract,
)
from harness.core.mission_store import MissionStore
from harness.core.opportunities import Opportunity
from harness.core.research_store import ResearchStore
from harness.core.shared_queue import (
    build_mission_work_queue,
    build_research_work_queue,
    build_shared_work_queue,
)


def test_build_mission_work_queue_marks_validation_ready(tmp_path) -> None:
    store = MissionStore(root=tmp_path / ".harness" / "missions")
    mission = Mission(
        id="mission-1",
        title="Mission demo",
        goal="Validate after feature completion.",
        status="running",
        current_milestone_id="milestone-1",
    )
    milestone = Milestone(
        id="milestone-1",
        mission_id=mission.id,
        title="Milestone 1",
        summary="Ship one slice.",
        status="active",
        order=1,
    )
    contract = ValidationContract(
        id="contract-1",
        mission_id=mission.id,
        summary="Assertions first.",
        assertions=(
            ValidationAssertion(
                id="assertion-1",
                contract_id="contract-1",
                title="A1",
                description="Feature is complete.",
                kind="behavior",
                verification_method="Run milestone validation.",
                covered_by_features=("feature-1",),
            ),
        ),
    )
    feature = MissionFeature(
        id="feature-1",
        mission_id=mission.id,
        milestone_id=milestone.id,
        title="Implement slice",
        summary="Build the first slice.",
        status="completed",
    )
    store.add_mission(mission)
    store.add_milestone(milestone)
    store.add_contract(contract)
    store.add_feature(feature)

    items = build_mission_work_queue(store)
    assert len(items) == 1
    assert items[0].kind == "mission.validation"
    assert items[0].ready is True


def test_build_shared_work_queue_combines_mission_and_research(tmp_path) -> None:
    mission_store = MissionStore(root=tmp_path / ".harness" / "missions")
    research_store = ResearchStore(root=tmp_path / ".harness" / "research")

    mission = Mission(
        id="mission-1",
        title="Mission demo",
        goal="Ship one slice.",
        status="approved",
        current_milestone_id="milestone-1",
    )
    milestone = Milestone(
        id="milestone-1",
        mission_id=mission.id,
        title="Milestone 1",
        summary="Ship one slice.",
        status="pending",
        order=1,
    )
    feature = MissionFeature(
        id="feature-1",
        mission_id=mission.id,
        milestone_id=milestone.id,
        title="Implement slice",
        summary="Build the first slice.",
        status="pending",
    )
    contract = ValidationContract(
        id="contract-1",
        mission_id=mission.id,
        summary="Assertions first.",
        assertions=(
            ValidationAssertion(
                id="assertion-1",
                contract_id="contract-1",
                title="A1",
                description="Feature is complete.",
                kind="behavior",
                verification_method="Run milestone validation.",
                covered_by_features=(feature.id,),
            ),
        ),
    )
    mission_store.add_mission(mission)
    mission_store.add_milestone(milestone)
    mission_store.add_contract(contract)
    mission_store.add_feature(feature)

    opportunity = Opportunity(
        id="opp-1",
        title="Research next step",
        summary="Explore a follow-on idea.",
        related_sections=("docs/",),
        change_modes=("improve",),
        priority="high",
        created_by="test",
    )
    research_store.add_opportunity(opportunity)

    mission_items = build_mission_work_queue(mission_store)
    research_items = build_research_work_queue(research_store)
    shared_items = build_shared_work_queue(
        mission_store=mission_store,
        research_store=research_store,
    )

    assert mission_items[0].source == "mission"
    assert research_items[0].source == "research"
    assert {item.source for item in shared_items} == {"mission", "research"}
