from __future__ import annotations

from pathlib import Path

import pytest

from harness.core.mission.models import (
    Milestone,
    Mission,
    MissionFeature,
    ValidationAssertion,
    ValidationContract,
)
from harness.core.mission.store import MissionStore, default_mission_root


def test_mission_store_writes_mission_files_and_lists_missions(tmp_path: Path) -> None:
    store = MissionStore(root=default_mission_root(tmp_path))
    mission = Mission(
        id=store.new_id("mission", "Autonomous QA"),
        title="Autonomous QA",
        goal="Plan and validate a QA-focused mission.",
        status="planned",
    )

    mission_path = store.add_mission(mission)

    assert (mission_path / "mission.json").is_file()
    assert (mission_path / "MISSION.md").is_file()
    assert store.load_mission(mission.id).title == "Autonomous QA"
    assert [item.id for item in store.list_missions(status="planned")] == [mission.id]


def test_mission_store_persists_milestones_features_and_contracts(tmp_path: Path) -> None:
    store = MissionStore(root=default_mission_root(tmp_path))
    mission = Mission(
        id="mission-demo",
        title="Mission demo",
        goal="Deliver one milestone.",
    )
    store.add_mission(mission)
    milestone = Milestone(
        id="milestone-1",
        mission_id="mission-demo",
        title="Milestone 1",
        summary="Ship the first slice.",
        order=1,
    )
    feature = MissionFeature(
        id="feature-1",
        mission_id="mission-demo",
        milestone_id="milestone-1",
        title="Feature 1",
        summary="Implement the login flow.",
        target_files=("app/login.py",),
    )
    contract = ValidationContract(
        id="contract-1",
        mission_id="mission-demo",
        summary="Assertions must exist before implementation.",
        assertions=(
            ValidationAssertion(
                id="assertion-1",
                contract_id="contract-1",
                title="Login works",
                description="Primary login flow succeeds.",
                kind="behavior",
                verification_method="Run browser validation.",
                covered_by_features=("feature-1",),
            ),
        ),
    )

    milestone_path = store.add_milestone(milestone)
    feature_path = store.add_feature(feature)
    contract_path = store.add_contract(contract)

    assert (milestone_path / "milestone.json").is_file()
    assert (feature_path / "feature.json").is_file()
    assert (contract_path / "contract.json").is_file()
    assert [item.id for item in store.list_milestones(mission_id="mission-demo")] == ["milestone-1"]
    assert [item.id for item in store.list_features(mission_id="mission-demo")] == ["feature-1"]
    loaded_contract = store.load_contract("contract-1")
    assert loaded_contract.assertions[0].covered_by_features == ("feature-1",)


def test_write_json_is_atomic_under_failure(tmp_path, monkeypatch):
    """A write that dies mid-flight must not corrupt the existing record.

    Mission records are written by long unattended runs that can be OOM-killed
    (observed on this repo), so a truncating write is a real corruption risk and
    not a theoretical one.
    """
    import json as _json

    from harness.core.mission import store as mission_store

    target = tmp_path / "mission.json"
    mission_store._write_json(target, {"id": "m-1", "status": "approved"})
    intact = target.read_text()

    def explode(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(_json, "dump", explode)
    with pytest.raises(OSError):
        mission_store._write_json(target, {"id": "m-1", "status": "completed"})

    # the original record survives byte for byte, and no temp file is left behind
    assert target.read_text() == intact
    assert _json.loads(target.read_text())["status"] == "approved"
    assert list(tmp_path.iterdir()) == [target]
