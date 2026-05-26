from __future__ import annotations

import json
import os
from pathlib import Path


def _load_singleton_json(directory: Path, filename: str) -> dict:
    matches = sorted(directory.glob(f"*/{filename}"))
    assert len(matches) == 1, f"expected exactly one {filename} in {directory}, got {matches!r}"
    return json.loads(matches[0].read_text(encoding="utf-8"))


def test_mission_planning_flow() -> None:
    root = Path(os.environ.get("HARNESS_EVAL_WORKSPACE", ".")) / ".harness" / "missions"
    mission = _load_singleton_json(root / "missions", "mission.json")
    assert mission["title"] == "Mission Planning System"
    assert (
        mission["goal"]
        == "Create a mission plan that defines milestones, assertions, and feature coverage before implementation."
    )
    assert mission["status"] == "approved"
    assert mission["planner_model"] == "gpt-planner"
    assert mission["worker_model"] == "gpt-worker"
    assert mission["validator_model"] == "gpt-validator"
    assert mission["reporter_model"] == "gpt-reporter"
    assert mission["budget_tokens"] == 9000
    assert mission["budget_runtime_minutes"] == 120

    contract = _load_singleton_json(root / "contracts", "contract.json")
    assert (
        contract["summary"]
        == "Mission assertions must be declared before execution and covered by planned feature work."
    )
    assertions = {item["title"]: item for item in contract["assertions"]}
    assert set(assertions) == {
        "Mission object exists",
        "Runtime dispatch works",
        "Validation gates milestones",
    }

    milestones = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "milestones").glob("*/milestone.json"))
    ]
    milestones.sort(key=lambda item: item["order"])
    assert [item["title"] for item in milestones] == ["Mission schema", "Mission runtime"]
    assert [item["order"] for item in milestones] == [1, 2]
    assert mission["current_milestone_id"] == milestones[0]["id"]

    features = {
        item["title"]: item
        for item in (
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / "features").glob("*/feature.json"))
        )
    }
    assert set(features) == {
        "Define mission schema",
        "Implement mission runtime",
        "Implement mission validator",
    }
    assert features["Define mission schema"]["assigned_role"] == "planner"
    assert features["Implement mission runtime"]["depends_on"] == [
        features["Define mission schema"]["id"]
    ]
    assert features["Implement mission validator"]["depends_on"] == [
        features["Implement mission runtime"]["id"]
    ]
    assert features["Implement mission runtime"]["target_files"] == [
        "packages/core/src/harness/core/mission_runtime.py"
    ]
    assert features["Implement mission validator"]["target_files"] == [
        "packages/core/src/harness/core/mission_validator.py"
    ]

    assert assertions["Mission object exists"]["covered_by_features"] == [
        features["Define mission schema"]["id"]
    ]
    assert assertions["Runtime dispatch works"]["covered_by_features"] == [
        features["Implement mission runtime"]["id"]
    ]
    assert assertions["Validation gates milestones"]["covered_by_features"] == [
        features["Implement mission validator"]["id"]
    ]
