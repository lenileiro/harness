from __future__ import annotations

import json
import os
from pathlib import Path


def _load_singleton_json(directory: Path, filename: str) -> dict:
    matches = sorted(directory.glob(f"*/{filename}"))
    assert len(matches) == 1, f"expected exactly one {filename} in {directory}, got {matches!r}"
    return json.loads(matches[0].read_text(encoding="utf-8"))


def test_mission_handoff_flow() -> None:
    root = Path(os.environ.get("HARNESS_EVAL_WORKSPACE", ".")) / ".harness" / "missions"

    mission = _load_singleton_json(root / "missions", "mission.json")
    assert mission["title"] == "Mission Handoff System"
    assert mission["status"] == "completed"
    assert mission["current_milestone_id"] == ""

    milestone = _load_singleton_json(root / "milestones", "milestone.json")
    assert milestone["status"] == "completed"

    feature = _load_singleton_json(root / "features", "feature.json")
    assert feature["title"] == "Complete handoff feature"
    assert feature["status"] == "validated"

    handoffs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "handoffs").glob("*/handoff.json"))
    ]
    assert len(handoffs) == 2
    worker = next(item for item in handoffs if item["role"] == "worker")
    assert (
        worker["completed_work"]
        == "Implemented the mission handoff consumption flow and recorded the worker completion state."
    )
    assert worker["remaining_work"] == "No remaining feature work."
    assert worker["known_issues"] == ["Validator pass still required."]
    assert (
        worker["next_recommendation"]
        == "Validate the milestone and publish the mission summary report."
    )
    assert worker["confidence"] == 0.95

    validator_runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "runs").glob("*/run.json"))
        if json.loads(path.read_text(encoding="utf-8"))["role"] == "validator"
    ]
    assert len(validator_runs) == 1
    assert validator_runs[0]["status"] == "completed"

    report = _load_singleton_json(root / "reports", "report.json")
    assert report["status"] == "completed"
    assert report["findings"] == []
    assert report["next_actions"] == ["Mission is complete."]
