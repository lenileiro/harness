from __future__ import annotations

import json
import os
from pathlib import Path


def test_mission_corrective_loop_flow() -> None:
    root = Path(os.environ.get("HARNESS_EVAL_WORKSPACE", ".")) / ".harness" / "missions"

    mission = json.loads(
        next((root / "missions").glob("*/mission.json")).read_text(encoding="utf-8")
    )
    assert mission["title"] == "Mission Corrective Loop"
    assert mission["status"] == "completed"
    assert mission["current_milestone_id"] == ""

    milestone = json.loads(
        next((root / "milestones").glob("*/milestone.json")).read_text(encoding="utf-8")
    )
    assert milestone["status"] == "completed"

    features = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "features").glob("*/feature.json"))
    ]
    by_title = {item["title"]: item for item in features}
    assert "Implement original slice" in by_title
    assert "Corrective: Implement original slice" in by_title
    assert by_title["Implement original slice"]["status"] == "validated"
    assert by_title["Corrective: Implement original slice"]["status"] == "validated"
    assert by_title["Corrective: Implement original slice"]["depends_on"] == [
        by_title["Implement original slice"]["id"]
    ]

    findings = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "findings").glob("*/finding.json"))
    ]
    assert len(findings) >= 2
    assert any(
        item["summary"] == "Feature 'Implement original slice' is not complete."
        for item in findings
    )
    assert any(
        item["summary"]
        == "Assertion 'Corrective loop works' is not yet satisfied by completed feature work."
        for item in findings
    )

    validator_runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "runs").glob("*/run.json"))
        if json.loads(path.read_text(encoding="utf-8"))["role"] == "validator"
    ]
    statuses = [item["status"] for item in validator_runs]
    assert statuses.count("failed") == 1
    assert statuses.count("completed") == 1

    worker_handoffs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((root / "handoffs").glob("*/handoff.json"))
        if json.loads(path.read_text(encoding="utf-8"))["role"] == "worker"
    ]
    completed_work = {item["completed_work"] for item in worker_handoffs}
    assert (
        "Implemented the original mission slice after the validator exposed the missing completion state."
        in completed_work
    )
    assert (
        "Implemented the corrective follow-up slice so the validator can close the milestone."
        in completed_work
    )

    report = json.loads(next((root / "reports").glob("*/report.json")).read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["next_actions"] == ["Mission is complete."]
