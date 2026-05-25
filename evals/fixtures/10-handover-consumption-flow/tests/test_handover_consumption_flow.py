from __future__ import annotations

import json
import os
from pathlib import Path


def _load_singleton_json(directory: Path, filename: str) -> dict:
    matches = sorted(directory.glob(f"*/{filename}"))
    assert len(matches) == 1, f"expected exactly one {filename} in {directory}, got {matches!r}"
    return json.loads(matches[0].read_text(encoding="utf-8"))


def test_handover_consumption_flow() -> None:
    root = Path(os.environ.get("HARNESS_EVAL_WORKSPACE", ".")) / ".harness"
    research_root = root / "research"
    assert research_root.exists(), "expected seeded .harness/research to exist"

    vision = json.loads(
        (research_root / "vision" / "current" / "vision.json").read_text(encoding="utf-8")
    )
    assert vision["title"] == "Autonomous Handover System"

    publication = _load_singleton_json(research_root / "publications", "publication.json")
    assert publication["title"] == "Handover checklist findings"

    opportunity = _load_singleton_json(research_root / "opportunities", "opportunity.json")
    assert opportunity["title"] == "Handover verification checklist"
    assert (
        opportunity["summary"]
        == "The next agent should verify that a handover names the next feature, the source publication, and the continuation path."
    )

    hypothesis = _load_singleton_json(research_root / "hypotheses", "hypothesis.json")
    assert hypothesis["opportunity_id"] == opportunity["id"]
    assert (
        hypothesis["claim"]
        == "A verification checklist will make handovers safer for the next agent to consume."
    )
    assert (
        hypothesis["expected_win"]
        == "handover artifacts become easier for follow-on agents to verify"
    )
    assert hypothesis["risk_level"] == "low"
    assert hypothesis["change_mode"] == "improve"

    candidate = _load_singleton_json(research_root / "promotions", "promotion_candidate.json")
    assert candidate["title"] == "Handover verification handoff"
    assert (
        candidate["summary"]
        == "Promote a verification-oriented continuation step after the handover is consumed."
    )
    assert candidate["source_publications"] == [publication["id"]]
    assert candidate["source_hypotheses"] == [hypothesis["id"]]
    assert (
        candidate["expected_metric"] == "handover artifacts include verification checklist coverage"
    )
    assert (
        candidate["validation_plan"]
        == "Run the fixture tests to confirm the continuation artifacts and resume contract are linked."
    )

    resume = json.loads((root / "resume.json").read_text(encoding="utf-8"))
    assert resume["current"] == "handover-verification"
    features = {item["name"]: item for item in resume["features"]}
    assert set(features) == {
        "handover-checklist",
        "handover-consumption",
        "handover-verification",
    }
    assert (
        features["handover-verification"]["description"]
        == "Verify the handover contract before the next continuation step."
    )
    assert features["handover-verification"]["phases"] == ["inspect", "verify", "continue"]
    assert features["handover-consumption"]["phases"] == ["review", "continue", "verify"]
