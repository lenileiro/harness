from __future__ import annotations

import json
from pathlib import Path


def _load_singleton_json(directory: Path, filename: str) -> dict:
    matches = sorted(directory.glob(f"*/{filename}"))
    assert len(matches) == 1, f"expected exactly one {filename} in {directory}, got {matches!r}"
    return json.loads(matches[0].read_text(encoding="utf-8"))


def test_handover_vision_flow() -> None:
    root = Path(".harness")
    research_root = root / "research"
    assert research_root.exists(), "expected .harness/research to be created by the research CLI"

    vision_path = research_root / "vision" / "current" / "vision.json"
    assert vision_path.exists(), "vision.json was not created"
    vision = json.loads(vision_path.read_text(encoding="utf-8"))
    assert vision["title"] == "Autonomous Handover System"
    assert vision["summary"] == "Leave structured continuation state for the next agent."

    theme = _load_singleton_json(research_root / "themes", "theme.json")
    assert theme["title"] == "Agent continuity"

    unknown = _load_singleton_json(research_root / "unknowns", "unknown.json")
    assert unknown["question"] == "What context must a handover always preserve?"
    assert (
        unknown["why_it_matters"]
        == "Future agents need enough structure to continue work without replaying history."
    )

    rabbit_hole = _load_singleton_json(research_root / "rabbitholes", "rabbit_hole.json")
    assert rabbit_hole["title"] == "Handover artifact checklist"
    assert (
        rabbit_hole["question"]
        == "What information should every agent handover preserve for the next agent?"
    )

    publication = _load_singleton_json(research_root / "publications", "publication.json")
    assert publication["title"] == "Handover checklist findings"
    assert publication["rabbit_hole_id"] == rabbit_hole["id"]
    assert publication["recommendations"] == [
        "Use the resume contract to point the next agent at a single current feature."
    ]
    assert publication["open_questions"] == [
        "Should handover artifacts also include a verification checklist for the next agent?"
    ]

    resume_path = root / "resume.json"
    assert resume_path.exists(), "resume.json was not created"
    resume = json.loads(resume_path.read_text(encoding="utf-8"))
    assert resume["current"] == "handover-consumption"
    features = {item["name"]: item for item in resume["features"]}
    assert set(features) == {"handover-checklist", "handover-consumption"}
    assert (
        features["handover-checklist"]["description"]
        == "Capture the checklist for what a durable handover must contain."
    )
    assert (
        features["handover-consumption"]["description"]
        == "Consume the handover and continue the next step without replaying prior context."
    )
    assert features["handover-consumption"]["phases"] == ["review", "continue", "verify"]
