from __future__ import annotations

import json
from pathlib import Path


def _load_singleton_json(directory: Path, filename: str) -> dict:
    matches = sorted(directory.glob(f"*/{filename}"))
    assert len(matches) == 1, f"expected exactly one {filename} in {directory}, got {matches!r}"
    return json.loads(matches[0].read_text(encoding="utf-8"))


def test_research_publication_flow() -> None:
    root = Path(".harness/research")
    assert root.exists(), "expected .harness/research to be created by the research CLI"

    vision_path = root / "vision" / "current" / "vision.json"
    assert vision_path.exists(), "vision.json was not created"
    vision = json.loads(vision_path.read_text(encoding="utf-8"))
    assert vision["title"] == "Autonomous Harness Research"

    theme = _load_singleton_json(root / "themes", "theme.json")
    assert theme["title"] == "Plugin reliability"

    unknown = _load_singleton_json(root / "unknowns", "unknown.json")
    assert unknown["question"] == "How should workspace plugins be validated?"

    rabbit_hole = _load_singleton_json(root / "rabbitholes", "rabbit_hole.json")
    assert rabbit_hole["title"] == "Workspace plugin import flow"

    publication_dir_matches = sorted(root.glob("publications/*"))
    assert len(publication_dir_matches) == 1, publication_dir_matches
    publication_dir = publication_dir_matches[0]
    publication = json.loads((publication_dir / "publication.json").read_text(encoding="utf-8"))
    assert publication["title"] == "Workspace plugin validation findings"
    assert publication["rabbit_hole_id"] == rabbit_hole["id"]
    markdown_path = publication_dir / "PUBLICATION.md"
    assert markdown_path.exists(), "publication markdown artifact was not created"
