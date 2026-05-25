from __future__ import annotations

import json
import os
from pathlib import Path


def _load_singleton_json(directory: Path, filename: str) -> dict:
    matches = sorted(directory.glob(f"*/{filename}"))
    assert len(matches) == 1, f"expected exactly one {filename} in {directory}, got {matches!r}"
    return json.loads(matches[0].read_text(encoding="utf-8"))


def _load_json_matches(directory: Path, filename: str) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.glob(f"*/{filename}"))
    ]


def test_promotion_draft_flow() -> None:
    root = Path(os.environ.get("HARNESS_EVAL_WORKSPACE", ".")) / ".harness" / "research"
    assert root.exists(), "expected .harness/research to be created by the research CLI"

    vision = json.loads((root / "vision" / "current" / "vision.json").read_text(encoding="utf-8"))
    assert vision["title"] == "Promotion Workflow Hardening"

    theme = _load_singleton_json(root / "themes", "theme.json")
    assert theme["title"] == "Promotion reliability"

    rabbit_hole = _load_singleton_json(root / "rabbitholes", "rabbit_hole.json")
    assert rabbit_hole["title"] == "Promotion evidence checklist"
    assert rabbit_hole["question"] == "What evidence should every promotion draft include?"

    publication = _load_singleton_json(root / "publications", "publication.json")
    assert publication["title"] == "Promotion evidence findings"
    assert publication["rabbit_hole_id"] == rabbit_hole["id"]
    assert publication["claims"] == [
        "Promotion drafts should include explicit evidence and validation sections."
    ]

    opportunity = _load_singleton_json(root / "opportunities", "opportunity.json")
    assert opportunity["title"] == "Tighten promotion drafts"

    hypothesis = _load_singleton_json(root / "hypotheses", "hypothesis.json")
    assert hypothesis["opportunity_id"] == opportunity["id"]
    assert (
        hypothesis["claim"]
        == "Explicit evidence sections will make promotion drafts easier to review."
    )

    candidate_matches = _load_json_matches(root / "promotions", "promotion_candidate.json")
    assert candidate_matches, "no promotion candidates were created"
    candidate = next(
        (item for item in candidate_matches if item["title"] == "Promotion draft evidence section"),
        None,
    )
    assert candidate is not None, candidate_matches
    assert candidate["title"] == "Promotion draft evidence section"
    assert candidate["source_publications"] == [publication["id"]]
    assert candidate["source_hypotheses"] == [hypothesis["id"]]
    assert candidate["expected_metric"] == "promotion drafts include evidence checklist coverage"

    candidate_dir = root / "promotions" / candidate["id"]
    draft = json.loads((candidate_dir / "promotion_draft.json").read_text(encoding="utf-8"))
    assert draft["candidate_id"] == candidate["id"]
    assert draft["branch_name"].startswith("research/")
    assert draft["pr_title"]

    pr_body = (candidate_dir / "PR_BODY.md").read_text(encoding="utf-8")
    assert "## Promotion Candidate" in pr_body
    assert "## Evidence Checklist" in pr_body
    assert "promotion drafts include evidence checklist coverage" in pr_body
