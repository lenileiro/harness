from __future__ import annotations

from pathlib import Path

from harness.core.pr_generation import (
    branch_name_for_candidate,
    build_promotion_draft,
    write_promotion_draft,
)
from harness.core.promotion_candidates import PromotionCandidate
from harness.core.research_models import ChangeIntent


def _candidate() -> PromotionCandidate:
    return PromotionCandidate(
        id="promo-research-completion-fix",
        title="Research completion fix",
        summary="Promote the repo-first research and loop-detection change.",
        source_publications=("pub-demo",),
        source_hypotheses=("hyp-demo",),
        target_files=("packages/core/src/harness/core/domain_profiles.py",),
        expected_metric="research smoke pass rate",
        validation_plan="Run pytest, pyright, and research smoke.",
        risk_level="low",
        change_intent=ChangeIntent(
            mode="improve",
            subsystem="research",
            rationale="Current research runs over-explore before finalizing.",
            expected_outcome="More final structured answers.",
        ),
    )


def test_build_promotion_draft_includes_candidate_evidence() -> None:
    candidate = _candidate()

    draft = build_promotion_draft(candidate, base_branch="main")

    assert branch_name_for_candidate(candidate).startswith("research/")
    assert draft.pr_title.startswith("improve:")
    assert "research smoke pass rate" in draft.pr_body
    assert "`pub-demo`" in draft.pr_body
    assert "`packages/core/src/harness/core/domain_profiles.py`" in draft.pr_body


def test_write_promotion_draft_persists_json_and_markdown(tmp_path: Path) -> None:
    draft = build_promotion_draft(_candidate(), base_branch="main")

    json_path, body_path = write_promotion_draft(draft=draft, target_dir=tmp_path)

    assert json_path.is_file()
    assert body_path.is_file()
    assert "Promotion Candidate" in body_path.read_text(encoding="utf-8")
