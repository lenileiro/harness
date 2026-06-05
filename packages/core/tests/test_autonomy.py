from __future__ import annotations

from pathlib import Path

from harness.core.autonomy import execute_next_research_item
from harness.core.hypotheses import Hypothesis
from harness.core.research_store import ResearchStore, default_research_root


def test_autonomy_hypothesis_plan_uses_language_agnostic_harness_checks(
    tmp_path: Path,
) -> None:
    store = ResearchStore(root=default_research_root(tmp_path))
    hypothesis = Hypothesis(
        id="hyp-generic-verification",
        opportunity_id="opp-verification",
        claim="Generic Harness checks improve verification reliability.",
        expected_win="Language-independent validation covers the change.",
        risk_level="low",
        change_mode="improve",
    )
    store.add_hypothesis(hypothesis)

    result = execute_next_research_item(store=store, cwd=tmp_path)

    assert result.status == "executed"
    assert result.queue_item_kind == "hypothesis"
    plan = store.list_experiment_plans(hypothesis_id=hypothesis.id)[0]
    assert plan.checks == ("uv run harness eval validate",)
    assert plan.eval_slices == ("workflow-smoke",)
    default_commands = " ".join((*plan.checks, *plan.eval_slices))
    assert "pytest" not in default_commands
