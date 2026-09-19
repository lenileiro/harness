from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from harness.core.experiment_plans import ExperimentPlan
from harness.core.experiment_runner import run_experiment_plan
from harness.core.experiments import CommandResult, Experiment, ExperimentResult
from harness.core.promotion_candidates import PromotionCandidate
from harness.core.promotion_evidence import PromotionEvidenceError, require_promotion_evidence
from harness.core.research.store import ResearchStore, default_research_root
from harness.core.workspace_snapshot import workspace_fingerprint


def _setup(cwd: Path) -> tuple[ResearchStore, PromotionCandidate, ExperimentPlan]:
    (cwd / "src").mkdir(exist_ok=True)
    (cwd / "src/app.py").write_text("value = 1\n")
    store = ResearchStore(root=default_research_root(cwd))
    candidate = PromotionCandidate(
        id="candidate",
        title="Fix",
        summary="fix",
        target_files=("src/app.py",),
        source_experiments=("exp",),
    )
    plan = ExperimentPlan(
        id="plan", hypothesis_id="hyp", plan="test", target_files=("src",), checks=("check",)
    )
    store.add_experiment_plan(plan)
    _save(store, plan, cwd=cwd)
    return store, candidate, plan


def _save(
    store: ResearchStore,
    plan: ExperimentPlan,
    *,
    cwd: Path,
    experiment_id: str = "exp",
    created_at: str = "2026-01-01T00:00:00.000001+00:00",
    **overrides: object,
) -> ExperimentResult:
    result = ExperimentResult(
        experiment_id=experiment_id,
        status="passed",
        command_results=(
            CommandResult(kind="check", command="check", exit_code=0, duration_seconds=1),
        ),
        started_at=created_at,
        finished_at=created_at,
        duration_seconds=1,
        workspace_fingerprint=workspace_fingerprint(cwd),
    )
    result = replace(result, **overrides)  # type: ignore[arg-type]
    store.add_experiment(
        Experiment(id=experiment_id, plan_id=plan.id, created_at=created_at), result
    )
    return result


def test_current_explicit_evidence_and_source_reference_round_trip(tmp_path: Path) -> None:
    store, candidate, _ = _setup(tmp_path)
    assert PromotionCandidate.from_dict(candidate.to_dict()) == candidate
    evidence = require_promotion_evidence(candidate=candidate, store=store, cwd=tmp_path)
    assert evidence.experiment_ids == ("exp",)
    assert evidence.workspace_fingerprint == workspace_fingerprint(tmp_path)


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"status": "failed"}, "failed"),
        ({"status": "timed_out"}, "timed_out"),
        ({"command_results": ()}, "command evidence"),
        ({"workspace_fingerprint": ""}, "fingerprint"),
        ({"workspace_fingerprint": "stale"}, "stale"),
        (
            {
                "command_results": (
                    CommandResult(kind="check", command="check", exit_code=1, duration_seconds=1),
                )
            },
            "command evidence",
        ),
        (
            {
                "command_results": (
                    CommandResult(
                        kind="check", command="different check", exit_code=0, duration_seconds=1
                    ),
                )
            },
            "all commands",
        ),
    ],
)
def test_invalid_execution_evidence_blocks(
    tmp_path: Path, overrides: dict[str, Any], reason: str
) -> None:
    store, candidate, plan = _setup(tmp_path)
    _save(store, plan, cwd=tmp_path, **overrides)
    with pytest.raises(PromotionEvidenceError, match=reason):
        require_promotion_evidence(candidate=candidate, store=store, cwd=tmp_path)


def test_unrelated_targets_and_modified_inputs_block(tmp_path: Path) -> None:
    store, candidate, _ = _setup(tmp_path)
    with pytest.raises(PromotionEvidenceError, match="targets"):
        require_promotion_evidence(
            candidate=replace(candidate, target_files=("other.py",)), store=store, cwd=tmp_path
        )
    (tmp_path / "src/app.py").write_text("value = 2\n")
    with pytest.raises(PromotionEvidenceError, match="stale"):
        require_promotion_evidence(candidate=candidate, store=store, cwd=tmp_path)


def test_missing_link_blocks_and_latest_failed_hypothesis_run_overrides_old_pass(
    tmp_path: Path,
) -> None:
    store, candidate, plan = _setup(tmp_path)
    with pytest.raises(PromotionEvidenceError, match="no linked"):
        require_promotion_evidence(
            candidate=replace(candidate, source_experiments=()), store=store, cwd=tmp_path
        )
    inferred = replace(candidate, source_experiments=(), source_hypotheses=("hyp",))
    assert require_promotion_evidence(
        candidate=inferred, store=store, cwd=tmp_path
    ).experiment_ids == ("exp",)
    _save(
        store,
        plan,
        cwd=tmp_path,
        experiment_id="newer",
        created_at="2026-01-01T00:00:00.000002+00:00",
        status="failed",
    )
    with pytest.raises(PromotionEvidenceError, match="newer is failed"):
        require_promotion_evidence(candidate=inferred, store=store, cwd=tmp_path)


def test_ambiguous_legacy_timestamps_require_explicit_reference(tmp_path: Path) -> None:
    store, candidate, plan = _setup(tmp_path)
    _save(store, plan, cwd=tmp_path, experiment_id="another")
    with pytest.raises(PromotionEvidenceError, match="ambiguous"):
        require_promotion_evidence(
            candidate=replace(candidate, source_experiments=(), source_hypotheses=("hyp",)),
            store=store,
            cwd=tmp_path,
        )
    assert require_promotion_evidence(
        candidate=candidate, store=store, cwd=tmp_path
    ).experiment_ids == ("exp",)


@pytest.mark.parametrize("target", [".", "../outside.py", ".harness/result.json"])
def test_target_scope_cannot_escape_or_include_generated_state(tmp_path: Path, target: str) -> None:
    store, candidate, _ = _setup(tmp_path)
    with pytest.raises(PromotionEvidenceError):
        require_promotion_evidence(
            candidate=replace(candidate, target_files=(target,)), store=store, cwd=tmp_path
        )


def _measured(
    cwd: Path, *, before: dict[str, float], after: dict[str, float]
) -> tuple[ResearchStore, PromotionCandidate, ExperimentPlan]:
    store, candidate, original_plan = _setup(cwd)
    measurement = CommandResult(
        kind="measurement", command="measure", exit_code=0, duration_seconds=1
    )
    _save(
        store,
        original_plan,
        cwd=cwd,
        experiment_id="baseline",
        command_results=(measurement,),
        metrics=before,
    )
    directions = {"quality": "maximize", "latency": "minimize"}
    plan = replace(
        original_plan,
        measurement_command="measure",
        metric_directions=directions,
        baseline_experiment_id="baseline",
    )
    store.add_experiment_plan(plan)
    check = CommandResult(kind="check", command="check", exit_code=0, duration_seconds=1)
    _save(
        store,
        plan,
        cwd=cwd,
        command_results=(check, measurement),
        metrics=after,
        metric_directions=directions,
        baseline_experiment_id="baseline",
    )
    return store, candidate, plan


def test_measured_improvement_accepts_nonregressing_goals(tmp_path: Path) -> None:
    store, candidate, _ = _measured(
        tmp_path, before={"quality": 4, "latency": 2}, after={"quality": 5, "latency": 2}
    )
    evidence = require_promotion_evidence(candidate=candidate, store=store, cwd=tmp_path)
    assert evidence.metric_improvements == ("quality",)


@pytest.mark.parametrize(
    ("after", "reason"),
    [
        ({"quality": 5, "latency": 3}, "regressed"),
        ({"quality": 4, "latency": 2}, "no improvement"),
        ({"quality": float("nan"), "latency": 1}, "finite"),
        ({"quality": 5}, "missing"),
    ],
)
def test_metrics_cannot_regress_or_hide_invalid_measurements(
    tmp_path: Path, after: dict[str, float], reason: str
) -> None:
    store, candidate, _ = _measured(tmp_path, before={"quality": 4, "latency": 2}, after=after)
    with pytest.raises(PromotionEvidenceError, match=reason):
        require_promotion_evidence(candidate=candidate, store=store, cwd=tmp_path)


def test_failed_baseline_blocks_comparison(tmp_path: Path) -> None:
    store, candidate, plan = _measured(
        tmp_path, before={"quality": 4, "latency": 2}, after={"quality": 5, "latency": 1}
    )
    baseline = store.load_experiment_result("baseline")
    store.add_experiment(
        Experiment(id="baseline", plan_id=plan.id), replace(baseline, status="failed")
    )
    with pytest.raises(PromotionEvidenceError, match="baseline baseline is failed"):
        require_promotion_evidence(candidate=candidate, store=store, cwd=tmp_path)


def test_actual_runner_evidence_passes_for_unchanged_inputs(tmp_path: Path) -> None:
    import shlex

    store, candidate, plan = _setup(tmp_path)
    plan = replace(plan, checks=(f"{shlex.quote(sys.executable)} -c 'assert 2 + 2 == 4'",))
    store.add_experiment_plan(plan)
    experiment, result = run_experiment_plan(store=store, plan=plan, cwd=tmp_path)
    assert result.status == "passed"
    evidence = require_promotion_evidence(
        candidate=replace(candidate, source_experiments=(experiment.id,)), store=store, cwd=tmp_path
    )
    assert evidence.experiment_ids == (experiment.id,)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"baseline_experiment_id": ""}, "requires a baseline"),
        ({"metric_directions": {}}, "must declare"),
        ({"measurement_command": ""}, "measurement command"),
    ],
)
def test_declared_measurement_goals_require_complete_comparison_contract(
    tmp_path: Path, changes: dict[str, Any], reason: str
) -> None:
    store, candidate, plan = _measured(
        tmp_path, before={"quality": 4, "latency": 2}, after={"quality": 5, "latency": 1}
    )
    plan = replace(plan, **changes)
    store.add_experiment_plan(plan)
    result = store.load_experiment_result("exp")
    store.add_experiment(
        store.load_experiment("exp"),
        replace(
            result,
            metric_directions=plan.metric_directions,
            baseline_experiment_id=plan.baseline_experiment_id,
        ),
    )
    with pytest.raises(PromotionEvidenceError, match=reason):
        require_promotion_evidence(candidate=candidate, store=store, cwd=tmp_path)


def test_changing_metric_goals_after_execution_requires_rerun(tmp_path: Path) -> None:
    store, candidate, plan = _measured(
        tmp_path, before={"quality": 4, "latency": 2}, after={"quality": 5, "latency": 1}
    )
    store.add_experiment_plan(replace(plan, metric_directions={"quality": "minimize"}))
    with pytest.raises(PromotionEvidenceError, match="stale metric goals"):
        require_promotion_evidence(candidate=candidate, store=store, cwd=tmp_path)
