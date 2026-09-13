import json
from dataclasses import replace
from pathlib import Path

import pytest

from harness.core.autonomy import execute_next_research_item, execute_research_burst
from harness.core.experiment_plans import ExperimentPlan
from harness.core.experiment_runner import run_experiment_plan
from harness.core.opportunities import Opportunity
from harness.core.promotion_candidates import PromotionCandidate
from harness.core.research_models import Theme, Unknown
from harness.core.research_store import ResearchStore, default_research_root


def promotion_store(tmp_path):
    store = ResearchStore(root=default_research_root(tmp_path))
    (tmp_path / "source.py").write_text("pass\n")
    candidate = PromotionCandidate(
        id="promotion",
        title="Improve source",
        summary="A bounded change",
        target_files=("source.py",),
        expected_metric="fewer failures",
        validation_plan="Run checks",
        risk_level="low",
        change_intent=store.parse_change_intent(
            mode="improve",
            subsystem="runtime",
            rationale="Improve reliability",
            expected_outcome="Fewer failures",
            risk="low",
        ),
    )
    store.add_promotion_candidate(candidate)
    return store, candidate


def test_prepared_promotion_can_advance_requested_stages_without_repeating(tmp_path, monkeypatch):
    store, candidate = promotion_store(tmp_path)
    calls = []
    monkeypatch.setattr(
        "harness.core.promotion_evidence.require_promotion_evidence", lambda **kwargs: None
    )
    for function in ["ensure_branch", "commit_paths", "push_branch", "create_pull_request"]:
        monkeypatch.setattr(
            f"harness.core.autonomy.{function}",
            lambda _name=function, **kwargs: calls.append(_name),
        )
    assert execute_next_research_item(store=store, cwd=tmp_path).status == "executed"
    for stage in ["create_branch", "commit", "push", "open_pr"]:
        result = execute_next_research_item(
            store=store,
            cwd=tmp_path,
            create_branch=stage == "create_branch",
            commit=stage == "commit",
            push=stage == "push",
            open_pr=stage == "open_pr",
        )
        assert result.status == "executed", (stage, result)
    assert calls == ["ensure_branch", "commit_paths", "push_branch", "create_pull_request"]
    record = json.loads(
        (store.promotion_candidates_dir / candidate.id / "promotion_execution.json").read_text()
    )
    assert record["stages"] == {
        "prepare": True,
        "branch": True,
        "commit": True,
        "push": True,
        "open_pr": True,
    }
    assert (
        execute_next_research_item(
            store=store, cwd=tmp_path, create_branch=True, commit=True, push=True, open_pr=True
        ).status
        == "no_work"
    )
    assert len(calls) == 4


def test_evidence_blocked_promotion_does_not_starve_other_work(tmp_path):
    store, candidate = promotion_store(tmp_path)
    store.add_opportunity(
        Opportunity(id="other", title="Independent work", summary="Make progress")
    )
    result = execute_next_research_item(store=store, cwd=tmp_path, commit=True)
    assert result.queue_item_kind == "opportunity"
    assert result.status == "executed"
    assert len(store.list_hypotheses()) == 1
    record = json.loads(
        (store.promotion_candidates_dir / candidate.id / "promotion_execution.json").read_text()
    )
    assert "no linked experiment" in record["deferred_reason"]


def test_resumed_promotion_restores_requested_branch_before_committing(tmp_path, monkeypatch):
    store, _candidate = promotion_store(tmp_path)
    checkout = "main"
    calls = []

    def ensure_branch(**kwargs):
        nonlocal checkout
        checkout = kwargs["branch_name"]
        calls.append("checkout")

    def commit(**kwargs):
        assert checkout != "unrelated"
        calls.append("commit")

    monkeypatch.setattr("harness.core.autonomy.ensure_branch", ensure_branch)
    monkeypatch.setattr("harness.core.autonomy.commit_paths", commit)
    monkeypatch.setattr(
        "harness.core.promotion_evidence.require_promotion_evidence", lambda **kwargs: None
    )
    assert (
        execute_next_research_item(store=store, cwd=tmp_path, create_branch=True).status
        == "executed"
    )
    checkout = "unrelated"
    result = execute_next_research_item(store=store, cwd=tmp_path, create_branch=True, commit=True)
    assert result.status == "executed"
    assert calls == ["checkout", "checkout", "commit"]


def test_only_evidence_blocked_work_returns_actionable_deferred(tmp_path):
    store, _candidate = promotion_store(tmp_path)
    for _ in range(2):
        result = execute_next_research_item(store=store, cwd=tmp_path, commit=True)
        assert result.status == "deferred"
        assert "no linked experiment" in result.message


def test_evidence_is_rechecked_after_commit_before_push_and_progress_persists(
    tmp_path, monkeypatch
):
    from harness.core.promotion_evidence import PromotionEvidenceError

    store, candidate = promotion_store(tmp_path)
    calls = []
    fresh = True

    def gate(**kwargs):
        calls.append("gate")
        if not fresh:
            raise PromotionEvidenceError("commit hook changed verified input")

    def commit(**kwargs):
        nonlocal fresh
        calls.append("commit")
        fresh = False

    monkeypatch.setattr("harness.core.promotion_evidence.require_promotion_evidence", gate)
    monkeypatch.setattr("harness.core.autonomy.commit_paths", commit)
    monkeypatch.setattr("harness.core.autonomy.push_branch", lambda **kwargs: calls.append("push"))
    monkeypatch.setattr(
        "harness.core.autonomy.create_pull_request", lambda **kwargs: calls.append("pr")
    )
    result = execute_next_research_item(
        store=store, cwd=tmp_path, commit=True, push=True, open_pr=True
    )
    assert result.status == "deferred"
    assert calls == ["gate", "commit", "gate"]
    record = json.loads(
        (store.promotion_candidates_dir / candidate.id / "promotion_execution.json").read_text()
    )
    assert record["stages"]["commit"] is True
    assert not record["stages"].get("push")
    fresh = True
    result = execute_next_research_item(
        store=store, cwd=tmp_path, commit=True, push=True, open_pr=True
    )
    assert result.status == "executed"
    assert calls == ["gate", "commit", "gate", "gate", "push", "gate", "pr"]


@pytest.mark.parametrize(
    "latest_status,tied", [("failed", False), ("running", False), ("passed", True)]
)
def test_candidate_creation_requires_unique_latest_passed_experiment(tmp_path, latest_status, tied):
    from harness.core.experiments import Experiment, ExperimentResult
    from harness.core.hypotheses import Hypothesis
    from harness.core.research_scheduler import build_research_queue

    store = ResearchStore(root=default_research_root(tmp_path))
    store.add_opportunity(Opportunity(id="o", title="Opportunity", summary="Improve"))
    store.add_hypothesis(
        Hypothesis(
            id="h",
            opportunity_id="o",
            claim="Claim",
            expected_win="Win",
            risk_level="low",
            change_mode="improve",
        )
    )
    store.add_experiment_plan(
        ExperimentPlan(
            id="p", hypothesis_id="h", plan="Check", checks=("true",), target_files=("source.py",)
        )
    )
    old = Experiment(id="old", plan_id="p", created_at="2026-01-01T00:00:00+00:00")
    new = replace(old, id="new", created_at=old.created_at if tied else "2026-01-02T00:00:00+00:00")
    for experiment, status in [(old, "passed"), (new, latest_status)]:
        store.add_experiment(
            experiment,
            ExperimentResult(
                experiment_id=experiment.id,
                status=status,
                command_results=(),
                started_at=experiment.created_at,
                finished_at=experiment.created_at,
                duration_seconds=0,
            ),
        )
    assert not any(item.kind == "experiment_result" for item in build_research_queue(store))
    assert execute_next_research_item(store=store, cwd=tmp_path).status == "no_work"
    assert not store.list_promotion_candidates()


def test_high_priority_opportunity_advances_without_duplicate_children(tmp_path, monkeypatch):
    store = ResearchStore(root=default_research_root(tmp_path))
    store.add_opportunity(
        Opportunity(id="opp", title="Improve", summary="Measure it", priority="high")
    )

    def fake_experiment(**kwargs):
        return run_experiment_plan(
            store=kwargs["store"],
            cwd=kwargs["cwd"],
            plan=ExperimentPlan(
                id=kwargs["plan"].id,
                hypothesis_id=kwargs["plan"].hypothesis_id,
                plan="Known failing evidence",
                checks=("exit 1",),
            ),
        )

    monkeypatch.setattr("harness.core.autonomy.run_experiment_plan", fake_experiment)
    result = execute_research_burst(store=store, cwd=tmp_path, max_steps=5)
    assert [step.queue_item_kind for step in result.results[:3]] == [
        "opportunity",
        "hypothesis",
        "experiment_plan",
    ]
    assert len(store.list_hypotheses()) == 1
    assert len(store.list_experiment_plans()) == 1


def test_unknown_has_one_durable_continuation(tmp_path: Path):
    store = ResearchStore(root=default_research_root(tmp_path))
    store.add_theme(Theme(id="theme", vision_id="current", title="Theme", description="Explore"))
    store.add_unknown(
        Unknown(id="unknown", theme_id="theme", question="Why?", why_it_matters="Learn")
    )
    assert execute_next_research_item(store=store, cwd=tmp_path).queue_item_kind == "unknown"
    assert execute_next_research_item(store=store, cwd=tmp_path).status == "no_work"
    assert len(store.list_rabbit_holes()) == 1


def test_empty_and_timed_out_experiments_are_not_successful(tmp_path: Path):
    store = ResearchStore(root=default_research_root(tmp_path))
    _, empty = run_experiment_plan(
        store=store, cwd=tmp_path, plan=ExperimentPlan(id="empty", hypothesis_id="h", plan="Empty")
    )
    assert empty.status == "inconclusive"
    experiment, timed = run_experiment_plan(
        store=store,
        cwd=tmp_path,
        timeout=0.02,
        plan=ExperimentPlan(id="slow", hypothesis_id="h", plan="Slow", checks=("sleep 2",)),
    )
    assert timed.status == "timed_out"
    assert timed.command_results[0].exit_code == 124
    assert store.load_experiment_result(experiment.id).status == "timed_out"
    assert Path(timed.command_results[0].stderr_path).is_file()
