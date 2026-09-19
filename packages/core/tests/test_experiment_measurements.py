from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from harness.core.experiment_plans import ExperimentPlan
from harness.core.experiment_runner import compare_experiment_results, run_experiment_plan
from harness.core.research.store import ResearchStore, default_research_root
from harness.core.workspace_snapshot import workspace_fingerprint


def test_measurement_comparison_and_freshness(tmp_path):
    store = ResearchStore(root=default_research_root(tmp_path))
    (tmp_path / "module.py").write_text("VALUE = 1\n")
    plan = ExperimentPlan(
        id="baseline",
        hypothesis_id="hyp",
        plan="Measure latency",
        checks=("exit 0",),
        measurement_command="printf '{\"latency\":20}'",
        metric_directions={"latency": "minimize"},
    )
    first, baseline = run_experiment_plan(store=store, plan=plan, cwd=tmp_path)
    (tmp_path / "module.py").write_text("VALUE = 2\n")
    plan = replace(
        plan,
        id="candidate",
        measurement_command="printf '{\"latency\":12}'",
        baseline_experiment_id=first.id,
    )
    experiment, result = run_experiment_plan(store=store, plan=plan, cwd=tmp_path)
    assert result.status == "passed"
    assert result.metrics == {"latency": 12}
    assert (
        result.workspace_fingerprint
        and result.workspace_fingerprint != baseline.workspace_fingerprint
    )
    restored = store.load_experiment_result(experiment.id)
    assert restored.metrics == result.metrics
    assert restored.baseline_experiment_id == first.id
    comparison = compare_experiment_results(baseline, result)
    assert comparison["metrics"] == {
        "latency": {"left": 20, "right": 12, "delta": -8, "direction": "minimize", "improved": True}
    }


def test_invalid_measurement_does_not_pass(tmp_path):
    store = ResearchStore(root=default_research_root(tmp_path))
    for output in ["{}", '{"latency":true}', '{"latency":NaN}', '{"other":12}']:
        plan = ExperimentPlan(
            id="bad",
            hypothesis_id="hyp",
            plan="Measure",
            measurement_command=f"printf '%s' '{output}'",
            metric_directions={"latency": "minimize"},
        )
        _, result = run_experiment_plan(store=store, plan=plan, cwd=tmp_path)
        assert result.status == "failed", output
        assert result.command_results[0].exit_code != 0


def test_checks_cannot_certify_changed_inputs(tmp_path):
    store = ResearchStore(root=default_research_root(tmp_path))
    (tmp_path / "source.txt").write_text("before")
    _, result = run_experiment_plan(
        store=store,
        cwd=tmp_path,
        plan=ExperimentPlan(
            id="changes",
            hypothesis_id="hyp",
            plan="Unstable check",
            checks=("printf after > source.txt",),
        ),
    )
    assert result.workspace_fingerprint == ""


def test_measurement_plan_round_trips():
    plan = ExperimentPlan(
        id="p",
        hypothesis_id="h",
        plan="Measure",
        measurement_command="measure",
        metric_directions={"rate": "maximize"},
        baseline_experiment_id="baseline",
    )
    assert ExperimentPlan.from_dict(json.loads(json.dumps(plan.to_dict()))) == plan


def test_fingerprint_handles_missing_git_and_detects_file_modes(tmp_path, monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git not installed")

    monkeypatch.setattr(subprocess, "run", missing_git)
    script = tmp_path / "check.sh"
    script.write_text("exit 0")
    script.chmod(0o644)
    first = workspace_fingerprint(tmp_path)
    script.chmod(0o755)
    assert workspace_fingerprint(tmp_path) != first
    state = tmp_path / ".harness"
    state.mkdir()
    before_state = workspace_fingerprint(tmp_path)
    (state / "result.json").write_text("{}")
    assert workspace_fingerprint(tmp_path) == before_state


def test_completed_experiment_stops_background_commands(tmp_path):
    store = ResearchStore(root=default_research_root(tmp_path))
    _, result = run_experiment_plan(
        store=store,
        cwd=tmp_path,
        plan=ExperimentPlan(
            id="background",
            hypothesis_id="h",
            plan="Bounded command",
            checks=("sleep 60 >/dev/null 2>&1 & echo $!",),
        ),
    )
    pid = Path(result.command_results[0].stdout_path).read_text().strip()
    process = subprocess.run(["ps", "-p", pid, "-o", "stat="], capture_output=True, text=True)
    assert not process.stdout.strip() or process.stdout.strip().startswith("Z")


def test_interrupted_experiment_saves_partial_outcome_and_can_be_retried(tmp_path, monkeypatch):
    store = ResearchStore(root=default_research_root(tmp_path))
    original = subprocess.Popen.communicate
    interrupted = []

    def interrupt_once(process, *args, **kwargs):
        if process.args == "sleep 60" and not interrupted:
            interrupted.append(process)
            raise KeyboardInterrupt
        return original(process, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupt_once)
    plan = ExperimentPlan(
        id="interrupt", hypothesis_id="h", plan="Interrupted check", checks=("exit 0", "sleep 60")
    )
    with pytest.raises(KeyboardInterrupt):
        run_experiment_plan(store=store, cwd=tmp_path, plan=plan)
    experiments = store.list_experiments()
    assert len(experiments) == 1
    result = store.load_experiment_result(experiments[0].id)
    assert result.status == "interrupted"
    assert [command.exit_code for command in result.command_results] == [0, 130]
    assert result.finished_at
    assert not result.workspace_fingerprint
    assert interrupted[0].poll() is not None
    retry, completed = run_experiment_plan(
        store=store, cwd=tmp_path, plan=replace(plan, checks=("exit 0",))
    )
    assert retry.id != experiments[0].id
    assert completed.status == "passed"
