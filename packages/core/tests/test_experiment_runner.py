from __future__ import annotations

import os
from pathlib import Path

from harness.core.experiment_plans import ExperimentPlan
from harness.core.experiment_runner import run_experiment_plan
from harness.core.research_store import ResearchStore, default_research_root


def test_run_experiment_plan_expands_known_eval_slice_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    fake_uv.write_text(
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > uv-args.txt\n",
        encoding="utf-8",
    )
    fake_uv.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")

    store = ResearchStore(root=default_research_root(tmp_path))
    plan = ExperimentPlan(
        id="plan-workflow-smoke",
        hypothesis_id="hyp-workflow",
        plan="Run workflow smoke.",
        eval_slices=("workflow-smoke",),
    )

    _experiment, result = run_experiment_plan(store=store, plan=plan, cwd=tmp_path)

    assert result.status == "passed"
    assert result.command_results[0].kind == "eval"
    assert (
        result.command_results[0].command
        == "uv run harness eval workflow --suite workflow-smoke --timeout 120"
    )
    assert (tmp_path / "uv-args.txt").read_text(encoding="utf-8").splitlines() == [
        "run",
        "harness",
        "eval",
        "workflow",
        "--suite",
        "workflow-smoke",
        "--timeout",
        "120",
    ]
