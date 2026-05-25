from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.cli import evals as eval_cli


def test_eval_workflow_runs_fixture_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    evals_root = tmp_path / "evals"
    (evals_root / "fixtures").mkdir(parents=True)

    workflow_fixture = SimpleNamespace(
        name="01-demo",
        path=evals_root / "workflow-fixtures" / "01-demo",
    )

    def _run_fixture(*_args, **kwargs):
        return SimpleNamespace(
            fixture_name="01-demo",
            passed=True,
            steps_total=3,
            steps_passed=3,
            step_results=[],
            artifact_dir=kwargs.get("artifact_dir"),
            duration_seconds=0.3,
            to_dict=lambda: {
                "fixture_name": "01-demo",
                "passed": True,
                "steps_total": 3,
                "steps_passed": 3,
                "step_results": [],
                "artifact_dir": str(kwargs.get("artifact_dir")),
                "duration_seconds": 0.3,
            },
        )

    class FakeWorkflowReport:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to_dict(self):
            return {
                "run_id": self.kwargs["run_id"],
                "artifact_root": str(self.kwargs["artifact_root"]),
                "results": [result.to_dict() for result in self.kwargs["results"]],
            }

    monkeypatch.setattr(eval_cli, "_find_evals_root", lambda: evals_root)

    def _load_eval_module(name: str, _root: Path):
        if name == "workflow_runner":
            return SimpleNamespace(
                discover_workflow_fixtures=lambda *_args, **_kwargs: [workflow_fixture],
                run_workflow_fixture=_run_fixture,
                WorkflowEvalReport=FakeWorkflowReport,
            )
        raise AssertionError(f"unexpected module load: {name}")

    monkeypatch.setattr(eval_cli, "_load_eval_module", _load_eval_module)

    out_dir = tmp_path / "workflow-artifacts"
    result = CliRunner().invoke(
        cli_main.app,
        [
            "eval",
            "workflow",
            "01-demo",
            "--output-dir",
            str(out_dir),
            "--json-out",
        ],
    )

    assert result.exit_code == 0, result.stdout
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["results"][0]["fixture_name"] == "01-demo"
    assert report["results"][0]["steps_passed"] == 3
