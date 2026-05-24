from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.cli import evals as eval_cli


def test_eval_review_runs_fixture_and_writes_report(tmp_path: Path, monkeypatch) -> None:
    evals_root = tmp_path / "evals"
    (evals_root / "fixtures").mkdir(parents=True)

    review_fixture = SimpleNamespace(
        name="01-demo",
        path=evals_root / "review-fixtures" / "01-demo",
    )

    def _run_fixture(*_args, **kwargs):
        return SimpleNamespace(
            fixture_name="01-demo",
            passed=True,
            findings_count=1,
            matched_expectations=1,
            missing_expectations=[],
            summary="review found a real issue",
            artifact_dir=kwargs.get("artifact_dir"),
            raw_output='{"summary":"review found a real issue","findings":[]}',
            report={"summary": "review found a real issue", "findings": []},
            duration_seconds=0.2,
            to_dict=lambda: {
                "fixture_name": "01-demo",
                "passed": True,
                "findings_count": 1,
                "matched_expectations": 1,
                "missing_expectations": [],
                "summary": "review found a real issue",
                "artifact_dir": str(kwargs.get("artifact_dir")),
                "raw_output": '{"summary":"review found a real issue","findings":[]}',
                "report": {"summary": "review found a real issue", "findings": []},
                "duration_seconds": 0.2,
            },
        )

    class FakeReviewReport:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def to_dict(self):
            return {
                "run_id": self.kwargs["run_id"],
                "provider": self.kwargs["provider"],
                "model": self.kwargs["model"],
                "artifact_root": str(self.kwargs["artifact_root"]),
                "results": [result.to_dict() for result in self.kwargs["results"]],
            }

    monkeypatch.setattr(eval_cli, "_find_evals_root", lambda: evals_root)

    def _load_eval_module(name: str, _root: Path):
        if name == "review_runner":
            return SimpleNamespace(
                discover_review_fixtures=lambda *_args, **_kwargs: [review_fixture],
                run_review_fixture=_run_fixture,
                ReviewEvalReport=FakeReviewReport,
            )
        raise AssertionError(f"unexpected module load: {name}")

    monkeypatch.setattr(eval_cli, "_load_eval_module", _load_eval_module)

    out_dir = tmp_path / "review-artifacts"
    result = CliRunner().invoke(
        cli_main.app,
        [
            "eval",
            "review",
            "01-demo",
            "--provider",
            "mock",
            "--model",
            "mock-model",
            "--output-dir",
            str(out_dir),
            "--json-out",
        ],
    )

    assert result.exit_code == 0, result.stdout
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["provider"] == "mock"
    assert report["model"] == "mock-model"
    assert report["results"][0]["fixture_name"] == "01-demo"
