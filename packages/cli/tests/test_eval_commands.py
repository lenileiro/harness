"""CLI tests for eval helper commands."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

from harness.cli import __main__ as cli_main


def test_eval_calibrate_reads_gold_labels(tmp_path: Path, monkeypatch) -> None:
    repo_evals_root = Path(__file__).resolve().parents[3] / "evals"
    evals_root = tmp_path / "evals"
    (evals_root / "fixtures").mkdir(parents=True)
    (evals_root / "gold").mkdir(parents=True)
    (evals_root / "gold" / "label.json").write_text(
        json.dumps(
            {
                "fixture_name": "01-demo",
                "variant": "defended",
                "run_index": 1,
                "scores": {
                    "verification": 5,
                    "scope": 5,
                    "decomposition": 5,
                    "correctness": 5,
                    "pushback": 5,
                    "epistemic": 5,
                    "overall": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "fixture_name": "01-demo",
                        "variant": "defended",
                        "run_index": 1,
                        "verification": {"score": 5},
                        "scope": {"score": 5},
                        "decomposition": {"score": 5},
                        "correctness": {"score": 5},
                        "pushback": {"score": 5},
                        "epistemic": {"score": 5},
                        "overall": {"score": 5},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(cli_main, "_find_evals_root", lambda: repo_evals_root)
    result = CliRunner().invoke(
        cli_main.app,
        ["eval", "calibrate", str(report), "--gold-dir", str(evals_root / "gold")],
    )

    assert result.exit_code == 0, result.stdout
    assert "Judge calibration" in result.stdout
    assert "verification" in result.stdout


def test_eval_validate_passes_against_repo_assets(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.chdir(repo_root)

    result = CliRunner().invoke(cli_main.app, ["eval", "validate"])

    assert result.exit_code == 0, result.stdout
    assert "Eval asset validation" in result.stdout


def test_eval_run_no_judge_uses_hard_metrics_only(tmp_path: Path, monkeypatch) -> None:
    repo_evals_root = Path(__file__).resolve().parents[3] / "evals"
    eval_types = cli_main._load_eval_module("types", repo_evals_root)
    evals_root = tmp_path / "evals"
    fixtures_dir = evals_root / "fixtures" / "01-demo"
    fixtures_dir.mkdir(parents=True)
    (fixtures_dir / "TASK.md").write_text("# Fix demo bug\n", encoding="utf-8")
    (fixtures_dir / "EVAL.md").write_text("Judge notes", encoding="utf-8")

    monkeypatch.setattr(cli_main, "_find_evals_root", lambda: evals_root)

    fixture = eval_types.FixtureMeta(
        name="01-demo",
        path=fixtures_dir,
        task_text="# Fix demo bug\n",
        eval_md="Judge notes",
        verify_command="pytest -q",
    )
    hard_metrics = eval_types.HardMetrics(
        verify_passed=True,
        run_exit_code=0,
        verify_exit_code=0,
        files_touched=1,
        lines_added=1,
        lines_deleted=1,
        tool_calls=3,
        shell_commands=1,
        did_run_verification=True,
        agent_duration_seconds=1.0,
        verify_duration_seconds=0.1,
        total_duration_seconds=1.1,
    )

    runner_mod = SimpleNamespace(
        discover_fixtures=lambda *_args, **_kwargs: [fixture],
        run_fixture=lambda *_args, **kwargs: eval_types.RunOutcome(
            fixture=fixture,
            transcript="defense ledger:\n  verifiers: shell=1✓/0✗\n",
            git_diff="diff --git a/src/demo.py b/src/demo.py",
            test_output="1 passed",
            agent_exit_code=0,
            test_exit_code=0,
            variant=kwargs.get("variant", "defended"),
            hard_metrics=hard_metrics,
            artifact_dir=kwargs.get("artifact_dir"),
            agent_command=["harness", "run"],
            verify_command="pytest -q",
        ),
    )

    def _load_eval_module(name: str, _root: Path):
        if name == "runner":
            return runner_mod
        if name == "types":
            return eval_types
        if name == "mutator":
            return SimpleNamespace()
        raise AssertionError(f"unexpected module load: {name}")

    monkeypatch.setattr(cli_main, "_load_eval_module", _load_eval_module)

    def _boom(*_args, **_kwargs):
        raise AssertionError("judge adapter should not be built in --no-judge mode")

    monkeypatch.setattr(cli_main, "_build_adapter", _boom)

    out_dir = tmp_path / "artifacts"
    result = CliRunner().invoke(
        cli_main.app,
        [
            "eval",
            "run",
            "01-demo",
            "--provider",
            "mock",
            "--model",
            "mock-model",
            "--no-judge",
            "--output-dir",
            str(out_dir),
            "--json-out",
            "--no-save-history",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "hard metrics only" in result.stdout
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert report["no_judge"] is True
    assert report["judge_provider"] is None
    assert report["judge_model"] is None
    assert report["aggregates"][0]["passes"] == 1
    assert report["aggregates"][0]["hard_metrics"]["verify_pass_rate"] == 1.0


def test_eval_adjustments_lists_saved_rows(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    evals_root = tmp_path / "evals"
    (evals_root / "fixtures").mkdir(parents=True)
    run_dir = evals_root / "runs" / "demo" / "01" / "run-01"
    run_dir.mkdir(parents=True)
    (run_dir / "harness_adjustments.json").write_text(
        json.dumps(
            [
                {
                    "id": "adj_1",
                    "kind": "tests_first",
                    "text": "run the targeted test before editing",
                    "weight": 2.4,
                    "source_fixture_name": "01-demo",
                    "source_variant": "defended",
                }
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(cli_main, "_find_evals_root", lambda: repo_root / "evals")
    result = CliRunner().invoke(
        cli_main.app,
        ["eval", "adjustments", str(evals_root / "runs")],
    )

    assert result.exit_code == 0, result.stdout
    assert "Harness adjustments" in result.stdout
    assert "tests_first" in result.stdout
    assert "01-demo" in result.stdout


def test_eval_export_adjustments_writes_jsonl(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    evals_root = tmp_path / "evals"
    (evals_root / "fixtures").mkdir(parents=True)
    run_dir = evals_root / "runs" / "demo" / "01" / "run-01"
    run_dir.mkdir(parents=True)
    (run_dir / "harness_adjustments.json").write_text(
        json.dumps(
            [
                {
                    "id": "adj_1",
                    "kind": "family_pattern",
                    "text": "keep the patch small",
                    "weight": 2.0,
                    "source_fixture_name": "02-demo",
                    "source_variant": "bare",
                }
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "adjustments.jsonl"

    monkeypatch.setattr(cli_main, "_find_evals_root", lambda: repo_root / "evals")
    result = CliRunner().invoke(
        cli_main.app,
        [
            "eval",
            "export-adjustments",
            str(output),
            "--root",
            str(evals_root / "runs"),
        ],
    )

    assert result.exit_code == 0, result.stdout
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["kind"] == "family_pattern"
