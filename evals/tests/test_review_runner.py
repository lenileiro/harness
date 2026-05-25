from __future__ import annotations

import json
from pathlib import Path

from evals import review_runner


def _write_review_fixture(root: Path, name: str) -> Path:
    fixture = root / "evals" / "review-fixtures" / name
    (fixture / "base" / "src").mkdir(parents=True)
    (fixture / "head" / "src").mkdir(parents=True)
    (fixture / "TASK.md").write_text("# Review fixture\n", encoding="utf-8")
    (fixture / "expected.json").write_text(
        json.dumps(
            {
                "summary_contains": "review",
                "min_findings": 1,
                "required_findings": [
                    {
                        "file": "src/demo.py",
                        "severity": "high",
                        "issue_substring": "None",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (fixture / "base" / "src" / "demo.py").write_text(
        "def render(user):\n    return 'ok'\n",
        encoding="utf-8",
    )
    (fixture / "head" / "src" / "demo.py").write_text(
        "def render(user):\n    return user['name']\n",
        encoding="utf-8",
    )
    return fixture


def test_discover_review_fixtures_reads_expected_layout(tmp_path: Path) -> None:
    _write_review_fixture(tmp_path, "01-demo")
    fixtures = review_runner.discover_review_fixtures(tmp_path / "evals")
    assert [fixture.name for fixture in fixtures] == ["01-demo"]
    assert fixtures[0].required_findings[0].file == "src/demo.py"


def test_evaluate_review_report_requires_expected_finding(tmp_path: Path) -> None:
    _write_review_fixture(tmp_path, "01-demo")
    fixture = review_runner.discover_review_fixtures(tmp_path / "evals")[0]
    parsed = review_runner.parse_review_report(
        json.dumps(
            {
                "summary": "review found a real issue",
                "findings": [
                    {
                        "severity": "high",
                        "file": "src/demo.py",
                        "line": 2,
                        "issue": "None input will crash here",
                        "rationale": "The code dereferences user without checking for None.",
                    }
                ],
            }
        )
    )
    passed, matched, missing = review_runner.evaluate_review_report(fixture, parsed)
    assert passed is True
    assert matched == 1
    assert missing == []


def test_evaluate_review_report_matches_issue_substring_in_rationale(tmp_path: Path) -> None:
    _write_review_fixture(tmp_path, "01-demo")
    fixture = review_runner.discover_review_fixtures(tmp_path / "evals")[0]
    parsed = review_runner.parse_review_report(
        json.dumps(
            {
                "summary": "Structured regression findings.",
                "findings": [
                    {
                        "severity": "high",
                        "file": "src/demo.py",
                        "line": 2,
                        "issue": "Potential crash on null input",
                        "rationale": "The code dereferences user without checking for None.",
                    }
                ],
            }
        )
    )
    passed, matched, missing = review_runner.evaluate_review_report(fixture, parsed)
    assert passed is True
    assert matched == 1
    assert missing == []


def test_run_review_fixture_executes_review_command_and_persists_artifacts(
    tmp_path: Path,
) -> None:
    fixture_dir = _write_review_fixture(tmp_path, "01-demo")
    fixture = review_runner.discover_review_fixtures(tmp_path / "evals")[0]
    log_path = tmp_path / "invocation.json"
    harness_bin = tmp_path / "fake-harness"
    harness_bin.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import json, pathlib, sys",
                f"log_path = pathlib.Path({str(log_path)!r})",
                "log_path.write_text(json.dumps(sys.argv[1:]))",
                "print(json.dumps({",
                '  "summary": "review found a real issue",',
                '  "findings": [',
                '    {"severity": "high", "file": "src/demo.py", "line": 2, "issue": "None input will crash here", "rationale": "The code dereferences user without checking for None."}',
                "  ]",
                "}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    harness_bin.chmod(0o755)
    artifact_dir = tmp_path / "artifacts"

    result = review_runner.run_review_fixture(
        fixture,
        provider="mock",
        model="mock-model",
        harness_bin=str(harness_bin),
        artifact_dir=artifact_dir,
        timeout=30,
    )

    argv = json.loads(log_path.read_text(encoding="utf-8"))
    assert argv[:2] == ["review", "--cwd"]
    assert "--base" in argv
    assert "HEAD~1" in argv
    assert result.passed is True
    assert result.matched_expectations == 1
    assert (artifact_dir / "review_output.txt").exists()
    assert (artifact_dir / "result.json").exists()
    assert fixture_dir.exists()
