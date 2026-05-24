from __future__ import annotations

import json
from pathlib import Path

from evals import docs_runner


def _write_docs_fixture(root: Path, name: str) -> Path:
    fixture = root / "evals" / "docs-fixtures" / name
    (fixture / "workspace").mkdir(parents=True)
    (fixture / "TASK.md").write_text("Audit plugin docs.", encoding="utf-8")
    (fixture / "expected.json").write_text(
        json.dumps(
            {
                "summary_contains": "plugin",
                "min_findings": 1,
                "required_findings": [
                    {
                        "path": "README.md",
                        "severity": "medium",
                        "issue_substring": "plugin",
                    }
                ],
                "missing_topics": ["plugin setup"],
            }
        ),
        encoding="utf-8",
    )
    (fixture / "workspace" / "README.md").write_text("# Harness\n", encoding="utf-8")
    return fixture


def test_discover_docs_fixtures_reads_expected_layout(tmp_path: Path) -> None:
    _write_docs_fixture(tmp_path, "01-demo")
    fixtures = docs_runner.discover_docs_fixtures(tmp_path / "evals")
    assert [fixture.name for fixture in fixtures] == ["01-demo"]
    assert fixtures[0].required_findings[0].path == "README.md"


def test_evaluate_docs_report_requires_expected_finding_and_topic(tmp_path: Path) -> None:
    _write_docs_fixture(tmp_path, "01-demo")
    fixture = docs_runner.discover_docs_fixtures(tmp_path / "evals")[0]
    parsed = docs_runner.parse_docs_audit_report(
        json.dumps(
            {
                "summary": "Plugin docs are incomplete.",
                "findings": [
                    {
                        "severity": "medium",
                        "path": "README.md",
                        "issue": "Plugin setup is missing.",
                        "rationale": "Users cannot discover the extension flow.",
                    }
                ],
                "missing_topics": ["plugin setup"],
            }
        )
    )
    passed, matched, matched_topics, missing = docs_runner.evaluate_docs_report(
        fixture,
        parsed,
    )
    assert passed is True
    assert matched == 1
    assert matched_topics == 1
    assert missing == []


def test_run_docs_fixture_executes_docs_command_and_persists_artifacts(
    tmp_path: Path,
) -> None:
    _write_docs_fixture(tmp_path, "01-demo")
    fixture = docs_runner.discover_docs_fixtures(tmp_path / "evals")[0]
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
                '  "summary": "Plugin docs are incomplete.",',
                '  "findings": [',
                '    {"severity": "medium", "path": "README.md", "issue": "Plugin setup is missing.", "rationale": "Users cannot discover the extension flow."}',
                "  ],",
                '  "missing_topics": ["plugin setup"]',
                "}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    harness_bin.chmod(0o755)
    artifact_dir = tmp_path / "artifacts"

    result = docs_runner.run_docs_fixture(
        fixture,
        provider="mock",
        model="mock-model",
        harness_bin=str(harness_bin),
        artifact_dir=artifact_dir,
        timeout=30,
    )

    argv = json.loads(log_path.read_text(encoding="utf-8"))
    assert argv[:2] == ["docs-audit", "Audit plugin docs."]
    assert "--cwd" in argv
    assert result.passed is True
    assert result.matched_expectations == 1
    assert result.matched_topics == 1
    assert (artifact_dir / "docs_output.txt").exists()
    assert (artifact_dir / "result.json").exists()
