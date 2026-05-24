from __future__ import annotations

import json
from pathlib import Path

from evals import research_runner


def _write_research_fixture(root: Path, name: str) -> Path:
    fixture = root / "evals" / "research-fixtures" / name
    (fixture / "workspace" / "docs").mkdir(parents=True)
    (fixture / "TASK.md").write_text("Research topic here.", encoding="utf-8")
    (fixture / "expected.json").write_text(
        json.dumps(
            {
                "summary_contains": "SQLite",
                "min_findings": 1,
                "min_sources": 1,
                "required_findings": ["zero-setup"],
                "required_sources": [{"url_substring": "docs/persistence.md"}],
            }
        ),
        encoding="utf-8",
    )
    (fixture / "workspace" / "docs" / "persistence.md").write_text(
        "# Persistence\nSQLite favors zero-setup local storage.\n",
        encoding="utf-8",
    )
    return fixture


def test_discover_research_fixtures_reads_expected_layout(tmp_path: Path) -> None:
    _write_research_fixture(tmp_path, "01-demo")
    fixtures = research_runner.discover_research_fixtures(tmp_path / "evals")
    assert [fixture.name for fixture in fixtures] == ["01-demo"]
    assert fixtures[0].required_sources[0].url_substring == "docs/persistence.md"


def test_evaluate_research_memo_requires_expected_finding_and_source(tmp_path: Path) -> None:
    _write_research_fixture(tmp_path, "01-demo")
    fixture = research_runner.discover_research_fixtures(tmp_path / "evals")[0]
    parsed = research_runner.parse_research_memo(
        json.dumps(
            {
                "summary": "SQLite is the default here.",
                "findings": ["SQLite favors zero-setup local storage."],
                "open_questions": [],
                "sources": [
                    {
                        "title": "Persistence",
                        "url": "docs/persistence.md",
                        "excerpt": "SQLite favors zero-setup local storage.",
                    }
                ],
            }
        )
    )
    passed, matched_findings, matched_sources, missing = research_runner.evaluate_research_memo(
        fixture,
        parsed,
    )
    assert passed is True
    assert matched_findings == 1
    assert matched_sources == 1
    assert missing == []


def test_run_research_fixture_executes_research_command_and_persists_artifacts(
    tmp_path: Path,
) -> None:
    _write_research_fixture(tmp_path, "01-demo")
    fixture = research_runner.discover_research_fixtures(tmp_path / "evals")[0]
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
                '  "summary": "SQLite is the default here.",',
                '  "findings": ["SQLite favors zero-setup local storage."],',
                '  "open_questions": [],',
                '  "sources": [',
                '    {"title": "Persistence", "url": "docs/persistence.md", "excerpt": "SQLite favors zero-setup local storage."}',
                "  ]",
                "}))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    harness_bin.chmod(0o755)
    artifact_dir = tmp_path / "artifacts"

    result = research_runner.run_research_fixture(
        fixture,
        provider="mock",
        model="mock-model",
        harness_bin=str(harness_bin),
        artifact_dir=artifact_dir,
        timeout=30,
    )

    argv = json.loads(log_path.read_text(encoding="utf-8"))
    assert argv[:2] == ["research", "Research topic here."]
    assert "--cwd" in argv
    assert result.passed is True
    assert result.matched_findings == 1
    assert result.matched_sources == 1
    assert (artifact_dir / "research_output.txt").exists()
    assert (artifact_dir / "result.json").exists()


def test_evaluate_research_memo_matches_semantic_tradeoff_wording(tmp_path: Path) -> None:
    fixture = tmp_path / "evals" / "research-fixtures" / "01-demo"
    (fixture / "workspace" / "docs").mkdir(parents=True)
    (fixture / "TASK.md").write_text("Research topic here.", encoding="utf-8")
    (fixture / "expected.json").write_text(
        json.dumps(
            {
                "summary_contains": "SQLite",
                "min_findings": 1,
                "min_sources": 1,
                "required_findings": ["concurrent writes"],
                "required_sources": [{"url_substring": "docs/persistence.md"}],
            }
        ),
        encoding="utf-8",
    )
    (fixture / "workspace" / "docs" / "persistence.md").write_text(
        "# Persistence\nTradeoffs: limited concurrent writes.\n",
        encoding="utf-8",
    )
    discovered = research_runner.discover_research_fixtures(tmp_path / "evals")[0]
    parsed = research_runner.parse_research_memo(
        json.dumps(
            {
                "summary": "SQLite is the default here.",
                "findings": [
                    "The primary architectural tradeoff is limited concurrency for writes."
                ],
                "open_questions": [],
                "sources": [
                    {
                        "title": "Persistence",
                        "url": "docs/persistence.md",
                        "excerpt": "Tradeoffs: limited concurrent writes.",
                    }
                ],
            }
        )
    )
    passed, matched_findings, matched_sources, missing = research_runner.evaluate_research_memo(
        discovered,
        parsed,
    )
    assert passed is True
    assert matched_findings == 1
    assert matched_sources == 1
    assert missing == []
