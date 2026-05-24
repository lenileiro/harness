"""Tests for fixture discovery."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals import runner


def _write_fixture(root: Path, name: str, *, metadata: str = "") -> Path:
    fixture = root / "evals" / "fixtures" / name
    fixture.mkdir(parents=True)
    (fixture / "TASK.md").write_text("Fix it.\n", encoding="utf-8")
    (fixture / "EVAL.md").write_text(
        "primary_dimension: verification\n\ntrap: >\n  run tests first\n",
        encoding="utf-8",
    )
    (fixture / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
    (fixture / "fixture.yaml").write_text(metadata, encoding="utf-8")
    return fixture


class TestDiscoverFixtures:
    def test_reads_fixture_yaml_metadata(self, tmp_path: Path) -> None:
        _write_fixture(
            tmp_path,
            "01-demo",
            metadata=(
                "verify_command: python -c \"print('ok')\"\n"
                "family: regression\n"
                "behavior_category: verification\n"
                "expected_first_step: run tests\n"
                "disallowed_paths:\n"
                "  - src/other.py\n"
            ),
        )

        fixtures = runner.discover_fixtures(tmp_path / "evals")

        assert len(fixtures) == 1
        fixture = fixtures[0]
        assert fixture.verify_command == "python -c \"print('ok')\""
        assert fixture.family == "regression"
        assert fixture.rules.expected_first_step == "run tests"
        assert fixture.rules.disallowed_paths == ["src/other.py"]

    def test_holdout_fixtures_are_excluded_by_default(self, tmp_path: Path) -> None:
        _write_fixture(tmp_path, "01-demo", metadata="holdout: true\n")

        hidden = runner.discover_fixtures(tmp_path / "evals")
        visible = runner.discover_fixtures(tmp_path / "evals", include_holdout=True)

        assert hidden == []
        assert len(visible) == 1
