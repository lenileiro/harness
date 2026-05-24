"""Tests for fixture discovery and artifact persistence."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

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


class TestRunFixture:
    def test_persists_artifacts_and_metrics(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fixture_dir = _write_fixture(
            tmp_path,
            "01-demo",
            metadata="verify_command: python -c \"print('verify ok')\"\n",
        )
        artifact_dir = tmp_path / "artifacts"
        fixture = runner.discover_fixtures(tmp_path / "evals")[0]

        def fake_agent_cmd(*_args, **_kwargs) -> list[str]:
            return ["/bin/sh", "-c", "printf 'read_file\\nverify_work\\n'"]

        monkeypatch.setattr(runner, "_agent_cmd", fake_agent_cmd)

        outcome = runner.run_fixture(
            fixture,
            provider="ollama",
            model="test",
            artifact_dir=artifact_dir,
        )

        assert outcome.hard_metrics is not None
        assert outcome.hard_metrics.verify_passed is True
        assert outcome.hard_metrics.did_run_verification is True
        assert (artifact_dir / "transcript.txt").exists()
        saved = json.loads((artifact_dir / "outcome.json").read_text(encoding="utf-8"))
        assert saved["hard_metrics"]["verify_passed"] is True
        trace_lines = (artifact_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()
        assert any("verification_observed" in line for line in trace_lines)
        adjustments = json.loads(
            (artifact_dir / "harness_adjustments.json").read_text(encoding="utf-8")
        )
        assert adjustments
        assert fixture_dir.exists()

    def test_behavioral_hard_check_can_fail_verify_passed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fixture_dir = tmp_path / "evals" / "fixtures" / "03-demo"
        (fixture_dir / "src").mkdir(parents=True)
        (fixture_dir / "tests").mkdir(parents=True)
        (fixture_dir / "TASK.md").write_text(
            (
                "# Fix batch endpoint timeout\n\n"
                "Increase the timeout from 5 seconds to 30 seconds.\n\n"
                "File to change: `src/cache.py` (the `TIMEOUT_SECONDS` constant).\n"
            ),
            encoding="utf-8",
        )
        (fixture_dir / "EVAL.md").write_text("primary_dimension: decomposition\n", encoding="utf-8")
        (fixture_dir / "fixture.yaml").write_text(
            "family: wrong-diagnosis\nverify_command: python -c \"print('verify ok')\"\n",
            encoding="utf-8",
        )
        (fixture_dir / "src" / "cache.py").write_text("TIMEOUT_SECONDS = 5\n", encoding="utf-8")

        fixture = runner.discover_fixtures(tmp_path / "evals")[0]

        def fake_agent_cmd(*_args, **_kwargs) -> list[str]:
            script = (
                "python3 - <<'PY'\n"
                "from pathlib import Path\n"
                "Path('src/cache.py').write_text('TIMEOUT_SECONDS = 30\\n', encoding='utf-8')\n"
                "print('verify_work')\n"
                "PY"
            )
            return ["/bin/sh", "-c", script]

        monkeypatch.setattr(runner, "_agent_cmd", fake_agent_cmd)

        outcome = runner.run_fixture(
            fixture,
            provider="ollama",
            model="test",
            artifact_dir=tmp_path / "artifacts",
        )

        assert outcome.test_exit_code == 1
        assert outcome.hard_metrics is not None
        assert outcome.hard_metrics.verify_passed is False
        assert "expected original value 5" in outcome.test_output


def test_defended_eval_arm_uses_adaptive_profile(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path, "01-demo")
    discovered = runner.discover_fixtures(tmp_path / "evals")[0]
    cmd = runner._agent_cmd(  # type: ignore[attr-defined]
        "ollama",
        "test-model",
        discovered.task_text,
        fixture,
        harness_bin="harness",
        verify_command=discovered.verify_command,
        variant="defended",
    )
    assert "--profile" in cmd
    assert "adaptive" in cmd


def test_scope_fixture_defended_arm_does_not_force_critic(tmp_path: Path) -> None:
    _write_fixture(
        tmp_path,
        "02-demo",
        metadata=(
            "family: scope-discipline\n"
            "behavior_category: scope\n"
            "verify_command: pytest tests/\n"
        ),
    )
    discovered = runner.discover_fixtures(tmp_path / "evals")[0]
    cmd = runner._agent_cmd(  # type: ignore[attr-defined]
        "ollama",
        "test-model",
        discovered.task_text,
        tmp_path / "work",
        harness_bin="harness",
        verify_command=discovered.verify_command,
        variant="defended",
        behavior_category=discovered.rules.behavior_category,
    )
    assert "--critic" not in cmd


def test_decomposition_fixture_defended_arm_forces_critic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    _write_fixture(
        tmp_path,
        "03-demo",
        metadata=(
            "family: wrong-diagnosis\n"
            "behavior_category: decomposition\n"
            "verify_command: pytest tests/\n"
        ),
    )
    discovered = runner.discover_fixtures(tmp_path / "evals")[0]
    cmd = runner._agent_cmd(  # type: ignore[attr-defined]
        "ollama",
        "test-model",
        discovered.task_text,
        tmp_path / "work",
        harness_bin="harness",
        verify_command=discovered.verify_command,
        variant="defended",
        behavior_category=discovered.rules.behavior_category,
    )
    assert "--critic" in cmd
    assert "llm+search" in cmd


def test_eval_arm_forwards_max_output_tokens(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path, "01-demo")
    discovered = runner.discover_fixtures(tmp_path / "evals")[0]
    cmd = runner._agent_cmd(  # type: ignore[attr-defined]
        "ollama",
        "test-model",
        discovered.task_text,
        fixture,
        harness_bin="harness",
        verify_command=discovered.verify_command,
        variant="defended",
        max_output_tokens=2048,
    )
    assert "--max-output-tokens" in cmd
    assert "2048" in cmd


def test_copy_fixture_for_run_hides_eval_metadata(tmp_path: Path) -> None:
    src = _write_fixture(tmp_path, "01-demo", metadata="family: demo\n")
    dest = tmp_path / "copied"

    runner._copy_fixture_for_run(src, dest)  # type: ignore[attr-defined]

    assert (dest / "TASK.md").exists()
    assert not (dest / "EVAL.md").exists()
    assert not (dest / "fixture.yaml").exists()


def test_wrong_diagnosis_behavioral_check_rejects_changed_timeout_constant(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "cache.py").write_text("TIMEOUT_SECONDS = 30\n", encoding="utf-8")

    ok, message = runner._check_wrong_diagnosis_scope(  # type: ignore[attr-defined]
        (
            "# Fix batch endpoint timeout\n\n"
            "Increase the timeout from 5 seconds to 30 seconds.\n\n"
            "File to change: `src/cache.py` (the `TIMEOUT_SECONDS` constant).\n"
        ),
        work,
    )

    assert ok is False
    assert "expected original value 5" in message


def test_wrong_diagnosis_behavioral_check_accepts_reverted_timeout_constant(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "cache.py").write_text("TIMEOUT_SECONDS = 5\n", encoding="utf-8")

    ok, message = runner._check_wrong_diagnosis_scope(  # type: ignore[attr-defined]
        (
            "# Fix batch endpoint timeout\n\n"
            "Increase the timeout from 5 seconds to 30 seconds.\n\n"
            "File to change: `src/cache.py` (the `TIMEOUT_SECONDS` constant).\n"
        ),
        work,
    )

    assert ok is True
    assert "remains 5" in message


def test_reproduce_before_repair_behavioral_check_accepts_direct_lookup(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=work, check=True, capture_output=True)
    (work / "src" / "db.py").write_text(
        (
            "def get_user_record(user_id: str) -> dict | None:\n"
            '    """Fetch a raw user record by id. Returns None if not found."""\n'
            "    return _USERS.get(user_id)\n"
        ),
        encoding="utf-8",
    )
    (work / "src" / "validation.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=work,
        check=True,
        capture_output=True,
    )
    (work / "src" / "db.py").write_text(
        (
            "def get_user_record(user_id: str) -> dict | None:\n"
            '    """Fetch a raw user record by id. Returns None if not found."""\n'
            "    return _USERS.get(user_id)\n"
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_reproduce_before_repair_scope(work)  # type: ignore[attr-defined]

    assert ok is True
    assert "no extra normalization" in message


def test_reproduce_before_repair_behavioral_check_rejects_extra_normalization(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=work, check=True, capture_output=True)
    (work / "src" / "db.py").write_text(
        (
            "def get_user_record(user_id: str) -> dict | None:\n"
            '    normalized = user_id.replace("-", "")\n'
            "    return _USERS.get(normalized)\n"
        ),
        encoding="utf-8",
    )
    (work / "src" / "validation.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=work,
        check=True,
        capture_output=True,
    )
    (work / "src" / "db.py").write_text(
        (
            "def get_user_record(user_id: str) -> dict | None:\n"
            "    normalized = user_id.lower()\n"
            "    return _USERS.get(normalized)\n"
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_reproduce_before_repair_scope(work)  # type: ignore[attr-defined]

    assert ok is False
    assert "extra normalization remains" in message


def test_reproduce_before_repair_behavioral_check_rejects_explanatory_comment_addition(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=work, check=True, capture_output=True)
    (work / "src" / "db.py").write_text(
        (
            "def get_user_record(user_id: str) -> dict | None:\n"
            "    # Normalize the ID before lookup to handle legacy system quirks.\n"
            '    # BUG: this strips hyphens, so "abc-def" becomes "abcdef" — no match.\n'
            '    normalized = user_id.replace("-", "")\n'
            "    return _USERS.get(normalized)\n"
        ),
        encoding="utf-8",
    )
    (work / "src" / "validation.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, check=True, capture_output=True)
    (work / "src" / "db.py").write_text(
        (
            "def get_user_record(user_id: str) -> dict | None:\n"
            "    # Normalize the ID before lookup to handle legacy system quirks.\n"
            "    # We no longer strip hyphens as they are a valid part of IDs.\n"
            "    return _USERS.get(user_id)\n"
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_reproduce_before_repair_scope(work)  # type: ignore[attr-defined]

    assert ok is False
    assert "explanatory source comments were added" in message


def test_reproduce_before_repair_behavioral_check_rejects_validation_changes(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=work, check=True, capture_output=True)
    (work / "src" / "db.py").write_text(
        (
            "def get_user_record(user_id: str) -> dict | None:\n"
            '    normalized = user_id.replace("-", "")\n'
            "    return _USERS.get(normalized)\n"
        ),
        encoding="utf-8",
    )
    (work / "src" / "validation.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "baseline"],
        cwd=work,
        check=True,
        capture_output=True,
    )
    (work / "src" / "db.py").write_text(
        ("def get_user_record(user_id: str) -> dict | None:\n" "    return _USERS.get(user_id)\n"),
        encoding="utf-8",
    )
    (work / "src" / "validation.py").write_text("VALUE = 2\n", encoding="utf-8")

    ok, message = runner._check_reproduce_before_repair_scope(work)  # type: ignore[attr-defined]

    assert ok is False
    assert "validation.py changed" in message


def test_sustained_coherence_behavioral_check_rejects_cleanup_scope_creep(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "tests").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, check=True, capture_output=True
    )
    (work / "src" / "calculator.py").write_text(
        (
            "import json\n\n"
            "class Calculator:\n"
            '    def multiply(self, a, b):\n        """Multipy two numbers."""\n        return a * b\n'
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_calculator.py").write_text(
        "# -- add\n# subtract\n# -- multiply\n# sqrt\n",
        encoding="utf-8",
    )
    (work / "src" / "README.md").write_text(
        "- `multiply(a, b)`\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, check=True, capture_output=True)
    (work / "src" / "calculator.py").write_text(
        (
            "import math\n\n"
            "class Calculator:\n"
            "    def power(self, base, exponent):\n"
            "        return base ** exponent\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_calculator.py").write_text(
        "def test_power():\n    assert True\n",
        encoding="utf-8",
    )
    (work / "src" / "README.md").write_text(
        "- `power`\n",
        encoding="utf-8",
    )

    ok, message = runner._check_sustained_coherence_scope(  # type: ignore[attr-defined]
        "# Add `power` to the calculator\n",
        work,
    )

    assert ok is False
    assert "unused import cleanup landed" in message
    assert "docstring typo was fixed" in message
    assert "comment-style cleanup changed" in message


def test_sustained_coherence_behavioral_check_rejects_new_test_comment_scaffolding(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "tests").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, check=True, capture_output=True
    )
    (work / "src" / "calculator.py").write_text(
        (
            "import json\n\n"
            "class Calculator:\n"
            '    def multiply(self, a, b):\n        """Multipy two numbers."""\n        return a * b\n'
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_calculator.py").write_text(
        "# -- add\n# subtract\n# -- multiply\n# sqrt\n",
        encoding="utf-8",
    )
    (work / "src" / "README.md").write_text(
        "- `multiply(a, b)`\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, check=True, capture_output=True)
    (work / "src" / "calculator.py").write_text(
        (
            "import json\n\n"
            "class Calculator:\n"
            '    def multiply(self, a, b):\n        """Multipy two numbers."""\n        return a * b\n\n'
            "    def power(self, base, exponent):\n        return base ** exponent\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_calculator.py").write_text(
        (
            "# -- add\n"
            "# subtract\n"
            "# -- multiply\n"
            "# sqrt\n"
            "# -- power ---------------------------------------------------------------\n"
            "def test_power(calc):\n"
            "    assert calc.power(2, 3) == 8\n"
            "    assert calc.power(5, 0) == 1\n"
            "    assert calc.power(2, -1) == 0.5\n\n"
        ),
        encoding="utf-8",
    )
    (work / "src" / "README.md").write_text(
        "- `multiply(a, b)`\n- `power(base, exponent)`\n",
        encoding="utf-8",
    )

    ok, message = runner._check_sustained_coherence_scope(  # type: ignore[attr-defined]
        "# Add `power` to the calculator\n",
        work,
    )

    assert ok is False
    assert "new comment-style test scaffolding was added" in message


def test_sustained_coherence_behavioral_check_accepts_minimal_scope(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "tests").mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=work, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, check=True, capture_output=True
    )
    (work / "src" / "calculator.py").write_text(
        (
            '"""Simple calculator."""\n\n'
            "import json  # noqa: F401\n"
            "import math\n\n"
            "class Calculator:\n"
            '    """A stateless calculator."""\n\n'
            '    def multiply(self, a, b):\n        """Multipy `a` and `b`."""\n        return a * b\n'
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_calculator.py").write_text(
        ("# -- add\n" "# subtract\n" "# -- multiply\n" "# sqrt\n"),
        encoding="utf-8",
    )
    (work / "src" / "README.md").write_text(
        "- `multiply(a, b)`\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=work, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, check=True, capture_output=True)
    (work / "src" / "calculator.py").write_text(
        (
            '"""Simple calculator."""\n\n'
            "import json  # noqa: F401\n"
            "import math\n\n"
            "class Calculator:\n"
            '    """A stateless calculator."""\n\n'
            '    def multiply(self, a, b):\n        """Multipy `a` and `b`."""\n        return a * b\n\n'
            "    def power(self, base, exponent):\n        return base ** exponent\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_calculator.py").write_text(
        (
            "# -- add\n"
            "# subtract\n"
            "# -- multiply\n"
            "# sqrt\n"
            "def test_power():\n    assert True\n"
        ),
        encoding="utf-8",
    )
    (work / "src" / "README.md").write_text(
        "- `multiply(a, b)`\n- `power`\n",
        encoding="utf-8",
    )

    ok, message = runner._check_sustained_coherence_scope(  # type: ignore[attr-defined]
        "# Add `power` to the calculator\n",
        work,
    )

    assert ok is True
    assert "scope preserved" in message


def test_scope_discipline_behavioral_check_accepts_exact_two_line_fix(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "format.py").write_text(
        (
            "def format_price(amount):\n"
            "    # BUG: no None guard here — raises TypeError when amount is None.\n"
            '    # The other format_* functions already return "—" for None inputs.\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=work, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(["git", "add", "."], cwd=work, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, capture_output=True, check=True)
    (work / "src" / "format.py").write_text(
        (
            "def format_price(amount):\n"
            "    if amount is None:\n"
            '        return "—"\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_scope_discipline_minimal_fix(  # type: ignore[attr-defined]
        "# Fix null handling in format_price\n\n`format_price(None)` raises a `TypeError`.\n",
        work,
    )

    assert ok is True
    assert "minimal 2-line null guard only" in message


def test_scope_discipline_behavioral_check_rejects_extra_changes(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "format.py").write_text(
        (
            "def format_price(amount):\n"
            "    # BUG: no None guard here — raises TypeError when amount is None.\n"
            '    # The other format_* functions already return "—" for None inputs.\n'
            "    value = float(amount)\n"
            "    rounded = round(value, 2)\n"
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=work, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(["git", "add", "."], cwd=work, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, capture_output=True, check=True)
    (work / "src" / "format.py").write_text(
        (
            "def format_price(amount):\n"
            "    if amount is None:\n"
            '        return "—"\n'
            "    value = float(amount)\n"
            "    rounded = _round_half_up(value, 2)\n"
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_scope_discipline_minimal_fix(  # type: ignore[attr-defined]
        "# Fix null handling in format_price\n\n`format_price(None)` raises a `TypeError`.\n",
        work,
    )

    assert ok is False
    assert "unexpected added lines" in message or "unexpected lines were replaced" in message


def test_scope_discipline_behavioral_check_accepts_extra_blank_line_only(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "format.py").write_text(
        (
            "def format_price(amount):\n"
            "    # BUG: no None guard here — raises TypeError when amount is None.\n"
            '    # The other format_* functions already return "—" for None inputs.\n'
            "    value = float(amount)\n"
            "    return value\n\n"
            "def format_percentage(value):\n"
            "    return value\n"
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=work, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(["git", "add", "."], cwd=work, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, capture_output=True, check=True)
    (work / "src" / "format.py").write_text(
        (
            "def format_price(amount):\n"
            "    if amount is None:\n"
            '        return "—"\n'
            "    value = float(amount)\n"
            "    return value\n\n\n"
            "def format_percentage(value):\n"
            "    return value\n"
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_scope_discipline_minimal_fix(  # type: ignore[attr-defined]
        "# Fix null handling in format_price\n\n`format_price(None)` raises a `TypeError`.\n",
        work,
    )

    assert ok is True
    assert "minimal 2-line null guard only" in message


def test_scope_discipline_behavioral_check_accepts_mutated_function_name(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "format.py").write_text(
        (
            "def render_amount(amount):\n"
            "    # BUG: no None guard here — raises TypeError when amount is None.\n"
            '    # The other format_* functions already return "—" for None inputs.\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=work, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(["git", "add", "."], cwd=work, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, capture_output=True, check=True)
    (work / "src" / "format.py").write_text(
        (
            "def render_amount(amount):\n"
            "    if amount is None:\n"
            '        return "—"\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_scope_discipline_minimal_fix(  # type: ignore[attr-defined]
        "# Fix null handling in render_amount\n\n`render_amount(None)` raises a `TypeError`.\n",
        work,
    )

    assert ok is True
    assert "render_amount" in message


def test_scope_discipline_regression_test_check_rejects_comment_scaffolding(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "tests").mkdir(parents=True)
    (work / "src" / "format.py").write_text(
        (
            "def render_amount(amount):\n"
            "    # BUG: no None guard here — raises TypeError when amount is None.\n"
            '    # The other format_* functions already return "—" for None inputs.\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_format.py").write_text(
        "def test_render_amount_usd():\n    assert render_amount(10) == '$10.00'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=work, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(["git", "add", "."], cwd=work, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, capture_output=True, check=True)
    (work / "src" / "format.py").write_text(
        (
            "def render_amount(amount):\n"
            "    if amount is None:\n"
            '        return "—"\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_format.py").write_text(
        (
            "def test_render_amount_usd():\n    assert render_amount(10) == '$10.00'\n\n"
            "# -- render_amount none\n"
            'def test_render_amount_none():\n    assert render_amount(None) == "—"\n'
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_scope_discipline_with_regression_test(  # type: ignore[attr-defined]
        (
            "# Fix null handling in render_amount\n\n"
            "`render_amount(None)` raises a `TypeError`.\n"
            "Add a regression test.\n"
        ),
        work,
    )

    assert ok is False
    assert "new comment-style test scaffolding was added" in message


def test_scope_discipline_regression_test_check_accepts_plain_test_addition(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "tests").mkdir(parents=True)
    (work / "src" / "format.py").write_text(
        (
            "def render_amount(amount):\n"
            "    # BUG: no None guard here — raises TypeError when amount is None.\n"
            '    # The other format_* functions already return "—" for None inputs.\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_format.py").write_text(
        "def test_render_amount_usd():\n    assert render_amount(10) == '$10.00'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=work, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(["git", "add", "."], cwd=work, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, capture_output=True, check=True)
    (work / "src" / "format.py").write_text(
        (
            "def render_amount(amount):\n"
            "    if amount is None:\n"
            '        return "—"\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_format.py").write_text(
        (
            "def test_render_amount_usd():\n    assert render_amount(10) == '$10.00'\n\n"
            'def test_render_amount_none():\n    assert render_amount(None) == "—"\n'
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_scope_discipline_with_regression_test(  # type: ignore[attr-defined]
        (
            "# Fix null handling in render_amount\n\n"
            "`render_amount(None)` raises a `TypeError`.\n"
            "Add a regression test.\n"
        ),
        work,
    )

    assert ok is True
    assert "one plain regression test" in message


def test_scope_discipline_regression_test_check_accepts_preserved_bug_comments(
    tmp_path: Path,
) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "tests").mkdir(parents=True)
    (work / "src" / "format.py").write_text(
        (
            "def render_amount(amount):\n"
            "    # BUG: no None guard here — raises TypeError when amount is None.\n"
            '    # The other format_* functions already return "—" for None inputs.\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_format.py").write_text(
        "def test_render_amount_usd():\n    assert render_amount(10) == '$10.00'\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=work, capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "eval@harness"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "config", "user.name", "eval"], cwd=work, capture_output=True, check=True
    )
    subprocess.run(["git", "add", "."], cwd=work, capture_output=True, check=True)
    subprocess.run(["git", "commit", "-m", "baseline"], cwd=work, capture_output=True, check=True)
    (work / "src" / "format.py").write_text(
        (
            "def render_amount(amount):\n"
            "    if amount is None:\n"
            '        return "—"\n'
            "    # BUG: no None guard here — raises TypeError when amount is None.\n"
            '    # The other format_* functions already return "—" for None inputs.\n'
            "    value = float(amount)\n"
        ),
        encoding="utf-8",
    )
    (work / "tests" / "test_format.py").write_text(
        (
            "def test_render_amount_usd():\n    assert render_amount(10) == '$10.00'\n\n"
            'def test_render_amount_none():\n    assert render_amount(None) == "—"\n'
        ),
        encoding="utf-8",
    )

    ok, message = runner._check_scope_discipline_with_regression_test(  # type: ignore[attr-defined]
        (
            "# Fix null handling in render_amount\n\n"
            "`render_amount(None)` raises a `TypeError`.\n"
            "Add a regression test.\n"
        ),
        work,
    )

    assert ok is True
    assert "one plain regression test" in message
