"""Tests for eval execution and artifact persistence."""

from __future__ import annotations

import json
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
            "family: scope-discipline\nbehavior_category: scope\nverify_command: pytest tests/\n"
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


def test_eval_env_exposes_project_root_and_workspace(tmp_path: Path) -> None:
    env = runner._eval_env(work=tmp_path)  # type: ignore[attr-defined]

    assert env["HARNESS_EVAL_WORKSPACE"] == str(tmp_path)
    assert Path(env["HARNESS_EVAL_PROJECT_ROOT"]).resolve() == Path(__file__).resolve().parents[2]
