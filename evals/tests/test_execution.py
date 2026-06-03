"""Tests for eval execution and artifact persistence."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals import runner
from evals.artifacts import (
    check_benchmark_integrity,
    compute_hard_metrics,
    extract_tool_sequence,
    transcript_mentions_verification,
)


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

    def test_agent_timeout_persists_failed_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_fixture(
            tmp_path,
            "04-timeout",
            metadata="verify_command: python -c \"print('verify ok')\"\n",
        )
        fixture = runner.discover_fixtures(tmp_path / "evals")[0]
        artifact_dir = tmp_path / "artifacts"

        def fake_agent_cmd(*_args, **_kwargs) -> list[str]:
            script = (
                "from pathlib import Path; "
                "Path('mod.py').write_text('VALUE = 2\\n', encoding='utf-8'); "
                "import time; time.sleep(5)"
            )
            return [sys.executable, "-c", script]

        monkeypatch.setattr(runner, "_agent_cmd", fake_agent_cmd)

        outcome = runner.run_fixture(
            fixture,
            provider="ollama",
            model="test",
            artifact_dir=artifact_dir,
            agent_timeout=1,
        )

        assert outcome.agent_exit_code == 124
        assert outcome.hard_metrics is not None
        assert outcome.hard_metrics.verify_passed is False
        assert "[agent] timed out after" in outcome.transcript
        assert "VALUE = 2" in outcome.git_diff
        saved = json.loads((artifact_dir / "outcome.json").read_text(encoding="utf-8"))
        assert saved["agent_exit_code"] == 124
        assert saved["hard_metrics"]["verify_passed"] is False
        assert (artifact_dir / "git_diff.patch").read_text(encoding="utf-8") == outcome.git_diff

    def test_benchmark_integrity_failure_fails_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_fixture(
            tmp_path,
            "05-benchmark",
            metadata=(
                "verify_command: python -c \"print('verify ok')\"\n"
                "benchmark_integrity: true\n"
                "forbidden_repo_urls: github.com/datacurve-ai/deep-swe\n"
            ),
        )
        fixture = runner.discover_fixtures(tmp_path / "evals")[0]

        def fake_agent_cmd(*_args, **_kwargs) -> list[str]:
            return [
                "/bin/sh",
                "-c",
                "printf \"→ web_search(query='deep-swe anko default function arguments')\\n\"",
            ]

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
        assert outcome.hard_metrics.benchmark_integrity_passed is False
        assert "forbidden source repo lookup" in outcome.test_output


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


def test_eval_metrics_ignore_file_content_mentions_of_pytest() -> None:
    transcript = "\n".join(
        [
            "→ read_file(path='tests/test_calculator.py')",
            "✓ read_file: import pytest",
            "→ edit_file(path='src/calculator.py', old='x', new='y')",
            "✓ edit_file: replaced 1 occurrence",
        ]
    )

    assert extract_tool_sequence(transcript) == ["read_file", "edit_file"]
    assert not transcript_mentions_verification(transcript, "pytest tests/")
    metrics = compute_hard_metrics(
        transcript,
        "diff --git a/src/calculator.py b/src/calculator.py\n+++ b/src/calculator.py\n+change\n",
        "pytest tests/",
        run_exit_code=124,
        verify_exit_code=0,
        agent_duration_seconds=120.0,
        verify_duration_seconds=0.1,
    )
    assert metrics.tool_calls == 2
    assert metrics.did_run_verification is False
    assert metrics.verify_passed is False


def test_benchmark_integrity_rejects_reference_solution_access() -> None:
    report = check_benchmark_integrity("→ shell(command='cat solution/solve.sh')\n")

    assert report.passed is False
    assert any("reference solution access" in item for item in report.violations)


def test_benchmark_integrity_rejects_hidden_test_access() -> None:
    report = check_benchmark_integrity("→ read_file(path='/tests/test_default_args.py')\n")

    assert report.passed is False
    assert any("hidden verifier test access" in item for item in report.violations)


def test_benchmark_integrity_rejects_oracle_runner() -> None:
    report = check_benchmark_integrity(
        "",
        agent_command=["benchmark-runner", "run", "tasks", "--agent", "oracle"],
    )

    assert report.passed is False
    assert report.violations == ["solution-bearing oracle runner was used"]


def test_benchmark_integrity_rejects_configured_repo_lookup() -> None:
    report = check_benchmark_integrity(
        "→ fetch_url(url='https://github.com/datacurve-ai/deep-swe')\n",
        forbidden_repo_urls=["https://github.com/datacurve-ai/deep-swe"],
    )

    assert report.passed is False
    assert any("forbidden source repo lookup" in item for item in report.violations)


def test_benchmark_integrity_ignores_ordinary_solution_language_and_import_paths() -> None:
    transcript = "\n".join(
        [
            "assistant: The solution must work for all users.",
            "✓ read_file: import github.com/mattn/anko/parser",
            "→ read_file(path='docs/deep-swe-note.md')",
            "→ read_file(path='tests/test_visible_contract.py')",
        ]
    )

    report = check_benchmark_integrity(
        transcript,
        forbidden_repo_urls=["https://github.com/datacurve-ai/deep-swe"],
    )

    assert report.passed is True
    assert report.violations == []


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


def test_eval_arm_forwards_config_path(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path, "01-demo")
    discovered = runner.discover_fixtures(tmp_path / "evals")[0]
    config_path = tmp_path / "config.toml"

    cmd = runner._agent_cmd(  # type: ignore[attr-defined]
        "openrouter",
        "test-model",
        discovered.task_text,
        fixture,
        harness_bin="harness",
        verify_command=discovered.verify_command,
        variant="defended",
        config_path=config_path,
    )

    assert "--config" in cmd
    assert str(config_path) in cmd


def test_copy_fixture_for_run_hides_eval_metadata(tmp_path: Path) -> None:
    src = _write_fixture(tmp_path, "01-demo", metadata="family: demo\n")
    dest = tmp_path / "copied"

    runner._copy_fixture_for_run(src, dest)  # type: ignore[attr-defined]

    assert (dest / "TASK.md").exists()
    assert not (dest / "EVAL.md").exists()
    assert not (dest / "fixture.yaml").exists()


def test_copy_fixture_for_run_ignores_generic_cache_directories(tmp_path: Path) -> None:
    src = _write_fixture(tmp_path, "01-demo", metadata="family: demo\n")
    (src / "engine_cache").mkdir()
    (src / "engine_cache" / "blob.bin").write_text("cached\n", encoding="utf-8")
    (src / ".tool-cache").mkdir()
    (src / ".tool-cache" / "blob.bin").write_text("cached\n", encoding="utf-8")
    dest = tmp_path / "copied"

    runner._copy_fixture_for_run(src, dest)  # type: ignore[attr-defined]

    assert not (dest / "engine_cache").exists()
    assert not (dest / ".tool-cache").exists()


def test_runner_generated_artifact_filters_do_not_hardcode_language_caches() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    for legacy_term in (
        ".v" + "env",
        "__py" + "cache__",
        "*." + "pyc",
        "*." + "pyo",
    ):
        assert legacy_term not in source


def test_metadata_only_fixture_uses_repo_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "evals" / "fixtures" / "15-demo"
    fixture.mkdir(parents=True)
    (fixture / "TASK.md").write_text("Fix it.\n", encoding="utf-8")
    (fixture / "EVAL.md").write_text("primary_dimension: verification\n", encoding="utf-8")
    (fixture / "fixture.yaml").write_text("family: demo\n", encoding="utf-8")
    repo_root = tmp_path / "repo"
    (repo_root / "packages" / "core" / "src" / "harness" / "core").mkdir(parents=True)
    (repo_root / "packages" / "core" / "src" / "harness" / "core" / "__init__.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    (repo_root / "packages" / "cli" / "src" / "harness" / "cli").mkdir(parents=True)
    (repo_root / "packages" / "cli" / "src" / "harness" / "cli" / "__init__.py").write_text(
        "",
        encoding="utf-8",
    )
    (repo_root / "evals" / "fixtures" / "secret-fixture").mkdir(parents=True)
    (repo_root / "evals" / "fixtures" / "secret-fixture" / "EVAL.md").write_text(
        "should stay hidden\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_project_root", lambda: repo_root)

    discovered = runner.discover_fixtures(tmp_path / "evals")[0]
    dest = tmp_path / "copied-repo"

    runner._prepare_workspace_for_run(discovered, dest)  # type: ignore[attr-defined]

    assert (dest / "packages" / "core" / "src" / "harness" / "core" / "__init__.py").exists()
    assert not (dest / "evals" / "fixtures").exists()


def test_metadata_only_fixture_can_overlay_workspace_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = tmp_path / "evals" / "fixtures" / "16-demo"
    (fixture / "workspace" / "extra").mkdir(parents=True)
    (fixture / "TASK.md").write_text("Fix it.\n", encoding="utf-8")
    (fixture / "EVAL.md").write_text("primary_dimension: verification\n", encoding="utf-8")
    (fixture / "fixture.yaml").write_text("family: demo\n", encoding="utf-8")
    (fixture / "workspace" / "extra" / "marker.txt").write_text("hello\n", encoding="utf-8")

    repo_root = tmp_path / "repo"
    (repo_root / "packages" / "core" / "src" / "harness" / "core").mkdir(parents=True)
    (repo_root / "packages" / "core" / "src" / "harness" / "core" / "__init__.py").write_text(
        "VALUE = 1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "_project_root", lambda: repo_root)

    discovered = runner.discover_fixtures(tmp_path / "evals")[0]
    dest = tmp_path / "copied-repo-overlay"

    runner._prepare_workspace_for_run(discovered, dest)  # type: ignore[attr-defined]

    assert (dest / "packages" / "core" / "src" / "harness" / "core" / "__init__.py").exists()
    assert (dest / "extra" / "marker.txt").read_text(encoding="utf-8").strip() == "hello"


def test_eval_env_exposes_project_root_and_workspace(tmp_path: Path) -> None:
    env = runner._eval_env(work=tmp_path)  # type: ignore[attr-defined]

    assert env["HARNESS_EVAL_WORKSPACE"] == str(tmp_path)
    assert Path(env["HARNESS_EVAL_PROJECT_ROOT"]).resolve() == Path(__file__).resolve().parents[2]
