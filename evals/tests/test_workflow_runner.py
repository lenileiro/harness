from __future__ import annotations

import json
import os
from pathlib import Path

from evals import workflow_runner


def _write_fixture(root: Path, name: str) -> None:
    fixture = root / "evals" / "workflow-fixtures" / name
    fixture.mkdir(parents=True)
    (fixture / "fixture.json").write_text(
        json.dumps(
            {
                "description": "workflow smoke",
                "steps": [
                    {
                        "name": "vision-update",
                        "argv": [
                            "vision",
                            "update",
                            "--title",
                            "Autonomous research harness",
                            "--summary",
                            "Turn Harness into a compounding research system.",
                            "--theme",
                            "autonomous-improvement",
                            "--success-metric",
                            "high-signal autonomous PRs",
                        ],
                        "stdout_contains": ["Updated vision"],
                        "files_exist": [".harness/research/vision/current/vision.json"],
                    },
                    {
                        "name": "add-theme",
                        "argv": [
                            "research",
                            "add-theme",
                            "--title",
                            "Verifier reliability",
                            "--description",
                            "Study routing opportunities.",
                        ],
                        "stdout_contains": ["Added theme"],
                        "files_exist": [".harness/research/themes/{latest_theme_id}/theme.json"],
                    },
                    {
                        "name": "open",
                        "argv": [
                            "research",
                            "open",
                            "--title",
                            "Verifier routing deep dive",
                            "--question",
                            "Can routing reduce verifier noise?",
                            "--scope",
                            "Inspect routing and eval impact.",
                            "--theme",
                            "verification",
                        ],
                        "stdout_contains": ["Opened rabbit hole"],
                        "files_exist": [
                            ".harness/research/rabbitholes/{latest_rabbit_hole_id}/rabbit_hole.json"
                        ],
                    },
                    {
                        "name": "publish",
                        "argv": [
                            "research",
                            "publish",
                            "--rabbit-hole",
                            "{latest_rabbit_hole_id}",
                            "--title",
                            "Verifier routing findings",
                            "--summary",
                            "Scoped routing helps.",
                            "--claim",
                            "Scoped routing reduces verifier noise.",
                        ],
                        "stdout_contains": ["Published"],
                        "files_exist": [
                            ".harness/research/publications/{latest_publication_id}/publication.json"
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_fixture_with_wrapper(root: Path, name: str) -> None:
    fixture = root / "evals" / "workflow-fixtures" / name
    workspace = fixture / "workspace"
    workspace.mkdir(parents=True)
    (fixture / "fixture.json").write_text(
        json.dumps(
            {
                "description": "workflow with custom harness wrapper",
                "harness_bin": "fake-harness",
                "steps": [
                    {
                        "name": "vision-update",
                        "argv": [
                            "vision",
                            "update",
                            "--title",
                            "Wrapped harness",
                            "--summary",
                            "Use a fixture-local harness binary.",
                        ],
                        "stdout_contains": ["Updated vision"],
                        "files_exist": [".harness/research/vision/current/vision.json"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    wrapper = workspace / "fake-harness"
    wrapper.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from harness.cli.__main__ import app",
                "if __name__ == '__main__':",
                "    app()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)


def _write_fixture_with_stdin(root: Path, name: str) -> None:
    fixture = root / "evals" / "workflow-fixtures" / name
    fixture.mkdir(parents=True)
    script = root / "echo-stdin"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "data = sys.stdin.read()",
                "print(data, end='')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    (fixture / "fixture.json").write_text(
        json.dumps(
            {
                "description": "workflow with stdin",
                "harness_bin": str(script),
                "steps": [
                    {
                        "name": "echo",
                        "argv": [],
                        "append_cwd": False,
                        "stdin_text": "hello from stdin\n",
                        "stdout_contains": ["hello from stdin"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_discover_workflow_fixtures_reads_expected_layout(tmp_path: Path) -> None:
    _write_fixture(tmp_path, "01-demo")
    fixtures = workflow_runner.discover_workflow_fixtures(tmp_path / "evals")
    assert [fixture.name for fixture in fixtures] == ["01-demo"]
    assert fixtures[0].steps[0].name == "vision-update"


def test_run_workflow_fixture_executes_cli_commands_and_persists_artifacts(tmp_path: Path) -> None:
    _write_fixture(tmp_path, "01-demo")
    fixture = workflow_runner.discover_workflow_fixtures(tmp_path / "evals")[0]
    harness_bin = tmp_path / "fake-harness"
    harness_bin.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "from harness.cli.__main__ import app",
                "if __name__ == '__main__':",
                "    app()",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    harness_bin.chmod(0o755)
    artifact_dir = tmp_path / "artifacts"

    result = workflow_runner.run_workflow_fixture(
        fixture,
        harness_bin=str(harness_bin),
        artifact_dir=artifact_dir,
        timeout=60,
    )

    assert result.passed is True
    assert result.steps_passed == result.steps_total == 4
    assert (artifact_dir / "result.json").exists()
    assert any("vision" in " ".join(step.argv) for step in result.step_results)
    first_step = json.loads((artifact_dir / "01-vision-update.json").read_text(encoding="utf-8"))
    assert first_step["passed"] is True


def test_context_for_workdir_exposes_previous_publication_id(tmp_path: Path) -> None:
    research_root = tmp_path / ".harness" / "research" / "publications"
    first = research_root / "pub-older"
    second = research_root / "pub-newer"
    first.mkdir(parents=True)
    second.mkdir(parents=True)

    older_time = 1_700_000_000
    newer_time = 1_700_000_100
    os.utime(first, (older_time, older_time))
    os.utime(second, (newer_time, newer_time))

    context = workflow_runner._context_for_workdir(tmp_path)
    assert context["latest_publication_id"] == "pub-newer"
    assert context["previous_publication_id"] == "pub-older"


def test_context_for_workdir_exposes_latest_mission_ids(tmp_path: Path) -> None:
    mission_root = tmp_path / ".harness" / "missions"
    mission_dir = mission_root / "missions" / "mission-demo"
    feature_dir = mission_root / "features" / "feature-demo"
    run_dir = mission_root / "runs" / "run-demo"
    for path in (mission_dir, feature_dir, run_dir):
        path.mkdir(parents=True)

    context = workflow_runner._context_for_workdir(tmp_path)
    assert context["latest_mission_id"] == "mission-demo"
    assert context["latest_mission_feature_id"] == "feature-demo"
    assert context["latest_mission_run_id"] == "run-demo"


def test_context_for_workdir_exposes_previous_mission_feature_id(tmp_path: Path) -> None:
    mission_root = tmp_path / ".harness" / "missions" / "features"
    older = mission_root / "feature-older"
    newer = mission_root / "feature-newer"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)

    older_time = 1_700_000_000
    newer_time = 1_700_000_100
    os.utime(older, (older_time, older_time))
    os.utime(newer, (newer_time, newer_time))

    context = workflow_runner._context_for_workdir(tmp_path)
    assert context["latest_mission_feature_id"] == "feature-newer"
    assert context["previous_mission_feature_id"] == "feature-older"


def test_run_workflow_fixture_uses_fixture_local_harness_bin(tmp_path: Path) -> None:
    _write_fixture_with_wrapper(tmp_path, "02-wrapper")
    fixture = workflow_runner.discover_workflow_fixtures(tmp_path / "evals")[0]

    result = workflow_runner.run_workflow_fixture(
        fixture,
        artifact_dir=tmp_path / "artifacts",
        timeout=60,
    )

    assert result.passed is True
    assert result.steps_passed == result.steps_total == 1


def test_run_workflow_fixture_supports_stdin_text(tmp_path: Path) -> None:
    _write_fixture_with_stdin(tmp_path, "03-stdin")
    fixture = workflow_runner.discover_workflow_fixtures(tmp_path / "evals")[0]

    result = workflow_runner.run_workflow_fixture(
        fixture,
        artifact_dir=tmp_path / "artifacts",
        timeout=60,
    )

    assert result.passed is True
    assert result.steps_passed == result.steps_total == 1
