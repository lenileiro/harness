from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from evals.deepswe_runner import (
    CommandResult,
    DeepSWETask,
    HostWorkspaceEnvironment,
    _capture_target_patch,
    _enforce_hidden_verifier_reward,
    _enforce_public_offline_verification,
    _failure_metadata,
    _git_diff_with_untracked_command,
    _hidden_verifier_command,
    _hidden_verifier_reward,
    _offline_verification_command,
    _policy_for_task,
    _prepare_agent_workspace,
    _public_offline_verification_command,
    _repository_forbidden_fragments,
    _run_diagnostics,
    _run_public_offline_verification,
    _run_timeout_seconds,
    _should_run_hidden_verifier,
    _should_run_public_offline_verification,
    build_parser,
    load_task,
    normalize_paths,
)


def _write_task(root: Path) -> Path:
    task_dir = root / "deep-swe" / "tasks" / "sample-task"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "solution").mkdir()
    (task_dir / "instruction.md").write_text("Fix observable behavior.\n", encoding="utf-8")
    (task_dir / "task.toml").write_text(
        """
version = "1.0"

[metadata]
task_id = "sample-task"
display_title = "Sample hidden benchmark title"
display_description = "Sample hidden benchmark description"
original_title = "Sample original issue title"
repository_url = "https://example.test/repo"
base_commit_hash = "abc123"

[verifier]
timeout_sec = 42.0

[agent]
timeout_sec = 123.0

[environment]
docker_image = "example/image:latest"
allow_internet = false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return task_dir


def test_load_task_reads_public_metadata_without_solution_or_tests(tmp_path: Path) -> None:
    task = load_task(_write_task(tmp_path))

    assert task.task_id == "sample-task"
    assert task.instruction == "Fix observable behavior."
    assert task.image == "example/image:latest"
    assert task.repository_url == "https://example.test/repo"
    assert task.base_commit == "abc123"
    assert task.allow_internet is False
    assert task.verifier_timeout_sec == 42
    assert task.agent_timeout_sec == 123
    assert task.display_title == "Sample hidden benchmark title"
    assert task.display_description == "Sample hidden benchmark description"
    assert task.original_title == "Sample original issue title"


def test_policy_blocks_deepswe_private_artifacts_but_not_repo_tests(tmp_path: Path) -> None:
    task_dir = _write_task(tmp_path)
    task = DeepSWETask(
        task_id="sample-task",
        task_dir=task_dir,
        instruction="Fix observable behavior.",
        image="example/image:latest",
        repository_url="https://github.com/example/project",
        base_commit="abc123",
        allow_internet=False,
        verifier_timeout_sec=42,
        agent_timeout_sec=123,
        display_title="Add flattened dataclass fields to Mashumaro field options",
        display_description="Add field_options support for flattening nested dataclass fields",
        original_title="Conditional Field Visibility",
    )
    policy = _policy_for_task(task, tmp_path / "results" / "sample-task-run")

    assert policy.references_forbidden_material(str(task_dir / "tests" / "test.patch"))
    assert policy.references_forbidden_material(str(task_dir / "solution" / "solution.patch"))
    assert policy.references_forbidden_material("curl https://github.com/datacurve-ai/deep-swe")
    assert policy.allow_web_access is True
    assert policy.required_git_base_commit == "abc123"
    assert policy.references_allowed_git_clone_material(
        "git clone https://github.com/example/project repo"
    )
    assert "restricted artifacts" in policy.refusal_message
    assert not policy.references_forbidden_web_material("search public docs")
    assert policy.references_forbidden_web_material("search github.com/example/project issues")
    assert policy.references_forbidden_web_material("example/project bug report")
    assert policy.references_forbidden_web_material(
        "https://deepswe.datacurve.ai/data/tasks/sample-task"
    )
    assert policy.references_forbidden_web_material("search deepswe.datacurve.ai")
    assert policy.references_forbidden_web_material("search sample-task solution")
    assert policy.references_forbidden_web_material(
        "Add flattened dataclass fields to Mashumaro field options"
    )
    assert policy.references_forbidden_web_material(
        "Add field_options support for flattening nested dataclass fields"
    )
    assert policy.references_forbidden_web_material("Conditional Field Visibility")
    assert policy.references_forbidden_web_material(
        "Add partial structuring with error recovery to cattrs - DeepSWE"
    )
    assert policy.references_forbidden_web_material("DataCurve benchmark task page")
    assert policy.required_no_network_verify_image == task.image
    assert not policy.references_forbidden_material("import github.com/example/project/pkg")
    assert policy.rejects_relative_path("solution/solution.patch")
    assert policy.rejects_relative_path("test.patch")
    assert not policy.rejects_relative_path("tests/test_public_behavior.go")
    assert not policy.rejects_relative_path("pkg/module/cache.go")
    assert not policy.references_forbidden_material("ls repo/tests")
    assert not policy.references_forbidden_material("pytest repo/tests/test_public.py")


def test_deepswe_policy_web_access_is_public_research_allowed_by_default(
    tmp_path: Path,
) -> None:
    task = load_task(_write_task(tmp_path))

    policy = _policy_for_task(task, tmp_path / "results" / "sample-task-run")
    disabled_policy = _policy_for_task(
        task,
        tmp_path / "results" / "sample-task-run-disabled",
        allow_web_access=False,
    )

    assert policy.allow_web_access is True
    assert not policy.references_forbidden_web_material("search Docker install docs")
    assert disabled_policy.allow_web_access is False


def test_repository_forbidden_fragments_cover_url_and_owner_repo() -> None:
    assert _repository_forbidden_fragments("https://github.com/python-attrs/cattrs") == (
        "https://github.com/python-attrs/cattrs",
        "github.com/python-attrs/cattrs",
        "python-attrs/cattrs",
    )


def test_cli_defaults_use_harness_external_runner_contract() -> None:
    args = build_parser().parse_args(["/tmp/deep-swe/tasks/sample"])

    assert isinstance(args, argparse.Namespace)
    assert args.model == "openai/gpt-5.4-nano"
    assert args.results_root == "evals/results/deepswe"
    assert args.max_steps == 50
    assert args.run_timeout_seconds == 0.0
    assert args.verifier_timeout_seconds == 300.0
    assert args.idle_timeout == 90.0
    assert args.turn_timeout == 120.0
    assert args.allow_web_access is True


def test_cli_can_opt_out_of_deepswe_web_access() -> None:
    args = build_parser().parse_args(["/tmp/deep-swe/tasks/sample", "--no-web-access"])

    assert args.allow_web_access is False


def test_hidden_verifier_is_post_run_grader_not_harness_sentinel() -> None:
    command = _hidden_verifier_command()

    assert "/tests/test.sh" in command
    assert "/model.patch" in command
    assert "reward.txt" in command


def test_hidden_verifier_runs_only_after_harness_accepted_completion() -> None:
    assert _should_run_hidden_verifier(status="passed", model_patch_captured=True)
    assert not _should_run_hidden_verifier(status="failed", model_patch_captured=True)
    assert not _should_run_hidden_verifier(status="passed", model_patch_captured=False)


def test_agent_workspace_contains_only_public_task_files(tmp_path: Path) -> None:
    task = load_task(_write_task(tmp_path))
    (task.task_dir / "environment").mkdir()
    (task.task_dir / "environment" / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")
    (task.task_dir / "tests" / "test.patch").write_text("hidden\n", encoding="utf-8")
    (task.task_dir / "solution" / "solution.patch").write_text("secret\n", encoding="utf-8")

    workspace = _prepare_agent_workspace(task, tmp_path / "run")

    assert (workspace / "task.toml").exists()
    assert (workspace / "instruction.md").exists()
    assert (workspace / "environment" / "Dockerfile").exists()
    assert not (workspace / "tests").exists()
    assert not (workspace / "solution").exists()
    assert (workspace / ".git").exists()
    assert ".harness-home/" in (workspace / ".git" / "info" / "exclude").read_text(encoding="utf-8")


async def test_host_workspace_environment_does_not_leak_parent_virtualenv(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness_venv = tmp_path / "harness-venv"
    harness_bin = harness_venv / "bin"
    harness_bin.mkdir(parents=True)
    monkeypatch.setenv("VIRTUAL_ENV", str(harness_venv))
    monkeypatch.setenv("PYTHONPATH", "/leaked/pythonpath")
    monkeypatch.setenv("PATH", f"{harness_bin}:/usr/bin:/bin")

    env = HostWorkspaceEnvironment(tmp_path)
    result = await env.exec(
        "printf 'HOME=%s\\nVIRTUAL_ENV=%s\\nPYTHONPATH=%s\\nPATH=%s\\nPYTHONUSERBASE=%s\\n' "
        '"${HOME-}" "${VIRTUAL_ENV-}" "${PYTHONPATH-}" "$PATH" '
        '"${PYTHONUSERBASE-}"'
    )

    assert result.return_code == 0
    assert f"HOME={tmp_path / '.harness-home'}" in result.stdout
    assert f"VIRTUAL_ENV={harness_venv}" not in result.stdout
    assert "PYTHONPATH=/leaked/pythonpath" not in result.stdout
    assert f"PYTHONUSERBASE={tmp_path / '.harness-home' / '.python-userbase'}" in result.stdout
    assert str(harness_bin) not in result.stdout
    assert "/usr/bin" in result.stdout


def test_deepswe_runner_does_not_wire_hidden_verifier_into_harness() -> None:
    import evals.deepswe_runner as runner

    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "run_harness_on_external_environment"
    ]

    assert calls, "runner should call Harness external workspace runtime"
    for call in calls:
        keyword = next(
            (item for item in call.keywords if item.arg == "default_verify_command"),
            None,
        )
        assert keyword is not None
        assert isinstance(keyword.value, ast.Constant)
        assert keyword.value.value is None
        regression_keyword = next(
            (item for item in call.keywords if item.arg == "require_regression_test_change"),
            None,
        )
        assert regression_keyword is not None
        assert isinstance(regression_keyword.value, ast.Constant)
        assert regression_keyword.value.value is True
    assert "__HARNESS_DEEPSWE_VERIFY__" not in source
    assert "docker exec" not in source
    assert '"-d",' not in source
    assert '"sleep",' not in source
    assert '"infinity",' not in source
    assert "treat it as a setup wrapper" not in source
    assert "prepare the patchable project in a subdirectory" not in source


def test_capture_target_patch_uses_agent_created_nested_repo(tmp_path: Path) -> None:
    task = load_task(_write_task(tmp_path))
    workspace = _prepare_agent_workspace(task, tmp_path / "run")
    repo = workspace / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "pkg.txt").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "pkg.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    task = DeepSWETask(
        task_id=task.task_id,
        task_dir=task.task_dir,
        instruction=task.instruction,
        image=task.image,
        repository_url=task.repository_url,
        base_commit=base_commit,
        allow_internet=task.allow_internet,
        verifier_timeout_sec=task.verifier_timeout_sec,
        agent_timeout_sec=task.agent_timeout_sec,
    )
    (repo / "pkg.txt").write_text("new\n", encoding="utf-8")

    result = _capture_target_patch(task=task, workspace=workspace, run_root=tmp_path / "run")

    assert result.return_code == 0
    assert "diff --git a/pkg.txt b/pkg.txt" in result.stdout
    assert (tmp_path / "run" / "model.patch").read_text(encoding="utf-8") == result.stdout


def test_capture_target_patch_allows_agent_checkout_at_workspace_root(
    tmp_path: Path,
) -> None:
    task = load_task(_write_task(tmp_path))
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=workspace,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=workspace, check=True)
    (workspace / "pkg.txt").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "pkg.txt"], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=workspace, check=True)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    task = DeepSWETask(
        task_id=task.task_id,
        task_dir=task.task_dir,
        instruction=task.instruction,
        image=task.image,
        repository_url=task.repository_url,
        base_commit=base_commit,
        allow_internet=task.allow_internet,
        verifier_timeout_sec=task.verifier_timeout_sec,
        agent_timeout_sec=task.agent_timeout_sec,
    )
    (workspace / "pkg.txt").write_text("new\n", encoding="utf-8")

    result = _capture_target_patch(task=task, workspace=workspace, run_root=tmp_path / "run")

    assert result.return_code == 0
    assert "diff --git a/pkg.txt b/pkg.txt" in result.stdout
    assert (tmp_path / "run" / "target_repo.txt").read_text(encoding="utf-8").strip() == str(
        workspace
    )


def test_capture_target_patch_includes_committed_nested_repo_changes(tmp_path: Path) -> None:
    task = load_task(_write_task(tmp_path))
    workspace = _prepare_agent_workspace(task, tmp_path / "run")
    repo = workspace / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "pkg.txt").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "pkg.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "base"], cwd=repo, check=True)
    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    task = DeepSWETask(
        task_id=task.task_id,
        task_dir=task.task_dir,
        instruction=task.instruction,
        image=task.image,
        repository_url=task.repository_url,
        base_commit=base_commit,
        allow_internet=task.allow_internet,
        verifier_timeout_sec=task.verifier_timeout_sec,
        agent_timeout_sec=task.agent_timeout_sec,
    )
    (repo / "pkg.txt").write_text("new\n", encoding="utf-8")
    subprocess.run(["git", "add", "pkg.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "agent-fix"], cwd=repo, check=True)

    result = _capture_target_patch(task=task, workspace=workspace, run_root=tmp_path / "run")

    assert result.return_code == 0
    assert "diff --git a/pkg.txt b/pkg.txt" in result.stdout
    assert "-old" in result.stdout
    assert "+new" in result.stdout


def test_deepswe_runner_script_help_uses_stdlib_types() -> None:
    runner = Path(__file__).resolve().parents[1] / "deepswe_runner.py"

    result = subprocess.run(
        [sys.executable, str(runner), "--help"],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    assert "Run one DeepSWE task through Harness." in result.stdout


def test_run_timeout_defaults_cover_harness_attempt_budget(tmp_path: Path) -> None:
    task = load_task(_write_task(tmp_path))
    args = build_parser().parse_args(
        [
            str(task.task_dir),
            "--pass-timeout-seconds",
            "100",
            "--source-change-retries",
            "1",
            "--verification-retries",
            "1",
        ]
    )

    assert _run_timeout_seconds(args, task) == 1100.0


def test_run_timeout_rejects_shorter_than_harness_attempt_budget(tmp_path: Path) -> None:
    task = load_task(_write_task(tmp_path))
    args = build_parser().parse_args(
        [
            str(task.task_dir),
            "--pass-timeout-seconds",
            "100",
            "--source-change-retries",
            "1",
            "--verification-retries",
            "1",
            "--run-timeout-seconds",
            "699",
        ]
    )

    try:
        _run_timeout_seconds(args, task)
    except ValueError as exc:
        assert "Harness attempt timeout budget" in str(exc)
    else:
        raise AssertionError("short run timeout was accepted")


def test_cli_path_normalization_uses_absolute_docker_mount_paths(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            str(tmp_path / "deep-swe" / "tasks" / "sample"),
            "--results-root",
            str(tmp_path / "relative-results"),
            "--env-file",
            str(tmp_path / ".env"),
        ]
    )

    normalized = normalize_paths(args)

    assert Path(normalized.task).is_absolute()
    assert Path(normalized.results_root).is_absolute()
    assert Path(normalized.env_file).is_absolute()


def test_outcome_shape_remains_json_serializable(tmp_path: Path) -> None:
    payload = {
        "task_id": "sample-task",
        "status": "failed",
        "error": "RuntimeError: refused unverified work",
        "run_root": str(tmp_path / "run"),
        "model": "google/gemma-4-31b-it",
        "duration_seconds": 1.25,
        "metadata": {"source_change_passed": True},
    }

    encoded = json.dumps(payload, sort_keys=True)

    assert "refused unverified work" in encoded


def test_hidden_verifier_reward_is_required_for_deepswe_pass(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    metadata: dict[str, object] = {"source_change_passed": True}

    status, error, updated = _enforce_hidden_verifier_reward(
        status="passed",
        error="",
        metadata=metadata,
        run_root=run_root,
    )

    assert status == "failed"
    assert "reward=1" in error
    assert updated["hidden_verifier_reward"] is None
    assert updated["run_error"] == error


def test_hidden_verifier_reward_allows_deepswe_pass_when_reward_is_one(tmp_path: Path) -> None:
    reward = tmp_path / "run" / "verifier" / "verifier" / "reward.txt"
    reward.parent.mkdir(parents=True)
    reward.write_text("1\n", encoding="utf-8")

    assert _hidden_verifier_reward(tmp_path / "run") == "1"

    status, error, updated = _enforce_hidden_verifier_reward(
        status="passed",
        error="",
        metadata={},
        run_root=tmp_path / "run",
    )

    assert status == "passed"
    assert error == ""
    assert updated["hidden_verifier_reward"] == "1"
    assert "run_error" not in updated


def test_public_offline_verification_strips_wrapper_repo_cd(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    target_repo = run_root / "agent-workspace" / "repo"
    target_repo.parent.mkdir(parents=True)
    (run_root / "target_repo.txt").write_text(str(target_repo) + "\n", encoding="utf-8")

    original, replay = _offline_verification_command(run_root, "cd repo && cargo test")

    assert original == "cd repo && cargo test"
    assert replay == "cargo test"
    assert _offline_verification_command(run_root, "cd other && cargo test")[1] == (
        "cd other && cargo test"
    )


def test_public_offline_verification_replays_inner_declared_docker_command(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    target_repo = run_root / "agent-workspace" / "aiomonitor_repo"
    target_repo.parent.mkdir(parents=True)
    (run_root / "target_repo.txt").write_text(str(target_repo) + "\n", encoding="utf-8")
    image = "public.ecr.aws/d3j8x8q7/swe-bench-202605:kh75rc2q0zhmsqwk7wewfwwtrx830v2n"
    command = (
        'cd aiomonitor_repo && docker run --rm --network none -v "$PWD":/app '
        f'-w /app {image} bash -lc "pytest -q"'
    )

    original, replay = _offline_verification_command(run_root, command, image=image)

    assert original == command
    assert replay == "pytest -q"
    assert (
        _offline_verification_command(
            run_root,
            command.replace("--network none", "--network bridge"),
            image=image,
        )[1]
        != "pytest -q"
    )
    assert (
        _offline_verification_command(
            run_root,
            command.replace(image, "other/image:latest"),
            image=image,
        )[1]
        != "pytest -q"
    )


def test_public_offline_verification_command_fails_on_patch_apply_error() -> None:
    command = _public_offline_verification_command("cargo test")

    assert "git apply --whitespace=nowarn /model.patch || exit $?" in command
    assert "bash -lc" in command


async def test_public_offline_verification_replays_without_hidden_test_mount(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import evals.deepswe_runner as runner

    task = load_task(_write_task(tmp_path))
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "model.patch").write_text("", encoding="utf-8")
    target_repo = run_root / "agent-workspace" / "repo"
    target_repo.parent.mkdir(parents=True)
    (run_root / "target_repo.txt").write_text(str(target_repo) + "\n", encoding="utf-8")
    calls: list[tuple[list[str], int | float | None]] = []

    async def fake_run_exec(
        argv: list[str],
        *,
        input_text: str | None = None,
        timeout_sec: int | float | None = None,
    ) -> CommandResult:
        del input_text
        calls.append((argv, timeout_sec))
        return CommandResult(stdout="ok\n", stderr="", return_code=0)

    monkeypatch.setattr(runner, "_run_exec", fake_run_exec)

    result = await _run_public_offline_verification(
        task=task,
        run_root=run_root,
        command="cd repo && cargo test",
        timeout_sec=12,
    )

    assert result.return_code == 0
    assert calls
    argv, timeout = calls[0]
    assert timeout == 12
    assert argv[:4] == ["docker", "run", "--rm", "--network"]
    assert argv[4] == "none"
    assert task.image in argv
    assert not any("/tests" in part for part in argv)
    assert "cargo test" in argv[-1]
    assert "cd repo" not in argv[-1]
    command_json = json.loads(
        (run_root / "public-offline-verification" / "command.json").read_text(encoding="utf-8")
    )
    assert command_json["original_command"] == "cd repo && cargo test"
    assert command_json["replay_command"] == "cargo test"


async def test_public_offline_verification_does_not_nest_declared_docker_verify(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import evals.deepswe_runner as runner

    task = load_task(_write_task(tmp_path))
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "model.patch").write_text("", encoding="utf-8")
    target_repo = run_root / "agent-workspace" / "repo"
    target_repo.parent.mkdir(parents=True)
    (run_root / "target_repo.txt").write_text(str(target_repo) + "\n", encoding="utf-8")
    calls: list[list[str]] = []

    async def fake_run_exec(
        argv: list[str],
        *,
        input_text: str | None = None,
        timeout_sec: int | float | None = None,
    ) -> CommandResult:
        del input_text, timeout_sec
        calls.append(argv)
        return CommandResult(stdout="ok\n", stderr="", return_code=0)

    monkeypatch.setattr(runner, "_run_exec", fake_run_exec)

    result = await _run_public_offline_verification(
        task=task,
        run_root=run_root,
        command=(
            'cd repo && docker run --rm --network none -v "$PWD":/app '
            f'-w /app {task.image} bash -lc "pytest -q"'
        ),
        timeout_sec=12,
    )

    assert result.return_code == 0
    assert calls
    assert "pytest -q" in calls[0][-1]
    assert "docker run" not in calls[0][-1]
    command_json = json.loads(
        (run_root / "public-offline-verification" / "command.json").read_text(encoding="utf-8")
    )
    assert command_json["replay_command"] == "pytest -q"


def test_public_offline_verification_runs_only_for_no_network_passes(tmp_path: Path) -> None:
    task = load_task(_write_task(tmp_path))
    metadata: dict[str, object] = {"latest_verification_command": "cargo test"}

    assert _should_run_public_offline_verification(
        task=task,
        status="passed",
        model_patch_captured=True,
        metadata=metadata,
    )
    assert not _should_run_public_offline_verification(
        task=task,
        status="failed",
        model_patch_captured=True,
        metadata=metadata,
    )
    network_task = DeepSWETask(
        task_id=task.task_id,
        task_dir=task.task_dir,
        instruction=task.instruction,
        image=task.image,
        repository_url=task.repository_url,
        base_commit=task.base_commit,
        allow_internet=True,
        verifier_timeout_sec=task.verifier_timeout_sec,
        agent_timeout_sec=task.agent_timeout_sec,
    )
    assert not _should_run_public_offline_verification(
        task=network_task,
        status="passed",
        model_patch_captured=True,
        metadata=metadata,
    )


def test_public_offline_verification_failure_blocks_deepswe_pass() -> None:
    result = CommandResult(
        stdout="",
        stderr="failed to download dependency: Could not resolve host",
        return_code=101,
    )

    status, error, updated = _enforce_public_offline_verification(
        status="passed",
        error="",
        metadata={"latest_verification_command": "cargo test"},
        result=result,
    )

    assert status == "failed"
    assert "network disabled" in error
    assert "Could not resolve host" in error
    assert updated["public_offline_verification_exit_code"] == 101
    assert updated["run_error"] == error


def test_failure_metadata_preserves_harness_runtime_cause() -> None:
    context = SimpleNamespace(
        metadata={
            "latest_runtime_error_kind": "rate_limit",
            "latest_runtime_error": "OpenRouter rate-limited (429)",
            "source_change_passed": False,
        }
    )

    metadata = _failure_metadata(context, "RuntimeError: rate_limit: OpenRouter rate-limited")

    assert metadata["latest_runtime_error_kind"] == "rate_limit"
    assert metadata["latest_runtime_error"] == "OpenRouter rate-limited (429)"
    assert metadata["source_change_passed"] is False
    assert metadata["run_error"] == "RuntimeError: rate_limit: OpenRouter rate-limited"


def test_git_diff_with_untracked_command_includes_new_files_without_mutating_index(
    tmp_path: Path,
) -> None:
    if not shutil.which("git"):
        return

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "tracked.txt").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)

    (repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
    (repo / "new.txt").write_text("new\n", encoding="utf-8")

    diff = subprocess.run(
        ["bash", "-lc", _git_diff_with_untracked_command("HEAD")],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    ).stdout

    assert "diff --git a/tracked.txt b/tracked.txt" in diff
    assert "diff --git a/new.txt b/new.txt" in diff
    assert " M tracked.txt" in status
    assert "?? new.txt" in status
    assert " A new.txt" not in status


def test_run_diagnostics_summarizes_timeout_artifacts_without_full_logs(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    events = run_root / "harness" / "harness-events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        "\n".join(
            [
                json.dumps({"type": "model_request"}),
                json.dumps({"type": "tool_call", "call": {"name": "shell"}}),
                json.dumps(
                    {
                        "type": "tool_result",
                        "result": {
                            "name": "shell",
                            "is_error": True,
                            "content": "exit_code: 1\nstdout: failed",
                        },
                    }
                ),
                json.dumps({"type": "prediction_mismatch"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_root / "final.diff").write_text(
        "diff --git a/src/a.go b/src/a.go\n--- a/src/a.go\n+++ b/src/a.go\n",
        encoding="utf-8",
    )

    diagnostics = _run_diagnostics(run_root)

    assert diagnostics["event_counts"] == {
        "model_request": 1,
        "tool_call": 1,
        "tool_result": 1,
        "prediction_mismatch": 1,
    }
    assert diagnostics["tool_counts"] == {"shell": 1}
    assert diagnostics["tool_error_count"] == 1
    assert diagnostics["prediction_mismatch_count"] == 1
    assert diagnostics["changed_files"] == ["src/a.go"]
