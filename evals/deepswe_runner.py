"""Run DeepSWE tasks through Harness' external workspace runtime.

The Harness agent receives only public task artifacts and ordinary tools. It is
responsible for discovering Docker/git/project setup itself and producing a
patchable target repository. Hidden tests and reference solutions stay outside
the Harness loop and are used only after the agent stops, to grade the final
patch.
"""

from __future__ import annotations

import sys

if __package__ in (None, "") and sys.path:
    _script_dir = sys.path[0]
    if _script_dir.endswith("/evals"):
        sys.path.pop(0)
        sys.path.insert(0, _script_dir.rsplit("/", 1)[0])

import argparse
import asyncio
import json
import os
import shlex
import shutil
import subprocess
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from harness.cli.external_workspace import (
    ExternalWorkspacePolicy,
    external_workspace_total_attempts,
    run_harness_on_external_environment,
)
from harness.core.command_env import clean_command_env


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    return_code: int


@dataclass(frozen=True)
class DeepSWETask:
    task_id: str
    task_dir: Path
    instruction: str
    image: str
    repository_url: str
    base_commit: str
    allow_internet: bool
    verifier_timeout_sec: int
    agent_timeout_sec: int
    display_title: str = ""
    display_description: str = ""
    original_title: str = ""


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ[key] = value


def _failure_metadata(context: SimpleNamespace, error: str) -> dict[str, object]:
    existing = getattr(context, "metadata", {})
    metadata = dict(existing) if isinstance(existing, dict) else {}
    metadata["run_error"] = error
    return metadata


def _harness_attempt_timeout_seconds(args: argparse.Namespace) -> float:
    attempts = max(
        1,
        external_workspace_total_attempts(
            source_change_retries=int(args.source_change_retries),
            verification_retries=int(args.verification_retries),
        ),
    )
    return max(1.0, float(args.pass_timeout_seconds)) * attempts


def _run_timeout_seconds(args: argparse.Namespace, task: DeepSWETask) -> float:
    harness_timeout = _harness_attempt_timeout_seconds(args)
    requested = float(args.run_timeout_seconds or 0.0)
    if requested > 0:
        if requested < harness_timeout:
            raise ValueError(
                "--run-timeout-seconds must be at least the Harness attempt timeout "
                f"budget ({harness_timeout:.1f}s for pass-timeout-seconds="
                f"{float(args.pass_timeout_seconds):.1f}, source-change-retries="
                f"{int(args.source_change_retries)}, verification-retries="
                f"{int(args.verification_retries)})."
            )
        return requested
    return max(float(task.agent_timeout_sec), harness_timeout)


def _run_diagnostics(run_root: Path) -> dict[str, object]:
    diagnostics: dict[str, object] = {}
    events_path = run_root / "harness" / "harness-events.jsonl"
    if events_path.exists():
        event_counts: dict[str, int] = {}
        tool_counts: dict[str, int] = {}
        tool_errors: list[dict[str, object]] = []
        prediction_mismatches = 0
        last_event_type = ""
        for raw_line in events_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            event_type = str(event.get("type") or "")
            if event_type:
                last_event_type = event_type
                event_counts[event_type] = event_counts.get(event_type, 0) + 1
            if event_type == "tool_call":
                call = event.get("call") if isinstance(event.get("call"), dict) else {}
                name = str(call.get("name") or "")
                if name:
                    tool_counts[name] = tool_counts.get(name, 0) + 1
            elif event_type == "tool_result":
                result = event.get("result") if isinstance(event.get("result"), dict) else {}
                if result.get("is_error") is True:
                    tool_errors.append(
                        {
                            "name": str(result.get("name") or ""),
                            "content_preview": str(result.get("content") or "")[:300],
                        }
                    )
            elif event_type == "prediction_mismatch":
                prediction_mismatches += 1
        diagnostics["event_counts"] = event_counts
        diagnostics["tool_counts"] = tool_counts
        diagnostics["tool_error_count"] = len(tool_errors)
        diagnostics["tool_error_previews"] = tool_errors[-5:]
        diagnostics["prediction_mismatch_count"] = prediction_mismatches
        diagnostics["last_event_type"] = last_event_type

    diff_path = run_root / "final.diff"
    if diff_path.exists():
        diff_text = diff_path.read_text(encoding="utf-8", errors="replace")
        diagnostics["final_diff_bytes"] = diff_path.stat().st_size
        changed_files: list[str] = []
        for line in diff_text.splitlines():
            if not line.startswith("diff --git "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                changed_files.append(parts[3].removeprefix("b/"))
        diagnostics["changed_files"] = changed_files
    return diagnostics


def _hidden_verifier_reward(run_root: Path) -> str | None:
    reward_path = run_root / "verifier" / "verifier" / "reward.txt"
    if not reward_path.exists():
        return None
    reward = reward_path.read_text(encoding="utf-8", errors="replace").strip()
    return reward or None


def _should_run_hidden_verifier(*, status: str, model_patch_captured: bool) -> bool:
    return status == "passed" and model_patch_captured


def _enforce_hidden_verifier_reward(
    *,
    status: str,
    error: str,
    metadata: dict[str, object],
    run_root: Path,
) -> tuple[str, str, dict[str, object]]:
    updated = dict(metadata)
    reward = _hidden_verifier_reward(run_root)
    updated["hidden_verifier_reward"] = reward
    if status == "passed" and reward != "1":
        error = (
            "RuntimeError: hidden verifier did not produce reward=1; refusing to "
            "report DeepSWE task completion without passing isolated grading."
        )
        updated["run_error"] = error
        status = "failed"
    return status, error, updated


def _hidden_verifier_command() -> str:
    return (
        "mkdir -p /logs/verifier /logs/artifacts; "
        "if [ -s /model.patch ]; then git apply --whitespace=nowarn /model.patch; fi; "
        "bash /tests/test.sh; "
        "reward=$(cat /logs/verifier/reward.txt 2>/dev/null || echo 0); "
        'test "$reward" = 1'
    )


def _git_diff_with_untracked_command(base_ref: str) -> str:
    quoted_base = shlex.quote(base_ref)
    return (
        "tmp_index=$(mktemp); "
        "trap 'rm -f \"$tmp_index\"' EXIT; "
        f'GIT_INDEX_FILE="$tmp_index" git read-tree {quoted_base}; '
        'GIT_INDEX_FILE="$tmp_index" git add -N -- .; '
        f'GIT_INDEX_FILE="$tmp_index" git diff --binary {quoted_base}'
    )


def _copy_public_task_files(task: DeepSWETask, workspace: Path) -> None:
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copy2(task.task_dir / "task.toml", workspace / "task.toml")
    shutil.copy2(task.task_dir / "instruction.md", workspace / "instruction.md")
    environment_dir = task.task_dir / "environment"
    if environment_dir.exists():
        shutil.copytree(environment_dir, workspace / "environment")


def _init_public_workspace_git(workspace: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    info_exclude = workspace / ".git" / "info" / "exclude"
    with info_exclude.open("a", encoding="utf-8") as handle:
        handle.write("\n.harness-home/\n")
    subprocess.run(
        ["git", "config", "user.email", "harness@example.test"], cwd=workspace, check=True
    )
    subprocess.run(["git", "config", "user.name", "Harness Eval"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "public task files"], cwd=workspace, check=True)


def _prepare_agent_workspace(task: DeepSWETask, run_root: Path) -> Path:
    workspace = run_root / "agent-workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    _copy_public_task_files(task, workspace)
    _init_public_workspace_git(workspace)
    return workspace


def _git_stdout(repo: Path, *args: str, timeout_sec: int = 30) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _git_returncode(repo: Path, *args: str, timeout_sec: int = 30) -> int:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        timeout=timeout_sec,
        check=False,
    )
    return result.returncode


def _candidate_target_repos(workspace: Path, base_commit: str) -> list[Path]:
    candidates: list[Path] = []
    for git_dir in workspace.rglob(".git"):
        repo = git_dir.parent
        if repo == workspace:
            continue
        head = _git_stdout(repo, "rev-parse", "HEAD")
        base_head = _git_stdout(repo, "rev-parse", base_commit)
        if not head or not base_head:
            continue
        if (
            head == base_head
            or _git_returncode(
                repo,
                "merge-base",
                "--is-ancestor",
                base_head,
                "HEAD",
            )
            == 0
        ):
            candidates.append(repo)
    return sorted(candidates, key=lambda path: len(path.parts), reverse=True)


def _capture_target_patch(
    *,
    task: DeepSWETask,
    workspace: Path,
    run_root: Path,
) -> CommandResult:
    patch_path = run_root / "model.patch"
    errors: list[str] = []
    for repo in _candidate_target_repos(workspace, task.base_commit):
        result = subprocess.run(
            ["bash", "-lc", _git_diff_with_untracked_command(task.base_commit)],
            cwd=repo,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            errors.append(f"{repo}: {result.stderr or result.stdout}")
            continue
        if not result.stdout.strip():
            errors.append(f"{repo}: no patch relative to base commit")
            continue
        patch_path.write_text(result.stdout, encoding="utf-8")
        (run_root / "target_repo.txt").write_text(str(repo) + "\n", encoding="utf-8")
        return CommandResult(stdout=result.stdout, stderr="", return_code=0)
    message = (
        "agent did not leave a modified nested git repository at the task base commit"
        if not errors
        else "\n".join(errors)
    )
    patch_path.write_text("", encoding="utf-8")
    return CommandResult(stdout="", stderr=message, return_code=1)


async def _run_hidden_verifier(
    *,
    task: DeepSWETask,
    run_root: Path,
    timeout_sec: int | None = None,
) -> CommandResult:
    patch_path = run_root / "model.patch"
    if not patch_path.exists():
        return CommandResult(
            stdout="",
            stderr="model.patch was not captured before hidden verifier run",
            return_code=2,
        )
    verify_timeout = timeout_sec or min(task.verifier_timeout_sec, 300)
    tests_dir = task.task_dir / "tests"
    verifier_dir = run_root / "verifier"
    verifier_dir.mkdir(parents=True, exist_ok=True)
    return await _run_exec(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--workdir",
            "/app",
            "-v",
            f"{tests_dir}:/tests:ro",
            "-v",
            f"{verifier_dir}:/logs",
            "-v",
            f"{patch_path}:/model.patch:ro",
            task.image,
            "bash",
            "-lc",
            _hidden_verifier_command(),
        ],
        timeout_sec=verify_timeout,
    )


def load_task(task_dir: Path) -> DeepSWETask:
    task_dir = task_dir.resolve()
    with (task_dir / "task.toml").open("rb") as handle:
        task_toml = tomllib.load(handle)
    metadata = task_toml["metadata"]
    environment = task_toml["environment"]
    verifier = task_toml.get("verifier", {})
    agent = task_toml.get("agent", {})
    return DeepSWETask(
        task_id=str(metadata["task_id"]),
        task_dir=task_dir,
        instruction=(task_dir / "instruction.md").read_text(encoding="utf-8").strip(),
        image=str(environment["docker_image"]),
        repository_url=str(metadata["repository_url"]),
        base_commit=str(metadata["base_commit_hash"]),
        allow_internet=bool(environment.get("allow_internet", False)),
        verifier_timeout_sec=int(float(verifier.get("timeout_sec", 1800))),
        agent_timeout_sec=int(float(agent.get("timeout_sec", 5400))),
        display_title=str(metadata.get("display_title") or ""),
        display_description=str(metadata.get("display_description") or ""),
        original_title=str(metadata.get("original_title") or ""),
    )


async def _run_exec(
    argv: list[str],
    *,
    input_text: str | None = None,
    timeout_sec: int | float | None = None,
) -> CommandResult:
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if input_text is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input_text.encode("utf-8") if input_text is not None else None),
            timeout=timeout_sec,
        )
    except TimeoutError:
        proc.kill()
        stdout, stderr = await proc.communicate()
        return CommandResult(
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace") + f"\ncommand timed out after {timeout_sec}s",
            124,
        )
    return CommandResult(
        stdout.decode("utf-8", errors="replace"),
        stderr.decode("utf-8", errors="replace"),
        proc.returncode or 0,
    )


class HostWorkspaceEnvironment:
    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self.home = workdir / ".harness-home"

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> CommandResult:
        working_dir = Path(cwd) if cwd else self.workdir

        def run() -> subprocess.CompletedProcess[str]:
            env = clean_command_env(working_dir)
            self.home.mkdir(parents=True, exist_ok=True)
            env["HOME"] = str(self.home)
            env["XDG_CACHE_HOME"] = str(self.home / ".cache")
            env["XDG_CONFIG_HOME"] = str(self.home / ".config")
            env["XDG_DATA_HOME"] = str(self.home / ".local" / "share")
            env["PYTHONUSERBASE"] = str(self.home / ".python-userbase")
            env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
            return subprocess.run(
                command,
                cwd=working_dir,
                env=env,
                shell=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=float(timeout_sec or 30),
                check=False,
            )

        try:
            completed = await asyncio.to_thread(run)
        except subprocess.TimeoutExpired as exc:
            return CommandResult(
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + f"\ncommand timed out after {timeout_sec}s",
                return_code=124,
            )
        return CommandResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            return_code=completed.returncode,
        )


def _policy_for_task(
    task: DeepSWETask,
    run_root: Path,
    *,
    allow_web_access: bool = True,
) -> ExternalWorkspacePolicy:
    hidden_tests = task.task_dir / "tests"
    solution = task.task_dir / "solution"
    repository_fragments = _repository_forbidden_fragments(task.repository_url)
    benchmark_web_fragments = (
        task.task_id,
        task.display_title,
        task.display_description,
        task.original_title,
        "DeepSWE",
        "deep swe",
        "deep-swe",
        "DataCurve",
        "deepswe.datacurve.ai",
        "deepswe.datacurve.ai/data/tasks",
        f"deepswe.datacurve.ai/data/tasks/{task.task_id}",
        f"data/tasks/{task.task_id}",
    )
    benchmark_web_fragments = tuple(
        dict.fromkeys(fragment for fragment in benchmark_web_fragments if fragment)
    )
    return ExternalWorkspacePolicy(
        forbidden_path_parts=("solution",),
        forbidden_path_names=("solution.patch", "solve.sh", "test.patch"),
        forbidden_absolute_paths=(
            str(hidden_tests),
            str(solution),
            "/tests",
            "/solution",
            "/logs",
        ),
        forbidden_text_fragments=(
            "github.com/datacurve-ai/deep-swe",
            "datacurve-ai/deep-swe",
            "deep-swe/tasks",
            str(hidden_tests),
            str(solution),
            "solution.patch",
            "solve.sh",
            "test.patch",
        ),
        forbidden_web_fragments=(*repository_fragments, *benchmark_web_fragments),
        allowed_git_clone_fragments=repository_fragments,
        required_git_base_commit=task.base_commit,
        allow_web_access=allow_web_access,
        block_root_filesystem_probe=True,
        refusal_message="refused: external workspace policy blocks access to restricted artifacts",
    )


def _repository_forbidden_fragments(repository_url: str) -> tuple[str, ...]:
    url = repository_url.strip().rstrip("/")
    if not url:
        return ()
    fragments = [url]
    if "://" in url:
        without_scheme = url.split("://", 1)[1]
        fragments.append(without_scheme)
    else:
        without_scheme = url
    parts = [part for part in without_scheme.split("/") if part]
    if len(parts) >= 3 or len(parts) >= 2:
        fragments.append("/".join(parts[-2:]))
    return tuple(dict.fromkeys(fragments))


def _write_jsonl(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


async def run_deepswe_task(args: argparse.Namespace) -> int:
    _load_dotenv(Path(args.env_file))
    task = load_task(Path(args.task))
    try:
        run_timeout = _run_timeout_seconds(args, task)
    except ValueError as exc:
        print(f"DEEPSWE_RUN_FAILED=ValueError: {exc}", flush=True)
        return 2
    run_id = uuid4().hex[:8]
    results_root = Path(args.results_root)
    run_root = results_root / f"{task.task_id}-{run_id}"
    harness_logs = run_root / "harness"
    run_root.mkdir(parents=True, exist_ok=True)
    agent_workspace = _prepare_agent_workspace(task, run_root)
    print(f"DEEPSWE_RUN_PID={os.getpid()}", flush=True)
    print(f"TASK={task.task_id}", flush=True)
    print(f"RUN_ROOT={run_root}", flush=True)
    print(f"AGENT_WORKSPACE={agent_workspace}", flush=True)

    status = "failed"
    error = ""
    metadata: dict[str, object] = {}
    context = SimpleNamespace(n_agent_steps=0, metadata={})
    started = asyncio.get_running_loop().time()
    hidden_verifier_result: CommandResult | None = None
    verifier_timeout_seconds = min(
        task.verifier_timeout_sec,
        max(1, int(args.verifier_timeout_seconds)),
    )
    try:
        await asyncio.wait_for(
            run_harness_on_external_environment(
                instruction=task.instruction,
                environment=HostWorkspaceEnvironment(agent_workspace),
                context=context,
                logs_dir=str(harness_logs),
                model_name=args.model,
                max_steps=args.max_steps,
                max_output_tokens=args.max_output_tokens,
                source_change_retries=args.source_change_retries,
                verification_retries=args.verification_retries,
                pass_timeout_seconds=args.pass_timeout_seconds,
                require_regression_test_change=True,
                policy=_policy_for_task(
                    task,
                    run_root,
                    allow_web_access=bool(args.allow_web_access),
                ),
                default_verify_command=None,
                default_verify_timeout_seconds=None,
                model_stream_idle_timeout_seconds=args.idle_timeout,
                model_turn_timeout_seconds=args.turn_timeout,
            ),
            timeout=run_timeout,
        )
        status = "passed"
        metadata = dict(context.metadata)
    except TimeoutError:
        error = f"TimeoutError: DeepSWE run exceeded wall-clock timeout of {run_timeout}s"
        print(f"DEEPSWE_RUN_FAILED={error}", flush=True)
        metadata = _failure_metadata(context, error)
        return_code = 1
    except Exception as exc:
        error = f"{exc.__class__.__name__}: {exc}"
        print(f"DEEPSWE_RUN_FAILED={error}", flush=True)
        metadata = _failure_metadata(context, error)
        return_code = 1
    else:
        return_code = 0
    finally:
        capture_result = _capture_target_patch(
            task=task,
            workspace=agent_workspace,
            run_root=run_root,
        )
        if capture_result.return_code == 0:
            (run_root / "final.diff").write_text(capture_result.stdout, encoding="utf-8")
        else:
            (run_root / "final.diff").write_text("", encoding="utf-8")
            metadata = {
                **metadata,
                "model_patch_capture_error": (capture_result.stderr or capture_result.stdout)[
                    :1000
                ],
                "hidden_verifier_skipped": "model patch was not captured",
            }
        model_patch_captured = capture_result.return_code == 0
        if _should_run_hidden_verifier(
            status=status,
            model_patch_captured=model_patch_captured,
        ):
            hidden_verifier_result = await _run_hidden_verifier(
                task=task,
                run_root=run_root,
                timeout_sec=verifier_timeout_seconds,
            )
            (run_root / "verifier" / "stdout.txt").write_text(
                hidden_verifier_result.stdout,
                encoding="utf-8",
                errors="replace",
            )
            (run_root / "verifier" / "stderr.txt").write_text(
                hidden_verifier_result.stderr,
                encoding="utf-8",
                errors="replace",
            )
        elif model_patch_captured and status != "passed":
            metadata = {
                **metadata,
                "hidden_verifier_skipped": "harness did not accept completion",
            }
        duration = asyncio.get_running_loop().time() - started
        metadata = {**metadata, "run_diagnostics": _run_diagnostics(run_root)}
        if hidden_verifier_result is not None:
            metadata = {
                **metadata,
                "hidden_verifier_exit_code": hidden_verifier_result.return_code,
            }
        status, error, metadata = _enforce_hidden_verifier_reward(
            status=status,
            error=error,
            metadata=metadata,
            run_root=run_root,
        )
        if status != "passed":
            return_code = 1
        outcome = {
            "task_id": task.task_id,
            "status": status,
            "error": error,
            "run_root": str(run_root),
            "model": args.model,
            "duration_seconds": round(duration, 3),
            "metadata": metadata,
        }
        (run_root / "outcome.json").write_text(
            json.dumps(outcome, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_jsonl(results_root.parent / "history.jsonl", outcome)
        print(f"RUN_ROOT={run_root}", flush=True)
        print(f"DEEPSWE_STATUS={status}", flush=True)
    return return_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one DeepSWE task through Harness.")
    parser.add_argument("task", help="Path to deep-swe/tasks/<task-id>")
    parser.add_argument("--results-root", default="evals/results/deepswe")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", default="openai/gpt-5.4-nano")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--source-change-retries", type=int, default=1)
    parser.add_argument("--verification-retries", type=int, default=1)
    parser.add_argument("--pass-timeout-seconds", type=float, default=900.0)
    web_access = parser.add_mutually_exclusive_group()
    web_access.add_argument(
        "--allow-web-access",
        dest="allow_web_access",
        action="store_true",
        help=(
            "Allow the Harness agent to use read-only public web tools for autonomous "
            "research. Task-scoped policy still blocks hidden benchmark artifacts, "
            "source-repository web search, and solution-bearing pages."
        ),
    )
    web_access.add_argument(
        "--no-web-access",
        dest="allow_web_access",
        action="store_false",
        help="Disable read-only public web tools for this run.",
    )
    parser.set_defaults(allow_web_access=True)
    parser.add_argument(
        "--verifier-timeout-seconds",
        type=float,
        default=300.0,
        help="Cap each hidden verifier run. The task timeout is still respected if lower.",
    )
    parser.add_argument(
        "--run-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Hard wall-clock timeout for the whole Harness attempt. "
            "Defaults to the task agent timeout when 0."
        ),
    )
    parser.add_argument("--idle-timeout", type=float, default=90.0)
    parser.add_argument("--turn-timeout", type=float, default=120.0)
    return parser


def normalize_paths(args: argparse.Namespace) -> argparse.Namespace:
    args.task = str(Path(args.task).resolve(strict=False))
    args.results_root = str(Path(args.results_root).resolve(strict=False))
    args.env_file = str(Path(args.env_file).resolve(strict=False))
    return args


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = normalize_paths(parser.parse_args(argv))
    return asyncio.run(run_deepswe_task(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
