"""Run a shell/Makefile external-workspace validation scenario through Harness."""

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
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from evals.external_scenario_checks import (
    announce_scenario_start,
    command_output_text,
    git_diff_with_untracked,
    independent_check_failure,
    load_dotenv,
    untracked_scratch_paths,
)
from harness.cli.external_workspace import run_harness_on_external_environment


@dataclass(frozen=True)
class LocalCommandResult:
    stdout: str
    stderr: str
    return_code: int


class LocalEnvironment:
    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_sec: float | int | None = None,
    ) -> LocalCommandResult:
        working_dir = Path(cwd) if cwd else self.workdir

        def run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                command,
                cwd=working_dir,
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
            return LocalCommandResult(
                stdout=command_output_text(exc.stdout),
                stderr=(
                    command_output_text(exc.stderr) + f"\ncommand timed out after {timeout_sec}s"
                ),
                return_code=124,
            )
        return LocalCommandResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            return_code=completed.returncode,
        )


def _write(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def create_workspace(root: Path) -> Path:
    workspace = root / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    _write(
        workspace / "Makefile",
        """
test:
\t./tests/run.sh
""".lstrip(),
    )
    _write(
        workspace / "bin" / "ini_get",
        """
#!/usr/bin/env bash
section="$1"
key="$2"
file="$3"
current=""

while IFS= read -r line; do
  case "$line" in
    "["*"]") current="${line#\\[}"; current="${current%\\]}" ;;
    "$key="*) if [ "$current" = "$section" ]; then printf '%s\\n' "${line#*=}"; exit 0; fi ;;
  esac
done < "$file"
exit 1
""".lstrip(),
        executable=True,
    )
    _write(
        workspace / "tests" / "run.sh",
        """
#!/usr/bin/env bash
set -euo pipefail

tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

cat > "$tmp" <<'INI'
[server]
port=8080

[client]
port=9090
INI

[ "$(./bin/ini_get server port "$tmp")" = "8080" ]
[ "$(./bin/ini_get client port "$tmp")" = "9090" ]

echo "ok"
""".lstrip(),
        executable=True,
    )
    _write(
        workspace / "README.md",
        """
# INI Tools Fixture

Use the Makefile to discover the project test command.
""".lstrip(),
    )

    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "harness@example.test"], cwd=workspace, check=True
    )
    subprocess.run(["git", "config", "user.name", "Harness Eval"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial fixture"], cwd=workspace, check=True)
    return workspace


def _run_ini_get(
    workspace: Path, section: str, key: str, file_path: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["./bin/ini_get", section, key, str(file_path.resolve())],
        cwd=workspace,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=30,
    )


def _status_paths(status_text: str) -> list[str]:
    paths: list[str] = []
    for line in status_text.splitlines():
        if len(line) < 4:
            continue
        paths.append(line[3:])
    return paths


def independent_check(workspace: Path) -> dict[str, object]:
    make_result = subprocess.run(
        ["make", "test"],
        cwd=workspace,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=60,
    )
    hidden_ini = workspace / ".hidden-check.ini"
    hidden_ini.write_text(
        """
# leading comments should be ignored
; semicolon comments should also be ignored

[ server ]
  port = 8080
token = abc=def=ghi
dsn = postgres://user:pass@example.test/db?sslmode=require
description = value with spaces

[client]
port = 9090
""".lstrip(),
        encoding="utf-8",
    )
    cases = [
        ("server", "port", 0, "8080"),
        ("server", "token", 0, "abc=def=ghi"),
        ("server", "dsn", 0, "postgres://user:pass@example.test/db?sslmode=require"),
        ("server", "description", 0, "value with spaces"),
        ("client", "port", 0, "9090"),
        ("server", "missing", 1, ""),
    ]
    behavior: list[dict[str, object]] = []
    behavior_passed = True
    for section, key, expected_code, expected_stdout in cases:
        result = _run_ini_get(workspace, section, key, hidden_ini)
        actual_stdout = result.stdout.strip()
        ok = result.returncode == expected_code and actual_stdout == expected_stdout
        behavior_passed = behavior_passed and ok
        behavior.append(
            {
                "section": section,
                "key": key,
                "return_code": result.returncode,
                "stdout": result.stdout[-500:],
                "stderr": result.stderr[-500:],
                "expected_code": expected_code,
                "expected_stdout": expected_stdout,
                "passed": ok,
            }
        )
    hidden_ini.unlink(missing_ok=True)
    status_result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=workspace,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=30,
    )
    status_text = status_result.stdout
    changed_paths = _status_paths(status_text)
    changed_test_paths = [
        path for path in changed_paths if path == "Makefile" or path.startswith("tests/")
    ]
    created_standalone_regression = any(
        path.startswith("tests/") and path != "tests/run.sh" for path in changed_test_paths
    )
    runner_wired_new_regression = "Makefile" in status_text or "tests/run.sh" in status_text
    leftover_scratch_paths = untracked_scratch_paths(
        status_text,
        allowed_untracked=set(changed_test_paths),
    )
    diff_text = git_diff_with_untracked(
        workspace,
        ["bin/ini_get", "tests"],
        status_text=status_text,
    )
    return {
        "make_test_return_code": make_result.returncode,
        "make_test_stdout": make_result.stdout[-1000:],
        "make_test_stderr": make_result.stderr[-1000:],
        "behavior_passed": behavior_passed,
        "behavior": behavior,
        "diff": diff_text,
        "status": status_text,
        "changed_source": "bin/ini_get" in status_result.stdout,
        "changed_tests": bool(changed_test_paths),
        "changed_test_paths": changed_test_paths,
        "created_standalone_regression": created_standalone_regression,
        "runner_wired_new_regression": (not created_standalone_regression)
        or runner_wired_new_regression,
        "leftover_scratch_paths": leftover_scratch_paths,
    }


async def run(args: argparse.Namespace) -> int:
    load_dotenv(Path(args.env_file))
    run_root = Path(args.results_root) / f"external-shell-ini-{uuid4().hex[:8]}"
    run_root.mkdir(parents=True, exist_ok=True)
    workspace = create_workspace(run_root)
    announce_scenario_start(run_root=run_root, workspace=workspace)
    logs_dir = run_root / "harness"
    context = SimpleNamespace(n_agent_steps=0, metadata={})
    error = ""
    try:
        await run_harness_on_external_environment(
            instruction=(
                "Fix the INI lookup CLI in this repository. bin/ini_get takes "
                "SECTION KEY FILE and prints the matching value. It should ignore blank "
                "lines and full-line comments beginning with # or ;, trim surrounding "
                "whitespace around section names, keys, and values, preserve equals "
                "signs inside values, keep section scoping correct, and exit non-zero "
                "with no output when the key is missing. Discover the project test "
                "command from the repository, add focused regression tests, and verify "
                "the work."
            ),
            environment=LocalEnvironment(workspace),
            context=context,
            logs_dir=str(logs_dir),
            provider_name=args.provider,
            model_name=args.model,
            goal_plan=not bool(args.no_goal_plan),
            max_steps=args.max_steps,
            max_output_tokens=args.max_output_tokens,
            source_change_retries=args.source_change_retries,
            verification_retries=args.verification_retries,
            pass_timeout_seconds=args.pass_timeout_seconds,
            model_stream_idle_timeout_seconds=args.idle_timeout,
            model_turn_timeout_seconds=args.turn_timeout,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    try:
        check = independent_check(workspace)
    except Exception as exc:
        error = error or f"{type(exc).__name__}: {exc}"
        check = independent_check_failure(exc)
    passed = (
        not error
        and check.get("make_test_return_code") == 0
        and check.get("behavior_passed") is True
        and check.get("changed_source") is True
        and check.get("changed_tests") is True
        and check.get("runner_wired_new_regression") is True
        and not check.get("leftover_scratch_paths")
    )
    outcome = {
        "status": "passed" if passed else "failed",
        "error": error,
        "run_root": str(run_root),
        "workspace": str(workspace),
        "context_metadata": context.metadata,
        "independent_check": check,
    }
    (run_root / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    (run_root / "final.diff").write_text(str(check.get("diff", "")), encoding="utf-8")
    print(json.dumps(outcome, indent=2))
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="evals/results/live-scenarios")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--provider", default="openrouter")
    parser.add_argument("--model", default="openai/gpt-5.4-nano")
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument(
        "--no-goal-plan",
        action="store_true",
        help="Skip the optional LLM planning pre-call and enter the tool loop directly.",
    )
    parser.add_argument("--source-change-retries", type=int, default=1)
    parser.add_argument("--verification-retries", type=int, default=1)
    parser.add_argument("--pass-timeout-seconds", type=float, default=260.0)
    parser.add_argument("--idle-timeout", type=float, default=120.0)
    parser.add_argument("--turn-timeout", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
