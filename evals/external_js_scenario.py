"""Run a JavaScript external-workspace validation scenario through Harness."""

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
                stdout=exc.stdout or "",
                stderr=(exc.stderr or "") + f"\ncommand timed out after {timeout_sec}s",
                return_code=124,
            )
        return LocalCommandResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            return_code=completed.returncode,
        )


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def create_workspace(root: Path) -> Path:
    workspace = root / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    _write(
        workspace / "package.json",
        """
{
  "name": "url-tools-fixture",
  "version": "1.0.0",
  "type": "commonjs",
  "scripts": {
    "test": "node --test"
  }
}
""".lstrip(),
    )
    _write(
        workspace / "src" / "urlJoin.js",
        """
function joinUrl(...segments) {
  return segments.filter(Boolean).join("/");
}

module.exports = { joinUrl };
""".lstrip(),
    )
    _write(
        workspace / "test" / "urlJoin.test.js",
        """
const assert = require("node:assert/strict");
const test = require("node:test");
const { joinUrl } = require("../src/urlJoin");

test("joins simple path segments", () => {
  assert.equal(joinUrl("api", "v1", "users"), "api/v1/users");
});
""".lstrip(),
    )
    _write(
        workspace / "README.md",
        """
# URL Tools Fixture

Use the package metadata to discover how to run the test suite.
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


def independent_check(workspace: Path) -> dict[str, object]:
    npm_result = subprocess.run(
        ["npm", "test"],
        cwd=workspace,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=60,
    )
    behavior_script = """
const assert = require("node:assert/strict");
const { joinUrl } = require("./src/urlJoin");
const cases = [
  [["https://example.com/", "/api/", "v1"], "https://example.com/api/v1"],
  [["https://example.com", "", null, "api"], "https://example.com/api"],
  [["/api/", "/v1/", "/users/"], "/api/v1/users"],
  [["api/", "/v1"], "api/v1"],
];
for (const [input, expected] of cases) {
  assert.equal(joinUrl(...input), expected);
}
"""
    behavior_result = subprocess.run(
        ["node", "-e", behavior_script],
        cwd=workspace,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=30,
    )
    diff_result = subprocess.run(
        ["git", "diff", "--", "src/urlJoin.js", "test/urlJoin.test.js"],
        cwd=workspace,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=30,
    )
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
    leftover_scratch_paths = untracked_scratch_paths(status_text)
    return {
        "npm_test_return_code": npm_result.returncode,
        "npm_test_stdout": npm_result.stdout[-1000:],
        "npm_test_stderr": npm_result.stderr[-1000:],
        "behavior_return_code": behavior_result.returncode,
        "behavior_stdout": behavior_result.stdout[-1000:],
        "behavior_stderr": behavior_result.stderr[-1000:],
        "diff": diff_result.stdout,
        "status": status_text,
        "changed_source": "src/urlJoin.js" in status_text,
        "changed_tests": "test/urlJoin.test.js" in status_text,
        "leftover_scratch_paths": leftover_scratch_paths,
    }


async def run(args: argparse.Namespace) -> int:
    load_dotenv(Path(args.env_file))
    run_root = Path(args.results_root) / f"external-js-url-{uuid4().hex[:8]}"
    run_root.mkdir(parents=True, exist_ok=True)
    workspace = create_workspace(run_root)
    announce_scenario_start(run_root=run_root, workspace=workspace)
    logs_dir = run_root / "harness"
    context = SimpleNamespace(n_agent_steps=0, metadata={})
    error = ""
    try:
        await run_harness_on_external_environment(
            instruction=(
                "Fix URL joining in this JavaScript package. joinUrl should preserve "
                "a URL scheme and host, ignore nullish or empty segments, collapse "
                "duplicate slashes between path segments, keep a leading slash for "
                "absolute paths, and trim trailing slashes except for the root URL. "
                "Discover the project test command from the repository, add focused "
                "regression tests, and verify the work."
            ),
            environment=LocalEnvironment(workspace),
            context=context,
            logs_dir=str(logs_dir),
            model_name=args.model,
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
        and check.get("npm_test_return_code") == 0
        and check.get("behavior_return_code") == 0
        and check.get("changed_source") is True
        and check.get("changed_tests") is True
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
    parser.add_argument("--model", default="openai/gpt-5.4-nano")
    parser.add_argument("--max-steps", type=int, default=45)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
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
