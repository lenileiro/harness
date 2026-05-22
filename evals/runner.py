"""Eval fixture runner.

Discovers fixtures under evals/fixtures/, runs each one by:
  1. shutil.copytree to a temp dir
  2. git init + initial commit (clean baseline)
  3. reads TASK.md as the agent prompt
  4. invokes `harness run` as a subprocess (captures stdout+stderr)
  5. captures `git diff HEAD` (what the agent changed)
  6. runs `python -m pytest tests/` (whether the fix is correct)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FixtureMeta:
    name: str
    path: Path
    task_text: str
    eval_md: str  # raw EVAL.md text, passed to judge as context


@dataclass
class RunOutcome:
    fixture: FixtureMeta
    transcript: str
    git_diff: str
    test_output: str
    agent_exit_code: int
    test_exit_code: int


def _find_evals_root() -> Path:
    """Walk CWD upward to find evals/fixtures/."""
    current = Path.cwd().resolve()
    while True:
        candidate = current / "evals" / "fixtures"
        if candidate.is_dir():
            return current / "evals"
        parent = current.parent
        if parent == current:
            raise FileNotFoundError(
                "Could not find evals/fixtures/ — run from inside the harness repo."
            )
        current = parent


def discover_fixtures(evals_root: Path | None = None) -> list[FixtureMeta]:
    """Return all fixtures sorted by directory name."""
    root = evals_root or _find_evals_root()
    fixtures_dir = root / "fixtures"
    result: list[FixtureMeta] = []
    for entry in sorted(fixtures_dir.iterdir()):
        if not entry.is_dir():
            continue
        task_path = entry / "TASK.md"
        eval_path = entry / "EVAL.md"
        if not task_path.exists() or not eval_path.exists():
            continue
        result.append(
            FixtureMeta(
                name=entry.name,
                path=entry,
                task_text=task_path.read_text(encoding="utf-8"),
                eval_md=eval_path.read_text(encoding="utf-8"),
            )
        )
    return result


def _git_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "eval",
            "GIT_AUTHOR_EMAIL": "eval@harness",
            "GIT_COMMITTER_NAME": "eval",
            "GIT_COMMITTER_EMAIL": "eval@harness",
        }
    )
    return env


def run_fixture(
    fixture: FixtureMeta,
    *,
    provider: str,
    model: str,
    harness_bin: str | None = None,
    agent_timeout: int = 300,
    test_timeout: int = 60,
) -> RunOutcome:
    """Run one fixture end-to-end in an isolated temp directory."""
    harness_cmd = harness_bin or shutil.which("harness") or "harness"

    with tempfile.TemporaryDirectory(prefix="harness_eval_") as tmp_str:
        work = Path(tmp_str) / fixture.name
        shutil.copytree(fixture.path, work)

        git_env = _git_env()

        # Create a clean git baseline so git diff HEAD captures only agent changes.
        subprocess.run(
            ["git", "-c", "init.defaultBranch=main", "init"],
            cwd=work,
            env=git_env,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "add", "-A"],
            cwd=work,
            env=git_env,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "-c", "commit.gpgsign=false", "commit", "-m", "initial", "--no-verify"],
            cwd=work,
            env=git_env,
            capture_output=True,
            check=True,
        )

        # Run the agent.
        agent_result = subprocess.run(
            [
                harness_cmd,
                "run",
                fixture.task_text.strip(),
                "--cwd",
                str(work),
                "--yes",
                "--verify",
                "none",
                "--in-memory",
                "--provider",
                provider,
                "--model",
                model,
            ],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=agent_timeout,
            env=os.environ.copy(),
        )
        transcript = agent_result.stdout + agent_result.stderr

        # Capture what the agent changed.
        diff_result = subprocess.run(
            ["git", "diff", "HEAD"],
            cwd=work,
            capture_output=True,
            text=True,
        )
        git_diff = diff_result.stdout

        # Run the test suite to check correctness.
        test_result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "--no-header"],
            cwd=work,
            capture_output=True,
            text=True,
            timeout=test_timeout,
            env=os.environ.copy(),
        )
        test_output = test_result.stdout + test_result.stderr

        return RunOutcome(
            fixture=fixture,
            transcript=transcript,
            git_diff=git_diff,
            test_output=test_output,
            agent_exit_code=agent_result.returncode,
            test_exit_code=test_result.returncode,
        )
