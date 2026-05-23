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
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FixtureMeta:
    name: str
    path: Path
    task_text: str
    eval_md: str  # raw EVAL.md text, passed to judge as context
    verify_command: str = "pytest tests/ -v --tb=short --no-header"


@dataclass
class RunOutcome:
    fixture: FixtureMeta
    transcript: str
    git_diff: str
    test_output: str
    agent_exit_code: int
    test_exit_code: int
    # "defended" = full structural verifier chain + critic active.
    # "bare"     = same model + tools but no structural defenses, no critic.
    # Used by the eval's A/B mode to measure harness value-add.
    variant: str = "defended"


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


def _agent_cmd(
    provider: str,
    model: str,
    task_text: str,
    work: Path,
    harness_bin: str | None,
    verify_command: str | None = None,
    variant: str = "defended",
) -> list[str]:
    """Return the command to invoke the agent for the given provider.

    'claude' provider shells out to `claude -p` (Claude Code CLI).
    All other providers go through `harness run` with ShellVerifier so the
    repair loop fires when the verify_command exits non-zero.

    `variant="bare"` adds the `--bare` flag to disable the structural
    verifier chain and critic — the agent runs with model + tools only.
    Used by A/B mode to measure harness value-add.
    """
    if provider == "claude":
        claude_bin = shutil.which("claude") or "claude"
        return [
            claude_bin,
            "-p",
            task_text.strip(),
            "--allowedTools",
            "Read,Write,Edit,Bash",
        ]
    harness_cmd = harness_bin or shutil.which("harness") or "harness"
    cmd = [
        harness_cmd,
        "run",
        task_text.strip(),
        "--cwd",
        str(work),
        "--yes",
        "--in-memory",
        "--provider",
        provider,
        "--model",
        model,
        # 2 attempts: empirically the right budget for "minimal fix" style
        # tasks. Raising it to 4 regressed fixture 02 because the extra
        # turns gave the agent room to add unwanted "robustness" code,
        # net-net worse on scope dimension. Keep tight.
        "--max-repair",
        "2",
    ]
    if variant == "bare":
        cmd.append("--bare")
    if verify_command:
        # In bare mode, keep ShellVerifier (the test runner) but skip the
        # critic — bare = agent + tools + test signal, nothing else.
        cmd += [
            "--verify",
            "shell",
            "--verify-command",
            verify_command,
        ]
        if variant != "bare":
            critic_mode = "llm+search" if os.environ.get("TAVILY_API_KEY") else "llm"
            cmd += ["--critic", critic_mode]
    else:
        cmd += ["--verify", "rule"]
    return cmd


def run_fixture(
    fixture: FixtureMeta,
    *,
    provider: str,
    model: str,
    harness_bin: str | None = None,
    agent_timeout: int = 300,
    test_timeout: int = 60,
    variant: str = "defended",
) -> RunOutcome:
    """Run one fixture end-to-end in an isolated temp directory."""
    with tempfile.TemporaryDirectory(prefix="harness_eval_") as tmp_str:
        work = Path(tmp_str) / fixture.name
        shutil.copytree(fixture.path, work)

        git_env = _git_env()

        # Create a clean git baseline so git diff HEAD captures only agent changes.
        # Write a .gitignore first so __pycache__ / *.pyc don't pollute the diff.
        (work / ".gitignore").write_text("__pycache__/\n*.pyc\n*.pyo\n")
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
        cmd = _agent_cmd(
            provider,
            model,
            fixture.task_text,
            work,
            harness_bin,
            verify_command=fixture.verify_command,
            variant=variant,
        )
        agent_result = subprocess.run(
            cmd,
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

        # Run the fixture's own verify_command to check correctness. We
        # invoke whatever shell command the fixture declared (pytest, go
        # test, cargo, jest, ...) rather than hardcoding pytest here — the
        # harness eval framework is language-agnostic.
        test_result = subprocess.run(
            fixture.verify_command,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=test_timeout,
            env=os.environ.copy(),
            shell=True,
        )
        test_output = test_result.stdout + test_result.stderr

        return RunOutcome(
            fixture=fixture,
            transcript=transcript,
            git_diff=git_diff,
            test_output=test_output,
            agent_exit_code=agent_result.returncode,
            test_exit_code=test_result.returncode,
            variant=variant,
        )
