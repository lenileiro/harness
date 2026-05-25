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
import time
from pathlib import Path

from evals import discovery as _discovery
from evals import hard_checks as _hard_checks
from evals.artifacts import (
    build_trace_events as _build_trace_events,
)
from evals.artifacts import (
    compute_hard_metrics as _compute_hard_metrics,
)
from evals.artifacts import (
    persist_artifacts as _persist_artifacts,
)
from evals.failure_analyzer import persist_harness_adjustments
from evals.types import FixtureMeta, RunOutcome

discover_fixtures = _discovery.discover_fixtures
_find_evals_root = _discovery.find_evals_root
_behavioral_hard_check = _hard_checks.behavioral_hard_check
_check_reproduce_before_repair_scope = _hard_checks.check_reproduce_before_repair_scope
_check_scope_discipline_minimal_fix = _hard_checks.check_scope_discipline_minimal_fix
_check_scope_discipline_with_regression_test = (
    _hard_checks.check_scope_discipline_with_regression_test
)
_check_sustained_coherence_scope = _hard_checks.check_sustained_coherence_scope
_check_wrong_diagnosis_scope = _hard_checks.check_wrong_diagnosis_scope


def _eval_env(*, work: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["HARNESS_EVAL_PROJECT_ROOT"] = str(Path(__file__).resolve().parents[1])
    env["HARNESS_EVAL_WORKSPACE"] = str(work)
    return env


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
    phases: list[str] | None = None,
    behavior_category: str | None = None,
    max_output_tokens: int | None = None,
) -> list[str]:
    """Return the command to invoke the agent for the given provider.

    'claude' provider shells out to `claude -p` (Claude Code CLI).
    All other providers go through `harness run` with ShellVerifier so the
    repair loop fires when the verify_command exits non-zero.

    `variant="bare"` uses `--profile bare` to disable the structural
    verifier chain and critic — the agent runs with model + tools only.
    The defended arm now uses `--profile adaptive`, which picks between
    minimal and strict from the task shape. Used by A/B mode to measure
    harness value-add.
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
    # "defended" → adaptive profile (current best harness path).
    # "bare"     → no chain, no critic.
    # Eval keeps using defended/bare labels so historical tables remain
    # comparable even as the defended implementation improves.
    cmd += ["--profile", "adaptive" if variant == "defended" else "bare"]
    if max_output_tokens is not None:
        cmd += ["--max-output-tokens", str(max_output_tokens)]
    if verify_command:
        # In bare mode, keep ShellVerifier (the test runner) but skip the
        # critic — bare = agent + tools + test signal, nothing else.
        cmd += [
            "--verify",
            "shell",
            "--verify-command",
            verify_command,
        ]
        # Critic helps on diagnosis-heavy decomposition traps (fixture 03), but
        # on strict minimal-fix tasks it can create extra repair-loop churn and
        # overthinking. Keep the defended arm lighter on pure scope /
        # verification fixtures.
        if variant != "bare" and (behavior_category or "").strip().lower() in {
            "decomposition",
            "diagnosis",
        }:
            critic_mode = "llm+search" if os.environ.get("TAVILY_API_KEY") else "llm"
            cmd += ["--critic", critic_mode]
    else:
        cmd += ["--verify", "rule"]
    # Phase tracking is useful for some live tasks, but in this benchmark it
    # adds substantial finish-line latency on the long sustained-coherence
    # fixture without improving judged outcomes. Since the eval objective here
    # is pass-rate-first A/B comparison, keep defended lighter on pure scope /
    # verification families and reserve explicit phase gating for tasks whose
    # primary challenge is multi-stage decomposition.
    if (
        phases
        and variant != "bare"
        and (behavior_category or "").strip().lower() not in {"scope", "verification"}
    ):
        cmd += ["--phases", ",".join(phases)]
    return cmd


def _copy_fixture_for_run(src: Path, dest: Path) -> None:
    """Copy a fixture into an isolated worktree without leaking judge metadata.

    `EVAL.md` and `fixture.yaml` are benchmark-side artifacts. The runner
    provides their contents to the judge and discovery layer directly, so the
    agent should not be able to read them from the workspace during execution.
    """
    shutil.copytree(
        src,
        dest,
        ignore=shutil.ignore_patterns("EVAL.md", "fixture.yaml", "__pycache__", "*.pyc", "*.pyo"),
    )


def run_fixture(
    fixture: FixtureMeta,
    *,
    provider: str,
    model: str,
    harness_bin: str | None = None,
    agent_timeout: int = 300,
    test_timeout: int = 60,
    variant: str = "defended",
    artifact_dir: Path | None = None,
    max_output_tokens: int | None = None,
) -> RunOutcome:
    """Run one fixture end-to-end in an isolated temp directory."""
    with tempfile.TemporaryDirectory(prefix="harness_eval_") as tmp_str:
        work = Path(tmp_str) / fixture.name
        _copy_fixture_for_run(fixture.path, work)

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
            phases=fixture.phases,
            behavior_category=fixture.rules.behavior_category or fixture.family,
            max_output_tokens=max_output_tokens,
        )
        agent_started = time.perf_counter()
        agent_env = _eval_env(work=work)
        experience_root = (_find_evals_root() / "runs").resolve()
        existing_roots = [
            raw.strip()
            for raw in agent_env.get("HARNESS_EXPERIENCE_ROOTS", "").split(os.pathsep)
            if raw.strip()
        ]
        if str(experience_root) not in existing_roots:
            existing_roots.append(str(experience_root))
        agent_env["HARNESS_EXPERIENCE_ROOTS"] = os.pathsep.join(existing_roots)
        agent_result = subprocess.run(
            cmd,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=agent_timeout,
            env=agent_env,
        )
        agent_duration = time.perf_counter() - agent_started
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
        verify_started = time.perf_counter()
        verify_env = _eval_env(work=work)
        test_result = subprocess.run(
            fixture.verify_command,
            cwd=work,
            capture_output=True,
            text=True,
            timeout=test_timeout,
            env=verify_env,
            shell=True,
        )
        verify_duration = time.perf_counter() - verify_started
        test_output = test_result.stdout + test_result.stderr
        combined_verify_exit_code = test_result.returncode
        if test_result.returncode == 0:
            behavioral_ok, behavioral_message = _behavioral_hard_check(fixture, work)
            if behavioral_message:
                separator = "\n" if test_output.endswith("\n") or not test_output else "\n\n"
                test_output = f"{test_output}{separator}{behavioral_message}\n"
            if not behavioral_ok:
                combined_verify_exit_code = 1
        trace_events = _build_trace_events(
            transcript,
            fixture.verify_command,
            agent_exit_code=agent_result.returncode,
            verify_exit_code=combined_verify_exit_code,
        )
        hard_metrics = _compute_hard_metrics(
            transcript,
            git_diff,
            fixture.verify_command,
            run_exit_code=agent_result.returncode,
            verify_exit_code=combined_verify_exit_code,
            agent_duration_seconds=agent_duration,
            verify_duration_seconds=verify_duration,
        )
        outcome = RunOutcome(
            fixture=fixture,
            transcript=transcript,
            git_diff=git_diff,
            test_output=test_output,
            agent_exit_code=agent_result.returncode,
            test_exit_code=combined_verify_exit_code,
            variant=variant,
            hard_metrics=hard_metrics,
            trace_events=trace_events,
            artifact_dir=artifact_dir,
            agent_command=cmd,
            verify_command=fixture.verify_command,
        )
        if artifact_dir is not None:
            _persist_artifacts(artifact_dir, outcome)
            persist_harness_adjustments(artifact_dir)
        return outcome
