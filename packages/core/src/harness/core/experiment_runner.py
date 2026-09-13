from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from harness.core.experiment_plans import ExperimentPlan
from harness.core.experiments import CommandResult, Experiment, ExperimentResult, ExperimentStatus
from harness.core.research_store import ResearchStore
from harness.core.workspace_snapshot import workspace_fingerprint

_EVAL_SLICE_COMMANDS = {
    "docs-smoke": "uv run harness eval docs-audit --suite docs-smoke",
    "research-smoke": "uv run harness eval research --suite research-smoke",
    "review-smoke": "uv run harness eval review --suite review-smoke",
    "workflow-smoke": "uv run harness eval workflow --suite workflow-smoke --timeout 120",
}


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _current_branch(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def _eval_slice_command(slice_or_command: str) -> str:
    normalized = slice_or_command.strip()
    key = normalized.removesuffix(".txt")
    return _EVAL_SLICE_COMMANDS.get(key, normalized)


def _stop_command(process: subprocess.Popen[str]) -> None:
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
    except (ProcessLookupError, PermissionError):
        # Darwin reports EPERM, not ESRCH, for an already-reaped group.
        pass


def run_experiment_plan(
    *,
    store: ResearchStore,
    plan: ExperimentPlan,
    cwd: Path,
    created_by: str = "human",
    timeout: float = 600,
) -> tuple[Experiment, ExperimentResult]:
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("experiment timeout must be positive")
    if any(
        direction not in {"minimize", "maximize"} for direction in plan.metric_directions.values()
    ):
        raise ValueError("metric directions must be minimize or maximize")
    started_at = _utcnow()
    experiment = Experiment(
        id=store.new_id("exp", plan.id),
        plan_id=plan.id,
        branch=_current_branch(cwd),
        worktree=str(cwd),
        created_by=created_by,
    )
    artifact_dir = store.root / "experiments" / experiment.id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    commands: list[tuple[str, str]] = []
    commands.extend(("check", command) for command in plan.checks)
    commands.extend(("eval", _eval_slice_command(command)) for command in plan.eval_slices)
    if plan.measurement_command:
        commands.append(("measurement", plan.measurement_command))

    command_results: list[CommandResult] = []
    interruption: BaseException | None = None
    status: ExperimentStatus = "passed" if commands else "inconclusive"
    started_perf = time.perf_counter()
    initial_fingerprint = workspace_fingerprint(cwd)
    metrics: dict[str, float] = {}
    store.add_experiment(
        experiment,
        ExperimentResult(
            experiment_id=experiment.id,
            status="running",
            command_results=(),
            started_at=started_at,
            finished_at="",
            duration_seconds=0,
            artifact_dir=str(artifact_dir),
        ),
    )
    for index, (kind, command) in enumerate(commands, start=1):
        command_started = time.perf_counter()
        timed_out = False
        try:
            with subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=os.name == "posix",
            ) as process:
                try:
                    stdout, stderr = process.communicate(timeout=timeout)
                    exit_code = process.returncode
                except subprocess.TimeoutExpired:
                    _stop_command(process)
                    stdout, stderr = process.communicate()
                    stderr += f"\nExperiment command timed out after {timeout}s.\n"
                    exit_code = 124
                    timed_out = True
                except BaseException as exc:
                    _stop_command(process)
                    stdout, stderr = process.communicate()
                    stderr += f"\nExperiment command interrupted: {type(exc).__name__}.\n"
                    exit_code = 130
                    interruption = exc
                finally:
                    # Background descendants must not outlive the evidence they
                    # produced, even when their parent exits successfully.
                    _stop_command(process)
        except OSError as exc:
            stdout, stderr, exit_code = "", str(exc), 127
        if kind == "measurement" and exit_code == 0:
            try:
                payload = json.loads(stdout)
                if not isinstance(payload, dict) or not payload:
                    raise ValueError("measurement output must be a nonempty JSON object")
                for name, value in payload.items():
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                    ):
                        raise ValueError(f"measurement {name!r} must be a finite number")
                metrics = {name: float(value) for name, value in payload.items()}
                if set(plan.metric_directions) - set(metrics):
                    raise ValueError("measurement output omits a declared metric")
            except (ValueError, TypeError) as exc:
                exit_code = 1
                stderr += f"\nInvalid measurements: {exc}\n"
        duration = time.perf_counter() - command_started
        stdout_path = artifact_dir / f"{index:02d}-{kind}.stdout.txt"
        stderr_path = artifact_dir / f"{index:02d}-{kind}.stderr.txt"
        stdout_path.write_text(stdout, encoding="utf-8")
        stderr_path.write_text(stderr, encoding="utf-8")
        if exit_code != 0:
            status = (
                "interrupted"
                if interruption is not None
                else "timed_out"
                if timed_out
                else "failed"
            )
        command_results.append(
            CommandResult(
                kind=kind,  # type: ignore[arg-type]
                command=command,
                exit_code=exit_code,
                duration_seconds=duration,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )
        )
        if timed_out or interruption is not None:
            break
    finished_at = _utcnow()
    final_fingerprint = workspace_fingerprint(cwd)
    # A successful command may generate files. Keep its execution outcome, but
    # do not certify a workspace fingerprint when its inputs changed mid-run.
    result = ExperimentResult(
        experiment_id=experiment.id,
        status=status,
        command_results=tuple(command_results),
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=time.perf_counter() - started_perf,
        artifact_dir=str(artifact_dir),
        workspace_fingerprint=(
            final_fingerprint
            if initial_fingerprint == final_fingerprint and interruption is None
            else ""
        ),
        metrics=metrics,
        metric_directions=plan.metric_directions,
        baseline_experiment_id=plan.baseline_experiment_id,
    )
    store.add_experiment(experiment, result)
    if interruption is not None:
        raise interruption
    return experiment, result


def compare_experiment_results(
    left: ExperimentResult,
    right: ExperimentResult,
) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for name in sorted(left.metrics.keys() | right.metrics.keys()):
        before, after = left.metrics.get(name), right.metrics.get(name)
        direction = right.metric_directions.get(name) or left.metric_directions.get(name)
        delta = after - before if before is not None and after is not None else None
        metrics[name] = {
            "left": before,
            "right": after,
            "delta": delta,
            "direction": direction,
            "improved": (
                (delta < 0 if direction == "minimize" else delta > 0)
                if delta is not None and direction in {"minimize", "maximize"}
                else None
            ),
        }
    return {
        "left_status": left.status,
        "right_status": right.status,
        "left_duration_seconds": left.duration_seconds,
        "right_duration_seconds": right.duration_seconds,
        "left_commands": len(left.command_results),
        "right_commands": len(right.command_results),
        "metrics": metrics,
    }


__all__ = ["compare_experiment_results", "run_experiment_plan"]
