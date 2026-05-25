from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from harness.core.experiment_plans import ExperimentPlan
from harness.core.experiments import CommandResult, Experiment, ExperimentResult
from harness.core.research_store import ResearchStore


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _current_branch(cwd: Path) -> str:
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def run_experiment_plan(
    *,
    store: ResearchStore,
    plan: ExperimentPlan,
    cwd: Path,
    created_by: str = "human",
    timeout: int = 600,
) -> tuple[Experiment, ExperimentResult]:
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
    commands.extend(("eval", command) for command in plan.eval_slices)

    command_results: list[CommandResult] = []
    overall_ok = True
    started_perf = time.perf_counter()
    for index, (kind, command) in enumerate(commands, start=1):
        command_started = time.perf_counter()
        completed = subprocess.run(
            command,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        duration = time.perf_counter() - command_started
        stdout_path = artifact_dir / f"{index:02d}-{kind}.stdout.txt"
        stderr_path = artifact_dir / f"{index:02d}-{kind}.stderr.txt"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            overall_ok = False
        command_results.append(
            CommandResult(
                kind=kind,  # type: ignore[arg-type]
                command=command,
                exit_code=completed.returncode,
                duration_seconds=duration,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
            )
        )
    finished_at = _utcnow()
    result = ExperimentResult(
        experiment_id=experiment.id,
        status="passed" if overall_ok else "failed",
        command_results=tuple(command_results),
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=time.perf_counter() - started_perf,
        artifact_dir=str(artifact_dir),
    )
    (artifact_dir / "experiment.json").write_text(
        json.dumps(experiment.to_dict(), indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "result.json").write_text(
        json.dumps(result.to_dict(), indent=2),
        encoding="utf-8",
    )
    return experiment, result


def compare_experiment_results(
    left: ExperimentResult,
    right: ExperimentResult,
) -> dict[str, object]:
    return {
        "left_status": left.status,
        "right_status": right.status,
        "left_duration_seconds": left.duration_seconds,
        "right_duration_seconds": right.duration_seconds,
        "left_commands": len(left.command_results),
        "right_commands": len(right.command_results),
    }


__all__ = ["compare_experiment_results", "run_experiment_plan"]
