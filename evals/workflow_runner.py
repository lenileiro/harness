"""Eval runner for local feature-workflow fixtures.

These fixtures exercise deterministic CLI workflows for the research/autonomy
stack without depending on external model providers.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class WorkflowFileContainsExpectation:
    path: str
    substring: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorkflowStep:
    name: str
    argv: tuple[str, ...]
    append_cwd: bool = True
    stdin_text: str | None = None
    stdout_contains: tuple[str, ...] = ()
    files_exist: tuple[str, ...] = ()
    file_contains: tuple[WorkflowFileContainsExpectation, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "append_cwd": self.append_cwd,
            "stdin_text": self.stdin_text,
            "stdout_contains": list(self.stdout_contains),
            "files_exist": list(self.files_exist),
            "file_contains": [item.to_dict() for item in self.file_contains],
        }


@dataclass(slots=True)
class WorkflowFixtureMeta:
    name: str
    path: Path
    description: str
    steps: tuple[WorkflowStep, ...]
    workspace_dir: Path | None = None
    harness_bin: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": str(self.path),
            "description": self.description,
            "steps": [step.to_dict() for step in self.steps],
            "workspace_dir": str(self.workspace_dir) if self.workspace_dir else None,
            "harness_bin": self.harness_bin,
        }


@dataclass(slots=True)
class WorkflowStepResult:
    name: str
    argv: tuple[str, ...]
    passed: bool
    returncode: int
    stdout: str
    stderr: str
    failures: tuple[str, ...] = ()
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "argv": list(self.argv),
            "passed": self.passed,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "failures": list(self.failures),
            "duration_seconds": self.duration_seconds,
        }


@dataclass(slots=True)
class WorkflowEvalResult:
    fixture_name: str
    passed: bool
    steps_total: int
    steps_passed: int
    step_results: list[WorkflowStepResult] = field(default_factory=list)
    artifact_dir: Path | None = None
    duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "fixture_name": self.fixture_name,
            "passed": self.passed,
            "steps_total": self.steps_total,
            "steps_passed": self.steps_passed,
            "step_results": [result.to_dict() for result in self.step_results],
            "artifact_dir": str(self.artifact_dir) if self.artifact_dir else None,
            "duration_seconds": self.duration_seconds,
        }


@dataclass(slots=True)
class WorkflowEvalReport:
    run_id: str
    artifact_root: Path
    results: list[WorkflowEvalResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "artifact_root": str(self.artifact_root),
            "results": [result.to_dict() for result in self.results],
        }


def find_workflow_fixtures_root(evals_root: Path | None = None) -> Path:
    root = evals_root or Path.cwd().resolve() / "evals"
    return root / "workflow-fixtures"


def _parse_step(payload: dict[str, Any]) -> WorkflowStep:
    return WorkflowStep(
        name=str(payload.get("name") or "").strip(),
        argv=tuple(str(item) for item in (payload.get("argv") or []) if str(item).strip()),
        append_cwd=bool(payload.get("append_cwd", True)),
        stdin_text=(
            str(payload.get("stdin_text")) if payload.get("stdin_text") is not None else None
        ),
        stdout_contains=tuple(
            str(item) for item in (payload.get("stdout_contains") or []) if str(item).strip()
        ),
        files_exist=tuple(
            str(item) for item in (payload.get("files_exist") or []) if str(item).strip()
        ),
        file_contains=tuple(
            WorkflowFileContainsExpectation(
                path=str(item.get("path") or "").strip(),
                substring=str(item.get("substring") or "").strip(),
            )
            for item in (payload.get("file_contains") or [])
            if isinstance(item, dict)
            and str(item.get("path") or "").strip()
            and str(item.get("substring") or "").strip()
        ),
    )


def discover_workflow_fixtures(evals_root: Path | None = None) -> list[WorkflowFixtureMeta]:
    fixtures_root = find_workflow_fixtures_root(evals_root)
    if not fixtures_root.exists():
        return []
    fixtures: list[WorkflowFixtureMeta] = []
    for entry in sorted(fixtures_root.iterdir()):
        if not entry.is_dir():
            continue
        fixture_path = entry / "fixture.json"
        if not fixture_path.exists():
            continue
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        steps = tuple(
            _parse_step(item)
            for item in (payload.get("steps") or [])
            if isinstance(item, dict) and str(item.get("name") or "").strip()
        )
        workspace_dir = entry / "workspace"
        fixtures.append(
            WorkflowFixtureMeta(
                name=entry.name,
                path=entry,
                description=str(payload.get("description") or "").strip(),
                steps=steps,
                workspace_dir=workspace_dir if workspace_dir.is_dir() else None,
                harness_bin=(str(payload.get("harness_bin") or "").strip() or None),
            )
        )
    return fixtures


def _copy_tree(src: Path, dest: Path) -> None:
    for child in src.iterdir():
        target = dest / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)


def _latest_dir_name(path: Path) -> str | None:
    if not path.exists():
        return None
    candidates = [item for item in path.iterdir() if item.is_dir()]
    if not candidates:
        return None
    latest = max(candidates, key=lambda item: item.stat().st_mtime)
    return latest.name


def _sorted_dir_names(path: Path) -> list[str]:
    if not path.exists():
        return []
    candidates = [item for item in path.iterdir() if item.is_dir()]
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return [item.name for item in candidates]


def _context_for_workdir(work: Path) -> dict[str, str]:
    root = work / ".harness" / "research"
    mission_root = work / ".harness" / "missions"
    mapping = {
        "latest_theme_id": root / "themes",
        "latest_unknown_id": root / "unknowns",
        "latest_inspiration_id": root / "inspiration",
        "latest_rabbit_hole_id": root / "rabbitholes",
        "latest_publication_id": root / "publications",
        "latest_citation_id": root / "citations",
        "latest_section_map_id": root / "section-maps",
        "latest_observation_id": root / "observations",
        "latest_opportunity_id": root / "opportunities",
        "latest_hypothesis_id": root / "hypotheses",
        "latest_experiment_plan_id": root / "experiment-plans",
        "latest_experiment_id": root / "experiments",
        "latest_promotion_candidate_id": root / "promotions",
        "latest_archive_id": root / "archive",
        "latest_mission_id": mission_root / "missions",
        "latest_mission_milestone_id": mission_root / "milestones",
        "latest_mission_feature_id": mission_root / "features",
        "latest_mission_contract_id": mission_root / "contracts",
        "latest_mission_handoff_id": mission_root / "handoffs",
        "latest_mission_finding_id": mission_root / "findings",
        "latest_mission_run_id": mission_root / "runs",
    }
    context: dict[str, str] = {}
    for key, path in mapping.items():
        latest = _latest_dir_name(path)
        if latest:
            context[key] = latest
    publication_ids = _sorted_dir_names(root / "publications")
    if len(publication_ids) > 1:
        context["previous_publication_id"] = publication_ids[1]
    experiment_ids = _sorted_dir_names(root / "experiments")
    if len(experiment_ids) > 1:
        context["previous_experiment_id"] = experiment_ids[1]
    mission_feature_ids = _sorted_dir_names(mission_root / "features")
    if len(mission_feature_ids) > 1:
        context["previous_mission_feature_id"] = mission_feature_ids[1]
    return context


def _resolve_placeholders(value: str, context: dict[str, str]) -> str:
    resolved = value
    for key, item in context.items():
        resolved = resolved.replace("{" + key + "}", item)
    return resolved


def _build_command(
    *,
    harness_bin: str | None,
    step: WorkflowStep,
    work: Path,
    context: dict[str, str],
) -> tuple[str, ...]:
    harness_cmd = harness_bin or shutil.which("harness") or "harness"
    resolved_argv = tuple(_resolve_placeholders(item, context) for item in step.argv)
    if step.append_cwd and "--cwd" not in resolved_argv:
        resolved_argv = (*resolved_argv, "--cwd", str(work))
    return (harness_cmd, *resolved_argv)


def run_workflow_fixture(
    fixture: WorkflowFixtureMeta,
    *,
    harness_bin: str | None = None,
    artifact_dir: Path | None = None,
    timeout: int = 180,
) -> WorkflowEvalResult:
    with tempfile.TemporaryDirectory(prefix="harness_workflow_eval_") as tmp_str:
        work = Path(tmp_str) / "workspace"
        work.mkdir(parents=True, exist_ok=True)
        if fixture.workspace_dir is not None:
            _copy_tree(fixture.workspace_dir, work)
        effective_harness_bin = harness_bin
        if fixture.harness_bin:
            effective_harness_bin = str((work / fixture.harness_bin).resolve())

        started_fixture = time.perf_counter()
        step_results: list[WorkflowStepResult] = []
        passed_steps = 0

        for index, step in enumerate(fixture.steps, start=1):
            context = _context_for_workdir(work)
            cmd = _build_command(
                harness_bin=effective_harness_bin,
                step=step,
                work=work,
                context=context,
            )
            started_step = time.perf_counter()
            proc = subprocess.run(
                list(cmd),
                cwd=work,
                capture_output=True,
                text=True,
                input=step.stdin_text,
                timeout=timeout,
                check=False,
            )
            duration = time.perf_counter() - started_step
            combined = f"{proc.stdout}\n{proc.stderr}"
            failures: list[str] = []
            if proc.returncode != 0:
                failures.append(f"returncode={proc.returncode}")
            post_context = _context_for_workdir(work)
            for snippet in step.stdout_contains:
                resolved_snippet = _resolve_placeholders(snippet, post_context)
                if resolved_snippet not in combined:
                    failures.append(f"stdout~{resolved_snippet}")
            for rel_path in step.files_exist:
                resolved_path = _resolve_placeholders(rel_path, post_context)
                if not (work / resolved_path).exists():
                    failures.append(f"missing_file~{resolved_path}")
            for item in step.file_contains:
                resolved_path = _resolve_placeholders(item.path, post_context)
                file_path = work / resolved_path
                if not file_path.exists():
                    failures.append(f"missing_file~{resolved_path}")
                    continue
                contents = file_path.read_text(encoding="utf-8")
                resolved_substring = _resolve_placeholders(item.substring, post_context)
                if resolved_substring not in contents:
                    failures.append(f"file_contains~{resolved_path}~{resolved_substring}")

            step_result = WorkflowStepResult(
                name=step.name,
                argv=cmd,
                passed=not failures,
                returncode=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                failures=tuple(failures),
                duration_seconds=duration,
            )
            step_results.append(step_result)
            if artifact_dir is not None:
                artifact_dir.mkdir(parents=True, exist_ok=True)
                stem = f"{index:02d}-{step.name}"
                (artifact_dir / f"{stem}.stdout.txt").write_text(proc.stdout, encoding="utf-8")
                (artifact_dir / f"{stem}.stderr.txt").write_text(proc.stderr, encoding="utf-8")
                (artifact_dir / f"{stem}.json").write_text(
                    json.dumps(step_result.to_dict(), indent=2),
                    encoding="utf-8",
                )
            if failures:
                break
            passed_steps += 1

        duration = time.perf_counter() - started_fixture
        result = WorkflowEvalResult(
            fixture_name=fixture.name,
            passed=passed_steps == len(fixture.steps),
            steps_total=len(fixture.steps),
            steps_passed=passed_steps,
            step_results=step_results,
            artifact_dir=artifact_dir,
            duration_seconds=duration,
        )
        if artifact_dir is not None:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "result.json").write_text(
                json.dumps(result.to_dict(), indent=2),
                encoding="utf-8",
            )
        return result


__all__ = [
    "WorkflowEvalReport",
    "WorkflowEvalResult",
    "WorkflowFixtureMeta",
    "WorkflowStep",
    "WorkflowStepResult",
    "discover_workflow_fixtures",
    "find_workflow_fixtures_root",
    "run_workflow_fixture",
]
