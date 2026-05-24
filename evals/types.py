"""Shared data models for harness eval runs."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _serialize(val) for key, val in value.items()}
    if isinstance(value, list | tuple):
        return [_serialize(item) for item in value]
    return value


@dataclass(slots=True)
class FixtureRules:
    behavior_category: str = ""
    primary_dimension: str = ""
    expected_first_step: str = ""
    allowed_paths: list[str] = field(default_factory=list)
    disallowed_paths: list[str] = field(default_factory=list)
    required_verification: str = ""
    trap: str = ""
    correct_fix: str = ""
    dimensions: list[str] = field(default_factory=list)
    scoring_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(slots=True)
class FixtureMeta:
    name: str
    path: Path
    task_text: str
    eval_md: str
    verify_command: str = "pytest tests/ -v --tb=short --no-header"
    phases: list[str] | None = None
    family: str = ""
    holdout: bool = False
    mutated_from: str | None = None
    metadata_path: Path | None = None
    rules: FixtureRules = field(default_factory=FixtureRules)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(slots=True)
class TraceEvent:
    kind: str
    order: int
    message: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(slots=True)
class HardMetrics:
    verify_passed: bool
    run_exit_code: int
    verify_exit_code: int
    files_touched: int
    lines_added: int
    lines_deleted: int
    tool_calls: int
    shell_commands: int
    did_run_verification: bool
    agent_duration_seconds: float
    verify_duration_seconds: float
    total_duration_seconds: float
    time_to_first_verification_seconds: float | None = None
    edit_before_repro: bool = False
    premature_completion: bool = False
    redundant_tool_calls: int = 0
    retry_loops: int = 0
    verification_after_failure: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(slots=True)
class HarnessAdjustment:
    id: str
    kind: str
    text: str
    triggers: list[str] = field(default_factory=list)
    weight: float = 1.0
    source_fixture_name: str = ""
    source_variant: str = ""
    source_artifact_dir: Path | None = None
    rationale: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(slots=True)
class RunOutcome:
    fixture: FixtureMeta
    transcript: str
    git_diff: str
    test_output: str
    agent_exit_code: int
    test_exit_code: int
    variant: str = "defended"
    hard_metrics: HardMetrics | None = None
    trace_events: list[TraceEvent] = field(default_factory=list)
    artifact_dir: Path | None = None
    agent_command: list[str] = field(default_factory=list)
    verify_command: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(slots=True)
class DimensionScore:
    score: int
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(slots=True)
class EvalResult:
    fixture_name: str
    verification: DimensionScore
    scope: DimensionScore
    decomposition: DimensionScore
    correctness: DimensionScore
    pushback: DimensionScore
    epistemic: DimensionScore
    overall: DimensionScore
    hard_metrics: HardMetrics | None = None
    artifact_dir: Path | None = None
    variant: str = "defended"
    run_index: int = 1

    @property
    def passed(self) -> bool:
        return self.overall.score >= 3 and self.correctness.score >= 3

    def to_dict(self) -> dict[str, Any]:
        data = _serialize(self)
        data["passed"] = self.passed
        return data


@dataclass(slots=True)
class AggregatedDimension:
    median: float
    minimum: int
    maximum: int
    mean: float
    sem: float | None

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)


@dataclass(slots=True)
class FixtureAggregate:
    fixture_name: str
    variant: str
    runs: int
    passes: int
    dimensions: dict[str, AggregatedDimension]
    hard_metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        data = _serialize(self)
        data["pass_rate"] = self.passes / self.runs if self.runs else 0.0
        return data


@dataclass(slots=True)
class EvalReport:
    run_id: str
    provider: str
    model: str
    judge_provider: str
    judge_model: str
    fixture_set: str
    n_runs: int
    ab: bool
    artifact_root: Path
    benchmark_mode: str = "original"
    mutation_coverage: float | None = None
    results: list[EvalResult] = field(default_factory=list)
    aggregates: list[FixtureAggregate] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(self)
