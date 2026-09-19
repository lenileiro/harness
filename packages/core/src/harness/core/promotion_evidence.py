"""Require current execution evidence before publishing research changes."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from harness.core.experiment_plans import ExperimentPlan
from harness.core.experiment_runner import _eval_slice_command
from harness.core.experiments import Experiment, ExperimentResult
from harness.core.promotion_candidates import PromotionCandidate
from harness.core.research.store import ResearchStore
from harness.core.workspace_snapshot import workspace_fingerprint


class PromotionEvidenceError(ValueError):
    """A candidate has no complete, current evidence for the requested mutation."""


@dataclass(frozen=True, slots=True)
class PromotionEvidence:
    experiment_ids: tuple[str, ...]
    workspace_fingerprint: str
    metric_improvements: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_ids": list(self.experiment_ids),
            "workspace_fingerprint": self.workspace_fingerprint,
            "metric_improvements": list(self.metric_improvements),
        }


def _targets(paths: tuple[str, ...], *, cwd: Path) -> tuple[Path, ...]:
    normalized = []
    for name in paths:
        path = Path(name)
        if not name.strip() or path.is_absolute() or ".harness" in path.parts:
            raise PromotionEvidenceError(
                f"invalid evidence target {name!r}; use workspace-relative source paths"
            )
        target = (cwd / path).resolve()
        if (
            target == cwd
            or not target.is_relative_to(cwd)
            or ".harness" in target.relative_to(cwd).parts
        ):
            raise PromotionEvidenceError(
                f"target {name!r} must stay inside a declared workspace subpath"
            )
        normalized.append(target)
    return tuple(normalized)


def _covers(target: Path, scopes: tuple[Path, ...]) -> bool:
    return any(target.is_relative_to(scope) for scope in scopes)


def _passed_commands(result: ExperimentResult, *, label: str) -> None:
    if result.status != "passed":
        raise PromotionEvidenceError(
            f"{label} is {result.status}; rerun its plan successfully before promotion"
        )
    if not result.command_results or any(
        not command.command.strip() or command.exit_code != 0 for command in result.command_results
    ):
        raise PromotionEvidenceError(
            f"{label} has no complete passed command evidence; run nonempty checks or measurements"
        )
    if not result.workspace_fingerprint:
        raise PromotionEvidenceError(
            f"{label} has no certified workspace fingerprint; rerun its plan without changing inputs"
        )


def _planned_commands(plan: ExperimentPlan, result: ExperimentResult) -> None:
    required = Counter(("check", command.strip()) for command in plan.checks)
    required.update(("eval", _eval_slice_command(command)) for command in plan.eval_slices)
    if plan.measurement_command:
        required.update([("measurement", plan.measurement_command.strip())])
    executed = Counter(
        (command.kind, command.command.strip()) for command in result.command_results
    )
    if not required or required - executed:
        raise PromotionEvidenceError(
            f"experiment {result.experiment_id} does not contain all commands from plan {plan.id}; rerun the current plan"
        )


def _finite_metrics(result: ExperimentResult, *, label: str) -> None:
    for name, value in result.metrics.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise PromotionEvidenceError(f"{label} metric {name!r} must be a finite number")


def _compare_metrics(
    *, plan: ExperimentPlan, result: ExperimentResult, store: ResearchStore
) -> tuple[str, ...]:
    if (
        result.metric_directions != plan.metric_directions
        or result.baseline_experiment_id != plan.baseline_experiment_id
    ):
        raise PromotionEvidenceError(
            f"experiment {result.experiment_id} has stale metric goals or baseline; rerun plan {plan.id}"
        )
    _finite_metrics(result, label=f"experiment {result.experiment_id}")
    if not plan.metric_directions and not plan.baseline_experiment_id:
        return ()
    if not plan.metric_directions:
        raise PromotionEvidenceError(
            f"plan {plan.id} must declare minimize/maximize metric goals for its baseline"
        )
    if not plan.baseline_experiment_id:
        raise PromotionEvidenceError(
            f"plan {plan.id} requires a baseline experiment for its metric goals"
        )
    if not plan.measurement_command:
        raise PromotionEvidenceError(
            f"plan {plan.id} requires a measurement command for its metric goals"
        )
    if plan.baseline_experiment_id == result.experiment_id:
        raise PromotionEvidenceError("an experiment cannot serve as its own baseline")
    baseline = store.load_experiment_result(plan.baseline_experiment_id)
    if baseline.experiment_id != plan.baseline_experiment_id:
        raise PromotionEvidenceError(
            "baseline experiment identity does not match its stored result"
        )
    _passed_commands(baseline, label=f"baseline {baseline.experiment_id}")
    _finite_metrics(baseline, label=f"baseline {baseline.experiment_id}")
    if not any(command.kind == "measurement" for command in baseline.command_results):
        raise PromotionEvidenceError(
            f"baseline {baseline.experiment_id} has no measurement command evidence"
        )
    improved = []
    for name, direction in plan.metric_directions.items():
        if direction not in {"minimize", "maximize"}:
            raise PromotionEvidenceError(f"metric {name!r} direction must be minimize or maximize")
        if baseline.metric_directions.get(name, direction) != direction:
            raise PromotionEvidenceError(
                f"baseline direction for metric {name!r} does not match plan {plan.id}"
            )
        if name not in baseline.metrics or name not in result.metrics:
            raise PromotionEvidenceError(
                f"metric {name!r} is missing from the baseline or candidate measurements"
            )
        before, after = baseline.metrics[name], result.metrics[name]
        delta = after - before if direction == "maximize" else before - after
        if delta < 0:
            raise PromotionEvidenceError(
                f"metric {name!r} regressed ({before} -> {after}, {direction}); promotion blocked"
            )
        if delta > 0:
            improved.append(name)
    if not improved:
        raise PromotionEvidenceError(
            "measurements show no improvement over the baseline; promotion blocked"
        )
    return tuple(improved)


def require_promotion_evidence(
    *, candidate: PromotionCandidate, store: ResearchStore, cwd: Path
) -> PromotionEvidence:
    """Validate immediately before commit, push, or PR creation.

    Explicit experiment references are authoritative. Otherwise, each relevant
    plan linked through source_hypotheses uses its newest execution; old passing
    runs cannot override a newer failed run. Multiple plans may cover disjoint
    candidate targets, but every selected execution must pass at the current
    workspace fingerprint. Draft and branch preparation do not need this gate.
    """
    cwd = cwd.resolve()
    try:
        targets = _targets(candidate.target_files, cwd=cwd)
        if not targets:
            raise PromotionEvidenceError("candidate must declare target_files before promotion")
        selected: list[tuple[Experiment, ExperimentPlan]] = []
        if candidate.source_experiments:
            for experiment_id in dict.fromkeys(candidate.source_experiments):
                experiment = store.load_experiment(experiment_id)
                if experiment.id != experiment_id:
                    raise PromotionEvidenceError(
                        f"experiment identity mismatch for {experiment_id!r}"
                    )
                selected.append((experiment, store.load_experiment_plan(experiment.plan_id)))
        elif candidate.source_hypotheses:
            plans = {
                plan.id: plan
                for plan in store.list_experiment_plans()
                if plan.hypothesis_id in candidate.source_hypotheses
                and any(_covers(target, _targets(plan.target_files, cwd=cwd)) for target in targets)
            }
            latest: dict[str, Experiment] = {}
            ambiguous: set[str] = set()
            for experiment in store.list_experiments():
                if experiment.plan_id not in plans:
                    continue
                previous = latest.get(experiment.plan_id)
                if previous is None or experiment.created_at > previous.created_at:
                    latest[experiment.plan_id] = experiment
                    ambiguous.discard(experiment.plan_id)
                elif experiment.created_at == previous.created_at:
                    ambiguous.add(experiment.plan_id)
            if ambiguous:
                raise PromotionEvidenceError(
                    "latest experiment timestamps are ambiguous; link exact source_experiments or rerun the plan"
                )
            selected = [(experiment, plans[plan_id]) for plan_id, experiment in latest.items()]
        if not selected:
            raise PromotionEvidenceError(
                "no linked experiment evidence; run an experiment covering the candidate targets and link it with source_experiments or source_hypotheses"
            )
        fingerprint = workspace_fingerprint(cwd)
        covered: set[Path] = set()
        improved: set[str] = set()
        for experiment, plan in selected:
            if plan.id != experiment.plan_id:
                raise PromotionEvidenceError(
                    f"plan identity mismatch for experiment {experiment.id}"
                )
            scopes = _targets(plan.target_files, cwd=cwd)
            matching = {target for target in targets if _covers(target, scopes)}
            if not matching:
                raise PromotionEvidenceError(
                    f"experiment {experiment.id} plan targets do not cover candidate targets"
                )
            result = store.load_experiment_result(experiment.id)
            if result.experiment_id != experiment.id:
                raise PromotionEvidenceError(
                    f"result identity mismatch for experiment {experiment.id}"
                )
            _passed_commands(result, label=f"experiment {experiment.id}")
            if result.workspace_fingerprint != fingerprint:
                raise PromotionEvidenceError(
                    f"experiment {experiment.id} is stale for the current workspace; rerun plan {plan.id}"
                )
            _planned_commands(plan, result)
            improved.update(_compare_metrics(plan=plan, result=result, store=store))
            covered.update(matching)
        missing = set(targets) - covered
        if missing:
            names = ", ".join(str(path.relative_to(cwd)) for path in sorted(missing))
            raise PromotionEvidenceError(
                f"candidate targets lack matching experiment evidence: {names}"
            )
        return PromotionEvidence(
            experiment_ids=tuple(experiment.id for experiment, _ in selected),
            workspace_fingerprint=fingerprint,
            metric_improvements=tuple(sorted(improved)),
        )
    except PromotionEvidenceError:
        raise
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise PromotionEvidenceError(
            f"promotion evidence is missing or unreadable: {exc}; rerun and link a valid experiment"
        ) from exc


__all__ = ["PromotionEvidence", "PromotionEvidenceError", "require_promotion_evidence"]
