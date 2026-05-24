"""Judge calibration helpers for comparing eval reports to gold labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DIMENSIONS = (
    "verification",
    "scope",
    "decomposition",
    "correctness",
    "pushback",
    "epistemic",
    "overall",
)


@dataclass(slots=True)
class GoldLabel:
    fixture_name: str
    variant: str
    run_index: int
    scores: dict[str, int]


@dataclass(slots=True)
class CalibrationDimension:
    dimension: str
    count: int
    exact_match_rate: float
    mean_absolute_error: float


def load_gold_labels(gold_dir: Path) -> list[GoldLabel]:
    labels: list[GoldLabel] = []
    if not gold_dir.exists():
        return labels
    for path in sorted(gold_dir.glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        labels.append(
            GoldLabel(
                fixture_name=str(raw["fixture_name"]),
                variant=str(raw.get("variant", "defended")),
                run_index=int(raw.get("run_index", 1)),
                scores={
                    dim: int(raw["scores"][dim]) for dim in _DIMENSIONS if dim in raw["scores"]
                },
            )
        )
    return labels


def load_report(report_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(report_path.read_text(encoding="utf-8"))
    return list(raw.get("results", []))


def compare_report_to_gold(
    report_results: list[dict[str, Any]],
    labels: list[GoldLabel],
) -> list[CalibrationDimension]:
    report_index = {
        (item["fixture_name"], item.get("variant", "defended"), int(item.get("run_index", 1))): item
        for item in report_results
    }
    rows: list[CalibrationDimension] = []
    for dim in _DIMENSIONS:
        diffs: list[int] = []
        exact = 0
        for label in labels:
            report = report_index.get((label.fixture_name, label.variant, label.run_index))
            if report is None or dim not in label.scores:
                continue
            report_score = int(report[dim]["score"])
            label_score = label.scores[dim]
            diff = abs(report_score - label_score)
            diffs.append(diff)
            if diff == 0:
                exact += 1
        if not diffs:
            continue
        rows.append(
            CalibrationDimension(
                dimension=dim,
                count=len(diffs),
                exact_match_rate=exact / len(diffs),
                mean_absolute_error=sum(diffs) / len(diffs),
            )
        )
    return rows
