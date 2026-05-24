"""Tests for eval judge calibration helpers."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.calibration import compare_report_to_gold, load_gold_labels, load_report


def test_load_gold_labels_and_compare(tmp_path: Path) -> None:
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    (gold_dir / "case.json").write_text(
        json.dumps(
            {
                "fixture_name": "01-demo",
                "variant": "defended",
                "run_index": 1,
                "scores": {
                    "verification": 5,
                    "scope": 4,
                    "decomposition": 5,
                    "correctness": 5,
                    "pushback": 5,
                    "epistemic": 4,
                    "overall": 5,
                },
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "fixture_name": "01-demo",
                        "variant": "defended",
                        "run_index": 1,
                        "verification": {"score": 5},
                        "scope": {"score": 3},
                        "decomposition": {"score": 5},
                        "correctness": {"score": 5},
                        "pushback": {"score": 5},
                        "epistemic": {"score": 4},
                        "overall": {"score": 4},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    labels = load_gold_labels(gold_dir)
    results = load_report(report_path)
    rows = compare_report_to_gold(results, labels)

    assert len(rows) == 7
    scope = next(row for row in rows if row.dimension == "scope")
    overall = next(row for row in rows if row.dimension == "overall")
    assert scope.mean_absolute_error == 1.0
    assert overall.exact_match_rate == 0.0
