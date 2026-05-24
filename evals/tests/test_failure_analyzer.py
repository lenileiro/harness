from __future__ import annotations

import json
from pathlib import Path

from evals.failure_analyzer import analyze_artifact_dir, persist_harness_adjustments


def _write_artifact_bundle(tmp_path: Path, *, family: str, task_text: str, git_diff: str) -> Path:
    artifact_dir = tmp_path / "run-01"
    artifact_dir.mkdir(parents=True)
    outcome = {
        "fixture": {
            "name": "03-wrong-diagnosis",
            "family": family,
            "task_text": task_text,
        },
        "variant": "bare",
        "test_exit_code": 1,
        "hard_metrics": {
            "verify_passed": False,
            "edit_before_repro": True,
            "verification_after_failure": True,
            "retry_loops": 2,
            "tool_calls": 8,
            "time_to_first_verification_seconds": 12.5,
        },
    }
    (artifact_dir / "outcome.json").write_text(json.dumps(outcome), encoding="utf-8")
    (artifact_dir / "trace.jsonl").write_text(
        json.dumps({"kind": "tool_call", "message": "shell"}) + "\n",
        encoding="utf-8",
    )
    (artifact_dir / "git_diff.patch").write_text(git_diff, encoding="utf-8")
    return artifact_dir


def test_analyze_artifact_dir_emits_structured_adjustments(tmp_path: Path) -> None:
    artifact_dir = _write_artifact_bundle(
        tmp_path,
        family="wrong-diagnosis",
        task_text=(
            "Increase the `TIMEOUT_SECONDS` constant to 30 and fix the real caching bug if needed."
        ),
        git_diff="+# temporary banner\n",
    )

    adjustments = analyze_artifact_dir(artifact_dir)

    assert adjustments
    assert any(adjustment.kind == "tests_first" for adjustment in adjustments)
    assert any(adjustment.kind == "prompt_surface_revert" for adjustment in adjustments)
    assert all(adjustment.source_artifact_dir == artifact_dir for adjustment in adjustments)


def test_persist_harness_adjustments_writes_json_file(tmp_path: Path) -> None:
    artifact_dir = _write_artifact_bundle(
        tmp_path,
        family="scope-discipline",
        task_text="`format_price(None)` raises a `TypeError`.",
        git_diff='@@\n+    if amount is None:\n+        return "—"\n',
    )

    persisted = persist_harness_adjustments(artifact_dir)

    saved = json.loads((artifact_dir / "harness_adjustments.json").read_text(encoding="utf-8"))
    assert len(saved) == len(persisted)
    assert any(item["kind"] == "scope-discipline" for item in saved) is False
    assert any(item["kind"] == "family_pattern" for item in saved)
