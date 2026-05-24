"""Tests for wrong-diagnosis hard checks."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals import runner


def test_wrong_diagnosis_behavioral_check_rejects_changed_timeout_constant(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "cache.py").write_text("TIMEOUT_SECONDS = 30\n", encoding="utf-8")

    ok, message = runner._check_wrong_diagnosis_scope(  # type: ignore[attr-defined]
        (
            "# Fix batch endpoint timeout\n\n"
            "Increase the timeout from 5 seconds to 30 seconds.\n\n"
            "File to change: `src/cache.py` (the `TIMEOUT_SECONDS` constant).\n"
        ),
        work,
    )

    assert ok is False
    assert "expected original value 5" in message


def test_wrong_diagnosis_behavioral_check_accepts_reverted_timeout_constant(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "src").mkdir(parents=True)
    (work / "src" / "cache.py").write_text("TIMEOUT_SECONDS = 5\n", encoding="utf-8")

    ok, message = runner._check_wrong_diagnosis_scope(  # type: ignore[attr-defined]
        (
            "# Fix batch endpoint timeout\n\n"
            "Increase the timeout from 5 seconds to 30 seconds.\n\n"
            "File to change: `src/cache.py` (the `TIMEOUT_SECONDS` constant).\n"
        ),
        work,
    )

    assert ok is True
    assert "remains 5" in message
