"""Tests for OutcomeCalibration."""

from __future__ import annotations

import pytest

from harness.core.calibration import CalibrationRecord, OutcomeCalibration
from harness.core.prediction import PredictionOutcome


def _outcome(*, matched: bool, severity: str = "none") -> PredictionOutcome:
    return PredictionOutcome(
        prediction_id="pred_abc123",
        tool_call_id="call_001",
        matched=matched,
        severity=severity,  # type: ignore[arg-type]
        lesson="prediction_matched" if matched else "future_plans_should_model_action_failure_before_retry",
        actual_status="ok" if matched else "error",
    )


class TestOutcomeCalibration:
    def test_match_increases_confidence(self) -> None:
        cal = OutcomeCalibration()
        adjusted = cal.calibrate(base_confidence=0.74, outcome=_outcome(matched=True))
        assert adjusted == pytest.approx(0.78)

    def test_low_mismatch_decreases_confidence(self) -> None:
        cal = OutcomeCalibration()
        adjusted = cal.calibrate(base_confidence=0.74, outcome=_outcome(matched=False, severity="low"))
        assert adjusted == pytest.approx(0.66)

    def test_medium_mismatch_decreases_confidence(self) -> None:
        cal = OutcomeCalibration()
        adjusted = cal.calibrate(base_confidence=0.74, outcome=_outcome(matched=False, severity="medium"))
        assert adjusted == pytest.approx(0.60)

    def test_high_mismatch_decreases_confidence(self) -> None:
        cal = OutcomeCalibration()
        adjusted = cal.calibrate(base_confidence=0.74, outcome=_outcome(matched=False, severity="high"))
        assert adjusted == pytest.approx(0.52)

    def test_critical_mismatch_hard_floor(self) -> None:
        cal = OutcomeCalibration()
        # 0.20 - 0.30 = -0.10, should floor at 0.1
        adjusted = cal.calibrate(base_confidence=0.20, outcome=_outcome(matched=False, severity="critical"))
        assert adjusted == pytest.approx(0.1)

    def test_match_ceiling_at_0_99(self) -> None:
        cal = OutcomeCalibration()
        # 0.97 + 0.04 = 1.01, should cap at 0.99
        adjusted = cal.calibrate(base_confidence=0.97, outcome=_outcome(matched=True))
        assert adjusted == pytest.approx(0.99)

    def test_record_returns_calibration_record(self) -> None:
        cal = OutcomeCalibration()
        outcome = _outcome(matched=True)
        record = cal.record(
            tool_name="write_file",
            effect_scope="workspace_durable",
            base_confidence=0.74,
            outcome=outcome,
        )
        assert isinstance(record, CalibrationRecord)
        assert record.base_confidence == pytest.approx(0.74)
        assert record.adjusted_confidence == pytest.approx(0.78)
        assert record.tool_name == "write_file"
        assert record.effect_scope == "workspace_durable"
        assert len(record.pattern_key) == 16

    def test_record_pattern_key_is_stable(self) -> None:
        cal = OutcomeCalibration()
        outcome = _outcome(matched=False, severity="low")
        r1 = cal.record(tool_name="read_file", effect_scope="read_only", base_confidence=0.88, outcome=outcome)
        r2 = cal.record(tool_name="read_file", effect_scope="read_only", base_confidence=0.88, outcome=outcome)
        assert r1.pattern_key == r2.pattern_key

    def test_record_pattern_key_differs_by_tool(self) -> None:
        cal = OutcomeCalibration()
        outcome = _outcome(matched=True)
        r1 = cal.record(tool_name="read_file", effect_scope="read_only", base_confidence=0.88, outcome=outcome)
        r2 = cal.record(tool_name="write_file", effect_scope="read_only", base_confidence=0.88, outcome=outcome)
        assert r1.pattern_key != r2.pattern_key

    def test_none_severity_on_matched_outcome(self) -> None:
        cal = OutcomeCalibration()
        # severity="none" with matched=True should use the match delta (+0.04)
        adjusted = cal.calibrate(
            base_confidence=0.80,
            outcome=_outcome(matched=True, severity="none"),
        )
        assert adjusted == pytest.approx(0.84)
