"""OutcomeCalibration — adjust confidence scores based on prediction accuracy.

After each tool call, `OutcomeCalibration.calibrate()` takes the base confidence
from `ConsequencePredictor` and returns an adjusted value:
- Matched predictions increase confidence slightly (+0.04)
- Mismatches decrease it, scaled by severity

Results are emitted as `calibration.updated` activity events so they're
queryable from the ledger. The calibration object itself is stateless —
it's a pure function that adjusts a single data point.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from harness.core.prediction import PredictionOutcome
from harness.core.schemas import EffectScope


class CalibrationRecord(BaseModel):
    """A single calibration data point for one tool call outcome."""

    model_config = ConfigDict(extra="forbid")

    pattern_key: str
    tool_name: str
    effect_scope: EffectScope | None
    base_confidence: float
    adjusted_confidence: float


_MATCH_DELTA = 0.04
_MISMATCH_DELTAS: dict[str, float] = {
    "low": -0.08,
    "medium": -0.14,
    "high": -0.22,
    "critical": -0.30,
}


class OutcomeCalibration:
    """Stateless confidence adjuster.

    Mirrors holt's OutcomeCalibration: a single calibrate() call takes a
    base confidence and a PredictionOutcome and returns the adjusted value.
    Hard floor of 0.1, ceiling of 0.99.
    """

    def calibrate(
        self,
        *,
        base_confidence: float,
        outcome: PredictionOutcome,
    ) -> float:
        if outcome.matched:
            delta = _MATCH_DELTA
        else:
            delta = _MISMATCH_DELTAS.get(outcome.severity, -0.08)
        return max(0.1, min(0.99, base_confidence + delta))

    def record(
        self,
        *,
        tool_name: str,
        effect_scope: EffectScope | None,
        base_confidence: float,
        outcome: PredictionOutcome,
    ) -> CalibrationRecord:
        """Return a CalibrationRecord for this outcome (for activity emission)."""
        import hashlib

        key_raw = (tool_name + (effect_scope or "")).encode()
        pattern_key = hashlib.sha256(key_raw).hexdigest()[:16]
        adjusted = self.calibrate(base_confidence=base_confidence, outcome=outcome)
        return CalibrationRecord(
            pattern_key=pattern_key,
            tool_name=tool_name,
            effect_scope=effect_scope,
            base_confidence=base_confidence,
            adjusted_confidence=adjusted,
        )


__all__ = [
    "CalibrationRecord",
    "OutcomeCalibration",
]
