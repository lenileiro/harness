"""Tests for ConsequencePredictor, compare_prediction, and runtime integration."""

from __future__ import annotations

import pytest

from harness.core.prediction import (
    ConsequencePredictor,
    PredictionOutcome,
    ToolPrediction,
    compare_prediction,
)
from harness.core.schemas import ToolCall, ToolResult


def _call(name: str = "some_tool") -> ToolCall:
    return ToolCall(id="call_001", name=name, arguments={})


def _result(*, is_error: bool = False) -> ToolResult:
    return ToolResult(
        tool_call_id="call_001",
        name="some_tool",
        content="ok" if not is_error else "something went wrong",
        is_error=is_error,
    )


class TestConsequencePredictor:
    def test_read_only_predicts_ok(self) -> None:
        predictor = ConsequencePredictor()
        call = _call("read_file")
        pred = predictor.predict(tool_name="read_file", call=call, effect_scope="read_only")
        assert pred.expected_status == "ok"
        assert pred.confidence == pytest.approx(0.88)
        assert pred.effect_scope == "read_only"
        assert pred.tool_call_id == call.id

    def test_workspace_durable_confidence(self) -> None:
        predictor = ConsequencePredictor()
        call = _call("write_file")
        pred = predictor.predict(tool_name="write_file", call=call, effect_scope="workspace_durable")
        assert pred.expected_status == "ok"
        assert pred.confidence == pytest.approx(0.74)
        assert pred.reversibility == "git_or_manual"

    def test_external_side_effect_is_ok_or_error(self) -> None:
        predictor = ConsequencePredictor()
        call = _call("http_call")
        pred = predictor.predict(tool_name="http_call", call=call, effect_scope="external_side_effect")
        assert pred.expected_status == "ok_or_error"
        assert pred.confidence == pytest.approx(0.56)
        assert pred.reversibility == "none"

    def test_none_scope_is_ok_or_error(self) -> None:
        predictor = ConsequencePredictor()
        call = _call("custom")
        pred = predictor.predict(tool_name="custom", call=call, effect_scope=None)
        assert pred.expected_status == "ok_or_error"
        assert pred.confidence == pytest.approx(0.60)

    def test_prediction_id_is_stable_and_prefixed(self) -> None:
        predictor = ConsequencePredictor()
        call = _call("read_file")
        pred = predictor.predict(tool_name="read_file", call=call, effect_scope="read_only")
        assert pred.prediction_id.startswith("pred_")
        # Same call + scope → same id (deterministic)
        pred2 = predictor.predict(tool_name="read_file", call=call, effect_scope="read_only")
        assert pred.prediction_id == pred2.prediction_id

    def test_prediction_id_differs_by_scope(self) -> None:
        predictor = ConsequencePredictor()
        call = _call("tool")
        pred_ro = predictor.predict(tool_name="tool", call=call, effect_scope="read_only")
        pred_ws = predictor.predict(tool_name="tool", call=call, effect_scope="workspace_durable")
        assert pred_ro.prediction_id != pred_ws.prediction_id

    def test_possible_failures_non_empty(self) -> None:
        predictor = ConsequencePredictor()
        call = _call()
        for scope in ("read_only", "workspace_durable", "external_side_effect", None):
            pred = predictor.predict(tool_name="t", call=call, effect_scope=scope)  # type: ignore[arg-type]
            assert len(pred.possible_failures) > 0


class TestComparePrediction:
    def _prediction(self, *, effect_scope: str | None, override_status: str | None = None) -> ToolPrediction:
        predictor = ConsequencePredictor()
        call = _call()
        pred = predictor.predict(tool_name="t", call=call, effect_scope=effect_scope)  # type: ignore[arg-type]
        if override_status is not None:
            pred = pred.model_copy(update={"expected_status": override_status})
        return pred

    def test_prediction_matched_on_success(self) -> None:
        pred = self._prediction(effect_scope="read_only")  # expected_status="ok"
        outcome = compare_prediction(pred, _result(is_error=False))
        assert outcome.matched is True
        assert outcome.severity == "none"
        assert outcome.lesson == "prediction_matched"

    def test_prediction_mismatch_on_error(self) -> None:
        pred = self._prediction(effect_scope="read_only")  # expected_status="ok"
        outcome = compare_prediction(pred, _result(is_error=True))
        assert outcome.matched is False
        assert outcome.severity == "low"  # read_only → low

    def test_ok_or_error_tolerates_error(self) -> None:
        # external_side_effect has expected_status="ok_or_error" — error is tolerated
        pred = self._prediction(effect_scope="external_side_effect")
        assert pred.expected_status == "ok_or_error"
        outcome = compare_prediction(pred, _result(is_error=True))
        assert outcome.matched is True
        assert outcome.severity == "none"

    def test_ok_or_error_also_matches_success(self) -> None:
        pred = self._prediction(effect_scope="external_side_effect")
        outcome = compare_prediction(pred, _result(is_error=False))
        assert outcome.matched is True

    def test_workspace_durable_mismatch_is_medium(self) -> None:
        pred = self._prediction(effect_scope="workspace_durable")  # expected_status="ok"
        outcome = compare_prediction(pred, _result(is_error=True))
        assert outcome.matched is False
        assert outcome.severity == "medium"

    def test_external_side_effect_unexpected_mismatch_is_high(self) -> None:
        # Force blocked_before_execution on external_side_effect, then pass ok result
        pred = self._prediction(effect_scope="external_side_effect", override_status="blocked_before_execution")
        # actual="ok" doesn't satisfy "blocked_before_execution"
        outcome = compare_prediction(pred, _result(is_error=False))
        assert outcome.matched is False
        assert outcome.severity == "high"

    def test_blocked_before_execution_matches_denied(self) -> None:
        # ToolResult with is_error=True maps to actual_status="error", which satisfies
        # "blocked_before_execution" (denied/queued/error are all accepted)
        pred = self._prediction(effect_scope=None, override_status="blocked_before_execution")
        outcome = compare_prediction(pred, _result(is_error=True))
        assert outcome.matched is True

    def test_outcome_carries_prediction_id(self) -> None:
        pred = self._prediction(effect_scope="read_only")
        outcome = compare_prediction(pred, _result())
        assert outcome.prediction_id == pred.prediction_id
        assert outcome.tool_call_id == pred.tool_call_id
