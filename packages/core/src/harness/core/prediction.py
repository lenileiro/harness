"""ConsequencePredictor — deterministic pre-execution prediction + post-execution diff.

Before each tool executes, `ConsequencePredictor.predict()` generates a structured
`ToolPrediction` that commits to an expected outcome. After execution,
`compare_prediction()` produces a `PredictionOutcome` that scores the match.

This is the core honesty mechanism: tool calls become auditable commitments.
Mismatches are severity-scored and fed into calibration + repair.

No LLM call — fully deterministic, keyed on effect_scope.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from harness.core.schemas import EffectScope, ToolCall, ToolResult

ExpectedStatus = Literal["ok", "ok_or_error", "blocked_before_execution"]
PredictionSeverity = Literal["none", "low", "medium", "high", "critical"]


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _pred_id(tool_call_id: str, effect_scope: str | None) -> str:
    raw = (tool_call_id + (effect_scope or "")).encode()
    return "pred_" + hashlib.sha256(raw).hexdigest()[:16]


# ---------------------------------------------------------------------------
# ToolPrediction — committed before execution
# ---------------------------------------------------------------------------


class ToolPrediction(BaseModel):
    """What the runtime expects a tool call to do before it executes."""

    model_config = ConfigDict(extra="forbid")

    prediction_id: str
    tool_call_id: str
    tool_name: str
    effect_scope: EffectScope | None
    expected_status: ExpectedStatus
    possible_failures: list[str]
    confidence: float
    reversibility: str
    predicted_at: datetime = Field(default_factory=_utcnow)


# ---------------------------------------------------------------------------
# PredictionOutcome — produced after execution
# ---------------------------------------------------------------------------


class PredictionOutcome(BaseModel):
    """Comparison of a ToolPrediction against what actually happened."""

    model_config = ConfigDict(extra="forbid")

    prediction_id: str
    tool_call_id: str
    matched: bool
    severity: PredictionSeverity
    lesson: str
    actual_status: str


# ---------------------------------------------------------------------------
# ConsequencePredictor
# ---------------------------------------------------------------------------

_SCOPE_CONFIG: dict[EffectScope | None, dict[str, Any]] = {
    "read_only": {
        "status": "ok",
        "confidence": 0.88,
        "reversibility": "always",
        "failures": ["file_not_found", "permission_denied", "encoding_error"],
    },
    "session_ephemeral": {
        "status": "ok",
        "confidence": 0.85,
        "reversibility": "always",
        "failures": ["resource_exhausted"],
    },
    "task_durable": {
        "status": "ok",
        "confidence": 0.80,
        "reversibility": "storage_delete",
        "failures": ["storage_write_error", "conflict"],
    },
    "agent_orchestration": {
        "status": "ok_or_error",
        "confidence": 0.74,
        "reversibility": "cancel_child",
        "failures": ["child_agent_error", "timeout", "delegation_loop"],
    },
    "workspace_durable": {
        "status": "ok",
        "confidence": 0.74,
        "reversibility": "git_or_manual",
        "failures": ["file_write_error", "permission_denied", "disk_full"],
    },
    "external_side_effect": {
        "status": "ok_or_error",
        "confidence": 0.56,
        "reversibility": "none",
        "failures": ["network_error", "auth_failed", "rate_limited", "api_error"],
    },
    "routed": {
        "status": "ok_or_error",
        "confidence": 0.60,
        "reversibility": "depends",
        "failures": ["routing_error", "downstream_error"],
    },
    None: {
        "status": "ok_or_error",
        "confidence": 0.60,
        "reversibility": "unknown",
        "failures": ["unknown_error"],
    },
}


class ConsequencePredictor:
    """Deterministic rule-based predictor — no LLM call.

    Generates a ToolPrediction keyed on effect_scope. Confidence reflects
    how predictable the outcome is for that scope category.
    """

    def predict(
        self,
        *,
        tool_name: str,
        call: ToolCall,
        effect_scope: EffectScope | None,
    ) -> ToolPrediction:
        cfg = _SCOPE_CONFIG[effect_scope]
        return ToolPrediction(
            prediction_id=_pred_id(call.id, effect_scope),
            tool_call_id=call.id,
            tool_name=tool_name,
            effect_scope=effect_scope,
            expected_status=cfg["status"],
            possible_failures=cfg["failures"],
            confidence=cfg["confidence"],
            reversibility=cfg["reversibility"],
        )


# ---------------------------------------------------------------------------
# compare_prediction
# ---------------------------------------------------------------------------


def _status_matched(expected: ExpectedStatus, actual: str) -> bool:
    """Return True when the actual status satisfies the expected status."""
    if expected == "ok":
        return actual == "ok"
    if expected == "ok_or_error":
        return actual in ("ok", "error")
    if expected == "blocked_before_execution":
        return actual in ("denied", "queued", "error")
    return False


def _mismatch_severity(effect_scope: EffectScope | None) -> PredictionSeverity:
    if effect_scope == "external_side_effect":
        return "high"
    if effect_scope in ("workspace_durable", "agent_orchestration"):
        return "medium"
    return "low"


def _lesson(effect_scope: EffectScope | None, actual: str) -> str:
    if actual == "error":
        return "future_plans_should_model_action_failure_before_retry"
    return "future_plans_should_verify_observation_before_continuing"


def compare_prediction(prediction: ToolPrediction, result: ToolResult) -> PredictionOutcome:
    """Compare a pre-execution prediction against the actual ToolResult."""
    actual_status = "error" if result.is_error else "ok"
    matched = _status_matched(prediction.expected_status, actual_status)
    severity: PredictionSeverity = "none" if matched else _mismatch_severity(prediction.effect_scope)
    lesson = "prediction_matched" if matched else _lesson(prediction.effect_scope, actual_status)
    return PredictionOutcome(
        prediction_id=prediction.prediction_id,
        tool_call_id=prediction.tool_call_id,
        matched=matched,
        severity=severity,
        lesson=lesson,
        actual_status=actual_status,
    )


__all__ = [
    "ComparePredictor",
    "ConsequencePredictor",
    "ExpectedStatus",
    "PredictionOutcome",
    "PredictionSeverity",
    "ToolPrediction",
    "compare_prediction",
]
