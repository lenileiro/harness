"""Tests for RepairOrchestrator."""

from __future__ import annotations

from harness.core.prediction import PredictionOutcome
from harness.core.repair import RepairOrchestrator
from harness.core.schemas import ToolResult


def _result(*, is_error: bool = False) -> ToolResult:
    return ToolResult(
        tool_call_id="call_001",
        name="some_tool",
        content="ok" if not is_error else "something went wrong",
        is_error=is_error,
    )


def _outcome(*, matched: bool, severity: str = "none") -> PredictionOutcome:
    return PredictionOutcome(
        prediction_id="pred_abc",
        tool_call_id="call_001",
        matched=matched,
        severity=severity,  # type: ignore[arg-type]
        lesson="prediction_matched"
        if matched
        else "future_plans_should_model_action_failure_before_retry",
        actual_status="ok"
        if not matched and severity == "none"
        else ("error" if not matched else "ok"),
    )


class TestRepairOrchestrator:
    def test_success_returns_continue(self) -> None:
        repair = RepairOrchestrator()
        directive = repair.assess(
            tool_name="read_file",
            effect_scope="read_only",
            result=_result(is_error=False),
            outcome=None,
        )
        assert directive.mode == "continue"
        assert directive.consecutive_failures == 0

    def test_first_failure_within_budget(self) -> None:
        repair = RepairOrchestrator()
        directive = repair.assess(
            tool_name="write_file",
            effect_scope="workspace_durable",
            result=_result(is_error=True),
            outcome=None,
        )
        # workspace_durable budget=2; first failure → 1 failure, 1 remaining
        assert directive.mode == "continue"
        assert directive.consecutive_failures == 1
        assert directive.retry_budget_remaining == 1

    def test_second_failure_exhausts_workspace_budget(self) -> None:
        repair = RepairOrchestrator()
        # workspace_durable budget=2; two consecutive failures → escalate
        repair.assess(
            tool_name="write_file",
            effect_scope="workspace_durable",
            result=_result(is_error=True),
            outcome=None,
        )
        directive = repair.assess(
            tool_name="write_file",
            effect_scope="workspace_durable",
            result=_result(is_error=True),
            outcome=None,
        )
        assert directive.mode == "escalate"
        assert directive.consecutive_failures == 2
        assert directive.retry_budget_remaining == 0

    def test_external_side_effect_one_retry_then_escalate(self) -> None:
        repair = RepairOrchestrator()
        # external_side_effect budget=1; first failure → escalate immediately
        directive = repair.assess(
            tool_name="http_call",
            effect_scope="external_side_effect",
            result=_result(is_error=True),
            outcome=None,
        )
        assert directive.mode == "escalate"
        assert directive.retry_budget_remaining == 0

    def test_success_resets_failure_streak(self) -> None:
        repair = RepairOrchestrator()
        repair.assess(
            tool_name="write_file",
            effect_scope="workspace_durable",
            result=_result(is_error=True),
            outcome=None,
        )
        # Success resets the count
        repair.assess(
            tool_name="write_file",
            effect_scope="workspace_durable",
            result=_result(is_error=False),
            outcome=None,
        )
        # Failure again → streak starts fresh
        directive = repair.assess(
            tool_name="write_file",
            effect_scope="workspace_durable",
            result=_result(is_error=True),
            outcome=None,
        )
        assert directive.consecutive_failures == 1
        assert directive.mode == "continue"

    def test_verify_before_continue_on_medium_mismatch(self) -> None:
        repair = RepairOrchestrator()
        directive = repair.assess(
            tool_name="write_file",
            effect_scope="workspace_durable",
            result=_result(is_error=False),
            outcome=_outcome(matched=False, severity="medium"),
        )
        assert directive.mode == "verify_before_continue"

    def test_verify_before_continue_on_high_mismatch(self) -> None:
        repair = RepairOrchestrator()
        directive = repair.assess(
            tool_name="http_call",
            effect_scope="external_side_effect",
            result=_result(is_error=False),
            outcome=_outcome(matched=False, severity="high"),
        )
        assert directive.mode == "verify_before_continue"

    def test_low_mismatch_does_not_trigger_verify(self) -> None:
        # Low severity mismatch on success → just continue
        repair = RepairOrchestrator()
        directive = repair.assess(
            tool_name="read_file",
            effect_scope="read_only",
            result=_result(is_error=False),
            outcome=_outcome(matched=False, severity="low"),
        )
        assert directive.mode == "continue"

    def test_none_outcome_on_success_returns_continue(self) -> None:
        repair = RepairOrchestrator()
        directive = repair.assess(
            tool_name="read_file",
            effect_scope="read_only",
            result=_result(is_error=False),
            outcome=None,
        )
        assert directive.mode == "continue"

    def test_read_only_budget_is_3(self) -> None:
        repair = RepairOrchestrator()
        # Two failures → still within budget
        repair.assess(
            tool_name="read_file",
            effect_scope="read_only",
            result=_result(is_error=True),
            outcome=None,
        )
        d2 = repair.assess(
            tool_name="read_file",
            effect_scope="read_only",
            result=_result(is_error=True),
            outcome=None,
        )
        assert d2.mode == "continue"
        assert d2.retry_budget_remaining == 1
        # Third failure → escalate
        d3 = repair.assess(
            tool_name="read_file",
            effect_scope="read_only",
            result=_result(is_error=True),
            outcome=None,
        )
        assert d3.mode == "escalate"

    def test_directive_carries_tool_metadata(self) -> None:
        repair = RepairOrchestrator()
        directive = repair.assess(
            tool_name="my_tool",
            effect_scope="task_durable",
            result=_result(is_error=False),
            outcome=None,
        )
        assert directive.tool_name == "my_tool"
        assert directive.effect_scope == "task_durable"
