"""Tests for EvidenceContract, evaluate_evidence, and VerificationGateway."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.core.activity import ActivityEvent
from harness.core.schemas import Session, VerificationResult
from harness.core.verification import (
    EvidenceContract,
    RuleVerifier,
    VerificationGateway,
    evaluate_evidence,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _completed_event(
    name: str,
    *,
    is_error: bool = False,
    metadata: dict | None = None,
    session_id: str = "sess_001",
) -> ActivityEvent:
    return ActivityEvent(
        session_id=session_id,
        kind="tool_call.completed",
        data={
            "tool_call_id": "call_001",
            "name": name,
            "is_error": is_error,
            "content_preview": "ok",
            "metadata": metadata or {},
        },
    )


def _prediction_error_event(
    *,
    tool_name: str = "some_tool",
    severity: str = "medium",
    matched: bool = False,
    session_id: str = "sess_001",
) -> ActivityEvent:
    return ActivityEvent(
        session_id=session_id,
        kind="tool_call.prediction_error",
        data={
            "prediction_id": "pred_abc",
            "tool_call_id": "call_001",
            "tool_name": tool_name,
            "matched": matched,
            "severity": severity,
        },
    )


def _session() -> Session:
    return Session(provider="ollama", model="llama3.2", cwd=Path("/tmp"))


# ---------------------------------------------------------------------------
# evaluate_evidence
# ---------------------------------------------------------------------------


class TestEvaluateEvidence:
    def test_command_evidence_satisfied(self) -> None:
        activity = [_completed_event("shell", metadata={"exit_code": 0})]
        contract = EvidenceContract(required_checks=["command_evidence"])
        result = evaluate_evidence(contract, activity)
        assert result.satisfied is True
        assert "command_evidence" in result.found_checks

    def test_command_evidence_unsatisfied_when_no_shell(self) -> None:
        activity = [_completed_event("read_file")]
        contract = EvidenceContract(required_checks=["command_evidence"])
        result = evaluate_evidence(contract, activity)
        assert result.satisfied is False
        assert "command_evidence" in result.missing_checks

    def test_command_evidence_unsatisfied_when_shell_errored(self) -> None:
        activity = [_completed_event("shell", is_error=True, metadata={"exit_code": 1})]
        contract = EvidenceContract(required_checks=["command_evidence"])
        result = evaluate_evidence(contract, activity)
        assert result.satisfied is False

    def test_changed_file_satisfied_by_write(self) -> None:
        activity = [_completed_event("write_file")]
        contract = EvidenceContract(required_checks=["changed_file"])
        result = evaluate_evidence(contract, activity)
        assert result.satisfied is True

    def test_changed_file_satisfied_by_edit(self) -> None:
        activity = [_completed_event("edit_file")]
        contract = EvidenceContract(required_checks=["changed_file"])
        result = evaluate_evidence(contract, activity)
        assert result.satisfied is True

    def test_changed_file_unsatisfied_when_write_errored(self) -> None:
        activity = [_completed_event("write_file", is_error=True)]
        contract = EvidenceContract(required_checks=["changed_file"])
        result = evaluate_evidence(contract, activity)
        assert result.satisfied is False

    def test_no_prediction_errors_satisfied_when_clean(self) -> None:
        activity = [_completed_event("read_file")]
        contract = EvidenceContract(required_checks=["no_prediction_errors"])
        result = evaluate_evidence(contract, activity)
        assert result.satisfied is True

    def test_no_prediction_errors_unsatisfied_on_medium_mismatch(self) -> None:
        activity = [_prediction_error_event(severity="medium", matched=False)]
        contract = EvidenceContract(required_checks=["no_prediction_errors"])
        result = evaluate_evidence(contract, activity)
        assert result.satisfied is False

    def test_no_prediction_errors_satisfied_on_matched_event(self) -> None:
        # A prediction_error event where matched=True should not fail the check
        activity = [_prediction_error_event(severity="medium", matched=True)]
        contract = EvidenceContract(required_checks=["no_prediction_errors"])
        result = evaluate_evidence(contract, activity)
        assert result.satisfied is True

    def test_no_prediction_errors_ignores_low_severity(self) -> None:
        activity = [_prediction_error_event(severity="low", matched=False)]
        contract = EvidenceContract(required_checks=["no_prediction_errors"])
        result = evaluate_evidence(contract, activity)
        assert result.satisfied is True

    def test_tool_succeeded_satisfied(self) -> None:
        activity = [_completed_event("pytest")]
        contract = EvidenceContract(
            required_checks=["tool_succeeded"],
            check_data={"tool": "pytest"},
        )
        result = evaluate_evidence(contract, activity)
        assert result.satisfied is True

    def test_tool_succeeded_unsatisfied_on_error(self) -> None:
        activity = [_completed_event("pytest", is_error=True)]
        contract = EvidenceContract(
            required_checks=["tool_succeeded"],
            check_data={"tool": "pytest"},
        )
        result = evaluate_evidence(contract, activity)
        assert result.satisfied is False

    def test_tool_succeeded_unsatisfied_when_different_tool(self) -> None:
        activity = [_completed_event("ruff")]
        contract = EvidenceContract(
            required_checks=["tool_succeeded"],
            check_data={"tool": "pytest"},
        )
        result = evaluate_evidence(contract, activity)
        assert result.satisfied is False

    def test_multiple_checks_all_required(self) -> None:
        activity = [
            _completed_event("write_file"),
            _completed_event("shell", metadata={"exit_code": 0}),
        ]
        contract = EvidenceContract(required_checks=["changed_file", "command_evidence"])
        result = evaluate_evidence(contract, activity)
        assert result.satisfied is True
        assert sorted(result.found_checks) == ["changed_file", "command_evidence"]

    def test_multiple_checks_partial_failure(self) -> None:
        activity = [_completed_event("write_file")]  # no shell
        contract = EvidenceContract(required_checks=["changed_file", "command_evidence"])
        result = evaluate_evidence(contract, activity)
        assert result.satisfied is False
        assert "changed_file" in result.found_checks
        assert "command_evidence" in result.missing_checks

    def test_empty_activity_all_missing(self) -> None:
        contract = EvidenceContract(required_checks=["changed_file", "command_evidence"])
        result = evaluate_evidence(contract, [])
        assert result.satisfied is False
        assert sorted(result.missing_checks) == ["changed_file", "command_evidence"]


# ---------------------------------------------------------------------------
# VerificationGateway
# ---------------------------------------------------------------------------


class TestVerificationGateway:
    """VerificationGateway: prediction mismatch gate + evidence contract gate + inner verifier."""

    def _make_gateway(self, contract: EvidenceContract | None = None) -> VerificationGateway:
        rule = RuleVerifier()
        return VerificationGateway(rule, contract)

    @pytest.mark.asyncio
    async def test_passes_through_to_inner_verifier_when_no_issues(self) -> None:
        gateway = self._make_gateway()
        session = _session()
        result = await gateway.verify(session=session, activity=[])
        assert isinstance(result, VerificationResult)
        # Pass-through: inner verifier's name, not "gateway"
        assert result.verifier_name == "rule"

    @pytest.mark.asyncio
    async def test_blocks_on_unresolved_prediction_mismatch(self) -> None:
        gateway = self._make_gateway()
        session = _session()
        activity = [_prediction_error_event(severity="high", matched=False)]
        result = await gateway.verify(session=session, activity=activity)
        assert result.can_finish is False
        assert "mismatch" in result.reason.lower()
        assert result.verifier_name == "gateway"

    @pytest.mark.asyncio
    async def test_allows_matched_prediction_event(self) -> None:
        gateway = self._make_gateway()
        session = _session()
        activity = [_prediction_error_event(severity="high", matched=True)]
        result = await gateway.verify(session=session, activity=activity)
        # matched=True means no mismatch → passes gate 1
        # inner verifier (RuleVerifier with empty session) → can_finish=True
        assert result.can_finish is True

    @pytest.mark.asyncio
    async def test_blocks_on_unsatisfied_contract(self) -> None:
        contract = EvidenceContract(required_checks=["command_evidence"])
        gateway = self._make_gateway(contract)
        session = _session()
        activity = [_completed_event("read_file")]  # no shell
        result = await gateway.verify(session=session, activity=activity)
        assert result.can_finish is False
        assert "command_evidence" in result.reason

    @pytest.mark.asyncio
    async def test_passes_when_contract_satisfied(self) -> None:
        contract = EvidenceContract(required_checks=["changed_file"])
        gateway = self._make_gateway(contract)
        session = _session()
        activity = [_completed_event("write_file")]
        result = await gateway.verify(session=session, activity=activity)
        assert result.can_finish is True

    @pytest.mark.asyncio
    async def test_mismatch_gate_runs_before_contract_gate(self) -> None:
        # Both prediction mismatch AND unsatisfied contract — mismatch reason should appear
        contract = EvidenceContract(required_checks=["command_evidence"])
        gateway = self._make_gateway(contract)
        session = _session()
        activity = [_prediction_error_event(severity="critical", matched=False)]
        result = await gateway.verify(session=session, activity=activity)
        assert result.can_finish is False
        assert "mismatch" in result.reason.lower()
        assert "command_evidence" not in result.reason

    @pytest.mark.asyncio
    async def test_ignores_low_severity_prediction_errors(self) -> None:
        gateway = self._make_gateway()
        session = _session()
        activity = [_prediction_error_event(severity="low", matched=False)]
        result = await gateway.verify(session=session, activity=activity)
        # Low severity → gate 1 passes
        assert result.can_finish is True
