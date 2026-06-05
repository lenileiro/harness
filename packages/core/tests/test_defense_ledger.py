from __future__ import annotations

from harness.core.activity import REPAIR_DIRECTIVE_ISSUED, ActivityEvent
from harness.core.defense_ledger import build_ledger, format_ledger


def _event(**data: object) -> ActivityEvent:
    return ActivityEvent(session_id="s1", kind=REPAIR_DIRECTIVE_ISSUED, data=dict(data))


def test_ledger_counts_only_verifier_failure_directives_as_repairs() -> None:
    ledger = build_ledger(
        [
            _event(mode="empty_final_verify_current_state"),
            _event(mode="model_timeout_verify_current_state"),
            _event(attempt=1, verifier="verify_before_done", critic=True),
        ]
    )

    assert ledger.repair_attempts == 1
    assert ledger.critic_invocations == 1
    assert "repair attempts: 1" in format_ledger(ledger)


def test_ledger_omits_repair_attempts_for_verify_current_state_handoffs() -> None:
    ledger = build_ledger([_event(mode="empty_final_verify_current_state")])

    assert ledger.repair_attempts == 0
    assert "repair attempts:" not in format_ledger(ledger)
