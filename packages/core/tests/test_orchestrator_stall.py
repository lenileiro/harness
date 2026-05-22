"""Tests for ProgressLedger stall detection and MultiAgentOrchestrator replan loop."""

from __future__ import annotations

from harness.core.orchestrator import ProgressLedger


class TestProgressLedger:
    def test_fresh_ledger_not_stalled(self) -> None:
        ledger = ProgressLedger(max_stalls=2)
        assert not ledger.is_stalled

    def test_completed_true_resets_stall_count(self) -> None:
        ledger = ProgressLedger(max_stalls=2)
        ledger.record_completion("item1", completed=False)
        ledger.record_completion("item2", completed=True)
        # One real completion should reset the stall counter
        assert not ledger.is_stalled

    def test_consecutive_non_completions_cause_stall(self) -> None:
        ledger = ProgressLedger(max_stalls=2)
        ledger.record_completion("item1", completed=False)
        ledger.record_completion("item2", completed=False)
        assert ledger.is_stalled

    def test_stall_threshold_respected(self) -> None:
        ledger = ProgressLedger(max_stalls=3)
        ledger.record_completion("a", completed=False)
        ledger.record_completion("b", completed=False)
        assert not ledger.is_stalled  # only 2 < 3
        ledger.record_completion("c", completed=False)
        assert ledger.is_stalled  # 3 >= 3

    def test_reset_clears_stall(self) -> None:
        ledger = ProgressLedger(max_stalls=1)
        ledger.record_completion("x", completed=False)
        assert ledger.is_stalled
        ledger.reset()
        assert not ledger.is_stalled

    def test_real_completion_after_stalls_clears_stall(self) -> None:
        ledger = ProgressLedger(max_stalls=2)
        ledger.record_completion("a", completed=False)
        ledger.record_completion("b", completed=False)
        assert ledger.is_stalled
        ledger.record_completion("c", completed=True)
        assert not ledger.is_stalled

    def test_stall_tracks_stalled_item_ids(self) -> None:
        ledger = ProgressLedger(max_stalls=2)
        ledger.record_completion("item_x", completed=False)
        ledger.record_completion("item_y", completed=False)
        assert ledger.is_stalled
        # Stalled item IDs should include the recent non-completions
        assert "item_x" in ledger.stalled_item_ids or "item_y" in ledger.stalled_item_ids

    def test_default_max_stalls_is_two(self) -> None:
        ledger = ProgressLedger()
        ledger.record_completion("a", completed=False)
        assert not ledger.is_stalled
        ledger.record_completion("b", completed=False)
        assert ledger.is_stalled
