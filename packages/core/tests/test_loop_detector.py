"""Tests for L4 — trajectory regulation / loop detector."""

from __future__ import annotations

import pytest

from harness.core import LoopDetector, LoopFinding
from harness.core.schemas import ToolCall


def _call(name: str, args: dict | None = None, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=args or {})


class TestLoopDetectorRepeat:
    def test_two_identical_calls_no_finding(self) -> None:
        det = LoopDetector(repeat_threshold=3, no_progress_threshold=10)
        assert det.observe(_call("read_file", {"path": "a"})) is None
        assert det.observe(_call("read_file", {"path": "a"})) is None

    def test_three_identical_calls_trips_finding(self) -> None:
        det = LoopDetector(repeat_threshold=3, no_progress_threshold=10)
        det.observe(_call("read_file", {"path": "a"}))
        det.observe(_call("read_file", {"path": "a"}))
        finding = det.observe(_call("read_file", {"path": "a"}))
        assert isinstance(finding, LoopFinding)
        assert finding.pattern == "tool_repeat"
        assert finding.tool == "read_file"
        assert finding.repeats == 3
        assert "identical arguments" in finding.directive

    def test_finding_is_idempotent_for_same_signature(self) -> None:
        """The detector emits at most one finding per signature so the
        user-message inbox doesn't fill with duplicates."""
        det = LoopDetector(repeat_threshold=3, no_progress_threshold=10)
        for _ in range(5):
            det.observe(_call("read_file", {"path": "a"}))
        # First emission happened at the 3rd call; nothing else after.
        assert det.observe(_call("read_file", {"path": "a"})) is None

    def test_different_args_do_not_count_as_repeat(self) -> None:
        det = LoopDetector(repeat_threshold=3, no_progress_threshold=10)
        det.observe(_call("read_file", {"path": "a"}))
        det.observe(_call("read_file", {"path": "b"}))
        # Same tool, different args → not a tool_repeat.
        assert det.observe(_call("read_file", {"path": "c"})) is None

    def test_order_of_dict_keys_does_not_matter(self) -> None:
        """args are hashed by content, not by key order."""
        det = LoopDetector(repeat_threshold=3, no_progress_threshold=10)
        det.observe(_call("shell", {"cmd": "ls", "cwd": "/tmp"}))
        det.observe(_call("shell", {"cwd": "/tmp", "cmd": "ls"}))
        finding = det.observe(_call("shell", {"cmd": "ls", "cwd": "/tmp"}))
        assert finding is not None and finding.pattern == "tool_repeat"


class TestLoopDetectorNoProgress:
    def test_six_read_only_calls_trips_no_progress(self) -> None:
        det = LoopDetector(repeat_threshold=10, no_progress_threshold=6, window=12)
        # Read-only spamming with varied args (so tool_repeat doesn't fire).
        for i in range(5):
            assert det.observe(_call("read_file", {"path": f"{i}"})) is None
        finding = det.observe(_call("read_file", {"path": "5"}))
        assert isinstance(finding, LoopFinding)
        assert finding.pattern == "no_progress"

    def test_a_mutating_tool_resets_no_progress(self) -> None:
        det = LoopDetector(repeat_threshold=10, no_progress_threshold=6, window=12)
        for i in range(4):
            det.observe(_call("read_file", {"path": f"{i}"}))
        # write_file inside the window → no_progress should not fire.
        det.observe(_call("write_file", {"path": "x", "content": "y"}))
        # Two more reads — window of last 6 is read,read,write,read,read so
        # the mutating call keeps no_progress quiet.
        assert det.observe(_call("read_file", {"path": "5"})) is None
        assert det.observe(_call("read_file", {"path": "6"})) is None


class TestLoopDetectorReset:
    def test_reset_clears_state(self) -> None:
        det = LoopDetector(repeat_threshold=3, no_progress_threshold=10)
        for _ in range(3):
            det.observe(_call("read_file", {"path": "a"}))
        det.reset()
        # After reset we should be able to trip the pattern again.
        det.observe(_call("read_file", {"path": "a"}))
        det.observe(_call("read_file", {"path": "a"}))
        finding = det.observe(_call("read_file", {"path": "a"}))
        assert finding is not None


class TestLoopDetectorValidation:
    def test_threshold_below_two_rejected(self) -> None:
        with pytest.raises(ValueError):
            LoopDetector(repeat_threshold=1, no_progress_threshold=5)
        with pytest.raises(ValueError):
            LoopDetector(repeat_threshold=3, no_progress_threshold=1)

    def test_window_smaller_than_threshold_rejected(self) -> None:
        with pytest.raises(ValueError):
            LoopDetector(repeat_threshold=5, no_progress_threshold=3, window=4)
