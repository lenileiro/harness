"""Tests for L4 — trajectory regulation / loop detector."""

from __future__ import annotations

import pytest

from harness.core import LoopDetector, LoopFinding
from harness.core.schemas import ToolCall, ToolResult


def _call(name: str, args: dict | None = None, call_id: str = "c1") -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=args or {})


def _error_result(name: str = "edit_file", content: str = "`old` not found") -> ToolResult:
    return ToolResult(tool_call_id="c1", name=name, content=content, is_error=True)


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
    def test_default_threshold_allows_initial_large_repo_exploration(self) -> None:
        det = LoopDetector(repeat_threshold=12)

        for i in range(6):
            assert det.observe(_call("read_file", {"path": f"{i}"})) is None

    def test_default_threshold_trips_after_sustained_no_progress(self) -> None:
        det = LoopDetector(repeat_threshold=12)

        for i in range(9):
            assert det.observe(_call("read_file", {"path": f"{i}"})) is None
        finding = det.observe(_call("read_file", {"path": "9"}))

        assert isinstance(finding, LoopFinding)
        assert finding.pattern == "no_progress"
        assert "materially different autonomous path" in finding.directive
        assert "blocker and stop" not in finding.directive
        assert finding.repeats == 10
        assert "state the current hypothesis" in finding.directive

    def test_six_read_only_calls_trips_no_progress(self) -> None:
        det = LoopDetector(repeat_threshold=10, no_progress_threshold=6, window=12)
        # Read-only spamming with varied args (so tool_repeat doesn't fire).
        for i in range(5):
            assert det.observe(_call("read_file", {"path": f"{i}"})) is None
        finding = det.observe(_call("read_file", {"path": "5"}))
        assert isinstance(finding, LoopFinding)
        assert finding.pattern == "no_progress"
        assert "state the current hypothesis" in finding.directive

    def test_no_progress_finding_is_not_repeated_for_every_new_read(self) -> None:
        det = LoopDetector(repeat_threshold=6, no_progress_threshold=3, window=8)
        assert det.observe(_call("read_file", {"path": "0"})) is None
        assert det.observe(_call("read_file", {"path": "1"})) is None
        first = det.observe(_call("read_file", {"path": "2"}))
        repeated = det.observe(_call("read_file", {"path": "3"}))

        assert isinstance(first, LoopFinding)
        assert first.pattern == "no_progress"
        assert repeated is None

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

    def test_failed_shell_commands_do_not_count_as_progress(self) -> None:
        det = LoopDetector(repeat_threshold=10, no_progress_threshold=3, window=10)

        det.observe(
            _call("shell", {"command": "python app.py one"}),
            _error_result("shell", "exit_code: 2\nstderr: usage: app.py arg\nerror: bad arg"),
        )
        det.observe(
            _call("shell", {"command": "python app.py two"}),
            _error_result("shell", "exit_code: 2\nstderr: usage: app.py arg\nerror: bad arg"),
        )
        finding = det.observe(
            _call("verify_work", {"command": "python app.py three"}),
            _error_result(
                "verify_work",
                "FAILED (exit 2)\nusage: app.py arg\nerror: bad arg",
            ),
        )

        assert isinstance(finding, LoopFinding)
        assert finding.pattern == "no_progress"


class TestLoopDetectorFailedToolRetry:
    def test_similar_failed_verification_variants_trip_retry_finding(self) -> None:
        det = LoopDetector(repeat_threshold=3, no_progress_threshold=10, window=10)

        det.observe(
            _call("verify_work", {"command": "python app.py one"}),
            _error_result(
                "verify_work",
                "FAILED (exit 2)\nfirst stdout\nusage: app.py phrase\nerror: missing",
            ),
        )
        det.observe(
            _call("shell", {"command": "python app.py two"}),
            _error_result(
                "shell",
                "exit_code: 2\nstdout: partial\nstderr: usage: app.py phrase\nerror: missing",
            ),
        )
        finding = det.observe(
            _call("shell", {"command": "python app.py three"}),
            _error_result(
                "shell",
                "exit_code: 2\nstdout: other\nstderr: usage: app.py phrase\nerror: missing",
            ),
        )

        assert isinstance(finding, LoopFinding)
        assert finding.pattern == "failed_tool_retry"
        assert finding.repeats == 3
        assert "same underlying error" in finding.directive
        assert "continue autonomously" in finding.directive
        assert "explain the blocker and stop" not in finding.directive


class TestLoopDetectorStaleEdit:
    def test_repeated_old_not_found_edits_trip_even_with_rereads(self) -> None:
        det = LoopDetector(repeat_threshold=10, no_progress_threshold=10, window=12)
        edit = _call("edit_file", {"path": "parser/parser.go.y", "old": "x", "new": "y"})
        assert det.observe(edit, _error_result()) is None
        assert det.observe(_call("read_file", {"path": "parser/parser.go.y"})) is None
        assert det.observe(edit, _error_result()) is None
        assert det.observe(_call("read_file", {"path": "parser/parser.go.y"})) is None

        finding = det.observe(edit, _error_result())

        assert isinstance(finding, LoopFinding)
        assert finding.pattern == "stale_edit_retry"
        assert finding.tool == "edit_file"
        assert finding.repeats == 3
        assert "old text is not present" in finding.directive

    def test_stale_edit_detection_is_per_path(self) -> None:
        det = LoopDetector(repeat_threshold=10, no_progress_threshold=10, window=12)
        for path in ["a.py", "b.py", "a.py", "b.py"]:
            det.observe(_call("edit_file", {"path": path}), _error_result())

        finding = det.observe(_call("edit_file", {"path": "a.py"}), _error_result())

        assert finding is not None
        assert finding.pattern == "stale_edit_retry"


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
