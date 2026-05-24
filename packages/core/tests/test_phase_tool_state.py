"""Regression tests for PhaseTool state derivation + feature-add bypass.

These cover two production bugs discovered during cross-model A/B
validation (research-borrow run, 2026-05-23):

  1. PhaseTool's `_derive_state` returned empty when `session_id=None`.
     The CLI registers the tool before a session exists, so every
     declare/complete looked like the first one and emitted spurious
     "[WARNING] no phase was in flight" messages. Qwen3-coder treated
     the warnings as actionable errors and burned through max_steps
     trying to "fix" them.

  2. `TestsBeforeEditVerifier` was firing on feature-add tasks where
     no failing test exists to reproduce. Fixed by detecting the
     first-line task verb (Add/Implement/Create vs Fix/Debug/Handle).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.core import PhaseTool, Session
from harness.core.schemas import Message, ToolCall
from harness.core.verification import (
    TestsBeforeEditVerifier as _TestsBeforeEditVerifier,
)
from harness.core.verification import (
    _looks_like_feature_add,
)


def _call(action: str, name: str = "", call_id: str = "c1") -> ToolCall:
    args: dict[str, object] = {"action": action}
    if name:
        args["name"] = name
    return ToolCall(id=call_id, name="phase", arguments=args)


# ---------------------------------------------------------------------------
# Bug 1: PhaseTool state derivation without session_id
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestPhaseToolStateWithoutSessionId:
    async def test_declare_then_complete_no_warnings(self) -> None:
        """The CLI wires PhaseTool with activity_store but session_id=None.
        State must still carry through declare → complete on the same
        instance — otherwise the agent gets a spurious 'no phase was in
        flight' warning and may loop trying to 'fix' it.
        """
        tool = PhaseTool()  # no session_id, no activity_store
        result_declare = await tool(_call("declare", "implement", call_id="c1"))
        assert not result_declare.is_error
        assert "[WARNING]" not in (result_declare.content or "")

        result_complete = await tool(_call("complete", "implement", call_id="c2"))
        assert not result_complete.is_error
        assert "[WARNING]" not in (result_complete.content or "")

    async def test_complete_already_completed_phase_is_noop(self) -> None:
        tool = PhaseTool()
        await tool(_call("declare", "x", call_id="c1"))
        await tool(_call("complete", "x", call_id="c2"))
        second_complete = await tool(_call("complete", "x", call_id="c3"))
        assert not second_complete.is_error
        # Re-complete should emit an INFO, not a WARNING.
        content = second_complete.content or ""
        assert "[INFO]" in content
        assert "[WARNING]" not in content

    async def test_redeclare_already_declared_phase_is_noop(self) -> None:
        tool = PhaseTool()
        await tool(_call("declare", "x", call_id="c1"))
        second_declare = await tool(_call("declare", "x", call_id="c2"))
        content = second_declare.content or ""
        assert "[INFO]" in content
        assert "[WARNING]" not in content

    async def test_multi_phase_sequence(self) -> None:
        """A four-phase task: implement, test, document, verify.
        Each declare→complete pair should be clean."""
        tool = PhaseTool()
        for phase in ("implement", "test", "document", "verify"):
            d = await tool(_call("declare", phase, call_id=f"d_{phase}"))
            c = await tool(_call("complete", phase, call_id=f"c_{phase}"))
            assert "[WARNING]" not in (d.content or "")
            assert "[WARNING]" not in (c.content or "")


# ---------------------------------------------------------------------------
# Bug 2: TestsBeforeEditVerifier on feature-add tasks
# ---------------------------------------------------------------------------


class TestFeatureAddDetector:
    def test_add_header_is_feature(self) -> None:
        prompt = "# Add `power` to the calculator\n\nOur Calculator class..."
        assert _looks_like_feature_add(prompt)

    def test_implement_header_is_feature(self) -> None:
        assert _looks_like_feature_add("# Implement payment retry logic\n...")

    def test_fix_header_is_not_feature(self) -> None:
        prompt = "# Fix hyphenated user ID lookup\n\nget_user() returns None..."
        assert not _looks_like_feature_add(prompt)

    def test_body_text_with_fix_doesnt_flip_feature_to_bug(self) -> None:
        """The classic F04 regression: TASK.md body says 'Do not fix them'
        about pre-existing noise. That should NOT cause the heading-add
        prompt to be classified as a bug fix."""
        prompt = (
            "# Add `power` to the calculator\n\n"
            "The codebase has pre-existing issues (typos, unused imports). "
            "**Do not fix them.** They're tracked separately."
        )
        assert _looks_like_feature_add(prompt)

    def test_empty_prompt(self) -> None:
        assert not _looks_like_feature_add("")

    def test_no_recognized_verb(self) -> None:
        assert not _looks_like_feature_add("# Make the thing better\n\n...")


@pytest.mark.asyncio
class TestTestsBeforeEditBypass:
    async def test_feature_add_bypasses(self) -> None:
        """Edit before verify_work on a feature-add task should not block."""
        verifier = _TestsBeforeEditVerifier()
        session = Session(provider="x", model="y", cwd=Path("/tmp"))
        session.messages.append(
            Message(role="user", content="# Add power to the calculator\n\nAdd a power method.")
        )
        # Synthesize an activity log with edit-then-verify ordering.
        from harness.core.activity import ActivityEvent

        activity = [
            ActivityEvent(
                kind="tool_call.completed",
                data={"name": "edit_file", "is_error": False, "content_preview": "edited"},
            ),
            ActivityEvent(
                kind="tool_call.completed",
                data={
                    "name": "verify_work",
                    "is_error": False,
                    "content_preview": "all tests pass",
                },
            ),
        ]
        result = await verifier.verify(session=session, activity=activity)
        assert result.can_finish
        assert "feature-add" in result.reason

    async def test_bug_fix_still_blocks(self) -> None:
        """The verifier should still fire on bug-fix tasks that edit before verify."""
        verifier = _TestsBeforeEditVerifier()
        session = Session(provider="x", model="y", cwd=Path("/tmp"))
        session.messages.append(
            Message(role="user", content="# Fix the hyphenated user lookup bug\n\nget_user fails.")
        )
        from harness.core.activity import ActivityEvent

        activity = [
            ActivityEvent(
                kind="tool_call.completed",
                data={"name": "edit_file", "is_error": False, "content_preview": "edited"},
            ),
            ActivityEvent(
                kind="tool_call.completed",
                data={"name": "verify_work", "is_error": False, "content_preview": "tests pass"},
            ),
        ]
        result = await verifier.verify(session=session, activity=activity)
        assert not result.can_finish

    async def test_shell_pytest_before_edit_counts_as_pre_edit_test_run(self) -> None:
        verifier = _TestsBeforeEditVerifier()
        session = Session(provider="x", model="y", cwd=Path("/tmp"))
        session.messages.append(
            Message(role="user", content="# Fix the format bug\n\nformat_price(None) fails.")
        )
        from harness.core.activity import ActivityEvent

        activity = [
            ActivityEvent(
                kind="tool_call.completed",
                data={
                    "name": "shell",
                    "is_error": True,
                    "arguments": {"command": "pytest tests/test_format.py -q"},
                    "content_preview": (
                        "FAILED tests/test_format.py::test_format_price_none - TypeError"
                    ),
                },
            ),
            ActivityEvent(
                kind="tool_call.completed",
                data={"name": "edit_file", "is_error": False, "content_preview": "edited"},
            ),
            ActivityEvent(
                kind="tool_call.completed",
                data={"name": "verify_work", "is_error": False, "content_preview": "tests pass"},
            ),
        ]
        result = await verifier.verify(session=session, activity=activity)
        assert result.can_finish
        assert "before the first edit" in result.reason

    async def test_non_test_shell_before_edit_does_not_bypass(self) -> None:
        verifier = _TestsBeforeEditVerifier()
        session = Session(provider="x", model="y", cwd=Path("/tmp"))
        session.messages.append(
            Message(role="user", content="# Fix the format bug\n\nformat_price(None) fails.")
        )
        from harness.core.activity import ActivityEvent

        activity = [
            ActivityEvent(
                kind="tool_call.completed",
                data={
                    "name": "shell",
                    "is_error": False,
                    "arguments": {"command": "find . -maxdepth 2"},
                    "content_preview": "./src\n./tests",
                },
            ),
            ActivityEvent(
                kind="tool_call.completed",
                data={"name": "edit_file", "is_error": False, "content_preview": "edited"},
            ),
            ActivityEvent(
                kind="tool_call.completed",
                data={"name": "verify_work", "is_error": False, "content_preview": "tests pass"},
            ),
        ]
        result = await verifier.verify(session=session, activity=activity)
        assert not result.can_finish
