"""Tests for ClaimGroundingVerifier, StateVerifier, ConsensusVerifier,
and the require_tool_use RunRequest flag."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest

from harness.core.activity import ActivityEvent
from harness.core.schemas import Message, RunRequest, Session
from harness.core.verification import (
    ClaimGroundingVerifier,
    ConsensusVerifier,
    StateVerifier,
)

from .conftest import MockAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(kind: str, data: dict) -> ActivityEvent:
    return ActivityEvent(
        id=uuid.uuid4().hex,
        task_id="t1",
        session_id="s1",
        timestamp=datetime.now(UTC),
        kind=kind,
        data=data,
    )


def _make_session(final_text: str) -> Session:
    return Session(
        id="s1",
        provider="mock",
        model="test",
        cwd=Path.cwd(),
        task_id=None,
        messages=[
            Message(role="user", content="user prompt"),
            Message(role="assistant", content=final_text),
        ],
    )


def _completed_event(
    name: str, content_preview: str = "", arguments: dict | None = None
) -> ActivityEvent:
    return _make_event(
        "tool_call.completed",
        {
            "name": name,
            "is_error": False,
            "content_preview": content_preview,
            "arguments": arguments or {},
        },
    )


# ---------------------------------------------------------------------------
# ClaimGroundingVerifier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestClaimGroundingVerifier:
    async def test_grounding_passes_when_number_in_preview(self) -> None:
        event = _completed_event("list_dir", content_preview="Found 42 files in directory")
        session = _make_session("There are 42 Python files in the project.")
        result = await ClaimGroundingVerifier().verify(session=session, activity=[event])
        assert result.can_finish is True
        assert result.confidence == pytest.approx(0.85)

    async def test_grounding_fails_when_number_not_in_preview(self) -> None:
        event = _completed_event("list_dir", content_preview="Found 10 files in directory")
        session = _make_session("There are 42 Python files in the project.")
        result = await ClaimGroundingVerifier().verify(session=session, activity=[event])
        assert result.can_finish is False
        assert result.confidence == pytest.approx(0.75)
        assert "42" in result.reason

    async def test_grounding_passes_no_tool_events(self) -> None:
        session = _make_session("There are 42 Python files in the project.")
        result = await ClaimGroundingVerifier().verify(session=session, activity=[])
        assert result.can_finish is True
        assert result.confidence == pytest.approx(0.4)

    async def test_grounding_write_claim_backed_by_event(self) -> None:
        event = _completed_event(
            "write_file",
            content_preview="wrote file",
            arguments={"path": "fib_demo.py"},
        )
        session = _make_session("I wrote to fib_demo.py with the implementation.")
        result = await ClaimGroundingVerifier().verify(session=session, activity=[event])
        assert result.can_finish is True
        assert result.confidence == pytest.approx(0.85)

    async def test_grounding_write_claim_no_backing_event(self) -> None:
        event = _completed_event("list_dir", content_preview="some output")
        session = _make_session("I saved to foo.py the new implementation.")
        result = await ClaimGroundingVerifier().verify(session=session, activity=[event])
        assert result.can_finish is False
        assert "foo.py" in result.reason


# ---------------------------------------------------------------------------
# StateVerifier
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStateVerifier:
    async def test_state_written_file_exists(self, tmp_path: Path) -> None:
        target = tmp_path / "output.py"
        target.write_text("# hello\n")
        event = _completed_event(
            "write_file",
            content_preview="wrote file",
            arguments={"path": str(target)},
        )
        session = _make_session("Done.")
        result = await StateVerifier(cwd=tmp_path).verify(session=session, activity=[event])
        assert result.can_finish is True
        assert result.confidence == pytest.approx(0.9)

    async def test_state_written_file_missing(self, tmp_path: Path) -> None:
        missing = tmp_path / "ghost.py"
        event = _completed_event(
            "write_file",
            content_preview="wrote file",
            arguments={"path": str(missing)},
        )
        session = _make_session("Done.")
        result = await StateVerifier(cwd=tmp_path).verify(session=session, activity=[event])
        assert result.can_finish is False
        assert result.confidence == pytest.approx(0.9)
        assert "ghost.py" in result.reason

    async def test_state_no_events(self) -> None:
        session = _make_session("Done.")
        result = await StateVerifier().verify(session=session, activity=[])
        assert result.can_finish is True
        assert result.confidence == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# ConsensusVerifier
# ---------------------------------------------------------------------------


def _consensus_response(payload: dict | str) -> list:
    """Build a mock adapter script returning one text message + Done."""
    from harness.core import Done, TextDelta

    body = payload if isinstance(payload, str) else json.dumps(payload)
    return [
        TextDelta(text=body),
        Done(final_message=Message(role="assistant", content=body)),
    ]


@pytest.mark.asyncio
class TestConsensusVerifier:
    async def test_consensus_agrees(self) -> None:
        adapter = MockAdapter(
            "consensus",
            scripts=[_consensus_response({"agrees": True, "reason": "correct", "confidence": 0.9})],
        )
        verifier = ConsensusVerifier(adapter=adapter, model="judge-m")
        session = _make_session("The answer is 42.")
        result = await verifier.verify(session=session, activity=[])
        assert result.can_finish is True
        assert result.confidence == pytest.approx(0.9)
        assert "correct" in result.reason

    async def test_consensus_disagrees(self) -> None:
        adapter = MockAdapter(
            "consensus",
            scripts=[
                _consensus_response({"agrees": False, "reason": "count wrong", "confidence": 0.85})
            ],
        )
        verifier = ConsensusVerifier(adapter=adapter, model="judge-m")
        session = _make_session("There are 999 files in the repo.")
        result = await verifier.verify(session=session, activity=[])
        assert result.can_finish is False
        assert result.confidence == pytest.approx(0.85)
        assert "count wrong" in result.reason


# ---------------------------------------------------------------------------
# require_tool_use flag on RunRequest
# ---------------------------------------------------------------------------


class TestRequireToolUseFlag:
    def test_require_tool_use_flag_on_run_request(self) -> None:
        req = RunRequest(prompt="x", require_tool_use=True)
        assert req.require_tool_use is True

    def test_require_tool_use_defaults_false(self) -> None:
        req = RunRequest(prompt="x")
        assert req.require_tool_use is False
