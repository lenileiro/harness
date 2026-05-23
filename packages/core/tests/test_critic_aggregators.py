"""Tests for the pluggable MultiCritic aggregators."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from harness.core import Session
from harness.core.critic import (
    Critic,
    MultiCritic,
    make_majority,
    make_unanimity,
)
from harness.core.schemas import VerificationResult


class _FixedCritic:
    """Returns a fixed string. Empty string = critic punts.

    Implements the Critic Protocol structurally — duck-typed at runtime,
    cast at call sites for pyright's invariant generic check.
    """

    def __init__(self, text: str) -> None:
        self._text = text

    async def critique(self, *, session, verification_result, activity) -> str:
        return self._text


def _critics(*pairs: tuple[str, _FixedCritic]) -> list[tuple[str, Critic]]:
    return cast(list[tuple[str, Critic]], list(pairs))


def _make_session() -> Session:
    return Session(provider="x", model="y", cwd=Path("/tmp"))


def _vr() -> VerificationResult:
    return VerificationResult(verifier_name="test", can_finish=False, reason="boom")


async def _run(mc: MultiCritic) -> str:
    return await mc.critique(
        session=_make_session(),
        verification_result=_vr(),
        activity=[],
    )


class TestConcat:
    async def test_emits_every_non_empty_critic(self) -> None:
        mc = MultiCritic(critics=_critics(("A", _FixedCritic("hi")), ("B", _FixedCritic("there"))))
        out = await _run(mc)
        assert "hi" in out
        assert "there" in out

    async def test_empty_critic_skipped(self) -> None:
        mc = MultiCritic(critics=_critics(("A", _FixedCritic("hi")), ("B", _FixedCritic(""))))
        out = await _run(mc)
        assert "hi" in out
        assert "B" not in out  # label not emitted when text empty

    async def test_legacy_positional_constructor_still_works(self) -> None:
        mc = MultiCritic(_FixedCritic("primary-text"), _FixedCritic("devil-text"))
        out = await _run(mc)
        assert "primary-text" in out
        assert "devil-text" in out


class TestMajority:
    async def test_below_quorum_returns_empty(self) -> None:
        critics = _critics(
            ("A", _FixedCritic("yes")),
            ("B", _FixedCritic("")),
            ("C", _FixedCritic("")),
        )
        mc = MultiCritic(critics=critics, aggregator=make_majority(len(critics)))
        # 1 of 3 spoke; quorum=0.5 → threshold = 1*0.5 = 0 → int(0.5*3) = 1, +1 = 2
        out = await _run(mc)
        assert out == ""

    async def test_at_quorum_returns_concat(self) -> None:
        critics = _critics(
            ("A", _FixedCritic("a")),
            ("B", _FixedCritic("b")),
            ("C", _FixedCritic("")),
        )
        mc = MultiCritic(critics=critics, aggregator=make_majority(len(critics)))
        out = await _run(mc)
        assert "a" in out and "b" in out


class TestUnanimity:
    async def test_one_silent_blocks_emission(self) -> None:
        critics = _critics(
            ("A", _FixedCritic("a")),
            ("B", _FixedCritic("")),
        )
        mc = MultiCritic(critics=critics, aggregator=make_unanimity(len(critics)))
        assert await _run(mc) == ""

    async def test_all_speak_emits(self) -> None:
        critics = _critics(
            ("A", _FixedCritic("a")),
            ("B", _FixedCritic("b")),
        )
        mc = MultiCritic(critics=critics, aggregator=make_unanimity(len(critics)))
        out = await _run(mc)
        assert "a" in out and "b" in out


class TestApproval:
    async def test_majority_approve_forwards(self) -> None:
        critics = _critics(
            ("A", _FixedCritic("[APPROVE] good")),
            ("B", _FixedCritic("[APPROVE] sure")),
            ("C", _FixedCritic("[REJECT] no")),
        )
        mc = MultiCritic(critics=critics, aggregator="approval")
        out = await _run(mc)
        assert "good" in out
        # Markers should be stripped from forwarded text.
        assert "[APPROVE]" not in out

    async def test_majority_reject_silences(self) -> None:
        critics = _critics(
            ("A", _FixedCritic("[REJECT] no")),
            ("B", _FixedCritic("[REJECT] never")),
            ("C", _FixedCritic("[APPROVE] maybe")),
        )
        mc = MultiCritic(critics=critics, aggregator="approval")
        assert await _run(mc) == ""

    async def test_no_markers_falls_back_to_concat(self) -> None:
        critics = _critics(
            ("A", _FixedCritic("plain text")),
            ("B", _FixedCritic("more")),
        )
        mc = MultiCritic(critics=critics, aggregator="approval")
        out = await _run(mc)
        assert "plain text" in out and "more" in out


class TestUnknownAggregator:
    def test_string_aggregator_must_be_registered(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            MultiCritic(critics=_critics(("A", _FixedCritic("x"))), aggregator="not-a-thing")
