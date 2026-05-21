"""Tests for enhanced RichApprovalHandler UX."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from harness.cli.approval import RichApprovalHandler
from harness.core import Session, ToolCall


def _console() -> tuple[Console, StringIO]:
    buf = StringIO()
    con = Console(file=buf, highlight=False, markup=True, no_color=True, width=120)
    return con, buf


def _session() -> Session:
    return Session(id="s1", provider="mock", model="m", cwd=Path.cwd())


def _tool(name: str, description: str = "A tool.") -> MagicMock:
    t = MagicMock()
    t.name = name
    t.description = description
    return t


def _call(**arguments: object) -> ToolCall:
    return ToolCall(id="c1", name="shell", arguments=arguments)


# ---------------------------------------------------------------------------
# Risk styling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRiskStyling:
    async def test_shell_prompt_contains_warning_icon(self) -> None:
        con, buf = _console()
        handler = RichApprovalHandler(console=con)
        with (
            patch("harness.cli.approval.Prompt.ask", return_value="n"),
            patch.object(sys.stdin, "isatty", return_value=True),
        ):
            await handler(_tool("shell"), _call(command="rm -rf /"), _session())
        output = buf.getvalue()
        assert "⚠" in output

    async def test_read_file_prompt_contains_circle_icon(self) -> None:
        con, buf = _console()
        handler = RichApprovalHandler(console=con)
        with (
            patch("harness.cli.approval.Prompt.ask", return_value="n"),
            patch.object(sys.stdin, "isatty", return_value=True),
        ):
            await handler(_tool("read_file"), _call(path="foo.txt"), _session())
        output = buf.getvalue()
        assert "○" in output

    async def test_write_file_shows_warning(self) -> None:
        con, buf = _console()
        handler = RichApprovalHandler(console=con)
        with (
            patch("harness.cli.approval.Prompt.ask", return_value="n"),
            patch.object(sys.stdin, "isatty", return_value=True),
        ):
            await handler(_tool("write_file"), _call(path="out.txt", content="x"), _session())
        assert "⚠" in buf.getvalue()


# ---------------------------------------------------------------------------
# Full args display
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_args_shown_no_truncation() -> None:
    con, buf = _console()
    handler = RichApprovalHandler(console=con)
    long_arg = "x" * 300
    with (
        patch("harness.cli.approval.Prompt.ask", return_value="n"),
        patch.object(sys.stdin, "isatty", return_value=True),
    ):
        await handler(_tool("shell"), _call(command=long_arg), _session())
    # Rich wraps long content across lines; strip newlines to check it's fully present.
    output = buf.getvalue().replace("\n", "")
    assert "x" * 100 in output  # At least 100 chars of the arg rendered (not truncated)


# ---------------------------------------------------------------------------
# Session trust (s choice)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_s_choice_populates_session_overrides() -> None:
    con, _ = _console()
    overrides: dict = {}
    handler = RichApprovalHandler(console=con, session_overrides=overrides)
    with (
        patch("harness.cli.approval.Prompt.ask", return_value="s"),
        patch.object(sys.stdin, "isatty", return_value=True),
    ):
        result = await handler(_tool("shell"), _call(command="echo hi"), _session())
    assert result is True
    assert overrides.get("shell") == "auto"


@pytest.mark.asyncio
async def test_session_trust_auto_approves_subsequent_calls() -> None:
    con, _ = _console()
    overrides: dict = {"shell": "auto"}
    handler = RichApprovalHandler(console=con, session_overrides=overrides)
    # Should return True without prompting at all.
    with (
        patch("harness.cli.approval.Prompt.ask", side_effect=AssertionError("should not prompt")),
        patch.object(sys.stdin, "isatty", return_value=True),
    ):
        result = await handler(_tool("shell"), _call(command="echo"), _session())
    assert result is True


# ---------------------------------------------------------------------------
# Standard choices
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStandardChoices:
    async def test_y_returns_true(self) -> None:
        con, _ = _console()
        handler = RichApprovalHandler(console=con)
        with (
            patch("harness.cli.approval.Prompt.ask", return_value="y"),
            patch.object(sys.stdin, "isatty", return_value=True),
        ):
            assert await handler(_tool("shell"), _call(), _session()) is True

    async def test_n_returns_false(self) -> None:
        con, _ = _console()
        handler = RichApprovalHandler(console=con)
        with (
            patch("harness.cli.approval.Prompt.ask", return_value="n"),
            patch.object(sys.stdin, "isatty", return_value=True),
        ):
            assert await handler(_tool("shell"), _call(), _session()) is False

    async def test_a_writes_auto_to_session_overrides(self) -> None:
        con, _ = _console()
        handler = RichApprovalHandler(console=con)
        session = _session()
        with (
            patch("harness.cli.approval.Prompt.ask", return_value="a"),
            patch.object(sys.stdin, "isatty", return_value=True),
        ):
            result = await handler(_tool("shell"), _call(), session)
        assert result is True
        assert session.approval_overrides.get("shell") == "auto"

    async def test_d_writes_deny_to_session_overrides(self) -> None:
        con, _ = _console()
        handler = RichApprovalHandler(console=con)
        session = _session()
        with (
            patch("harness.cli.approval.Prompt.ask", return_value="d"),
            patch.object(sys.stdin, "isatty", return_value=True),
        ):
            result = await handler(_tool("shell"), _call(), session)
        assert result is False
        assert session.approval_overrides.get("shell") == "deny"

    async def test_non_tty_returns_false_without_prompting(self) -> None:
        con, buf = _console()
        handler = RichApprovalHandler(console=con)
        with patch.object(sys.stdin, "isatty", return_value=False):
            result = await handler(_tool("shell"), _call(), _session())
        assert result is False
        assert "not a TTY" in buf.getvalue()
