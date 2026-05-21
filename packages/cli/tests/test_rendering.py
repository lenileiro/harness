"""Tests for the Renderer class in __main__.py."""

from __future__ import annotations

from io import StringIO

from rich.console import Console

from harness.cli.__main__ import Renderer, _preprocess_markdown
from harness.core import (
    Done,
    ErrorEvent,
    StepCompleted,
    StepStarted,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    ToolResultEvent,
    Usage,
    Verification,
    VerificationResult,
)


def _console() -> tuple[Console, StringIO]:
    buf = StringIO()
    con = Console(file=buf, highlight=False, markup=True, no_color=True, width=120)
    return con, buf


def _make_call(name: str = "read_file", **args: object) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=dict(args))


def _make_result(
    name: str = "read_file", content: str = "ok", is_error: bool = False
) -> ToolResult:
    return ToolResult(tool_call_id="c1", name=name, content=content, is_error=is_error)


# ---------------------------------------------------------------------------
# Basic event rendering
# ---------------------------------------------------------------------------


class TestRendererBasics:
    def test_text_delta_rendered_as_markdown_on_flush(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(TextDelta(text="hello "))
        r.render(TextDelta(text="world"))
        # Text is buffered; flush by sending Done
        r.render(Done())
        output = buf.getvalue()
        assert "hello" in output
        assert "world" in output

    def test_text_delta_markdown_code_block_rendered(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(TextDelta(text="```python\nprint('hi')\n```"))
        r.render(Done())
        output = buf.getvalue()
        # Rich renders the code block — raw backticks should not appear
        assert "print" in output

    def test_tool_call_prints_arrow_and_name(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(ToolCallEvent(call=_make_call("shell", command="echo hi")))
        r.render(ToolResultEvent(result=_make_result("shell", "hi")))
        output = buf.getvalue()
        assert "shell" in output
        assert "→" in output

    def test_tool_result_success_shows_checkmark(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(ToolCallEvent(call=_make_call()))
        r.render(ToolResultEvent(result=_make_result("read_file", "file contents")))
        output = buf.getvalue()
        assert "✓" in output
        assert "read_file" in output

    def test_tool_result_error_shows_x(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(ToolCallEvent(call=_make_call()))
        r.render(ToolResultEvent(result=_make_result(is_error=True, content="not found")))
        output = buf.getvalue()
        assert "✗" in output

    def test_tool_result_shows_elapsed(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(ToolCallEvent(call=_make_call()))
        r.render(ToolResultEvent(result=_make_result()))
        output = buf.getvalue()
        assert "s)" in output  # e.g. "(0.0s)"

    def test_large_result_shows_byte_count(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        long_content = "x" * 500
        r.render(ToolCallEvent(call=_make_call()))
        r.render(ToolResultEvent(result=_make_result(content=long_content)))
        output = buf.getvalue()
        assert "bytes" in output

    def test_short_result_no_byte_hint(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(ToolCallEvent(call=_make_call()))
        r.render(ToolResultEvent(result=_make_result(content="hi")))
        output = buf.getvalue()
        assert "bytes" not in output

    def test_error_event_prints_kind_and_message(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(ErrorEvent(error="boom", kind="NetworkError"))
        output = buf.getvalue()
        assert "NetworkError" in output
        assert "boom" in output


# ---------------------------------------------------------------------------
# Step counter
# ---------------------------------------------------------------------------


class TestStepCounter:
    def test_step_header_shown_when_total_steps_gt_1(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(StepStarted(step=0, description="gather context", total_steps=3))
        output = buf.getvalue()
        assert "Step 1/3" in output
        assert "gather context" in output

    def test_step_header_not_shown_when_total_steps_zero(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(StepStarted(step=0, description="something", total_steps=0))
        assert buf.getvalue() == ""

    def test_step_header_not_shown_when_total_steps_one(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(StepStarted(step=0, total_steps=1))
        assert buf.getvalue() == ""

    def test_step_completed_is_silent(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(StepCompleted(step=0))
        assert buf.getvalue() == ""


# ---------------------------------------------------------------------------
# Token usage
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_done_with_usage_shows_token_line(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(Done(usage=Usage(prompt_tokens=1234, completion_tokens=89)))
        output = buf.getvalue()
        assert "1,234" in output
        assert "89" in output

    def test_done_without_usage_no_token_line(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(Done())
        assert "tokens" not in buf.getvalue()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class TestVerification:
    def test_can_finish_shows_green_check(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(
            Verification(
                result=VerificationResult(
                    can_finish=True,
                    reason="all good",
                    verifier_name="rule",
                    confidence=None,
                )
            )
        )
        output = buf.getvalue()
        assert "✓" in output
        assert "all good" in output

    def test_cannot_finish_shows_x(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(
            Verification(
                result=VerificationResult(
                    can_finish=False,
                    reason="tool failed",
                    verifier_name="rule",
                    confidence=None,
                )
            )
        )
        output = buf.getvalue()
        assert "✗" in output

    def test_confidence_shown_when_present(self) -> None:
        con, buf = _console()
        r = Renderer(con)
        r.render(
            Verification(
                result=VerificationResult(
                    can_finish=True,
                    reason="ok",
                    verifier_name="llm",
                    confidence=0.95,
                )
            )
        )
        assert "0.95" in buf.getvalue()


# ---------------------------------------------------------------------------
# Markdown preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessMarkdown:
    def test_rightarrow_converted(self) -> None:
        result = _preprocess_markdown("Think $\\rightarrow$ Act")
        assert "→" in result
        assert "$" not in result

    def test_multiple_arrows_in_sequence(self) -> None:
        result = _preprocess_markdown(
            "Think $\\rightarrow$ Tool Call $\\rightarrow$ Observe $\\rightarrow$ Repeat"
        )
        assert result.count("→") == 3
        assert "$" not in result

    def test_display_math_converted(self) -> None:
        result = _preprocess_markdown("$$\\alpha + \\beta = \\gamma$$")
        assert "α" in result  # noqa: RUF001
        assert "β" in result
        assert "γ" in result  # noqa: RUF001
        assert "$" not in result

    def test_plain_text_unchanged(self) -> None:
        text = "Here is a code block:\n\n```python\nprint('hi')\n```"
        assert _preprocess_markdown(text) == text

    def test_think_block_collapsed(self) -> None:
        result = _preprocess_markdown("<think>reasoning goes here</think>Answer")
        assert "reasoning" in result
        assert "<think>" not in result

    def test_unknown_latex_command_stripped(self) -> None:
        result = _preprocess_markdown("$\\unknowncmd{x}$")
        assert "$" not in result
        assert "\\unknown" not in result

    def test_leq_geq_converted(self) -> None:
        result = _preprocess_markdown("If $x \\leq y$ and $y \\geq z$")
        assert "≤" in result
        assert "≥" in result
