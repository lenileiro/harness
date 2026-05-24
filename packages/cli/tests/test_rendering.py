"""Tests for the Renderer class in __main__.py."""

from __future__ import annotations

from io import StringIO

import unicodeitplus
from rich.console import Console

from harness.cli.markdown_render import Renderer, _preprocess_markdown, _render_mermaid
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
        # unicodeitplus uses italic math letters, not plain greek — check via library directly
        alpha = unicodeitplus.replace("\\alpha")
        beta = unicodeitplus.replace("\\beta")
        result = _preprocess_markdown("$$\\alpha + \\beta$$")
        assert alpha in result
        assert beta in result
        assert "$$" not in result

    def test_plain_text_unchanged(self) -> None:
        text = "Here is a code block:\n\n```python\nprint('hi')\n```"
        assert _preprocess_markdown(text) == text

    def test_think_block_collapsed(self) -> None:
        result = _preprocess_markdown("<think>reasoning goes here</think>Answer")
        assert "reasoning" in result
        assert "<think>" not in result

    def test_subscript_superscript_converted(self) -> None:
        # unicodeitplus handles _{...}^{...} as Unicode sub/superscripts
        result = _preprocess_markdown("$\\sum_{i=0}^{n}$")
        assert "∑" in result
        assert "$" not in result

    def test_leq_geq_converted(self) -> None:
        result = _preprocess_markdown("If $x \\leq y$ and $y \\geq z$")
        assert "≤" in result
        assert "≥" in result

    def test_mermaid_fence_replaced_when_complete(self) -> None:
        src = "graph LR\n    A --> B"
        text = f"```mermaid\n{src}\n```"
        result = _preprocess_markdown(text)
        # Fenced block replaced — the raw ```mermaid marker should be gone
        assert "```mermaid" not in result
        # Result contains a plain code block wrapping either ascii art or the source
        assert "```" in result

    def test_mermaid_fence_not_replaced_when_incomplete(self) -> None:
        # Opening fence without closing — should not be touched
        text = "```mermaid\ngraph LR\n    A --> B"
        result = _preprocess_markdown(text)
        assert "```mermaid" in result

    def test_mermaid_non_fence_code_block_unchanged(self) -> None:
        text = "```python\nprint('hi')\n```"
        assert _preprocess_markdown(text) == text


# ---------------------------------------------------------------------------
# Mermaid rendering
# ---------------------------------------------------------------------------


class TestRenderMermaid:
    def test_fallback_when_not_installed(self) -> None:
        import sys
        from unittest.mock import patch

        src = "graph LR\n    FALLBACK_A --> FALLBACK_B"
        with patch.dict(sys.modules, {"mermaid_ascii": None}):
            result = _render_mermaid(src)
        # Should fall back to a plain code block containing the source
        assert "FALLBACK_A" in result
        assert "```" in result

    def test_renders_ascii_when_available(self) -> None:
        import sys
        from unittest.mock import MagicMock, patch

        fake_module = MagicMock()
        fake_module.mermaid_to_ascii.return_value = "+--ASCII--+"
        src = "graph LR\n    ASCII_A --> ASCII_B"
        with patch.dict(sys.modules, {"mermaid_ascii": fake_module}):
            result = _render_mermaid(src)
        assert "+--ASCII--+" in result
        assert "```" in result

    def test_cache_reuses_result(self) -> None:
        src = "graph LR\n    CACHE_X --> CACHE_Y"
        first = _render_mermaid(src)
        second = _render_mermaid(src)
        assert first is second  # same object from cache
