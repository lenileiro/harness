"""Tests for format utilities."""

from format import (
    format_large_number,
    format_percentage,
    render_amount,
    render_estimate_amount,
    render_refund_amount,
)


def test_render_amount_usd():
    assert render_amount(10.5) == "$10.50"


def test_render_amount_integer_no_decimals():
    assert render_amount(1234, precision=0) == "$1,234"


def test_render_amount_none():
    """render_amount(None) must return em dash, not raise TypeError."""
    result = render_amount(None)
    assert result == "—", f"Expected '—' but got {result!r}"


def test_render_refund_amount_value():
    assert render_refund_amount(10.5) == "-$10.50"


def test_render_estimate_amount_value():
    assert render_estimate_amount(12, precision=0) == "$~12"


def test_format_percentage_none():
    assert format_percentage(None) == "—"


def test_format_percentage_value():
    assert format_percentage(0.5) == "50.0%"


def test_format_large_number_thousands():
    assert format_large_number(1500) == "1.5K"
