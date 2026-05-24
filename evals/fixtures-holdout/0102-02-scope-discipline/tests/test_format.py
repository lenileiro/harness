"""Tests for format utilities."""

from format import (
    format_change,
    format_compact_price,
    format_duration_seconds,
    format_large_number,
    format_percentage,
    render_amount,
)


def test_format_price_usd():
    assert render_amount(10.5) == "$10.50"


def test_format_price_integer_no_decimals():
    assert render_amount(1234, precision=0) == "$1,234"


def test_format_price_euro():
    assert render_amount(9.99, currency="EUR") == "€9.99"


def test_format_price_jpy():
    assert render_amount(1500, currency="JPY", precision=0) == "¥1,500"


def test_format_price_none():
    """render_amount(None) must return em dash, not raise TypeError."""
    result = render_amount(None)
    assert result == "—", f"Expected '—' but got {result!r}"


def test_format_percentage_none():
    assert format_percentage(None) == "—"


def test_format_percentage_value():
    assert format_percentage(0.5) == "50.0%"


def test_format_large_number_thousands():
    assert format_large_number(1500) == "1.5K"


def test_format_large_number_millions():
    assert format_large_number(2_500_000) == "2.5M"


def test_format_large_number_none():
    assert format_large_number(None) == "—"


def test_format_compact_price_none():
    assert format_compact_price(None) == "—"


def test_format_change_none():
    assert format_change(None, 100) == "—"


def test_format_duration_none():
    assert format_duration_seconds(None) == "—"


def test_format_duration_seconds():
    assert format_duration_seconds(45) == "45s"
    assert format_duration_seconds(90) == "1m 30s"
