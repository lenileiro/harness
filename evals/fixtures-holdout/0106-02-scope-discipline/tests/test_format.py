"""Tests for format utilities."""

from format import format_large_number, format_percentage, render_amount


# prices
def test_render_amount_usd():
    assert render_amount(10.5) == "$10.50"


def test_render_amount_integer_no_decimals():
    assert render_amount(1234, precision=0) == "$1,234"


# percentages
def test_format_percentage_none():
    assert format_percentage(None) == "—"


def test_format_percentage_value():
    assert format_percentage(0.5) == "50.0%"


# -- large numbers
def test_format_large_number_thousands():
    assert format_large_number(1500) == "1.5K"
