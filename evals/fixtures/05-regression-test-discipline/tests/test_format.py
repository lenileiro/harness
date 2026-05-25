from format import format_compact_price, format_percentage, format_price


def test_format_price_still_handles_none() -> None:
    assert format_price(None) == "—"


def test_format_percentage_still_handles_none() -> None:
    assert format_percentage(None) == "—"


def test_format_compact_price_rounds_thousands() -> None:
    assert format_compact_price(1_250) == "$1.2K"


def test_format_compact_price_rounds_millions() -> None:
    assert format_compact_price(2_500_000) == "$2.5M"
