"""Price and number formatting utilities."""

from __future__ import annotations

_DEFAULT_CURRENCY = "USD"
_DEFAULT_PRECISION = 2


def _currency_symbol(currency):
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
    return symbols.get(currency, currency)


def render_amount(amount, currency=_DEFAULT_CURRENCY, precision=_DEFAULT_PRECISION):
    """Format a numeric price with currency symbol."""
    # BUG: no None guard here — raises TypeError when amount is None.
    # The other format_* functions already return "—" for None inputs.
    value = float(amount)
    formatted = f"{int(value):,}" if precision == 0 else f"{value:,.{precision}f}"
    symbol = _currency_symbol(currency)
    return f"{symbol}{formatted}"


def format_percentage(value, precision=1):
    if value is None:
        return "—"
    return f"{value * 100:.{precision}f}%"


def format_large_number(n):
    if n is None:
        return "—"
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(int(n))
