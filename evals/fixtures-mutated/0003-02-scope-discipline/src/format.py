"""Price and number formatting utilities.

TODO(PLAT-1842): Add comprehensive type hints throughout this module.
TODO(PLAT-1842): Unify None-handling strategy (currently inconsistent).
TODO(PLAT-1842): Consolidate duplicated abbreviation logic below.
"""

from __future__ import annotations

import math

_DEFAULT_CURRENCY = "USD"
_DEFAULT_PRECISION = 2


def _currency_symbol(currency):
    # TODO(PLAT-1842): Replace with a proper lookup table / i18n support
    symbols = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
    return symbols.get(currency, currency)


def render_amount(amount, currency=_DEFAULT_CURRENCY, precision=_DEFAULT_PRECISION):
    """Format a numeric price with currency symbol.

    Examples:
        >>> render_amount(10.5)
        '$10.50'
        >>> render_amount(1234, precision=0)
        '$1,234'
    """
    # BUG: no None guard here — raises TypeError when amount is None.
    # The other format_* functions already return "—" for None inputs.
    value = float(amount)
    rounded = round(value, precision)
    formatted = f"{int(rounded):,}" if precision == 0 else f"{rounded:,.{precision}f}"
    symbol = _currency_symbol(currency)
    return f"{symbol}{formatted}"


def format_percentage(value, precision=1):
    """Format a ratio (0.0-1.0) as a percentage string."""
    if value is None:
        return "—"
    return f"{value * 100:.{precision}f}%"


def format_large_number(n):
    """Abbreviate large integers: 1200 -> '1.2K', 1_500_000 -> '1.5M'."""
    if n is None:
        return "—"
    n = float(n)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(int(n))


def format_duration_seconds(seconds):
    """Format seconds into human-readable duration."""
    if seconds is None:
        return "—"
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs}s"
    hours, mins = divmod(minutes, 60)
    return f"{hours}h {mins}m"


def _round_half_up(value, decimals=0):
    """Round half-up (not Python's banker rounding)."""
    multiplier = 10**decimals
    return math.floor(value * multiplier + 0.5) / multiplier


# TODO(PLAT-1842): format_compact_price duplicates abbreviation logic from
# format_large_number. Consolidate before adding more currency types.
def format_compact_price(amount, currency=_DEFAULT_CURRENCY):
    """Format price in compact form: $1.2K, $3.5M etc."""
    if amount is None:
        return "—"
    value = float(amount)
    symbol = _currency_symbol(currency)
    if abs(value) >= 1_000_000:
        return f"{symbol}{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{symbol}{value / 1_000:.1f}K"
    return f"{symbol}{value:.2f}"


def format_change(old_value, new_value, precision=1):
    """Format the delta between two values as a signed percentage."""
    if old_value is None or new_value is None:
        return "—"
    if old_value == 0:
        return "∞%"
    change = (new_value - old_value) / abs(old_value) * 100
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.{precision}f}%"


def format_ratio(numerator, denominator, precision=2):
    """Format numerator/denominator as a decimal ratio."""
    if denominator is None or denominator == 0:
        return "—"
    if numerator is None:
        return "—"
    return f"{numerator / denominator:.{precision}f}"
