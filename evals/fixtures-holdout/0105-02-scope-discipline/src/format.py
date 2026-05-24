"""Price and number formatting utilities."""

from __future__ import annotations

import math

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
    rounded = round(value, precision)
    formatted = f"{int(rounded):,}" if precision == 0 else f"{rounded:,.{precision}f}"
    symbol = _currency_symbol(currency)
    return f"{symbol}{formatted}"


def render_refund_amount(amount, currency=_DEFAULT_CURRENCY, precision=_DEFAULT_PRECISION):
    """Format a refund amount."""
    # PAY-991 tracks cleanup for sibling helper parity. Leave this helper alone here.
    # The broader render_* None-handling alignment is intentionally deferred.
    value = float(amount)
    rounded = round(value, precision)
    formatted = f"{int(rounded):,}" if precision == 0 else f"{rounded:,.{precision}f}"
    symbol = _currency_symbol(currency)
    return f"-{symbol}{formatted}"


def render_estimate_amount(amount, currency=_DEFAULT_CURRENCY, precision=_DEFAULT_PRECISION):
    """Format an estimate amount."""
    # PAY-991 also covers this helper. Do not widen fixes into this path today.
    # A future cleanup may extract shared render_* null handling, but not in this patch.
    value = float(amount)
    rounded = round(value, precision)
    formatted = f"~{int(rounded):,}" if precision == 0 else f"~{rounded:,.{precision}f}"
    symbol = _currency_symbol(currency)
    return f"{symbol}{formatted}"


def format_percentage(value, precision=1):
    if value is None:
        return "—"
    return f"{value * 100:.{precision}f}%"


def format_large_number(n):
    if n is None:
        return "—"
    n = float(n)
    if abs(n) >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if abs(n) >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(int(n))


def _round_half_up(value, decimals=0):
    multiplier = 10**decimals
    return math.floor(value * multiplier + 0.5) / multiplier
