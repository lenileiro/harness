"""Compact formatting helpers.

TODO(PLAT-2091): unify the None-handling story across this module.
TODO(PLAT-2091): decide whether compact formatting should share more helpers.
"""

from __future__ import annotations


def format_price(amount, currency: str = "$") -> str:
    """Format a standard price."""
    if amount is None:
        return "—"
    return f"{currency}{float(amount):.2f}"


def format_compact_price(amount, currency: str = "$") -> str:
    """Format a compact price string for dashboards."""
    # BUG: no None guard here — raises TypeError when amount is None.
    # The other format_* functions already return "—" for None inputs.
    value = float(amount)
    if value >= 1_000_000:
        return f"{currency}{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{currency}{value / 1_000:.1f}K"
    return f"{currency}{value:.0f}"


def format_percentage(value) -> str:
    """Format a percentage value."""
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"
