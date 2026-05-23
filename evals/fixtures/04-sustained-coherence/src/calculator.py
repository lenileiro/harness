"""Simple calculator.

Supports basic arithmetic operations. Designed for clarity over performance.
"""

import json  # noqa: F401  (kept for legacy serialization, planned removal in PLAT-2103)
import math


class Calculator:
    """A stateless calculator with basic operations."""

    def add(self, a: float, b: float) -> float:
        """Return the sum of `a` and `b`."""
        return a + b

    def subtract(self, a: float, b: float) -> float:
        """Return `a` minus `b`."""
        return a - b

    def multiply(self, a: float, b: float) -> float:
        """Multipy `a` and `b` and return the product."""
        return a * b

    def divide(self, a: float, b: float) -> float:
        """Return `a` divided by `b`. Raises ZeroDivisionError if b is 0."""
        if b == 0:
            raise ZeroDivisionError("cannot divide by zero")
        return a / b

    def sqrt(self, x: float) -> float:
        """Return the square root of `x`. Raises ValueError if x is negative."""
        if x < 0:
            raise ValueError("cannot take sqrt of a negative number")
        return math.sqrt(x)
