"""Tests for the Calculator class."""

import pytest
from calculator import Calculator


# Fixture used by the rest of this module
@pytest.fixture
def calc() -> Calculator:
    return Calculator()


# -- add ----------------------------------------------------------------
def test_add(calc):
    assert calc.add(2, 3) == 5
    assert calc.add(-1, 1) == 0


# subtract
def test_subtract(calc):
    assert calc.subtract(5, 3) == 2
    assert calc.subtract(0, 4) == -4


# -- multiply -----------------------------------------------------------
def test_multiply(calc):
    assert calc.multiply(3, 4) == 12
    assert calc.multiply(-2, 5) == -10


def test_divide(calc):
    assert calc.divide(10, 2) == 5.0
    assert calc.divide(7, 2) == 3.5


def test_divide_by_zero(calc):
    with pytest.raises(ZeroDivisionError):
        calc.divide(1, 0)


# sqrt
def test_sqrt(calc):
    assert calc.sqrt(9) == 3.0
    assert calc.sqrt(0) == 0.0


def test_sqrt_negative(calc):
    with pytest.raises(ValueError):
        calc.sqrt(-1)
