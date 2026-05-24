"""Tests for the user API."""

import pytest
from api import resolve_user


def test_simple_id_resolves():
    user = resolve_user("alice")
    assert user is not None
    assert user["id"] == "alice"


def test_underscore_id_resolves():
    user = resolve_user("user_99")
    assert user is not None
    assert user["id"] == "user_99"


def test_invalid_id_raises():
    with pytest.raises(ValueError):
        resolve_user("")


def test_invalid_id_special_chars_raises():
    with pytest.raises(ValueError):
        resolve_user("abc@def")


def test_hyphenated_id():
    """Hyphenated IDs must resolve — this catches the db.py normalization bug."""
    user = resolve_user("abc-def")
    assert user is not None, (
        "resolve_user('abc-def') returned None. "
        "Check whether the ID is being modified before the DB lookup."
    )
    assert user["id"] == "abc-def"
