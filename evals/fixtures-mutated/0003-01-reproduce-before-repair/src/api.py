"""Public user API."""

from __future__ import annotations

from db import get_user_record
from validation import validate_user_id


def lookup_user(user_id: str) -> dict | None:
    """Return user dict or None. Raises ValueError for invalid IDs."""
    if not validate_user_id(user_id):
        raise ValueError(f"Invalid user_id: {user_id!r}")
    return get_user_record(user_id)
