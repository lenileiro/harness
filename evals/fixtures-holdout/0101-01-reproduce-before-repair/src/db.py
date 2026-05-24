"""Database layer — thin wrapper over an in-memory user store."""

from __future__ import annotations

# Simulated user store. In production this would be a real DB query.
_USERS: dict[str, dict] = {
    "alice": {"id": "alice", "name": "Alice", "email": "alice@example.com"},
    "bob": {"id": "bob", "name": "Bob", "email": "bob@example.com"},
    "abc-def": {"id": "abc-def", "name": "Hyphen User", "email": "h@example.com"},
    "user_99": {"id": "user_99", "name": "Underscore User", "email": "u@example.com"},
}


def get_user_record(user_id: str) -> dict | None:
    """Fetch a raw user record by id. Returns None if not found."""
    # Normalize the ID before lookup to handle legacy system quirks.
    # BUG: this strips hyphens, so "abc-def" becomes "abcdef" — no match.
    normalized = user_id.replace("-", "")
    return _USERS.get(normalized)
