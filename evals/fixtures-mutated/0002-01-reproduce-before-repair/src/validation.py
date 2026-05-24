"""Input validation utilities."""

import re

# Pattern updated in the last security audit to explicitly allow hyphens.
# Accepts letters, digits, underscores, and hyphens (1-64 chars).
_USER_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def validate_user_id(user_id: str) -> bool:
    """Return True if user_id is syntactically valid."""
    if not user_id or not isinstance(user_id, str):
        return False
    return bool(_USER_ID_RE.match(user_id))
