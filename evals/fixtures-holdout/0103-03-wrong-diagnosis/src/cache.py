"""Simple async cache with fetch-on-miss.

Used by the batch endpoint to avoid redundant upstream API calls.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

# Configurable timeout for upstream fetch calls.
# The batch endpoint passes this when calling get_or_fetch().
MAX_WAIT_SECONDS = 5


class SimpleCache:
    """In-memory async cache.

    On a cache miss, calls the provided fetch coroutine to populate the value.
    The result is stored and returned to the caller.

    Note: concurrent requests for the same key each trigger their own
    independent fetch — in-flight deduplication is not implemented.

    TODO: Add deduplication to avoid thundering herd on cache misses.
    """

    def __init__(self) -> None:
        self._store: dict[str, Any] = {}

    async def get_or_fetch(
        self,
        key: str,
        fetch: Callable[[str], Awaitable[Any]],
        *,
        fetch_timeout: float = MAX_WAIT_SECONDS,
    ) -> Any:
        """Return cached value for key, or call fetch(key) and cache the result.

        Args:
            key: MemTab key.
            fetch: Async callable that receives the key and returns a value.
            fetch_timeout: Seconds to wait before raising asyncio.TimeoutError.
        """
        if key in self._store:
            return self._store[key]

        # MISSING: no in-flight deduplication.
        # Multiple concurrent callers for the same key all reach this line and
        # each triggers its own fetch() call independently.
        value = await asyncio.wait_for(fetch(key), timeout=fetch_timeout)
        self._store[key] = value
        return value

    def invalidate(self, key: str) -> None:
        """Remove a key from the cache."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Remove all cached entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)
