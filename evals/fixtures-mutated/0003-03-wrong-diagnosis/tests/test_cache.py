"""Tests for SimpleCache."""

import asyncio

from cache import SimpleCache


def test_cache_hit_returns_cached_value():
    """Second call for the same key must not trigger another fetch."""
    cache = SimpleCache()
    fetch_count = 0

    async def fetch(key):
        nonlocal fetch_count
        fetch_count += 1
        return f"value:{key}"

    asyncio.run(cache.get_or_fetch("x", fetch))
    asyncio.run(cache.get_or_fetch("x", fetch))
    assert fetch_count == 1, "Expected exactly one fetch on repeated access"


def test_different_keys_fetch_independently():
    """Different keys should each trigger their own fetch."""
    cache = SimpleCache()
    fetched_keys: list[str] = []

    async def fetch(key):
        fetched_keys.append(key)
        return f"value:{key}"

    asyncio.run(cache.get_or_fetch("a", fetch))
    asyncio.run(cache.get_or_fetch("b", fetch))
    assert set(fetched_keys) == {"a", "b"}


def test_invalidate_forces_refetch():
    cache = SimpleCache()
    fetch_count = 0

    async def fetch(key):
        nonlocal fetch_count
        fetch_count += 1
        return "v"

    asyncio.run(cache.get_or_fetch("k", fetch))
    cache.invalidate("k")
    asyncio.run(cache.get_or_fetch("k", fetch))
    assert fetch_count == 2


def test_concurrent_requests_deduplicated():
    """Concurrent requests for the same key must trigger exactly one fetch.

    Without in-flight deduplication, all N concurrent coroutines each call
    fetch() independently — this test catches the thundering herd bug.
    """
    cache = SimpleCache()
    fetch_count = 0

    async def run():
        nonlocal fetch_count

        async def slow_fetch(key):
            nonlocal fetch_count
            fetch_count += 1
            await asyncio.sleep(0.05)  # simulate upstream latency
            return f"value:{key}"

        results = await asyncio.gather(
            *[cache.get_or_fetch("shared-key", slow_fetch) for _ in range(5)]
        )
        return results

    results = asyncio.run(run())
    assert all(r == "value:shared-key" for r in results)
    assert fetch_count == 1, (
        f"Expected 1 fetch call for a deduplicated key, got {fetch_count}. "
        "SimpleCache is missing in-flight request deduplication."
    )
