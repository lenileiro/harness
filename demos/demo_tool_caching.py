"""Demo: tool result caching — expensive lookup called multiple times.

A "geo_lookup" tool simulates an expensive geocoding API call. The agent is
asked about two cities but mentions one twice; the second call for the same
city hits the cache and the tool body never runs.

Run: uv run python demos/demo_tool_caching.py
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, ClassVar

from harness.adapters.ollama import OllamaAdapter
from harness.core import Agent, FailoverPolicy, RunRequest, ToolRegistry
from harness.core.schemas import ToolCall, ToolResult
from harness.storage.memory import InMemoryStorage

MODEL = "gemma4:latest"


class GeoLookupTool:
    """Fake geocoding tool — cache=True means results are memoized by (name, args)."""

    name = "geo_lookup"
    description = (
        "Look up the latitude and longitude of a city. Returns a JSON string with lat/lon."
    )
    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name to geocode."}},
        "required": ["city"],
    }
    approval = "auto"
    cache = True  # ← opt-in to caching

    call_count: ClassVar[int] = 0
    cities_called: ClassVar[list[str]] = []

    async def __call__(self, call: ToolCall) -> ToolResult:
        GeoLookupTool.call_count += 1
        city = call.arguments.get("city", "")
        GeoLookupTool.cities_called.append(city)

        # Simulate slow I/O
        await asyncio.sleep(0.5)
        print(f"  [geo_lookup] EXECUTING for {city!r} (call #{GeoLookupTool.call_count})")

        # Hardcoded lat/lon for demo cities
        data = {
            "Sydney": {"lat": -33.8688, "lon": 151.2093},
            "Tokyo": {"lat": 35.6762, "lon": 139.6503},
            "London": {"lat": 51.5074, "lon": -0.1278},
        }
        coords = data.get(city, {"lat": 0.0, "lon": 0.0})
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=f'{{"city": "{city}", "lat": {coords["lat"]}, "lon": {coords["lon"]}}}',
        )


async def main() -> None:
    adapter = OllamaAdapter()
    storage = InMemoryStorage()
    registry = ToolRegistry()
    registry.register(GeoLookupTool())
    failover = FailoverPolicy(chain=["ollama"], max_attempts=1)

    agent = Agent(
        adapters={"ollama": adapter},
        tools=registry,
        storage=storage,
        failover=failover,
        default_model=MODEL,
        system_prompt=(
            "You are a geography assistant. When asked about cities, "
            "use the geo_lookup tool to fetch their coordinates. "
            "For each city mentioned look it up, even if mentioned multiple times."
        ),
    )

    prompt = (
        "What are the coordinates of Sydney and Tokyo? "
        "Then double-check Sydney's coordinates again to confirm."
    )
    print(f"Model : {MODEL}")
    print(f"Prompt: {prompt}\n")

    t0 = time.perf_counter()
    async for _ in agent.run(RunRequest(prompt=prompt)):
        pass
    elapsed = time.perf_counter() - t0

    print(f"\nTool body executed {GeoLookupTool.call_count} time(s)")
    print(f"Cities called (in order): {GeoLookupTool.cities_called}")
    print(f"Total elapsed: {elapsed:.2f}s")
    print("\n✓ Sydney was looked up once (second call was a cache hit — no extra 0.5s sleep).")


if __name__ == "__main__":
    asyncio.run(main())
