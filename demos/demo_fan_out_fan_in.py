"""Demo: fan-out / fan-in Flow — barrier semantics for multi-source merge.

FlowRunner executes steps in BFS topological order (not concurrently).
Fan-out means multiple steps can proceed from one predecessor; fan-in means
a step waits for ALL of its listed predecessors before it runs.

Architecture:
  fetch_weather ─┐
  fetch_news    ─┼─ @listen([weather, news, stocks]) ─► morning_brief
  fetch_stocks  ─┘

The three @start steps run in BFS order; morning_brief only runs after the
BFS has visited all three — even if they finished on different iterations.
Without the fan-in list, you'd have to manually coordinate or risk running
the merge before all data is ready.

Run: uv run python demos/demo_fan_out_fan_in.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from harness.core.flow import Flow, FlowRunner, listen, start


class BriefingState(BaseModel):
    weather: str = ""
    news: str = ""
    stocks: str = ""
    briefing: str = ""
    execution_order: list[str] = []


class MorningBriefFlow(Flow[BriefingState]):
    @start
    async def fetch_weather(self) -> None:
        print("  [1/4] fetch_weather running...")
        self.state.weather = "Sydney: 22°C, partly cloudy."
        self.state.execution_order.append("fetch_weather")

    @start
    async def fetch_news(self) -> None:
        print("  [2/4] fetch_news running...")
        self.state.news = "Top story: room-temperature superconductors confirmed."
        self.state.execution_order.append("fetch_news")

    @start
    async def fetch_stocks(self) -> None:
        print("  [3/4] fetch_stocks running...")
        self.state.stocks = "ASX 200: +0.8%. BTC: $98,400 (+2.1%)."
        self.state.execution_order.append("fetch_stocks")

    @listen([fetch_weather, fetch_news, fetch_stocks])
    async def morning_brief(self) -> None:
        print("  [4/4] morning_brief running (all 3 sources ready)...")
        self.state.briefing = (
            f"Good morning!\n\n"
            f"WEATHER: {self.state.weather}\n"
            f"NEWS: {self.state.news}\n"
            f"MARKETS: {self.state.stocks}"
        )
        self.state.execution_order.append("morning_brief")


class BadFlow_NoFanIn(Flow[BriefingState]):
    """What happens WITHOUT fan-in: morning_brief fires after fetch_weather alone."""

    @start
    async def fetch_weather(self) -> None:
        self.state.weather = "Sydney: 22°C"
        self.state.execution_order.append("fetch_weather")

    @start
    async def fetch_news(self) -> None:
        self.state.news = "Breaking news..."
        self.state.execution_order.append("fetch_news")

    @start
    async def fetch_stocks(self) -> None:
        self.state.stocks = "ASX +0.8%"
        self.state.execution_order.append("fetch_stocks")

    @listen(fetch_weather)  # Only listens to weather — wrong!
    async def morning_brief(self) -> None:
        # news and stocks may not be set yet
        self.state.briefing = (
            f"WEATHER: {self.state.weather} | "
            f"NEWS: {self.state.news!r} | "  # likely empty ""
            f"MARKETS: {self.state.stocks!r}"  # likely empty ""
        )
        self.state.execution_order.append("morning_brief")


async def main() -> None:
    print("=" * 60)
    print("WITH fan-in: @listen([weather, news, stocks])")
    print("=" * 60)
    state = await FlowRunner(MorningBriefFlow()).run()
    print(f"\nExecution order: {state.execution_order}")
    print(f"\n{state.briefing}")
    assert state.execution_order[-1] == "morning_brief", "merge ran last"
    assert state.news != "", "news was populated before merge"
    assert state.stocks != "", "stocks was populated before merge"
    print("\n✓ morning_brief ran AFTER all three fetches. All fields populated.")

    print("\n" + "=" * 60)
    print("WITHOUT fan-in: @listen(fetch_weather) only")
    print("=" * 60)
    bad = await FlowRunner(BadFlow_NoFanIn()).run()
    print(f"\nExecution order: {bad.execution_order}")
    print(f"\n{bad.briefing}")
    # morning_brief ran right after weather, before news/stocks were ready
    if bad.execution_order.index("morning_brief") < bad.execution_order.index("fetch_news"):
        print("\n⚠ morning_brief ran BEFORE fetch_news — incomplete data!")
    else:
        print("\n(morning_brief happened to run after all in this BFS ordering)")


if __name__ == "__main__":
    asyncio.run(main())
