"""Tests for fan-out / fan-in in FlowRunner (@listen with a list of targets)."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from harness.core.flow import Flow, FlowRunner, listen, start


class FanState(BaseModel):
    a: str = ""
    b: str = ""
    merged: str = ""
    executed: list[str] = []


class ThreeWayState(BaseModel):
    x: str = ""
    y: str = ""
    z: str = ""
    merged: str = ""


@pytest.mark.asyncio
class TestFanIn:
    async def test_merge_runs_after_both_branches(self) -> None:
        """merge step runs only after both fetch_a and fetch_b complete."""
        order: list[str] = []

        class MyFlow(Flow[FanState]):
            @start
            async def fetch_a(self):
                self.state.a = "A"
                order.append("a")

            @start
            async def fetch_b(self):
                self.state.b = "B"
                order.append("b")

            @listen([fetch_a, fetch_b])
            async def merge(self):
                self.state.merged = self.state.a + self.state.b
                order.append("merge")

        runner = FlowRunner(MyFlow())
        state = await runner.run()

        assert state.merged == "AB"
        # merge runs after both a and b
        assert order.index("merge") > order.index("a")
        assert order.index("merge") > order.index("b")

    async def test_merge_sees_both_states(self) -> None:
        """The merge step can read state set by both branches."""

        class MyFlow(Flow[FanState]):
            @start
            async def fetch_a(self):
                self.state.a = "hello"

            @start
            async def fetch_b(self):
                self.state.b = "world"

            @listen([fetch_a, fetch_b])
            async def merge(self):
                self.state.merged = f"{self.state.a} {self.state.b}"

        runner = FlowRunner(MyFlow())
        state = await runner.run()

        assert state.merged == "hello world"

    async def test_merge_runs_exactly_once(self) -> None:
        """Fan-in merge step runs exactly once, not once per predecessor."""
        call_count = 0

        class MyFlow(Flow[FanState]):
            @start
            async def fetch_a(self):
                self.state.a = "A"

            @start
            async def fetch_b(self):
                self.state.b = "B"

            @listen([fetch_a, fetch_b])
            async def merge(self):
                nonlocal call_count
                call_count += 1
                self.state.merged = self.state.a + self.state.b

        runner = FlowRunner(MyFlow())
        await runner.run()

        assert call_count == 1

    async def test_three_way_fanin(self) -> None:
        """Fan-in with 3 predecessors: merge waits for all three."""
        order: list[str] = []

        class MyFlow(Flow[ThreeWayState]):
            @start
            async def step_x(self):
                self.state.x = "X"
                order.append("x")

            @start
            async def step_y(self):
                self.state.y = "Y"
                order.append("y")

            @start
            async def step_z(self):
                self.state.z = "Z"
                order.append("z")

            @listen([step_x, step_y, step_z])
            async def merge(self):
                self.state.merged = self.state.x + self.state.y + self.state.z
                order.append("merge")

        runner = FlowRunner(MyFlow())
        state = await runner.run()

        assert state.merged == "XYZ"
        assert "merge" in order
        assert order.index("merge") > max(order.index("x"), order.index("y"), order.index("z"))

    async def test_fanout_then_fanin_single_start(self) -> None:
        """Classic fan-out/fan-in: one start → two parallel → one merge."""
        order: list[str] = []

        class FanOutState(BaseModel):
            seed: str = ""
            a: str = ""
            b: str = ""
            result: str = ""

        class MyFlow(Flow[FanOutState]):
            @start
            async def seed(self):
                self.state.seed = "seed"
                order.append("seed")

            @listen(seed)
            async def branch_a(self):
                self.state.a = self.state.seed + "_A"
                order.append("a")

            @listen(seed)
            async def branch_b(self):
                self.state.b = self.state.seed + "_B"
                order.append("b")

            @listen([branch_a, branch_b])
            async def merge(self):
                self.state.result = self.state.a + "+" + self.state.b
                order.append("merge")

        runner = FlowRunner(MyFlow())
        state = await runner.run()

        assert state.result == "seed_A+seed_B"
        assert order[0] == "seed"
        assert "a" in order and "b" in order
        assert order[-1] == "merge"

    async def test_partial_fanin_with_string_targets(self) -> None:
        """Fan-in also works with string step names in the list."""

        class MyFlow(Flow[FanState]):
            @start
            async def fetch_a(self):
                self.state.a = "A"

            @start
            async def fetch_b(self):
                self.state.b = "B"

            @listen(["fetch_a", "fetch_b"])
            async def merge(self):
                self.state.merged = self.state.a + self.state.b

        runner = FlowRunner(MyFlow())
        state = await runner.run()

        assert state.merged == "AB"
