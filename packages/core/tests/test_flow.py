"""Tests for Flow[StateT] / FlowRunner — decorator-driven workflow DAG."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from harness.core.flow import Flow, FlowRunner, listen, router, start

# ---------------------------------------------------------------------------
# State models
# ---------------------------------------------------------------------------


class SimpleState(BaseModel):
    value: str = ""
    calls: list[str] = []


class RouterState(BaseModel):
    path: str = ""
    result: str = ""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFlowBasics:
    def test_flow_requires_state_type(self) -> None:
        with pytest.raises(TypeError):

            class BadFlow(Flow):  # type: ignore[type-arg]
                pass

            BadFlow()

    def test_flow_uses_default_state(self) -> None:
        class MyFlow(Flow[SimpleState]):
            @start
            async def init(self):
                self.state.value = "initialized"

        flow = MyFlow()
        assert flow.state.value == ""

    def test_flow_accepts_initial_state(self) -> None:
        class MyFlow(Flow[SimpleState]):
            @start
            async def init(self):
                pass

        flow = MyFlow(state=SimpleState(value="preset"))
        assert flow.state.value == "preset"


@pytest.mark.asyncio
class TestFlowRunner:
    async def test_start_step_runs(self) -> None:
        class MyFlow(Flow[SimpleState]):
            @start
            async def entry(self):
                self.state.value = "ran"

        runner = FlowRunner(MyFlow())
        state = await runner.run()
        assert state.value == "ran"

    async def test_listen_runs_after_start(self) -> None:
        class MyFlow(Flow[SimpleState]):
            @start
            async def first(self):
                self.state.calls.append("first")

            @listen(first)
            async def second(self):
                self.state.calls.append("second")

        runner = FlowRunner(MyFlow())
        state = await runner.run()
        assert state.calls == ["first", "second"]

    async def test_chain_three_steps(self) -> None:
        class MyFlow(Flow[SimpleState]):
            @start
            async def a(self):
                self.state.calls.append("a")

            @listen(a)
            async def b(self):
                self.state.calls.append("b")

            @listen(b)
            async def c(self):
                self.state.calls.append("c")

        runner = FlowRunner(MyFlow())
        state = await runner.run()
        assert state.calls == ["a", "b", "c"]

    async def test_router_branches(self) -> None:
        class MyFlow(Flow[RouterState]):
            @start
            async def classify(self):
                self.state.path = "fast"

            @listen(classify)
            @router()
            async def decide(self) -> str:
                return self.state.path

            @listen("fast")
            async def handle_fast(self):
                self.state.result = "fast path taken"

            @listen("slow")
            async def handle_slow(self):
                self.state.result = "slow path taken"

        runner = FlowRunner(MyFlow())
        state = await runner.run()
        assert state.result == "fast path taken"

    async def test_router_takes_slow_branch(self) -> None:
        class MyFlow(Flow[RouterState]):
            @start
            async def classify(self):
                self.state.path = "slow"

            @listen(classify)
            @router()
            async def decide(self) -> str:
                return self.state.path

            @listen("fast")
            async def handle_fast(self):
                self.state.result = "fast"

            @listen("slow")
            async def handle_slow(self):
                self.state.result = "slow"

        runner = FlowRunner(MyFlow())
        state = await runner.run()
        assert state.result == "slow"

    async def test_each_step_runs_once(self) -> None:
        """Steps should run exactly once even if they appear in multiple paths."""

        class MyFlow(Flow[SimpleState]):
            @start
            async def a(self):
                self.state.calls.append("a")

            @listen(a)
            async def b(self):
                self.state.calls.append("b")

            @listen(a)
            async def c(self):
                self.state.calls.append("c")

        runner = FlowRunner(MyFlow())
        state = await runner.run()
        # a runs once, then b and c each run once
        assert state.calls.count("a") == 1
        assert state.calls.count("b") == 1
        assert state.calls.count("c") == 1

    async def test_sync_steps_work(self) -> None:
        class MyFlow(Flow[SimpleState]):
            @start
            def sync_start(self):
                self.state.value = "sync"

        runner = FlowRunner(MyFlow())
        state = await runner.run()
        assert state.value == "sync"

    async def test_state_preserved_across_steps(self) -> None:
        class MyFlow(Flow[SimpleState]):
            @start
            async def write(self):
                self.state.value = "hello"

            @listen(write)
            async def read(self):
                self.state.calls.append(self.state.value)

        runner = FlowRunner(MyFlow())
        state = await runner.run()
        assert state.calls == ["hello"]
