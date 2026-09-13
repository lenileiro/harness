"""Checkpoints preserve pending branches, join barriers, and router decisions."""

import pytest
from pydantic import BaseModel, Field

from harness.core.flow import Flow, FlowRunner, listen, persist, router, start
from harness.core.flow_checkpoint import FlowCheckpoint, InMemoryCheckpointStore


class State(BaseModel):
    steps: list[str] = Field(default_factory=list)


class Routed(Flow[State]):
    @start
    @persist
    @router()
    async def choose(self):
        self.state.steps.append("choose")
        return "chosen"

    @listen("chosen")
    async def finish(self):
        self.state.steps.append("finish")


class Joined(Flow[State]):
    @start
    @persist
    async def a(self):
        self.state.steps.append("a")

    @start
    @persist
    async def b(self):
        self.state.steps.append("b")

    @listen([a, b])
    async def merge(self):
        self.state.steps.append("merge")


@pytest.mark.parametrize("flow_type,step", [(Routed, "choose"), (Joined, "a"), (Joined, "b")])
async def test_resume_preserves_full_execution_frontier(flow_type, step):
    store = InMemoryCheckpointStore()
    original = await FlowRunner(flow_type(), checkpoint_store=store, flow_id="run").run()
    checkpoint = await store.load("run", step)
    assert checkpoint is not None
    restored = FlowCheckpoint.model_validate_json(checkpoint.model_dump_json())
    resumed = await FlowRunner.from_checkpoint(restored, flow_type()).run()
    assert resumed.steps == original.steps


def test_legacy_ambiguous_checkpoint_is_rejected():
    checkpoint = FlowCheckpoint(
        flow_id="old", step_name="choose", state_json='{"steps":["choose"]}'
    )
    with pytest.raises(ValueError, match="legacy"):
        FlowRunner.from_checkpoint(checkpoint, Routed())


def test_checkpoint_unknown_step_is_rejected():
    checkpoint = FlowCheckpoint(flow_id="old", step_name="removed", state_json="{}")
    with pytest.raises(ValueError, match="step"):
        FlowRunner.from_checkpoint(checkpoint, Routed())


async def test_legacy_linear_checkpoint_resumes_without_repeating_ancestors():
    class Linear(Flow[State]):
        @start
        async def a(self):
            self.state.steps.append("a")

        @listen(a)
        async def b(self):
            self.state.steps.append("b")

    checkpoint = FlowCheckpoint(flow_id="old", step_name="a", state_json='{"steps":["a"]}')
    assert (await FlowRunner.from_checkpoint(checkpoint, Linear()).run()).steps == ["a", "b"]


async def test_changed_graph_is_rejected_before_mutating_flow():
    store = InMemoryCheckpointStore()
    await FlowRunner(Routed(), checkpoint_store=store, flow_id="run").run()
    checkpoint = await store.load("run", "choose")
    assert checkpoint is not None

    class Changed(Routed):
        @listen("chosen")
        async def extra(self):
            self.state.steps.append("extra")

    flow = Changed()
    with pytest.raises(ValueError, match="graph differs"):
        FlowRunner.from_checkpoint(checkpoint, flow)
    assert flow.state.steps == []
