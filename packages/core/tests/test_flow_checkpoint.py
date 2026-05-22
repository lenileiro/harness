"""Tests for @persist + fork-from-checkpoint in Flow / FlowRunner."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from harness.core.flow import Flow, FlowRunner, listen, persist, start
from harness.core.flow_checkpoint import (
    FileCheckpointStore,
    FlowCheckpoint,
    InMemoryCheckpointStore,
)

# ---------------------------------------------------------------------------
# State models
# ---------------------------------------------------------------------------


class PipelineState(BaseModel):
    step_a: str = ""
    step_b: str = ""
    step_c: str = ""
    calls: list[str] = []


class BranchState(BaseModel):
    value: int = 0
    result: str = ""


# ---------------------------------------------------------------------------
# InMemoryCheckpointStore unit tests
# ---------------------------------------------------------------------------


class TestInMemoryCheckpointStore:
    @pytest.mark.asyncio
    async def test_save_and_load(self) -> None:
        store = InMemoryCheckpointStore()
        cp = FlowCheckpoint(flow_id="run-1", step_name="step_a", state_json='{"x": 1}')
        await store.save(cp)
        loaded = await store.load("run-1", "step_a")
        assert loaded is not None
        assert loaded.flow_id == "run-1"
        assert loaded.step_name == "step_a"
        assert loaded.state_json == '{"x": 1}'

    @pytest.mark.asyncio
    async def test_load_missing_returns_none(self) -> None:
        store = InMemoryCheckpointStore()
        assert await store.load("nope", "also_nope") is None

    @pytest.mark.asyncio
    async def test_list_flow_filters_by_flow_id(self) -> None:
        store = InMemoryCheckpointStore()
        await store.save(FlowCheckpoint(flow_id="a", step_name="s1", state_json="{}"))
        await store.save(FlowCheckpoint(flow_id="a", step_name="s2", state_json="{}"))
        await store.save(FlowCheckpoint(flow_id="b", step_name="s1", state_json="{}"))
        items_a = await store.list_flow("a")
        assert len(items_a) == 2
        assert all(cp.flow_id == "a" for cp in items_a)

    @pytest.mark.asyncio
    async def test_save_overwrites_existing(self) -> None:
        store = InMemoryCheckpointStore()
        await store.save(FlowCheckpoint(flow_id="r", step_name="s", state_json='{"v": 1}'))
        await store.save(FlowCheckpoint(flow_id="r", step_name="s", state_json='{"v": 2}'))
        loaded = await store.load("r", "s")
        assert loaded is not None
        assert '"v": 2' in loaded.state_json


# ---------------------------------------------------------------------------
# FileCheckpointStore unit tests
# ---------------------------------------------------------------------------


class TestFileCheckpointStore:
    @pytest.mark.asyncio
    async def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        cp = FlowCheckpoint(flow_id="run-1", step_name="analyze", state_json='{"done": true}')
        await store.save(cp)
        loaded = await store.load("run-1", "analyze")
        assert loaded is not None
        assert loaded.flow_id == "run-1"
        assert loaded.step_name == "analyze"
        assert '"done": true' in loaded.state_json

    @pytest.mark.asyncio
    async def test_creates_subdirectory(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        await store.save(FlowCheckpoint(flow_id="nested/run", step_name="s", state_json="{}"))
        assert (tmp_path / "nested/run" / "s.json").exists()

    @pytest.mark.asyncio
    async def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        assert await store.load("missing", "step") is None

    @pytest.mark.asyncio
    async def test_list_flow(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        await store.save(FlowCheckpoint(flow_id="run-x", step_name="a", state_json="{}"))
        await store.save(FlowCheckpoint(flow_id="run-x", step_name="b", state_json="{}"))
        items = await store.list_flow("run-x")
        assert len(items) == 2
        names = {cp.step_name for cp in items}
        assert names == {"a", "b"}

    @pytest.mark.asyncio
    async def test_list_flow_empty_dir(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)
        assert await store.list_flow("nonexistent") == []


# ---------------------------------------------------------------------------
# @persist decorator + FlowRunner checkpointing
# ---------------------------------------------------------------------------


class TestPersistDecorator:
    @pytest.mark.asyncio
    async def test_persist_step_saves_checkpoint(self) -> None:
        store = InMemoryCheckpointStore()

        class MyFlow(Flow[PipelineState]):
            @start
            @persist
            async def step_a(self):
                self.state.step_a = "done"
                self.state.calls.append("a")

        runner = FlowRunner(MyFlow(), checkpoint_store=store, flow_id="test-run")
        await runner.run()

        cp = await store.load("test-run", "step_a")
        assert cp is not None
        assert "done" in cp.state_json

    @pytest.mark.asyncio
    async def test_non_persist_step_does_not_save(self) -> None:
        store = InMemoryCheckpointStore()

        class MyFlow(Flow[PipelineState]):
            @start
            async def step_a(self):
                self.state.step_a = "done"

        runner = FlowRunner(MyFlow(), checkpoint_store=store, flow_id="no-save")
        await runner.run()

        assert await store.load("no-save", "step_a") is None

    @pytest.mark.asyncio
    async def test_persist_saves_state_at_that_point(self) -> None:
        store = InMemoryCheckpointStore()

        class MyFlow(Flow[PipelineState]):
            @start
            @persist
            async def step_a(self):
                self.state.step_a = "A"

            @listen(step_a)
            async def step_b(self):
                self.state.step_b = "B"

        runner = FlowRunner(MyFlow(), checkpoint_store=store, flow_id="partial")
        await runner.run()

        cp = await store.load("partial", "step_a")
        assert cp is not None
        state = PipelineState.model_validate_json(cp.state_json)
        assert state.step_a == "A"
        assert state.step_b == ""  # not yet set when checkpoint was taken

    @pytest.mark.asyncio
    async def test_multiple_persist_steps_each_saved(self) -> None:
        store = InMemoryCheckpointStore()

        class MyFlow(Flow[PipelineState]):
            @start
            @persist
            async def step_a(self):
                self.state.step_a = "A"

            @listen(step_a)
            @persist
            async def step_b(self):
                self.state.step_b = "B"

        runner = FlowRunner(MyFlow(), checkpoint_store=store, flow_id="multi")
        await runner.run()

        assert await store.load("multi", "step_a") is not None
        assert await store.load("multi", "step_b") is not None

    @pytest.mark.asyncio
    async def test_no_checkpoint_store_persist_is_noop(self) -> None:
        class MyFlow(Flow[PipelineState]):
            @start
            @persist
            async def step_a(self):
                self.state.step_a = "done"

        # No checkpoint_store — should run without error
        runner = FlowRunner(MyFlow())
        state = await runner.run()
        assert state.step_a == "done"

    @pytest.mark.asyncio
    async def test_flow_id_auto_assigned_if_not_provided(self) -> None:
        store = InMemoryCheckpointStore()

        class MyFlow(Flow[PipelineState]):
            @start
            @persist
            async def step_a(self):
                self.state.step_a = "done"

        runner = FlowRunner(MyFlow(), checkpoint_store=store)
        await runner.run()

        # flow_id was auto-assigned
        assert runner._flow_id is not None
        items = await store.list_flow(runner._flow_id)
        assert len(items) == 1


# ---------------------------------------------------------------------------
# Fork-from-checkpoint tests
# ---------------------------------------------------------------------------


class TestForkFromCheckpoint:
    @pytest.mark.asyncio
    async def test_fork_resumes_state_correctly(self) -> None:
        store = InMemoryCheckpointStore()
        fork_executions: list[str] = []

        class MyFlow(Flow[PipelineState]):
            @start
            @persist
            async def step_a(self):
                self.state.step_a = "A"

            @listen(step_a)
            async def step_b(self):
                self.state.step_b = "B"
                fork_executions.append("b")

        # Run original
        runner = FlowRunner(MyFlow(), checkpoint_store=store, flow_id="orig")
        await runner.run()

        # Fork from step_a — step_a's state is restored, step_b reruns
        cp = await store.load("orig", "step_a")
        assert cp is not None
        fork_executions.clear()
        forked = FlowRunner.from_checkpoint(cp, MyFlow())
        state = await forked.run()

        assert state.step_a == "A"
        assert state.step_b == "B"
        # Only step_b ran in the forked run
        assert fork_executions == ["b"]

    @pytest.mark.asyncio
    async def test_fork_does_not_rerun_persisted_step(self) -> None:
        """The @persist step itself should NOT re-execute when forking from it."""
        store = InMemoryCheckpointStore()
        ran = []

        class MyFlow(Flow[PipelineState]):
            @start
            @persist
            async def expensive(self):
                ran.append("expensive")
                self.state.step_a = "computed"

            @listen(expensive)
            async def use_result(self):
                self.state.step_b = self.state.step_a + "_used"

        runner = FlowRunner(MyFlow(), checkpoint_store=store, flow_id="exp")
        await runner.run()
        assert ran == ["expensive"]

        cp = await store.load("exp", "expensive")
        assert cp is not None
        forked = FlowRunner.from_checkpoint(cp, MyFlow())
        state = await forked.run()

        # "expensive" ran only once (in the original, not the fork)
        assert ran == ["expensive"]
        assert state.step_b == "computed_used"

    @pytest.mark.asyncio
    async def test_fork_with_file_store(self, tmp_path: Path) -> None:
        store = FileCheckpointStore(tmp_path)

        class MyFlow(Flow[PipelineState]):
            @start
            @persist
            async def step_a(self):
                self.state.step_a = "file-persisted"

            @listen(step_a)
            async def step_b(self):
                self.state.step_b = "from-fork"

        runner = FlowRunner(MyFlow(), checkpoint_store=store, flow_id="file-run")
        await runner.run()

        cp = await store.load("file-run", "step_a")
        assert cp is not None
        forked = FlowRunner.from_checkpoint(cp, MyFlow(), checkpoint_store=store)
        state = await forked.run()

        assert state.step_a == "file-persisted"
        assert state.step_b == "from-fork"

    @pytest.mark.asyncio
    async def test_fork_mid_pipeline_skips_earlier_steps(self) -> None:
        """Forking from step_b should skip step_a and run step_c."""
        store = InMemoryCheckpointStore()
        fork_executions: list[str] = []

        class MyFlow(Flow[PipelineState]):
            @start
            async def step_a(self):
                self.state.step_a = "A"

            @listen(step_a)
            @persist
            async def step_b(self):
                self.state.step_b = "B"

            @listen(step_b)
            async def step_c(self):
                self.state.step_c = "C"
                fork_executions.append("c")

        runner = FlowRunner(MyFlow(), checkpoint_store=store, flow_id="mid")
        await runner.run()

        cp = await store.load("mid", "step_b")
        assert cp is not None
        fork_executions.clear()
        forked = FlowRunner.from_checkpoint(cp, MyFlow())
        state = await forked.run()

        # state was restored from step_b checkpoint (includes step_a result)
        assert state.step_a == "A"
        assert state.step_b == "B"
        assert state.step_c == "C"
        # only step_c ran in the fork
        assert fork_executions == ["c"]

    @pytest.mark.asyncio
    async def test_fork_at_end_of_chain_is_noop(self) -> None:
        """Forking from a terminal step (no listeners) returns restored state unchanged."""
        store = InMemoryCheckpointStore()

        class MyFlow(Flow[PipelineState]):
            @start
            @persist
            async def terminal(self):
                self.state.step_a = "end"

        runner = FlowRunner(MyFlow(), checkpoint_store=store, flow_id="end")
        await runner.run()

        cp = await store.load("end", "terminal")
        assert cp is not None
        forked = FlowRunner.from_checkpoint(cp, MyFlow())
        state = await forked.run()

        assert state.step_a == "end"
