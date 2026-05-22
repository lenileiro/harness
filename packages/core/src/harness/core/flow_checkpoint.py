"""Checkpoint persistence for Flow runs.

After any ``@persist``-decorated step completes, :class:`FlowRunner` serialises
the current state into a :class:`FlowCheckpoint` and hands it to the configured
:class:`CheckpointStore`. That checkpoint can later be loaded to fork a new run
from the same point — useful for retrying a branch, exploring alternatives, or
recovering from a crash mid-flow.

Example::

    store = InMemoryCheckpointStore()
    runner = FlowRunner(MyFlow(), checkpoint_store=store, flow_id="run-1")
    await runner.run()

    cp = await store.load("run-1", "expensive_step")
    forked = FlowRunner.from_checkpoint(cp, MyFlow(), checkpoint_store=store)
    await forked.run()  # resumes from after "expensive_step"
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class FlowCheckpoint(BaseModel):
    """Serialised snapshot of a flow's state after a ``@persist`` step."""

    flow_id: str
    step_name: str
    state_json: str
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )


@runtime_checkable
class CheckpointStore(Protocol):
    """Persistence backend for :class:`FlowCheckpoint` objects."""

    async def save(self, checkpoint: FlowCheckpoint) -> None: ...

    async def load(self, flow_id: str, step_name: str) -> FlowCheckpoint | None: ...

    async def list_flow(self, flow_id: str) -> list[FlowCheckpoint]: ...


class InMemoryCheckpointStore:
    """Volatile in-memory store — checkpoints are lost when the process exits."""

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], FlowCheckpoint] = {}

    async def save(self, checkpoint: FlowCheckpoint) -> None:
        self._data[(checkpoint.flow_id, checkpoint.step_name)] = checkpoint

    async def load(self, flow_id: str, step_name: str) -> FlowCheckpoint | None:
        return self._data.get((flow_id, step_name))

    async def list_flow(self, flow_id: str) -> list[FlowCheckpoint]:
        return [cp for (fid, _), cp in self._data.items() if fid == flow_id]


class FileCheckpointStore:
    """File-system store. One JSON file per ``(flow_id, step_name)`` pair.

    Layout: ``{base_dir}/{flow_id}/{step_name}.json``.
    The directory is created on first write.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir)

    def _path(self, flow_id: str, step_name: str) -> Path:
        return self._base / flow_id / f"{step_name}.json"

    async def save(self, checkpoint: FlowCheckpoint) -> None:
        path = self._path(checkpoint.flow_id, checkpoint.step_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(checkpoint.model_dump_json())

    async def load(self, flow_id: str, step_name: str) -> FlowCheckpoint | None:
        path = self._path(flow_id, step_name)
        if not path.exists():
            return None
        return FlowCheckpoint.model_validate_json(path.read_text())

    async def list_flow(self, flow_id: str) -> list[FlowCheckpoint]:
        flow_dir = self._base / flow_id
        if not flow_dir.exists():
            return []
        result = []
        for p in sorted(flow_dir.glob("*.json")):
            with contextlib.suppress(Exception):
                result.append(FlowCheckpoint.model_validate_json(p.read_text()))
        return result


__all__ = [
    "CheckpointStore",
    "FileCheckpointStore",
    "FlowCheckpoint",
    "InMemoryCheckpointStore",
]
