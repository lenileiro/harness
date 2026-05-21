"""Tests for the evidence-ledger fields on `tool_call.completed`.

The runtime records timing + tool metadata + size info as part of the
existing `tool_call.completed` activity event. This file verifies the
shape of that event end-to-end.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.core import (
    ActivityEvent,
    ActivityStore,
    Agent,
    AutoApprove,
    FailoverPolicy,
    RunRequest,
    ToolCall,
    ToolRegistry,
    ToolResult,
)
from harness.core import activity as activity_kinds

from .conftest import MockAdapter, MockStorage, text_turn, tool_call_turn


class _Sink(ActivityStore):
    def __init__(self) -> None:
        self.events: list[ActivityEvent] = []

    async def append_activity(self, event: ActivityEvent) -> None:
        self.events.append(event)

    async def list_activity(
        self,
        *,
        task_id: str | None = None,
        session_id: str | None = None,
        kinds: tuple[str, ...] | None = None,
        limit: int = 200,
    ) -> list[ActivityEvent]:
        return list(self.events)[:limit]


class _MetadataTool:
    """A tool that emits a fixed metadata payload."""

    name = "evidencer"
    description = "tool that emits structured metadata"
    approval = "auto"
    phases: tuple[str, ...] = ("*",)

    def __init__(self) -> None:
        self.parameters_schema: dict = {"type": "object", "properties": {}}
        self.calls: list = []

    async def __call__(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content="ok",
            metadata={"custom": "value", "number": 7},
        )


class _SlowTool:
    name = "slow"
    description = "tool that sleeps briefly"
    approval = "auto"
    phases: tuple[str, ...] = ("*",)

    def __init__(self) -> None:
        self.parameters_schema: dict = {"type": "object", "properties": {}}

    async def __call__(self, call: ToolCall) -> ToolResult:
        import asyncio

        await asyncio.sleep(0.02)
        return ToolResult(tool_call_id=call.id, name=self.name, content="done")


def _agent_with(adapter: MockAdapter, tool: object, sink: ActivityStore) -> Agent:
    registry = ToolRegistry()
    registry.register(tool)  # type: ignore[arg-type]
    return Agent(
        adapters={"mock": adapter},  # type: ignore[arg-type]
        tools=registry,
        storage=MockStorage(),
        failover=FailoverPolicy(chain=["mock"], max_attempts=1),
        approval_handler=AutoApprove(),
        activity_store=sink,
        default_model="m",
    )


async def _drain(it):
    async for _ in it:
        pass


@pytest.mark.asyncio
class TestEvidenceShape:
    async def test_tool_call_completed_includes_metadata_and_timing(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="evidencer", arguments={"x": 1}),
                text_turn("done"),
            ],
        )
        tool = _MetadataTool()
        sink = _Sink()
        agent = _agent_with(adapter, tool, sink)

        await _drain(agent.run(RunRequest(prompt="run", model="m")))

        completed = [e for e in sink.events if e.kind == activity_kinds.TOOL_CALL_COMPLETED]
        assert len(completed) == 1
        data = completed[0].data
        # Required ledger fields:
        assert data["tool_call_id"] == "c1"
        assert data["name"] == "evidencer"
        assert data["is_error"] is False
        assert data["arguments"] == {"x": 1}
        assert data["content_size"] == len("ok")
        assert data["content_preview"] == "ok"
        # Tool-supplied metadata is round-tripped.
        assert data["metadata"] == {"custom": "value", "number": 7}
        # Timing is captured for executed tools.
        assert isinstance(data["duration_ms"], int)
        assert data["duration_ms"] >= 0

    async def test_duration_reflects_real_time(self, tmp_path: Path) -> None:
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="slow", arguments={}),
                text_turn("done"),
            ],
        )
        tool = _SlowTool()
        sink = _Sink()
        agent = _agent_with(adapter, tool, sink)

        await _drain(agent.run(RunRequest(prompt="x", model="m")))

        completed = next(e for e in sink.events if e.kind == activity_kinds.TOOL_CALL_COMPLETED)
        # Our tool sleeps 20 ms, so duration should be at least ~15.
        assert completed.data["duration_ms"] >= 15

    async def test_short_circuit_paths_have_no_duration(self, tmp_path: Path) -> None:
        """Denied / unknown / out-of-phase calls skip execution; duration is None."""
        adapter = MockAdapter(
            "mock",
            scripts=[
                tool_call_turn(call_id="c1", name="ghost", arguments={}),
                text_turn("done"),
            ],
        )
        sink = _Sink()
        # No tool registered for "ghost"
        registry = ToolRegistry()
        agent = Agent(
            adapters={"mock": adapter},  # type: ignore[arg-type]
            tools=registry,
            storage=MockStorage(),
            failover=FailoverPolicy(chain=["mock"], max_attempts=1),
            activity_store=sink,
            approval_handler=AutoApprove(),
            default_model="m",
        )
        await _drain(agent.run(RunRequest(prompt="x", model="m")))

        completed = next(e for e in sink.events if e.kind == activity_kinds.TOOL_CALL_COMPLETED)
        # Unknown tool short-circuits before execution — no timing.
        assert completed.data["duration_ms"] is None
        assert completed.data["is_error"] is True
        assert completed.data["metadata"] == {}


@pytest.mark.asyncio
class TestBuiltInToolMetadata:
    """Each built-in tool emits its own structured metadata payload."""

    async def test_read_file_metadata(self, tmp_path: Path) -> None:
        from harness.tools.fs import ReadFileTool

        (tmp_path / "f.txt").write_text("hello world", encoding="utf-8")
        result = await ReadFileTool(cwd=tmp_path)(
            ToolCall(id="c1", name="read_file", arguments={"path": "f.txt"})
        )
        assert result.is_error is False
        assert result.metadata is not None
        assert result.metadata["bytes"] == len("hello world")
        assert result.metadata["encoding"] == "utf-8"
        assert result.metadata["path"] == "f.txt"

    async def test_write_file_metadata(self, tmp_path: Path) -> None:
        from harness.tools.fs import WriteFileTool

        result = await WriteFileTool(cwd=tmp_path)(
            ToolCall(id="c1", name="write_file", arguments={"path": "new.txt", "content": "hi"})
        )
        assert result.metadata is not None
        assert result.metadata["bytes_written"] == 2
        assert result.metadata["created"] is True

    async def test_edit_file_metadata(self, tmp_path: Path) -> None:
        from harness.tools.fs import EditFileTool

        (tmp_path / "x.txt").write_text("abc def", encoding="utf-8")
        result = await EditFileTool(cwd=tmp_path)(
            ToolCall(
                id="c1",
                name="edit_file",
                arguments={"path": "x.txt", "old": "abc", "new": "ABC"},
            )
        )
        assert result.metadata is not None
        assert result.metadata["occurrences_replaced"] == 1
        assert result.metadata["bytes_before"] == 7
        assert result.metadata["bytes_after"] == 7

    async def test_list_dir_metadata(self, tmp_path: Path) -> None:
        from harness.tools.fs import ListDirTool

        (tmp_path / "a").mkdir()
        (tmp_path / "b.txt").write_text("", encoding="utf-8")
        result = await ListDirTool(cwd=tmp_path)(ToolCall(id="c1", name="list_dir", arguments={}))
        assert result.metadata is not None
        assert result.metadata["entries"] == 2

    async def test_glob_metadata(self, tmp_path: Path) -> None:
        from harness.tools.fs import GlobTool

        for i in range(3):
            (tmp_path / f"f{i}.py").write_text("", encoding="utf-8")
        result = await GlobTool(cwd=tmp_path)(
            ToolCall(id="c1", name="glob", arguments={"pattern": "*.py"})
        )
        assert result.metadata is not None
        assert result.metadata["matches"] == 3
        assert result.metadata["capped"] is False

    async def test_glob_capped_metadata(self, tmp_path: Path) -> None:
        from harness.tools.fs import GlobTool

        for i in range(5):
            (tmp_path / f"f{i}.txt").write_text("", encoding="utf-8")
        result = await GlobTool(cwd=tmp_path, max_results=2)(
            ToolCall(id="c1", name="glob", arguments={"pattern": "*.txt"})
        )
        assert result.metadata is not None
        assert result.metadata["capped"] is True

    async def test_shell_metadata(self, tmp_path: Path) -> None:
        from harness.tools.shell import ShellTool

        result = await ShellTool(cwd=tmp_path)(
            ToolCall(id="c1", name="shell", arguments={"command": "printf hello"})
        )
        assert result.metadata is not None
        assert result.metadata["exit_code"] == 0
        assert result.metadata["stdout_bytes"] == len("hello")
        assert result.metadata["timed_out"] is False
        assert isinstance(result.metadata["duration_ms"], int)

    async def test_shell_timeout_metadata(self, tmp_path: Path) -> None:
        import sys

        from harness.tools.shell import ShellTool

        result = await ShellTool(cwd=tmp_path, default_timeout=0.1)(
            ToolCall(
                id="c1",
                name="shell",
                arguments={"command": f"{sys.executable} -c 'import time; time.sleep(2)'"},
            )
        )
        assert result.metadata is not None
        assert result.metadata["timed_out"] is True
        assert result.is_error is True

    async def test_fetch_url_metadata(self) -> None:
        import httpx

        from harness.tools.web import FetchUrlTool

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _r: httpx.Response(
                    200, headers={"content-type": "text/plain"}, content=b"hello"
                )
            )
        ) as client:
            tool = FetchUrlTool(client=client)
            result = await tool(
                ToolCall(
                    id="c1",
                    name="fetch_url",
                    arguments={"url": "https://example.test/x"},
                )
            )
        assert result.metadata is not None
        assert result.metadata["status_code"] == 200
        assert result.metadata["content_type"] == "text/plain"
        assert result.metadata["bytes"] == len("hello")
        assert result.metadata["url"] == "https://example.test/x"
