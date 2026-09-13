"""Opt-in live autonomy check: Claude proposes calls, Harness executes every tool.

Run explicitly with HARNESS_CLAUDE_TEST_BINARY=/absolute/path/to/claude.
This exercises the shipped `harness.adapters.claude.ClaudeAdapter`; the wrapper
below only records turns and enforces the fixture's caps. Maximum: eight print
processes at $0.30 each, 90 seconds per process and 300 seconds for the complete
fixture. Claude runs in an empty sibling directory with native tools, skills and
MCP disabled by the adapter itself.
"""

from __future__ import annotations

import asyncio
import json
import os
import secrets
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from harness.adapters.claude import ClaudeAdapter
from harness.cli.config import HarnessConfig
from harness.cli.runtime_agent import build_agent
from harness.cli.runtime_helpers import AUTONOMOUS_CONTEXT_POLICY
from harness.core import (
    Capabilities,
    Done,
    Event,
    Message,
    RunRequest,
    ToolCallEvent,
    ToolRegistry,
    ToolResultEvent,
)
from harness.core.errors import ConfigurationError
from harness.storage.sqlite import SQLiteStorage
from harness.tools.fs import ListDirTool, ReadFileTool

_BINARY = os.environ.get("HARNESS_CLAUDE_TEST_BINARY", "")
pytestmark = pytest.mark.skipif(
    not _BINARY or os.name != "posix",
    reason="explicit HARNESS_CLAUDE_TEST_BINARY and POSIX process-group ownership required",
)


class _RecordingClaudeAdapter:
    """Thin recorder around the shipped adapter: caps spend, keeps evidence."""

    name = "claude"

    def __init__(self, binary: str, *, cwd: Path):
        self.inner = ClaudeAdapter(
            claude_bin=binary,
            cwd=cwd,
            timeout=90.0,
            idle_timeout=60.0,
            effort="low",
            max_budget_usd=0.30,
        )
        self.turns = 0
        self.proposed_calls: list[dict[str, Any]] = []
        self.tool_schemas: list[list[dict[str, Any]]] = []

    @property
    def cost_usd(self) -> float:
        return self.inner.cost_usd

    async def capabilities(self) -> Capabilities:
        return await self.inner.capabilities()

    async def cancel(self, session_id: str) -> None:
        await self.inner.cancel(session_id)

    def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[Event]:
        if self.turns >= 8:
            raise ConfigurationError("Live Claude fixture exhausted its eight-call budget")
        self.turns += 1
        self.tool_schemas.append(tools or [])
        names = {tool["function"]["name"] for tool in tools or []}
        assert "clarify" not in names
        system = "\n\n".join(
            message.content or "" for message in messages if message.role == "system"
        )
        assert AUTONOMOUS_CONTEXT_POLICY in system
        return self._record(
            self.inner.stream(model=model, messages=messages, tools=tools, **kwargs), names
        )

    async def _record(self, events: AsyncIterator[Event], names: set[str]) -> AsyncIterator[Event]:
        async for event in events:
            if isinstance(event, ToolCallEvent):
                assert event.call.name in names
                self.proposed_calls.append(event.call.model_dump())
            yield event


async def test_claude_print_discovers_context_through_actual_harness_tools(tmp_path: Path):
    binary = Path(_BINARY)
    assert binary.is_absolute() and await asyncio.to_thread(binary.is_file), (
        "HARNESS_CLAUDE_TEST_BINARY must name an installed executable"
    )
    workspace, cli_cwd = tmp_path / "workspace", tmp_path / "model-only"
    workspace.mkdir()
    cli_cwd.mkdir()
    (workspace / "catalog").mkdir()
    directory = workspace / "records" / secrets.token_hex(5)
    directory.mkdir(parents=True)
    source = directory / f"{secrets.token_hex(5)}.json"
    relative_source = source.relative_to(workspace).as_posix()
    expected = "release-" + secrets.token_hex(8)
    (workspace / f"release-guide-{secrets.token_hex(3)}.md").write_text(
        "Release records\n\nThe active release is selected by catalog/state-map.json. "
        "Historical labels are not authoritative.\n",
        encoding="utf-8",
    )
    (workspace / "catalog/state-map.json").write_text(
        json.dumps({"active_record": relative_source, "historical_label": "retired-example"}),
        encoding="utf-8",
    )
    source.write_text(json.dumps({"release_label": expected, "status": "active"}), encoding="utf-8")
    snapshot = {
        path.relative_to(workspace): path.read_bytes()
        for path in workspace.rglob("*")
        if path.is_file()
    }
    adapter = _RecordingClaudeAdapter(str(binary), cwd=cli_cwd)
    storage = SQLiteStorage(path=tmp_path / "sessions.db")

    def discovery_tools(cwd: Path) -> ToolRegistry:
        registry = ToolRegistry()
        registry.register(ListDirTool(cwd=cwd))
        registry.register(ReadFileTool(cwd=cwd))
        return registry

    config = HarnessConfig(approval={"list_dir": "auto", "read_file": "auto"})
    agent = build_agent(
        chain=[adapter.name],
        base_url=None,
        model=os.environ.get("HARNESS_CLAUDE_TEST_MODEL", "sonnet"),
        storage=storage,
        cwd=workspace,
        config=config,
        yes=False,
        inbox=True,
        pause_on_approval=True,
        approval_store=storage,
        activity_store=storage,
        build_adapter=lambda *args, **kwargs: adapter,
        build_tools=discovery_tools,
        build_search_fn=lambda: None,
        console=None,
        auxiliary_tools_enabled=False,
        project_context_enabled=False,
        memory_tools_enabled=False,
        skip_builtin_verify_before_done=True,
    )
    try:
        assert not config.clarification_enabled and agent.question_store_factory is None
        request = RunRequest(
            session_id="live-claude-discovery",
            prompt="What is the active release label for this workspace? Cite the file that establishes it.",
            max_steps=8,
        )
        async with asyncio.timeout(300):
            events = [event async for event in agent.run(request)]
        session = await storage.get(request.session_id)
        assert session is not None and session.status == "done"
        final = next(event for event in reversed(events) if isinstance(event, Done))
        assert final.final_message is not None
        answer = final.final_message.content or ""
        results = [event.result for event in events if isinstance(event, ToolResultEvent)]
        assert any(result.name == "list_dir" and not result.is_error for result in results)
        assert any(
            result.name == "read_file" and not result.is_error and expected in result.content
            for result in results
        )
        assert expected in answer and relative_source in answer
        assert adapter.proposed_calls and all(
            call["name"] in {"list_dir", "read_file"} for call in adapter.proposed_calls
        )
        assert await storage.list_approvals(session_id=request.session_id) == []
        assert agent.question_store is None and "pending_question_id" not in session.metadata
        assert snapshot == {
            path.relative_to(workspace): path.read_bytes()
            for path in workspace.rglob("*")
            if path.is_file()
        }
        assert adapter.cost_usd <= 2.40
        report = {
            "model_turns": adapter.turns,
            "reported_cost_usd": adapter.cost_usd,
            "tool_calls": adapter.proposed_calls,
            "successful_tool_results": sum(not result.is_error for result in results),
            "final_answer": answer,
        }
        report_path = tmp_path / "claude-live-result.json"
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("CLAUDE_LIVE_RESULT " + json.dumps({**report, "report_path": str(report_path)}))
    finally:
        await adapter.cancel("live-claude-discovery")
        await agent.aclose()
        await storage.close()
