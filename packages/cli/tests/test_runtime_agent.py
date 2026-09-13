from __future__ import annotations

import sqlite3
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

import pytest

from harness.cli.config import HarnessConfig
from harness.cli.runtime_agent import build_agent
from harness.cli.runtime_helpers import AUTONOMOUS_CONTEXT_POLICY
from harness.core import Capabilities, Done, Event, Message, RunRequest, ToolRegistry
from harness.storage.sqlite import SQLiteStorage


class _FakeAdapter:
    name = "fake"

    def __init__(self, cwd: Path) -> None:
        self.cwd = cwd

    def stream(self, **_kwargs: Any) -> Any:
        raise NotImplementedError

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=True)

    async def cancel(self, session_id: str) -> None:
        del session_id


def _build_runtime_agent(*, provider: str, adapter: _FakeAdapter, cwd: Path):
    return build_agent(
        chain=[provider],
        base_url=None,
        model="m",
        storage=object(),  # type: ignore[arg-type]
        cwd=cwd,
        config=HarnessConfig(),
        yes=True,
        build_adapter=lambda *_args, **_kwargs: adapter,
        build_tools=lambda _cwd: ToolRegistry(),
        build_search_fn=lambda: None,
        console=None,
        auxiliary_tools_enabled=False,
        project_context_enabled=False,
    )


def test_build_agent_aligns_codex_adapter_cwd_to_runtime_cwd(tmp_path: Path) -> None:
    adapter = _FakeAdapter(Path("/wrong"))

    agent = _build_runtime_agent(provider="codex", adapter=adapter, cwd=tmp_path)

    assert cast(_FakeAdapter, agent.adapters["codex"]).cwd == tmp_path.resolve()


def test_build_agent_does_not_rewrite_non_codex_adapter_cwd(tmp_path: Path) -> None:
    original_cwd = Path("/provider-default")
    adapter = _FakeAdapter(original_cwd)

    agent = _build_runtime_agent(provider="openrouter", adapter=adapter, cwd=tmp_path)

    assert cast(_FakeAdapter, agent.adapters["openrouter"]).cwd == original_cwd


@pytest.mark.parametrize("enabled", [False, True])
async def test_clarification_schema_and_database_require_operator_opt_in(
    tmp_path: Path, enabled: bool
):
    class RecordingAdapter(_FakeAdapter):
        def __init__(self, cwd: Path) -> None:
            super().__init__(cwd)
            self.calls: list[dict[str, Any]] = []

        async def stream(self, **kwargs: Any) -> AsyncIterator[Event]:
            self.calls.append(kwargs)
            yield Done(
                final_message=Message(
                    role="assistant", content="Completed from repository context."
                )
            )

    storage = SQLiteStorage(path=tmp_path / "sessions.db")
    adapter = RecordingAdapter(tmp_path)
    agent = build_agent(
        chain=["fake"],
        base_url=None,
        model="m",
        storage=storage,
        cwd=tmp_path,
        config=HarnessConfig(clarification_enabled=enabled),
        yes=True,
        build_adapter=lambda *args, **kwargs: adapter,
        build_tools=lambda path: ToolRegistry(),
        build_search_fn=lambda: None,
        console=None,
        auxiliary_tools_enabled=False,
        project_context_enabled=False,
        skip_builtin_verify_before_done=True,
        system_prompt="Specialist task instructions.",
    )
    try:
        assert (agent.question_store_factory is not None) is enabled
        [
            event
            async for event in agent.run(
                RunRequest(session_id="s", prompt="Complete the task using available context.")
            )
        ]
        names = {item["function"]["name"] for item in adapter.calls[0]["tools"]}
        system = adapter.calls[0]["messages"][0].content
        assert system.startswith(AUTONOMOUS_CONTEXT_POLICY)
        assert "Specialist task instructions." in system
        assert ("clarify" in names) is enabled
        assert (agent.question_store is not None) is enabled
        with sqlite3.connect(storage.path) as db:
            tables = {
                row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
        assert ("clarifications" in tables) is enabled
    finally:
        await agent.aclose()
        await storage.close()
