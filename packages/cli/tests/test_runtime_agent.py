from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from harness.cli.config import HarnessConfig
from harness.cli.runtime_agent import build_agent
from harness.core import Capabilities, ToolRegistry


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
