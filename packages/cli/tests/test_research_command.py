from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from harness.cli.config import HarnessConfig
from harness.cli.research_commands import build_research_prompt, research_command


def test_build_research_prompt_includes_topic() -> None:
    prompt = build_research_prompt(topic="compare sqlite and postgres for local agents")
    assert "compare sqlite and postgres" in prompt
    assert "Return JSON only" in prompt


def test_research_command_uses_research_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_once(**kwargs: object) -> str:
        captured.update(kwargs)
        return (
            '{"summary":"Two key tradeoffs.","findings":["SQLite is simpler."],'
            '"open_questions":["How much concurrency is needed?"],'
            '"sources":[{"title":"SQLite docs","url":"https://sqlite.org","excerpt":"Reliable embedded DB."}]}'
        )

    def fake_run_async(awaitable: object) -> object:
        return asyncio.run(awaitable)  # type: ignore[arg-type]

    stream = StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None)

    research_command(
        topic="compare sqlite and postgres for local agents",
        model="test-model",
        provider="ollama",
        failover=None,
        base_url=None,
        cwd=tmp_path,
        max_steps=8,
        max_output_tokens=None,
        db=None,
        in_memory=True,
        yes=True,
        verbose=False,
        json_output=False,
        config_path=None,
        console=console,
        load_cli_config=lambda _path: HarnessConfig(),
        resolve_chain=lambda **_kwargs: ["ollama"],
        run_async=fake_run_async,
        run_once=fake_run_once,
    )

    assert captured["domain"] == "research"
    assert captured["profile"] == "minimal"
    assert captured["require_tools"] is False
    assert captured["loop_detect"] is True
    output = stream.getvalue()
    assert "Two key tradeoffs." in output
    assert "Findings" in output
    assert "Sources" in output
