from __future__ import annotations

import asyncio
import json
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.cli.config import HarnessConfig
from harness.cli.research_commands import (
    build_catch_up_prompt,
    build_context_packet_prompt,
    build_research_prompt,
    catch_up_command,
    context_packet_command,
    research_command,
)


def test_build_research_prompt_includes_topic() -> None:
    prompt = build_research_prompt(topic="compare sqlite and postgres for local agents")
    assert "compare sqlite and postgres" in prompt
    assert "Return JSON only" in prompt


def test_build_catch_up_prompt_includes_mode_and_mental_model() -> None:
    prompt = build_catch_up_prompt(
        topic="how scheduler reminders flow through WhatsApp",
        mode="feature-trace",
    )
    assert "how scheduler reminders flow through WhatsApp" in prompt
    assert "Exploration mode: feature-trace" in prompt
    assert "mental model" in prompt
    assert "Stay read-only" in prompt
    assert "Flow" in prompt


def test_build_catch_up_prompt_rejects_unknown_mode() -> None:
    with pytest.raises(Exception, match="--mode must be one of"):
        build_catch_up_prompt(topic="gateway", mode="vibes")


def test_build_context_packet_prompt_includes_context_engine_checks() -> None:
    prompt = build_context_packet_prompt(task="implement a Zendesk integration")
    assert "implement a Zendesk integration" in prompt
    assert "context-engine work" in prompt
    assert "not naive RAG" in prompt
    assert "first plausible match" in prompt
    assert "Sources of truth" in prompt
    assert "Conflict checks" in prompt
    assert "Boundaries and permissions" in prompt
    assert "token-optimized" in prompt


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
    assert captured["profile"] == "bare"
    assert captured["require_tools"] is False
    assert captured["loop_detect"] is True
    output = stream.getvalue()
    assert "Two key tradeoffs." in output
    assert "Findings" in output
    assert "Sources" in output


def test_catch_up_command_uses_comprehension_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_once(**kwargs: object) -> str:
        captured.update(kwargs)
        return "## Mental model\n- Scheduler reminders are routed through the gateway."

    def fake_run_async(awaitable: object) -> object:
        return asyncio.run(awaitable)  # type: ignore[arg-type]

    catch_up_command(
        topic="how scheduler reminders flow through WhatsApp",
        mode="feature-trace",
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
        config_path=None,
        console=Console(file=StringIO(), force_terminal=False, color_system=None),
        load_cli_config=lambda _path: HarnessConfig(),
        resolve_chain=lambda **_kwargs: ["ollama"],
        run_async=fake_run_async,
        run_once=fake_run_once,
    )

    assert captured["domain"] == "comprehension"
    assert captured["profile"] == "bare"
    assert captured["verify"] == "none"
    assert "Exploration mode: feature-trace" in str(captured["prompt"])


def test_context_packet_command_uses_comprehension_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_once(**kwargs: object) -> str:
        captured.update(kwargs)
        return "## Context packet\n- Reuse the existing gateway factory."

    def fake_run_async(awaitable: object) -> object:
        return asyncio.run(awaitable)  # type: ignore[arg-type]

    context_packet_command(
        task="implement a Zendesk integration",
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
        config_path=None,
        console=Console(file=StringIO(), force_terminal=False, color_system=None),
        load_cli_config=lambda _path: HarnessConfig(),
        resolve_chain=lambda **_kwargs: ["ollama"],
        run_async=fake_run_async,
        run_once=fake_run_once,
    )

    assert captured["domain"] == "comprehension"
    assert captured["profile"] == "bare"
    assert captured["verify"] == "none"
    assert "context-engine work" in str(captured["prompt"])


def test_research_run_command_uses_injected_run_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_once(**kwargs: object) -> str:
        captured.update(kwargs)
        return json.dumps(
            {
                "summary": "Two key tradeoffs.",
                "findings": ["SQLite is simpler."],
                "open_questions": ["How much concurrency is needed?"],
                "sources": [
                    {
                        "title": "SQLite docs",
                        "url": "https://sqlite.org",
                        "excerpt": "Reliable embedded DB.",
                    }
                ],
            }
        )

    monkeypatch.setattr(cli_main, "_run_once", fake_run_once)

    result = CliRunner().invoke(
        cli_main.app,
        [
            "research",
            "run",
            "compare sqlite and postgres for local agents",
            "--cwd",
            str(tmp_path),
            "--in-memory",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["domain"] == "research"
    assert "Two key tradeoffs." in result.stdout


def test_research_catch_up_command_uses_injected_run_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_once(**kwargs: object) -> str:
        captured.update(kwargs)
        return "## Mental model\n- The gateway owns message routing."

    monkeypatch.setattr(cli_main, "_run_once", fake_run_once)

    result = CliRunner().invoke(
        cli_main.app,
        [
            "research",
            "catch-up",
            "how gateway routing works",
            "--mode",
            "architecture",
            "--cwd",
            str(tmp_path),
            "--in-memory",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["domain"] == "comprehension"
    assert "how gateway routing works" in str(captured["prompt"])


def test_research_context_packet_command_uses_injected_run_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_once(**kwargs: object) -> str:
        captured.update(kwargs)
        return "## Context packet\n- Reuse the existing integration pattern."

    monkeypatch.setattr(cli_main, "_run_once", fake_run_once)

    result = CliRunner().invoke(
        cli_main.app,
        [
            "research",
            "context-packet",
            "implement a Zendesk integration",
            "--cwd",
            str(tmp_path),
            "--in-memory",
            "--yes",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["domain"] == "comprehension"
    assert "implement a Zendesk integration" in str(captured["prompt"])
