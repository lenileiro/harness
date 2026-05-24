from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from harness.cli.config import HarnessConfig
from harness.cli.docs_commands import build_docs_audit_prompt, docs_audit_command


def test_build_docs_audit_prompt_includes_focus() -> None:
    prompt = build_docs_audit_prompt(focus="plugin setup")
    assert "plugin setup" in prompt
    assert "Return JSON only" in prompt


def test_docs_audit_command_uses_docs_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    async def fake_run_once(**kwargs: object) -> str:
        captured.update(kwargs)
        return (
            '{"summary":"Docs need one update.",'
            '"findings":[{"severity":"medium","path":"README.md","issue":"Missing plugin example",'
            '"rationale":"Extension workflow is unclear"}],'
            '"missing_topics":["plugin setup"]}'
        )

    def fake_run_async(awaitable: object) -> object:
        return asyncio.run(awaitable)  # type: ignore[arg-type]

    stream = StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None)

    docs_audit_command(
        focus="plugin setup",
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

    assert captured["domain"] == "docs-audit"
    assert captured["profile"] == "bare"
    assert captured["require_tools"] is True
    output = stream.getvalue()
    assert "Docs need one update." in output
    assert "Findings" in output
    assert "Missing Topics" in output
