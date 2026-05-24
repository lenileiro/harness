from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

from harness.cli import review_commands as review_mod
from harness.cli.config import HarnessConfig
from harness.cli.review_commands import build_review_prompt, review_command


def test_build_review_prompt_includes_base_and_diff() -> None:
    prompt = build_review_prompt(base="main", diff="diff --git a/x b/x")
    assert "main" in prompt
    assert "BEGIN DIFF" in prompt
    assert "diff --git" in prompt


def test_review_command_uses_code_review_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(review_mod, "_git_diff", lambda cwd, *, base: "diff --git a/x b/x")

    async def fake_run_once(**kwargs: object) -> str:
        captured.update(kwargs)
        return (
            '{"summary":"One issue.","findings":['
            '{"severity":"high","file":"src/app.py","line":3,'
            '"issue":"Bug","rationale":"Breaks logic"}]}'
        )

    def fake_run_async(awaitable: object) -> object:
        return asyncio.run(awaitable)  # type: ignore[arg-type]

    stream = StringIO()
    console = Console(file=stream, force_terminal=False, color_system=None)

    review_command(
        base="HEAD~1",
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

    assert captured["domain"] == "code-review"
    assert captured["profile"] == "bare"
    assert captured["require_tools"] is True
    assert "One issue." in stream.getvalue()
