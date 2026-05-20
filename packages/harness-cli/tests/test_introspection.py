"""Tests for `harness providers` and `harness tools` introspection commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main


def _run(cli_args: list[str]) -> Result:
    return CliRunner().invoke(cli_main.app, cli_args)


class TestProvidersList:
    def test_lists_known_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        result = _run(["providers", "list"])
        assert result.exit_code == 0
        assert "ollama" in result.stdout
        assert "openrouter" in result.stdout
        assert "missing OPENROUTER_API_KEY" in result.stdout

    def test_openrouter_ready_when_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        result = _run(["providers", "list"])
        assert result.exit_code == 0
        assert "ready" in result.stdout


class TestProvidersCapabilities:
    def test_ollama_capabilities(self) -> None:
        result = _run(["providers", "capabilities", "ollama"])
        assert result.exit_code == 0
        assert "streaming" in result.stdout
        assert "tool_use" in result.stdout

    def test_unknown_provider_exits_2(self) -> None:
        result = _run(["providers", "capabilities", "nope"])
        assert result.exit_code == 2

    def test_openrouter_without_key_exits_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        result = _run(["providers", "capabilities", "openrouter"])
        assert result.exit_code == 2


class TestToolsList:
    def test_includes_every_built_in_tool(self, tmp_path: Path) -> None:
        result = _run(["tools", "list", "--cwd", str(tmp_path)])
        assert result.exit_code == 0
        for needle in (
            "read_file",
            "write_file",
            "edit_file",
            "list_dir",
            "glob",
            "shell",
            "fetch_url",
        ):
            assert needle in result.stdout

    def test_shows_effective_approval(self, tmp_path: Path) -> None:
        result = _run(["tools", "list", "--cwd", str(tmp_path)])
        assert result.exit_code == 0
        # Defaults: read_file=auto, write_file/shell=prompt
        assert "auto" in result.stdout
        assert "prompt" in result.stdout

    def test_config_overrides_apply(self, tmp_path: Path) -> None:
        cfg = tmp_path / "config.toml"
        cfg.write_text('[approval]\nshell = "deny"\n', encoding="utf-8")
        result = _run(["tools", "list", "--cwd", str(tmp_path), "--config", str(cfg)])
        assert result.exit_code == 0
        # `deny` appears for shell now.
        assert "deny" in result.stdout
