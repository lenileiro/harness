"""Tests for `harness providers`, `tools`, and `plugins` introspection commands."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from click.testing import Result
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main


def _run(cli_args: list[str]) -> Result:
    return CliRunner().invoke(cli_main.app, cli_args)


class TestProvidersList:
    def test_lists_known_providers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        monkeypatch.setattr("harness.cli.introspection.codex_cli_available", lambda: False)
        monkeypatch.setattr("harness.cli.introspection.inspect_codex_cli_auth", lambda: None)
        result = _run(["providers", "list"])
        assert result.exit_code == 0
        assert "ollama" in result.stdout
        assert "codex" in result.stdout
        assert "openai" in result.stdout
        assert "openrouter" in result.stdout
        assert "missing OPENAI_API_KEY" in result.stdout
        assert "missing OPENROUTER_API_KEY" in result.stdout

    def test_codex_ready_when_cli_and_auth_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("harness.cli.introspection.codex_cli_available", lambda: True)
        monkeypatch.setattr(
            "harness.cli.introspection.inspect_codex_cli_auth",
            lambda: {
                "auth_mode": "chatgpt",
                "has_openai_api_key": False,
                "has_access_token": True,
            },
        )
        result = _run(["providers", "list"])
        assert result.exit_code == 0
        assert "codex" in result.stdout
        assert "ChatGPT/Codex OAuth present" in result.stdout

    def test_openai_ready_when_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setattr("harness.cli.introspection.codex_cli_available", lambda: False)
        monkeypatch.setattr("harness.cli.introspection.inspect_codex_cli_auth", lambda: None)
        result = _run(["providers", "list"])
        assert result.exit_code == 0
        assert "ready" in result.stdout

    def test_openai_ready_when_codex_auth_has_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            "harness.cli.introspection.load_codex_openai_api_key",
            lambda: "codex-key",
        )
        result = _run(["providers", "list"])
        assert result.exit_code == 0
        assert "openai" in result.stdout
        assert "codex auth: OPENAI_API_KEY set" in result.stdout

    def test_openai_lists_chatgpt_codex_auth_as_insufficient(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(
            "harness.cli.introspection.inspect_codex_openai_auth",
            lambda: {"auth_mode": "chatgpt", "has_openai_api_key": False},
        )
        monkeypatch.setattr(
            "harness.cli.introspection.load_codex_openai_api_key",
            lambda: None,
        )
        result = _run(["providers", "list"])
        assert result.exit_code == 0
        assert "ChatGPT OAuth present" in result.stdout
        assert "not usable for model calls" in result.stdout

    def test_openrouter_ready_when_key_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        result = _run(["providers", "list"])
        assert result.exit_code == 0
        assert "ready" in result.stdout


class TestProvidersCapabilities:
    def test_openai_without_key_exits_2(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        result = _run(["providers", "capabilities", "openai"])
        assert result.exit_code == 2

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

    def test_workspace_plugin_tools_are_included(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "tools_demo_provider.py").write_text(
            textwrap.dedent(
                """
                from harness.core.tool_entry import ToolSpec


                class DemoTool:
                    name = "demo_tool"
                    description = "Tool from workspace plugin"
                    parameters_schema = {"type": "object", "properties": {}}
                    approval = "auto"

                    async def __call__(self, call):
                        raise NotImplementedError


                class DemoProvider:
                    def specs(self):
                        return [ToolSpec(name="demo_tool", factory=lambda ctx: DemoTool())]
                """
            ),
            encoding="utf-8",
        )
        plugin_dir = tmp_path / ".harness" / "plugins"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "demo.toml").write_text(
            'name = "workspace-demo"\nprovider = "tools_demo_provider:DemoProvider"\n',
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        result = _run(["tools", "list", "--cwd", str(tmp_path)])
        assert result.exit_code == 0
        assert "demo_tool" in result.stdout


class TestPluginsList:
    def test_lists_builtin_and_workspace_plugins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "plugins_demo_provider.py").write_text(
            "class DemoProvider:\n    def specs(self):\n        return []\n",
            encoding="utf-8",
        )
        plugin_dir = tmp_path / ".harness" / "plugins"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "demo.toml").write_text(
            textwrap.dedent(
                """
                name = "workspace-demo"
                provider = "plugins_demo_provider:DemoProvider"
                description = "Workspace plugin"
                """
            ),
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        result = _run(["plugins", "list", "--cwd", str(tmp_path)])
        assert result.exit_code == 0
        assert "builtin" in result.stdout
        assert "workspace-demo" in result.stdout
        assert "tool" in result.stdout
        assert "workspace" in result.stdout

    def test_filters_plugins_by_kind(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / ".harness" / "plugins"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "exp.toml").write_text(
            textwrap.dedent(
                """
                name = "experience-demo"
                kind = "experience"
                provider = "experience_demo:Provider"
                """
            ),
            encoding="utf-8",
        )
        (plugin_dir / "domain.toml").write_text(
            textwrap.dedent(
                """
                name = "domain-demo"
                kind = "domain_profile"
                provider = "domain_demo:Provider"
                """
            ),
            encoding="utf-8",
        )

        result = _run(["plugins", "list", "--cwd", str(tmp_path), "--kind", "experience"])
        assert result.exit_code == 0
        assert "experience-demo" in result.stdout
        assert "domain-demo" not in result.stdout

    def test_invalid_plugin_kind_exits_2(self, tmp_path: Path) -> None:
        result = _run(["plugins", "list", "--cwd", str(tmp_path), "--kind", "nope"])
        assert result.exit_code == 2


class TestPluginsValidate:
    def test_validates_workspace_plugins(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "plugins_demo_provider.py").write_text(
            "class DemoProvider:\n    def specs(self):\n        return []\n",
            encoding="utf-8",
        )
        plugin_dir = tmp_path / ".harness" / "plugins"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "demo.toml").write_text(
            textwrap.dedent(
                """
                name = "workspace-demo"
                provider = "plugins_demo_provider:DemoProvider"
                description = "Workspace plugin"
                """
            ),
            encoding="utf-8",
        )
        monkeypatch.syspath_prepend(str(tmp_path))

        result = _run(["plugins", "validate", "--cwd", str(tmp_path)])
        assert result.exit_code == 0
        assert "workspace-demo" in result.stdout
        assert "ok" in result.stdout

    def test_validate_exits_1_for_broken_plugin(self, tmp_path: Path) -> None:
        plugin_dir = tmp_path / ".harness" / "plugins"
        plugin_dir.mkdir(parents=True)
        (plugin_dir / "broken.toml").write_text(
            textwrap.dedent(
                """
                name = "broken-demo"
                provider = "missing_plugin_module:DemoProvider"
                """
            ),
            encoding="utf-8",
        )

        result = _run(["plugins", "validate", "--cwd", str(tmp_path), "--kind", "tool"])
        assert result.exit_code == 1
        assert "broken-demo" in result.stdout
        assert "error" in result.stdout
