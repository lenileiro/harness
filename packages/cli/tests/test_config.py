"""Tests for the CLI config loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.cli.config import (
    ConfigError,
    HarnessConfig,
    default_config_path,
    load_config,
)


class TestDefaultConfigPath:
    def test_uses_xdg_config_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/cfg-test")
        assert default_config_path() == Path("/tmp/cfg-test/harness/config.toml")

    def test_falls_back_to_home_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        path = default_config_path()
        assert path.parts[-3:] == (".config", "harness", "config.toml")


class TestLoadConfig:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        cfg = load_config(tmp_path / "no-such.toml")
        assert isinstance(cfg, HarnessConfig)
        assert cfg.default_provider is None
        assert cfg.default_model is None
        assert cfg.provider_settings == {}
        assert cfg.approval == {}
        assert cfg.plugins_enabled == ()
        assert cfg.plugins_disabled == ()
        assert cfg.include_plugin_entry_points is False

    def test_full_config(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text(
            """
            [default]
            provider = "openrouter"
            model = "anthropic/claude-3.5-sonnet"

            [provider.ollama]
            base_url = "http://lm:11434"

            [provider.openrouter]
            http_referer = "https://example.com"
            x_title = "MyApp"

            [approval]
            shell = "prompt"
            write_file = "prompt"
            read_file = "auto"

            [plugins]
            enabled = ["workspace-demo"]
            disabled = ["legacy-tools"]
            include_entry_points = true
            """,
            encoding="utf-8",
        )
        cfg = load_config(target)
        assert cfg.default_provider == "openrouter"
        assert cfg.default_model == "anthropic/claude-3.5-sonnet"
        assert cfg.provider("ollama") == {"base_url": "http://lm:11434"}
        assert cfg.provider("openrouter")["x_title"] == "MyApp"
        assert cfg.approval == {
            "shell": "prompt",
            "write_file": "prompt",
            "read_file": "auto",
        }
        assert cfg.plugins_enabled == ("workspace-demo",)
        assert cfg.plugins_disabled == ("legacy-tools",)
        assert cfg.include_plugin_entry_points is True
        assert cfg.research_scheduler.max_steps is None

    def test_research_scheduler_config(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text(
            """
            [research_scheduler]
            max_steps = 7
            max_risk = "low"
            base_branch = "develop"
            create_branch = true
            commit = true
            push = false
            open_pr = false
            draft_pr = true
            """,
            encoding="utf-8",
        )
        cfg = load_config(target)
        assert cfg.research_scheduler.max_steps == 7
        assert cfg.research_scheduler.max_risk == "low"
        assert cfg.research_scheduler.base_branch == "develop"
        assert cfg.research_scheduler.create_branch is True
        assert cfg.research_scheduler.commit is True
        assert cfg.research_scheduler.push is False
        assert cfg.research_scheduler.open_pr is False
        assert cfg.research_scheduler.draft_pr is True

    def test_invalid_approval_value_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text('[approval]\nshell = "sometimes"\n', encoding="utf-8")
        with pytest.raises(ConfigError, match=r"approval\.shell"):
            load_config(target)

    def test_bad_default_type_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text("[default]\nprovider = 5\n", encoding="utf-8")
        with pytest.raises(ConfigError, match="provider"):
            load_config(target)

    def test_malformed_toml_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text("this is not [valid toml", encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(target)

    def test_provider_section_must_be_table(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text('[provider]\nollama = "not-a-table"\n', encoding="utf-8")
        with pytest.raises(ConfigError, match=r"provider\.ollama"):
            load_config(target)

    def test_plugins_section_validates_types(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text(
            """
            [plugins]
            enabled = "demo"
            """,
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match=r"plugins\.enabled"):
            load_config(target)

    def test_research_scheduler_section_validates_types(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text(
            """
            [research_scheduler]
            max_steps = 0
            """,
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match=r"research_scheduler\.max_steps"):
            load_config(target)
