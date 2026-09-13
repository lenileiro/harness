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
    """These exercise the fallback chain, so they own the whole environment.

    `HARNESS_CONFIG` wins over both branches below and is set for every test by
    the isolation fixture in `packages/conftest.py`; clear it here.
    """

    def test_uses_xdg_config_home(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HARNESS_CONFIG", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/cfg-test")
        assert default_config_path() == Path("/tmp/cfg-test/harness/config.toml")

    def test_falls_back_to_home_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("HARNESS_CONFIG", raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        path = default_config_path()
        assert path.parts[-3:] == (".config", "harness", "config.toml")

    def test_harness_config_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HARNESS_CONFIG", "/tmp/explicit/config.toml")
        monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/cfg-test")
        assert default_config_path() == Path("/tmp/explicit/config.toml")


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
        assert cfg.clarification_enabled is False

    def test_clarification_requires_explicit_boolean_opt_in(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text("[clarification]\nenabled=true\n", encoding="utf-8")
        assert load_config(target).clarification_enabled is True
        target.write_text('[clarification]\nenabled="true"\n', encoding="utf-8")
        with pytest.raises(ConfigError, match="clarification"):
            load_config(target)

    def test_full_config(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text(
            """
            [default]
            provider = "openrouter"
            model = "anthropic/claude-3.5-sonnet"

            [provider.ollama]
            base_url = "http://lm:11434"

            [provider.openai]
            base_url = "https://api.openai.example/v1"

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
        assert cfg.provider("openai") == {"base_url": "https://api.openai.example/v1"}
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
        assert cfg.mission_scheduler.max_steps is None

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

    def test_mission_scheduler_config(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text(
            """
            [mission_scheduler]
            max_steps = 7
            auto_complete = true
            """,
            encoding="utf-8",
        )
        cfg = load_config(target)
        assert cfg.mission_scheduler.max_steps == 7
        assert cfg.mission_scheduler.auto_complete is True

    def test_mission_role_defaults_config(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text(
            """
            [mission_roles.planner]
            model = "gpt-planner"
            brief = "Plan before coding."

            [mission_roles.worker]
            model = "gpt-worker"

            [mission_roles.validator]
            brief = "Check assertions independently."
            """,
            encoding="utf-8",
        )
        cfg = load_config(target)
        assert cfg.mission_roles.planner.model == "gpt-planner"
        assert cfg.mission_roles.planner.brief == "Plan before coding."
        assert cfg.mission_roles.worker.model == "gpt-worker"
        assert cfg.mission_roles.validator.brief == "Check assertions independently."

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

    def test_mission_scheduler_section_validates_types(self, tmp_path: Path) -> None:
        target = tmp_path / "config.toml"
        target.write_text(
            """
            [mission_scheduler]
            max_steps = 0
            """,
            encoding="utf-8",
        )
        with pytest.raises(ConfigError, match=r"mission_scheduler\.max_steps"):
            load_config(target)


def test_suite_never_reads_the_developers_real_config() -> None:
    """A configured Harness install must not change what the tests do.

    `default_config_path()` falls back to `~/.config/harness/config.toml`, so
    without the autouse isolation in `packages/conftest.py` a developer who has
    run `harness setup` runs a different suite than CI does.
    """

    from pathlib import Path

    from harness.cli.config import default_config_path, load_config

    resolved = default_config_path()
    real = Path.home() / ".config" / "harness" / "config.toml"
    assert resolved != real, "tests are resolving the developer's real config"
    assert not resolved.exists(), "the isolated config path should not exist"

    config = load_config()
    assert config.default_provider is None
    assert config.default_model is None
    assert config.approval == {}
