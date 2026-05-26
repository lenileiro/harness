"""TOML config loader for the Harness CLI.

Location follows XDG: `$XDG_CONFIG_HOME/harness/config.toml`, or
`~/.config/harness/config.toml` otherwise. Missing file = empty config (all
defaults). The file is intentionally optional — every setting also has a CLI
flag or sensible default.

Example:

```toml
[default]
provider = "ollama"
model    = "llama3.2"

[provider.ollama]
base_url = "http://localhost:11434"

[provider.openrouter]
http_referer = "https://example.com"
x_title      = "MyApp"

[approval]
shell      = "prompt"
write_file = "prompt"
edit_file  = "prompt"
fetch_url  = "prompt"
read_file  = "auto"
list_dir   = "auto"
glob       = "auto"
```
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

from harness.core import ApprovalDecision

_VALID_DECISIONS: tuple[str, ...] = get_args(ApprovalDecision)


def default_config_path() -> Path:
    """`$XDG_CONFIG_HOME/harness/config.toml` or `~/.config/harness/config.toml`."""
    base = os.environ.get("XDG_CONFIG_HOME")
    config_home = Path(base) if base else Path.home() / ".config"
    return config_home / "harness" / "config.toml"


@dataclass(frozen=True, slots=True)
class ResearchSchedulerConfig:
    max_steps: int | None = None
    max_risk: str | None = None
    base_branch: str | None = None
    create_branch: bool | None = None
    commit: bool | None = None
    push: bool | None = None
    open_pr: bool | None = None
    draft_pr: bool | None = None


@dataclass(frozen=True, slots=True)
class MissionSchedulerConfig:
    max_steps: int | None = None
    auto_complete: bool | None = None


@dataclass(frozen=True, slots=True)
class MissionRoleConfig:
    model: str | None = None
    brief: str | None = None


@dataclass(frozen=True, slots=True)
class MissionRoleDefaults:
    planner: MissionRoleConfig = field(default_factory=MissionRoleConfig)
    worker: MissionRoleConfig = field(default_factory=MissionRoleConfig)
    validator: MissionRoleConfig = field(default_factory=MissionRoleConfig)
    reporter: MissionRoleConfig = field(default_factory=MissionRoleConfig)


@dataclass(frozen=True, slots=True)
class HarnessConfig:
    """Resolved CLI configuration.

    All fields default to None so the CLI can layer flags > config > builtin
    defaults without ambiguity over which level set a value.
    """

    default_provider: str | None = None
    default_model: str | None = None
    provider_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    approval: dict[str, ApprovalDecision] = field(default_factory=dict)
    plugins_enabled: tuple[str, ...] = ()
    plugins_disabled: tuple[str, ...] = ()
    include_plugin_entry_points: bool = False
    research_scheduler: ResearchSchedulerConfig = field(default_factory=ResearchSchedulerConfig)
    mission_scheduler: MissionSchedulerConfig = field(default_factory=MissionSchedulerConfig)
    mission_roles: MissionRoleDefaults = field(default_factory=MissionRoleDefaults)

    def provider(self, name: str) -> dict[str, Any]:
        """Return the per-provider settings dict (empty if unset)."""
        return self.provider_settings.get(name, {})


class ConfigError(ValueError):
    """Raised when the config file exists but is malformed."""


def load_config(path: Path | None = None) -> HarnessConfig:
    """Read and validate the config TOML; missing file → empty `HarnessConfig`."""
    target = path or default_config_path()
    if not target.exists():
        return HarnessConfig()

    try:
        with target.open("rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"could not parse {target}: {exc}") from exc
    except OSError as exc:
        raise ConfigError(f"could not read {target}: {exc}") from exc

    default_section = raw.get("default", {})
    if not isinstance(default_section, dict):
        raise ConfigError("`[default]` must be a table")

    default_provider = default_section.get("provider")
    default_model = default_section.get("model")
    if default_provider is not None and not isinstance(default_provider, str):
        raise ConfigError("`default.provider` must be a string")
    if default_model is not None and not isinstance(default_model, str):
        raise ConfigError("`default.model` must be a string")

    provider_section = raw.get("provider", {})
    if not isinstance(provider_section, dict):
        raise ConfigError("`[provider]` must be a table")
    provider_settings: dict[str, dict[str, Any]] = {}
    for name, settings in provider_section.items():
        if not isinstance(settings, dict):
            raise ConfigError(f"`[provider.{name}]` must be a table")
        provider_settings[name] = dict(settings)

    approval_section = raw.get("approval", {})
    if not isinstance(approval_section, dict):
        raise ConfigError("`[approval]` must be a table")
    approval: dict[str, ApprovalDecision] = {}
    for tool_name, decision in approval_section.items():
        if decision not in _VALID_DECISIONS:
            raise ConfigError(
                f"`approval.{tool_name}` = {decision!r} is not one of {_VALID_DECISIONS}"
            )
        approval[tool_name] = decision

    plugins_section = raw.get("plugins", {})
    if not isinstance(plugins_section, dict):
        raise ConfigError("`[plugins]` must be a table")

    def _string_list(name: str) -> tuple[str, ...]:
        value = plugins_section.get(name, [])
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ConfigError(f"`plugins.{name}` must be an array of strings")
        return tuple(value)

    include_plugin_entry_points = plugins_section.get("include_entry_points", False)
    if not isinstance(include_plugin_entry_points, bool):
        raise ConfigError("`plugins.include_entry_points` must be a boolean")

    scheduler_section = raw.get("research_scheduler", {})
    if not isinstance(scheduler_section, dict):
        raise ConfigError("`[research_scheduler]` must be a table")

    def _optional_bool(name: str) -> bool | None:
        value = scheduler_section.get(name)
        if value is None:
            return None
        if not isinstance(value, bool):
            raise ConfigError(f"`research_scheduler.{name}` must be a boolean")
        return value

    max_steps = scheduler_section.get("max_steps")
    if max_steps is not None and (not isinstance(max_steps, int) or max_steps < 1):
        raise ConfigError("`research_scheduler.max_steps` must be a positive integer")

    max_risk = scheduler_section.get("max_risk")
    if max_risk is not None and not isinstance(max_risk, str):
        raise ConfigError("`research_scheduler.max_risk` must be a string")

    base_branch = scheduler_section.get("base_branch")
    if base_branch is not None and not isinstance(base_branch, str):
        raise ConfigError("`research_scheduler.base_branch` must be a string")

    mission_scheduler_section = raw.get("mission_scheduler", {})
    if not isinstance(mission_scheduler_section, dict):
        raise ConfigError("`[mission_scheduler]` must be a table")

    mission_max_steps = mission_scheduler_section.get("max_steps")
    if mission_max_steps is not None and (
        not isinstance(mission_max_steps, int) or mission_max_steps < 1
    ):
        raise ConfigError("`mission_scheduler.max_steps` must be a positive integer")

    mission_auto_complete = mission_scheduler_section.get("auto_complete")
    if mission_auto_complete is not None and not isinstance(mission_auto_complete, bool):
        raise ConfigError("`mission_scheduler.auto_complete` must be a boolean")

    mission_roles_section = raw.get("mission_roles", {})
    if not isinstance(mission_roles_section, dict):
        raise ConfigError("`[mission_roles]` must be a table")

    def _mission_role_config(role: str) -> MissionRoleConfig:
        section = mission_roles_section.get(role, {})
        if not isinstance(section, dict):
            raise ConfigError(f"`[mission_roles.{role}]` must be a table")
        model = section.get("model")
        if model is not None and not isinstance(model, str):
            raise ConfigError(f"`mission_roles.{role}.model` must be a string")
        brief = section.get("brief")
        if brief is not None and not isinstance(brief, str):
            raise ConfigError(f"`mission_roles.{role}.brief` must be a string")
        return MissionRoleConfig(model=model, brief=brief)

    return HarnessConfig(
        default_provider=default_provider,
        default_model=default_model,
        provider_settings=provider_settings,
        approval=approval,
        plugins_enabled=_string_list("enabled"),
        plugins_disabled=_string_list("disabled"),
        include_plugin_entry_points=include_plugin_entry_points,
        research_scheduler=ResearchSchedulerConfig(
            max_steps=max_steps,
            max_risk=max_risk,
            base_branch=base_branch,
            create_branch=_optional_bool("create_branch"),
            commit=_optional_bool("commit"),
            push=_optional_bool("push"),
            open_pr=_optional_bool("open_pr"),
            draft_pr=_optional_bool("draft_pr"),
        ),
        mission_scheduler=MissionSchedulerConfig(
            max_steps=mission_max_steps,
            auto_complete=mission_auto_complete,
        ),
        mission_roles=MissionRoleDefaults(
            planner=_mission_role_config("planner"),
            worker=_mission_role_config("worker"),
            validator=_mission_role_config("validator"),
            reporter=_mission_role_config("reporter"),
        ),
    )


__all__ = [
    "ConfigError",
    "HarnessConfig",
    "MissionRoleConfig",
    "MissionRoleDefaults",
    "MissionSchedulerConfig",
    "ResearchSchedulerConfig",
    "default_config_path",
    "load_config",
]
