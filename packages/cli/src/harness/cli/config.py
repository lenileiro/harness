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

    return HarnessConfig(
        default_provider=default_provider,
        default_model=default_model,
        provider_settings=provider_settings,
        approval=approval,
        plugins_enabled=_string_list("enabled"),
        plugins_disabled=_string_list("disabled"),
        include_plugin_entry_points=include_plugin_entry_points,
    )


__all__ = ["ConfigError", "HarnessConfig", "default_config_path", "load_config"]
