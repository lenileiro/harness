"""Discovery and loading for provider plugins."""

from __future__ import annotations

import importlib
import os
import tomllib
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from typing import Literal

from harness.core.extensions import (
    CriticProvider,
    DomainProfileProvider,
    ExperienceProvider,
    ToolProvider,
    VerifierProvider,
)

PluginKind = Literal["tool", "experience", "domain_profile", "verifier", "critic"]
PluginSource = Literal["bundled", "workspace", "user", "entry_point"]

_ENTRY_POINT_GROUPS: dict[PluginKind, str] = {
    "tool": "harness.tool_providers",
    "experience": "harness.experience_providers",
    "domain_profile": "harness.domain_profile_providers",
    "verifier": "harness.verifier_providers",
    "critic": "harness.critic_providers",
}
_VALID_KINDS = tuple(sorted(_ENTRY_POINT_GROUPS))


@dataclass(frozen=True, slots=True)
class ProviderPlugin:
    """Declarative description of a discovered provider plugin."""

    name: str
    provider_ref: str
    source: PluginSource
    kind: PluginKind = "tool"
    description: str | None = None
    path: Path | None = None
    enabled: bool = True


ToolProviderPlugin = ProviderPlugin
ExperienceProviderPlugin = ProviderPlugin
DomainProfileProviderPlugin = ProviderPlugin
VerifierProviderPlugin = ProviderPlugin
CriticProviderPlugin = ProviderPlugin


def default_user_plugins_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    config_home = Path(base) if base else Path.home() / ".config"
    return config_home / "harness" / "plugins"


def _manifest_paths(root: Path) -> list[Path]:
    if not root.exists():
        return []
    paths = sorted(root.glob("*.toml"))
    paths.extend(sorted(root.glob("*/plugin.toml")))
    return paths


def _coerce_kind(raw: object, *, path: Path) -> PluginKind:
    if not isinstance(raw, str) or raw not in _VALID_KINDS:
        raise ValueError(
            f"plugin manifest {path} has invalid `kind`; expected one of {_VALID_KINDS}"
        )
    return raw


def _parse_manifest(path: Path, *, source: PluginSource) -> ProviderPlugin:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"plugin manifest {path} must define a table")

    name = raw.get("name")
    provider_ref = raw.get("provider")
    if not isinstance(name, str) or not name:
        raise ValueError(f"plugin manifest {path} must define a string `name`")
    if not isinstance(provider_ref, str) or not provider_ref:
        raise ValueError(f"plugin manifest {path} must define a string `provider`")

    description = raw.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError(f"plugin manifest {path} has invalid `description`")

    enabled = raw.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"plugin manifest {path} has invalid `enabled`")

    kind = _coerce_kind(raw.get("kind", "tool"), path=path)

    return ProviderPlugin(
        name=name,
        provider_ref=provider_ref,
        source=source,
        kind=kind,
        description=description,
        path=path,
        enabled=enabled,
    )


def discover_manifest_plugins(
    root: Path,
    *,
    source: Literal["workspace", "user"],
    kind: PluginKind | None = None,
) -> list[ProviderPlugin]:
    plugins: list[ProviderPlugin] = []
    for path in _manifest_paths(root):
        plugin = _parse_manifest(path, source=source)
        if kind is not None and plugin.kind != kind:
            continue
        plugins.append(plugin)
    return plugins


def discover_entry_point_plugins(
    *,
    kind: PluginKind = "tool",
    group: str | None = None,
) -> list[ProviderPlugin]:
    plugins: list[ProviderPlugin] = []
    entry_point_group = group or _ENTRY_POINT_GROUPS[kind]
    for ep in entry_points(group=entry_point_group):
        plugins.append(
            ProviderPlugin(
                name=ep.name,
                provider_ref=ep.value,
                source="entry_point",
                kind=kind,
                description=f"Entry point {ep.value}",
            )
        )
    return plugins


def resolve_provider_plugins(
    *,
    cwd: Path,
    kind: PluginKind,
    bundled: list[ProviderPlugin] | None = None,
    enabled: set[str] | None = None,
    disabled: set[str] | None = None,
    include_entry_points: bool = False,
    user_plugins_dir: Path | None = None,
) -> list[ProviderPlugin]:
    disabled_names = disabled or set()
    enabled_names = enabled or set()

    workspace_dir = cwd / ".harness" / "plugins"
    user_dir = user_plugins_dir or default_user_plugins_dir()
    ordered: dict[str, ProviderPlugin] = {}

    for plugin in bundled or []:
        if plugin.kind == kind:
            ordered[plugin.name] = plugin
    for plugin in discover_manifest_plugins(user_dir, source="user", kind=kind):
        ordered[plugin.name] = plugin
    if include_entry_points:
        for plugin in discover_entry_point_plugins(kind=kind):
            ordered[plugin.name] = plugin
    for plugin in discover_manifest_plugins(workspace_dir, source="workspace", kind=kind):
        ordered[plugin.name] = plugin

    selected: list[ProviderPlugin] = []
    for plugin in ordered.values():
        if not plugin.enabled:
            continue
        if plugin.name in disabled_names:
            continue
        if enabled_names and plugin.source != "bundled" and plugin.name not in enabled_names:
            continue
        selected.append(plugin)
    return selected


def resolve_tool_provider_plugins(
    *,
    cwd: Path,
    bundled: list[ToolProviderPlugin] | None = None,
    enabled: set[str] | None = None,
    disabled: set[str] | None = None,
    include_entry_points: bool = False,
    user_plugins_dir: Path | None = None,
) -> list[ToolProviderPlugin]:
    return resolve_provider_plugins(
        cwd=cwd,
        kind="tool",
        bundled=bundled,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
        user_plugins_dir=user_plugins_dir,
    )


def resolve_experience_provider_plugins(
    *,
    cwd: Path,
    bundled: list[ExperienceProviderPlugin] | None = None,
    enabled: set[str] | None = None,
    disabled: set[str] | None = None,
    include_entry_points: bool = False,
    user_plugins_dir: Path | None = None,
) -> list[ExperienceProviderPlugin]:
    return resolve_provider_plugins(
        cwd=cwd,
        kind="experience",
        bundled=bundled,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
        user_plugins_dir=user_plugins_dir,
    )


def resolve_domain_profile_provider_plugins(
    *,
    cwd: Path,
    bundled: list[DomainProfileProviderPlugin] | None = None,
    enabled: set[str] | None = None,
    disabled: set[str] | None = None,
    include_entry_points: bool = False,
    user_plugins_dir: Path | None = None,
) -> list[DomainProfileProviderPlugin]:
    return resolve_provider_plugins(
        cwd=cwd,
        kind="domain_profile",
        bundled=bundled,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
        user_plugins_dir=user_plugins_dir,
    )


def resolve_verifier_provider_plugins(
    *,
    cwd: Path,
    bundled: list[VerifierProviderPlugin] | None = None,
    enabled: set[str] | None = None,
    disabled: set[str] | None = None,
    include_entry_points: bool = False,
    user_plugins_dir: Path | None = None,
) -> list[VerifierProviderPlugin]:
    return resolve_provider_plugins(
        cwd=cwd,
        kind="verifier",
        bundled=bundled,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
        user_plugins_dir=user_plugins_dir,
    )


def resolve_critic_provider_plugins(
    *,
    cwd: Path,
    bundled: list[CriticProviderPlugin] | None = None,
    enabled: set[str] | None = None,
    disabled: set[str] | None = None,
    include_entry_points: bool = False,
    user_plugins_dir: Path | None = None,
) -> list[CriticProviderPlugin]:
    return resolve_provider_plugins(
        cwd=cwd,
        kind="critic",
        bundled=bundled,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
        user_plugins_dir=user_plugins_dir,
    )


def _instantiate_provider(
    candidate: object,
    *,
    provider_ref: str,
    expected_type: type[object],
    label: str,
) -> object:
    instance = candidate() if isinstance(candidate, type) else candidate
    if not isinstance(instance, expected_type):
        raise TypeError(f"{provider_ref!r} did not resolve to a {label}")
    return instance


def _load_provider(
    plugin: ProviderPlugin,
    *,
    expected_type: type[object],
    label: str,
) -> object:
    importlib.invalidate_caches()
    if ":" in plugin.provider_ref:
        module_name, attr_name = plugin.provider_ref.split(":", 1)
        module = importlib.import_module(module_name)
        target = getattr(module, attr_name)
        return _instantiate_provider(
            target,
            provider_ref=plugin.provider_ref,
            expected_type=expected_type,
            label=label,
        )
    module = importlib.import_module(plugin.provider_ref)
    return _instantiate_provider(
        module,
        provider_ref=plugin.provider_ref,
        expected_type=expected_type,
        label=label,
    )


def load_tool_provider(plugin: ToolProviderPlugin) -> ToolProvider:
    return _load_provider(plugin, expected_type=ToolProvider, label="ToolProvider")  # type: ignore[return-value]


def load_experience_provider(plugin: ExperienceProviderPlugin) -> ExperienceProvider:
    return _load_provider(  # type: ignore[return-value]
        plugin,
        expected_type=ExperienceProvider,
        label="ExperienceProvider",
    )


def load_domain_profile_provider(plugin: DomainProfileProviderPlugin) -> DomainProfileProvider:
    return _load_provider(  # type: ignore[return-value]
        plugin,
        expected_type=DomainProfileProvider,
        label="DomainProfileProvider",
    )


def load_verifier_provider(plugin: VerifierProviderPlugin) -> VerifierProvider:
    return _load_provider(  # type: ignore[return-value]
        plugin,
        expected_type=VerifierProvider,
        label="VerifierProvider",
    )


def load_critic_provider(plugin: CriticProviderPlugin) -> CriticProvider:
    return _load_provider(  # type: ignore[return-value]
        plugin,
        expected_type=CriticProvider,
        label="CriticProvider",
    )


def load_provider_plugin(plugin: ProviderPlugin) -> object:
    if plugin.kind == "tool":
        return load_tool_provider(plugin)
    if plugin.kind == "experience":
        return load_experience_provider(plugin)
    if plugin.kind == "domain_profile":
        return load_domain_profile_provider(plugin)
    if plugin.kind == "verifier":
        return load_verifier_provider(plugin)
    if plugin.kind == "critic":
        return load_critic_provider(plugin)
    raise ValueError(f"unsupported plugin kind: {plugin.kind}")


def validate_provider_plugin(plugin: ProviderPlugin) -> tuple[bool, str]:
    try:
        load_provider_plugin(plugin)
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


def load_tool_providers(
    *,
    cwd: Path,
    bundled: list[ToolProviderPlugin] | None = None,
    enabled: set[str] | None = None,
    disabled: set[str] | None = None,
    include_entry_points: bool = False,
    user_plugins_dir: Path | None = None,
) -> list[ToolProvider]:
    plugins = resolve_tool_provider_plugins(
        cwd=cwd,
        bundled=bundled,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
        user_plugins_dir=user_plugins_dir,
    )
    return [load_tool_provider(plugin) for plugin in plugins]


def load_experience_providers(
    *,
    cwd: Path,
    bundled: list[ExperienceProviderPlugin] | None = None,
    enabled: set[str] | None = None,
    disabled: set[str] | None = None,
    include_entry_points: bool = False,
    user_plugins_dir: Path | None = None,
) -> list[ExperienceProvider]:
    plugins = resolve_experience_provider_plugins(
        cwd=cwd,
        bundled=bundled,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
        user_plugins_dir=user_plugins_dir,
    )
    return [load_experience_provider(plugin) for plugin in plugins]


def load_domain_profile_providers(
    *,
    cwd: Path,
    bundled: list[DomainProfileProviderPlugin] | None = None,
    enabled: set[str] | None = None,
    disabled: set[str] | None = None,
    include_entry_points: bool = False,
    user_plugins_dir: Path | None = None,
) -> list[DomainProfileProvider]:
    plugins = resolve_domain_profile_provider_plugins(
        cwd=cwd,
        bundled=bundled,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
        user_plugins_dir=user_plugins_dir,
    )
    return [load_domain_profile_provider(plugin) for plugin in plugins]


def load_verifier_providers(
    *,
    cwd: Path,
    bundled: list[VerifierProviderPlugin] | None = None,
    enabled: set[str] | None = None,
    disabled: set[str] | None = None,
    include_entry_points: bool = False,
    user_plugins_dir: Path | None = None,
) -> list[VerifierProvider]:
    plugins = resolve_verifier_provider_plugins(
        cwd=cwd,
        bundled=bundled,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
        user_plugins_dir=user_plugins_dir,
    )
    return [load_verifier_provider(plugin) for plugin in plugins]


def load_critic_providers(
    *,
    cwd: Path,
    bundled: list[CriticProviderPlugin] | None = None,
    enabled: set[str] | None = None,
    disabled: set[str] | None = None,
    include_entry_points: bool = False,
    user_plugins_dir: Path | None = None,
) -> list[CriticProvider]:
    plugins = resolve_critic_provider_plugins(
        cwd=cwd,
        bundled=bundled,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
        user_plugins_dir=user_plugins_dir,
    )
    return [load_critic_provider(plugin) for plugin in plugins]


__all__ = [
    "CriticProviderPlugin",
    "DomainProfileProviderPlugin",
    "ExperienceProviderPlugin",
    "PluginKind",
    "PluginSource",
    "ProviderPlugin",
    "ToolProviderPlugin",
    "VerifierProviderPlugin",
    "default_user_plugins_dir",
    "discover_entry_point_plugins",
    "discover_manifest_plugins",
    "load_critic_provider",
    "load_critic_providers",
    "load_domain_profile_provider",
    "load_domain_profile_providers",
    "load_experience_provider",
    "load_experience_providers",
    "load_provider_plugin",
    "load_tool_provider",
    "load_tool_providers",
    "load_verifier_provider",
    "load_verifier_providers",
    "resolve_critic_provider_plugins",
    "resolve_domain_profile_provider_plugins",
    "resolve_experience_provider_plugins",
    "resolve_provider_plugins",
    "resolve_tool_provider_plugins",
    "resolve_verifier_provider_plugins",
    "validate_provider_plugin",
]
