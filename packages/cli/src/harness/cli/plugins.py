from __future__ import annotations

from pathlib import Path

from harness.cli.config import HarnessConfig
from harness.core.extensions import (
    CriticProvider,
    DomainProfileProvider,
    ExperienceProvider,
    HookProvider,
    ToolProvider,
    VerifierProvider,
)
from harness.core.plugin_loader import (
    CriticProviderPlugin,
    HookProviderPlugin,
    PluginKind,
    ProviderPlugin,
    ToolProviderPlugin,
    VerifierProviderPlugin,
    load_critic_providers,
    load_domain_profile_providers,
    load_experience_providers,
    load_hook_providers,
    load_tool_providers,
    load_verifier_providers,
    resolve_critic_provider_plugins,
    resolve_domain_profile_provider_plugins,
    resolve_experience_provider_plugins,
    resolve_hook_provider_plugins,
    resolve_provider_plugins,
    resolve_tool_provider_plugins,
    resolve_verifier_provider_plugins,
)

_BUILTIN_PLUGIN = ToolProviderPlugin(
    name="builtin",
    provider_ref="harness.cli.builtin_tools:BuiltinToolProvider",
    source="bundled",
    kind="tool",
    description="Standard Harness CLI toolset.",
)

_BUILTIN_HOOK_PLUGIN = HookProviderPlugin(
    name="builtin-hooks",
    provider_ref="harness.cli.gateway_hooks:BuiltinHookProvider",
    source="bundled",
    kind="hook",
    description="Built-in lifecycle hooks for gateway notifications.",
)


def _plugin_selection(
    config: HarnessConfig | None,
) -> tuple[set[str] | None, set[str] | None, bool]:
    cfg = config or HarnessConfig()
    enabled = set(cfg.plugins_enabled)
    disabled = set(cfg.plugins_disabled)
    return enabled or None, disabled or None, cfg.include_plugin_entry_points


def discover_cli_plugins(
    cwd: Path,
    *,
    config: HarnessConfig | None = None,
    kind: PluginKind | None = None,
) -> list[ProviderPlugin]:
    enabled, disabled, include_entry_points = _plugin_selection(config)
    if kind is None:
        plugins: list[ProviderPlugin] = []
        for plugin_kind in (
            "tool",
            "experience",
            "domain_profile",
            "verifier",
            "critic",
            "hook",
        ):
            plugins.extend(
                resolve_provider_plugins(
                    cwd=cwd,
                    kind=plugin_kind,
                    bundled=(
                        [_BUILTIN_PLUGIN]
                        if plugin_kind == "tool"
                        else ([_BUILTIN_HOOK_PLUGIN] if plugin_kind == "hook" else None)
                    ),
                    enabled=enabled,
                    disabled=disabled,
                    include_entry_points=include_entry_points,
                )
            )
        return plugins
    return resolve_provider_plugins(
        cwd=cwd,
        kind=kind,
        bundled=(
            [_BUILTIN_PLUGIN]
            if kind == "tool"
            else ([_BUILTIN_HOOK_PLUGIN] if kind == "hook" else None)
        ),
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
    )


def discover_cli_tool_plugins(
    cwd: Path,
    *,
    config: HarnessConfig | None = None,
) -> list[ToolProviderPlugin]:
    enabled, disabled, include_entry_points = _plugin_selection(config)
    return resolve_tool_provider_plugins(
        cwd=cwd,
        bundled=[_BUILTIN_PLUGIN],
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
    )


def load_cli_tool_providers(
    cwd: Path,
    *,
    config: HarnessConfig | None = None,
) -> list[ToolProvider]:
    enabled, disabled, include_entry_points = _plugin_selection(config)
    return load_tool_providers(
        cwd=cwd,
        bundled=[_BUILTIN_PLUGIN],
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
    )


def discover_cli_experience_plugins(
    cwd: Path,
    *,
    config: HarnessConfig | None = None,
) -> list[ProviderPlugin]:
    enabled, disabled, include_entry_points = _plugin_selection(config)
    return resolve_experience_provider_plugins(
        cwd=cwd,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
    )


def load_cli_experience_providers(
    cwd: Path,
    *,
    config: HarnessConfig | None = None,
) -> list[ExperienceProvider]:
    enabled, disabled, include_entry_points = _plugin_selection(config)
    return load_experience_providers(
        cwd=cwd,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
    )


def discover_cli_domain_profile_plugins(
    cwd: Path,
    *,
    config: HarnessConfig | None = None,
) -> list[ProviderPlugin]:
    enabled, disabled, include_entry_points = _plugin_selection(config)
    return resolve_domain_profile_provider_plugins(
        cwd=cwd,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
    )


def load_cli_domain_profile_providers(
    cwd: Path,
    *,
    config: HarnessConfig | None = None,
) -> list[DomainProfileProvider]:
    enabled, disabled, include_entry_points = _plugin_selection(config)
    return load_domain_profile_providers(
        cwd=cwd,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
    )


def discover_cli_verifier_plugins(
    cwd: Path,
    *,
    config: HarnessConfig | None = None,
) -> list[VerifierProviderPlugin]:
    enabled, disabled, include_entry_points = _plugin_selection(config)
    return resolve_verifier_provider_plugins(
        cwd=cwd,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
    )


def load_cli_verifier_providers(
    cwd: Path,
    *,
    config: HarnessConfig | None = None,
) -> list[VerifierProvider]:
    enabled, disabled, include_entry_points = _plugin_selection(config)
    return load_verifier_providers(
        cwd=cwd,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
    )


def discover_cli_critic_plugins(
    cwd: Path,
    *,
    config: HarnessConfig | None = None,
) -> list[CriticProviderPlugin]:
    enabled, disabled, include_entry_points = _plugin_selection(config)
    return resolve_critic_provider_plugins(
        cwd=cwd,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
    )


def load_cli_critic_providers(
    cwd: Path,
    *,
    config: HarnessConfig | None = None,
) -> list[CriticProvider]:
    enabled, disabled, include_entry_points = _plugin_selection(config)
    return load_critic_providers(
        cwd=cwd,
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
    )


def discover_cli_hook_plugins(
    cwd: Path,
    *,
    config: HarnessConfig | None = None,
) -> list[HookProviderPlugin]:
    enabled, disabled, include_entry_points = _plugin_selection(config)
    return resolve_hook_provider_plugins(
        cwd=cwd,
        bundled=[_BUILTIN_HOOK_PLUGIN],
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
    )


def load_cli_hook_providers(
    cwd: Path,
    *,
    config: HarnessConfig | None = None,
) -> list[HookProvider]:
    enabled, disabled, include_entry_points = _plugin_selection(config)
    return load_hook_providers(
        cwd=cwd,
        bundled=[_BUILTIN_HOOK_PLUGIN],
        enabled=enabled,
        disabled=disabled,
        include_entry_points=include_entry_points,
    )


__all__ = [
    "discover_cli_critic_plugins",
    "discover_cli_domain_profile_plugins",
    "discover_cli_experience_plugins",
    "discover_cli_hook_plugins",
    "discover_cli_plugins",
    "discover_cli_tool_plugins",
    "discover_cli_verifier_plugins",
    "load_cli_critic_providers",
    "load_cli_domain_profile_providers",
    "load_cli_experience_providers",
    "load_cli_hook_providers",
    "load_cli_tool_providers",
    "load_cli_verifier_providers",
]
