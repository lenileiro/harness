from __future__ import annotations

from pathlib import Path

import pytest

from harness.core.extensions import ToolProvider
from harness.core.plugin_loader import (
    CriticProviderPlugin,
    ToolProviderPlugin,
    VerifierProviderPlugin,
    discover_entry_point_plugins,
    discover_manifest_plugins,
    load_critic_provider,
    load_domain_profile_provider,
    load_experience_provider,
    load_provider_plugin,
    load_tool_provider,
    load_verifier_provider,
    resolve_critic_provider_plugins,
    resolve_domain_profile_provider_plugins,
    resolve_experience_provider_plugins,
    resolve_tool_provider_plugins,
    resolve_verifier_provider_plugins,
    validate_provider_plugin,
)


def _write_provider_module(root: Path, module_name: str) -> None:
    (root / f"{module_name}.py").write_text(
        """
from harness.core.tool_entry import ToolSpec


class DummyTool:
    name = "dummy_tool"
    description = "Dummy tool from plugin"
    parameters_schema = {"type": "object", "properties": {}}
    approval = "auto"

    async def __call__(self, call):
        raise NotImplementedError


class DemoProvider:
    def specs(self):
        return [ToolSpec(name="dummy_tool", factory=lambda ctx: DummyTool())]
""",
        encoding="utf-8",
    )


def test_discover_manifest_plugins_reads_flat_and_nested_layouts(tmp_path: Path) -> None:
    root = tmp_path / "plugins"
    root.mkdir()
    (root / "flat.toml").write_text(
        'name = "flat"\nprovider = "flat_provider:DemoProvider"\n',
        encoding="utf-8",
    )
    nested = root / "nested"
    nested.mkdir()
    (nested / "plugin.toml").write_text(
        'name = "nested"\nprovider = "nested_provider:DemoProvider"\ndescription = "Nested"\n',
        encoding="utf-8",
    )

    plugins = discover_manifest_plugins(root, source="workspace")
    assert [plugin.name for plugin in plugins] == ["flat", "nested"]
    assert plugins[1].description == "Nested"


def test_discover_manifest_plugins_filters_by_kind(tmp_path: Path) -> None:
    root = tmp_path / "plugins"
    root.mkdir()
    (root / "experience.toml").write_text(
        'name = "exp"\nkind = "experience"\nprovider = "exp_provider:DemoProvider"\n',
        encoding="utf-8",
    )
    (root / "tool.toml").write_text(
        'name = "tool"\nprovider = "tool_provider:DemoProvider"\n',
        encoding="utf-8",
    )

    plugins = discover_manifest_plugins(root, source="workspace", kind="experience")
    assert [plugin.name for plugin in plugins] == ["exp"]
    assert plugins[0].kind == "experience"


def test_resolve_tool_provider_plugins_applies_precedence(tmp_path: Path) -> None:
    workspace_plugins = tmp_path / ".harness" / "plugins"
    workspace_plugins.mkdir(parents=True)
    (workspace_plugins / "demo.toml").write_text(
        'name = "demo"\nprovider = "workspace_provider:DemoProvider"\n',
        encoding="utf-8",
    )

    user_plugins = tmp_path / "user-plugins"
    user_plugins.mkdir()
    (user_plugins / "demo.toml").write_text(
        'name = "demo"\nprovider = "user_provider:DemoProvider"\n',
        encoding="utf-8",
    )

    bundled = [
        ToolProviderPlugin(
            name="builtin",
            provider_ref="builtin_provider:DemoProvider",
            source="bundled",
        )
    ]
    resolved = resolve_tool_provider_plugins(
        cwd=tmp_path,
        bundled=bundled,
        user_plugins_dir=user_plugins,
    )

    assert [plugin.name for plugin in resolved] == ["builtin", "demo"]
    assert resolved[1].provider_ref == "workspace_provider:DemoProvider"


def test_resolve_non_tool_plugins_respects_manifest_kind(tmp_path: Path) -> None:
    workspace_plugins = tmp_path / ".harness" / "plugins"
    workspace_plugins.mkdir(parents=True)
    (workspace_plugins / "domain.toml").write_text(
        'name = "review-pack"\nkind = "domain_profile"\nprovider = "profiles:Provider"\n',
        encoding="utf-8",
    )
    (workspace_plugins / "experience.toml").write_text(
        'name = "repair-pack"\nkind = "experience"\nprovider = "tips:Provider"\n',
        encoding="utf-8",
    )
    (workspace_plugins / "verifier.toml").write_text(
        'name = "verify-pack"\nkind = "verifier"\nprovider = "verifier_mod:Provider"\n',
        encoding="utf-8",
    )
    (workspace_plugins / "critic.toml").write_text(
        'name = "critic-pack"\nkind = "critic"\nprovider = "critic_mod:Provider"\n',
        encoding="utf-8",
    )

    experience = resolve_experience_provider_plugins(cwd=tmp_path)
    domains = resolve_domain_profile_provider_plugins(cwd=tmp_path)
    verifiers = resolve_verifier_provider_plugins(cwd=tmp_path)
    critics = resolve_critic_provider_plugins(cwd=tmp_path)

    assert [plugin.name for plugin in experience] == ["repair-pack"]
    assert [plugin.name for plugin in domains] == ["review-pack"]
    assert [plugin.name for plugin in verifiers] == ["verify-pack"]
    assert [plugin.name for plugin in critics] == ["critic-pack"]


def test_load_tool_provider_imports_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_provider_module(tmp_path, "plugin_loader_demo_provider")
    monkeypatch.syspath_prepend(str(tmp_path))

    plugin = ToolProviderPlugin(
        name="demo",
        provider_ref="plugin_loader_demo_provider:DemoProvider",
        source="workspace",
    )
    provider = load_tool_provider(plugin)
    specs = provider.specs()
    assert [spec.name for spec in specs] == ["dummy_tool"]


def test_load_experience_and_domain_profile_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "experience_provider.py").write_text(
        """
from harness.core.tips_models import Tip


class DemoProvider:
    def query(self, task_text: str, *, top_k: int = 3):
        return [Tip(text="use the smaller reproduction", triggers=("pytest",), weight=2.0)]
""",
        encoding="utf-8",
    )
    (tmp_path / "domain_provider.py").write_text(
        """
from harness.core.domain_profiles import DomainProfile


class DemoProvider:
    def profiles(self):
        return [DomainProfile(name="docs-review", description="docs only")]
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    exp_plugin = ToolProviderPlugin(
        name="exp",
        provider_ref="experience_provider:DemoProvider",
        source="workspace",
        kind="experience",
    )
    domain_plugin = ToolProviderPlugin(
        name="domain",
        provider_ref="domain_provider:DemoProvider",
        source="workspace",
        kind="domain_profile",
    )

    experience_provider = load_experience_provider(exp_plugin)
    domain_provider = load_domain_profile_provider(domain_plugin)
    assert [tip.text for tip in experience_provider.query("pytest", top_k=5)] == [
        "use the smaller reproduction"
    ]
    assert [profile.name for profile in domain_provider.profiles()] == ["docs-review"]


def test_load_verifier_and_critic_providers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "verifier_provider.py").write_text(
        """
from harness.core.schemas import VerificationResult


class DemoVerifier:
    async def verify(self, *, session, activity):
        return VerificationResult(can_finish=True, confidence=1.0, reason="ok")


class DemoProvider:
    def verifiers(self):
        return [DemoVerifier()]
""",
        encoding="utf-8",
    )
    (tmp_path / "critic_provider.py").write_text(
        """
class DemoCritic:
    async def critique(self, *, session, verification_result, activity):
        return "challenge the assumption"


class DemoProvider:
    def critics(self):
        return [DemoCritic()]
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    verifier_plugin = VerifierProviderPlugin(
        name="verify",
        provider_ref="verifier_provider:DemoProvider",
        source="workspace",
        kind="verifier",
    )
    critic_plugin = CriticProviderPlugin(
        name="critic",
        provider_ref="critic_provider:DemoProvider",
        source="workspace",
        kind="critic",
    )

    verifier_provider = load_verifier_provider(verifier_plugin)
    critic_provider = load_critic_provider(critic_plugin)
    assert len(verifier_provider.verifiers()) == 1
    assert len(critic_provider.critics()) == 1


def test_discover_entry_point_plugins_can_be_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeEntryPoint:
        name = "ep-demo"
        value = "entry_provider:DemoProvider"

    monkeypatch.setattr(
        "harness.core.plugin_loader.entry_points",
        lambda *, group: [FakeEntryPoint()] if group == "harness.tool_providers" else [],
    )

    plugins = discover_entry_point_plugins(kind="tool")
    assert len(plugins) == 1
    assert plugins[0].name == "ep-demo"
    assert plugins[0].source == "entry_point"


def test_load_provider_plugin_dispatches_by_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_provider_module(tmp_path, "dispatch_demo_provider")
    monkeypatch.syspath_prepend(str(tmp_path))

    plugin = ToolProviderPlugin(
        name="demo",
        provider_ref="dispatch_demo_provider:DemoProvider",
        source="workspace",
        kind="tool",
    )
    provider = load_provider_plugin(plugin)
    assert isinstance(provider, ToolProvider)
    assert [spec.name for spec in provider.specs()] == ["dummy_tool"]


def test_validate_provider_plugin_reports_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.syspath_prepend(str(tmp_path))
    plugin = ToolProviderPlugin(
        name="broken",
        provider_ref="missing_plugin_module:DemoProvider",
        source="workspace",
        kind="tool",
    )
    ok, detail = validate_provider_plugin(plugin)
    assert ok is False
    assert "missing_plugin_module" in detail
