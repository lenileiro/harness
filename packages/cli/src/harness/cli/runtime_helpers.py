from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from harness.core import (
    ChainedVerifier,
    ClaimGroundingVerifier,
    Critic,
    LLMJudgeVerifier,
    RuleVerifier,
    ShellVerifier,
    StateVerifier,
    Storage,
    Verifier,
    VerifierRouter,
    build_ledger,
    format_ledger,
    make_multi_critic,
)
from harness.storage.memory import InMemoryStorage
from harness.storage.sqlite import SQLiteStorage, default_db_path

if TYPE_CHECKING:
    from rich.console import Console

    from harness.cli.config import HarnessConfig
    from harness.core import Adapter, ToolResult


@dataclass(frozen=True)
class RuntimeStrategy:
    structural_profile: str
    critic_mode: str | None
    rationale: str


def workspace_db(cwd: Path) -> Path | None:
    candidate = cwd / ".harness" / "harness.db"
    return candidate if candidate.exists() else None


def build_storage(*, db: Path | None, in_memory: bool, cwd: Path | None = None) -> Storage:
    if in_memory:
        return InMemoryStorage()
    resolved = db or (cwd and workspace_db(cwd)) or default_db_path()
    return SQLiteStorage(path=resolved)


def build_verifier(
    verify: str | None,
    *,
    chain: list[str],
    model: str,
    config: HarnessConfig,
    build_adapter: Callable[..., Adapter],
    cwd: Path | None = None,
    verify_command: str | None = None,
) -> Verifier | None:
    if not verify or verify == "none":
        return None
    if verify.startswith("plugin:"):
        from harness.cli.plugins import discover_cli_verifier_plugins, load_cli_verifier_providers

        plugin_name = verify.split(":", 1)[1].strip()
        providers = load_cli_verifier_providers(cwd or Path.cwd(), config=config)
        plugins = discover_cli_verifier_plugins(cwd or Path.cwd(), config=config)
        provider_map = {
            plugin.name: provider for plugin, provider in zip(plugins, providers, strict=False)
        }
        provider = provider_map.get(plugin_name)
        if provider is None:
            raise typer.BadParameter(f"unknown verifier plugin: {plugin_name!r}")
        verifiers = provider.verifiers()
        if len(verifiers) != 1:
            raise typer.BadParameter(
                f"verifier plugin {plugin_name!r} must provide exactly one verifier"
            )
        return verifiers[0]
    if verify == "grounding":
        return ClaimGroundingVerifier()
    if verify == "state":
        return StateVerifier(cwd=cwd or Path.cwd())
    if verify == "rule":
        return RuleVerifier()
    if verify == "shell":
        if not verify_command:
            raise typer.BadParameter("--verify shell requires --verify-command <cmd>")
        return ShellVerifier(verify_command, cwd=cwd)
    if verify == "llm":
        adapter = build_adapter(chain[0], base_url=None, config=config)
        return LLMJudgeVerifier(adapter=adapter, model=model)
    if verify == "auto":
        adapter = build_adapter(chain[0], base_url=None, config=config)
        return ChainedVerifier(
            ClaimGroundingVerifier(),
            StateVerifier(cwd=cwd or Path.cwd()),
            VerifierRouter(
                rule=RuleVerifier(),
                llm=LLMJudgeVerifier(adapter=adapter, model=model),
            ),
        )
    raise typer.BadParameter(
        f"unknown --verify value: {verify!r} (use grounding|state|rule|shell|llm|auto|none)"
    )


def build_search_fn() -> Any:
    try:
        from harness.core import ToolCall
        from harness.tools.web import WebSearchTool

        searcher = WebSearchTool()

        async def _search(query: str) -> str:
            call = ToolCall(id=f"s_{query[:8]}", name="web_search", arguments={"query": query})
            result: ToolResult = await searcher(call)
            return result.content or ""

        return _search
    except Exception:
        return None


def build_critic(
    critic: str | None,
    *,
    chain: list[str],
    model: str,
    config: HarnessConfig,
    build_adapter: Callable[..., Adapter],
) -> Critic | None:
    if not critic or critic == "none":
        return None
    if critic.startswith("plugin:"):
        from harness.cli.plugins import discover_cli_critic_plugins, load_cli_critic_providers

        plugin_name = critic.split(":", 1)[1].strip()
        providers = load_cli_critic_providers(Path.cwd(), config=config)
        plugins = discover_cli_critic_plugins(Path.cwd(), config=config)
        provider_map = {
            plugin.name: provider for plugin, provider in zip(plugins, providers, strict=False)
        }
        provider = provider_map.get(plugin_name)
        if provider is None:
            raise typer.BadParameter(f"unknown critic plugin: {plugin_name!r}")
        critics = provider.critics()
        if len(critics) != 1:
            raise typer.BadParameter(
                f"critic plugin {plugin_name!r} must provide exactly one critic"
            )
        return critics[0]
    if critic in ("llm", "llm+search"):
        adapter = build_adapter(chain[0], base_url=None, config=config)
        search_fn = build_search_fn() if critic == "llm+search" else None
        return make_multi_critic(adapter=adapter, model=model, search_fn=search_fn)
    raise typer.BadParameter(
        f"unknown --critic value: {critic!r} (use llm|llm+search|plugin:<name>|none)"
    )


def resolve_runtime_strategy(
    *,
    prompt: str,
    requested_profile: str,
    verify_command: str | None,
    phases: str | None,
    requested_critic: str | None,
) -> RuntimeStrategy:
    if requested_profile != "adaptive":
        return RuntimeStrategy(
            structural_profile=requested_profile,
            critic_mode=requested_critic,
            rationale=f"explicit profile={requested_profile}",
        )

    structural_profile = "minimal"
    if phases:
        structural_profile = "strict"
    elif verify_command:
        structural_profile = "diagnostic"

    critic_mode = requested_critic

    reasons: list[str] = []
    if phases:
        reasons.append("explicit phases -> strict structural checks")
    elif verify_command:
        reasons.append("explicit verifier command -> diagnostic structural checks")
    else:
        reasons.append("no explicit verifier/phases -> minimal structural checks")
    if critic_mode:
        reasons.append(f"critic={critic_mode}")
    return RuntimeStrategy(
        structural_profile=structural_profile,
        critic_mode=critic_mode,
        rationale="; ".join(reasons),
    )


async def print_defense_ledger(
    storage: Storage, session_id: str | None, *, console: Console
) -> None:
    try:
        target_session_id = session_id
        if target_session_id is None:
            sessions = await storage.list(limit=1)  # type: ignore[attr-defined]
            if sessions:
                target_session_id = sessions[0].id
        activity_store: Any = storage
        if target_session_id is not None:
            events = await activity_store.list_activity(session_id=target_session_id, limit=500)
        else:
            events = await activity_store.list_activity(limit=500)
        ledger = build_ledger(events)
        if ledger.is_empty():
            return
        console.print(f"\n[dim]{format_ledger(ledger)}[/dim]")
    except Exception as exc:
        console.print(f"[dim]defense ledger unavailable: {exc!s}[/dim]")
