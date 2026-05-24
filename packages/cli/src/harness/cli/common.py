from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import typer
from rich.console import Console

from harness.adapters.anthropic import AnthropicAdapter
from harness.adapters.ollama import OllamaAdapter
from harness.adapters.openrouter import OpenRouterAdapter
from harness.cli.config import HarnessConfig, load_config
from harness.cli.plugins import load_cli_tool_providers
from harness.core import Adapter, ToolRegistry

console = Console()

KNOWN_PROVIDERS: tuple[str, ...] = ("ollama", "openrouter")

_T = TypeVar("_T")


def _run_async(awaitable: Awaitable[_T]) -> _T:
    """Run a CLI coroutine in an explicitly managed event loop."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(awaitable)
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        finally:
            try:
                loop.run_until_complete(loop.shutdown_default_executor())
            finally:
                asyncio.set_event_loop(None)
                loop.close()


def _build_adapter(provider: str, *, base_url: str | None, config: HarnessConfig) -> Adapter:
    settings = config.provider(provider)
    effective_base_url = base_url or settings.get("base_url")
    if provider == "ollama":
        timeout = float(settings.get("timeout", 120.0))
        return (
            OllamaAdapter(base_url=effective_base_url, timeout=timeout)
            if effective_base_url
            else OllamaAdapter(timeout=timeout)
        )
    if provider == "openrouter":
        return OpenRouterAdapter(
            base_url=effective_base_url,
            http_referer=settings.get("http_referer"),
            x_title=settings.get("x_title"),
        )
    if provider == "anthropic":
        return AnthropicAdapter(base_url=effective_base_url)
    raise typer.BadParameter(f"unknown provider: {provider!r}")


def _build_tools(
    cwd: Path,
    *,
    config: HarnessConfig | None = None,
    include: set[str] | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    for provider in load_cli_tool_providers(cwd, config=config):
        registry.register_provider(provider)
    built = registry.materialize_specs(cwd=cwd)
    if include is None:
        return built
    filtered = ToolRegistry()
    for name in sorted(include):
        if built.has(name):
            filtered.register(built.get(name))
    return filtered


def _resolve_chain(
    *,
    failover_flag: str | None,
    provider_flag: str | None,
    config: HarnessConfig,
) -> list[str]:
    """Resolve the provider chain from --failover > --provider > config > 'ollama'."""
    if failover_flag:
        chain = [p.strip() for p in failover_flag.split(",") if p.strip()]
        if not chain:
            raise typer.BadParameter("--failover chain is empty")
        return chain
    return [provider_flag or config.default_provider or "ollama"]


def _load_cli_config(config_path: Path | None) -> HarnessConfig:
    try:
        return load_config(config_path)
    except Exception as exc:
        console.print(f"[red]Bad config:[/red] {exc}")
        raise typer.Exit(2) from None


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def _args_preview(args: dict) -> str:
    if not args:
        return ""
    parts = [f"{k}={_truncate(repr(v), 40)}" for k, v in args.items()]
    return ", ".join(parts)


def _ago(dt: datetime) -> str:
    now = datetime.now(UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    delta = now - dt
    seconds = int(delta.total_seconds())
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    if seconds < 86400:
        return f"{seconds // 3600}h ago"
    return f"{seconds // 86400}d ago"
