from __future__ import annotations

import os
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
    if not os.environ.get("TAVILY_API_KEY"):
        return None
    try:
        from harness.core import ToolCall
        from harness.tools.web import TavilySearchTool

        searcher = TavilySearchTool()

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
    if critic in ("llm", "llm+search"):
        adapter = build_adapter(chain[0], base_url=None, config=config)
        search_fn = build_search_fn() if critic == "llm+search" else None
        return make_multi_critic(adapter=adapter, model=model, search_fn=search_fn)
    raise typer.BadParameter(f"unknown --critic value: {critic!r} (use llm|llm+search|none)")


def normalize_task_header(prompt: str) -> str:
    for line in prompt.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped.lower()
    return ""


def is_feature_task(prompt: str) -> bool:
    header = normalize_task_header(prompt)
    feature_verbs = ("add ", "implement ", "create ", "support ", "introduce ")
    bug_verbs = ("fix ", "debug ", "handle ", "resolve ", "repair ", "patch ")
    return any(header.startswith(v) for v in feature_verbs) and not any(
        header.startswith(v) for v in bug_verbs
    )


def looks_scope_sensitive(prompt: str, phases: str | None) -> bool:
    lower = prompt.lower()
    if phases:
        return True
    markers = (
        "do not touch",
        "do not fix",
        "stay focused",
        "nothing else should change",
        "only modify",
        "minimal fix",
        "just fix",
        "while i'm here",
        "scope creep",
    )
    return any(marker in lower for marker in markers)


def looks_diagnosis_heavy(prompt: str, verify_command: str | None) -> bool:
    lower = prompt.lower()
    markers = (
        "likely",
        "downstream",
        "root cause",
        "timeout",
        "concurrent",
        "deduplic",
        "flaky",
        "real bug",
        "wrong layer",
    )
    if any(marker in lower for marker in markers):
        return True
    return bool(verify_command and ("pytest" in verify_command or "test" in verify_command))


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

    feature_task = is_feature_task(prompt)
    scope_sensitive = looks_scope_sensitive(prompt, phases)
    diagnosis_heavy = looks_diagnosis_heavy(prompt, verify_command)

    structural_profile = "minimal"
    if scope_sensitive:
        structural_profile = "strict"
    elif feature_task or diagnosis_heavy:
        structural_profile = "diagnostic" if diagnosis_heavy and not feature_task else "minimal"

    critic_mode = requested_critic
    if requested_critic is None and diagnosis_heavy and not feature_task:
        critic_mode = "llm"

    reasons: list[str] = []
    if scope_sensitive:
        reasons.append("scope-sensitive prompt -> strict structural checks")
    elif feature_task:
        reasons.append("feature task -> keep structure light")
    elif diagnosis_heavy:
        reasons.append("diagnosis-heavy bugfix -> diagnostic structure + critic")
    else:
        reasons.append("default adaptive path -> minimal structure")
    if critic_mode and requested_critic is None:
        reasons.append(f"implicit critic={critic_mode}")
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
