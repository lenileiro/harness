from __future__ import annotations

import asyncio
import hashlib
import os
import re
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime
from inspect import isawaitable
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console

from harness.cli.common import (
    _build_adapter,
    _build_tools,
    _load_cli_config,
    _resolve_chain,
)
from harness.cli.plugins import load_cli_hook_providers
from harness.cli.runtime_helpers import build_storage
from harness.core import (
    ApprovalStore,
    GatewayMessage,
    Message,
    default_gateway_root,
)
from harness.core.extensions import LifecycleHook
from harness.core.gateway_evidence import successful_tool_evidence_reply
from harness.core.gateway_router import dispatch_gateway_message, is_gateway_control_message
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.gateway_whatsapp import (
    default_whatsapp_config_path,
    load_whatsapp_bridge_config,
)
from harness.core.scheduler_store import SchedulerStore
from harness.core.verification_structural import (
    tool_event_changes_state,
    verify_work_event_changes_state,
)

console = Console()
_GATEWAY_TURN_TIMEOUT_SECONDS = 590.0
_GATEWAY_MAX_REPAIR_ATTEMPTS = 3
_GATEWAY_WRITE_TOOL_NAMES = frozenset({"write_file", "edit_file", "apply_diff", "patch", "shell"})
_THREAD_CONTEXT_USER_LIMIT = 240
_THREAD_CONTEXT_ASSISTANT_LIMIT = 520
_RUNTIME_VERIFICATION_HANDOFF_TEXT = (
    "[harness:runtime] The model timed out after changing the workspace. "
    "Handing the current state to verification."
)
_RUNTIME_SESSION_METADATA_RE = re.compile(
    r"^harness_session_id_[a-z0-9-]+-[0-9a-f]{8}" r"(?:_(?:chat|workflow|live)-[0-9a-f]{8})?$"
)


def _default_gateway_model(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "openrouter":
        return "openai/gpt-5.4-nano"
    return "gemma4:latest"


def _default_gateway_provider() -> str:
    if os.environ.get("OPENROUTER_API_KEY", "").strip():
        return "openrouter"
    if os.environ.get("OPENAI_API_KEY", "").strip():
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY", "").strip():
        return "anthropic"
    return "ollama"


def _gateway_env_override(name: str) -> str:
    return os.environ.get(name, "").strip()


def _gateway_turn_timeout_seconds() -> float:
    raw_override = os.environ.get("HARNESS_GATEWAY_WORKFLOW_TIMEOUT", "").strip()
    if raw_override:
        with suppress(ValueError):
            return max(1.0, float(raw_override))
    raw_child_timeout = os.environ.get("HARNESS_WHATSAPP_CHILD_TIMEOUT_MS", "").strip()
    if raw_child_timeout:
        with suppress(ValueError):
            child_timeout_seconds = float(raw_child_timeout) / 1000.0
            return max(1.0, child_timeout_seconds - 10.0)
    return _GATEWAY_TURN_TIMEOUT_SECONDS


def _gateway_max_repair_attempts() -> int:
    raw_override = os.environ.get("HARNESS_GATEWAY_MAX_REPAIR", "").strip()
    if raw_override:
        with suppress(ValueError):
            return max(0, int(raw_override))
    return _GATEWAY_MAX_REPAIR_ATTEMPTS


def _runtime_session_key(*, provider: str, model: str) -> str:
    provider_slug = re.sub(r"[^a-z0-9]+", "-", provider.lower()).strip("-") or "provider"
    digest = hashlib.sha1(f"{provider}\0{model}".encode()).hexdigest()[:8]
    return f"{provider_slug}-{digest}"


def _load_hooks(cwd: Path) -> tuple[LifecycleHook, ...]:
    hooks: list[LifecycleHook] = []
    for provider in load_cli_hook_providers(cwd):
        hooks.extend(provider.hooks())
    return tuple(hooks)


def _utcnow_text() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _compact_context_text(text: str, *, limit: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "..."


def _compact_thread_context_line(line: str) -> str:
    stripped = str(line or "").strip()
    lowered = stripped.lower()
    if lowered.startswith("user:"):
        return "user: " + _compact_context_text(
            stripped.split(":", 1)[1],
            limit=_THREAD_CONTEXT_USER_LIMIT,
        )
    if lowered.startswith("assistant:"):
        return "assistant: " + _compact_context_text(
            stripped.split(":", 1)[1],
            limit=_THREAD_CONTEXT_ASSISTANT_LIMIT,
        )
    return _compact_context_text(stripped, limit=_THREAD_CONTEXT_ASSISTANT_LIMIT)


def _updated_thread_context(
    thread_context: list[str],
    *,
    user_message: str,
    assistant_reply: str,
) -> list[str]:
    compacted = [_compact_thread_context_line(line) for line in thread_context if line.strip()]
    compacted.extend(
        [
            "user: " + _compact_context_text(user_message, limit=_THREAD_CONTEXT_USER_LIMIT),
            "assistant: "
            + _compact_context_text(assistant_reply, limit=_THREAD_CONTEXT_ASSISTANT_LIMIT),
        ]
    )
    return compacted[-8:]


def _prune_gateway_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metadata.items()
        if not _RUNTIME_SESSION_METADATA_RE.match(str(key))
    }


def _thread_context_from_session(session: Any) -> list[str]:
    raw_thread_context = session.metadata.get("thread_context", [])
    if not isinstance(raw_thread_context, list):
        return []
    return [
        _compact_thread_context_line(str(item)) for item in raw_thread_context if str(item).strip()
    ]


def _with_shared_gateway_context(
    *,
    session_store: GatewaySessionStore,
    session: Any,
    message: str,
) -> str:
    profile = session_store.get_or_create_profile(
        transport=session.transport,
        user_id=session.user_id,
    )
    sections: list[str] = []
    active_lines: list[str] = []
    for work in profile.active_work[-8:]:
        title = _compact_context_text(work.title, limit=120)
        summary = _compact_context_text(work.summary, limit=220)
        label = f"{title} [{work.kind}]" if work.kind else title
        active_lines.append(f"- {label}: {summary}" if summary else f"- {label}")
    if active_lines:
        sections.append("Shared active work for this user:\n" + "\n".join(active_lines))

    sessions_by_thread = {
        item.thread_id: item
        for item in session_store.list_user_sessions(
            transport=session.transport,
            user_id=session.user_id,
        )
    }
    other_lines: list[str] = []
    for thread_id in profile.recent_threads[-8:]:
        if thread_id == session.thread_id:
            continue
        other = sessions_by_thread.get(thread_id)
        if other is None:
            continue
        summary = _compact_context_text(
            _shared_context_summary(str(other.metadata.get("thread_summary") or "")),
            limit=260,
        )
        if summary:
            other_lines.append(f"- {thread_id}: {summary}")
    if other_lines:
        sections.append("Other recent chats for this user:\n" + "\n".join(other_lines))

    if not sections:
        return message
    return "\n\n".join(
        [
            (
                "Continuity context only. It is not verified evidence; re-check factual "
                "or current claims with tools before relying on it."
            ),
            *sections,
            f"User message:\n{message}",
        ]
    )


def _shared_context_summary(summary: str) -> str:
    """Keep cross-thread continuity without recycling prior assistant answers as evidence."""

    text = " ".join(str(summary or "").split())
    marker = " Last reply:"
    if marker in text:
        text = text.split(marker, 1)[0].rstrip()
    return text


_TOOL_FAILURE_POLL_SECONDS = 0.5


def _tool_failure_reply_from_messages(messages: list[Message]) -> str | None:
    for message in reversed(messages):
        if message.role != "tool":
            continue
        content = (message.content or "").strip()
        if not content:
            continue
        if message.name == "verify_work" and "failed" not in content.lower():
            return None
        if "failed" not in content.lower() and "timed out" not in content.lower():
            continue
        tool_name = str(message.name or "tool").strip() or "tool"
        return f"I tried to use `{tool_name}`, but it failed: {' '.join(content.split())}"
    return None


def _run_failure_reply_from_activity(events: list[Any]) -> str | None:
    for event in reversed(events):
        if getattr(event, "kind", "") != "agent_run.failed":
            continue
        data = getattr(event, "data", {}) or {}
        if not isinstance(data, dict):
            continue
        kind = str(data.get("kind") or "").strip()
        error = " ".join(str(data.get("error") or "").split())
        if kind == "rate_limit":
            return (
                "Harness hit a provider rate limit while generating this reply. "
                "The core runtime retried the request; please try again shortly."
            )
        if kind == "model_unavailable":
            return (
                "Harness could not use the selected model right now. "
                "The core runtime retried the request, but no provider completed it."
            )
        if kind == "timeout":
            return (
                "Harness timed out while waiting for the model after retrying through "
                "the core runtime. Please try again shortly."
            )
        if kind == "configuration" and error:
            return f"Harness provider configuration failed: {error}"
        if error:
            return f"Harness model run failed after core runtime retries: {error}"
    return None


def _verification_failure_reply_from_activity(events: list[Any]) -> str | None:
    for event in reversed(events):
        if getattr(event, "kind", "") != "verification.completed":
            continue
        data = getattr(event, "data", {}) or {}
        if not isinstance(data, dict):
            continue
        if bool(data.get("can_finish", True)):
            return None
        reason = " ".join(str(data.get("reason") or "").split())
        if reason:
            return f"Harness core verification blocked the final reply: {reason}"
        return "Harness core verification blocked the final reply."
    return None


def _successful_verified_reply_from_activity(events: list[Any]) -> str | None:
    latest_verification_index = -1
    latest_reason = ""
    for index, event in enumerate(events):
        if getattr(event, "kind", "") != "verification.completed":
            continue
        data = getattr(event, "data", {}) or {}
        if not isinstance(data, dict):
            continue
        if bool(data.get("can_finish", True)):
            latest_verification_index = index
            latest_reason = " ".join(str(data.get("reason") or "").split())
        else:
            latest_verification_index = -1
            latest_reason = ""
    if latest_verification_index < 0:
        return None

    evidence_events = [
        event
        for event in events[: latest_verification_index + 1]
        if getattr(event, "kind", "") == "tool_call.completed"
    ]
    evidence_reply = successful_tool_evidence_reply(evidence_events)
    if evidence_reply:
        return evidence_reply
    if latest_reason:
        return f"Harness verified the completed work: {latest_reason}"
    return "Harness verified the completed work."


def _event_path_label(event: Any) -> str:
    data = getattr(event, "data", {}) or {}
    if not isinstance(data, dict):
        return ""
    for source in (data.get("arguments"), data.get("metadata")):
        if not isinstance(source, dict):
            continue
        path = str(source.get("path") or source.get("file") or "").strip()
        if path:
            return path
    return ""


def _gateway_event_changes_workspace(event: Any) -> bool:
    if getattr(event, "kind", "") != "tool_call.completed":
        return False
    try:
        return bool(
            tool_event_changes_state(event, _GATEWAY_WRITE_TOOL_NAMES)
            or verify_work_event_changes_state(event)
        )
    except Exception:
        return False


def _unverified_workspace_change_reply_from_activity(events: list[Any]) -> str | None:
    latest_verification_index = -1
    for index, event in enumerate(events):
        if getattr(event, "kind", "") == "verification.completed":
            latest_verification_index = index
    for event in reversed(events[latest_verification_index + 1 :]):
        if not _gateway_event_changes_workspace(event):
            continue
        data = getattr(event, "data", {}) or {}
        tool_name = str(data.get("name") or "tool").strip() if isinstance(data, dict) else "tool"
        path = _event_path_label(event)
        detail = f"`{tool_name}`"
        if path:
            detail += f" on `{path}`"
        return (
            "Harness stopped before completing verification after the final workspace "
            f"change. Latest unverified change: {detail}. Continue the task so Harness "
            "can run `verify_work` after that change before producing a final reply."
        )
    return None


async def _latest_tool_failure_reply(*, cwd: Path, session_id: str) -> str | None:
    storage = build_storage(db=None, in_memory=False, cwd=cwd)
    try:
        session = await storage.get(session_id)
    finally:
        close = getattr(storage, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                await result
    if session is None:
        return None
    return _tool_failure_reply_from_messages(list(session.messages))


async def _latest_run_failure_reply(*, cwd: Path, session_id: str) -> str | None:
    storage = build_storage(db=None, in_memory=False, cwd=cwd)
    try:
        events = await storage.list_activity(  # type: ignore[attr-defined]
            session_id=session_id,
            kinds=("agent_run.failed",),
            limit=20,
        )
    finally:
        close = getattr(storage, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                await result
    return _run_failure_reply_from_activity(events)


async def _latest_verification_failure_reply(*, cwd: Path, session_id: str) -> str | None:
    storage = build_storage(db=None, in_memory=False, cwd=cwd)
    try:
        events = await storage.list_activity(  # type: ignore[attr-defined]
            session_id=session_id,
            kinds=("verification.completed",),
            limit=20,
        )
    finally:
        close = getattr(storage, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                await result
    return _verification_failure_reply_from_activity(events)


async def _latest_successful_verified_reply(*, cwd: Path, session_id: str) -> str | None:
    storage = build_storage(db=None, in_memory=False, cwd=cwd)
    try:
        events = await storage.list_activity(  # type: ignore[attr-defined]
            session_id=session_id,
            kinds=("tool_call.completed", "verification.completed"),
            limit=500,
        )
    finally:
        close = getattr(storage, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                await result
    return _successful_verified_reply_from_activity(events)


async def _latest_unverified_workspace_change_reply(*, cwd: Path, session_id: str) -> str | None:
    storage = build_storage(db=None, in_memory=False, cwd=cwd)
    try:
        events = await storage.list_activity(  # type: ignore[attr-defined]
            session_id=session_id,
            kinds=("tool_call.completed", "verification.completed"),
            limit=500,
        )
    finally:
        close = getattr(storage, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                await result
    return _unverified_workspace_change_reply_from_activity(events)


async def _latest_successful_tool_evidence_reply(*, cwd: Path, session_id: str) -> str | None:
    storage = build_storage(db=None, in_memory=False, cwd=cwd)
    try:
        events = await storage.list_activity(  # type: ignore[attr-defined]
            session_id=session_id,
            kinds=("tool_call.completed",),
            limit=100,
        )
    finally:
        close = getattr(storage, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                await result
    return successful_tool_evidence_reply(events)


async def _poll_tool_failure_reply(*, cwd: Path, session_id: str) -> str:
    while True:
        reply = await _latest_tool_failure_reply(cwd=cwd, session_id=session_id)
        if reply:
            return reply
        await asyncio.sleep(_TOOL_FAILURE_POLL_SECONDS)


async def _await_gateway_run(
    run: Any,
    *,
    cwd: Path,
    session_id: str,
    timeout_seconds: float,
    watch_tool_failures: bool,
) -> str | None:
    run_task = asyncio.create_task(run)
    tasks: set[asyncio.Task[Any]] = {run_task}
    failure_task: asyncio.Task[str] | None = None
    if watch_tool_failures:
        failure_task = asyncio.create_task(_poll_tool_failure_reply(cwd=cwd, session_id=session_id))
        tasks.add(failure_task)

    try:
        done, pending = await asyncio.wait(
            tasks,
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            raise TimeoutError
        if failure_task is not None and failure_task in done:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            return failure_task.result()
        if run_task in done:
            if failure_task is not None:
                failure_task.cancel()
                await asyncio.gather(failure_task, return_exceptions=True)
            return run_task.result()
        raise TimeoutError
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _noop_task_attachment(*_args: object, **_kwargs: object) -> tuple[None, None]:
    return None, None


async def _noop_print_defense_ledger(*_args: object, **_kwargs: object) -> None:
    return None


async def _run_gateway_chat_turn(
    *,
    cwd: Path,
    prompt: str,
    chain: list[str],
    model: str,
    session_id: str,
    max_steps: int,
    config: Any,
    system_prompt: str,
) -> str:
    from harness.cli.__main__ import _build_agent
    from harness.cli.run_commands import run_once as _run_once_impl
    from harness.cli.runtime_helpers import build_critic as _build_critic
    from harness.cli.runtime_helpers import build_verifier as _build_verifier
    from harness.cli.runtime_helpers import print_defense_ledger as _print_defense_ledger
    from harness.cli.runtime_helpers import (
        resolve_runtime_strategy as _resolve_runtime_strategy,
    )

    return (
        await _run_once_impl(
            prompt=prompt,
            model=model,
            chain=chain,
            base_url=None,
            cwd=cwd,
            max_steps=max(6, max_steps),
            max_output_tokens=None,
            session_id=session_id,
            task_ref=None,
            db=cwd / ".harness" / "harness.db",
            in_memory=False,
            yes=True,
            inbox=False,
            verify="auto",
            verify_command=None,
            critic=None,
            require_tools=False,
            goal=False,
            max_context_tokens=None,
            predict=True,
            auto_compact=False,
            max_repair=_gateway_max_repair_attempts(),
            profile="adaptive",
            domain="coding",
            phases=None,
            loop_detect=True,
            contracts=True,
            tips=True,
            include_workspace_context=False,
            silent=True,
            config=config,
            build_storage=build_storage,
            resolve_task_attachment=_noop_task_attachment,
            resolve_runtime_strategy=_resolve_runtime_strategy,
            build_verifier=_build_verifier,
            build_critic=_build_critic,
            build_adapter=_build_adapter,
            build_tools=_build_tools,
            build_agent=_build_agent,
            print_defense_ledger=_print_defense_ledger,
            render=lambda _event: None,
            default_system_prompt=system_prompt,
            console=console,
        )
        or ""
    )


async def _run_gateway_conversation(
    *,
    cwd: Path,
    session_store: GatewaySessionStore,
    transport: str,
    user_id: str,
    thread_id: str,
    message: str,
    max_steps: int = 20,
) -> dict[str, object]:
    from harness.cli.__main__ import _DEFAULT_SYSTEM_PROMPT

    wa_config = load_whatsapp_bridge_config(cwd)
    session = session_store.get_or_create_session(
        transport=transport,
        user_id=user_id,
        thread_id=thread_id,
    )
    thread_context = _thread_context_from_session(session)
    cfg = _load_cli_config(None)
    has_whatsapp_config = default_whatsapp_config_path(cwd).exists()
    whatsapp_provider = wa_config.provider if has_whatsapp_config else ""
    whatsapp_model = wa_config.model if has_whatsapp_config else ""
    provider = (
        _gateway_env_override("HARNESS_GATEWAY_PROVIDER")
        or whatsapp_provider
        or cfg.default_provider
        or _default_gateway_provider()
    )
    chain = _resolve_chain(failover_flag=None, provider_flag=provider, config=cfg)
    model = (
        _gateway_env_override("HARNESS_GATEWAY_MODEL")
        or whatsapp_model
        or cfg.default_model
        or _default_gateway_model(provider)
    )
    harness_base_session_id = (
        str(session.metadata.get("harness_session_base_id", "")).strip()
        or str(session.metadata.get("harness_session_id", "")).strip()
        or f"sess_{session.id.replace('-', '_')}"
    )
    harness_session_mode = "general"
    harness_session_id = (
        str(session.metadata.get(f"harness_session_id_{harness_session_mode}", "")).strip()
        or str(session.metadata.get("harness_session_id", "")).strip()
        or f"{harness_base_session_id}_{harness_session_mode}"
    )
    if harness_base_session_id.endswith(("_work", "_general")):
        harness_base_session_id = harness_base_session_id.rsplit("_", 1)[0]
        harness_session_id = (
            str(session.metadata.get(f"harness_session_id_{harness_session_mode}", "")).strip()
            or f"{harness_base_session_id}_{harness_session_mode}"
        )
    runtime_session_key = _runtime_session_key(provider=chain[0], model=model)
    keyed_session_metadata_name = f"harness_session_id_{runtime_session_key}"
    harness_session_id = (
        str(session.metadata.get(keyed_session_metadata_name, "")).strip()
        or f"{harness_base_session_id}_{harness_session_mode}_{runtime_session_key}"
    )
    runtime_prompt = _with_shared_gateway_context(
        session_store=session_store,
        session=session,
        message=message,
    )
    try:
        final_text = await asyncio.wait_for(
            _run_gateway_chat_turn(
                cwd=cwd,
                prompt=runtime_prompt,
                chain=chain,
                model=model,
                session_id=harness_session_id,
                max_steps=max_steps,
                config=cfg,
                system_prompt=_DEFAULT_SYSTEM_PROMPT,
            ),
            timeout=_gateway_turn_timeout_seconds(),
        )
    except typer.Exit:
        final_text = (
            await _latest_run_failure_reply(cwd=cwd, session_id=harness_session_id)
            or await _latest_unverified_workspace_change_reply(
                cwd=cwd,
                session_id=harness_session_id,
            )
            or await _latest_verification_failure_reply(
                cwd=cwd,
                session_id=harness_session_id,
            )
            or await _latest_successful_tool_evidence_reply(
                cwd=cwd,
                session_id=harness_session_id,
            )
            or ""
        )
    except TimeoutError:
        final_text = (
            await _latest_run_failure_reply(cwd=cwd, session_id=harness_session_id)
            or await _latest_unverified_workspace_change_reply(
                cwd=cwd,
                session_id=harness_session_id,
            )
            or await _latest_verification_failure_reply(
                cwd=cwd,
                session_id=harness_session_id,
            )
            or await _latest_tool_failure_reply(cwd=cwd, session_id=harness_session_id)
            or await _latest_successful_tool_evidence_reply(
                cwd=cwd,
                session_id=harness_session_id,
            )
            or (
                "Harness chat timed out while waiting for the model. "
                "The WhatsApp bridge is still running; please try again."
            )
        )
    if (final_text or "").strip() == _RUNTIME_VERIFICATION_HANDOFF_TEXT:
        final_text = (
            await _latest_unverified_workspace_change_reply(
                cwd=cwd,
                session_id=harness_session_id,
            )
            or await _latest_verification_failure_reply(
                cwd=cwd,
                session_id=harness_session_id,
            )
            or await _latest_successful_verified_reply(
                cwd=cwd,
                session_id=harness_session_id,
            )
            or final_text
        )
    reply_text = (final_text or "").strip()
    if not reply_text:
        reply_text = (
            "Harness could not generate a conversational reply. "
            "Configure a working model/provider for chat runs, then try again."
        )
    metadata = {
        **_prune_gateway_metadata(session.metadata),
        "harness_session_id": harness_session_id,
        "harness_session_base_id": harness_base_session_id,
        f"harness_session_id_{harness_session_mode}": harness_session_id,
        "harness_session_mode": harness_session_mode,
        keyed_session_metadata_name: harness_session_id,
        "thread_context": _updated_thread_context(
            thread_context,
            user_message=message,
            assistant_reply=reply_text,
        ),
        "thread_summary": (
            f"Latest user ask: {_compact_context_text(message, limit=120)}. "
            f"Last reply: {_compact_context_text(reply_text, limit=180)}"
        ).strip(),
    }
    updated = replace(
        session,
        last_command="chat",
        updated_at=_utcnow_text(),
        metadata=metadata,
    )
    session_store.save_session(updated)
    profile = session_store.get_or_create_profile(transport=transport, user_id=user_id)
    recent_threads = [item for item in profile.recent_threads if item != thread_id]
    recent_threads.append(thread_id)
    session_store.save_profile(
        replace(
            profile,
            recent_threads=recent_threads[-8:],
            updated_at=updated.updated_at,
        )
    )
    reply = {
        "session_id": updated.id,
        "command": "chat",
        "status": "ok",
        "text": reply_text,
        "data": {
            "harness_session_id": harness_session_id,
            "harness_session_mode": harness_session_mode,
        },
    }
    return {
        "reply": reply,
        "session": updated.to_dict(),
    }


async def _run_gateway_converse_payload(
    *,
    working_dir: Path,
    message: str,
    transport: str,
    user_id: str,
    thread_id: str,
    max_steps: int = 20,
) -> dict[str, object]:
    session_store = GatewaySessionStore(root=default_gateway_root(working_dir))
    return await _run_gateway_conversation(
        cwd=working_dir,
        session_store=session_store,
        transport=transport,
        user_id=user_id,
        thread_id=thread_id,
        message=message,
        max_steps=max_steps,
    )


async def _run_gateway_dispatch_payload(
    *,
    working_dir: Path,
    message: str,
    transport: str,
    user_id: str,
    thread_id: str,
    db: Path | None = None,
    in_memory: bool = False,
) -> dict[str, object]:
    hooks = _load_hooks(working_dir)
    session_store = GatewaySessionStore(root=default_gateway_root(working_dir))
    scheduler_store = SchedulerStore(root=working_dir / ".harness" / "scheduler")
    storage = build_storage(db=db, in_memory=in_memory, cwd=working_dir)
    approval_store = cast(ApprovalStore, storage)
    try:
        reply, session = await dispatch_gateway_message(
            cwd=working_dir,
            session_store=session_store,
            scheduler_store=scheduler_store,
            message=GatewayMessage(
                id=f"{transport}-{user_id}-{thread_id}",
                transport=transport,
                user_id=user_id,
                thread_id=thread_id,
                text=message,
            ),
            approval_store=approval_store,
            hooks=hooks,
        )
        return {
            "reply": reply.to_dict(),
            "session": session.to_dict(),
        }
    finally:
        close = getattr(storage, "close", None)
        if callable(close):
            result = close()
            if isawaitable(result):
                await result


async def _run_gateway_receive_payload(
    *,
    working_dir: Path,
    message: str,
    transport: str,
    user_id: str,
    thread_id: str,
    max_steps: int = 20,
) -> dict[str, object]:
    if is_gateway_control_message(message):
        return await _run_gateway_dispatch_payload(
            working_dir=working_dir,
            message=message,
            transport=transport,
            user_id=user_id,
            thread_id=thread_id,
        )
    return await _run_gateway_converse_payload(
        working_dir=working_dir,
        message=message,
        transport=transport,
        user_id=user_id,
        thread_id=thread_id,
        max_steps=max_steps,
    )


__all__ = [
    "_default_gateway_model",
    "_default_gateway_provider",
    "_gateway_max_repair_attempts",
    "_gateway_turn_timeout_seconds",
    "_run_failure_reply_from_activity",
    "_run_gateway_chat_turn",
    "_run_gateway_conversation",
    "_run_gateway_converse_payload",
    "_run_gateway_dispatch_payload",
    "_run_gateway_receive_payload",
    "_successful_verified_reply_from_activity",
    "_tool_failure_reply_from_messages",
    "_unverified_workspace_change_reply_from_activity",
    "_verification_failure_reply_from_activity",
]
