from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console
from rich.table import Table

from harness.cli.common import (
    _build_adapter,
    _build_tools,
    _load_cli_config,
    _resolve_chain,
)
from harness.cli.common import (
    console as common_console,
)
from harness.cli.plugins import load_cli_hook_providers
from harness.cli.run_commands import run_once as _run_once_impl
from harness.cli.runtime_helpers import (
    build_critic as _build_critic,
)
from harness.cli.runtime_helpers import build_storage
from harness.cli.runtime_helpers import (
    build_verifier as _build_verifier,
)
from harness.cli.runtime_helpers import (
    resolve_runtime_strategy as _resolve_runtime_strategy,
)
from harness.core import (
    ApprovalStore,
    Done,
    GatewayMessage,
    Message,
    TextDelta,
    default_gateway_root,
)
from harness.core.extensions import LifecycleHook
from harness.core.gateway_router import dispatch_gateway_message
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.gateway_whatsapp import (
    WhatsAppBridgeConfig,
    clear_whatsapp_session,
    install_whatsapp_bridge_dependencies,
    is_whatsapp_paired,
    load_whatsapp_bridge_config,
    read_whatsapp_bridge_status,
    run_whatsapp_pairing,
    save_whatsapp_bridge_config,
    send_whatsapp_text_message,
    start_whatsapp_bridge,
)
from harness.core.scheduler_store import SchedulerStore

console = Console()


def _default_gateway_model(provider: str) -> str:
    normalized = provider.strip().lower()
    if normalized == "openrouter":
        return "google/gemma-4-31b-it"
    return "gemma4:latest"


gateway_app = typer.Typer(
    name="gateway",
    help="Dispatch transport-neutral remote control messages.",
    no_args_is_help=True,
)
whatsapp_app = typer.Typer(
    name="whatsapp",
    help="Manage local WhatsApp Web pairing and bridge runtime.",
    no_args_is_help=True,
)
gateway_app.add_typer(whatsapp_app, name="whatsapp")


def _emit_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2))


def _load_hooks(cwd: Path) -> tuple[LifecycleHook, ...]:
    hooks: list[LifecycleHook] = []
    for provider in load_cli_hook_providers(cwd):
        hooks.extend(provider.hooks())
    return tuple(hooks)


def _utcnow_text() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _thread_context_from_session(session: Any) -> list[str]:
    raw_thread_context = session.metadata.get("thread_context", [])
    if not isinstance(raw_thread_context, list):
        return []
    return [str(item).strip() for item in raw_thread_context if str(item).strip()]


def _linked_work_refs_from_session(session: Any) -> list[str]:
    raw_linked = session.metadata.get("linked_work_items", [])
    if not isinstance(raw_linked, list):
        return []
    return [str(item).strip() for item in raw_linked if str(item).strip()]


def _thread_summary_for_session(session: Any) -> str:
    return str(session.metadata.get("thread_summary", "")).strip()


def _work_context_lines(
    *,
    session_store: GatewaySessionStore,
    transport: str,
    user_id: str,
    thread_id: str,
    linked_work_refs: list[str],
) -> list[str]:
    profile = session_store.get_or_create_profile(transport=transport, user_id=user_id)
    active_by_ref = {item.ref: item for item in profile.active_work}
    selected_refs = linked_work_refs or [item.ref for item in profile.active_work[-3:]]
    lines: list[str] = []
    for ref in selected_refs:
        item = active_by_ref.get(ref)
        if item is None:
            continue
        title = item.title or item.ref
        summary = item.summary or item.status
        suffix = ""
        if item.source_thread_id and item.source_thread_id != thread_id:
            suffix = f" from chat {item.source_thread_id}"
        lines.append(f"{title} [{item.kind}] - {summary}{suffix}")
    return lines[-4:]


def _other_thread_context_lines(
    *,
    session_store: GatewaySessionStore,
    transport: str,
    user_id: str,
    thread_id: str,
) -> list[str]:
    lines: list[str] = []
    for item in session_store.list_user_sessions(transport=transport, user_id=user_id):
        if item.thread_id == thread_id:
            continue
        summary = _thread_summary_for_session(item)
        if not summary:
            continue
        lines.append(f"{item.thread_id}: {summary}")
    return lines[-3:]


def _contextualize_user_prompt(
    *,
    message: str,
    thread_context: list[str],
    work_context: list[str],
    related_threads: list[str],
) -> str:
    sections: list[str] = []
    if thread_context:
        sections.append(
            "Current chat context:\n" + "\n".join(f"- {line}" for line in thread_context[-6:])
        )
    if work_context:
        sections.append(
            "Shared active work for this user:\n" + "\n".join(f"- {line}" for line in work_context)
        )
    if related_threads:
        sections.append(
            "Other recent chats for this user:\n"
            + "\n".join(f"- {line}" for line in related_threads)
        )
    if not sections:
        return message
    sections.append(
        "Use the shared work and other chat context only when it is relevant. Do not merge unrelated threads."
    )
    sections.append(f"User message:\n{message}")
    return "\n\n".join(sections)


async def _noop_task_attachment(*_args: object, **_kwargs: object) -> tuple[None, None]:
    return None, None


async def _noop_print_defense_ledger(*_args: object, **_kwargs: object) -> None:
    return None


def _progress_summary_from_event(event: Any) -> str | None:
    event_type = getattr(event, "type", "")
    if event_type == "step_started":
        description = str(getattr(event, "description", "") or "").strip()
        if description:
            return f"Working step {getattr(event, 'step', '?')}: {description}"
        return f"Working step {getattr(event, 'step', '?')}."
    if event_type == "tool_call":
        call = getattr(event, "call", None)
        tool_name = str(getattr(call, "name", "") or "").strip()
        arguments = getattr(call, "arguments", {}) or {}
        if tool_name == "write_file":
            path = str(arguments.get("path", "")).strip()
            return f"Writing {path}." if path else "Writing a file."
        if tool_name == "edit_file":
            path = str(arguments.get("path", "")).strip()
            return f"Editing {path}." if path else "Editing a file."
        if tool_name == "shell":
            command = str(arguments.get("command", "")).strip()
            if command:
                compact = command if len(command) <= 80 else command[:77] + "..."
                return f"Running: {compact}"
            return "Running a shell command."
        if tool_name == "verify_work":
            return "Verifying the result."
        if tool_name == "fetch_url":
            url = str(arguments.get("url", "")).strip()
            return f"Fetching {url}." if url else "Fetching data."
        if tool_name:
            return f"Using {tool_name}."
    if event_type == "verification":
        result = getattr(event, "result", None)
        if result is None:
            return "Verification finished."
        if bool(getattr(result, "can_finish", False)):
            return "Verification passed."
        reason = str(getattr(result, "reason", "") or "").strip()
        return (
            f"Verification found an issue: {reason}" if reason else "Verification found an issue."
        )
    if event_type == "critique":
        return "Revising the approach after a failed attempt."
    if event_type == "error":
        error_text = str(getattr(event, "error", "") or "").strip()
        return f"Hit an error: {error_text}" if error_text else "Hit an error."
    return None


async def _stream_text_response(
    *,
    provider: str,
    model: str,
    config: Any,
    messages: list[Message],
) -> str:
    adapter = _build_adapter(provider, base_url=None, config=config)
    text_parts: list[str] = []
    async for event in adapter.stream(
        model=model, messages=messages, max_tokens=80, temperature=0.4
    ):
        if isinstance(event, TextDelta):
            text_parts.append(event.text)
        elif isinstance(event, Done):
            if event.final_message and event.final_message.content:
                return event.final_message.content.strip()
            break
    return "".join(text_parts).strip()


async def _generate_progress_note(
    *,
    provider: str,
    model: str,
    config: Any,
    thread_context: list[str],
    user_prompt: str,
    event_summary: str,
    work_context: list[str],
) -> str:
    thread_lines = "\n".join(f"- {line}" for line in thread_context if line.strip())
    context_lines = "\n".join(f"- {line}" for line in work_context if line.strip())
    messages = [
        Message(
            role="system",
            content=(
                "You write short live progress updates for a user while an AI coding agent works. "
                "Use the real observed work context. Write one sentence, under 24 words, plain text, "
                "first person, no markdown, no promises, and no filler. Mention the concrete work being done."
            ),
        ),
        Message(
            role="user",
            content=(
                f"Recent conversation context:\n{thread_lines or '- no prior thread context'}\n"
                f"Original user request: {user_prompt}\n"
                f"Recent work context:\n{context_lines or '- no prior work yet'}\n"
                f"Observed runtime event: {event_summary}\n"
                "Write the progress update now."
            ),
        ),
    ]
    return await _stream_text_response(
        provider=provider,
        model=model,
        config=config,
        messages=messages,
    )


def _make_gateway_progress_renderer(
    *,
    cwd: Path,
    transport: str,
    user_id: str,
    thread_id: str,
    provider: str,
    model: str,
    config: Any,
    thread_context: list[str],
    user_prompt: str,
) -> tuple[Callable[[Any], None], set[asyncio.Task[object]]]:
    pending: set[asyncio.Task[object]] = set()
    last_sent: dict[str, float] = {"ts": 0.0}
    last_text: dict[str, str] = {"value": ""}
    last_summary: dict[str, str] = {"value": ""}
    recent_summaries: list[str] = []

    def _schedule_message(summary: str) -> None:
        if transport != "whatsapp":
            return
        normalized_summary = summary.strip()
        if not normalized_summary:
            return
        loop = asyncio.get_running_loop()
        now = loop.time()
        if normalized_summary == last_summary["value"]:
            return
        if now - last_sent["ts"] < 2.5:
            return
        recent_summaries.append(normalized_summary)
        del recent_summaries[:-5]
        last_summary["value"] = normalized_summary
        last_sent["ts"] = now

        async def _send() -> None:
            text = await _generate_progress_note(
                provider=provider,
                model=model,
                config=config,
                thread_context=thread_context,
                user_prompt=user_prompt,
                event_summary=normalized_summary,
                work_context=list(recent_summaries),
            )
            final_text = text.strip() or normalized_summary
            if final_text == last_text["value"]:
                return
            last_text["value"] = final_text
            await asyncio.to_thread(
                send_whatsapp_text_message,
                cwd=cwd,
                to=user_id,
                text=final_text,
            )

        task = asyncio.create_task(_send())
        pending.add(task)
        task.add_done_callback(lambda finished: pending.discard(finished))

    def _render(event: Any) -> None:
        summary = _progress_summary_from_event(event)
        if summary:
            _schedule_message(summary)

    return _render, pending


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
    from harness.cli.__main__ import _DEFAULT_SYSTEM_PROMPT, _build_agent

    wa_config = load_whatsapp_bridge_config(cwd)
    session = session_store.get_or_create_session(
        transport=transport,
        user_id=user_id,
        thread_id=thread_id,
    )
    thread_context = _thread_context_from_session(session)
    linked_work_refs = _linked_work_refs_from_session(session)
    work_context = _work_context_lines(
        session_store=session_store,
        transport=transport,
        user_id=user_id,
        thread_id=thread_id,
        linked_work_refs=linked_work_refs,
    )
    related_threads = _other_thread_context_lines(
        session_store=session_store,
        transport=transport,
        user_id=user_id,
        thread_id=thread_id,
    )
    contextual_prompt = _contextualize_user_prompt(
        message=message,
        thread_context=thread_context,
        work_context=work_context,
        related_threads=related_threads,
    )
    harness_session_id = (
        str(session.metadata.get("harness_session_id", "")).strip()
        or f"sess_{session.id.replace('-', '_')}"
    )
    cfg = _load_cli_config(None)
    provider = wa_config.provider or cfg.default_provider or "ollama"
    chain = _resolve_chain(failover_flag=None, provider_flag=provider, config=cfg)
    model = wa_config.model or cfg.default_model or _default_gateway_model(provider)
    progress_render, progress_tasks = _make_gateway_progress_renderer(
        cwd=cwd,
        transport=transport,
        user_id=user_id,
        thread_id=thread_id,
        provider=provider,
        model=model,
        config=cfg,
        thread_context=list(thread_context),
        user_prompt=message,
    )
    final_text = await _run_once_impl(
        prompt=contextual_prompt,
        model=model,
        chain=chain,
        base_url=None,
        cwd=cwd,
        max_steps=max_steps,
        max_output_tokens=None,
        session_id=harness_session_id,
        task_ref=None,
        db=None,
        in_memory=False,
        yes=True,
        inbox=False,
        verify=None,
        verify_command=None,
        critic=None,
        require_tools=False,
        goal=False,
        max_context_tokens=None,
        predict=False,
        auto_compact=False,
        max_repair=3,
        profile="minimal",
        domain="coding",
        phases=None,
        loop_detect=True,
        contracts=True,
        tips=True,
        silent=True,
        config=cfg,
        build_storage=build_storage,
        resolve_task_attachment=_noop_task_attachment,
        resolve_runtime_strategy=_resolve_runtime_strategy,
        build_verifier=_build_verifier,
        build_critic=_build_critic,
        build_adapter=_build_adapter,
        build_tools=_build_tools,
        build_agent=_build_agent,
        print_defense_ledger=_noop_print_defense_ledger,
        render=progress_render,
        default_system_prompt=_DEFAULT_SYSTEM_PROMPT,
        console=common_console,
    )
    if progress_tasks:
        await asyncio.gather(*tuple(progress_tasks), return_exceptions=True)
    reply_text = (final_text or "").strip()
    if not reply_text:
        reply_text = (
            "Harness could not generate a conversational reply. "
            "Configure a working model/provider for chat runs, then try again."
        )
    updated = replace(
        session,
        last_command="chat",
        updated_at=_utcnow_text(),
        metadata={
            **session.metadata,
            "harness_session_id": harness_session_id,
            "thread_context": ([*thread_context, f"user: {message}", f"assistant: {reply_text}"])[
                -8:
            ],
            "thread_summary": (
                f"Latest user ask: {message}. " f"Last reply: {' '.join(reply_text.split())[:180]}"
            ).strip(),
        },
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
        "data": {"harness_session_id": harness_session_id},
    }
    return {
        "reply": reply,
        "session": updated.to_dict(),
    }


@gateway_app.command("dispatch")
def gateway_dispatch_command(
    *,
    message: str = typer.Option(..., "--message"),
    transport: str = typer.Option("local", "--transport"),
    user_id: str = typer.Option(..., "--user"),
    thread_id: str = typer.Option("default", "--thread"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    db: Path | None = typer.Option(None, "--db"),
    in_memory: bool = typer.Option(False, "--in-memory"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    hooks = _load_hooks(working_dir)
    session_store = GatewaySessionStore(root=default_gateway_root(working_dir))
    scheduler_store = SchedulerStore(root=working_dir / ".harness" / "scheduler")
    storage = build_storage(db=db, in_memory=in_memory, cwd=working_dir)
    approval_store = cast(ApprovalStore, storage)

    async def _go() -> dict[str, object]:
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

    try:
        payload = asyncio.run(_go())
        if json_output:
            _emit_json(payload)
            return
        reply = payload["reply"]
        assert isinstance(reply, dict)
        color = "green" if reply["status"] == "ok" else "red"
        console.print(f"[{color}]{reply['command']}[/{color}] {reply['text']}")
        session = payload["session"]
        assert isinstance(session, dict)
        console.print(f"session_id={session['id']}")
    finally:
        if hasattr(storage, "close"):
            asyncio.run(storage.close())  # type: ignore[attr-defined]


@gateway_app.command("list-sessions")
def gateway_list_sessions_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    session_store = GatewaySessionStore(root=default_gateway_root(working_dir))
    sessions = session_store.list_sessions()
    if json_output:
        _emit_json([item.to_dict() for item in sessions])
        return
    if not sessions:
        console.print("[dim]No gateway sessions found.[/dim]")
        return
    table = Table("id", "transport", "user", "thread", "last_command", "last_run_id")
    for item in sessions:
        table.add_row(
            item.id,
            item.transport,
            item.user_id,
            item.thread_id,
            item.last_command or "-",
            item.last_run_id or "-",
        )
    console.print(table)


@gateway_app.command("converse")
def gateway_converse_command(
    *,
    message: str = typer.Option(..., "--message"),
    transport: str = typer.Option("whatsapp", "--transport"),
    user_id: str = typer.Option(..., "--user"),
    thread_id: str = typer.Option("default", "--thread"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    max_steps: int = typer.Option(20, "--max-steps"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    session_store = GatewaySessionStore(root=default_gateway_root(working_dir))
    payload = asyncio.run(
        _run_gateway_conversation(
            cwd=working_dir,
            session_store=session_store,
            transport=transport,
            user_id=user_id,
            thread_id=thread_id,
            message=message,
            max_steps=max_steps,
        )
    )
    if json_output:
        _emit_json(payload)
        return
    reply = payload["reply"]
    assert isinstance(reply, dict)
    console.print(reply["text"])


def _normalize_allowed_users(values: list[str] | tuple[str, ...]) -> list[str]:
    normalized: list[str] = []
    for item in values:
        for part in item.split(","):
            value = part.strip().replace(" ", "")
            if value:
                normalized.append(value)
    return normalized


def _prompt_whatsapp_mode() -> str:
    choice = typer.prompt(
        "Choose WhatsApp mode: 1=personal number (self-chat), 2=separate bot number",
        default="1",
    ).strip()
    return "bot" if choice == "2" else "self-chat"


@whatsapp_app.command("setup")
def gateway_whatsapp_setup_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    mode: str | None = typer.Option(None, "--mode"),
    allowed_user: list[str] | None = typer.Option(None, "--allowed-user"),
    install: bool = typer.Option(True, "--install/--no-install"),
    pair: bool = typer.Option(True, "--pair/--no-pair"),
    force_repair: bool = typer.Option(False, "--force-repair"),
    bridge_port: int | None = typer.Option(None, "--bridge-port"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    existing = load_whatsapp_bridge_config(working_dir)
    selected_provider = (provider or existing.provider or "ollama").strip() or "ollama"
    if model is not None:
        selected_model = model.strip()
    elif provider is not None and selected_provider != existing.provider:
        selected_model = _default_gateway_model(selected_provider)
    else:
        selected_model = (existing.model or _default_gateway_model(selected_provider)).strip()
    selected_mode = (mode or existing.mode or _prompt_whatsapp_mode()).strip() or "self-chat"
    allowed_users = _normalize_allowed_users(allowed_user or [])
    if not allowed_users and existing.allowed_users:
        allowed_users = list(existing.allowed_users)
    if not allowed_users and selected_mode == "self-chat":
        owner = typer.prompt("Your personal WhatsApp number (digits, with country code)")
        allowed_users = _normalize_allowed_users((owner,))
    elif not allowed_users:
        raw = typer.prompt(
            "Allowed WhatsApp numbers (comma-separated, or * for anyone)",
            default="",
            show_default=False,
        )
        allowed_users = _normalize_allowed_users((raw,))

    config = WhatsAppBridgeConfig(
        enabled=existing.enabled,
        provider=selected_provider,
        model=selected_model or _default_gateway_model(selected_provider),
        mode="bot" if selected_mode == "bot" else "self-chat",
        allowed_users=allowed_users,
        bridge_port=bridge_port or existing.bridge_port,
        reply_prefix=existing.reply_prefix,
    )
    save_whatsapp_bridge_config(working_dir, config)

    if force_repair:
        clear_whatsapp_session(working_dir)

    if install:
        install_whatsapp_bridge_dependencies(working_dir)

    paired_now = is_whatsapp_paired(working_dir)
    if pair and (force_repair or not paired_now):
        run_whatsapp_pairing(working_dir)
        paired_now = is_whatsapp_paired(working_dir)

    config.enabled = paired_now or existing.enabled
    save_whatsapp_bridge_config(working_dir, config)
    status = read_whatsapp_bridge_status(working_dir)
    if json_output:
        _emit_json(status.to_dict())
        return
    console.print(f"[green]mode[/green]={status.config.mode}")
    console.print(f"[green]provider[/green]={status.config.provider}")
    console.print(f"[green]model[/green]={status.config.model}")
    console.print(f"[green]allowed_users[/green]={', '.join(status.config.allowed_users) or '-'}")
    console.print(f"[green]paired[/green]={status.paired}")
    console.print(f"[green]enabled[/green]={status.config.enabled}")


@whatsapp_app.command("status")
def gateway_whatsapp_status_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    status = read_whatsapp_bridge_status(working_dir)
    if json_output:
        _emit_json(status.to_dict())
        return
    table = Table("field", "value")
    table.add_row("enabled", str(status.config.enabled))
    table.add_row("mode", status.config.mode)
    table.add_row("allowed_users", ", ".join(status.config.allowed_users) or "-")
    table.add_row("paired", str(status.paired))
    table.add_row("dependencies_installed", str(status.dependencies_installed))
    table.add_row("bridge_running", str(status.bridge_running))
    table.add_row("bridge_connected", str(status.bridge_connected))
    table.add_row("bridge_port", str(status.config.bridge_port))
    table.add_row("session_dir", str(status.session_dir))
    console.print(table)


@whatsapp_app.command("pair")
def gateway_whatsapp_pair_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    install: bool = typer.Option(True, "--install/--no-install"),
    force_repair: bool = typer.Option(False, "--force-repair"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    if force_repair:
        clear_whatsapp_session(working_dir)
    if install:
        install_whatsapp_bridge_dependencies(working_dir)
    run_whatsapp_pairing(working_dir)
    config = load_whatsapp_bridge_config(working_dir)
    config.enabled = is_whatsapp_paired(working_dir)
    save_whatsapp_bridge_config(working_dir, config)
    status = read_whatsapp_bridge_status(working_dir)
    if json_output:
        _emit_json(status.to_dict())
        return
    console.print(f"[green]paired[/green]={status.paired}")


@whatsapp_app.command("start")
def gateway_whatsapp_start_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    install: bool = typer.Option(True, "--install/--no-install"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    if install:
        install_whatsapp_bridge_dependencies(working_dir)
    start_whatsapp_bridge(working_dir)


@whatsapp_app.command("send")
def gateway_whatsapp_send_command(
    *,
    to: str = typer.Option(..., "--to"),
    text: str = typer.Option(..., "--text"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    payload = send_whatsapp_text_message(cwd=working_dir, to=to, text=text)
    if json_output:
        _emit_json(payload)
        return
    console.print(f"[green]sent[/green] to={to}")


__all__ = [
    "gateway_app",
    "is_whatsapp_paired",
    "run_whatsapp_pairing",
    "send_whatsapp_text_message",
    "whatsapp_app",
]
