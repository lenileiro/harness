from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from harness.cli.gateway_runtime import (
    _default_gateway_model,
    _run_gateway_converse_payload,
    _run_gateway_dispatch_payload,
    _run_gateway_receive_payload,
)
from harness.core import (
    GatewaySessionStore,
    WhatsAppBridgeConfig,
    default_gateway_root,
)
from harness.core.gateway_whatsapp import (
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

console = Console()

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
    payload = asyncio.run(
        _run_gateway_dispatch_payload(
            working_dir=working_dir,
            message=message,
            transport=transport,
            user_id=user_id,
            thread_id=thread_id,
            db=db,
            in_memory=in_memory,
        )
    )
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


@gateway_app.command("receive")
def gateway_receive_command(
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
    payload = asyncio.run(
        _run_gateway_receive_payload(
            working_dir=working_dir,
            message=message,
            transport=transport,
            user_id=user_id,
            thread_id=thread_id,
            max_steps=max_steps,
        )
    )
    if json_output:
        _emit_json(payload)
        return
    reply = payload["reply"]
    assert isinstance(reply, dict)
    console.print(reply["text"])


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
    payload = asyncio.run(
        _run_gateway_converse_payload(
            working_dir=working_dir,
            message=message,
            transport=transport,
            user_id=user_id,
            thread_id=thread_id,
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
    max_gateway_concurrency: int | None = typer.Option(None, "--max-gateway-concurrency"),
    max_gateway_queue: int | None = typer.Option(None, "--max-gateway-queue"),
    gateway_child_timeout_seconds: int | None = typer.Option(
        None, "--gateway-child-timeout-seconds"
    ),
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
        max_gateway_concurrency=max_gateway_concurrency or existing.max_gateway_concurrency,
        max_gateway_queue=(
            existing.max_gateway_queue if max_gateway_queue is None else max(0, max_gateway_queue)
        ),
        gateway_child_timeout_seconds=(
            gateway_child_timeout_seconds or existing.gateway_child_timeout_seconds
        ),
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
    console.print(f"[green]max_gateway_concurrency[/green]={status.config.max_gateway_concurrency}")
    console.print(f"[green]max_gateway_queue[/green]={status.config.max_gateway_queue}")
    console.print(
        f"[green]gateway_child_timeout_seconds[/green]={status.config.gateway_child_timeout_seconds}"
    )
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
    table.add_row("max_gateway_concurrency", str(status.config.max_gateway_concurrency))
    table.add_row("max_gateway_queue", str(status.config.max_gateway_queue))
    table.add_row(
        "gateway_child_timeout_seconds",
        str(status.config.gateway_child_timeout_seconds),
    )
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
    "_default_gateway_model",
    "_run_gateway_converse_payload",
    "_run_gateway_dispatch_payload",
    "_run_gateway_receive_payload",
    "gateway_app",
    "is_whatsapp_paired",
    "run_whatsapp_pairing",
    "send_whatsapp_text_message",
    "whatsapp_app",
]
