from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.table import Table

from harness.cli.plugins import load_cli_hook_providers
from harness.cli.runtime_helpers import build_storage
from harness.core import ApprovalStore, GatewayMessage, default_gateway_root
from harness.core.extensions import LifecycleHook
from harness.core.gateway_router import dispatch_gateway_message
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.gateway_whatsapp import extract_whatsapp_messages, send_whatsapp_text_message
from harness.core.scheduler_store import SchedulerStore

console = Console()

gateway_app = typer.Typer(
    name="gateway",
    help="Dispatch transport-neutral remote control messages.",
    no_args_is_help=True,
)


def _emit_json(payload: object) -> None:
    typer.echo(json.dumps(payload, indent=2))


def _load_hooks(cwd: Path) -> tuple[LifecycleHook, ...]:
    hooks: list[LifecycleHook] = []
    for provider in load_cli_hook_providers(cwd):
        hooks.extend(provider.hooks())
    return tuple(hooks)


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


@gateway_app.command("whatsapp-dispatch")
def gateway_whatsapp_dispatch_command(
    *,
    payload_path: Path = typer.Option(..., "--payload"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    db: Path | None = typer.Option(None, "--db"),
    in_memory: bool = typer.Option(False, "--in-memory"),
    send: bool = typer.Option(False, "--send"),
    phone_number_id: str | None = typer.Option(None, "--phone-number-id"),
    access_token: str | None = typer.Option(None, "--access-token"),
    api_version: str | None = typer.Option(None, "--api-version"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    incoming = extract_whatsapp_messages(payload)
    hooks = _load_hooks(working_dir)
    session_store = GatewaySessionStore(root=default_gateway_root(working_dir))
    scheduler_store = SchedulerStore(root=working_dir / ".harness" / "scheduler")
    storage = build_storage(db=db, in_memory=in_memory, cwd=working_dir)
    approval_store = cast(ApprovalStore, storage)

    async def _go() -> list[dict[str, object]]:
        replies: list[dict[str, object]] = []
        for item in incoming:
            reply, session = await dispatch_gateway_message(
                cwd=working_dir,
                session_store=session_store,
                scheduler_store=scheduler_store,
                message=item,
                approval_store=approval_store,
                hooks=hooks,
            )
            sent_payload: dict[str, object] | None = None
            if send:
                sent_payload = send_whatsapp_text_message(
                    to=item.user_id,
                    text=reply.text,
                    phone_number_id=phone_number_id,
                    access_token=access_token,
                    api_version=api_version,
                )
            replies.append(
                {
                    "message": item.to_dict(),
                    "reply": reply.to_dict(),
                    "session": session.to_dict(),
                    "sent": sent_payload,
                }
            )
        return replies

    try:
        results = asyncio.run(_go())
        if json_output:
            _emit_json(results)
            return
        if not results:
            console.print("[dim]No WhatsApp text messages found in payload.[/dim]")
            return
        for item in results:
            reply = item["reply"]
            assert isinstance(reply, dict)
            color = "green" if reply["status"] == "ok" else "red"
            console.print(f"[{color}]{reply['command']}[/{color}] {reply['text']}")
    finally:
        if hasattr(storage, "close"):
            asyncio.run(storage.close())  # type: ignore[attr-defined]


@gateway_app.command("whatsapp-send")
def gateway_whatsapp_send_command(
    *,
    to: str = typer.Option(..., "--to"),
    text: str = typer.Option(..., "--text"),
    phone_number_id: str | None = typer.Option(None, "--phone-number-id"),
    access_token: str | None = typer.Option(None, "--access-token"),
    api_version: str | None = typer.Option(None, "--api-version"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    payload = send_whatsapp_text_message(
        to=to,
        text=text,
        phone_number_id=phone_number_id,
        access_token=access_token,
        api_version=api_version,
    )
    if json_output:
        _emit_json(payload)
        return
    console.print(f"[green]sent[/green] to={to}")


__all__ = ["gateway_app"]
