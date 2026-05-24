from __future__ import annotations

import asyncio
import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from harness.cli.config import HarnessConfig
from harness.core import Agent, ContextBudget, ContextCompactor, RunRequest, Session, Storage
from harness.storage.sqlite import SQLiteStorage

_HELP_TEXT = (
    "/help              show this help\n"
    "/quit, /exit, /q   exit the chat\n"
    "/tools             list registered tools and effective approval\n"
    "/session           show current session id and turn count\n"
    "/diff              show file changes made this session\n"
    "/clear             clear the terminal\n"
    "/model [name]      show or switch the active model mid-session\n"
)

SlashHandler = Callable[..., Awaitable[bool]]


def run_chat_command(
    *,
    model: str | None,
    provider: str | None,
    base_url: str | None,
    cwd: Path | None,
    db: Path | None,
    in_memory: bool,
    session_id: str | None,
    task_ref: str | None,
    max_steps: int,
    failover: str | None,
    yes: bool,
    inbox: bool,
    verify: str | None,
    require_tools: bool,
    max_context_tokens: int | None,
    config_path: Path | None,
    auto_compact: bool,
    verbose: bool,
    console: Console,
    configure_logging: Callable[..., None],
    load_cli_config: Callable[[Path | None], HarnessConfig],
    resolve_chain: Callable[..., list[str]],
    run_async: Callable[[Any], Any],
    build_storage: Callable[..., Storage],
    resolve_task_attachment: Callable[..., Any],
    build_verifier: Callable[..., Any],
    build_adapter: Callable[..., Any],
    build_agent: Callable[..., Agent],
    render: Callable[[Any], None],
    render_session_diff: Callable[[Any, Console], None],
    default_system_prompt: str,
) -> None:
    configure_logging(level="DEBUG" if verbose else "INFO")
    if not yes and os.environ.get("HARNESS_YES"):
        yes = True
    if verify == "none":
        verify = None
    cfg = load_cli_config(config_path)
    chain = resolve_chain(failover_flag=failover, provider_flag=provider, config=cfg)
    effective_model = model or cfg.default_model or "llama3.2"
    working_dir = (cwd or Path.cwd()).resolve()
    if not working_dir.exists() or not working_dir.is_dir():
        console.print(f"[red]--cwd does not exist or is not a directory: {working_dir}[/red]")
        raise typer.Exit(2)

    try:
        run_async(
            chat_loop(
                chain=chain,
                base_url=base_url,
                model=effective_model,
                cwd=working_dir,
                db=db,
                in_memory=in_memory,
                session_id=session_id,
                task_ref=task_ref,
                max_steps=max_steps,
                yes=yes,
                inbox=inbox,
                verify=verify,
                require_tools=require_tools,
                max_context_tokens=max_context_tokens,
                auto_compact=auto_compact,
                config=cfg,
                console=console,
                build_storage=build_storage,
                resolve_task_attachment=resolve_task_attachment,
                build_verifier=build_verifier,
                build_adapter=build_adapter,
                build_agent=build_agent,
                render=render,
                render_session_diff=render_session_diff,
                default_system_prompt=default_system_prompt,
            )
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]bye[/yellow]")
        raise typer.Exit(130) from None


async def chat_loop(
    *,
    chain: list[str],
    base_url: str | None,
    model: str,
    cwd: Path,
    db: Path | None,
    in_memory: bool,
    session_id: str | None,
    task_ref: str | None,
    max_steps: int,
    yes: bool,
    inbox: bool,
    verify: str | None,
    require_tools: bool = False,
    max_context_tokens: int | None,
    auto_compact: bool = False,
    config: HarnessConfig,
    console: Console,
    build_storage: Callable[..., Storage],
    resolve_task_attachment: Callable[..., Any],
    build_verifier: Callable[..., Any],
    build_adapter: Callable[..., Any],
    build_agent: Callable[..., Agent],
    render: Callable[[Any], None],
    render_session_diff: Callable[[Any, Console], None],
    default_system_prompt: str,
) -> None:
    from uuid import uuid4

    storage = build_storage(db=db, in_memory=in_memory, cwd=cwd)
    try:
        existing: Session | None = None
        if session_id:
            existing = await storage.get(session_id)
        current_session_id = session_id or f"sess_{uuid4().hex[:12]}"

        task_id, _task = await resolve_task_attachment(storage, task_ref, current_session_id)

        verifier = build_verifier(
            verify,
            chain=chain,
            model=model,
            config=config,
            build_adapter=build_adapter,
            cwd=cwd,
        )
        budget = (
            ContextBudget(max_tokens=max_context_tokens) if max_context_tokens is not None else None
        )
        compactor: ContextCompactor | None = None
        if auto_compact:
            adapter = build_adapter(chain[0], base_url=base_url, config=config)
            compactor = ContextCompactor(adapter=adapter, model=model)
        agent = build_agent(
            chain=chain,
            base_url=base_url,
            model=model,
            storage=storage,
            cwd=cwd,
            config=config,
            yes=yes,
            inbox=inbox,
            activity_store=storage,  # type: ignore[arg-type]
            approval_store=storage,  # type: ignore[arg-type]
            verifier=verifier,
            budget=budget,
            memory_store=storage,  # type: ignore[arg-type]
            system_prompt=default_system_prompt,
            compactor=compactor,
        )

        first_turn = existing is None

        chain_label = chain[0]
        if len(chain) > 1:
            chain_label += "  [dim](failover: " + ", ".join(chain[1:]) + ")[/dim]"
        intro = (
            f"[bold]Session:[/bold] {current_session_id}"
            + (" [dim](resumed)[/dim]" if existing else "")
            + f"\n[bold]Provider:[/bold] {chain_label}"
            f"\n[bold]Model:[/bold] {model}"
            f"\n[bold]Tools:[/bold] {', '.join(agent.tools.names())}"
            f"\n[bold]CWD:[/bold] {cwd}\n\n"
            f"[dim]Type /help for commands. /quit to exit.[/dim]"
        )
        console.print(Panel(intro, title="harness chat", expand=False))

        slash_handler = _make_slash_handler(
            console=console, render_session_diff=render_session_diff
        )

        while True:
            try:
                user_input = console.input("\n[bold cyan]> [/bold cyan]").strip()
            except EOFError:
                console.print("\n[yellow]bye[/yellow]")
                return
            except KeyboardInterrupt:
                console.print("\n[yellow]bye[/yellow]")
                return

            if not user_input:
                continue

            if user_input.startswith("/"):
                keep_going = await slash_handler(
                    user_input, agent=agent, session_id=current_session_id, storage=storage
                )
                if not keep_going:
                    return
                continue

            try:
                if first_turn:
                    request_kwargs: dict[str, object] = {
                        "prompt": user_input,
                        "session_id": current_session_id,
                        "model": model,
                        "max_steps": max_steps,
                        "require_tool_use": require_tools,
                    }
                    if task_id:
                        request_kwargs["task_id"] = task_id
                    request = RunRequest(**request_kwargs)  # type: ignore[arg-type]
                    async for event in agent.run(request):
                        render(event)
                    first_turn = False
                else:
                    async for event in agent.resume(
                        current_session_id, prompt=user_input, max_steps=max_steps
                    ):
                        render(event)
            except (KeyboardInterrupt, asyncio.CancelledError):
                console.print("\n[yellow]cancelled[/yellow]")
            except Exception as exc:
                console.print(f"\n[red]Error:[/red] {exc!s}")
    finally:
        if isinstance(storage, SQLiteStorage):
            await storage.close()


def _make_slash_handler(
    *,
    console: Console,
    render_session_diff: Callable[[Any, Console], None],
) -> Callable[..., Awaitable[bool]]:
    registry: dict[str, SlashHandler] = {}

    def slash(name: str) -> Callable[[SlashHandler], SlashHandler]:
        def decorator(fn: SlashHandler) -> SlashHandler:
            registry[name] = fn
            return fn

        return decorator

    @slash("/quit")
    @slash("/exit")
    @slash("/q")
    async def slash_quit(line: str, *, agent: Agent, session_id: str, storage: Storage) -> bool:
        console.print("[yellow]bye[/yellow]")
        return False

    @slash("/help")
    async def slash_help(line: str, *, agent: Agent, session_id: str, storage: Storage) -> bool:
        console.print(Panel(_HELP_TEXT.rstrip(), title="commands", expand=False))
        return True

    @slash("/tools")
    async def slash_tools(line: str, *, agent: Agent, session_id: str, storage: Storage) -> bool:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Tool")
        table.add_column("Approval")
        for tool in agent.tools.all():
            effective = agent.approval_policy.decide(tool)
            color = {"auto": "green", "prompt": "yellow", "deny": "red"}.get(effective, "white")
            table.add_row(tool.name, f"[{color}]{effective}[/{color}]")
        console.print(table)
        return True

    @slash("/session")
    async def slash_session(line: str, *, agent: Agent, session_id: str, storage: Storage) -> bool:
        session = await storage.get(session_id)
        if session is None:
            console.print(f"[dim]Session {session_id} (no turns yet)[/dim]")
        else:
            console.print(
                f"[dim]Session {session_id}, status: {session.status}, "
                f"{len(session.messages)} messages[/dim]"
            )
        return True

    @slash("/diff")
    async def slash_diff(line: str, *, agent: Agent, session_id: str, storage: Storage) -> bool:
        activity = await storage.list_activity(session_id=session_id)  # type: ignore[attr-defined]
        render_session_diff(activity, console)
        return True

    @slash("/clear")
    async def slash_clear(line: str, *, agent: Agent, session_id: str, storage: Storage) -> bool:
        console.clear()
        return True

    @slash("/model")
    async def slash_model(line: str, *, agent: Agent, session_id: str, storage: Storage) -> bool:
        parts = line.split(None, 1)
        if len(parts) == 1:
            console.print(f"[dim]Active model: {agent.default_model}[/dim]")
        else:
            new_model = parts[1].strip()
            agent.default_model = new_model
            console.print(f"[green]Switched model to:[/green] {new_model}")
        return True

    async def handle_slash(line: str, *, agent: Agent, session_id: str, storage: Storage) -> bool:
        cmd = line.split(None, 1)[0].lower()
        handler = registry.get(cmd)
        if handler is None:
            console.print(f"[red]Unknown command:[/red] {cmd}.  Try /help.")
            return True
        return await handler(line, agent=agent, session_id=session_id, storage=storage)

    return handle_slash
