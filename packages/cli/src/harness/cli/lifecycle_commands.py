from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from harness.storage.sqlite import SQLiteStorage, default_db_path
from harness.tasks import ActivityEvent


async def latest_session_id(storage: Any) -> str | None:
    sessions = await storage.list(limit=1)  # type: ignore[attr-defined]
    return sessions[0].id if sessions else None


async def append_phase_event(storage: Any, kind: str, name: str, notes: str) -> str | None:
    sid = await latest_session_id(storage)
    if sid is None:
        return None
    data: dict[str, Any] = {"phase": name}
    if notes:
        data["notes"] = notes
    event = ActivityEvent(session_id=sid, kind=kind, data=data)
    await storage.append_activity(event)  # type: ignore[attr-defined]
    return sid


def phase_declare_command(
    *,
    name: str,
    notes: str | None,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    run_async: Any,
) -> None:
    async def go() -> None:
        storage = build_storage(db=db, in_memory=in_memory)
        try:
            sid = await append_phase_event(
                storage, "phase.declared", name.strip().lower(), (notes or "").strip()
            )
            if sid is None:
                console.print(
                    "[yellow]No sessions found in workspace storage. Run `harness run` "
                    "at least once before declaring phases.[/yellow]"
                )
                raise typer.Exit(1)
            console.print(f"[green]declared[/green] phase {name!r}  (session {sid})")
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    run_async(go())


def phase_complete_command(
    *,
    name: str,
    notes: str | None,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    run_async: Any,
) -> None:
    async def go() -> None:
        storage = build_storage(db=db, in_memory=in_memory)
        try:
            sid = await append_phase_event(
                storage, "phase.completed", name.strip().lower(), (notes or "").strip()
            )
            if sid is None:
                console.print("[yellow]No sessions found in workspace storage.[/yellow]")
                raise typer.Exit(1)
            console.print(f"[green]completed[/green] phase {name!r}  (session {sid})")
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    run_async(go())


def phase_status_command(
    *,
    db: Path | None,
    in_memory: bool,
    console: Console,
    build_storage: Any,
    run_async: Any,
) -> None:
    async def go() -> None:
        storage = build_storage(db=db, in_memory=in_memory)
        try:
            sid = await latest_session_id(storage)
            if sid is None:
                console.print("[dim]No sessions yet.[/dim]")
                return
            events = await storage.list_activity(  # type: ignore[attr-defined]
                session_id=sid,
                kinds=("phase.declared", "phase.completed"),
                limit=200,
            )
            declared: list[str] = []
            completed: list[str] = []
            for event in events:
                pname = str((event.data or {}).get("phase", "")).strip()
                if not pname:
                    continue
                if event.kind == "phase.declared" and pname not in declared:
                    declared.append(pname)
                elif event.kind == "phase.completed" and pname not in completed:
                    completed.append(pname)
            console.print(f"session: [dim]{sid}[/dim]")
            if not declared:
                console.print("[dim]no phases declared[/dim]")
                return
            console.print(f"declared (in order): {', '.join(declared)}")
            if completed:
                console.print(f"completed: {', '.join(completed)}")
            outstanding = [phase for phase in declared if phase not in completed]
            if outstanding:
                console.print(f"[yellow]outstanding:[/yellow] {', '.join(outstanding)}")
            else:
                console.print("[green]all declared phases completed[/green]")
        finally:
            if isinstance(storage, SQLiteStorage):
                await storage.close()

    run_async(go())


def contracts_list_command(*, cwd: Path | None, console: Console) -> None:
    from harness.core import ContractRegistry

    working = (cwd or Path.cwd()).resolve()
    registry = ContractRegistry.from_paths(
        [working / ".harness" / "contracts", Path.home() / ".harness" / "contracts"]
    )
    if not registry:
        console.print("[dim]No contracts loaded.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Name", no_wrap=True)
    table.add_column("Priority", justify="right")
    table.add_column("Triggers")
    table.add_column("Rules", overflow="fold")
    table.add_column("Source", style="dim", overflow="fold")
    for contract in sorted(registry.contracts, key=lambda x: (-x.priority, x.name)):
        table.add_row(
            contract.name,
            str(contract.priority),
            ", ".join(contract.triggers) or "[dim](always)[/dim]",
            "\n".join(f"- {rule}" for rule in contract.rules),
            contract.source or "",
        )
    console.print(table)


def contracts_test_command(*, task: str, cwd: Path | None, console: Console) -> None:
    from harness.core import ContractRegistry

    working = (cwd or Path.cwd()).resolve()
    registry = ContractRegistry.from_paths(
        [working / ".harness" / "contracts", Path.home() / ".harness" / "contracts"]
    )
    rendered = registry.render(task)
    if rendered is None:
        console.print("[dim]No contracts match.[/dim]")
        return
    console.print(rendered)


def tips_list_command(*, cwd: Path | None, console: Console) -> None:
    from harness.core import TipLibrary

    working = (cwd or Path.cwd()).resolve()
    library = TipLibrary.load(
        [working / ".harness" / "tips.jsonl", Path.home() / ".harness" / "tips.jsonl"]
    )
    if not library:
        console.print("[dim]No tips loaded.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", no_wrap=True, style="dim")
    table.add_column("Weight", justify="right")
    table.add_column("Triggers")
    table.add_column("Tip", overflow="fold")
    table.add_column("Source session", style="dim", no_wrap=True)
    for tip in sorted(library.tips, key=lambda t: t.weight, reverse=True):
        table.add_row(
            tip.id,
            f"{tip.weight:.1f}",
            ", ".join(tip.triggers) or "[dim](always)[/dim]",
            tip.text,
            tip.source_session_id or "",
        )
    console.print(table)


def tips_add_command(
    *,
    text: str,
    triggers: str | None,
    weight: float,
    scope: str,
    console: Console,
) -> None:
    from harness.core import Tip, TipLibrary

    if scope not in ("repo", "user"):
        console.print("[red]--scope must be 'repo' or 'user'.[/red]")
        raise typer.Exit(2)
    target = (
        Path.cwd() / ".harness" / "tips.jsonl"
        if scope == "repo"
        else Path.home() / ".harness" / "tips.jsonl"
    )
    library = TipLibrary.load([target])
    library.path = target
    triggers_tuple = tuple(t.strip() for t in (triggers or "").split(",") if t.strip())
    tip = Tip(text=text.strip(), triggers=triggers_tuple, weight=weight)
    library.add(tip, persist=True)
    console.print(f"[green]Added tip {tip.id}[/green] to {target}")


def tips_test_command(*, task: str, top_k: int, cwd: Path | None, console: Console) -> None:
    from harness.core import TipLibrary

    working = (cwd or Path.cwd()).resolve()
    library = TipLibrary.load(
        [working / ".harness" / "tips.jsonl", Path.home() / ".harness" / "tips.jsonl"]
    )
    rendered = library.render(task, top_k=top_k)
    if rendered is None:
        console.print("[dim]No tips match.[/dim]")
        return
    console.print(rendered)


def tips_mine_command(
    *,
    session_id: str,
    model: str | None,
    provider: str | None,
    db: Path | None,
    scope: str,
    dry_run: bool,
    config_path: Path | None,
    console: Console,
    load_cli_config: Any,
    resolve_chain: Any,
    build_adapter: Any,
    run_async: Any,
) -> None:
    from harness.core import MiningInput, Tip, TipLibrary, parse_mined_tips, render_mining_prompt

    cfg = load_cli_config(config_path)
    chain = resolve_chain(failover_flag=None, provider_flag=provider, config=cfg)
    effective_model = model or cfg.default_model or "gemma2:2b"
    adapter = build_adapter(chain[0], base_url=None, config=cfg)

    target = (
        Path.cwd() / ".harness" / "tips.jsonl"
        if scope == "repo"
        else Path.home() / ".harness" / "tips.jsonl"
    )

    async def go() -> list[Tip]:
        storage = SQLiteStorage(path=db or default_db_path())
        try:
            session = await storage.get(session_id)
            if session is None:
                console.print(f"[red]Session {session_id} not found.[/red]")
                raise typer.Exit(1)
            user_msg = next((m for m in session.messages if m.role == "user"), None)
            task_text = ((user_msg.content if user_msg else None) or "").strip()
            transcript_tail = "\n".join(
                f"[{m.role}] {(m.content or '')[:400]}" for m in session.messages[-12:]
            )
            failure_summary = f"Session ended with status={session.status}."

            inp = MiningInput(
                session_id=session_id,
                task_text=task_text,
                failure_summary=failure_summary,
                transcript_excerpt=transcript_tail,
            )
            prompt = render_mining_prompt(inp)

            response_parts: list[str] = []
            from harness.core.events import Done as _Done
            from harness.core.events import TextDelta as _TextDelta
            from harness.core.schemas import Message as _Message

            async for event in adapter.stream(
                model=effective_model,
                messages=[_Message(role="user", content=prompt)],
                temperature=0.0,
                max_tokens=512,
            ):
                if isinstance(event, _TextDelta):
                    response_parts.append(event.text)
                elif isinstance(event, _Done):
                    break
            response = "".join(response_parts)
            return parse_mined_tips(response, source_session_id=session_id)
        finally:
            await storage.close()

    tips = run_async(go())
    if not tips:
        console.print("[yellow]No tips extracted.[/yellow]")
        return

    if dry_run:
        console.print(json.dumps([tip.as_dict() for tip in tips], indent=2))
        return

    library = TipLibrary.load([target])
    library.path = target
    for tip in tips:
        library.add(tip, persist=True)
    console.print(f"[green]Added {len(tips)} tip(s)[/green] to {target}")


def resume_show_command(*, cwd: Path | None, console: Console) -> None:
    from harness.core import DEFAULT_RESUME_PATH, ResumeContract

    working = (cwd or Path.cwd()).resolve()
    path = working / DEFAULT_RESUME_PATH
    contract = ResumeContract.load(path)
    if contract is None:
        console.print(f"[dim]No resume contract at {path}.[/dim]")
        console.print("Run `harness resume init` to create one.")
        return
    rendered = contract.render_for_prompt()
    if rendered:
        console.print(rendered)
    else:
        console.print(
            "[yellow]Resume contract loaded but `current` is unset.[/yellow]\n"
            f"Edit {path} to point at the feature this session should work on."
        )


def resume_init_command(
    *,
    cwd: Path | None,
    feature: str | None,
    description: str | None,
    console: Console,
) -> None:
    from harness.core import DEFAULT_RESUME_PATH, FeatureItem, ResumeContract

    working = (cwd or Path.cwd()).resolve()
    path = working / DEFAULT_RESUME_PATH
    if path.exists():
        console.print(f"[yellow]{path} already exists. Use `resume show`.[/yellow]")
        raise typer.Exit(1)
    fname = (feature or "first-feature").strip()
    contract = ResumeContract(
        current=fname,
        features=[
            FeatureItem(
                name=fname,
                description=(description or "Describe what shipping this feature means.").strip(),
                status="in_progress",
            )
        ],
    )
    contract.save(path)
    console.print(f"[green]Wrote {path}[/green]")


def resume_set_current_command(
    *,
    feature_name: str,
    cwd: Path | None,
    console: Console,
) -> None:
    from harness.core import DEFAULT_RESUME_PATH, ResumeContract

    working = (cwd or Path.cwd()).resolve()
    path = working / DEFAULT_RESUME_PATH
    contract = ResumeContract.load(path)
    if contract is None:
        console.print(f"[red]No resume contract at {path}.[/red]")
        raise typer.Exit(1)
    if contract.feature(feature_name) is None:
        console.print(
            f"[red]Feature {feature_name!r} not on the roadmap. Edit {path} to add it.[/red]"
        )
        raise typer.Exit(1)
    contract.current = feature_name
    contract.save(path)
    console.print(f"[green]Current feature is now {feature_name!r}[/green]")


def resume_add_feature_command(
    *,
    name: str,
    description: str | None,
    phases: str | None,
    cwd: Path | None,
    console: Console,
) -> None:
    from harness.core import DEFAULT_RESUME_PATH, FeatureItem, ResumeContract

    working = (cwd or Path.cwd()).resolve()
    path = working / DEFAULT_RESUME_PATH
    contract = ResumeContract.load(path) or ResumeContract()
    if contract.feature(name) is not None:
        console.print(f"[yellow]Feature {name!r} already exists. No change.[/yellow]")
        raise typer.Exit(1)
    phase_list = [p.strip().lower() for p in phases.split(",") if p.strip()] if phases else []
    contract.features.append(
        FeatureItem(
            name=name,
            description=(description or "").strip(),
            status="pending",
            phases=phase_list,
        )
    )
    if contract.current is None:
        contract.current = name
    contract.save(path)
    console.print(f"[green]Added {name!r} to {path}[/green]")
