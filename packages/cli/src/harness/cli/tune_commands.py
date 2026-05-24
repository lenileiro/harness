from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table


def tune_list_command(*, cwd: Path | None, console: Console) -> None:
    """List versioned tunable prompts under `.harness/tuned-prompts/`."""
    from harness.core.verifier_tuner import DEFAULT_TUNED_DIR, TunablePrompt

    working = (cwd or Path.cwd()).resolve()
    target_dir = working / DEFAULT_TUNED_DIR
    if not target_dir.is_dir():
        console.print(f"[dim]No tuned prompts at {target_dir}.[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Key", no_wrap=True)
    table.add_column("Current version", justify="right")
    table.add_column("Rationale", overflow="fold")
    for entry in sorted(target_dir.glob("*.json")):
        prompt = TunablePrompt.load(entry)
        if prompt is None or not prompt.versions:
            continue
        current = prompt.current
        assert current is not None
        table.add_row(prompt.key, str(current.version), current.rationale or "[dim](none)[/dim]")
    console.print(table)


def tune_show_command(
    *,
    key: str,
    version: int | None,
    cwd: Path | None,
    console: Console,
) -> None:
    """Print one version (default: current) of a tunable prompt."""
    from harness.core.verifier_tuner import DEFAULT_TUNED_DIR, TunablePrompt

    working = (cwd or Path.cwd()).resolve()
    path = working / DEFAULT_TUNED_DIR / f"{key}.json"
    prompt = TunablePrompt.load(path)
    if prompt is None or not prompt.versions:
        console.print(f"[red]No versions for {key!r} at {path}.[/red]")
        raise typer.Exit(1)
    if version is None:
        selected = prompt.current
    else:
        selected = next((item for item in prompt.versions if item.version == version), None)
    if selected is None:
        console.print(f"[red]Version {version} not found for {key!r}.[/red]")
        raise typer.Exit(1)
    console.print(f"[bold]{key} v{selected.version}[/bold]")
    if selected.rationale:
        console.print(f"[dim]{selected.rationale}[/dim]")
    console.print()
    console.print(selected.text)


def tune_propose_command(
    *,
    key: str,
    current_prompt_file: Path,
    pairs_file: Path,
    model: str | None,
    provider: str | None,
    notes: str | None,
    config_path: Path | None,
    dry_run: bool,
    console: Console,
    load_cli_config: Any,
    resolve_chain: Any,
    build_adapter: Any,
    run_async: Any,
) -> None:
    """Ask the configured LLM for a prompt-delta proposal."""
    from harness.core.events import Done as _Done
    from harness.core.events import TextDelta as _TextDelta
    from harness.core.schemas import Message as _Message
    from harness.core.verifier_tuner import (
        DEFAULT_TUNED_DIR,
        TUNER_SYSTEM,
        TrajectoryPair,
        TunablePrompt,
        TuneRequest,
        parse_proposal,
        render_tune_prompt,
    )

    cfg = load_cli_config(config_path)
    chain = resolve_chain(failover_flag=None, provider_flag=provider, config=cfg)
    effective_model = model or cfg.default_model or "gemma2:2b"
    adapter = build_adapter(chain[0], base_url=None, config=cfg)

    current_text = current_prompt_file.read_text(encoding="utf-8").strip()
    raw_pairs = json.loads(pairs_file.read_text(encoding="utf-8"))
    pairs = [
        TrajectoryPair(
            fixture=str(item.get("fixture", "")),
            defended_excerpt=str(item.get("defended_excerpt", "")),
            defended_outcome=str(item.get("defended_outcome", "")),
            bare_excerpt=str(item.get("bare_excerpt", "")),
            bare_outcome=str(item.get("bare_outcome", "")),
            differing_dimension=item.get("differing_dimension"),
        )
        for item in raw_pairs
        if isinstance(item, dict)
    ]

    request = TuneRequest(
        prompt_key=key,
        current_prompt=current_text,
        pairs=pairs,
        notes=notes or "",
    )
    user_msg = render_tune_prompt(request)

    async def go() -> str:
        chunks: list[str] = []
        async for event in adapter.stream(
            model=effective_model,
            messages=[
                _Message(role="system", content=TUNER_SYSTEM),
                _Message(role="user", content=user_msg),
            ],
            temperature=0.0,
            max_tokens=1500,
        ):
            if isinstance(event, _TextDelta):
                chunks.append(event.text)
            elif isinstance(event, _Done):
                break
        return "".join(chunks)

    response = run_async(go())
    delta = parse_proposal(response, prompt_key=key)
    if delta is None:
        console.print("[red]Tuner LLM returned no parseable proposal.[/red]")
        console.print("[dim]Raw response:[/dim]")
        console.print(response[:1000])
        raise typer.Exit(1)

    console.print(f"[bold]Proposed delta for {key!r}[/bold]")
    console.print()
    console.print(f"[dim]Rationale: {delta.rationale}[/dim]")
    console.print()
    console.print(delta.new_prompt)

    if dry_run:
        console.print("\n[dim]--dry-run: not saving.[/dim]")
        return

    target = Path.cwd().resolve() / DEFAULT_TUNED_DIR / f"{key}.json"
    prompt = TunablePrompt.load(target) or TunablePrompt(key=key)
    if not prompt.versions:
        prompt.add_version(current_text, rationale="seed: pre-tune baseline")
    prompt.add_version(delta.new_prompt, rationale=delta.rationale)
    prompt.save(target)
    saved = prompt.current
    assert saved is not None
    console.print(f"\n[green]Saved v{saved.version} to {target}[/green]")


def tune_rollback_command(*, key: str, cwd: Path | None, console: Console) -> None:
    """Drop the latest version, restoring the previous one as current."""
    from harness.core.verifier_tuner import DEFAULT_TUNED_DIR, TunablePrompt

    working = (cwd or Path.cwd()).resolve()
    path = working / DEFAULT_TUNED_DIR / f"{key}.json"
    prompt = TunablePrompt.load(path)
    if prompt is None or len(prompt.versions) <= 1:
        console.print(f"[red]Nothing to roll back at {path}.[/red]")
        raise typer.Exit(1)
    dropped = prompt.versions.pop()
    prompt.save(path)
    remaining = prompt.current
    assert remaining is not None
    console.print(
        f"[yellow]Rolled back v{dropped.version}. Current is now v{remaining.version}.[/yellow]"
    )
