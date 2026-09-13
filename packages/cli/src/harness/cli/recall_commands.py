"""Local workspace-scoped memory management and transcript retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from harness.cli.common import _run_async, console
from harness.cli.runtime_helpers import build_storage
from harness.core.memory import MemoryScope
from harness.core.schemas import ToolCall
from harness.core.tools_conversation_window import ConversationWindowTool
from harness.core.tools_durable_memory import ConversationSearchTool, DurableMemoryTool
from harness.storage.sqlite import SQLiteStorage

recall_app = typer.Typer(
    help="Manage workspace memories and search past conversations.", no_args_is_help=True
)


def _invoke(
    arguments: dict[str, Any],
    *,
    cwd: Path | None,
    db: Path | None,
    conversations: bool = False,
    window: bool = False,
) -> None:
    workspace = (cwd or Path.cwd()).resolve()

    async def run() -> None:
        storage = build_storage(db=db, in_memory=False, cwd=workspace)
        assert isinstance(storage, SQLiteStorage)
        scope = MemoryScope(workspace=str(workspace))
        tool = (
            ConversationWindowTool(storage, scope=scope)
            if window
            else ConversationSearchTool(storage, scope=scope)
            if conversations
            else DurableMemoryTool(storage, scope=scope)
        )
        try:
            result = await tool(ToolCall(id="cli_recall", name=tool.name, arguments=arguments))
            if result.is_error:
                console.print(result.content, markup=False)
                raise typer.Exit(1)
            console.print(
                json.dumps(json.loads(result.content), indent=2, ensure_ascii=False),
                markup=False,
                soft_wrap=True,
            )
        finally:
            await storage.close()

    _run_async(run())


@recall_app.command("search")
def search_command(
    query: str = typer.Argument(...),
    cwd: Path | None = typer.Option(None, "--cwd"),
    db: Path | None = typer.Option(None, "--db"),
    limit: int = typer.Option(10, "--limit", min=1, max=50),
) -> None:
    """Search matching transcript excerpts and session IDs in this workspace."""
    _invoke({"query": query, "limit": limit}, cwd=cwd, db=db, conversations=True)


@recall_app.command("list")
def list_command(
    cwd: Path | None = typer.Option(None, "--cwd"),
    db: Path | None = typer.Option(None, "--db"),
    limit: int = typer.Option(20, "--limit", min=1, max=50),
) -> None:
    _invoke({"action": "list", "limit": limit}, cwd=cwd, db=db)


@recall_app.command("window")
def window_command(
    session_id: str,
    reference: str | None = typer.Option(None, "--reference"),
    index: int = typer.Option(0, "--index", min=0),
    before: int = typer.Option(2, "--before", min=0, max=10),
    after: int = typer.Option(2, "--after", min=0, max=10),
    cwd: Path | None = typer.Option(None, "--cwd"),
    db: Path | None = typer.Option(None, "--db"),
) -> None:
    """Browse messages around a returned content reference or transcript index."""
    _invoke(
        {
            "session_id": session_id,
            "reference": reference,
            "index": index,
            "before": before,
            "after": after,
        },
        cwd=cwd,
        db=db,
        conversations=True,
        window=True,
    )


@recall_app.command("add")
def add_command(
    text: str = typer.Argument(...),
    kind: str = typer.Option("project_fact", "--kind"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    db: Path | None = typer.Option(None, "--db"),
) -> None:
    _invoke({"action": "add", "text": text, "kind": kind}, cwd=cwd, db=db)


@recall_app.command("get")
def get_command(
    entry_id: str = typer.Argument(...),
    cwd: Path | None = typer.Option(None, "--cwd"),
    db: Path | None = typer.Option(None, "--db"),
) -> None:
    _invoke({"action": "get", "id": entry_id}, cwd=cwd, db=db)


@recall_app.command("update")
def update_command(
    entry_id: str = typer.Argument(...),
    text: str = typer.Argument(...),
    kind: str | None = typer.Option(None, "--kind"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    db: Path | None = typer.Option(None, "--db"),
) -> None:
    _invoke({"action": "update", "id": entry_id, "text": text, "kind": kind}, cwd=cwd, db=db)


@recall_app.command("delete")
def delete_command(
    entry_id: str = typer.Argument(...),
    cwd: Path | None = typer.Option(None, "--cwd"),
    db: Path | None = typer.Option(None, "--db"),
) -> None:
    _invoke({"action": "delete", "id": entry_id}, cwd=cwd, db=db)


__all__ = ["recall_app"]
