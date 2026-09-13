"""Explicit, noninteractive local question inspection and answers."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from harness.cli.common import console
from harness.cli.runtime_helpers import workspace_db
from harness.core.clarification import QuestionStore, parse_question_answer, render_question
from harness.core.memory import MemoryScope
from harness.storage.sqlite import default_db_path

clarify_app = typer.Typer(
    help="Inspect and answer pending clarification questions.", no_args_is_help=True
)


def local_question_store(cwd: Path | None, db: Path | None) -> tuple[QuestionStore, MemoryScope]:
    workspace = (cwd or Path.cwd()).resolve()
    return QuestionStore(db or workspace_db(workspace) or default_db_path()), MemoryScope(
        workspace=str(workspace)
    )


@clarify_app.command("list")
def list_questions(
    cwd: Path | None = typer.Option(None, "--cwd"),
    db: Path | None = typer.Option(None, "--db"),
    session_id: str | None = typer.Option(None, "--session"),
) -> None:
    store, scope = local_question_store(cwd, db)
    try:
        rows = store.list_pending(scope=scope, session_id=session_id)
        console.print(
            json.dumps([row.model_dump(mode="json") for row in rows], indent=2, ensure_ascii=False),
            markup=False,
        )
    finally:
        store.close()


@clarify_app.command("show")
def show_question(
    question_id: str,
    cwd: Path | None = typer.Option(None, "--cwd"),
    db: Path | None = typer.Option(None, "--db"),
) -> None:
    store, scope = local_question_store(cwd, db)
    try:
        record = store.get(question_id, scope=scope)
        if record is None:
            raise typer.BadParameter("question not found in the local workspace scope")
        console.print(render_question(record), markup=False)
    finally:
        store.close()


@clarify_app.command("answer")
def answer_question(
    question_id: str,
    answer: str,
    session_id: str = typer.Option(..., "--session"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    db: Path | None = typer.Option(None, "--db"),
) -> None:
    store, scope = local_question_store(cwd, db)
    try:
        record = store.get(question_id, scope=scope, session_id=session_id)
        if record is None:
            raise ValueError("question not found in this local session")
        record = store.answer(
            question_id,
            scope=scope,
            session_id=session_id,
            answers=parse_question_answer(record, answer),
        )
        console.print(
            render_question(record)
            if record.status == "pending"
            else f"Answers recorded. Resume session {session_id} with `harness sessions resume` and the same --db/--cwd.",
            markup=False,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    finally:
        store.close()


@clarify_app.command("cancel")
def cancel_question(
    question_id: str,
    session_id: str = typer.Option(..., "--session"),
    cwd: Path | None = typer.Option(None, "--cwd"),
    db: Path | None = typer.Option(None, "--db"),
) -> None:
    store, scope = local_question_store(cwd, db)
    try:
        store.cancel(question_id, scope=scope, session_id=session_id)
        console.print(
            f"Question cancelled. Resume session {session_id} to continue with a cancelled question result.",
            markup=False,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    finally:
        store.close()


__all__ = ["clarify_app"]
