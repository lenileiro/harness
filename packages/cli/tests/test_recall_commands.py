"""Scoped recall commands with real persistent storage and no model calls."""

import asyncio
import json

from typer.testing import CliRunner

from harness.cli import __main__ as cli
from harness.cli.recall_commands import recall_app
from harness.core.memory import MemoryEntry, MemoryScope
from harness.core.schemas import Message, Session
from harness.storage.sqlite import SQLiteStorage


def test_recall_memory_crud_is_scoped_and_persistent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    db = tmp_path / "shared.db"
    options = ["--db", str(db), "--cwd", str(tmp_path / "one")]
    result = runner.invoke(recall_app, ["add", "persistent preference", *options])
    assert result.exit_code == 0, result.output
    entry_id = json.loads(result.output)["id"]
    result = runner.invoke(recall_app, ["get", entry_id, *options])
    assert json.loads(result.output)["text"] == "persistent preference"
    result = runner.invoke(recall_app, ["update", entry_id, "corrected preference", *options])
    assert result.exit_code == 0, result.output
    result = runner.invoke(recall_app, ["list", *options])
    assert json.loads(result.output)["memories"][0]["text"] == "corrected preference"
    result = runner.invoke(
        recall_app, ["get", entry_id, "--db", str(db), "--cwd", str(tmp_path / "two")]
    )
    assert result.exit_code == 1
    assert "corrected preference" not in result.output
    result = runner.invoke(recall_app, ["delete", entry_id, *options])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["deleted"]
    assert not json.loads(runner.invoke(recall_app, ["list", *options]).output)["memories"]


def test_recall_and_existing_memory_commands_cannot_read_other_users(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = tmp_path / "shared.db"
    local = MemoryScope(workspace=str(tmp_path))
    private = MemoryScope(workspace=str(tmp_path), user_id="alice")

    async def seed():
        store = SQLiteStorage(path=db)
        try:
            for scope, entry_id, text in (
                (local, "local", "nebula local"),
                (private, "private", "nebula secret"),
            ):
                await store.save_scoped_memory(
                    MemoryEntry(id=entry_id, kind="project_fact", text=text), scope=scope
                )
                await store.save(
                    Session(
                        id=entry_id,
                        provider="mock",
                        model="mock",
                        cwd=tmp_path,
                        metadata={"memory_scope": scope.model_dump()},
                        messages=[Message(role="user", content=text)],
                    )
                )
        finally:
            await store.close()

    asyncio.run(seed())
    runner = CliRunner()
    result = runner.invoke(recall_app, ["search", "nebula", "--db", str(db)])
    assert result.exit_code == 0, result.output
    assert [item["session_id"] for item in json.loads(result.output)["sessions"]] == ["local"]
    for command in (["memory", "list"], ["memory", "search", "nebula"]):
        result = runner.invoke(cli.app, [*command, "--db", str(db)])
        assert result.exit_code == 0, result.output
        assert "nebula local" in result.output
        assert "nebula secret" not in result.output
    result = runner.invoke(cli.app, ["memory", "rm", "private", "--db", str(db), "--yes"])
    assert result.exit_code == 1
    assert "nebula secret" not in result.output
