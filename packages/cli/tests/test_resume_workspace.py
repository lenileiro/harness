"""Real CLI resumes from another directory retain their original local identity."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.cli import chat_commands
from harness.core import Capabilities, Done, Event, Message, Session, TextDelta, ToolCall
from harness.core.memory import MemoryEntry, MemoryScope
from harness.storage.sqlite import SQLiteStorage, default_db_path


@pytest.fixture
def resumed_workspace(tmp_path, monkeypatch):
    original, launch = tmp_path / "original", tmp_path / "launch"
    original.mkdir()
    launch.mkdir()
    (original / "note.txt").write_text("ORIGINAL_FILE_EVIDENCE")
    (launch / "note.txt").write_text("FOREIGN_LAUNCH_FILE")
    (launch / "AGENTS.md").write_text("FOREIGN_LAUNCH_INSTRUCTIONS")
    config = tmp_path / "config.toml"
    config.write_text("")
    monkeypatch.setenv("HARNESS_HOME", str(tmp_path / "home"))
    monkeypatch.chdir(launch)
    monkeypatch.setattr(
        chat_commands,
        "_classify_chat_turn_policy",
        lambda **kwargs: asyncio.sleep(0, result=chat_commands._GENERAL_TURN_POLICY),
    )
    calls: list[dict[str, Any]] = []
    constructions = []

    class Adapter:
        name = "ollama"

        def __init__(self, *args, **kwargs):
            constructions.append(self)

        async def capabilities(self):
            return Capabilities(tool_use=True)

        async def cancel(self, session_id):
            pass

        async def stream(self, **kwargs) -> AsyncIterator[Event]:
            calls.append(kwargs)
            if len(calls) == 1:
                yield Done(
                    final_message=Message(
                        role="assistant",
                        tool_calls=[
                            ToolCall(id="file", name="read_file", arguments={"path": "note.txt"}),
                            ToolCall(
                                id="recall",
                                name="recall_memory",
                                arguments={"action": "search", "query": "telescope"},
                            ),
                            ToolCall(
                                id="search",
                                name="search_sessions",
                                arguments={"query": "telescope"},
                            ),
                            ToolCall(
                                id="foreign",
                                name="conversation_window",
                                arguments={"session_id": "foreign-history"},
                            ),
                        ],
                    )
                )
            else:
                yield TextDelta(text="Scoped recall complete.")
                yield Done(
                    final_message=Message(role="assistant", content="Scoped recall complete.")
                )

    monkeypatch.setattr(cli_main, "OllamaAdapter", Adapter)
    return original, launch, config, calls, constructions


async def seed(original, launch, *, metadata=None):
    scope = MemoryScope(workspace=str(original))
    storage = SQLiteStorage(path=default_db_path())
    try:
        await storage.save(
            Session(
                id="existing",
                cwd=original,
                provider="ollama",
                model="test",
                messages=[Message(role="user", content="Continue this local project")],
                metadata=metadata if metadata is not None else {"memory_scope": scope.model_dump()},
            )
        )
        for label, owner in (
            ("OWN", scope),
            ("FOREIGN_LAUNCH", MemoryScope(workspace=str(launch))),
            ("FOREIGN_USER", MemoryScope(workspace=str(original), user_id="bob")),
        ):
            await storage.save_scoped_memory(
                MemoryEntry(kind="project_fact", text=f"{label} telescope memory"), scope=owner
            )
            await storage.save(
                Session(
                    id="own-history" if label == "OWN" else f"{label.lower()}-history",
                    cwd=Path(owner.workspace),
                    provider="ollama",
                    model="test",
                    metadata={"memory_scope": owner.model_dump()},
                    messages=[Message(role="user", content=f"{label} telescope history")],
                )
            )
        await storage.save(
            Session(
                id="foreign-history",
                cwd=original,
                provider="ollama",
                model="test",
                metadata={
                    "memory_scope": MemoryScope(workspace=str(original), user_id="bob").model_dump()
                },
                messages=[Message(role="user", content="FOREIGN_USER_PRIVATE_TRANSCRIPT")],
            )
        )
    finally:
        await storage.close()


def invoke(command, config, *extras):
    prompt = "Read note.txt and recall telescope facts."
    args = [command, *([prompt, "--bare"] if command == "run" else [])]
    args += ["--session", "existing", "--provider", "ollama", "--config", str(config), "--yes"]
    args += list(extras)
    return CliRunner().invoke(
        cli_main.app, args, input=f"{prompt}\n/quit\n" if command == "chat" else None
    )


@pytest.mark.parametrize("command", ["run", "chat"])
@pytest.mark.parametrize("legacy", [False, True])
def test_implicit_resume_uses_saved_workspace_and_only_its_local_recall(
    resumed_workspace, command, legacy
):
    original, launch, config, calls, _ = resumed_workspace
    asyncio.run(seed(original, launch, metadata={} if legacy else None))

    async def competing_database():
        storage = SQLiteStorage(path=original / ".harness" / "harness.db")
        try:
            await storage.save(
                Session(
                    id="existing",
                    cwd=original,
                    provider="ollama",
                    model="test",
                    messages=[Message(role="user", content="FOREIGN_DATABASE_TRANSCRIPT")],
                )
            )
        finally:
            await storage.close()

    # Resolving the workspace must not reopen a different database in it.
    asyncio.run(competing_database())
    result = invoke(command, config)
    assert result.exit_code == 0, result.output
    assert "Scoped recall complete." in result.output
    assert len(calls) == 2
    messages = calls[-1]["messages"]
    content = "\n".join(message.content or "" for message in messages)
    assert "ORIGINAL_FILE_EVIDENCE" in content
    assert "OWN telescope memory" in content and "OWN telescope history" in content
    assert "FOREIGN_" not in content
    foreign = next(message for message in messages if message.tool_call_id == "foreign")
    assert "not found" in foreign.content.lower()

    async def saved():
        storage = SQLiteStorage(path=default_db_path())
        try:
            return await storage.get("existing")
        finally:
            await storage.close()

    session = asyncio.run(saved())
    assert session is not None and session.cwd == original
    assert session.metadata["memory_scope"] == MemoryScope(workspace=str(original)).model_dump()


@pytest.mark.parametrize("command", ["run", "chat"])
def test_matching_explicit_workspace_and_selected_database_are_allowed(resumed_workspace, command):
    original, launch, config, calls, _ = resumed_workspace
    asyncio.run(seed(original, launch))
    result = invoke(command, config, "--cwd", str(original), "--db", str(default_db_path()))
    assert result.exit_code == 0, result.output
    assert "Scoped recall complete." in result.output
    assert "ORIGINAL_FILE_EVIDENCE" in "\n".join(
        message.content or "" for message in calls[-1]["messages"]
    )


@pytest.mark.parametrize("command", ["run", "chat"])
def test_unused_session_id_still_creates_session_in_launch_workspace(resumed_workspace, command):
    _, launch, config, calls, _ = resumed_workspace
    result = invoke(command, config)
    assert result.exit_code == 0, result.output
    assert "FOREIGN_LAUNCH_FILE" in "\n".join(
        message.content or "" for message in calls[-1]["messages"]
    )

    async def saved():
        storage = SQLiteStorage(path=default_db_path())
        try:
            return await storage.get("existing")
        finally:
            await storage.close()

    session = asyncio.run(saved())
    assert session is not None and session.cwd == launch
    assert session.metadata["memory_scope"] == MemoryScope(workspace=str(launch)).model_dump()


@pytest.mark.parametrize("command", ["run", "chat"])
@pytest.mark.parametrize(
    "problem", ["override", "foreign", "scope_conflict", "malformed", "missing"]
)
def test_invalid_resume_stops_before_adapter_tools_or_session_mutation(
    resumed_workspace, monkeypatch, command, problem
):
    original, launch, config, calls, constructions = resumed_workspace
    metadata = None
    if problem == "foreign":
        metadata = {
            "memory_scope": MemoryScope(workspace=str(original), user_id="bob").model_dump()
        }
    elif problem == "scope_conflict":
        metadata = {"memory_scope": MemoryScope(workspace=str(launch)).model_dump()}
    elif problem == "malformed":
        metadata = {"memory_scope": {"workspace": str(original), "unknown_identity": "bob"}}
    asyncio.run(seed(original, launch, metadata=metadata))
    if problem == "missing":
        original.rename(original.with_name("moved-original"))

    def forbidden_builder(*args, **kwargs):
        raise AssertionError("Invalid resume must stop before building tools")

    monkeypatch.setattr(cli_main, "_build_tools", forbidden_builder)
    extra = ("--cwd", str(launch)) if problem == "override" else ()
    result = invoke(command, config, *extra)
    assert result.exit_code == 2, result.output
    assert calls == [] and constructions == []
    expected = {
        "override": "conflicts with the saved session workspace",
        "foreign": "not a local CLI session",
        "scope_conflict": "not a local CLI session",
        "malformed": "not a local CLI session",
        "missing": "Saved session workspace is unavailable",
    }
    assert expected[problem] in result.output

    async def saved():
        storage = SQLiteStorage(path=default_db_path())
        try:
            return await storage.get("existing")
        finally:
            await storage.close()

    session = asyncio.run(saved())
    assert session is not None and len(session.messages) == 1
    assert session.cwd == original
