"""Real backend tests for constructor-bound memory and transcript access."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.core.memory import MemoryEntry, MemoryScope, ScopedMemoryStore
from harness.core.schemas import Message, Session, ToolCall
from harness.core.tools_durable_memory import (
    ConversationSearchTool,
    DurableMemoryTool,
    RecallMemoryTool,
)
from harness.storage.memory import InMemoryStorage
from harness.storage.sqlite import SQLiteStorage


@pytest.fixture(params=["memory", "sqlite"])
async def store(request, tmp_path):
    storage = (
        InMemoryStorage()
        if request.param == "memory"
        else SQLiteStorage(path=tmp_path / "sessions.db")
    )
    yield storage
    if isinstance(storage, SQLiteStorage):
        await storage.close()


async def invoke(tool, **arguments):
    return await tool(ToolCall(id="call", name=tool.name, arguments=arguments))


async def test_read_only_recall_rejects_mutation_and_foreign_ids(store, tmp_path):
    scope = MemoryScope(workspace=str(tmp_path), user_id="alice")
    own = await store.save_scoped_memory(
        MemoryEntry(kind="project_fact", text="owned telescope fact"), scope=scope
    )
    foreign = await store.save_scoped_memory(
        MemoryEntry(kind="project_fact", text="foreign telescope fact"),
        scope=scope.model_copy(update={"user_id": "bob"}),
    )
    tool = RecallMemoryTool(store, scope=scope)
    assert tool.effect_scope == "read_only"
    for action in ("list", "search"):
        result = await invoke(tool, action=action, query="telescope")
        assert not result.is_error
        assert [item["id"] for item in json.loads(result.content)["memories"]] == [own.id]
    assert json.loads((await invoke(tool, action="get", id=own.id)).content)["text"] == own.text
    assert (await invoke(tool, action="get", id=foreign.id)).is_error
    for arguments in (
        {"action": "add", "text": "overwrite"},
        {"action": "update", "id": own.id, "text": "overwrite"},
        {"action": "delete", "id": own.id},
        {"action": "list", "scope": {"user_id": "bob"}},
        {"action": "list", "user_id": "bob"},
        {"action": "get"},
        {"action": "search", "query": " "},
    ):
        assert (await invoke(tool, **arguments)).is_error
    assert (await store.get_scoped_memory(own.id, scope=scope)) == own


async def test_scoped_memory_crud_roundtrip_and_returned_values_are_isolated(store, tmp_path):
    scope = MemoryScope(workspace=str(tmp_path))
    assert isinstance(store, ScopedMemoryStore)
    tool = DurableMemoryTool(store, scope=scope)
    added = await invoke(
        tool, action="add", text="Prefers concise responses", kind="user_preference"
    )
    assert not added.is_error
    entry_id = json.loads(added.content)["id"]
    result = await invoke(tool, action="get", id=entry_id)
    assert json.loads(result.content)["text"] == "Prefers concise responses"
    result = await invoke(tool, action="search", query="concise")
    assert [item["id"] for item in json.loads(result.content)["memories"]] == [entry_id]
    result = await invoke(tool, action="update", id=entry_id, text="Prefers detailed responses")
    assert json.loads(result.content)["kind"] == "user_preference"
    listed = await store.list_scoped_memory(scope=scope)
    listed[0].text = "do not mutate storage"
    loaded = await store.get_scoped_memory(entry_id, scope=scope)
    assert loaded is not None
    assert loaded.text == "Prefers detailed responses"
    assert not (await invoke(tool, action="delete", id=entry_id)).is_error
    assert (await invoke(tool, action="get", id=entry_id)).is_error


async def test_memory_cannot_cross_workspace_or_user_even_with_known_id(store, tmp_path):
    owner = MemoryScope(workspace=str(tmp_path / "one"), user_id="alice")
    entry = await store.save_scoped_memory(
        MemoryEntry(kind="user_fact", text="private secret"), scope=owner
    )
    for scope in (
        MemoryScope(workspace=str(tmp_path / "one"), user_id="bob"),
        MemoryScope(workspace=str(tmp_path / "one")),
        MemoryScope(workspace=str(tmp_path / "two"), user_id="alice"),
    ):
        tool = DurableMemoryTool(store, scope=scope)
        assert not json.loads((await invoke(tool, action="list")).content)["memories"]
        assert not json.loads((await invoke(tool, action="search", query="secret")).content)[
            "memories"
        ]
        for action in ("get", "update", "delete"):
            result = await invoke(tool, action=action, id=entry.id, text="overwrite")
            assert result.is_error
            assert "private secret" not in result.content
        with pytest.raises(KeyError):
            await store.save_scoped_memory(entry.model_copy(update={"text": "stolen"}), scope=scope)
        assert not await store.delete_scoped_memory(entry.id, scope=scope)
    loaded = await store.get_scoped_memory(entry.id, scope=owner)
    assert loaded is not None
    assert loaded.text == "private secret"


async def test_model_arguments_cannot_override_scope_and_search_is_literal(store, tmp_path):
    scope = MemoryScope(workspace=str(tmp_path))
    tool = DurableMemoryTool(store, scope=scope)
    for arguments in (
        {"action": "add", "text": "bad", "scope": {"workspace": "/"}},
        {"action": "list", "user_id": "victim"},
        {"action": "list", "limit": -1},
    ):
        assert (await invoke(tool, **arguments)).is_error
    await invoke(tool, action="add", text="100% complete")
    await invoke(tool, action="add", text="unrelated")
    result = await invoke(tool, action="search", query="%")
    assert [entry["text"] for entry in json.loads(result.content)["memories"]] == ["100% complete"]


async def test_transcript_search_is_scoped_updated_and_deleted(store, tmp_path):
    workspace = str(tmp_path)
    owner = MemoryScope(workspace=workspace, user_id="alice")
    scopes = [
        owner,
        MemoryScope(workspace=workspace, user_id="bob"),
        MemoryScope(workspace=workspace),
        MemoryScope(workspace=str(tmp_path / "other"), user_id="alice"),
    ]
    for index, scope in enumerate(scopes):
        await store.save(
            Session(
                id=f"session-{index}",
                provider="mock",
                model="mock",
                cwd=Path(scope.workspace),
                metadata={"memory_scope": scope.model_dump()},
                messages=[Message(role="user", content=f"nebula telescope secret-{index}")],
            )
        )
    tool = ConversationSearchTool(store, scope=owner)
    result = await invoke(tool, query="telescope nebula")
    matches = json.loads(result.content)["sessions"]
    assert [match["session_id"] for match in matches] == ["session-0"]
    assert "secret-0" in matches[0]["excerpt"]
    assert (await invoke(tool, query="nebula", user_id="bob")).is_error
    assert not json.loads((await invoke(tool, query='" OR 1=1 --')).content)["sessions"]
    saved = await store.get("session-0")
    assert saved is not None
    saved.messages = [Message(role="assistant", content="different result")]
    await store.save(saved)
    assert not await store.search_sessions("nebula", scope=owner)
    assert len(await store.search_sessions("different", scope=owner)) == 1
    await store.delete("session-0")
    assert not await store.search_sessions("different", scope=owner)


async def test_legacy_conversations_are_local_and_malformed_identity_is_not_searchable(
    store, tmp_path
):
    for session_id, metadata in (
        ("local", {}),
        ("malformed", {"memory_scope": {"user_id": "alice"}}),
    ):
        await store.save(
            Session(
                id=session_id,
                provider="mock",
                model="mock",
                cwd=tmp_path,
                metadata=metadata,
                messages=[Message(role="user", content="starship")],
            )
        )
    local = MemoryScope(workspace=str(tmp_path))
    assert [item.session_id for item in await store.search_sessions("starship", scope=local)] == [
        "local"
    ]
    assert not await store.search_sessions(
        "starship", scope=MemoryScope(workspace=str(tmp_path), user_id="alice")
    )


async def test_unscoped_memory_is_not_exposed_without_ownership_evidence(store, tmp_path):
    await store.save_memory(MemoryEntry(kind="project_fact", text="unowned legacy record"))
    assert not await store.list_scoped_memory(scope=MemoryScope(workspace=str(tmp_path)))
    assert len(await store.list_memory()) == 1
