"""Tests for fork_session logic."""

from __future__ import annotations

import pytest

from harness.core.errors import ConfigurationError
from harness.core.runtime import fork_session
from harness.core.schemas import Message, Note, PhaseStatus, Session
from harness.storage.memory import InMemoryStorage


@pytest.fixture
def store() -> InMemoryStorage:
    return InMemoryStorage()


async def _save_session(store: InMemoryStorage, **kwargs) -> Session:
    defaults = {
        "provider": "ollama",
        "model": "llama3.2",
        "cwd": "/tmp",
        "messages": [Message(role="user", content="hello")],
        "approval_overrides": {"shell": "auto"},
        "task_id": "task_123",
    }
    defaults.update(kwargs)
    session = Session(**defaults)  # type: ignore[arg-type]
    await store.save(session)
    return session


@pytest.mark.asyncio
async def test_fork_creates_new_session(store: InMemoryStorage) -> None:
    parent = await _save_session(store)
    forked = await fork_session(store, parent.id)

    assert forked.id != parent.id
    assert forked.id.startswith("sess_")


@pytest.mark.asyncio
async def test_fork_copies_messages(store: InMemoryStorage) -> None:
    parent = await _save_session(store)
    forked = await fork_session(store, parent.id)

    assert len(forked.messages) == len(parent.messages)
    assert forked.messages[0].content == parent.messages[0].content


@pytest.mark.asyncio
async def test_fork_message_isolation(store: InMemoryStorage) -> None:
    parent = await _save_session(store)
    forked = await fork_session(store, parent.id)

    # Mutating the fork's messages should not affect the parent.
    forked.messages.append(Message(role="user", content="extra"))
    reloaded_parent = await store.get(parent.id)
    assert reloaded_parent is not None
    assert len(reloaded_parent.messages) == 1


@pytest.mark.asyncio
async def test_fork_sets_forked_from(store: InMemoryStorage) -> None:
    parent = await _save_session(store)
    forked = await fork_session(store, parent.id)

    assert forked.forked_from == parent.id


@pytest.mark.asyncio
async def test_fork_preserves_task_id(store: InMemoryStorage) -> None:
    parent = await _save_session(store, task_id="task_abc")
    forked = await fork_session(store, parent.id)

    assert forked.task_id == "task_abc"


@pytest.mark.asyncio
async def test_fork_preserves_approval_overrides(store: InMemoryStorage) -> None:
    parent = await _save_session(store, approval_overrides={"shell": "auto"})
    forked = await fork_session(store, parent.id)

    assert forked.approval_overrides == {"shell": "auto"}


@pytest.mark.asyncio
async def test_fork_resets_status(store: InMemoryStorage) -> None:
    parent = await _save_session(store)
    parent.status = "done"
    await store.save(parent)

    forked = await fork_session(store, parent.id)
    assert forked.status == "pending"


@pytest.mark.asyncio
async def test_fork_explicit_id(store: InMemoryStorage) -> None:
    parent = await _save_session(store)
    forked = await fork_session(store, parent.id, new_session_id="sess_custom")

    assert forked.id == "sess_custom"


@pytest.mark.asyncio
async def test_fork_persists(store: InMemoryStorage) -> None:
    parent = await _save_session(store)
    forked = await fork_session(store, parent.id)

    loaded = await store.get(forked.id)
    assert loaded is not None
    assert loaded.forked_from == parent.id


@pytest.mark.asyncio
async def test_fork_not_found(store: InMemoryStorage) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        await fork_session(store, "sess_doesnotexist")


@pytest.mark.asyncio
async def test_fork_preserves_continuity_without_sharing_mutable_state(
    store: InMemoryStorage,
) -> None:
    parent = await _save_session(
        store,
        notes=[Note(text="keep this finding", tags=["research"])],
        phases=[PhaseStatus(name="implement", notes=["planned"])],
        metadata={"custom": {"history": ["first"]}, "verification": {"passed": True}},
    )
    forked = await fork_session(store, parent.id)
    assert forked.notes == parent.notes
    assert forked.phases == parent.phases
    assert forked.metadata["custom"] == parent.metadata["custom"]
    assert "verification" not in forked.metadata
    forked.notes[0].tags.append("fork")
    forked.phases[0].notes.append("fork")
    forked.metadata["custom"]["history"].append("fork")
    assert parent.notes[0].tags == ["research"]
    assert parent.phases[0].notes == ["planned"]
    assert parent.metadata["custom"]["history"] == ["first"]
