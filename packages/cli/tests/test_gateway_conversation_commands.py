import json
from dataclasses import replace
from typing import Any

import pytest

from harness.cli import gateway_conversation_commands as commands
from harness.cli import gateway_runtime
from harness.cli.config import HarnessConfig
from harness.core import Capabilities, Done, Message, PendingApproval, Session
from harness.core.activity import ActivityEvent
from harness.core.gateway_models import GatewayRuntimeBinding, default_gateway_root
from harness.core.gateway_sessions import GatewaySessionStore
from harness.storage.sqlite import SQLiteStorage


async def make_session(tmp_path, *, provider="fake", user="owner", runtime_id="runtime"):
    sessions = GatewaySessionStore(root=default_gateway_root(tmp_path))
    gateway = sessions.get_or_create_session(transport="telegram", user_id=user, thread_id="thread")
    gateway = replace(
        gateway,
        metadata={
            "harness_session_id": runtime_id,
            "thread_summary": "last turn",
            "thread_context": ["last turn"],
        },
    )
    sessions.save_session(gateway)
    binding = GatewayRuntimeBinding(
        session_id=runtime_id,
        gateway_session_id=gateway.id,
        transport="telegram",
        user_id=user,
        thread_id="thread",
        provider=provider,
        model="original",
        max_steps=7,
    )
    sessions.bind_runtime_session(binding)
    session = Session(
        id=runtime_id,
        cwd=tmp_path,
        provider=provider,
        model="original",
        status="done",
        metadata={
            "memory_scope": {
                "workspace": str(tmp_path),
                "user_id": json.dumps(["telegram", user], separators=(",", ":")),
            }
        },
        messages=[
            Message(role="user", content="first question"),
            Message(role="assistant", content="first answer"),
            Message(role="user", content="latest private question"),
            Message(role="assistant", content="latest private answer"),
        ],
    )
    store = SQLiteStorage(path=tmp_path / ".harness/harness.db")
    await store.save(session)
    await store.close()
    return sessions, gateway, binding, session


async def receive(tmp_path, message, user="owner") -> dict[str, Any]:
    return await gateway_runtime._run_gateway_receive_payload(
        working_dir=tmp_path,
        message=message,
        transport="telegram",
        user_id=user,
        thread_id="thread",
    )


async def _receive_handoff(**kwargs: Any) -> dict[str, Any]:
    return await gateway_runtime._run_gateway_receive_payload(**kwargs)


async def test_explicit_handoff_is_destination_bound_recoverable_and_preserves_ownership(
    tmp_path, monkeypatch
):
    sessions, source, _, old = await make_session(tmp_path)
    target: dict[str, Any] = {"transport": "slack", "user_id": "T:U", "thread_id": "T:D:"}
    issued = await receive(tmp_path, "/handoff " + json.dumps(target))
    assert issued["reply"]["status"] == "ok", issued
    code = issued["reply"]["data"]["code"]
    from harness.core.gateway_handoffs import HandoffStore

    handoffs = HandoffStore(sessions.root)
    assert code not in (sessions.root / "handoffs.sqlite3").read_bytes().decode(errors="ignore")
    # A crashed claimant retains its fixed import ID and can finish on restart.
    reserved = handoffs.claim(code, target)["session_id"]
    handoffs.close()
    wrong = await _receive_handoff(
        working_dir=tmp_path,
        message="/continue " + code,
        transport="slack",
        user_id="other",
        thread_id="T:D:",
    )
    assert wrong["reply"]["status"] == "invalid" and "latest private" not in json.dumps(wrong)
    imported = await _receive_handoff(working_dir=tmp_path, message="/continue " + code, **target)
    assert imported["reply"]["status"] == "ok", imported
    assert imported["reply"]["data"]["harness_session_id"] == reserved
    database = SQLiteStorage(path=tmp_path / ".harness/harness.db")
    try:
        copied = await database.get(reserved)
        assert (
            copied
            and copied.messages == old.messages
            and copied.metadata["memory_scope"]["user_id"] == '["slack","T:U"]'
        )
        assert copied.notes == [] and copied.phases == [] and copied.approval_overrides == {}
        assert sessions.load_session(source.id).metadata["harness_session_id"] == old.id
        bound = sessions.load_runtime_binding(reserved)
        assert bound and bound.belongs_to(**target) and not bound.local_only
        copied.messages.append(Message(role="user", content="destination follow-up"))
        await database.save(copied)
        again = await _receive_handoff(working_dir=tmp_path, message="/continue " + code, **target)
        assert "already imported" in again["reply"]["text"]
        restored = await database.get(reserved)
        assert restored and restored.messages[-1].content == "destination follow-up"
        destination = sessions.get_or_create_session(**target)
        assert destination.metadata["provider_override"] == "fake"
        key = gateway_runtime._runtime_session_key(provider="fake", model="original")
        assert destination.metadata["harness_session_id_" + key] == reserved
    finally:
        await database.close()


async def test_handoff_expiry_revocation_and_source_approval_gate(tmp_path):
    sessions, _, _, old = await make_session(tmp_path)
    target: dict[str, Any] = {"transport": "slack", "user_id": "T:U", "thread_id": "T:D:"}
    issued = await receive(tmp_path, "/handoff " + json.dumps(target))
    code = issued["reply"]["data"]["code"]
    revoked = await receive(tmp_path, "/handoff revoke " + code)
    assert revoked["reply"]["text"] == "Handoff revoked."
    result = await _receive_handoff(working_dir=tmp_path, message="/continue " + code, **target)
    assert result["reply"]["status"] == "invalid"
    issued = await receive(tmp_path, "/handoff " + json.dumps(target))
    from harness.core.gateway_handoffs import HandoffStore

    handoffs = HandoffStore(sessions.root)
    with handoffs.db:
        handoffs.db.execute("UPDATE handoffs SET expires=0")
    handoffs.close()
    result = await _receive_handoff(
        working_dir=tmp_path, message="/continue " + issued["reply"]["data"]["code"], **target
    )
    assert result["reply"]["status"] == "invalid"
    database = SQLiteStorage(path=tmp_path / ".harness/harness.db")
    old.status = "paused"
    await database.save(old)
    await database.close()
    refused = await receive(tmp_path, "/handoff " + json.dumps(target))
    assert refused["reply"]["status"] == "busy"


@pytest.mark.parametrize("status", ["paused", "done"])
@pytest.mark.parametrize(
    "command", ["/model replacement", "/retry", "/undo", "/compress", "/handoff {}"]
)
async def test_question_pointer_prevents_conversation_copy_or_mutation(tmp_path, status, command):
    sessions, gateway, binding, original = await make_session(tmp_path)
    original.status = status
    original.metadata["pending_question_id"] = "owned-question"
    database = SQLiteStorage(path=tmp_path / ".harness/harness.db")
    try:
        await database.save(original)
        result = await receive(tmp_path, command)
        assert result["reply"]["status"] == "question_required"
        saved = await database.get(original.id)
        assert saved is not None and saved.messages == original.messages
        assert saved.metadata == original.metadata
        assert (
            sessions.load_session(gateway.id).metadata["harness_session_id"] == binding.session_id
        )
    finally:
        await database.close()


@pytest.mark.parametrize("command", ["/model", "/retry", "/undo", "/compress", "/usage"])
async def test_commands_never_follow_another_users_runtime_binding(tmp_path, command):
    sessions, owner, _, _ = await make_session(tmp_path)
    other = sessions.get_or_create_session(
        transport="telegram", user_id="other", thread_id="thread"
    )
    sessions.save_session(replace(other, metadata={"harness_session_id": "runtime"}))
    result = await receive(tmp_path, command, user="other")
    assert result["reply"]["status"] == "forbidden"
    assert "latest private" not in json.dumps(result)
    unchanged = sessions.load_session(owner.id)
    assert unchanged.metadata["harness_session_id"] == "runtime"


async def test_model_switch_preserves_context_and_immutable_binding_on_restart(
    tmp_path, monkeypatch
):
    sessions, gateway, binding, old = await make_session(tmp_path)
    monkeypatch.setattr(commands, "_load_cli_config", lambda _: HarnessConfig())
    shown = await receive(tmp_path, "/model")
    assert shown["reply"]["data"] == {"provider": "fake", "model": "original"}
    selected = await receive(tmp_path, "/model replacement")
    assert selected["reply"]["status"] == "ok"
    fresh = GatewaySessionStore(root=default_gateway_root(tmp_path)).load_session(gateway.id)
    new_id = fresh.metadata["harness_session_id"]
    assert new_id != old.id and fresh.metadata["model_override"] == "replacement"
    store = SQLiteStorage(path=tmp_path / ".harness/harness.db")
    try:
        copied = await store.get(new_id)
        assert copied is not None and copied.messages == old.messages
        assert copied.model == "replacement" and copied.forked_from == old.id
        assert sessions.load_runtime_binding(old.id) == binding
        new_binding = sessions.load_runtime_binding(new_id)
        assert new_binding is not None and new_binding.belongs_to(
            transport="telegram", user_id="owner", thread_id="thread"
        )
        assert new_binding.model == "replacement"
    finally:
        await store.close()


async def test_undo_archives_turn_and_clears_stale_thread_summary_without_changing_files(tmp_path):
    sessions, gateway, _, old = await make_session(tmp_path)
    artifact = tmp_path / "result.txt"
    artifact.write_text("completed external effect")
    result = await receive(tmp_path, "/undo")
    assert (
        result["reply"]["status"] == "ok" and "Workspace changes remain" in result["reply"]["text"]
    )
    store = SQLiteStorage(path=tmp_path / ".harness/harness.db")
    try:
        saved = await store.get(old.id)
        assert saved is not None and saved.messages == old.messages[:2]
        assert saved.metadata["undo_archive"][0]["content"] == "latest private question"
        assert sessions.load_session(gateway.id).metadata["thread_summary"] == ""
        assert artifact.read_text() == "completed external effect"
    finally:
        await store.close()


@pytest.mark.parametrize("command", ["/model changed", "/retry", "/undo", "/compress"])
async def test_mutating_conversation_controls_refuse_unfinished_approval(tmp_path, command):
    await make_session(tmp_path)
    store = SQLiteStorage(path=tmp_path / ".harness/harness.db")
    try:
        pending = await store.create_approval(
            PendingApproval(
                session_id="runtime",
                tool_call_id="call",
                tool_name="write_file",
                arguments={"path": "result"},
            )
        )
        await store.resolve_approval(pending.id, status="granted")
        await store.claim_replay(pending.id, session_id="runtime")
    finally:
        await store.close()
    result = await receive(tmp_path, command)
    assert result["reply"]["status"] == "approval_required"


async def test_retry_uses_owned_actual_prompt_and_scoped_runtime_with_approval_pause(
    tmp_path, monkeypatch
):
    await make_session(tmp_path)
    calls = []

    async def run(**kwargs):
        calls.append(kwargs)
        return "new result"

    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", run)
    monkeypatch.setattr(commands, "_load_cli_config", lambda _: HarnessConfig())
    result = await receive(tmp_path, "/retry")
    assert result["reply"]["text"] == "new result"
    assert len(calls) == 1 and calls[0]["prompt"] == "latest private question"
    assert calls[0]["session_id"] == "runtime" and calls[0]["user_id"] == "owner"
    assert calls[0]["transport"] == "telegram" and calls[0]["max_steps"] == 7
    assert "yes" not in calls[0]


async def test_usage_reports_only_current_session(tmp_path):
    await make_session(tmp_path)
    store = SQLiteStorage(path=tmp_path / ".harness/harness.db")
    try:
        await store.append_activity(
            ActivityEvent(
                session_id="runtime",
                kind="usage.recorded",
                data={"prompt_tokens": 100, "completion_tokens": 9},
            )
        )
        await store.append_activity(
            ActivityEvent(
                session_id="other",
                kind="usage.recorded",
                data={"prompt_tokens": 9999, "private": "other user's text"},
            )
        )
    finally:
        await store.close()
    result = await receive(tmp_path, "/usage")
    assert result["reply"]["data"]["usage"]["prompt_tokens"] == 100
    assert "other user's text" not in json.dumps(result) and "9999" not in json.dumps(result)


@pytest.mark.parametrize("fail", [False, True])
async def test_compress_uses_no_tools_isolated_lifecycle_and_preserves_failure_history(
    tmp_path, monkeypatch, fail
):
    _, _, _, session = await make_session(tmp_path, provider="codex")
    session.messages[0].content = "older owned context " * 3000
    store = SQLiteStorage(path=tmp_path / ".harness/harness.db")
    await store.save(session)
    await store.close()

    class Adapter:
        name = "codex"

        def __init__(self):
            self.calls = []
            self.closed = []

        async def stream(self, **kwargs):
            self.calls.append(kwargs)
            if fail:
                raise RuntimeError("unavailable")
            yield Done(
                final_message=Message(role="assistant", content="Earlier context summarized.")
            )

        async def capabilities(self):
            return Capabilities()

        async def cancel(self, session_id):
            pass

        async def end_run(self, session_id):
            self.closed.append(session_id)

    adapter = Adapter()
    configs = []

    def build(provider, **kwargs):
        configs.append(kwargs["config"])
        return adapter

    monkeypatch.setattr(commands, "_load_cli_config", lambda _: HarnessConfig())
    monkeypatch.setattr(commands, "_build_adapter", build)
    result = await receive(tmp_path, "/compress")
    assert configs[0].provider("codex")["mode"] == "app-server"
    assert adapter.calls[0]["tools"] == []
    assert adapter.closed == [adapter.calls[0]["session_id"]]
    store = SQLiteStorage(path=tmp_path / ".harness/harness.db")
    try:
        saved = await store.get("runtime")
        assert saved is not None
        if fail:
            assert result["reply"]["status"] == "unchanged" and saved.messages == session.messages
        else:
            assert result["reply"]["status"] == "ok"
            assert sum(len(item.content or "") for item in saved.messages) < sum(
                len(item.content or "") for item in session.messages
            )
            assert (
                saved.metadata["pre_compaction_archive"][0]["content"]
                == session.messages[0].content
            )
    finally:
        await store.close()


@pytest.mark.parametrize("message", ["", "   "])
async def test_attachment_only_gateway_input_runs_owned_conversation(
    tmp_path, monkeypatch, message
):
    from harness.core import MediaAttachment

    calls = []

    async def chat_turn(**kwargs):
        calls.append(kwargs)
        return "I received the image"

    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", chat_turn)
    monkeypatch.setattr(
        gateway_runtime,
        "_load_cli_config",
        lambda _: HarnessConfig(default_provider="openai", default_model="test"),
    )
    media = MediaAttachment(kind="image", mime_type="image/png", data="eA==")
    payload = await _receive_handoff(
        working_dir=tmp_path,
        message=message,
        transport="telegram",
        user_id="media-owner",
        thread_id="media-thread",
        attachments=[media],
    )
    assert len(calls) == 1 and calls[0]["attachments"] == [media]
    assert calls[0]["user_id"] == "media-owner" and calls[0]["transport"] == "telegram"
    assert payload["reply"]["command"] == "chat"
    sessions = GatewaySessionStore(root=default_gateway_root(tmp_path))
    binding = sessions.load_runtime_binding(calls[0]["session_id"])
    assert binding and binding.belongs_to(
        transport="telegram", user_id="media-owner", thread_id="media-thread"
    )


async def test_media_does_not_reclassify_explicit_gateway_control(tmp_path, monkeypatch):
    from harness.core import MediaAttachment

    async def chat_turn(**kwargs):
        pytest.fail("An explicit status command must keep the control route")

    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", chat_turn)
    payload = await _receive_handoff(
        working_dir=tmp_path,
        message="status",
        transport="telegram",
        user_id="owner",
        thread_id="thread",
        attachments=[MediaAttachment(kind="image", mime_type="image/png", data="eA==")],
    )
    assert payload["reply"]["command"] == "status"
