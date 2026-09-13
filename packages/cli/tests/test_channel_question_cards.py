from __future__ import annotations

import json
import time
from dataclasses import replace
from typing import Any

import httpx
import pytest

from harness.cli import gateway_clarification
from harness.cli.channels.question_interactions import accept_question_choice
from harness.cli.channels.runtime import deliver_message, process_message
from harness.cli.channels.transports import DiscordTransport, SlackTransport, TelegramTransport
from harness.core.approval import PendingApproval
from harness.core.clarification import QuestionSpec, QuestionStore
from harness.core.gateway_channels import ChannelConfig, ChannelMessage, ChannelStore
from harness.core.gateway_models import GatewayRuntimeBinding, default_gateway_root
from harness.core.gateway_sessions import GatewaySessionStore
from harness.storage.sqlite import SQLiteStorage

CASES = [
    (TelegramTransport, "42", "-10", "-10:7"),
    (DiscordTransport, "42", "100", "100"),
    (SlackTransport, "T:U", "T:C", "T:C:1.0"),
]


def make_transport(cls: Any, owner, channel):
    requests = []

    def handler(request):
        requests.append(request)
        if "/interactions/" in request.url.path:
            return httpx.Response(204)
        return httpx.Response(
            200,
            json={
                "ok": True,
                "result": {"message_id": 900},
                "id": "900",
                "ts": "900",
                "code": 0,
                "data": {"message_id": "900"},
            },
        )

    transport = cls(
        config=ChannelConfig(
            app_id="app",
            tenant_id="tenant",
            allowed_users=[owner],
            allowed_channels=[channel],
            allow_groups=True,
        ),
        token="secret",
        app_token="verify",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    transport.bot_id, transport.team_id = "bot", "T"
    transport.access_token, transport.token_expires = "access", time.time() + 3600
    transport.encrypt_key = "encrypt-key"
    return transport, requests


async def callback(transport: Any, store, token, *, actor="owner", message_id="900"):
    if transport.name == "telegram":
        await transport.approval_callback(
            {
                "id": "click",
                "data": token,
                "from": {"id": "42" if actor == "owner" else "43"},
                "message": {
                    "message_id": message_id,
                    "from": {"id": "bot"},
                    "chat": {"id": "-10", "type": "supergroup"},
                    "message_thread_id": 7,
                },
            },
            store,
        )
    elif transport.name == "discord":
        await transport.approval_callback(
            {
                "id": "999",
                "token": "callback-secret",
                "type": 3,
                "application_id": "bot",
                "channel_id": "100",
                "guild_id": "guild",
                "member": {"user": {"id": "42" if actor == "owner" else "43"}},
                "message": {"id": message_id, "author": {"id": "bot"}},
                "data": {"custom_id": token, "component_type": 2},
            },
            store,
        )
    elif transport.name == "slack":
        await transport.approval_callback(
            {
                "type": "block_actions",
                "team": {"id": "T"},
                "user": {"id": "U" if actor == "owner" else "other"},
                "channel": {"id": "C"},
                "container": {"type": "message", "message_ts": message_id},
                "message": {"user": "bot", "thread_ts": "1.0"},
                "actions": [
                    {"action_id": "harness_question_0", "value": token, "action_ts": "3.0"}
                ],
            },
            store,
        )


async def setup_question(tmp_path, cls, owner, channel, thread, questions):
    transport, requests = make_transport(cls, owner, channel)
    store = ChannelStore(cwd=tmp_path, transport=transport.name)
    sessions = GatewaySessionStore(root=default_gateway_root(tmp_path))
    session = sessions.get_or_create_session(
        transport=transport.name, user_id=owner, thread_id=thread
    )
    sessions.bind_runtime_session(
        GatewayRuntimeBinding(
            session_id="runtime",
            gateway_session_id=session.id,
            transport=transport.name,
            user_id=owner,
            thread_id=thread,
            provider="fake",
            model="original-model",
        )
    )
    ledger = QuestionStore(tmp_path / ".harness" / "harness.db")
    record = ledger.create(
        session_id="runtime",
        scope=gateway_clarification.gateway_question_scope(tmp_path, transport.name, owner),
        tool_call_id="clarify-call",
        questions=questions,
    )
    incoming = ChannelMessage(
        id="prompt",
        user_id=owner,
        channel_id=channel,
        thread_id=thread,
        text="start",
        group=True,
        mentioned=True,
    )
    store.ingest(incoming)

    async def receiver(**kwargs):
        return gateway_clarification.question_reply(record, session)

    assert await process_message(cwd=tmp_path, transport=transport, store=store, receiver=receiver)
    while await deliver_message(transport=transport, store=store):
        pass
    return transport, requests, store, ledger, record


def latest_card(store):
    return dict(
        store.db.execute(
            "SELECT * FROM question_interactions ORDER BY rowid DESC LIMIT 1"
        ).fetchone()
    )


def tokens(store, row):
    return {
        x["action"]: x["token"]
        for x in store.db.execute(
            "SELECT * FROM question_interaction_tokens WHERE interaction_id=?", (row["id"],)
        )
    }


@pytest.mark.parametrize("cls,owner,channel,thread", CASES[:3])
async def test_native_question_partial_answer_restart_and_original_session_resume(
    tmp_path, monkeypatch, cls, owner, channel, thread
):
    transport, requests, store, ledger, record = await setup_question(
        tmp_path,
        cls,
        owner,
        channel,
        thread,
        [
            QuestionSpec(question="Choose a format", choices=["Concise", "Detailed"]),
            QuestionSpec(question="What name?", choices=["Sample", "Other example"]),
        ],
    )
    resumed = []

    async def resume(**kwargs):
        assert (
            kwargs["binding"].model == "original-model"
            and kwargs["binding"].session_id == "runtime"
        )
        resumed.append(kwargs["record"].answers)
        ledger.mark_applied(record.id, scope=record.scope, session_id="runtime")
        return {"reply": {"text": "Continued original session", "data": {}}}

    monkeypatch.setattr(gateway_clarification, "resume_gateway_question", resume)

    async def receiver(**kwargs):
        result = await gateway_clarification.dispatch_gateway_question(
            cwd=tmp_path,
            message=kwargs["message"],
            transport=transport.name,
            user_id=kwargs["user_id"],
            thread_id=kwargs["thread_id"],
        )
        assert result is not None
        return result

    storage = SQLiteStorage(path=tmp_path / ".harness" / "harness.db")
    approval = await storage.create_approval(
        PendingApproval(session_id="runtime", tool_call_id="mutation", tool_name="write_file")
    )
    try:
        first = latest_card(store)
        buttons = tokens(store, first)
        native = json.loads(requests[-1].content)
        assert buttons["0"] in json.dumps(native) and "Concise" in json.dumps(native)
        await callback(transport, store, buttons["0"], actor="stranger")
        await callback(transport, store, buttons["0"], message_id="another-message")
        assert store.claim_message() is None
        await callback(transport, store, buttons["0"])
        await callback(transport, store, buttons["1"])
        assert await process_message(
            cwd=tmp_path, transport=transport, store=store, receiver=receiver
        )
        partial = ledger.get(record.id, scope=record.scope, session_id="runtime")
        assert partial and partial.status == "pending" and partial.answers == {"q0": "Concise"}
        assert not resumed
        while await deliver_message(transport=transport, store=store):
            pass
        second = latest_card(store)
        assert second["question_key"] == "q1"
        store.close()
        store = ChannelStore(cwd=tmp_path, transport=transport.name)
        await callback(transport, store, tokens(store, second)["other"])
        # Other authorizes this owner's next answer without another group mention.
        answer = ChannelMessage(
            id="typed",
            user_id=owner,
            channel_id=channel,
            thread_id=thread,
            text="My chosen name",
            group=True,
        )
        transport.accept(replace(answer, id="foreign", user_id="stranger"), store)
        transport.accept(answer, store)
        transport.accept(replace(answer, id="too-late", text="Do not start another run"), store)
        assert await process_message(
            cwd=tmp_path, transport=transport, store=store, receiver=receiver
        )
        complete = ledger.get(record.id, scope=record.scope, session_id="runtime")
        assert (
            complete
            and complete.answers == {"q0": "Concise", "q1": "My chosen name"}
            and complete.applied
        )
        assert resumed == [{"q0": "Concise", "q1": "My chosen name"}]
        assert await process_message(
            cwd=tmp_path, transport=transport, store=store, receiver=receiver
        )
        assert len(resumed) == 1  # stale captured text becomes /questions, never a new model run
        assert store.claim_message() is None
        unchanged = await storage.get_approval(approval.id)
        assert unchanged and unchanged.status == "pending"
    finally:
        store.close()
        ledger.close()
        await storage.close()
        await transport.client.aclose()
        await transport.close()


async def test_multi_select_persists_toggles_deduplicates_event_and_queues_exact_list(tmp_path):
    cls, owner, channel, thread = CASES[0]
    transport, _, store, ledger, record = await setup_question(
        tmp_path,
        cls,
        owner,
        channel,
        thread,
        [
            QuestionSpec(
                question="Choose formats",
                choices=["Plain text", "Markdown", "HTML"],
                multi_select=True,
            )
        ],
    )
    try:
        row = latest_card(store)
        buttons = tokens(store, row)

        async def choice(action: str, event: str) -> str:
            return await accept_question_choice(
                transport,
                store,
                token=buttons[action],
                user_id=owner,
                channel_id=channel,
                thread_id=thread,
                native_message_id="900",
                event_id=event,
            )

        assert "Plain text" in await choice("0", "click-one")
        assert "already received" in await choice("0", "click-one")
        assert "Markdown" in await choice("1", "click-two")
        store.close()
        store = ChannelStore(cwd=tmp_path, transport=transport.name)
        assert json.loads(latest_card(store)["selected"]) == ["Plain text", "Markdown"]
        assert store.claim_message() is None
        assert await choice("submit", "submit") == "Answer queued."
        assert (await choice("0", "too-late")).startswith("Unavailable")
        message = store.claim_message()
        assert message and json.loads(message.text.split(None, 2)[2]) == {
            "q0": ["Plain text", "Markdown"]
        }
        current = ledger.get(record.id, scope=record.scope, session_id="runtime")
        assert current and current.answers == {}  # only gateway /answer may update the ledger
    finally:
        store.close()
        ledger.close()
        await transport.client.aclose()
        await transport.close()


@pytest.mark.parametrize("condition", ["expired", "cancelled", "session", "spec", "allowlist"])
async def test_native_question_revalidates_authority_and_snapshot(tmp_path, condition):
    cls, owner, channel, thread = CASES[0]
    transport, _, store, ledger, record = await setup_question(
        tmp_path,
        cls,
        owner,
        channel,
        thread,
        [QuestionSpec(question="Question", choices=["A", "B"])],
    )
    try:
        row = latest_card(store)
        token = tokens(store, row)["0"]
        if condition == "expired":
            with store.db:
                store.db.execute("UPDATE question_interactions SET expires=?", (time.time() - 1,))
        elif condition == "cancelled":
            ledger.cancel(record.id, scope=record.scope, session_id="runtime")
        elif condition in {"session", "spec"}:
            field, value = (
                ("session_id", "foreign")
                if condition == "session"
                else (
                    "spec",
                    json.dumps(
                        {"question": "Changed", "choices": ["A", "B"], "multi_select": False}
                    ),
                )
            )
            with store.db:
                store.db.execute(f"UPDATE question_interactions SET {field}=?", (value,))
        else:
            transport.config = replace(transport.config, allowed_users=[])
        result = await accept_question_choice(
            transport,
            store,
            token=token,
            user_id=owner,
            channel_id=channel,
            thread_id=thread,
            native_message_id="900",
            event_id="click",
        )
        assert result.startswith("Unavailable") and store.claim_message() is None
    finally:
        store.close()
        ledger.close()
        await transport.client.aclose()
        await transport.close()
