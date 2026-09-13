"""Google's authenticated cards drive the durable question ledger offline."""

from __future__ import annotations

import json
from collections.abc import Mapping

import httpx
import pytest

from harness.cli.channels.google_chat import QUESTION_ACTION, GoogleChatTransport
from harness.cli.channels.runtime import deliver_message
from harness.cli.channels.webhooks import AuthenticationError
from harness.cli.gateway_clarification import gateway_question_scope
from harness.core.clarification import QuestionSpec, QuestionStore
from harness.core.gateway_channels import ChannelConfig, ChannelMessage, ChannelStore
from harness.core.gateway_models import GatewayRuntimeBinding, default_gateway_root
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.question_interactions import queue_card


class GoogleFixture(GoogleChatTransport):
    async def _access(self) -> str:
        return "bot-token"

    async def _verify(self, headers: Mapping[str, str]) -> None:
        if headers.get("Authorization") != "verified":
            raise AuthenticationError("Invalid fixture signature")


@pytest.fixture
async def native_question(tmp_path, request):
    multi = getattr(request, "param", False)
    thread = json.dumps(["spaces/s", "spaces/s/threads/t"], separators=(",", ":"))
    sessions = GatewaySessionStore(root=default_gateway_root(tmp_path))
    gateway = sessions.get_or_create_session(
        transport="google_chat", user_id="users/u", thread_id=thread
    )
    sessions.bind_runtime_session(
        GatewayRuntimeBinding(
            session_id="session",
            gateway_session_id=gateway.id,
            transport="google_chat",
            user_id="users/u",
            thread_id=thread,
            provider="fake",
            model="model",
            max_steps=4,
        )
    )
    ledger = QuestionStore(tmp_path / ".harness/harness.db")
    spec = QuestionSpec(
        question="Choose <one> & explain", choices=["SQLite", "Postgres"], multi_select=multi
    )
    record = ledger.create(
        session_id="session",
        tool_call_id="call",
        scope=gateway_question_scope(tmp_path, "google_chat", "users/u"),
        questions=[spec],
    )
    store = ChannelStore(cwd=tmp_path, transport="google_chat")
    message = ChannelMessage(
        id="spaces/s/messages/input",
        user_id="users/u",
        channel_id="spaces/s",
        thread_id=thread,
        text="task",
        group=True,
        mentioned=True,
    )
    with store.db:
        queue_card(
            store,
            message,
            {
                "question_id": record.id,
                "session_id": "session",
                "question_key": "q0",
                "spec": spec.model_dump(mode="json"),
                "expires": record.expires_at.timestamp(),
            },
            start=0,
        )
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer bot-token"
        return httpx.Response(200, json={"name": "spaces/s/messages/card"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = GoogleFixture(
        config=ChannelConfig(
            allowed_users=["users/u", "users/other"],
            allowed_channels=["spaces/s"],
            allow_groups=True,
        ),
        token="fixture",
        client=client,
    )
    try:
        assert await deliver_message(transport=transport, store=store)
        row = store.db.execute("SELECT * FROM question_interactions").fetchone()
        yield transport, store, ledger, record, dict(row), requests
    finally:
        ledger.close()
        store.close()
        await transport.close()
        await client.aclose()


def click(
    token,
    *,
    tick="1",
    user="users/u",
    native="spaces/s/messages/card",
    root="spaces/s/threads/t",
    common=False,
):
    event = {
        "type": "CARD_CLICKED",
        "eventTime": f"2026-09-13T08:00:0{tick}.000Z",
        "user": {"name": user, "type": "HUMAN"},
        "space": {"name": "spaces/s", "type": "ROOM"},
        "message": {"name": native, "thread": {"name": root}},
    }
    if common:
        event["common"] = {"invokedFunction": QUESTION_ACTION, "parameters": {"token": token}}
    else:
        event["action"] = {
            "actionMethodName": QUESTION_ACTION,
            "parameters": [{"key": "token", "value": token}],
        }
    return json.dumps(event).encode()


async def test_google_question_card_receipt_and_authenticated_owner_thread_binding(native_question):
    transport, store, ledger, record, row, requests = native_question
    assert row["native_message_id"] == "spaces/s/messages/card"
    sent = requests[0]
    payload = json.loads(sent.content)
    assert payload["thread"] == {"name": "spaces/s/threads/t"}
    assert (
        sent.url.params["messageReplyOption"] == "REPLY_MESSAGE_OR_FAIL"
        and sent.url.params["requestId"]
    )
    widgets = payload["cardsV2"][0]["card"]["sections"][0]["widgets"]
    assert "&lt;one&gt; &amp;" in widgets[0]["textParagraph"]["text"]
    buttons = json.loads(row["buttons"])
    token = buttons[1]["token"]
    assert widgets[1]["buttonList"]["buttons"][1]["onClick"]["action"]["parameters"] == [
        {"key": "token", "value": token}
    ]
    with pytest.raises(AuthenticationError):
        await transport.handle_event({}, click(token), store)
    for change in (
        {"user": "users/other"},
        {"native": "spaces/s/messages/different"},
        {"root": "spaces/s/threads/other"},
    ):
        result = await transport.handle_event(
            {"Authorization": "verified"},
            click(
                token,
                user=change.get("user", "users/u"),
                native=change.get("native", "spaces/s/messages/card"),
                root=change.get("root", "spaces/s/threads/t"),
            ),
            store,
        )
        assert "Unavailable" in result["text"] and store.claim_message() is None
    result = await transport.handle_event(
        {"Authorization": "verified"}, click(token, common=True), store
    )
    assert result == {"text": "Answer queued.", "privateMessageViewer": {"name": "users/u"}}
    queued = store.claim_message()
    assert (
        queued is not None and queued.user_id == "users/u" and queued.thread_id == row["thread_id"]
    )
    assert queued.text == "/answer " + record.id + ' {"q0": "Postgres"}'
    await transport.handle_event({"Authorization": "verified"}, click(token, tick="2"), store)
    assert store.claim_message() is None
    unchanged = ledger.get(record.id, scope=record.scope)
    assert unchanged is not None and unchanged.answers == {}  # Gateway applies queued data.


@pytest.mark.parametrize("native_question", [True], indirect=True)
async def test_google_multi_selection_is_durable_until_submit_and_retries_do_not_toggle(
    native_question, tmp_path
):
    transport, store, _, _, row, _ = native_question
    buttons = json.loads(row["buttons"])
    for index in (0, 1):
        event = click(buttons[index]["token"], tick=str(index + 1))
        result = await transport.handle_event({"Authorization": "verified"}, event, store)
        assert result["text"].startswith("Selected:")
        await transport.handle_event({"Authorization": "verified"}, event, store)
    assert store.claim_message() is None
    restarted = ChannelStore(cwd=tmp_path, transport="google_chat")
    try:
        result = await transport.handle_event(
            {"Authorization": "verified"}, click(buttons[-1]["token"], tick="3"), restarted
        )
        assert result["text"] == "Answer queued."
        queued = restarted.claim_message()
        assert queued is not None and json.loads(queued.text.split(" ", 2)[2]) == {
            "q0": ["SQLite", "Postgres"]
        }
    finally:
        restarted.close()


async def test_google_card_allowlist_rechecked_at_callback(native_question):
    transport, store, _, _, row, _ = native_question
    transport.config.allowed_users.clear()
    token = json.loads(row["buttons"])[0]["token"]
    result = await transport.handle_event({"Authorization": "verified"}, click(token), store)
    assert "Unavailable" in result["text"] and store.claim_message() is None
