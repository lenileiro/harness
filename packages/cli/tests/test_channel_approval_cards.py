from __future__ import annotations

import asyncio
import hashlib
import json
import time
from dataclasses import replace
from typing import Any

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from harness.cli.channels.approval_interactions import accept_choice
from harness.cli.channels.feishu import FeishuTransport, LarkTransport
from harness.cli.channels.runtime import deliver_message, process_message
from harness.cli.channels.teams import TeamsTransport
from harness.cli.channels.transports import DiscordTransport, SlackTransport, TelegramTransport
from harness.core.approval import PendingApproval
from harness.core.channel_interactions import lookup
from harness.core.gateway_channels import ChannelConfig, ChannelMessage, ChannelStore
from harness.core.gateway_models import GatewayMessage, GatewayRuntimeBinding, default_gateway_root
from harness.core.gateway_router import dispatch_gateway_message
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.scheduler_store import SchedulerStore
from harness.storage.sqlite import SQLiteStorage

CASES = [
    (TelegramTransport, "42", "-10", "-10:7"),
    (DiscordTransport, "42", "100", "100"),
    (SlackTransport, "T:U", "T:C", "T:C:1.0"),
    (FeishuTransport, "tenant:owner", "chat", '["chat","root"]'),
    (LarkTransport, "tenant:owner", "chat", '["chat","root"]'),
    (TeamsTransport, "tenant:owner", "tenant:conversation", '["tenant","conversation","root"]'),
]


async def seed(tmp_path, transport: Any, owner, channel, thread):
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
            model="fake",
        )
    )
    storage = SQLiteStorage(path=tmp_path / ".harness" / "harness.db")
    approval = await storage.create_approval(
        PendingApproval(
            session_id="runtime",
            tool_call_id="call",
            tool_name="write_file",
            arguments={"path": "result.txt", "content": "reviewed contents"},
        )
    )
    message = ChannelMessage(
        id="prompt",
        user_id=owner,
        channel_id=channel,
        thread_id=thread,
        text="write result",
        group=True,
        mentioned=True,
    )
    store.ingest(message)

    async def receiver(**kwargs):
        return {
            "reply": {
                "text": "write_file: result.txt, reviewed contents",
                "data": {"approval_ids": [approval.id], "harness_session_id": "runtime"},
            }
        }

    assert await process_message(cwd=tmp_path, transport=transport, store=store, receiver=receiver)
    return store, storage, approval


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
                "actions": [{"action_id": "harness_approve", "value": token, "action_ts": "3.0"}],
            },
            store,
        )
    elif transport.name in {"feishu", "lark"}:
        body = json.dumps(
            {
                "header": {
                    "event_type": "card.action.trigger",
                    "event_id": "click",
                    "app_id": "app",
                    "token": "verify",
                    "tenant_key": "tenant",
                },
                "event": {
                    "operator": {"open_id": actor, "tenant_key": "tenant"},
                    "context": {"open_chat_id": "chat", "open_message_id": message_id},
                    "action": {"tag": "button", "value": {"harness_approval": token}},
                },
            }
        ).encode()
        stamp = str(int(time.time()))
        signature = hashlib.sha256((stamp + "nonce" + "encrypt-key").encode() + body).hexdigest()
        await transport.handle_event(
            {
                "X-Lark-Request-Timestamp": stamp,
                "X-Lark-Request-Nonce": "nonce",
                "X-Lark-Signature": signature,
            },
            body,
            store,
        )
    else:
        activity = {
            "type": "invoke",
            "name": "adaptiveCard/action",
            "id": "click",
            "replyToId": message_id,
            "serviceUrl": "https://smba.trafficmanager.net/emea/",
            "channelId": "msteams",
            "channelData": {"tenant": {"id": "tenant"}},
            "conversation": {"id": "conversation", "conversationType": "channel"},
            "from": {"id": "29:" + actor, "aadObjectId": actor},
            "recipient": {"id": "28:app"},
            "value": {
                "action": {
                    "type": "Action.Execute",
                    "verb": "harness_approval",
                    "data": {"harness_approval": token},
                }
            },
        }
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public = json.loads(RSAAlgorithm.to_jwk(key.public_key()))
        public.update(kid="key", endorsements=["msteams"])
        transport.keys, transport.keys_fetched = {"key": public}, time.time()
        signed = jwt.encode(
            {
                "iss": "https://api.botframework.com",
                "aud": "app",
                "nbf": int(time.time()) - 1,
                "exp": int(time.time()) + 3600,
                "serviceurl": activity["serviceUrl"],
            },
            key,
            algorithm="RS256",
            headers={"kid": "key"},
        )
        await transport.handle_event(
            {"Authorization": "Bearer " + signed}, json.dumps(activity).encode(), store
        )


@pytest.mark.parametrize("choice", ["approve", "deny"])
@pytest.mark.parametrize("cls,owner,channel,thread", CASES)
async def test_native_cards_persist_owner_scope_and_resume_once(
    tmp_path, cls, owner, channel, thread, choice
):
    transport, requests = make_transport(cls, owner, channel)
    store, storage, approval = await seed(tmp_path, transport, owner, channel, thread)
    if transport.name == "teams":
        transport.store = store
        store.set(
            "teams-route:" + thread,
            {
                "service_url": "https://smba.trafficmanager.net/emea/",
                "recipient": {"id": "28:app"},
                "sender": {"id": "29:owner"},
                "user_id": owner,
            },
        )
    try:
        assert await deliver_message(transport=transport, store=store)  # full action text
        assert await deliver_message(transport=transport, store=store)  # native decision card
        card = dict(store.db.execute("SELECT * FROM approval_interactions").fetchone())
        assert card["native_message_id"] == "900"
        body = json.loads(requests[-1].content)
        encoded = json.dumps(body)
        # Payload carries opaque decisions, never an arbitrary command or expanded scope.
        assert card["approve_token"] in encoded and card["deny_token"] in encoded
        assert "reviewed contents" not in encoded and len(card["approve_token"].encode()) < 64
        if transport.name == "telegram":
            assert "inline_keyboard" in encoded
        elif transport.name == "discord":
            assert body["components"][0]["components"][0]["style"] == 3
        elif transport.name == "slack":
            assert body["thread_ts"] == "1.0" and "blocks" in body
        elif transport.name in {"feishu", "lark"}:
            assert body["msg_type"] == "interactive" and body["reply_in_thread"]
        else:
            assert "Action.Execute" in encoded and requests[-1].url.path.endswith("/root")
        store.close()
        store = ChannelStore(cwd=tmp_path, transport=transport.name)
        if transport.name == "teams":
            transport.store = store
        await callback(transport, store, card["approve_token"], actor="stranger")
        await callback(transport, store, card["approve_token"], message_id="another-card")
        assert store.claim_message() is None
        persisted = await storage.get_approval(approval.id)
        assert persisted and persisted.status == "pending"
        await callback(transport, store, card[choice + "_token"])
        await callback(transport, store, card["deny_token"])
        await callback(transport, store, card["approve_token"])
        resumed = []

        async def resume(binding):
            resumed.append(binding.session_id)
            return "Executed once"

        async def receiver(**kwargs):
            reply, _ = await dispatch_gateway_message(
                cwd=tmp_path,
                session_store=GatewaySessionStore(root=default_gateway_root(tmp_path)),
                scheduler_store=SchedulerStore(root=tmp_path / ".harness" / "scheduler"),
                approval_store=storage,
                resume_approval=resume,
                message=GatewayMessage(
                    id="decision",
                    transport=transport.name,
                    user_id=kwargs["user_id"],
                    thread_id=kwargs["thread_id"],
                    text=kwargs["message"],
                ),
            )
            return {"reply": reply.to_dict()}

        assert await process_message(
            cwd=tmp_path, transport=transport, store=store, receiver=receiver
        )
        assert not await process_message(
            cwd=tmp_path, transport=transport, store=store, receiver=receiver
        )
        assert resumed == (["runtime"] if choice == "approve" else [])
        persisted = await storage.get_approval(approval.id)
        assert persisted and persisted.status == ("granted" if choice == "approve" else "denied")
    finally:
        store.close()
        await storage.close()
        await transport.client.aclose()
        await transport.close()


@pytest.mark.parametrize(
    "failure", ["expired", "session", "resolved", "unbound", "thread", "allowlist"]
)
async def test_approval_control_revalidates_before_enqueue(tmp_path, failure):
    transport, _ = make_transport(TelegramTransport, "42", "-10")
    store, storage, approval = await seed(tmp_path, transport, "42", "-10", "-10:7")
    try:
        await deliver_message(transport=transport, store=store)
        await deliver_message(transport=transport, store=store)
        row = dict(store.db.execute("SELECT * FROM approval_interactions").fetchone())
        if failure in {"expired", "session", "unbound"}:
            column, value = {
                "expired": ("expires", 0),
                "session": ("session_id", "another"),
                "unbound": ("native_message_id", ""),
            }[failure]
            with store.db:
                store.db.execute(f"UPDATE approval_interactions SET {column}=?", (value,))
        elif failure == "resolved":
            await storage.resolve_approval(approval.id, status="denied")
        elif failure == "allowlist":
            transport.config = replace(transport.config, allowed_users=[])
        assert not await accept_choice(
            transport,
            store,
            token=row["approve_token"],
            user_id="42",
            channel_id="-10",
            thread_id="-10:8" if failure == "thread" else "-10:7",
            native_message_id="900",
            event_id="click",
        )
        assert store.claim_message() is None
    finally:
        store.close()
        await storage.close()
        await transport.client.aclose()
        await transport.close()


async def test_slack_socket_acks_persisted_approval_without_running_model(tmp_path):
    transport, _ = make_transport(SlackTransport, "T:U", "T:C")
    store, storage, _ = await seed(tmp_path, transport, "T:U", "T:C", "T:C:1.0")
    await deliver_message(transport=transport, store=store)
    await deliver_message(transport=transport, store=store)
    row = dict(store.db.execute("SELECT * FROM approval_interactions").fetchone())
    received = []

    async def server(socket):
        await socket.send(
            json.dumps(
                {
                    "envelope_id": "envelope",
                    "type": "interactive",
                    "payload": {
                        "type": "block_actions",
                        "team": {"id": "T"},
                        "user": {"id": "U"},
                        "channel": {"id": "C"},
                        "container": {"message_ts": "900"},
                        "message": {"user": "bot", "thread_ts": "1.0"},
                        "actions": [
                            {
                                "action_id": "harness_approve",
                                "value": row["approve_token"],
                                "action_ts": "2.0",
                            }
                        ],
                    },
                }
            )
        )
        received.append(json.loads(await socket.recv()))
        persisted = lookup(store, row["approve_token"])
        assert persisted and persisted["consumed"] == 1
        await socket.send(json.dumps({"type": "disconnect"}))

    try:
        async with serve(server, "127.0.0.1", 0) as local:

            async def call(method, data=None, *, app=False):
                return {"url": "wss://wss.slack.com/test"}

            transport.call = call
            transport.socket_connect = lambda url, **kw: connect(
                f"ws://127.0.0.1:{local.sockets[0].getsockname()[1]}", **kw
            )
            await asyncio.wait_for(transport.receive(store), 5)
        assert received == [{"envelope_id": "envelope"}]
        command = store.claim_message()
        assert command and command.text.startswith("approve ")
    finally:
        store.close()
        await storage.close()
        await transport.client.aclose()
        await transport.close()


async def test_scheduled_approval_notification_creates_one_owner_bound_card(tmp_path):
    from harness.cli.gateway_hooks import ChannelNotificationHook

    transport, _ = make_transport(TelegramTransport, "42", "-10")
    store, storage, approval = await seed(tmp_path, transport, "42", "-10", "-10:7")
    try:
        # Reconstruct the scheduler's first notification, with a persisted prior
        # conversation but no ordinary prompt reply card.
        with store.db:
            store.db.execute("DELETE FROM outbox")
            store.db.execute("DELETE FROM approval_interactions")
        hook = ChannelNotificationHook()
        hook.on_approval_requested(cwd=tmp_path, approval=approval)
        hook.on_approval_requested(cwd=tmp_path, approval=approval)
        assert store.db.execute("SELECT COUNT(*) FROM approval_interactions").fetchone()[0] == 1
        card = dict(store.db.execute("SELECT * FROM approval_interactions").fetchone())
        assert (
            card["user_id"] == "42"
            and card["thread_id"] == "-10:7"
            and card["session_id"] == "runtime"
        )
        assert await deliver_message(transport=transport, store=store)
        assert await deliver_message(transport=transport, store=store)
        await callback(transport, store, card["deny_token"])
        command = store.claim_message()
        assert command and command.text == "deny " + approval.id
    finally:
        store.close()
        await storage.close()
        await transport.client.aclose()
        await transport.close()


async def test_concurrent_paired_choices_enqueue_only_one_decision(tmp_path):
    transport, _ = make_transport(TelegramTransport, "42", "-10")
    store, storage, _ = await seed(tmp_path, transport, "42", "-10", "-10:7")
    try:
        await deliver_message(transport=transport, store=store)
        await deliver_message(transport=transport, store=store)
        row = dict(store.db.execute("SELECT * FROM approval_interactions").fetchone())
        other = ChannelStore(cwd=tmp_path, transport="telegram")
        try:
            results = await asyncio.gather(
                *[
                    accept_choice(
                        transport,
                        target,
                        token=row[action + "_token"],
                        user_id="42",
                        channel_id="-10",
                        thread_id="-10:7",
                        native_message_id="900",
                        event_id=action,
                    )
                    for target, action in [(store, "approve"), (other, "deny")]
                ]
            )
            assert sum(results) == 1
            assert store.claim_message() is not None and other.claim_message() is None
        finally:
            other.close()
    finally:
        store.close()
        await storage.close()
        await transport.client.aclose()
        await transport.close()


async def test_expired_card_is_not_sent_after_earlier_outbox_delay(tmp_path):
    transport, requests = make_transport(TelegramTransport, "42", "-10")
    store, storage, _ = await seed(tmp_path, transport, "42", "-10", "-10:7")
    try:
        assert await deliver_message(transport=transport, store=store)
        with store.db:
            store.db.execute("UPDATE approval_interactions SET expires=0")
        before = len(requests)
        assert await deliver_message(transport=transport, store=store)
        assert len(requests) == before
        assert store.status()["outbox"][0]["status"] == "failed"
        assert "expired" in store.status()["outbox"][0]["error"]
    finally:
        store.close()
        await storage.close()
        await transport.client.aclose()
        await transport.close()
