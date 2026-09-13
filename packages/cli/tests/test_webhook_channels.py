import base64
import hashlib
import json
import time
from dataclasses import replace
from urllib.parse import parse_qs

import httpx
import jwt
import pytest
from aiohttp import web
from cryptography.hazmat.primitives import padding, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from jwt.algorithms import RSAAlgorithm

from harness.cli.channels.feishu import FeishuTransport, LarkTransport
from harness.cli.channels.google_chat import GoogleChatTransport
from harness.cli.channels.matrix import MatrixTransport
from harness.cli.channels.runtime import deliver_message, process_message
from harness.cli.channels.transports import ChannelError
from harness.cli.channels.webhooks import AuthenticationError
from harness.core.gateway_channels import ChannelConfig, ChannelMessage, ChannelStore


@pytest.fixture(scope="module")
def keys():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()
    ).decode()
    public = json.loads(RSAAlgorithm.to_jwk(key.public_key()))
    public.update(kid="google-key", alg="RS256")
    return private, public


def google_token(keys, **overrides):
    now = int(time.time())
    claims = {
        "iss": "https://accounts.google.com",
        "aud": "https://bot.example/events",
        "iat": now,
        "exp": now + 3600,
        "email": "chat@system.gserviceaccount.com",
        "email_verified": True,
    }
    claims.update(overrides)
    return jwt.encode(claims, keys[0], algorithm="RS256", headers={"kid": "google-key"})


def google_event():
    return {
        "type": "MESSAGE",
        "user": {"name": "users/owner", "type": "HUMAN"},
        "space": {"name": "spaces/space", "type": "SPACE"},
        "message": {
            "name": "spaces/space/messages/one",
            "text": "@bot status",
            "argumentText": "status",
            "thread": {"name": "spaces/space/threads/topic"},
            "annotations": [
                {
                    "type": "USER_MENTION",
                    "userMention": {"user": {"name": "users/bot", "type": "BOT"}},
                }
            ],
        },
    }


async def make_google(keys):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == "oauth2.googleapis.com":
            fields = parse_qs(request.content.decode())
            assertion = jwt.decode(
                fields["assertion"][0],
                jwt.PyJWK.from_dict(keys[1]).key,
                algorithms=["RS256"],
                audience="https://oauth2.googleapis.com/token",
            )
            assert assertion["scope"] == "https://www.googleapis.com/auth/chat.bot"
            return httpx.Response(200, json={"access_token": "access-secret", "expires_in": 3600})
        if request.url.host == "www.googleapis.com":
            return httpx.Response(200, json={"keys": [keys[1]]})
        assert request.url.host == "chat.googleapis.com"
        assert request.headers["authorization"] == "Bearer access-secret"
        return httpx.Response(200, json={"name": "spaces/space/messages/reply"})

    config = ChannelConfig(
        allowed_users=["users/owner"],
        allowed_channels=["spaces/space"],
        audience="https://bot.example/events",
        app_id="users/bot",
        allow_groups=True,
    )
    account = {
        "type": "service_account",
        "client_email": "bot@project.iam.gserviceaccount.com",
        "private_key": keys[0],
    }
    transport = GoogleChatTransport(
        config=config,
        token=json.dumps(account),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await transport.authenticate()
    return transport, requests


async def test_google_listener_verifies_before_durable_ack_and_approval_dispatch(tmp_path, keys):
    transport, requests = await make_google(keys)
    store = ChannelStore(cwd=tmp_path, transport="google_chat")
    runner = web.AppRunner(transport.application(store), access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    address = runner.addresses[0]
    url = f"http://127.0.0.1:{address[1]}/events"
    try:
        async with httpx.AsyncClient() as client:
            rejected = await client.post(url, json=google_event())
            assert rejected.status_code == 401 and store.claim_message() is None
            headers = {"Authorization": "Bearer " + google_token(keys)}
            accepted = await client.post(url, json=google_event(), headers=headers)
            duplicate = await client.post(url, json=google_event(), headers=headers)
            assert accepted.status_code == duplicate.status_code == 200
            row = store.db.execute("SELECT status FROM inbox").fetchall()
            assert len(row) == 1 and row[0]["status"] == "pending"
            # No model runs inside the request handler; durable ACK is already sent.
            called = []

            async def receiver(**kwargs):
                called.append(kwargs)
                return {"reply": {"text": "approval status", "data": {}}}

            assert await process_message(
                cwd=tmp_path, transport=transport, store=store, receiver=receiver
            )
            assert not await process_message(
                cwd=tmp_path, transport=transport, store=store, receiver=receiver
            )
            assert called[0]["user_id"] == "users/owner" and called[0]["message"] == "status"
            assert json.loads(called[0]["thread_id"]) == [
                "spaces/space",
                "spaces/space/threads/topic",
            ]
            assert await deliver_message(transport=transport, store=store)
            sent = requests[-1]
            assert sent.url.params["messageReplyOption"] == "REPLY_MESSAGE_OR_FAIL"
            assert json.loads(sent.content)["thread"]["name"] == "spaces/space/threads/topic"
            too_large = await client.post(
                url,
                content=b"x" * (1024 * 1024 + 1),
                headers={**headers, "Content-Type": "application/json"},
            )
            assert too_large.status_code == 413
    finally:
        await runner.cleanup()
        store.close()
        await transport.client.aclose()
        await transport.close()


@pytest.mark.parametrize(
    "changes",
    [
        {"aud": "https://other.example/events"},
        {"iss": "https://attacker.example"},
        {"email": "other@system.gserviceaccount.com"},
        {"email_verified": False},
        {"exp": 1},
    ],
)
async def test_google_invalid_claims_cannot_ingest(tmp_path, keys, changes):
    transport, _ = await make_google(keys)
    store = ChannelStore(cwd=tmp_path, transport="google_chat")
    try:
        with pytest.raises(AuthenticationError):
            await transport.handle_event(
                {"Authorization": "Bearer " + google_token(keys, **changes)},
                json.dumps(google_event()).encode(),
                store,
            )
        assert store.claim_message() is None
    finally:
        store.close()
        await transport.client.aclose()
        await transport.close()


def encrypt_feishu(payload, key="encrypt-secret"):
    plaintext = json.dumps(payload).encode()
    padder = padding.PKCS7(128).padder()
    padded = padder.update(plaintext) + padder.finalize()
    iv = b"0" * 16
    encryptor = Cipher(
        algorithms.AES(hashlib.sha256(key.encode()).digest()), modes.CBC(iv)
    ).encryptor()
    return json.dumps(
        {"encrypt": base64.b64encode(iv + encryptor.update(padded) + encryptor.finalize()).decode()}
    ).encode()


def feishu_event():
    return {
        "schema": "2.0",
        "header": {
            "event_id": "event-one",
            "app_id": "app",
            "tenant_key": "tenant",
            "token": "verification-secret",
            "event_type": "im.message.receive_v1",
        },
        "event": {
            "sender": {"sender_type": "user", "sender_id": {"open_id": "owner"}},
            "message": {
                "message_id": "message",
                "chat_id": "chat",
                "chat_type": "group",
                "message_type": "text",
                "content": json.dumps({"text": "@_user_1 approve approval-id"}),
                "mentions": [{"key": "@_user_1", "id": {"open_id": "bot"}}],
            },
        },
    }


def feishu_headers(body, timestamp=None):
    timestamp = str(int(time.time())) if timestamp is None else str(timestamp)
    return {
        "X-Lark-Request-Timestamp": timestamp,
        "X-Lark-Request-Nonce": "nonce",
        "X-Lark-Signature": hashlib.sha256(
            (timestamp + "nonce" + "encrypt-secret").encode() + body
        ).hexdigest(),
    }


@pytest.mark.parametrize("transport_type", [FeishuTransport, LarkTransport])
async def test_feishu_encrypted_signature_challenge_owner_thread_and_delivery(
    tmp_path, monkeypatch, transport_type
):
    monkeypatch.setenv("ENCRYPT_KEY", "encrypt-secret")
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/auth/v3/tenant_access_token/internal"):
            assert json.loads(request.content) == {"app_id": "app", "app_secret": "app-secret"}
            return httpx.Response(
                200, json={"code": 0, "tenant_access_token": "tenant-secret", "expire": 7200}
            )
        if request.url.path.endswith("/bot/v3/info"):
            return httpx.Response(200, json={"code": 0, "bot": {"open_id": "bot"}})
        return httpx.Response(200, json={"code": 0})

    config = ChannelConfig(
        app_id="app",
        signing_secret_env="ENCRYPT_KEY",
        allowed_users=["tenant:owner"],
        allowed_channels=["chat"],
        allow_groups=True,
    )
    transport = transport_type(
        config=config,
        token="app-secret",
        app_token="verification-secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await transport.authenticate()
    store = ChannelStore(cwd=tmp_path, transport=transport.name)
    try:
        challenge = encrypt_feishu(
            {"type": "url_verification", "token": "verification-secret", "challenge": "challenge"}
        )
        assert await transport.handle_event({}, challenge, store) == {"challenge": "challenge"}
        assert store.claim_message() is None
        body = encrypt_feishu(feishu_event())
        with pytest.raises(AuthenticationError):
            await transport.handle_event({}, body, store)
        with pytest.raises(AuthenticationError):
            await transport.handle_event(feishu_headers(body, 1), body, store)
        with pytest.raises(AuthenticationError):
            await transport.handle_event(feishu_headers(body), body + b" ", store)
        await transport.handle_event(feishu_headers(body), body, store)
        await transport.handle_event(feishu_headers(body), body, store)
        message = store.claim_message()
        assert message is not None and message.text == "approve approval-id"
        assert message.user_id == "tenant:owner" and json.loads(message.thread_id) == [
            "chat",
            "message",
        ]
        assert store.claim_message() is None
        await transport.send(message.thread_id, "approved result", "delivery")
        assert requests[-1].url.path.endswith("/im/v1/messages/message/reply")
        assert json.loads(requests[-1].content)["reply_in_thread"] is True
        assert requests[-1].url.host == (
            "open.feishu.cn" if transport.name == "feishu" else "open.larksuite.com"
        )
    finally:
        store.close()
        await transport.client.aclose()
        await transport.close()


async def test_matrix_persisted_sync_ignores_history_deduplicates_and_sends_transactions(tmp_path):
    requests = []
    step = 0
    event = {
        "event_id": "$one",
        "sender": "@owner:matrix.org",
        "type": "m.room.message",
        "content": {
            "msgtype": "m.text",
            "body": "status",
            "m.mentions": {"user_ids": ["@bot:matrix.org"]},
            "m.relates_to": {"rel_type": "m.thread", "event_id": "$root"},
        },
    }

    def handler(request):
        nonlocal step
        requests.append(request)
        assert request.headers["authorization"] == "Bearer matrix-secret"
        if request.url.path.endswith("/whoami"):
            return httpx.Response(200, json={"user_id": "@bot:matrix.org", "device_id": "device"})
        if request.url.path.endswith("/sync"):
            step += 1
            if step > 1:
                assert request.url.params["since"] == f"cursor-{step - 1}"
            return httpx.Response(
                200,
                json={
                    "next_batch": f"cursor-{step}",
                    "rooms": {"join": {"!room:matrix.org": {"timeline": {"events": [event]}}}},
                },
            )
        if request.url.path.endswith("/state/m.room.encryption"):
            return httpx.Response(404, json={"errcode": "M_NOT_FOUND"})
        return httpx.Response(200, json={"event_id": "$sent"})

    transport = MatrixTransport(
        config=ChannelConfig(
            homeserver="https://matrix.org", allowed_users=["@owner:matrix.org"], allow_groups=True
        ),
        token="matrix-secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await transport.authenticate()
    store = ChannelStore(cwd=tmp_path, transport="matrix")
    try:
        await transport.receive(store)
        assert store.claim_message() is None
        await transport.receive(store)
        message = store.claim_message()
        assert message is not None and message.user_id == "@owner:matrix.org"
        assert json.loads(message.thread_id) == ["!room:matrix.org", "$root"]
        store.complete(message, "reply", limit=4000)
        store.close()
        store = ChannelStore(cwd=tmp_path, transport="matrix")
        await transport.receive(store)
        assert store.claim_message() is None
        await transport.send(message.thread_id, "reply", "transaction")
        await transport.send(message.thread_id, "reply", "transaction")
        sends = [r for r in requests if r.method == "PUT"]
        assert len(sends) == 2 and sends[0].url == sends[1].url
        assert json.loads(sends[0].content)["m.relates_to"]["event_id"] == "$root"
    finally:
        store.close()
        await transport.client.aclose()
        await transport.close()


async def test_matrix_media_is_bounded_platform_resource_and_never_plaintext_in_encrypted_room(
    tmp_path,
):
    encrypted = False
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("m.room.encryption"):
            return httpx.Response(200 if encrypted else 404, json={})
        if "/download/" in request.url.path:
            return httpx.Response(200, content=b"image")
        if request.url.path.endswith("/upload"):
            return httpx.Response(200, json={"content_uri": "mxc://matrix.org/new"})
        return httpx.Response(200, json={"event_id": "$sent"})

    transport = MatrixTransport(
        config=ChannelConfig(homeserver="https://matrix.org"),
        token="secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    thread = json.dumps(["!room:matrix.org", ""])
    message = ChannelMessage(
        id="$photo",
        user_id="owner",
        channel_id="room",
        thread_id=thread,
        text="photo",
        attachments=[
            {
                "url": "mxc://remote.example/file",
                "mime_type": "image/png",
                "name": "../../photo.png",
            }
        ],
    )
    try:
        [attachment] = await transport.prepare_media(message)
        assert (
            attachment.name == "photo.png" and base64.b64decode(attachment.data or "") == b"image"
        )
        assert requests[-1].url.host == "matrix.org"
        await transport.send_media(thread, attachment, "media-delivery")
        assert json.loads(requests[-1].content)["url"] == "mxc://matrix.org/new"
        bad = replace(message, attachments=[{"url": "https://private.example/secret"}])
        with pytest.raises(ChannelError):
            await transport.prepare_media(bad)
        encrypted = True
        before = len(requests)
        with pytest.raises(ChannelError, match="encrypted"):
            await transport.send(thread, "private plaintext", "forbidden")
        assert len(requests) == before + 1 and requests[-1].method == "GET"
    finally:
        await transport.client.aclose()
        await transport.close()
