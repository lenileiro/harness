import asyncio
import json
import logging
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from harness.cli.channels.dingtalk import OPEN_CONNECTION, TOPIC, DingTalkTransport
from harness.cli.channels.runtime import deliver_message, process_message, run_transport
from harness.cli.channels.teams import OPENID, TeamsTransport
from harness.cli.channels.transports import ChannelError, Transport
from harness.cli.channels.webhooks import AuthenticationError
from harness.core.gateway_channels import ChannelConfig, ChannelStore


@pytest.fixture(scope="module")
def signing_key():
    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = json.loads(RSAAlgorithm.to_jwk(private.public_key()))
    public.update(kid="connector-key", alg="RS256", endorsements=["msteams"])
    return private, public


def teams_event(kind="channel"):
    return {
        "type": "message",
        "id": "message-1",
        "replyToId": "root-1",
        "channelId": "msteams",
        "serviceUrl": "https://smba.trafficmanager.net/emea/",
        "channelData": {"tenant": {"id": "tenant"}},
        "from": {"id": "29:owner", "aadObjectId": "owner"},
        "recipient": {"id": "28:app"},
        "conversation": {"id": "19:conversation", "conversationType": kind},
        "text": "<at>Bot</at> approve pending-id",
        "entities": [{"type": "mention", "mentioned": {"id": "28:app"}, "text": "<at>Bot</at>"}],
    }


def teams_token(signing_key, **overrides):
    claims = {
        "iss": "https://api.botframework.com",
        "aud": "app",
        "nbf": int(time.time()) - 1,
        "exp": int(time.time()) + 3600,
        "serviceurl": "https://smba.trafficmanager.net/emea/",
    }
    claims.update(overrides)
    return jwt.encode(claims, signing_key[0], algorithm="RS256", headers={"kid": "connector-key"})


async def make_teams(signing_key):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == "login.microsoftonline.com":
            assert b"client_secret=app-secret" in request.content
            return httpx.Response(200, json={"access_token": "access-secret", "expires_in": 3600})
        if str(request.url) == OPENID:
            return httpx.Response(
                200,
                json={
                    "jwks_uri": "https://login.botframework.com/v1/.well-known/keys",
                    "id_token_signing_alg_values_supported": ["RS256"],
                },
            )
        if request.url.host == "login.botframework.com":
            return httpx.Response(200, json={"keys": [signing_key[1]]})
        assert request.url.host == "smba.trafficmanager.net"
        assert request.headers["authorization"] == "Bearer access-secret"
        return httpx.Response(200, json={"id": "sent"})

    transport = TeamsTransport(
        config=ChannelConfig(
            app_id="app", tenant_id="tenant", allowed_users=["tenant:owner"], allow_groups=True
        ),
        token="app-secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await transport.authenticate()
    return transport, requests


async def test_teams_authenticated_thread_approval_dedup_and_restart_route(tmp_path, signing_key):
    transport, requests = await make_teams(signing_key)
    store = ChannelStore(cwd=tmp_path, transport="teams")
    headers = {"Authorization": "Bearer " + teams_token(signing_key)}
    body = json.dumps(teams_event()).encode()
    await transport.handle_event(headers, body, store)
    await transport.handle_event(headers, body, store)
    received = []

    async def receiver(**kwargs):
        received.append(kwargs)
        return {"reply": {"text": "Action approved"}}

    assert await process_message(cwd=tmp_path, transport=transport, store=store, receiver=receiver)
    assert not await process_message(
        cwd=tmp_path, transport=transport, store=store, receiver=receiver
    )
    assert received[0]["user_id"] == "tenant:owner"
    assert received[0]["thread_id"] == '["tenant","19:conversation","root-1"]'
    assert received[0]["message"] == "approve pending-id"
    store.close()
    store = ChannelStore(cwd=tmp_path, transport="teams")
    transport.store = store
    assert await deliver_message(transport=transport, store=store)
    outbound = requests[-1]
    assert outbound.url.path.endswith("/activities/root-1")
    assert json.loads(outbound.content)["text"] == "Action approved"
    assert all(row["status"] == "sent" for row in store.status()["outbox"])
    store.close()
    await transport.close()


@pytest.mark.parametrize(
    "change", ["audience", "issuer", "expired", "service", "host", "tenant", "endorsement"]
)
async def test_teams_rejects_untrusted_callbacks_before_routing(tmp_path, signing_key, change):
    transport, requests = await make_teams(signing_key)
    store = ChannelStore(cwd=tmp_path, transport="teams")
    event = teams_event()
    overrides = {}
    if change == "audience":
        overrides["aud"] = "another-app"
    if change == "issuer":
        overrides["iss"] = "attacker"
    if change == "expired":
        overrides["exp"] = int(time.time()) - 1000
    if change == "service":
        event["serviceUrl"] = "https://smba.trafficmanager.net/other/"
    if change == "host":
        event["serviceUrl"] = "https://attacker.example/"
        overrides["serviceurl"] = event["serviceUrl"]
    if change == "tenant":
        event["channelData"]["tenant"]["id"] = "other"
    if change == "endorsement":
        transport.keys = {"connector-key": {**signing_key[1], "endorsements": ["other"]}}
        transport.keys_fetched = time.time()
    with pytest.raises(AuthenticationError):
        await transport.handle_event(
            {"Authorization": "Bearer " + teams_token(signing_key, **overrides)},
            json.dumps(event).encode(),
            store,
        )
    assert store.claim_message() is None
    assert not any(request.url.host == "attacker.example" for request in requests)
    store.close()
    await transport.close()


async def test_teams_group_chat_stable_and_identity_allowlist(tmp_path, signing_key):
    transport, _ = await make_teams(signing_key)
    store = ChannelStore(cwd=tmp_path, transport="teams")
    headers = {"Authorization": "Bearer " + teams_token(signing_key)}
    for identifier in ("one", "two"):
        event = teams_event("groupChat")
        event["id"] = identifier
        event.pop("replyToId")
        await transport.handle_event(headers, json.dumps(event).encode(), store)
    first, second = store.claim_message(), store.claim_message()
    assert (
        first
        and second
        and first.thread_id == second.thread_id == '["tenant","19:conversation",""]'
    )
    event["from"]["aadObjectId"] = "stranger"
    await transport.handle_event(headers, json.dumps(event).encode(), store)
    assert store.claim_message() is None
    store.close()
    await transport.close()


def ding_event():
    return {
        "msgtype": "text",
        "msgId": "one",
        "senderCorpId": "corp",
        "senderStaffId": "owner",
        "conversationId": "group",
        "conversationType": "2",
        "isInAtList": True,
        "text": {"content": "approve pending-id"},
        "sessionWebhook": "https://oapi.dingtalk.com/robot/sendBySession?session=capability-secret",
        "sessionWebhookExpiredTime": int((time.time() + 3600) * 1000),
    }


async def test_dingtalk_real_socket_durable_ack_dedup_reply_and_secret_redaction(tmp_path):
    store = ChannelStore(cwd=tmp_path, transport="dingtalk")
    frames, requests = [], []

    async def server(socket):
        for index in range(2):
            await socket.send(
                json.dumps(
                    {
                        "type": "CALLBACK",
                        "headers": {"messageId": str(index), "topic": TOPIC},
                        "data": json.dumps(ding_event()),
                    }
                )
            )
            frames.append(json.loads(await socket.recv()))
            assert len(store.status()["inbox"]) == 1  # Ack follows the durable write.
        await socket.send(
            json.dumps(
                {
                    "type": "SYSTEM",
                    "headers": {"messageId": "bye", "topic": "disconnect"},
                    "data": json.dumps({"reason": "reconnect"}),
                }
            )
        )
        frames.append(json.loads(await socket.recv()))

    def handler(request):
        requests.append(request)
        if str(request.url) == OPEN_CONNECTION:
            data = json.loads(request.content)
            assert data["clientId"] == "app" and data["clientSecret"] == "app-secret"
            assert data["subscriptions"] == [{"type": "CALLBACK", "topic": TOPIC}]
            return httpx.Response(
                200,
                json={"endpoint": "wss://stream.dingtalk.com/connect", "ticket": "ticket-secret"},
            )
        assert request.url.host == "oapi.dingtalk.com"
        return httpx.Response(200, json={"errcode": 0})

    async with serve(server, "127.0.0.1", 0) as local:
        address = f"ws://127.0.0.1:{local.sockets[0].getsockname()[1]}"

        def socket_connect(url, **kwargs):
            assert url == "wss://stream.dingtalk.com/connect?ticket=ticket-secret"
            return connect(address, **kwargs)

        transport = DingTalkTransport(
            config=ChannelConfig(app_id="app", allowed_users=["corp:owner"], allow_groups=True),
            token="app-secret",
            client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
            socket_connect=socket_connect,
        )
        await transport.authenticate()
        await asyncio.wait_for(transport.receive(store), timeout=5)
    assert frames[0]["code"] == 200 and json.loads(frames[-1]["data"]) == {"reason": "reconnect"}
    message = store.claim_message()
    assert message and message.thread_id == '["corp","group"]'
    await transport.send(message.thread_id, "Approved", "delivery")
    assert json.loads(requests[-1].content)["text"]["content"] == "Approved"
    record = logging.LogRecord(
        "harness.channels.websocket",
        logging.DEBUG,
        "",
        1,
        "GET /connect?ticket=ticket-secret app-secret session=capability-secret",
        (),
        None,
    )
    transport.log_filter.filter(record)
    assert "secret" not in record.getMessage()
    store.close()
    await transport.close()


async def test_dingtalk_disallows_untrusted_capability_and_expired_delivery(tmp_path):
    store = ChannelStore(cwd=tmp_path, transport="dingtalk")
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={})

    transport = DingTalkTransport(
        config=ChannelConfig(app_id="app", allowed_users=["corp:owner"], allow_groups=True),
        token="secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    transport.store = store
    event = ding_event()
    event["sessionWebhook"] = "https://attacker.example/robot/sendBySession"
    with pytest.raises(ChannelError):
        transport._ingest(event, store)
    assert store.claim_message() is None
    event = ding_event()
    event["sessionWebhookExpiredTime"] = 0
    transport._ingest(event, store)
    with pytest.raises(ChannelError, match="expired"):
        await transport.send('["corp","group"]', "private reply", "delivery")
    assert requests == []
    store.close()
    await transport.close()


async def test_channel_lock_excludes_second_runner_and_releases_on_cancellation(tmp_path):
    from filelock import FileLock

    transport = DingTalkTransport(
        config=ChannelConfig(app_id="app", allowed_users=["owner"]), token="fake"
    )
    store = ChannelStore(cwd=tmp_path, transport="dingtalk")
    lock = FileLock(store.path.with_suffix(".lock"), timeout=0)
    with lock, pytest.raises(ChannelError, match="already running"):
        await run_transport(cwd=tmp_path, transport=transport)
    # Failure must not release the first runner's independent lock.
    with lock:
        assert lock.is_locked
    store.close()


async def test_channel_safe_connection_diagnostic_redacts_and_clears(tmp_path):
    class FakeTransport(Transport):
        name = "telegram"

        async def authenticate(self):
            self.identity = "bot"

        async def receive(self, store):
            raise ChannelError("Session expired; pair again (credential=private-token)")

    transport = FakeTransport(config=ChannelConfig(allowed_users=["owner"]), token="private-token")
    task = asyncio.create_task(run_transport(cwd=tmp_path, transport=transport))
    store = ChannelStore(cwd=tmp_path, transport="telegram")
    try:
        for _ in range(100):
            if store.status()["connection_error"]:
                break
            await asyncio.sleep(0.01)
        assert (
            store.status()["connection_error"]
            == "Session expired; pair again (credential=[REDACTED])"
        )
        from harness.core.gateway_channels import ChannelMessage

        transport.accept(
            ChannelMessage(
                id="resumed", user_id="owner", channel_id="chat", thread_id="chat", text="hello"
            ),
            store,
        )
        assert store.status()["connection_error"] == ""
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        store.close()
