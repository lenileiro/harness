"""Offline protocol contracts and real loopback webhook/durable queue checks."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
from contextlib import asynccontextmanager
from dataclasses import replace

import httpx
import pytest
from aiohttp import web

from harness.cli.channels.bluebubbles import BlueBubblesTransport
from harness.cli.channels.msgraph_webhook import MSGraphWebhookTransport
from harness.cli.channels.runtime import deliver_message, process_message
from harness.cli.channels.transports import ChannelError
from harness.cli.channels.webhooks import AuthenticationError
from harness.cli.channels.whatsapp_cloud import WhatsAppCloudTransport
from harness.core.gateway_channels import ChannelConfig, ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment


@asynccontextmanager
async def listener(transport, store):
    runner = web.AppRunner(transport.application(store), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    try:
        async with httpx.AsyncClient(trust_env=False) as client:
            yield client, f"http://127.0.0.1:{runner.addresses[0][1]}/events"
    finally:
        await runner.cleanup()


async def whatsapp(monkeypatch, handler=None):
    requests = []

    def api(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer access-token"
        assert request.url.host == "graph.facebook.com"
        if request.method == "GET" and request.url.path == "/v23.0/123":
            return httpx.Response(200, json={"id": "123"})
        if handler:
            return handler(request)
        return httpx.Response(200, json={"messages": [{"id": "wamid.sent"}]})

    monkeypatch.setenv("TEST_META_SECRET", "meta-secret")
    transport = WhatsAppCloudTransport(
        config=ChannelConfig(
            phone_number_id="123",
            waba_id="456",
            api_version="v23.0",
            signing_secret_env="TEST_META_SECRET",
            allowed_users=["15550001"],
        ),
        token="access-token",
        app_token="verification-token",
        client=httpx.AsyncClient(transport=httpx.MockTransport(api)),
    )
    await transport.authenticate()
    return transport, requests


def wa_event(**message_overrides):
    return {
        "object": "whatsapp_business_account",
        "entry": [
            {
                "id": "456",
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {"phone_number_id": "123"},
                            "messages": [
                                {
                                    "id": "wamid.one",
                                    "from": "15550001",
                                    "type": "text",
                                    "text": {"body": "hello"},
                                    **message_overrides,
                                }
                            ],
                        },
                    }
                ],
            }
        ],
    }


def signed(event):
    body = json.dumps(event, separators=(",", ":")).encode()
    return body, {
        "Content-Type": "application/json",
        "X-Hub-Signature-256": "sha256="
        + hmac.new(b"meta-secret", body, hashlib.sha256).hexdigest(),
    }


async def test_whatsapp_signed_listener_durability_and_delivery(tmp_path, monkeypatch):
    transport, requests = await whatsapp(monkeypatch)
    store = ChannelStore(cwd=tmp_path, transport="whatsapp_cloud")
    store.bind_identity(transport.identity)
    body, headers = signed(wa_event())
    try:
        async with listener(transport, store) as (client, url):
            validation = await client.get(
                url,
                params={
                    "hub.mode": "subscribe",
                    "hub.verify_token": "verification-token",
                    "hub.challenge": "challenge",
                },
            )
            assert validation.status_code == 200 and validation.text == "challenge"
            assert (
                await client.get(
                    url,
                    params={
                        "hub.mode": "subscribe",
                        "hub.verify_token": "wrong",
                        "hub.challenge": "challenge",
                    },
                )
            ).status_code == 403
            assert (await client.post(url, content=body + b" ", headers=headers)).status_code == 401
            assert not store.status()["inbox"]
            assert (await client.post(url, content=body, headers=headers)).status_code == 200
            assert (await client.post(url, content=body, headers=headers)).status_code == 200
        store.close()
        store = ChannelStore(cwd=tmp_path, transport="whatsapp_cloud")
        assert len(store.status()["inbox"]) == 1
        seen = []

        async def receiver(**kwargs):
            seen.append(kwargs)
            return {"reply": {"text": "reply"}}

        assert await process_message(
            cwd=tmp_path, transport=transport, store=store, receiver=receiver
        )
        assert seen[0]["transport"] == "whatsapp_cloud" and seen[0]["user_id"] == "15550001"
        assert await deliver_message(transport=transport, store=store)
        sent = json.loads(requests[-1].content)
        assert requests[-1].url.path == "/v23.0/123/messages"
        assert sent["to"] == "15550001" and sent["text"] == {"body": "reply", "preview_url": False}
        assert sent["biz_opaque_callback_data"] == store.status()["outbox"][0]["id"]
        assert store.status()["outbox"][0]["status"] == "sent"
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


@pytest.mark.parametrize("field", ["account", "phone", "product"])
async def test_whatsapp_rejects_cross_account_batches_before_ingestion(
    tmp_path, monkeypatch, field
):
    transport, _ = await whatsapp(monkeypatch)
    store = ChannelStore(cwd=tmp_path, transport="whatsapp_cloud")
    event = wa_event()
    other = wa_event(id="wamid.two")["entry"][0]
    if field == "account":
        other["id"] = "999"
    elif field == "phone":
        other["changes"][0]["value"]["metadata"]["phone_number_id"] = "999"
    else:
        other["changes"][0]["value"]["messaging_product"] = "other"
    event["entry"].append(other)
    body, headers = signed(event)
    try:
        with pytest.raises(AuthenticationError):
            await transport.handle_event(headers, body, store)
        assert not store.status()["inbox"]
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


async def test_whatsapp_denied_sender_and_status_events_never_dispatch(tmp_path, monkeypatch):
    transport, _ = await whatsapp(monkeypatch)
    store = ChannelStore(cwd=tmp_path, transport="whatsapp_cloud")
    try:
        event = wa_event(**{"from": "999"})
        body, headers = signed(event)
        await transport.handle_event(headers, body, store)
        event["entry"][0]["changes"][0]["value"].pop("messages")
        body, headers = signed(event)
        await transport.handle_event(headers, body, store)
        assert store.claim_message() is None
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


@pytest.mark.parametrize("status,expected", [(429, "pending"), (500, "uncertain")])
async def test_whatsapp_outbox_preserves_retry_and_uncertain_outcomes(
    tmp_path, monkeypatch, status, expected
):
    transport, _ = await whatsapp(
        monkeypatch,
        lambda _: httpx.Response(status, json={"error": {}}, headers={"Retry-After": "2"}),
    )
    store = ChannelStore(cwd=tmp_path, transport="whatsapp_cloud")
    try:
        store.queue(source="one", user_id="15550001", thread_id="15550001", text="hi", limit=4096)
        assert await deliver_message(transport=transport, store=store)
        assert store.status()["outbox"][0]["status"] == expected
        assert not await deliver_message(transport=transport, store=store)
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


async def test_whatsapp_media_cannot_redirect_bearer_and_upload_uses_media_id(monkeypatch):
    transport, requests = await whatsapp(monkeypatch)
    urls = []

    def api(request):
        urls.append(str(request.url))
        if request.url.path == "/v23.0/789":
            return httpx.Response(200, json={"url": "https://attacker.example/steal"})
        if request.url.path == "/v23.0/123/media":
            assert b"PNG" in request.content and b"messaging_product" in request.content
            return httpx.Response(200, json={"id": "789"})
        assert json.loads(request.content)["image"] == {"id": "789"}
        return httpx.Response(200, json={"messages": [{"id": "sent"}]})

    await transport.client.aclose()
    transport.client = httpx.AsyncClient(transport=httpx.MockTransport(api))
    try:
        message = ChannelMessage(
            id="m",
            user_id="15550001",
            thread_id="15550001",
            channel_id="15550001",
            text="image",
            attachments=[{"id": "789", "mime_type": "image/png", "name": "photo"}],
        )
        with pytest.raises(ValueError, match="outside"):
            await transport.prepare_media(message)
        assert len(urls) == 1
        await transport.send_media(
            "15550001",
            MediaAttachment(
                kind="image", mime_type="image/png", data=base64.b64encode(b"PNG").decode()
            ),
            "delivery",
        )
        assert len(urls) == 3 and requests[0].url.path == "/v23.0/123"
    finally:
        await transport.close()
        await transport.client.aclose()


async def bluebubbles(*, config=None, handler=None):
    requests = []

    def api(request):
        requests.append(request)
        assert request.url.params["password"] == "server & password"
        if request.url.path == "/api/v1/server/info":
            return httpx.Response(
                200,
                json={
                    "status": 200,
                    "data": {"computer_id": "computer-1", "detected_imessage": "bot@example.test"},
                },
            )
        if handler:
            return handler(request)
        return httpx.Response(200, json={"status": 200, "data": {"guid": "sent-guid"}})

    transport = BlueBubblesTransport(
        config=config
        or ChannelConfig(homeserver="http://127.0.0.1:1234", allowed_users=["+15550001"]),
        token="server & password",
        app_token="webhook-secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(api)),
    )
    await transport.authenticate()
    return transport, requests


def bb_event(**overrides):
    return {
        "type": "new-message",
        "data": {
            "guid": "message-guid",
            "isFromMe": False,
            "associatedMessageType": None,
            "text": "hello",
            "handle": {"address": "+15550001"},
            "chats": [{"guid": "iMessage;-;+15550001"}],
            **overrides,
        },
    }


async def test_bluebubbles_query_auth_durable_identity_and_exact_reply(tmp_path):
    transport, requests = await bluebubbles()
    store = ChannelStore(cwd=tmp_path, transport="bluebubbles")
    store.bind_identity(transport.identity)
    try:
        async with listener(transport, store) as (client, url):
            assert (await client.post(url, json=bb_event())).status_code == 401
            assert (await client.post(url + "?token=wrong", json=bb_event())).status_code == 401
            assert (
                await client.post(
                    url + "?token=webhook-secret&token=webhook-secret", json=bb_event()
                )
            ).status_code == 401
            assert store.claim_message() is None
            assert (
                await client.post(url + "?token=webhook-secret", json=bb_event())
            ).status_code == 200
            assert (
                await client.post(url + "?token=webhook-secret", json=bb_event())
            ).status_code == 200
        assert len(store.status()["inbox"]) == 1
        message = store.claim_message()
        assert message is not None
        store.complete(message, "hi", limit=4000)
        assert await deliver_message(transport=transport, store=store)
        sent = json.loads(requests[-1].content)
        assert sent["chatGuid"] == "iMessage;-;+15550001"
        assert sent["tempGuid"].startswith("harness-")
        with pytest.raises(ValueError, match="another bot"):
            store.bind_identity("different account")
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


async def test_bluebubbles_filters_self_receipts_and_requires_explicit_group_mention(tmp_path):
    config = ChannelConfig(
        homeserver="http://localhost:1234",
        allowed_users=["+15550001"],
        allow_groups=True,
        username="assistant",
    )
    transport, _ = await bluebubbles(config=config)
    store = ChannelStore(cwd=tmp_path, transport="bluebubbles")
    headers = {"X-BlueBubbles-Token": "webhook-secret"}
    try:
        for event in [
            bb_event(isFromMe=True),
            bb_event(associatedMessageType=2001),
            {**bb_event(), "type": "updated-message"},
            bb_event(chats=[{"guid": "iMessage;+;group"}]),
            bb_event(text="@assistantx hello", chats=[{"guid": "iMessage;+;group"}]),
        ]:
            await transport.handle_event(headers, json.dumps(event).encode(), store)
        assert store.claim_message() is None
        event = bb_event(text="@assistant hello", chats=[{"guid": "iMessage;+;group"}])
        await transport.handle_event(headers, json.dumps(event).encode(), store)
        message = store.claim_message()
        assert message is not None and message.group and message.thread_id == "iMessage;+;group"
        with pytest.raises(ValueError, match="does not match"):
            await transport.handle_event(
                headers,
                json.dumps(bb_event(chats=[{"guid": "iMessage;-;+19999999"}])).encode(),
                store,
            )
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


@pytest.mark.parametrize(
    "base",
    [
        "http://remote.example",
        "https://user:secret@example.com",
        "https://example.com/?password=secret",
    ],
)
async def test_bluebubbles_rejects_insecure_or_credentialed_endpoint(base):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: pytest.fail("no request expected"))
    )
    transport = BlueBubblesTransport(
        config=ChannelConfig(homeserver=base), token="token", app_token="secret", client=client
    )
    try:
        with pytest.raises(ChannelError, match="HTTPS"):
            await transport.authenticate()
    finally:
        await transport.close()
        await client.aclose()


@pytest.mark.parametrize("existing", [False, True])
async def test_bluebubbles_registration_cleanup_only_deletes_owned(tmp_path, existing):
    url = "http://localhost:8766/events?token=webhook-secret"
    ready = asyncio.Event()

    def api(request):
        if request.method == "GET":
            ready.set() if existing else None
            return httpx.Response(
                200,
                json={
                    "status": 200,
                    "data": [{"id": 7, "url": url, "events": ["new-message"]}] if existing else [],
                },
            )
        if request.method == "POST":
            assert json.loads(request.content) == {"url": url, "events": ["new-message"]}
            ready.set()
        return httpx.Response(200, json={"status": 200, "data": {"id": 7}})

    config = ChannelConfig(
        homeserver="http://localhost:1234",
        webhook_url="http://localhost:8766/events",
        listen_port=0,
    )
    transport, requests = await bluebubbles(config=config, handler=api)
    store = ChannelStore(cwd=tmp_path, transport="bluebubbles")
    task = asyncio.create_task(transport.receive(store))
    try:
        await asyncio.wait_for(ready.wait(), 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        deletes = [request for request in requests if request.method == "DELETE"]
        assert len(deletes) == (0 if existing else 1)
        if deletes:
            assert deletes[0].url.path == "/api/v1/webhook/7"
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        store.close()
        await transport.close()
        await transport.client.aclose()


async def test_bluebubbles_cancel_during_registration_recovers_late_owned_id(tmp_path):
    transport, requests = await bluebubbles(
        config=ChannelConfig(
            homeserver="http://localhost:1234",
            webhook_url="http://localhost:8766/events",
            listen_port=0,
        )
    )
    await transport.client.aclose()
    started, release = asyncio.Event(), asyncio.Event()

    async def api(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json={"status": 200, "data": []})
        if request.method == "POST":
            started.set()
            await release.wait()
        return httpx.Response(200, json={"status": 200, "data": {"id": 9}})

    transport.client = httpx.AsyncClient(transport=httpx.MockTransport(api))
    store = ChannelStore(cwd=tmp_path, transport="bluebubbles")
    task = asyncio.create_task(transport.receive(store))
    try:
        await asyncio.wait_for(started.wait(), 2)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert requests[-1].method == "DELETE" and requests[-1].url.path.endswith("/9")
    finally:
        release.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        store.close()
        await transport.close()
        await transport.client.aclose()


async def graph():
    transport = MSGraphWebhookTransport(
        config=ChannelConfig(
            tenant_id="tenant-one",
            subscription_id="sub-one",
            accepted_resources=["users/owner/messages"],
            allowed_users=["tenant-one:sub-one"],
        ),
        token="a-private-client-state-token-of-32-bytes",
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: pytest.fail("Graph ingress must not make outbound requests")
            )
        ),
    )
    await transport.authenticate()
    return transport


def graph_event(**overrides):
    return {
        "value": [
            {
                "clientState": "a-private-client-state-token-of-32-bytes",
                "tenantId": "tenant-one",
                "subscriptionId": "sub-one",
                "changeType": "created",
                "resource": "users/owner/messages/m1",
                "resourceData": {"id": "m1", "@odata.type": "#Microsoft.Graph.Message"},
                **overrides,
            }
        ]
    }


async def test_graph_validation_secret_scoping_and_durable_local_response(tmp_path):
    transport = await graph()
    store = ChannelStore(cwd=tmp_path, transport="msgraph_webhook")
    transport.store = store
    try:
        async with listener(transport, store) as (client, url):
            validation = await client.post(
                url + "?validationToken=unencoded%20token", headers={"Content-Type": "text/plain"}
            )
            assert validation.status_code == 200 and validation.text == "unencoded token"
            assert store.claim_message() is None
            assert (
                await client.post(url, json=graph_event(clientState="wrong"))
            ).status_code == 401
            assert (await client.post(url, json=graph_event())).status_code == 202
            assert (await client.post(url, json=graph_event())).status_code == 202
        assert len(store.status()["inbox"]) == 1
        message = store.claim_message()
        assert message is not None and transport.token not in message.text
        store.complete(message, "notification received", limit=4000)
        assert await deliver_message(transport=transport, store=store)
        assert "local_responses" not in store.status()
        private = store.status(include_private=True)
        assert private["local_responses"][0]["text"] == "notification received"
        assert private["local_responses"][0]["destination"] == "local"
        store.close()
        store = ChannelStore(cwd=tmp_path, transport="msgraph_webhook")
        assert store.get("local_responses")[0]["destination"] == "local"
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


@pytest.mark.parametrize(
    "overrides",
    [
        {"tenantId": "other"},
        {"subscriptionId": "other"},
        {"resource": "users/owner/messages-elsewhere/m1"},
        {"resource": "users/owner/messages/../private"},
        {"resource": "https://attacker.example"},
    ],
)
async def test_graph_rejects_out_of_scope_batch_without_partial_ingestion(tmp_path, overrides):
    transport = await graph()
    store = ChannelStore(cwd=tmp_path, transport="msgraph_webhook")
    event = graph_event()
    event["value"].append(graph_event(**overrides)["value"][0])
    try:
        with pytest.raises((AuthenticationError, ValueError)):
            await transport.handle_event({}, json.dumps(event).encode(), store)
        assert not store.status()["inbox"]
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


async def test_graph_rich_notifications_rejected_and_unversioned_updates_not_lost(tmp_path):
    transport = await graph()
    store = ChannelStore(cwd=tmp_path, transport="msgraph_webhook")
    try:
        with pytest.raises(ValueError, match="rich"):
            await transport.handle_event(
                {},
                json.dumps({**graph_event(), "validationTokens": ["unvalidated"]}).encode(),
                store,
            )
        for _ in range(2):
            await transport.handle_event(
                {}, json.dumps(graph_event(changeType="updated")).encode(), store
            )
        assert len(store.status()["inbox"]) == 2
        lifecycle = graph_event(lifecycleEvent="reauthorizationRequired")
        await transport.handle_event({}, json.dumps(lifecycle).encode(), store)
        assert store.status()["subscription_lifecycle"]["event"] == "reauthorizationRequired"
        assert len(store.status()["inbox"]) == 2
        assert transport.token not in json.dumps(store.status(include_private=True))
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


async def test_graph_missing_scope_or_weak_secret_refuses_start():
    transport = await graph()
    try:
        transport.config = replace(transport.config, accepted_resources=[])
        with pytest.raises(ChannelError, match="accepted_resources"):
            await transport.authenticate()
    finally:
        await transport.close()
        await transport.client.aclose()


@pytest.mark.parametrize("kind", ["whatsapp", "bluebubbles"])
async def test_media_transfers_use_platform_bytes_and_bounded_inline_uploads(monkeypatch, kind):
    transport, _ = await (whatsapp(monkeypatch) if kind == "whatsapp" else bluebubbles())
    await transport.client.aclose()
    requests = []

    def api(request):
        requests.append(request)
        if request.method == "GET":
            if request.url.path == "/v23.0/789":
                return httpx.Response(
                    200,
                    json={
                        "url": "https://lookaside.fbsbx.com/whatsapp_business/attachments/?mid=789"
                    },
                )
            if kind == "whatsapp":
                assert request.url.host == "lookaside.fbsbx.com"
                assert request.headers["Authorization"] == "Bearer access-token"
            else:
                assert request.url.path == "/api/v1/attachment/attachment-guid/download"
                assert request.url.params["password"] == "server & password"
            return httpx.Response(200, content=b"binary\x00data")
        if request.url.path.endswith("/messages"):
            assert json.loads(request.content)["document"] == {
                "id": "789",
                "filename": "report.txt",
            }
            return httpx.Response(200, json={"messages": [{"id": "sent"}]})
        assert b"binary\x00data" in request.content and b"report.txt" in request.content
        return httpx.Response(
            200,
            json={"id": "789"}
            if kind == "whatsapp"
            else {"status": 200, "data": {"guid": "sent-guid"}},
        )

    transport.client = httpx.AsyncClient(transport=httpx.MockTransport(api))
    descriptor = {"id": "789"} if kind == "whatsapp" else {"guid": "attachment-guid"}
    message = ChannelMessage(
        id="m",
        user_id="u",
        channel_id="c",
        thread_id="t",
        text="file",
        attachments=[{**descriptor, "mime_type": "text/plain", "name": "../report.txt"}],
    )
    target = "15550001" if kind == "whatsapp" else "iMessage;-;+15550001"
    try:
        prepared = await transport.prepare_media(message)
        assert len(prepared) == 1 and prepared[0].name == "report.txt"
        assert prepared[0].data and base64.b64decode(prepared[0].data) == b"binary\x00data"
        await transport.send_media(target, prepared[0], "delivery")
        before = len(requests)
        with pytest.raises(ChannelError, match="inline"):
            await transport.send_media(
                target,
                MediaAttachment(
                    kind="file", mime_type="text/plain", url="https://attacker.example"
                ),
                "delivery",
            )
        assert len(requests) == before
    finally:
        await transport.close()
        await transport.client.aclose()


@pytest.mark.parametrize("kind", ["whatsapp", "bluebubbles"])
async def test_media_redirects_never_forward_platform_credentials(monkeypatch, kind):
    transport, _ = await (whatsapp(monkeypatch) if kind == "whatsapp" else bluebubbles())
    await transport.client.aclose()
    requests = []

    def api(request):
        requests.append(request)
        assert request.url.host != "attacker.example"
        if request.url.path == "/v23.0/789":
            return httpx.Response(200, json={"url": "https://lookaside.fbsbx.com/media"})
        return httpx.Response(302, headers={"Location": "https://attacker.example"})

    transport.client = httpx.AsyncClient(transport=httpx.MockTransport(api), follow_redirects=True)
    descriptor = {"id": "789"} if kind == "whatsapp" else {"guid": "attachment-guid"}
    message = ChannelMessage(
        id="m",
        user_id="u",
        channel_id="c",
        thread_id="t",
        text="file",
        attachments=[{**descriptor, "mime_type": "text/plain", "name": "file"}],
    )
    try:
        with pytest.raises(ChannelError, match="download rejected"):
            await transport.prepare_media(message)
        assert len(requests) == (2 if kind == "whatsapp" else 1)
    finally:
        await transport.close()
        await transport.client.aclose()


async def test_bluebubbles_crash_restart_recognizes_owned_registration(tmp_path):
    records = []

    def api(request):
        if request.method == "GET":
            return httpx.Response(200, json={"status": 200, "data": list(records)})
        data = {**json.loads(request.content), "id": 17}
        records.append(data)
        return httpx.Response(200, json={"status": 200, "data": data})

    transport, requests = await bluebubbles(
        config=ChannelConfig(
            homeserver="http://localhost:1234", webhook_url="http://localhost:8766/events"
        ),
        handler=api,
    )
    store = ChannelStore(cwd=tmp_path, transport="bluebubbles")
    try:
        assert await transport.register(store) == "17"
        store.close()
        store = ChannelStore(cwd=tmp_path, transport="bluebubbles")
        assert await transport.register(store) == "17"
        assert sum(request.method == "POST" for request in requests) == 1
        assert "webhook-secret" not in json.dumps(store.get("webhook_registration"))
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


@pytest.mark.parametrize("kind", ["whatsapp", "bluebubbles"])
async def test_platform_media_stream_cap_stops_before_unbounded_buffering(monkeypatch, kind):
    transport, _ = await (whatsapp(monkeypatch) if kind == "whatsapp" else bluebubbles())
    await transport.client.aclose()
    counts = []

    class LargeStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            for index in range(100):
                counts.append(index)
                yield b"a" * (1024 * 1024)

    def api(request):
        if request.url.path == "/v23.0/789":
            return httpx.Response(200, json={"url": "https://lookaside.fbsbx.com/media"})
        return httpx.Response(200, stream=LargeStream())

    transport.client = httpx.AsyncClient(transport=httpx.MockTransport(api))
    descriptor = {"id": "789"} if kind == "whatsapp" else {"guid": "attachment-guid"}
    message = ChannelMessage(
        id="m",
        user_id="u",
        channel_id="c",
        thread_id="t",
        text="file",
        attachments=[{**descriptor, "mime_type": "text/plain", "name": "file"}],
    )
    try:
        with pytest.raises(ChannelError, match="20 MiB"):
            await transport.prepare_media(message)
        assert len(counts) == 21
    finally:
        await transport.close()
        await transport.client.aclose()
