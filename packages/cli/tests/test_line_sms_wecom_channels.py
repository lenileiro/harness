import base64
import hashlib
import hmac
import json
import struct
import time
from urllib.parse import parse_qs, urlencode

import httpx
import pytest
from aiohttp import web
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from harness.cli.channels.line import LINETransport
from harness.cli.channels.runtime import deliver_message, process_message
from harness.cli.channels.sms import SMSTransport, twilio_signature
from harness.cli.channels.transports import ChannelError
from harness.cli.channels.webhooks import AuthenticationError
from harness.cli.channels.wecom import WeComTransport, xml_fields
from harness.core.gateway_channels import ChannelConfig, ChannelStore


async def test_line_raw_signature_utf16_mention_owner_and_idempotent_reply(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("info"):
            return httpx.Response(200, json={"userId": "Ubot"})
        return httpx.Response(409, headers={"x-line-accepted-request-id": "previous"}, json={})

    transport = LINETransport(
        config=ChannelConfig(allowed_users=["Uowner"], allow_groups=True),
        token="access",
        app_token="channel-secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await transport.authenticate()
    store = ChannelStore(cwd=tmp_path, transport="line")
    event = {
        "destination": "Ubot",
        "events": [
            {
                "type": "message",
                "webhookEventId": "event",
                "source": {"type": "group", "groupId": "Cgroup", "userId": "Uowner"},
                "message": {
                    "id": "message",
                    "type": "text",
                    "text": "😀 @bot approve pending",
                    "mention": {
                        "mentionees": [{"type": "user", "userId": "Ubot", "index": 3, "length": 4}]
                    },
                },
            }
        ],
    }
    body = json.dumps(event).encode()
    signature = base64.b64encode(
        hmac.new(b"channel-secret", body, hashlib.sha256).digest()
    ).decode()
    with pytest.raises(AuthenticationError):
        await transport.handle_event({"X-Line-Signature": signature}, body + b" ", store)
    for _ in range(2):
        await transport.handle_event({"X-Line-Signature": signature}, body, store)
    message = store.claim_message()
    assert message and message.text == "😀  approve pending" and message.user_id == "Uowner"
    assert store.claim_message() is None
    await transport.send(message.thread_id, "Result", "delivery")
    first_key = requests[-1].headers["x-line-retry-key"]
    await transport.send(message.thread_id, "Result", "delivery")
    assert requests[-1].headers["x-line-retry-key"] == first_key
    assert json.loads(requests[-1].content)["to"] == "Cgroup"
    store.close()
    await transport.close()


async def test_sms_actual_form_listener_signature_account_number_and_scoped_reply(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        assert (
            request.headers["authorization"]
            == "Basic " + base64.b64encode(b"ACaccount:auth-secret").decode()
        )
        if request.method == "GET":
            return httpx.Response(200, json={"sid": "ACaccount"})
        return httpx.Response(201, json={"sid": "SMreply"})

    config = ChannelConfig(
        app_id="ACaccount",
        username="+15550000001",
        webhook_url="https://bot.example/sms?route=1",
        webhook_path="/sms",
        allowed_users=["+15550000002"],
    )
    transport = SMSTransport(
        config=config,
        token="auth-secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await transport.authenticate()
    store = ChannelStore(cwd=tmp_path, transport="sms")
    runner = web.AppRunner(transport.application(store), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    local = f"http://127.0.0.1:{runner.addresses[0][1]}/sms?route=1"
    data = {
        "AccountSid": "ACaccount",
        "To": "+15550000001",
        "From": "+15550000002",
        "MessageSid": "SMmessage",
        "Body": "approve pending",
        "FutureParameter": "included-in-signature",
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(local, data=data)
            assert response.status_code == 401
            signature = twilio_signature(config.webhook_url, data, "auth-secret")
            headers = {"X-Twilio-Signature": signature}
            for _ in range(2):
                response = await client.post(local, data=data, headers=headers)
                assert response.status_code == 200 and response.text.endswith("<Response/>")
            response = await client.post(
                local, data={**data, "To": "+15559999999"}, headers=headers
            )
            assert response.status_code == 401
            response = await client.post(
                local.replace("route=1", "route=2"), data=data, headers=headers
            )
            assert response.status_code == 401
            response = await client.post(
                local,
                content=urlencode(data) + "&From=%2B15559999999",
                headers={**headers, "Content-Type": "application/x-www-form-urlencoded"},
            )
            assert response.status_code == 401
        received = []

        async def receiver(**kwargs):
            received.append(kwargs)
            return {"reply": {"text": "Approved"}}

        assert await process_message(
            cwd=tmp_path, transport=transport, store=store, receiver=receiver
        )
        assert not await process_message(
            cwd=tmp_path, transport=transport, store=store, receiver=receiver
        )
        assert (
            received[0]["user_id"] == "+15550000002"
            and received[0]["thread_id"] == '["+15550000001","+15550000002"]'
        )
        assert await deliver_message(transport=transport, store=store)
        assert parse_qs(requests[-1].content.decode()) == {
            "To": ["+15550000002"],
            "From": ["+15550000001"],
            "Body": ["Approved"],
        }
        with pytest.raises(ChannelError, match="different"):
            await transport.send('["+15559999999","+15550000002"]', "private", "id")
    finally:
        await runner.cleanup()
        store.close()
        await transport.close()


def wecom_encrypted(body: bytes, *, corp="corp", key=b"k" * 32):
    plain = b"r" * 16 + struct.pack("!I", len(body)) + body + corp.encode()
    padding = 32 - len(plain) % 32
    plain += bytes([padding]) * padding
    encryptor = Cipher(algorithms.AES(key), modes.CBC(key[:16])).encryptor()
    return base64.b64encode(encryptor.update(plain) + encryptor.finalize()).decode()


def wecom_query(encrypted, timestamp=None):
    timestamp = str(int(time.time())) if timestamp is None else str(timestamp)
    nonce = "nonce"
    signature = hashlib.sha1(
        "".join(sorted(["callback-token", timestamp, nonce, encrypted])).encode()
    ).hexdigest()
    return {"timestamp": timestamp, "nonce": nonce, "msg_signature": signature}


async def test_wecom_actual_encrypted_challenge_callback_binding_and_token_refresh(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("WECOM_AES", base64.b64encode(b"k" * 32).decode().rstrip("="))
    requests = []
    tokens = 0
    sends = 0

    def handler(request):
        nonlocal tokens, sends
        requests.append(request)
        assert request.url.host == "qyapi.weixin.qq.com"
        if request.url.path.endswith("gettoken"):
            tokens += 1
            assert (
                request.url.params["corpid"] == "corp"
                and request.url.params["corpsecret"] == "app-secret"
            )
            return httpx.Response(
                200, json={"access_token": f"access-{tokens}", "expires_in": 7200}
            )
        if request.url.path.endswith("agent/get"):
            return httpx.Response(200, json={"agentid": 10001})
        sends += 1
        return httpx.Response(200, json={"errcode": 42001 if sends == 1 else 0})

    transport = WeComTransport(
        config=ChannelConfig(
            tenant_id="corp",
            app_id="10001",
            allowed_users=["corp:owner"],
            signing_secret_env="WECOM_AES",
        ),
        token="app-secret",
        app_token="callback-token",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await transport.authenticate()
    store = ChannelStore(cwd=tmp_path, transport="wecom")
    runner = web.AppRunner(transport.application(store), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    url = f"http://127.0.0.1:{runner.addresses[0][1]}/events"
    plain = b"<xml><ToUserName>corp</ToUserName><FromUserName>owner</FromUserName><AgentID>10001</AgentID><MsgType>text</MsgType><MsgId>message</MsgId><Content>approve pending</Content></xml>"
    try:
        async with httpx.AsyncClient() as client:
            echo = wecom_encrypted(b"challenge")
            response = await client.get(url, params={**wecom_query(echo), "echostr": echo})
            assert response.status_code == 200 and response.text == "challenge"
            encrypted = wecom_encrypted(plain)
            for _ in range(2):
                response = await client.post(
                    url,
                    params=wecom_query(encrypted),
                    content=f"<xml><Encrypt>{encrypted}</Encrypt></xml>",
                    headers={"Content-Type": "text/xml"},
                )
                assert response.status_code == 200 and response.text == "success"
            for bad in (
                wecom_encrypted(plain, corp="other"),
                wecom_encrypted(plain.replace(b"10001", b"10002")),
            ):
                response = await client.post(
                    url,
                    params=wecom_query(bad),
                    content=f"<xml><Encrypt>{bad}</Encrypt></xml>",
                    headers={"Content-Type": "text/xml"},
                )
                assert response.status_code == 401
            response = await client.get(url, params={**wecom_query(echo, 0), "echostr": echo})
            assert response.status_code == 401
        message = store.claim_message()
        assert (
            message
            and message.user_id == "corp:owner"
            and message.thread_id == '["corp","10001","owner"]'
        )
        assert store.claim_message() is None
        await transport.send(message.thread_id, "Result", "delivery")
        assert tokens == sends == 2
        body = json.loads(requests[-1].content)
        assert body["touser"] == "owner" and body["agentid"] == 10001
        assert requests[-1].url.params["access_token"] == "access-2"
        with pytest.raises(ChannelError):
            await transport.send('["corp","10001","@all"]', "private", "id")
    finally:
        await runner.cleanup()
        store.close()
        await transport.close()


def test_wecom_rejects_xml_entity_and_duplicate_fields():
    for xml in (
        b'<!DOCTYPE xml [<!ENTITY x "expand">]><xml><Content>&x;</Content></xml>',
        b"<xml><AgentID>1</AgentID><AgentID>2</AgentID></xml>",
    ):
        with pytest.raises(ValueError):
            xml_fields(xml)
