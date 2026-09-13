from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import replace
from datetime import UTC, datetime

import httpx
import pytest
import typer
from cryptography.hazmat.primitives import serialization
from typer.testing import CliRunner
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from harness.cli.channel_weixin_commands import register_weixin_commands, save_credentials
from harness.cli.channels.qqbot import QQBotTransport
from harness.cli.channels.runtime import build_transport, deliver_message
from harness.cli.channels.transports import ChannelError, RateLimited
from harness.cli.channels.webhooks import AuthenticationError
from harness.cli.channels.weixin import WeixinTransport, crypt_media
from harness.cli.channels.weixin_pairing import WeixinPairing
from harness.core.gateway_channels import ChannelConfig, ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment


def wx_message(**overrides):
    return {
        "message_type": 1,
        "message_state": 2,
        "message_id": 1,
        "seq": 1,
        "from_user_id": "owner@im.wechat",
        "to_user_id": "bot@im.bot",
        "context_token": "private-peer-token",
        "item_list": [{"type": 1, "text_item": {"text": "hello"}}],
        **overrides,
    }


async def weixin(handler=None):
    requests = []

    def api(request):
        requests.append(request)
        assert request.headers["Authorization"] == "Bearer bot-token"
        assert request.headers["AuthorizationType"] == "ilink_bot_token"
        assert base64.b64decode(request.headers["X-WECHAT-UIN"]).isdigit()
        if request.url.path.endswith("getconfig"):
            return httpx.Response(200, json={"ret": 0})
        if handler:
            return handler(request)
        if request.url.path.endswith("getupdates"):
            return httpx.Response(
                200, json={"ret": 0, "msgs": [wx_message()], "get_updates_buf": "cursor-one"}
            )
        return httpx.Response(200, json={"ret": 0})

    transport = WeixinTransport(
        config=ChannelConfig(app_id="bot@im.bot", allowed_users=["owner@im.wechat"]),
        token="bot-token",
        client=httpx.AsyncClient(transport=httpx.MockTransport(api)),
    )
    await transport.authenticate()
    return transport, requests


async def test_weixin_cursor_and_peer_context_survive_restart_without_transcript_secret(tmp_path):
    transport, requests = await weixin()
    store = ChannelStore(cwd=tmp_path, transport="weixin")
    try:
        await transport.poll_once(store)
        store.close()
        store = ChannelStore(cwd=tmp_path, transport="weixin")
        transport.store = store
        await transport.poll_once(store)
        assert json.loads(requests[-1].content)["get_updates_buf"] == "cursor-one"
        assert len(store.status()["inbox"]) == 1
        message = store.claim_message()
        assert message is not None and "private-peer-token" not in message.text
        store.complete(message, "reply", limit=2000)
        assert await deliver_message(transport=transport, store=store)
        payload = json.loads(requests[-1].content)["msg"]
        assert payload["context_token"] == "private-peer-token"
        assert payload["to_user_id"] == "owner@im.wechat"
        assert payload["client_id"].startswith("harness-")
        assert "private-peer-token" not in json.dumps(store.status())
        with pytest.raises(ChannelError, match="prior authenticated"):
            await transport.send("other@im.wechat", "private", "delivery")
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


async def test_weixin_filters_owners_and_does_not_regress_context_sequence(tmp_path):
    records = [wx_message(), wx_message(message_id=2, seq=2, context_token="new-token")]
    transport, _ = await weixin(
        lambda _: httpx.Response(200, json={"ret": 0, "msgs": records, "get_updates_buf": "cursor"})
    )
    store = ChannelStore(cwd=tmp_path, transport="weixin")
    transport.store = store
    try:
        await transport.poll_once(store)
        records[:] = [
            wx_message(),
            wx_message(message_id=3, from_user_id="attacker", context_token="attacker-token"),
        ]
        await transport.poll_once(store)
        assert transport.context("owner@im.wechat") == "new-token"
        assert store.get("context:attacker") is None
        assert len(store.status()["inbox"]) == 2
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


async def test_weixin_bad_account_and_failed_ingest_never_advance_cursor(tmp_path, monkeypatch):
    record = wx_message(to_user_id="other-bot")
    transport, _ = await weixin(
        lambda _: httpx.Response(200, json={"ret": 0, "msgs": [record], "get_updates_buf": "next"})
    )
    store = ChannelStore(cwd=tmp_path, transport="weixin")
    try:
        with pytest.raises(ChannelError, match="another bot"):
            await transport.poll_once(store)
        assert store.get("cursor") is None
        record["to_user_id"] = "bot@im.bot"
        monkeypatch.setattr(store, "ingest", lambda _: (_ for _ in ()).throw(OSError("disk full")))
        with pytest.raises(OSError, match="disk full"):
            await transport.poll_once(store)
        assert store.get("cursor") is None
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


@pytest.mark.parametrize("code,error", [(-14, ChannelError), (-2, RateLimited)])
async def test_weixin_auth_expiry_and_rate_limit_never_retry_without_context(tmp_path, code, error):
    transport, requests = await weixin(lambda _: httpx.Response(200, json={"ret": code}))
    store = ChannelStore(cwd=tmp_path, transport="weixin")
    store.set("context:owner@im.wechat", {"account": "bot@im.bot", "token": "peer"})
    transport.store = store
    try:
        with pytest.raises(error):
            await transport.send("owner@im.wechat", "hello", "one")
        assert len(requests) == 2
        assert json.loads(requests[-1].content)["msg"]["context_token"] == "peer"
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


async def test_weixin_real_aes_media_roundtrip_and_cdn_upload_route(tmp_path):
    transport, _ = await weixin()
    await transport.client.aclose()
    key = bytes(range(16))
    plain = b"a private file\x00"
    encrypted = crypt_media(plain, key, encrypt=True)
    requests, uploaded = [], {}

    def api(request):
        requests.append(request)
        if request.url.host == "novac2c.cdn.weixin.qq.com":
            assert "Authorization" not in request.headers
            if request.method == "GET":
                return httpx.Response(200, content=encrypted)
            assert crypt_media(request.content, uploaded["key"], encrypt=False) == plain
            return httpx.Response(200, headers={"x-encrypted-param": "download-token"})
        payload = json.loads(request.content)
        if request.url.path.endswith("getuploadurl"):
            uploaded["key"] = bytes.fromhex(payload["aeskey"])
            assert payload["rawsize"] == len(plain) and payload["media_type"] == 3
            return httpx.Response(200, json={"ret": 0, "upload_param": "upload-token"})
        item = payload["msg"]["item_list"][0]
        assert item["type"] == 4 and item["file_item"]["file_name"] == "report.txt"
        assert item["file_item"]["media"]["encrypt_query_param"] == "download-token"
        return httpx.Response(200, json={"ret": 0})

    transport.client = httpx.AsyncClient(transport=httpx.MockTransport(api))
    store = ChannelStore(cwd=tmp_path, transport="weixin")
    store.set("context:owner@im.wechat", {"account": "bot@im.bot", "token": "peer"})
    transport.store = store
    message = ChannelMessage(
        id="m",
        user_id="u",
        thread_id="t",
        channel_id="c",
        text="file",
        attachments=[
            {
                "kind": "file",
                "item": {
                    "file_name": "../report.txt",
                    "media": {
                        "encrypt_query_param": "download-token",
                        "aes_key": base64.b64encode(key.hex().encode()).decode(),
                    },
                },
            }
        ],
    )
    try:
        media = await transport.prepare_media(message)
        assert media[0].data and base64.b64decode(media[0].data) == plain
        await transport.send_media("owner@im.wechat", media[0], "delivery")
        assert len(requests) == 4
        message.attachments[0]["item"]["media"]["full_url"] = "https://attacker.example"
        with pytest.raises(ChannelError, match="CDN"):
            await transport.prepare_media(message)
        assert len(requests) == 4
        with pytest.raises(ValueError):
            crypt_media(encrypted[:-1], key, encrypt=False)
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


async def qq(*, handler=None, mode="websocket"):
    requests = []

    def api(request):
        requests.append(request)
        if request.url.host == "bots.qq.com":
            assert json.loads(request.content) == {"appId": "123", "clientSecret": "qq-secret"}
            return httpx.Response(200, json={"access_token": "access-token", "expires_in": "7200"})
        assert request.headers["Authorization"] == "QQBot access-token"
        if request.url.path == "/gateway":
            return httpx.Response(200, json={"url": "wss://api.sgroup.qq.com/websocket/"})
        if handler:
            return handler(request)
        return httpx.Response(200, json={"id": "sent"})

    transport = QQBotTransport(
        config=ChannelConfig(
            app_id="123",
            allowed_users=["123:c2c:owner", "123:group:member", "123:guild:member"],
            allow_groups=True,
            receive_mode=mode,
        ),
        token="qq-secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(api)),
    )
    await transport.authenticate()
    return transport, requests


def qq_event(kind="C2C_MESSAGE_CREATE", **overrides):
    return {
        "op": 0,
        "t": kind,
        "s": 2,
        "d": {
            "id": "incoming",
            "timestamp": datetime.now(UTC).isoformat(),
            "author": {"user_openid": "owner", "member_openid": "member", "id": "member"},
            "group_openid": "group",
            "channel_id": "channel",
            "guild_id": "guild",
            "content": "hello",
            **overrides,
        },
    }


@pytest.mark.parametrize(
    "event,kind,target,userkind",
    [
        ("C2C_MESSAGE_CREATE", "c2c", "owner", "c2c"),
        ("GROUP_AT_MESSAGE_CREATE", "group", "group", "group"),
        ("AT_MESSAGE_CREATE", "guild", "channel", "guild"),
        ("DIRECT_MESSAGE_CREATE", "dm", "guild", "guild"),
    ],
)
async def test_qq_exact_routes_and_durable_idempotency_sequence(
    tmp_path, event, kind, target, userkind
):
    transport, requests = await qq()
    store = ChannelStore(cwd=tmp_path, transport="qqbot")
    transport.store = store
    try:
        await transport.ingest(qq_event(event), store)
        message = store.claim_message()
        assert message is not None and message.channel_id == f"123:{kind}:{target}"
        assert message.user_id.startswith(f"123:{userkind}:")
        store.complete(message, "reply", limit=4000)
        assert await deliver_message(transport=transport, store=store)
        first = json.loads(requests[-1].content)
        assert first["msg_id"] == "incoming"
        assert first.get("msg_seq") == (1 if kind in {"c2c", "group"} else None)
        delivery = store.status()["outbox"][0]["id"]
        store.close()
        store = ChannelStore(cwd=tmp_path, transport="qqbot")
        transport.store = store
        await transport.send(message.thread_id, "reply", delivery)
        assert json.loads(requests[-1].content).get("msg_seq") == (
            1 if kind in {"c2c", "group"} else None
        )
        await transport.send(message.thread_id, "second part", "next-part")
        assert json.loads(requests[-1].content).get("msg_seq") == (
            2 if kind in {"c2c", "group"} else None
        )
        expected = (
            f"/v2/{'users' if kind == 'c2c' else 'groups'}/{target}/messages"
            if kind in {"c2c", "group"}
            else f"/{'channels' if kind == 'guild' else 'dms'}/{target}/messages"
        )
        assert requests[-1].url.path == expected
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


async def test_qq_signed_webhook_and_challenge_are_not_authority_to_dispatch(tmp_path):
    transport, _ = await qq(mode="webhook")
    store = ChannelStore(cwd=tmp_path, transport="qqbot")
    try:
        challenge = {"op": 13, "d": {"plain_token": "challenge", "event_ts": "1"}}
        response = await transport.handle_event({}, json.dumps(challenge).encode(), store)
        transport.signing_key.public_key().verify(
            bytes.fromhex(response["signature"]), b"1challenge"
        )
        assert store.claim_message() is None
        body = json.dumps(qq_event()).encode()
        timestamp = str(int(time.time()))
        headers = {
            "X-Signature-Timestamp": timestamp,
            "X-Signature-Ed25519": transport.signing_key.sign(timestamp.encode() + body).hex(),
        }
        with pytest.raises(AuthenticationError):
            await transport.handle_event(headers, body + b" ", store)
        assert store.claim_message() is None
        await transport.handle_event(headers, body, store)
        await transport.handle_event(headers, body, store)
        assert len(store.status()["inbox"]) == 1
        stale = str(int(time.time()) - 301)
        with pytest.raises(AuthenticationError):
            await transport.handle_event(
                {
                    "X-Signature-Timestamp": stale,
                    "X-Signature-Ed25519": transport.signing_key.sign(stale.encode() + body).hex(),
                },
                body,
                store,
            )
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


async def test_qq_signing_key_matches_published_tencent_vector():
    transport, _ = await qq()
    try:
        transport.token = "naOC0ocQE3shWLAfffVLB1rhYPG7"

        # Skip the fresh exchange: this test checks the published cryptographic vector.
        async def access():
            return "unused"

        transport.access = access
        await transport.authenticate()
        public = transport.signing_key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        assert list(public) == [
            215,
            195,
            98,
            254,
            120,
            174,
            248,
            31,
            242,
            50,
            135,
            180,
            147,
            98,
            139,
            93,
            176,
            42,
            60,
            79,
            227,
            11,
            33,
            94,
            77,
            25,
            96,
            155,
            93,
            118,
            103,
            58,
        ]
        # The published seed/public-key vector is independent of our roundtrip
        # test. Payload bytes are tested separately without JSON normalization.
    finally:
        await transport.close()
        await transport.client.aclose()


async def test_qq_real_gateway_checkpoints_after_ingest_and_resumes(tmp_path):
    transport, _ = await qq()
    store = ChannelStore(cwd=tmp_path, transport="qqbot")
    auths, heartbeats = [], []

    async def server(socket):
        await socket.send(json.dumps({"op": 10, "d": {"heartbeat_interval": 50}}))
        auths.append(json.loads(await socket.recv()))
        if len(auths) == 1:
            await socket.send(
                json.dumps(
                    {
                        "op": 0,
                        "t": "READY",
                        "s": 1,
                        "d": {"session_id": "session-one", "user": {"id": "bot"}},
                    }
                )
            )
            await socket.send(json.dumps(qq_event()))
            heartbeat = json.loads(await socket.recv())
            heartbeats.append(heartbeat)
            assert store.get("resume")["sequence"] == 2
            assert len(store.status()["inbox"]) == 1
            await socket.send(json.dumps({"op": 11}))
        await socket.send(json.dumps({"op": 7}))

    try:
        async with serve(server, "127.0.0.1", 0) as service:
            url = f"ws://127.0.0.1:{service.sockets[0].getsockname()[1]}"
            transport.socket_connect = lambda _, **kwargs: connect(url, **kwargs)
            await transport.receive(store)
            await transport.receive(store)
        assert auths[0]["op"] == 2 and auths[0]["d"]["token"] == "QQBot access-token"
        assert auths[1]["op"] == 6 and auths[1]["d"]["seq"] == 2
        assert heartbeats[0] == {"op": 1, "d": 2}
        assert transport.store is None
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


async def test_qq_filters_non_mentions_and_enforces_reply_expiry(tmp_path):
    transport, requests = await qq()
    store = ChannelStore(cwd=tmp_path, transport="qqbot")
    transport.store = store
    try:
        await transport.ingest(qq_event("MESSAGE_CREATE"), store)
        await transport.ingest(qq_event(author={"user_openid": "other"}), store)
        assert store.claim_message() is None
        await transport.ingest(qq_event(timestamp="2020-01-01T00:00:00Z"), store)
        message = store.claim_message()
        assert message is not None
        with pytest.raises(ChannelError, match="expired"):
            await transport.send(message.thread_id, "late", "delivery")
        with pytest.raises(AuthenticationError):
            await transport.send(json.dumps(["other-app", "c2c", "owner"]), "secret", "delivery")
        assert len(requests) == 1
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


async def test_pairing_verification_code_credentials_and_no_background_work():
    calls, displayed = [], []
    states = [
        {"status": "need_verifycode"},
        {
            "status": "confirmed",
            "bot_token": "private-bot-token",
            "ilink_bot_id": "bot@im.bot",
            "ilink_user_id": "owner@im.wechat",
            "baseurl": "https://ilinkai.weixin.qq.com",
        },
    ]

    def api(request):
        calls.append(request)
        assert "Authorization" not in request.headers
        if request.method == "POST":
            assert json.loads(request.content) == {"local_token_list": []}
            return httpx.Response(
                200,
                json={
                    "qrcode": "qr-handle",
                    "qrcode_img_content": "https://weixin.example/qr-display",
                },
            )
        return httpx.Response(200, json=states.pop(0))

    async def display(value):
        displayed.append(value)

    async def code():
        return "123456"

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(api)) as client,
        WeixinPairing(client=client) as pairing,
    ):
        credentials = await pairing.pair(
            display_qr=display, verification_code=code, timeout_seconds=5
        )
    assert credentials["bot_token"] == "private-bot-token" and len(displayed) == 1
    assert calls[-1].url.params["verify_code"] == "123456"


@pytest.mark.parametrize("mode", ["redirect", "expired", "cancel"])
async def test_pairing_refuses_untrusted_redirect_and_closes_pending_work(mode):
    pending = asyncio.Event()

    async def api(request):
        if request.method == "POST":
            return httpx.Response(200, json={"qrcode": "qr", "qrcode_img_content": "display"})
        pending.set()
        if mode == "cancel":
            await asyncio.Event().wait()
        return httpx.Response(
            200,
            json={"status": "scaned_but_redirect", "redirect_host": "attacker.example"}
            if mode == "redirect"
            else {"status": "expired"},
        )

    async def display(_):
        pass

    async with (
        httpx.AsyncClient(transport=httpx.MockTransport(api)) as client,
        WeixinPairing(client=client) as pairing,
    ):
        task = asyncio.create_task(pairing.pair(display_qr=display, timeout_seconds=5))
        await asyncio.wait_for(pending.wait(), 2)
        if mode == "cancel":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(ChannelError):
                await task


def test_pair_credentials_are_private_and_loadable_without_echoing_token(tmp_path, monkeypatch):
    account = {
        "bot_token": "private-bot-token",
        "ilink_bot_id": "bot@im.bot",
        "ilink_user_id": "owner@im.wechat",
        "base_url": "https://ilinkai.weixin.qq.com",
    }
    path = save_credentials(tmp_path, account)
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text()) == account
    with pytest.raises(ValueError, match="exist"):
        save_credentials(tmp_path, account)
    config = ChannelConfig(account_file=str(path), allowed_users=[account["ilink_user_id"]])
    transport = build_transport("weixin", config)
    assert json.loads(transport.token) == account
    asyncio.run(transport.close())
    app = typer.Typer()
    register_weixin_commands(app)
    result = CliRunner().invoke(app, ["--cwd", str(tmp_path)])
    assert result.exit_code == 1 and "private-bot-token" not in result.output


def test_pair_command_writes_config_snippet_without_secret(tmp_path, monkeypatch):
    async def fake_pair(self, **kwargs):
        await kwargs["display_qr"]("local fake QR payload")
        return {
            "bot_token": "must-not-print",
            "ilink_bot_id": "bot@im.bot",
            "ilink_user_id": "owner@im.wechat",
            "base_url": "https://ilinkai.weixin.qq.com",
        }

    monkeypatch.setattr(WeixinPairing, "pair", fake_pair)
    app = typer.Typer()
    register_weixin_commands(app)
    result = CliRunner().invoke(app, ["--cwd", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert (
        "account_file = " in result.output
        and 'allowed_users = ["owner@im.wechat"]' in result.output
    )
    assert "must-not-print" not in result.output
    assert (
        json.loads((tmp_path / ".harness/channels/weixin-account.json").read_text())["bot_token"]
        == "must-not-print"
    )


async def test_weixin_json_credentials_match_account_and_group_scope_is_explicit():
    transport, requests = await weixin()
    try:
        transport.token = json.dumps({"bot_token": "bot-token", "ilink_bot_id": "other"})
        with pytest.raises(ChannelError, match="another configured"):
            await transport.authenticate()
        assert len(requests) == 1
        transport.token = json.dumps(
            {
                "bot_token": "bot-token",
                "ilink_bot_id": "bot@im.bot",
                "base_url": "https://ilinkai.weixin.qq.com",
            }
        )
        await transport.authenticate()
        assert transport.bot_id == "bot@im.bot"
        transport.config = replace(transport.config, allow_groups=True)
        with pytest.raises(ChannelError, match="group routing"):
            await transport.authenticate()
    finally:
        await transport.close()
        await transport.client.aclose()


async def test_qq_media_uses_no_auto_send_upload_then_acknowledged_message(tmp_path):
    def api(request):
        payload = json.loads(request.content)
        if request.url.path.endswith("/files"):
            assert payload["srv_send_msg"] is False
            assert base64.b64decode(payload["file_data"]) == b"image bytes"
            assert payload["file_type"] == 1
            return httpx.Response(200, json={"file_info": "private-file-reference"})
        assert payload["media"] == {"file_info": "private-file-reference"}
        assert payload["msg_seq"] == 1 and payload["msg_id"] == "incoming"
        return httpx.Response(200, json={"id": "sent-image"})

    transport, requests = await qq(handler=api)
    store = ChannelStore(cwd=tmp_path, transport="qqbot")
    transport.store = store
    try:
        await transport.ingest(qq_event(), store)
        message = store.claim_message()
        assert message is not None
        image = MediaAttachment(
            kind="image",
            mime_type="image/png",
            data=base64.b64encode(b"image bytes").decode(),
            name="image.png",
        )
        await transport.send_media(message.thread_id, image, "delivery")
        assert requests[-2].url.path == "/v2/users/owner/files"
        assert requests[-1].url.path == "/v2/users/owner/messages"
        transport.token_expires = 0
        await asyncio.gather(*(transport.access() for _ in range(5)))
        assert sum(request.url.host == "bots.qq.com" for request in requests) == 2
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


async def test_qq_gateway_failed_ingest_does_not_checkpoint_unstored_message(tmp_path, monkeypatch):
    transport, _ = await qq()
    store = ChannelStore(cwd=tmp_path, transport="qqbot")

    async def server(socket):
        await socket.send(json.dumps({"op": 10, "d": {"heartbeat_interval": 1000}}))
        await socket.recv()
        await socket.send(
            json.dumps({"op": 0, "t": "READY", "s": 1, "d": {"session_id": "session"}})
        )
        await socket.send(json.dumps(qq_event()))
        await socket.wait_closed()

    monkeypatch.setattr(store, "ingest", lambda _: (_ for _ in ()).throw(OSError("disk full")))
    try:
        async with serve(server, "127.0.0.1", 0) as service:
            url = f"ws://127.0.0.1:{service.sockets[0].getsockname()[1]}"
            transport.socket_connect = lambda _, **kwargs: connect(url, **kwargs)
            with pytest.raises(OSError, match="disk full"):
                await transport.receive(store)
        assert store.get("resume")["sequence"] == 1
        assert transport.store is None
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()


async def test_qq_missing_heartbeat_ack_closes_owned_socket(tmp_path):
    transport, _ = await qq()
    store = ChannelStore(cwd=tmp_path, transport="qqbot")
    closed = asyncio.Event()

    async def server(socket):
        await socket.send(json.dumps({"op": 10, "d": {"heartbeat_interval": 50}}))
        await socket.recv()
        await socket.recv()  # Intentionally omit the first heartbeat ACK.
        await socket.wait_closed()
        closed.set()

    try:
        async with serve(server, "127.0.0.1", 0) as service:
            url = f"ws://127.0.0.1:{service.sockets[0].getsockname()[1]}"
            transport.socket_connect = lambda _, **kwargs: connect(url, **kwargs)
            await asyncio.wait_for(transport.receive(store), 2)
            await asyncio.wait_for(closed.wait(), 1)
    finally:
        store.close()
        await transport.close()
        await transport.client.aclose()
