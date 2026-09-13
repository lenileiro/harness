import asyncio
import base64
import hashlib
import json
import time

import httpx
import pytest
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from harness.cli.channels.runtime import build_transport, process_message
from harness.cli.channels.transports import ChannelError, DeliveryRejected
from harness.cli.channels.wecom_bot import WS_URL, WeComBotTransport
from harness.core.gateway_channels import ChannelConfig, ChannelStore
from harness.core.schemas import MediaAttachment


def callback(identifier="message", *, owner="owner", group=False, bot="bot"):
    return {
        "cmd": "aibot_msg_callback",
        "headers": {"req_id": "callback-" + identifier},
        "body": {
            "msgid": identifier,
            "aibotid": bot,
            "chattype": "group" if group else "single",
            "chatid": "room" if group else "",
            "from": {"userid": owner},
            "msgtype": "text",
            "text": {"content": "@Harness /usage" if group else "hello"},
        },
    }


async def test_wecom_bot_real_socket_auth_scoped_group_and_chunk_upload(tmp_path):
    sent, chunks = [], []

    async def server(socket):
        subscription = json.loads(await socket.recv())
        assert subscription["cmd"] == "aibot_subscribe"
        assert subscription["body"] == {"bot_id": "bot", "secret": "secret"}
        await socket.send(json.dumps({"headers": subscription["headers"], "errcode": 0}))
        for frame in [
            callback(),
            callback(),
            callback("other", owner="stranger"),
            callback("wrong", bot="other"),
            callback("group", group=True),
        ]:
            await socket.send(json.dumps(frame))
        async for raw in socket:
            frame = json.loads(raw)
            body = {}
            if frame["cmd"] == "aibot_upload_media_init":
                assert frame["body"]["md5"] == hashlib.md5(b"file-bytes").hexdigest()
                assert frame["body"]["type"] == "file" and frame["body"]["total_chunks"] == 1
                body = {"upload_id": "upload"}
            elif frame["cmd"] == "aibot_upload_media_chunk":
                assert frame["body"]["chunk_index"] == 0 and frame["body"]["upload_id"] == "upload"
                chunks.append(base64.b64decode(frame["body"]["base64_data"]))
            elif frame["cmd"] == "aibot_upload_media_finish":
                body = {"media_id": "media"}
            elif frame["cmd"] != "ping":
                sent.append(frame)
            await socket.send(json.dumps({"headers": frame["headers"], "errcode": 0, "body": body}))

    store = ChannelStore(cwd=tmp_path, transport="wecom")
    async with serve(server, "127.0.0.1", 0) as local:

        def socket_factory(url, **kwargs):
            assert url == WS_URL
            return connect(f"ws://127.0.0.1:{local.sockets[0].getsockname()[1]}", **kwargs)

        transport = WeComBotTransport(
            config=ChannelConfig(
                wecom_mode="bot",
                app_id="bot",
                username="Harness",
                allowed_users=["owner"],
                allow_groups=True,
            ),
            token="secret",
            socket_connect=socket_factory,
        )
        await transport.authenticate()
        store.bind_identity(transport.identity)
        task = asyncio.create_task(transport.receive(store))
        try:
            for _ in range(100):
                if len(store.status()["inbox"]) == 2:
                    break
                await asyncio.sleep(0.01)
            direct, group = store.claim_message(), store.claim_message()
            assert direct and group and group.text == "/usage"
            assert direct.thread_id == '["bot","dm","owner","owner"]'
            assert group.thread_id == '["bot","group","room","owner"]'
            assert store.claim_message() is None
            await transport.send(direct.thread_id, "private reply", "direct-delivery")
            await transport.send(group.thread_id, "group reply", "group-delivery")
            assert sent[0]["cmd"] == "aibot_send_msg" and sent[0]["body"]["chatid"] == "owner"
            assert (
                sent[1]["headers"]["req_id"] == "callback-group"
                and sent[1]["body"]["stream"]["finish"] is True
            )
            media = MediaAttachment(
                kind="file",
                mime_type="application/pdf",
                name="report.pdf",
                data=base64.b64encode(b"file-bytes").decode(),
            )
            await transport.send_media(direct.thread_id, media, "file-delivery")
            assert chunks == [b"file-bytes"] and sent[-1]["body"]["file"]["media_id"] == "media"
            with pytest.raises(ChannelError, match="owner"):
                await transport.send('["bot","dm","stranger","owner"]', "secret", "wrong")
            store.set(
                "wecom-bot-route:" + group.thread_id,
                {"request_id": "callback-group", "expires": time.time() - 1},
            )
            with pytest.raises(DeliveryRejected, match="expired"):
                await transport.send(group.thread_id, "late", "late")
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await transport.close()
    store.close()
    reopened = ChannelStore(cwd=tmp_path, transport="wecom")
    original_route = reopened.get('wecom-bot-route:["bot","dm","owner","owner"]')
    transport.ingest(callback(), reopened)
    assert len(reopened.status()["inbox"]) == 2
    assert reopened.get('wecom-bot-route:["bot","dm","owner","owner"]') == original_route
    reopened.close()


async def test_wecom_encrypted_attachment_only_reaches_gateway_and_rejects_bad_key(tmp_path):
    key = bytes(range(32))
    plaintext = b"image-payload"
    padded = plaintext + bytes([32 - len(plaintext)]) * (32 - len(plaintext))
    cipher = Cipher(algorithms.AES(key), modes.CBC(key[:16])).encryptor()
    encrypted = cipher.update(padded) + cipher.finalize()

    def http(request):
        assert request.url.host == "images.myqcloud.com"
        assert "Authorization" not in request.headers
        return httpx.Response(200, content=encrypted)

    received = []

    async def receiver(**kwargs):
        received.append(kwargs)
        return {"reply": {"text": "Read the image"}}

    store = ChannelStore(cwd=tmp_path, transport="wecom")
    async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
        transport = WeComBotTransport(
            config=ChannelConfig(wecom_mode="bot", app_id="bot", allowed_users=["owner"]),
            token="secret",
            client=client,
        )
        transport.bot_id = "bot"
        frame = callback()
        frame["body"].update(
            msgtype="image",
            image={
                "url": "https://images.myqcloud.com/opaque",
                "aeskey": base64.b64encode(key).decode().rstrip("="),
            },
        )
        transport.ingest(frame, store)
        assert await process_message(
            cwd=tmp_path, transport=transport, store=store, receiver=receiver
        )
        assert len(received) == 1 and received[0]["message"] == ""
        attachment = received[0]["attachments"][0]
        assert attachment.kind == "image" and base64.b64decode(attachment.data) == plaintext
        frame["body"]["msgid"] = "bad-key"
        frame["body"]["image"]["aeskey"] = "invalid"
        transport.ingest(frame, store)
        message = store.claim_message()
        assert message
        with pytest.raises((ChannelError, ValueError)):
            await transport.prepare_media(message)
        await transport.close()
    store.close()


async def test_wecom_bot_explicit_mode_auth_mismatch_and_revocation(tmp_path, monkeypatch):
    monkeypatch.setenv("WECOM_BOT_SECRET", "secret")
    transport = build_transport(
        "wecom",
        ChannelConfig.from_dict(
            {
                "wecom_mode": "bot",
                "app_id": "bot",
                "token_env": "WECOM_BOT_SECRET",
                "allowed_users": ["owner"],
            }
        ),
    )
    assert isinstance(transport, WeComBotTransport)
    await transport.close()
    with pytest.raises(ValueError, match="wecom_mode"):
        ChannelConfig.from_dict({"wecom_mode": "unknown"})

    async def server(socket):
        request = json.loads(await socket.recv())
        await socket.send(json.dumps({"headers": request["headers"], "errcode": 40001}))

    async with serve(server, "127.0.0.1", 0) as local:

        def socket_factory(url, **kwargs):
            return connect(f"ws://127.0.0.1:{local.sockets[0].getsockname()[1]}", **kwargs)

        transport = WeComBotTransport(
            config=ChannelConfig(app_id="bot"), token="secret", socket_connect=socket_factory
        )
        with pytest.raises(ChannelError, match="authentication rejected"):
            await transport.authenticate()
        assert transport.ws is None
        transport.revoked = True
        with pytest.raises(ChannelError, match="competing"):
            await transport.connect()
        await transport.close()
