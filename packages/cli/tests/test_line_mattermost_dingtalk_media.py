import base64
import gzip
import hashlib
import hmac
import json
import time

import httpx
import pytest
from aiohttp import web

from harness.cli.channels.attachment_io import MediaPublisher, download
from harness.cli.channels.dingtalk import DingTalkTransport
from harness.cli.channels.line import LINETransport
from harness.cli.channels.mattermost import MattermostTransport
from harness.cli.channels.transports import ChannelError
from harness.core.gateway_channels import ChannelConfig, ChannelStore
from harness.core.schemas import MediaAttachment


async def test_line_media_auth_native_duration_and_actual_capability_listener(tmp_path):
    native = []

    def http(request):
        assert request.headers["Authorization"] == "Bearer line-token"
        if request.url.path.endswith("/info"):
            return httpx.Response(200, json={"userId": "bot"})
        if request.url.host == "api-data.line.me":
            assert request.url.path == "/v2/bot/message/123/content"
            return httpx.Response(
                200, content=b"image-bytes", headers={"Content-Type": "image/jpeg"}
            )
        native.append(json.loads(request.content))
        return httpx.Response(200, json={})

    store = ChannelStore(cwd=tmp_path, transport="line")
    async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
        transport = LINETransport(
            config=ChannelConfig(
                allowed_users=["owner"],
                webhook_url="https://bot.example/events",
                webhook_path="/events",
            ),
            token="line-token",
            app_token="signing-secret",
            client=client,
        )
        await transport.authenticate()
        body = json.dumps(
            {
                "destination": "bot",
                "events": [
                    {
                        "type": "message",
                        "webhookEventId": "event",
                        "source": {"type": "user", "userId": "owner"},
                        "message": {
                            "type": "image",
                            "id": "123",
                            "contentProvider": {"type": "line"},
                        },
                    }
                ],
            }
        ).encode()
        signature = base64.b64encode(
            hmac.new(b"signing-secret", body, hashlib.sha256).digest()
        ).decode()
        await transport.handle_event({"X-Line-Signature": signature}, body, store)
        message = store.claim_message()
        assert message and not message.text and len(message.attachments) == 1
        media = (await transport.prepare_media(message))[0]
        assert media.kind == "image" and base64.b64decode(media.data or "") == b"image-bytes"
        runner = web.AppRunner(transport.application(store), access_log=None)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        assert runner.addresses
        port = runner.addresses[0][1]
        try:
            await transport.send_media(message.thread_id, media, "image-delivery")
            url = native[-1]["messages"][0]["originalContentUrl"]
            assert url.startswith("https://bot.example/events/media/")
            async with httpx.AsyncClient() as local:
                endpoint = f"http://127.0.0.1:{port}" + httpx.URL(url).path
                response = await local.get(endpoint)
                assert (
                    response.content == b"image-bytes"
                    and response.headers["X-Content-Type-Options"] == "nosniff"
                )
                assert (
                    await local.get(endpoint, headers={"Range": "bytes=0-4"})
                ).content == b"image"
                assert (
                    await local.get(f"http://127.0.0.1:{port}/events/media/not-a-token")
                ).status_code == 404
                with pytest.raises(ChannelError, match="owner/content"):
                    transport.publisher.publish("image-delivery", '["user","other"]', media)
                restarted = MediaPublisher(
                    store, public_url="https://bot.example/events", callback_path="/events"
                )
                assert restarted.publish("image-delivery", message.thread_id, media) == url
                with store.db:
                    store.db.execute("UPDATE published_media SET expires=0")
                assert (await local.get(endpoint)).status_code == 404
            audio = MediaAttachment(
                kind="audio",
                mime_type="audio/mp4",
                data=base64.b64encode(b"mp4").decode(),
                duration_ms=1234,
            )
            await transport.send_media(message.thread_id, audio, "audio-delivery")
            assert native[-1]["messages"][0]["duration"] == 1234
            with pytest.raises(ChannelError, match="duration_ms"):
                await transport.send_media(
                    message.thread_id,
                    audio.model_copy(update={"duration_ms": None}),
                    "audio-missing",
                )
        finally:
            await runner.cleanup()
            await transport.close()
    store.close()


async def test_mattermost_attachments_verify_post_ownership_and_upload_thread(tmp_path):
    sent = []

    def http(request):
        assert request.headers["Authorization"] == "Bearer token"
        if request.url.path.endswith("/info"):
            return httpx.Response(
                200,
                json={
                    "id": "file",
                    "post_id": "post",
                    "name": "report.pdf",
                    "mime_type": "application/pdf",
                },
            )
        if request.method == "GET":
            return httpx.Response(200, content=b"%PDF", headers={"Content-Type": "application/pdf"})
        if request.url.path.endswith("/files"):
            assert (
                b'name="channel_id"' in request.content
                and b"report.pdf" in request.content
                and b"%PDF" in request.content
            )
            return httpx.Response(201, json={"file_infos": [{"id": "uploaded"}]})
        sent.append(json.loads(request.content))
        return httpx.Response(201, json={"id": "sent"})

    store = ChannelStore(cwd=tmp_path, transport="mattermost")
    async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
        transport = MattermostTransport(
            config=ChannelConfig(
                homeserver="https://mm.example",
                allowed_users=["owner"],
                allowed_channels=["channel"],
            ),
            token="token",
            client=client,
        )
        transport.bot_id, transport.username = "bot", "harness"
        transport.channels = {"channel": {"type": "D"}}
        transport.ingest(
            {"id": "post", "user_id": "owner", "channel_id": "channel", "file_ids": ["file"]}, store
        )
        message = store.claim_message()
        assert message and message.attachments
        media = (await transport.prepare_media(message))[0]
        await transport.send_media(message.thread_id, media, "delivery")
        assert sent == [
            {
                "channel_id": "channel",
                "root_id": "",
                "message": "",
                "file_ids": ["uploaded"],
                "pending_post_id": "delivery",
            }
        ]
        message.attachments[0]["post_id"] = "private-other-post"
        with pytest.raises(ChannelError, match="different post"):
            await transport.prepare_media(message)
        await transport.close()
    store.close()


async def test_dingtalk_media_code_exchange_credential_boundary_and_remote_image(tmp_path):
    posts = []

    def http(request):
        if request.url.path.endswith("accessToken"):
            assert json.loads(request.content) == {"appKey": "app", "appSecret": "secret"}
            return httpx.Response(200, json={"accessToken": "access", "expireIn": 7200})
        if request.url.path.endswith("/download"):
            assert request.headers["x-acs-dingtalk-access-token"] == "access"
            assert json.loads(request.content) == {"robotCode": "app", "downloadCode": "opaque"}
            return httpx.Response(
                200, json={"downloadUrl": "https://static.dingtalk.com/image?cap=private"}
            )
        if request.url.host == "static.dingtalk.com":
            assert (
                "Authorization" not in request.headers
                and "x-acs-dingtalk-access-token" not in request.headers
            )
            return httpx.Response(200, content=b"jpeg", headers={"Content-Type": "image/jpeg"})
        posts.append(json.loads(request.content))
        return httpx.Response(200, json={"errcode": 0})

    store = ChannelStore(cwd=tmp_path, transport="dingtalk")
    async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
        transport = DingTalkTransport(
            config=ChannelConfig(app_id="app", allowed_users=["corp:owner"]),
            token="secret",
            client=client,
        )
        transport.store, transport.access_token, transport.access_expires = store, "", 0
        event = {
            "robotCode": "app",
            "msgtype": "picture",
            "senderCorpId": "corp",
            "senderStaffId": "owner",
            "conversationId": "dm",
            "conversationType": "1",
            "msgId": "message",
            "content": {"downloadCode": "opaque"},
            "sessionWebhook": "https://oapi.dingtalk.com/robot/sendBySession?session=private",
            "sessionWebhookExpiredTime": int((time.time() + 300) * 1000),
        }
        transport._ingest(event, store)
        message = store.claim_message()
        assert message and message.attachments
        media = (await transport.prepare_media(message))[0]
        assert media.kind == "image"
        with pytest.raises(ChannelError, match="inline file"):
            await transport.send_media(message.thread_id, media, "unsupported")
        await transport.send_media(
            message.thread_id,
            MediaAttachment(
                kind="image", mime_type="image/png", url="https://images.example/image.png"
            ),
            "hosted",
        )
        assert posts[-1]["markdown"]["text"] == "![](https://images.example/image.png)"
        assert posts[-1]["at"] == {"isAtAll": False}
        await transport.close()
    store.close()


async def test_native_download_bounds_and_rejects_cross_origin_credentials():
    calls = []

    def http(request):
        calls.append(request)
        if request.url.host == "api.example":
            return httpx.Response(302, headers={"Location": "https://cdn.example/media"})
        assert "Authorization" not in request.headers
        return httpx.Response(200, content=b"too-long")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(http), follow_redirects=True
    ) as client:
        with pytest.raises(ChannelError, match="byte limit"):
            await download(
                client,
                "https://api.example/media",
                domains=("api.example", "cdn.example"),
                headers={"Authorization": "Bearer secret"},
                max_bytes=3,
            )
        assert len(calls) == 2
        with pytest.raises(ValueError, match="outside"):
            await download(client, "https://127.0.0.1/private", domains=("api.example",))


async def test_native_download_rejects_compressed_response():
    def http(request):
        assert request.headers["Accept-Encoding"] == "identity"
        return httpx.Response(
            200, content=gzip.compress(b"large" * 1000), headers={"Content-Encoding": "gzip"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(http)) as client:
        with pytest.raises(ChannelError, match="Compressed"):
            await download(client, "https://api.example/media", domains=("api.example",))
