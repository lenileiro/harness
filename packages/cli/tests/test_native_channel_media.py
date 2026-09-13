"""Offline native transfer contracts; JWT/signature verification has separate suites."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from collections.abc import Mapping
from typing import Any, Literal

import httpx
import pytest

from harness.cli.channels.feishu import FeishuTransport, LarkTransport
from harness.cli.channels.google_chat import GoogleChatTransport
from harness.cli.channels.native_media import MEDIA_LIMIT, download_bytes
from harness.cli.channels.runtime import deliver_message, process_message
from harness.cli.channels.teams import FILE_CONSENT, FILE_DOWNLOAD, FILE_INFO, TeamsTransport
from harness.cli.channels.transports import ChannelError, DeliveryDeferred, DeliveryRejected
from harness.cli.channels.webhooks import AuthenticationError
from harness.core.gateway_channels import ChannelConfig, ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment


class GoogleFixture(GoogleChatTransport):
    async def _access(self) -> str:
        return "bot-token"

    async def _verify(self, headers: Mapping[str, str]) -> None:
        if headers.get("Authorization") != "verified":
            raise AuthenticationError("Invalid fixture signature")


class TeamsFixture(TeamsTransport):
    async def _access(self) -> str:
        return "bot-token"

    async def _verify(self, headers: Mapping[str, str], activity: dict[str, Any]) -> None:
        if headers.get("Authorization") != "verified":
            raise AuthenticationError("Invalid fixture signature")


def file_attachment(
    data=b"file bytes",
    name="result.txt",
    kind: Literal["image", "audio", "file"] = "file",
    mime="text/plain",
):
    return MediaAttachment(
        kind=kind, mime_type=mime, name=name, data=base64.b64encode(data).decode()
    )


def google_message():
    return ChannelMessage(
        id="spaces/s/messages/m",
        user_id="users/u",
        channel_id="spaces/s",
        thread_id=json.dumps(["spaces/s", "spaces/s/threads/t"]),
        text="look",
        attachments=[
            {
                "name": "spaces/s/messages/m/attachments/a",
                "contentName": "../image.png",
                "contentType": "image/png",
                "attachmentDataRef": {"resourceName": "opaque/media-token"},
                "downloadUri": "http://127.0.0.1/private",
            }
        ],
    )


@pytest.fixture
async def google():
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == "oauth2.googleapis.com":
            assert b"grant_type=refresh_token" in request.content
            assert b"refresh_token=refresh-secret" in request.content
            return httpx.Response(200, json={"access_token": "user-token", "expires_in": 3600})
        if request.method == "GET":
            assert request.headers["authorization"] == "Bearer bot-token"
            return httpx.Response(200, content=b"png bytes", headers={"Content-Type": "image/png"})
        assert request.headers["authorization"] == "Bearer user-token"
        return httpx.Response(
            200, json={"attachmentDataRef": {"attachmentUploadToken": "uploaded"}}
        )

    transport = GoogleFixture(
        config=ChannelConfig(
            app_id="users/bot", audience="https://bot.example/events", allowed_users=["users/u"]
        ),
        token=json.dumps(
            {
                "type": "service_account",
                "client_email": "a@p.iam.gserviceaccount.com",
                "private_key": "unused-fixture",
            }
        ),
        app_token=json.dumps(
            {
                "type": "authorized_user",
                "client_id": "client",
                "client_secret": "client-secret",
                "refresh_token": "refresh-secret",
                "scopes": ["https://www.googleapis.com/auth/chat.messages.create"],
            }
        ),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await transport.authenticate()
    try:
        yield transport, requests
    finally:
        await transport.client.aclose()
        await transport.close()


async def test_google_native_download_ignores_display_urls_and_upload_uses_user_identity(google):
    transport, requests = google
    result = await transport.prepare_media(google_message())
    assert result[0].name == "image.png" and result[0].kind == "image"
    assert base64.b64decode(result[0].data) == b"png bytes"
    assert (
        str(requests[0].url) == "https://chat.googleapis.com/v1/media/opaque/media-token?alt=media"
    )
    await transport.send_media(google_message().thread_id, file_attachment(), "delivery")
    upload, post = requests[-2:]
    assert upload.url.path == "/upload/v1/spaces/s/attachments:upload"
    assert upload.url.params["uploadType"] == "multipart"
    assert upload.headers["content-type"].startswith("multipart/related;")
    assert b'"filename": "result.txt"' in upload.content and b"file bytes" in upload.content
    assert json.loads(post.content) == {
        "attachment": [{"attachmentDataRef": {"attachmentUploadToken": "uploaded"}}],
        "thread": {"name": "spaces/s/threads/t"},
    }
    assert post.url.params["messageReplyOption"] == "REPLY_MESSAGE_OR_FAIL"


@pytest.mark.parametrize(
    "change",
    [
        {"driveDataRef": {"driveFileId": "drive"}},
        {"name": "spaces/other/messages/m/attachments/a"},
        {"attachmentDataRef": {"resourceName": "../escape"}},
    ],
)
async def test_google_invalid_or_drive_resources_rejected_before_network(google, change):
    transport, requests = google
    message = google_message()
    message.attachments[0].update(change)
    with pytest.raises(ChannelError):
        await transport.prepare_media(message)
    assert requests == []


async def test_google_bot_cannot_upload_without_explicit_user_credentials(google):
    transport, requests = google
    transport.user_account = None
    with pytest.raises(ChannelError, match="authorized-user"):
        await transport.send_media(google_message().thread_id, file_attachment(), "delivery")
    assert requests == []


async def test_google_media_only_event_is_durable_before_download(google, tmp_path):
    transport, requests = google
    store = ChannelStore(cwd=tmp_path, transport="google_chat")
    try:
        event = {
            "type": "MESSAGE",
            "space": {"name": "spaces/s", "type": "DM"},
            "user": {"name": "users/u", "type": "HUMAN"},
            "message": {"name": "spaces/s/messages/m", "attachment": google_message().attachments},
        }
        with pytest.raises(AuthenticationError):
            await transport.handle_event({}, json.dumps(event).encode(), store)
        await transport.handle_event(
            {"Authorization": "verified"}, json.dumps(event).encode(), store
        )
        message = store.claim_message()
        assert message is not None and message.attachments and message.text == ""
        assert not requests
    finally:
        store.close()


@pytest.mark.parametrize("transport_type", [FeishuTransport, LarkTransport])
@pytest.mark.parametrize(
    "kind,content",
    [
        ("image", {"image_key": "img_key"}),
        ("audio", {"file_key": "audio_key"}),
        ("media", {"file_key": "video_key", "image_key": "poster"}),
        ("file", {"file_key": "file_key", "file_name": "notes.txt"}),
        (
            "post",
            {
                "title": "look",
                "content": [
                    [{"tag": "text", "text": "attached"}, {"tag": "img", "image_key": "img_key"}]
                ],
            },
        ),
    ],
)
async def test_feishu_native_receive_and_upload_preserve_reply_root(
    tmp_path, monkeypatch, transport_type, kind, content
):
    monkeypatch.setenv("ENCRYPT_KEY", "encryption-secret")
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("tenant_access_token/internal"):
            return httpx.Response(200, json={"tenant_access_token": "tenant-token"})
        assert request.headers["authorization"] == "Bearer tenant-token"
        if request.url.path.endswith("/bot/v3/info"):
            return httpx.Response(200, json={"bot": {"open_id": "bot"}})
        if "/resources/" in request.url.path:
            return httpx.Response(
                200,
                content=b"resource",
                headers={
                    "Content-Type": "image/png"
                    if request.url.params["type"] == "image"
                    else "application/octet-stream"
                },
            )
        return httpx.Response(
            200, json={"data": {"image_key": "uploaded_image", "file_key": "uploaded_file"}}
        )

    transport = transport_type(
        config=ChannelConfig(
            app_id="app",
            signing_secret_env="ENCRYPT_KEY",
            allowed_users=["tenant:owner"],
            allow_groups=True,
        ),
        token="secret",
        app_token="verification",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await transport.authenticate()
    store = ChannelStore(cwd=tmp_path, transport=transport.name)
    event = {
        "header": {
            "event_id": "event",
            "event_type": "im.message.receive_v1",
            "tenant_key": "tenant",
            "app_id": "app",
            "token": "verification",
        },
        "event": {
            "sender": {"sender_type": "user", "sender_id": {"open_id": "owner"}},
            "message": {
                "message_id": "om_message",
                "chat_id": "oc_chat",
                "root_id": "om_root",
                "chat_type": "group",
                "message_type": kind,
                "content": json.dumps(content),
                "mentions": [{"id": {"open_id": "bot"}}],
            },
        },
    }
    body = json.dumps(event).encode()
    timestamp = str(int(time.time()))
    headers = {
        "X-Lark-Request-Timestamp": timestamp,
        "X-Lark-Request-Nonce": "nonce",
        "X-Lark-Signature": hashlib.sha256(
            (timestamp + "nonceencryption-secret").encode() + body
        ).hexdigest(),
    }
    try:
        await transport.handle_event(headers, body, store)
        message = store.claim_message()
        assert message is not None and len(message.attachments) == 1
        assert message.user_id == "tenant:owner" and json.loads(message.thread_id) == [
            "oc_chat",
            "om_root",
        ]
        media = await transport.prepare_media(message)
        assert base64.b64decode(media[0].data) == b"resource"
        assert "/im/v1/messages/om_message/resources/" in requests[-1].url.path
        await transport.send_media(message.thread_id, media[0], "delivery")
        upload, sent = requests[-2:]
        assert (
            "multipart/form-data" in upload.headers["content-type"]
            and b"resource" in upload.content
        )
        assert sent.url.path.endswith("/im/v1/messages/om_root/reply")
        assert json.loads(sent.content)["reply_in_thread"] is True
        assert sent.url.host == (
            "open.larksuite.com" if transport.name == "lark" else "open.feishu.cn"
        )
    finally:
        store.close()
        await transport.client.aclose()
        await transport.close()


def teams_event():
    return {
        "type": "message",
        "id": "event",
        "serviceUrl": "https://smba.trafficmanager.net/emea/",
        "channelId": "msteams",
        "channelData": {"tenant": {"id": "tenant"}},
        "from": {"id": "29:owner", "aadObjectId": "owner"},
        "recipient": {"id": "28:app"},
        "conversation": {"id": "19:conversation", "conversationType": "personal"},
        "text": "hello",
    }


@pytest.fixture
async def teams(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.host == "smba.trafficmanager.net":
            assert request.headers["authorization"] == "Bearer bot-token"
            if request.method == "GET":
                return httpx.Response(200, content=b"image", headers={"Content-Type": "image/png"})
            return httpx.Response(200, json={"id": "sent"})
        assert request.url.host.endswith(".sharepoint.com") or request.url.host.endswith(
            ".1drv.com"
        )
        assert "authorization" not in request.headers
        if request.method == "PUT":
            assert request.headers["Content-Range"] == "bytes 0-9/10"
            assert request.content == b"file bytes"
            return httpx.Response(201, json={"id": "drive-file"})
        return httpx.Response(
            200, content=b"downloaded file", headers={"Content-Type": "text/plain"}
        )

    transport = TeamsFixture(
        config=ChannelConfig(
            app_id="app", tenant_id="tenant", allowed_users=["tenant:owner"], allow_groups=True
        ),
        token="secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    await transport.authenticate()
    store = ChannelStore(cwd=tmp_path, transport="teams")
    await transport.handle_event(
        {"Authorization": "verified"}, json.dumps(teams_event()).encode(), store
    )
    message = store.claim_message()
    assert message is not None
    try:
        yield transport, store, message, requests
    finally:
        store.close()
        await transport.client.aclose()
        await transport.close()


def consent_event(consent_id, action="accept"):
    event = teams_event()
    event.update(
        type="invoke",
        name="fileConsent/invoke",
        value={
            "type": "fileUpload",
            "action": action,
            "context": {"harness_consent": consent_id},
            "uploadInfo": {
                "uploadUrl": "https://upload.1drv.com/up/token?auth=secret",
                "contentUrl": "https://tenant.sharepoint.com/personal/owner/result.txt",
                "name": "result.txt",
                "uniqueId": "file-id",
                "fileType": "txt",
            },
        },
    )
    return event


async def test_teams_consent_remains_pending_then_uploads_only_after_owner_accepts(teams):
    transport, store, message, requests = teams
    store.complete(
        message, "report", limit=4000, attachments=[file_attachment().model_dump(mode="json")]
    )
    assert await deliver_message(transport=transport, store=store)  # text
    assert await deliver_message(transport=transport, store=store)  # consent
    row = store.db.execute("SELECT * FROM outbox WHERE attachment != ''").fetchone()
    assert row["status"] == "pending" and "consent" in row["error"].lower()
    assert store.get("rate_limit_until", 0) == 0
    consent_id = store.get("teams-delivery-consent:" + row["id"])
    assert json.loads(requests[-1].content)["attachments"][0]["contentType"] == FILE_CONSENT
    before = len(requests)
    # No duplicate card while waiting. Unrelated sends can proceed.
    with pytest.raises(DeliveryDeferred):
        await transport.send_media(message.thread_id, file_attachment(), row["id"])
    await transport.send(message.thread_id, "other work", "other")
    assert len(requests) == before + 1
    event = consent_event(consent_id)
    response = await transport.handle_event(
        {"Authorization": "verified"}, json.dumps(event).encode(), store
    )
    assert response == {"status": 200} and not any(r.method == "PUT" for r in requests)
    assert store.claim_message() is None  # consent never dispatches a model
    # Recreate the transport to prove consent and destination survive a restart.
    resumed = TeamsFixture(config=transport.config, token="secret", client=transport.client)
    await resumed.authenticate()
    resumed.store = store
    try:
        await resumed.send_media(message.thread_id, file_attachment(), row["id"])
        assert requests[-2].method == "PUT"
        sent = json.loads(requests[-1].content)
        assert sent["recipient"]["id"] == "29:owner"
        assert sent["attachments"][0]["contentType"] == FILE_INFO
        assert store.get("teams-consent:" + consent_id)["status"] == "delivered"
        count = len(requests)
        await resumed.send_media(message.thread_id, file_attachment(), row["id"])
        assert len(requests) == count
    finally:
        await resumed.close()


@pytest.mark.parametrize(
    "mutation", ["sender", "conversation", "recipient", "service", "url", "unsigned"]
)
async def test_teams_file_consent_rejects_other_identity_routes_and_urls(teams, mutation):
    transport, store, message, requests = teams
    with pytest.raises(DeliveryDeferred):
        await transport.send_media(message.thread_id, file_attachment(), "delivery")
    event = consent_event(store.get("teams-delivery-consent:delivery"))
    if mutation == "sender":
        event["from"]["aadObjectId"] = "other"
    if mutation == "conversation":
        event["conversation"]["id"] = "other"
    if mutation == "recipient":
        event["recipient"]["id"] = "other"
    if mutation == "service":
        event["serviceUrl"] = "https://other.botframework.com/"
    if mutation == "url":
        event["value"]["uploadInfo"]["uploadUrl"] = "https://127.0.0.1/private"
    before = len(requests)
    with pytest.raises((AuthenticationError, ChannelError)):
        await transport.handle_event(
            {} if mutation == "unsigned" else {"Authorization": "verified"},
            json.dumps(event).encode(),
            store,
        )
    assert len(requests) == before


@pytest.mark.parametrize("action", ["decline", "expire"])
async def test_teams_declined_or_expired_consent_is_terminal(teams, action):
    transport, store, message, requests = teams
    with pytest.raises(DeliveryDeferred):
        await transport.send_media(message.thread_id, file_attachment(), "delivery")
    consent_id = store.get("teams-delivery-consent:delivery")
    if action == "decline":
        await transport.handle_event(
            {"Authorization": "verified"},
            json.dumps(consent_event(consent_id, "decline")).encode(),
            store,
        )
    else:
        record = store.get("teams-consent:" + consent_id)
        record["expires"] = 1
        store.set("teams-consent:" + consent_id, record)
    before = len(requests)
    with pytest.raises(DeliveryRejected):
        await transport.send_media(message.thread_id, file_attachment(), "delivery")
    assert len(requests) == before


async def test_teams_inline_images_and_personal_files_use_distinct_auth(teams):
    transport, _, message, requests = teams
    message.attachments[:] = [
        {
            "contentType": "image/png",
            "contentUrl": "https://smba.trafficmanager.net/emea/v3/attachments/a/views/original",
            "service_url": teams_event()["serviceUrl"],
        },
        {
            "contentType": FILE_DOWNLOAD,
            "personal": True,
            "name": "notes.txt",
            "contentUrl": "https://tenant.sharepoint.com/personal/owner/notes.txt",
            "content": {"downloadUrl": "https://download.1drv.com/file?auth=secret"},
        },
    ]
    media = await transport.prepare_media(message)
    assert [x.kind for x in media] == ["image", "file"]
    assert "authorization" not in requests[-1].headers and "authorization" in requests[-2].headers
    await transport.send_media(
        message.thread_id, file_attachment(b"png", "image.png", "image", "image/png"), "image"
    )
    assert (
        json.loads(requests[-1].content)["attachments"][0]["contentUrl"]
        == "data:image/png;base64,cG5n"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/file",
        "http://tenant.sharepoint.com/file",
        "https://tenant.sharepoint.com:8443/file",
        "https://tenant.sharepoint.com.evil.example/file",
    ],
)
async def test_teams_rejects_arbitrary_file_download_urls_without_fetch(teams, url):
    transport, _, message, requests = teams
    message.attachments[:] = [
        {
            "contentType": FILE_DOWNLOAD,
            "personal": True,
            "contentUrl": "https://tenant.sharepoint.com/a",
            "content": {"downloadUrl": url},
        }
    ]
    with pytest.raises(ChannelError):
        await transport.prepare_media(message)
    assert not requests


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self):
        self.closed = False

    async def __aiter__(self):
        yield b"1234"
        yield b"5678"
        raise AssertionError("should stop before third chunk")

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("mode", ["redirect", "length", "stream", "compression"])
async def test_native_transfer_bounds_redirects_and_closes_stream(google, mode):
    transport, _ = google
    requests = []
    stream = ChunkStream()

    def handler(request):
        requests.append(request)
        if mode == "redirect":
            return httpx.Response(302, headers={"Location": "http://127.0.0.1/private"})
        if mode == "length":
            return httpx.Response(
                200, headers={"Content-Length": str(MEDIA_LIMIT + 1)}, stream=stream
            )
        if mode == "compression":
            return httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=stream)
        return httpx.Response(200, stream=stream)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), follow_redirects=True
    ) as client:
        old = transport.client
        transport.client = client
        try:
            with pytest.raises(ChannelError):
                await download_bytes(transport, "https://chat.googleapis.com/native", remaining=5)
            assert len(requests) == 1
            if mode != "redirect":
                assert stream.closed
        finally:
            transport.client = old


async def test_teams_media_only_callback_reaches_scoped_receiver_after_durable_ack(teams, tmp_path):
    transport, store, _, requests = teams
    event = teams_event()
    event.update(
        id="media-event",
        text="",
        attachments=[
            {
                "contentType": "image/png",
                "contentUrl": "https://smba.trafficmanager.net/emea/v3/attachments/image/views/original",
            }
        ],
    )
    await transport.handle_event({"Authorization": "verified"}, json.dumps(event).encode(), store)
    assert not requests
    received = []

    async def receiver(**kwargs):
        received.append(kwargs)
        return {"reply": {"text": "I see the image"}}

    assert await process_message(cwd=tmp_path, transport=transport, store=store, receiver=receiver)
    assert len(received) == 1 and received[0]["user_id"] == "tenant:owner"
    assert json.loads(received[0]["thread_id"]) == ["tenant", "19:conversation", ""]
    assert received[0]["message"] == "" and received[0]["attachments"][0].kind == "image"
    assert len(requests) == 1


async def test_teams_completed_upload_is_not_repeated_when_info_card_needs_retry(
    teams, monkeypatch
):
    transport, store, message, requests = teams
    with pytest.raises(DeliveryDeferred):
        await transport.send_media(message.thread_id, file_attachment(), "delivery")
    consent_id = store.get("teams-delivery-consent:delivery")
    await transport.handle_event(
        {"Authorization": "verified"}, json.dumps(consent_event(consent_id)).encode(), store
    )
    send = transport._send_activity

    async def fail_card(*args, **kwargs):
        raise ChannelError("fixture interruption after upload")

    monkeypatch.setattr(transport, "_send_activity", fail_card)
    with pytest.raises(ChannelError, match="after upload"):
        await transport.send_media(message.thread_id, file_attachment(), "delivery")
    assert store.get("teams-consent:" + consent_id)["status"] == "uploaded"
    monkeypatch.setattr(transport, "_send_activity", send)
    await transport.send_media(message.thread_id, file_attachment(), "delivery")
    assert sum(request.method == "PUT" for request in requests) == 1


async def test_rejected_teams_consent_records_failed_outbox_status(teams):
    transport, store, message, _ = teams
    store.complete(
        message, "report", limit=4000, attachments=[file_attachment().model_dump(mode="json")]
    )
    await deliver_message(transport=transport, store=store)
    await deliver_message(transport=transport, store=store)
    row = store.db.execute("SELECT * FROM outbox WHERE attachment != ''").fetchone()
    consent_id = store.get("teams-delivery-consent:" + row["id"])
    await transport.handle_event(
        {"Authorization": "verified"},
        json.dumps(consent_event(consent_id, "decline")).encode(),
        store,
    )
    store.delivery_result(row["id"], status="pending")
    assert await deliver_message(transport=transport, store=store)
    status = store.db.execute("SELECT status,error FROM outbox WHERE id=?", (row["id"],)).fetchone()
    assert status["status"] == "failed" and "declined" in status["error"]


@pytest.mark.parametrize("opus", [True, False])
async def test_feishu_audio_upload_preserves_codec_and_known_duration(opus):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"data": {"file_key": "uploaded"}})

    transport = FeishuTransport(
        config=ChannelConfig(),
        token="secret",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    transport.access_token, transport.token_expires = "tenant-token", time.time() + 3600
    attachment = file_attachment(
        b"OggS" + (b"OpusHead" if opus else b"vorbis"), "clip.ogg", "audio", "audio/ogg"
    )
    attachment.duration_ms = 1200
    try:
        await transport.send_media(json.dumps(["chat", ""]), attachment, "delivery")
        assert requests[0].url.path == "/open-apis/im/v1/files"
        sent = json.loads(requests[1].content)
        assert sent["msg_type"] == ("audio" if opus else "file")
        assert (b'name="duration"' in requests[0].content) is opus
        if opus:
            assert b"1200" in requests[0].content
    finally:
        await transport.client.aclose()
        await transport.close()
