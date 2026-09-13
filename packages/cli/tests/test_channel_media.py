from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from harness.cli.channels.runtime import deliver_message, process_message
from harness.cli.channels.transports import DiscordTransport, SlackTransport, TelegramTransport
from harness.core.gateway_channels import ChannelConfig, ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment


@pytest.mark.parametrize("field", ["browser", "execution"])
async def test_remote_gateway_rejects_configured_host_backends(tmp_path, field):
    from harness.cli.gateway_runtime import _run_gateway_chat_turn
    from harness.core.errors import ConfigurationError

    config = SimpleNamespace(**{field: object()})
    with pytest.raises(ConfigurationError, match="unavailable for remote"):
        await _run_gateway_chat_turn(
            cwd=tmp_path,
            prompt="hello",
            chain=["fake"],
            model="fake",
            session_id="session",
            max_steps=1,
            config=config,
            system_prompt="",
            transport="telegram",
            user_id="123",
        )


def test_gateway_receive_accepts_media_from_bridge_stdin(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from harness.cli.gateway_commands import gateway_app

    captured = []

    async def receive(**kwargs):
        captured.append(kwargs)
        return {"reply": {"text": "Image received"}}

    monkeypatch.setattr("harness.cli.gateway_commands._run_gateway_receive_payload", receive)
    attachment = {"kind": "image", "mime_type": "image/png", "data": "aW1hZ2U="}
    result = CliRunner().invoke(
        gateway_app,
        [
            "receive",
            "--cwd",
            str(tmp_path),
            "--user",
            "123",
            "--message",
            "Describe this",
            "--attachments-stdin",
            "--json",
        ],
        input=json.dumps([attachment]),
    )
    assert result.exit_code == 0, result.output
    assert captured[0]["attachments"][0].kind == "image"


@pytest.mark.parametrize("name", ["telegram", "discord", "slack"])
def test_media_reaches_gateway_and_durable_upload(name, tmp_path):
    data = b"image-bytes"
    requests = []

    def request(request):
        requests.append(request)
        path = request.url.path
        if path.endswith("getFile"):
            return httpx.Response(
                200, json={"ok": True, "result": {"file_path": "photos/image.jpg"}}
            )
        if request.method == "GET":
            if name == "slack":
                assert request.headers["authorization"] == "Bearer token"
            return httpx.Response(200, content=data)
        if path.endswith("files.getUploadURLExternal"):
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "upload_url": "https://files.slack.com/upload/v1/private",
                    "file_id": "F1",
                },
            )
        if path.startswith("/upload/"):
            assert request.content == data
            return httpx.Response(200, content=b"OK")
        return httpx.Response(200, json={"ok": True, "result": {"id": "sent"}, "id": "sent"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
            cls = {
                "telegram": TelegramTransport,
                "discord": DiscordTransport,
                "slack": SlackTransport,
            }[name]
            user = "T1:U1" if name == "slack" else "123"
            thread = "T1:D1:" if name == "slack" else "123"
            transport = cls(
                config=ChannelConfig(allowed_users=[user]),
                token="token",
                app_token="app",
                client=client,
            )
            if isinstance(transport, SlackTransport):
                transport.team_id = "T1"
            descriptor = (
                {"file_id": "file", "mime_type": "image/jpeg", "name": "image.jpg"}
                if name == "telegram"
                else {
                    "url": "https://files.slack.com/image"
                    if name == "slack"
                    else "https://cdn.discordapp.com/image",
                    "mime_type": "image/jpeg",
                    "name": "image.jpg",
                }
            )
            store = ChannelStore(cwd=tmp_path, transport=name)
            store.ingest(
                ChannelMessage(
                    "media", user, thread, "Describe this", thread, attachments=[descriptor]
                )
            )

            async def receiver(**kwargs):
                image = kwargs["attachments"][0]
                assert isinstance(image, MediaAttachment)
                assert image.data is not None
                assert base64.b64decode(image.data) == data
                return {
                    "reply": {
                        "text": "Here is the image",
                        "data": {"attachments": [image.model_dump(mode="json")]},
                    }
                }

            await process_message(cwd=tmp_path, transport=transport, store=store, receiver=receiver)
            store.close()
            store = ChannelStore(cwd=tmp_path, transport=name)
            assert await deliver_message(transport=transport, store=store)
            assert await deliver_message(transport=transport, store=store)
            assert all(item["status"] == "sent" for item in store.status()["outbox"])
            if name in {"telegram", "discord"}:
                upload = requests[-1]
                assert "multipart/form-data" in upload.headers["content-type"]
                assert data in upload.content and b"image/jpeg" in upload.content
            else:
                completed = json.loads(requests[-1].content)
                assert completed["channel_id"] == "D1" and completed["files"][0]["id"] == "F1"
            store.close()
            await transport.close()

    asyncio.run(run())


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/private",
        "https://files.slack.com.evil.example/private",
        "file:///etc/passwd",
        "https://user:password@files.slack.com/private",
    ],
)
def test_media_never_fetches_arbitrary_host_or_local_paths(url):
    async def run():
        def request(request):
            raise AssertionError("Unsafe media URL must fail before network access")

        async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
            transport = SlackTransport(config=ChannelConfig(), token="secret", client=client)
            message = ChannelMessage(
                "id", "T1:U1", "T1:D1:", "media", "T1:D1", attachments=[{"url": url}]
            )
            with pytest.raises(ValueError, match="authenticated platform"):
                await transport.prepare_media(message)
            await transport.close()

    asyncio.run(run())
