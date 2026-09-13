import asyncio
import json
import time

import httpx
import pytest
from typer.testing import CliRunner

from harness.cli.account_auth import AccountAuth, OAuthAccountConfig
from harness.cli.config import load_config
from harness.cli.portal_tools import PortalConfig, PortalToolset
from harness.core import ToolCall, ToolResult


def account(tmp_path):
    auth = AccountAuth(
        "nous",
        OAuthAccountConfig(
            device_authorization_endpoint="https://auth.example/device",
            token_endpoint="https://auth.example/token",
            client_id="test",
        ),
        resource="https://model.example/v1",
        home=tmp_path / "home",
    )
    auth._store_login(
        {
            "access_token": "private-portal-token",
            "refresh_token": "private-refresh",
            "expires_at": time.time() + 3600,
        }
    )
    return auth


@pytest.mark.asyncio
async def test_account_bundle_uses_real_vendor_contracts_and_persists_media(tmp_path):
    seen = []
    cfg = PortalConfig(enabled=True, routes=("web", "images", "speech", "transcription"))

    def respond(request):
        seen.append(request)
        if request.url.host == "v3.fal.media":
            assert "authorization" not in request.headers
            return httpx.Response(
                200, content=b"image bytes", headers={"content-type": "image/webp"}
            )
        expected = (
            "Key private-portal-token"
            if "fal-queue" in request.url.host
            else "Bearer private-portal-token"
        )
        assert request.headers["authorization"] == expected
        if request.url.path == "/v2/search":
            assert json.loads(request.content) == {"query": "facts", "limit": 5}
            return httpx.Response(
                200, json={"data": {"web": [{"title": "Fact", "url": "https://example.com"}]}}
            )
        if request.url.path == "/v1/audio/speech":
            return httpx.Response(
                200, content=b"speech bytes", headers={"content-type": "audio/mpeg"}
            )
        if request.url.path == "/v1/audio/transcriptions":
            assert b"speech bytes" in request.content
            return httpx.Response(200, json={"text": "spoken words"})
        if request.url.path == "/" + cfg.image_model:
            assert request.headers["x-idempotency-key"].startswith("harness-image-")
            return httpx.Response(
                200,
                json={
                    "request_id": "job-1",
                    **{
                        key: cfg.image_url + "/jobs/1/" + key
                        for key in ("response_url", "status_url", "cancel_url")
                    },
                },
            )
        if request.url.path.endswith("/status_url"):
            return httpx.Response(200, json={"status": "COMPLETED"})
        if request.url.path.endswith("/response_url"):
            return httpx.Response(
                200, json={"images": [{"url": "https://v3.fal.media/generated.webp"}]}
            )
        raise AssertionError(str(request.url))

    async with PortalToolset(
        cfg, account(tmp_path), cwd=tmp_path, transport=httpx.MockTransport(respond)
    ) as owner:
        tools = {tool.name: tool for tool in owner.tools}
        found = await tools["web_search"](
            ToolCall(id="search", name="web_search", arguments={"query": "facts"})
        )
        assert not found.is_error and "Fact" in found.content
        image = await tools["image_generate"](
            ToolCall(id="image", name="image_generate", arguments={"prompt": "draw"})
        )
        assert not image.is_error and image.attachments[0].mime_type == "image/webp"
        speech = await tools["speech_generate"](
            ToolCall(id="speech", name="speech_generate", arguments={"text": "hello"})
        )
        assert not speech.is_error and speech.metadata is not None
        transcript = await tools["audio_transcribe"](
            ToolCall(
                id="transcript",
                name="audio_transcribe",
                arguments={"path": speech.metadata["path"]},
            )
        )
        assert transcript.content == "spoken words"
    assert len(list((tmp_path / "artifacts/media").iterdir())) == 2
    assert all(
        "private-portal-token" not in result.content
        for result in [found, image, speech, transcript]
    )


@pytest.mark.asyncio
async def test_browser_is_created_after_approval_tool_and_released_on_context_close(tmp_path):
    requests, closed = [], []

    class FakeTool:
        name = "browser"

        async def __call__(self, call):
            return ToolResult(tool_call_id=call.id, name=call.name, content="owned snapshot")

    class FakeBrowser:
        def __init__(self, config, **kwargs):
            assert (
                config.cdp_url.get_secret_value()
                == "wss://connect.browser-use.com/owned?key=private"
            )
            self.tools = [FakeTool()]

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            closed.append(True)

    def respond(request):
        requests.append(request)
        assert request.headers["x-browser-use-api-key"] == "private-portal-token"
        if request.method == "POST":
            return httpx.Response(
                200,
                json={"id": "owned-1", "cdpUrl": "wss://connect.browser-use.com/owned?key=private"},
            )
        assert request.method == "PATCH" and json.loads(request.content) == {"action": "stop"}
        return httpx.Response(200, json={})

    cfg = PortalConfig(enabled=True, routes=("browser",))
    async with PortalToolset(
        cfg,
        account(tmp_path),
        cwd=tmp_path,
        transport=httpx.MockTransport(respond),
        browser_factory=FakeBrowser,
    ) as owner:
        tools = {tool.name: tool for tool in owner.tools}
        assert tools["browser"].approval == "prompt"
        empty = await tools["browser_snapshot"](
            ToolCall(id="s", name="browser_snapshot", arguments={})
        )
        assert empty.is_error and not requests
        result = await tools["browser"](
            ToolCall(
                id="b",
                name="browser",
                arguments={"action": "navigate", "url": "https://example.com"},
            )
        )
        assert result.content == "owned snapshot"
    assert len(requests) == 2 and closed == [True]


@pytest.mark.asyncio
async def test_cancelled_generation_cancels_owned_job_without_new_submission(tmp_path):
    polling = asyncio.Event()
    requests = []
    cfg = PortalConfig(enabled=True, routes=("images",))

    def respond(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(
                200,
                json={
                    key: cfg.image_url + "/" + key
                    for key in ("status_url", "response_url", "cancel_url")
                },
            )
        if request.method == "PUT":
            return httpx.Response(200, json={})
        polling.set()
        return httpx.Response(200, json={"status": "IN_PROGRESS"})

    async with PortalToolset(
        cfg, account(tmp_path), cwd=tmp_path, transport=httpx.MockTransport(respond)
    ) as owner:
        task = asyncio.create_task(
            owner.tools[0](ToolCall(id="i", name="image_generate", arguments={"prompt": "draw"}))
        )
        await asyncio.wait_for(polling.wait(), 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert [request.method for request in requests] == ["POST", "GET", "PUT"]


def test_portal_cli_configuration_is_inspectable_and_preserves_unrelated_settings(tmp_path):
    from harness.cli.__main__ import app

    path = tmp_path / "config.toml"
    path.write_text('[approval]\nshell="deny"\n')
    result = CliRunner().invoke(
        app,
        [
            "portal",
            "configure",
            "--client-id",
            "public-test-client",
            "--model",
            "selected-model",
            "--config",
            str(path),
            "--route",
            "web",
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = load_config(path)
    assert cfg.approval["shell"] == "deny"
    assert cfg.default_model == "selected-model" and cfg.portal.routes == ("web",)
    assert cfg.provider("nous")["oauth"]["client_id"] == "public-test-client"
    assert not list(tmp_path.rglob("auth/*"))
