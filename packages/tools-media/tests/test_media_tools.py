import base64
import json

import httpx
import pytest

from harness.core import ToolCall
from harness.tools.media import MediaConfig, MediaToolset


@pytest.mark.asyncio
async def test_real_http_contracts_persist_outputs_and_transcribe(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDIA_TEST_KEY", "private-media-key")
    seen = []

    def respond(request):
        seen.append(request)
        assert request.headers["authorization"] == "Bearer private-media-key"
        if request.url.path.endswith("images/generations"):
            payload = json.loads(request.content)
            assert payload["model"] == "image-test" and payload["prompt"] == "draw"
            return httpx.Response(
                200, json={"data": [{"b64_json": base64.b64encode(b"image-data").decode()}]}
            )
        if request.url.path.endswith("audio/speech"):
            return httpx.Response(
                200, content=b"audio-data", headers={"content-type": "audio/mpeg"}
            )
        assert b"audio-data" in request.content and b"transcription-test" in request.content
        return httpx.Response(200, json={"text": "Transcript"})

    config = MediaConfig(
        enabled=True,
        api_key_env="MEDIA_TEST_KEY",
        image_model="image-test",
        speech_model="speech-test",
        transcription_model="transcription-test",
    )
    async with MediaToolset(
        config, cwd=tmp_path, transport=httpx.MockTransport(respond)
    ) as toolset:
        tools = {tool.name: tool for tool in toolset.tools}
        image = await tools["image_generate"](
            ToolCall(id="i", name="image_generate", arguments={"prompt": "draw"})
        )
        assert not image.is_error and image.attachments[0].kind == "image"
        speech = await tools["speech_generate"](
            ToolCall(id="s", name="speech_generate", arguments={"text": "Speak"})
        )
        assert not speech.is_error and not speech.attachments[0].model_visible
        assert speech.metadata is not None
        transcript = await tools["audio_transcribe"](
            ToolCall(id="t", name="audio_transcribe", arguments={"path": speech.metadata["path"]})
        )
        assert transcript.content == "Transcript"
    assert len(seen) == 3
    assert len(list((tmp_path / "artifacts/media").iterdir())) == 2


@pytest.mark.asyncio
async def test_no_retry_or_secret_leak_on_http_error(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDIA_TEST_KEY", "secret")
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(429, text="secret provider response")

    async with MediaToolset(
        MediaConfig(enabled=True, api_key_env="MEDIA_TEST_KEY", image_model="m"),
        cwd=tmp_path,
        transport=httpx.MockTransport(respond),
    ) as toolset:
        result = await toolset.tools[-1](
            ToolCall(id="i", name="image_generate", arguments={"prompt": "draw"})
        )
        assert result.is_error and "429" in result.content and "secret" not in result.content
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_media_path_boundary_and_response_size(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDIA_TEST_KEY", "secret")
    private = tmp_path / ".harness"
    private.mkdir()
    (private / "key.png").write_bytes(b"private")
    async with MediaToolset(MediaConfig(enabled=True), cwd=tmp_path) as toolset:
        for path in [".harness/key.png", "../outside.png"]:
            result = await toolset.tools[0](
                ToolCall(id="r", name="read_media", arguments={"path": path})
            )
            assert result.is_error and not result.attachments
    async with MediaToolset(
        MediaConfig(enabled=True, api_key_env="MEDIA_TEST_KEY", speech_model="m", max_bytes=1024),
        cwd=tmp_path,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"x" * 90000)),
    ) as toolset:
        result = await toolset.tools[-1](
            ToolCall(id="s", name="speech_generate", arguments={"text": "hello"})
        )
        assert result.is_error and "size limit" in result.content


@pytest.mark.asyncio
@pytest.mark.parametrize("output_format", ["jpeg", "webp"])
async def test_image_format_is_preserved_on_attachment_and_artifact(
    tmp_path, monkeypatch, output_format
):
    from pathlib import Path

    monkeypatch.setenv("MEDIA_TEST_KEY", "secret")
    config = MediaConfig(
        enabled=True,
        api_key_env="MEDIA_TEST_KEY",
        image_model="m",
        image_parameters={"output_format": output_format},
    )
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json={"data": [{"b64_json": base64.b64encode(b"image-data").decode()}]}
        )
    )
    async with MediaToolset(config, cwd=tmp_path, transport=transport) as toolset:
        result = await toolset.tools[-1](
            ToolCall(id="i", name="image_generate", arguments={"prompt": "draw"})
        )
    assert not result.is_error
    assert result.attachments[0].mime_type == f"image/{output_format}"
    assert result.metadata is not None
    assert Path(result.metadata["path"]).suffix == f".{output_format}"


@pytest.mark.asyncio
async def test_generated_speech_carries_real_duration_for_channel_delivery(tmp_path, monkeypatch):
    from pathlib import Path

    from harness.core import MediaAttachment

    monkeypatch.setenv("MEDIA_TEST_KEY", "secret")
    raw = (Path(__file__).parent / "fixtures/silence.mp3").read_bytes()
    async with MediaToolset(
        MediaConfig(enabled=True, api_key_env="MEDIA_TEST_KEY", speech_model="voice"),
        cwd=tmp_path,
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=raw, headers={"content-type": "audio/mpeg"})
        ),
    ) as owner:
        result = await owner.tools[-1](
            ToolCall(id="speech", name="speech_generate", arguments={"text": "Hello"})
        )
    assert not result.is_error, result.content
    media = result.attachments[0]
    assert media.duration_ms is not None and 200 <= media.duration_ms <= 400
    assert result.metadata is not None
    reloaded = MediaAttachment.from_file(Path(result.metadata["path"]))
    assert reloaded.duration_ms == media.duration_ms
