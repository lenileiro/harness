from __future__ import annotations

import base64
from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.core import (
    Agent,
    Capabilities,
    FailoverPolicy,
    MediaAttachment,
    Message,
    RunRequest,
    ToolRegistry,
)
from harness.core._openai import messages_to_wire
from harness.core.events import ErrorEvent
from harness.core.paths import read_regular_file
from harness.core.schemas import Session

from .conftest import MockAdapter, MockStorage, text_turn


def attachment():
    return MediaAttachment(
        kind="image", mime_type="image/png", data=base64.b64encode(b"test-image").decode()
    )


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"data": "%%%%"},
        {"data": ""},
        {"url": "file:///etc/passwd"},
        {"url": "https://user:pass@example.com/a"},
        {"data": "eA==", "url": "https://example.com/a"},
    ],
)
def test_invalid_media_source_rejected(extra):
    with pytest.raises(ValidationError):
        MediaAttachment(kind="image", mime_type="image/png", **extra)


def test_old_sessions_load_and_new_media_survives_serialization():
    session = Session(
        provider="mock",
        model="m",
        cwd=Path("/tmp"),
        messages=[Message(role="user", content="hello")],
    )
    assert session.messages[0].attachments == []
    session.messages.append(Message(role="user", attachments=[attachment()]))
    assert Session.model_validate_json(session.model_dump_json()).messages[-1].attachments == [
        attachment()
    ]


def test_openai_tool_media_follows_all_parallel_results():
    messages = [
        Message(
            role="tool",
            tool_call_id="a",
            name="screen",
            content="first",
            attachments=[attachment()],
        ),
        Message(
            role="tool",
            tool_call_id="b",
            name="screen",
            content="second",
            attachments=[attachment()],
        ),
    ]
    wire = messages_to_wire(messages)
    assert [m["role"] for m in wire] == ["tool", "tool", "user"]
    assert wire[0]["content"] == "first"
    assert wire[2]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_bounded_file_loading_refuses_symlink_and_oversize(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"image")
    assert MediaAttachment.from_file(path).kind == "image"
    with pytest.raises(ValueError):
        read_regular_file(path, max_bytes=2)
    link = tmp_path / "link.png"
    link.symlink_to(path)
    with pytest.raises((OSError, ValueError)):
        MediaAttachment.from_file(link)


async def test_media_reaches_adapter_and_survives_resume():
    store = MockStorage()
    adapter = MockAdapter(
        "mock",
        capabilities=Capabilities(input_media=["image"]),
        scripts=[text_turn("seen"), text_turn("remembered")],
    )
    agent = Agent(
        adapters={"mock": adapter},
        tools=ToolRegistry(),
        storage=store,
        failover=FailoverPolicy(chain=["mock"]),
    )
    async for _ in agent.run(
        RunRequest(prompt="describe", attachments=[attachment()], model="m", session_id="media")
    ):
        pass
    async for _ in agent.resume("media", prompt="what did you see?"):
        pass
    saved = await store.get("media")
    assert saved and any(m.attachments == [attachment()] for m in saved.messages)


async def test_unsupported_adapter_reports_media_error_without_sending():
    adapter = MockAdapter("mock", scripts=[text_turn("must not happen")])
    agent = Agent(
        adapters={"mock": adapter},
        tools=ToolRegistry(),
        storage=MockStorage(),
        failover=FailoverPolicy(chain=["mock"]),
    )
    events = [
        event
        async for event in agent.run(
            RunRequest(prompt="describe", attachments=[attachment()], model="m")
        )
    ]
    assert any(isinstance(event, ErrorEvent) and "input media" in str(event) for event in events)


def test_audio_duration_is_measured_from_bytes_and_survives_storage(tmp_path):
    import wave

    from harness.core.media_metadata import audio_duration_ms

    path = tmp_path / "quarter-second.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b"\0\0" * 2000)
    media = MediaAttachment.from_file(path)
    assert media.duration_ms == 250
    assert MediaAttachment.model_validate_json(media.model_dump_json()).duration_ms == 250
    assert audio_duration_ms(b"not valid audio") is None
    for value in [0, -1, True, 1.5, 86_400_001]:
        with pytest.raises(ValidationError):
            MediaAttachment(kind="audio", mime_type="audio/wav", data=media.data, duration_ms=value)
