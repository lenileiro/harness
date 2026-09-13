from __future__ import annotations

import asyncio
import json
import struct
import threading
import zlib
from pathlib import Path

import pytest

from harness.core import ToolCall
from harness.tools.computer import ComputerConfig, ComputerToolset


def png(width, height):
    def chunk(kind, data):
        return (
            struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress((b"\0" + b"\xff" * 4 * width) * height))
        + chunk(b"IEND", b"")
    )


class FakeImage:
    def __init__(self, size):
        self.size = size

    def resize(self, size):
        return FakeImage(size)

    def save(self, stream, format):
        assert format == "PNG"
        stream.write(png(*self.size))


class Driver:
    FAILSAFE = False
    KEYBOARD_KEYS = ("ctrl", "alt", "a", "enter", "tab")

    def __init__(self):
        self.calls = []

    def size(self):
        return 200, 100

    def position(self):
        return 10, 20

    def screenshot(self):
        self.calls.append(("screenshot",))
        return FakeImage((400, 200))

    def click(self, *args, **kwargs):
        self.calls.append(("click", args, kwargs))

    def moveTo(self, *args, **kwargs):
        self.calls.append(("move", args, kwargs))

    def dragTo(self, *args, **kwargs):
        self.calls.append(("drag", args, kwargs))

    def write(self, *args, **kwargs):
        self.calls.append(("type", args, kwargs))

    def press(self, *args, **kwargs):
        self.calls.append(("press", args, kwargs))

    def hotkey(self, *args, **kwargs):
        self.calls.append(("hotkey", args, kwargs))

    def scroll(self, *args, **kwargs):
        self.calls.append(("scroll", args, kwargs))


def make_toolset(tmp_path, driver=None, **kwargs):
    return ComputerToolset(
        ComputerConfig(enabled=True, **kwargs),
        cwd=tmp_path,
        driver=driver or Driver(),
        lock_path=tmp_path / "desktop.lock",
    )


async def invoke(toolset, **arguments):
    return await toolset.tools[0](ToolCall(id="call", name="computer", arguments=arguments))


async def test_enabled_configuration_and_desktop_ownership(tmp_path):
    disabled = ComputerToolset(
        ComputerConfig(), cwd=tmp_path, driver=Driver(), lock_path=tmp_path / "lock"
    )
    with pytest.raises(ValueError, match="enabled=true"):
        await disabled.__aenter__()
    first = make_toolset(tmp_path)
    second = make_toolset(tmp_path)
    async with first:
        assert first.driver.FAILSAFE is True
        with pytest.raises(RuntimeError, match="owns"):
            await second.__aenter__()
        assert first.tools[0].approval == "prompt"
        assert first.tools[0].effect_scope == "external_side_effect"
    assert first.driver.FAILSAFE is False
    async with second:
        assert second.active


async def test_screenshot_coordinates_artifact_and_media_agree(tmp_path):
    async with make_toolset(tmp_path) as tools:
        result = await invoke(tools, action="screenshot")
        assert not result.is_error
        assert result.metadata is not None
        assert (result.metadata["screen_width"], result.metadata["screen_height"]) == (200, 100)
        path = Path(result.metadata["path"])
        raw = await asyncio.to_thread(path.read_bytes)
        assert struct.unpack("!2I", raw[16:24]) == (200, 100)
        assert len(result.attachments) == 1 and result.attachments[0].model_visible
        assert result.attachments[0].mime_type == "image/png"
        assert path.is_relative_to(tmp_path / "artifacts/computer")
    assert (await invoke(tools, action="screenshot")).is_error


async def test_pointer_keyboard_and_scroll_routes_are_real_driver_operations(tmp_path):
    async with make_toolset(tmp_path) as tools:
        for action in ["click", "move", "drag"]:
            assert not (await invoke(tools, action=action, x=20, y=30)).is_error
        for arguments in [
            dict(action="type", text="Hello"),
            dict(action="press", key="enter"),
            dict(action="hotkey", keys=["ctrl", "a"]),
            dict(action="scroll", amount=-3),
        ]:
            assert not (await invoke(tools, **arguments)).is_error
        assert [call[0] for call in tools.driver.calls] == [
            "click",
            "move",
            "drag",
            "type",
            "press",
            "hotkey",
            "scroll",
        ]
        assert tools.driver.calls[0][1] == (20, 30)
        assert tools.driver.calls[-2][1] == ("ctrl", "a")
        position = await invoke(tools, action="position")
        assert json.loads(position.content)["x"] == 10


@pytest.mark.parametrize(
    "arguments",
    [
        dict(action="click", x=200, y=30),
        dict(action="click", x=20),
        dict(action="press", key="not-a-key"),
        dict(action="hotkey", keys=[]),
        dict(action="type", text="Привет"),
        dict(action="scroll", amount=101),
    ],
)
async def test_invalid_or_unsupported_input_never_reaches_driver(tmp_path, arguments):
    async with make_toolset(tmp_path) as tools:
        assert (await invoke(tools, **arguments)).is_error
        assert not tools.driver.calls


@pytest.mark.parametrize(
    "directory", ["../outside", "/tmp/outside", ".harness/screens", "ok/.harness"]
)
async def test_private_or_escaping_artifact_configuration_fails_before_desktop_access(
    tmp_path, directory
):
    tools = make_toolset(tmp_path, artifact_directory=directory)
    with pytest.raises(ValueError):
        await tools.__aenter__()
    assert not tools.driver.calls


async def test_artifact_symlink_is_rechecked_after_context_open(tmp_path):
    async with make_toolset(tmp_path) as tools:
        artifacts = tmp_path / "artifacts"
        artifacts.mkdir()
        (tmp_path / "outside").mkdir()
        (artifacts / "computer").symlink_to(tmp_path / "outside", target_is_directory=True)
        result = await invoke(tools, action="screenshot")
        assert result.is_error and "symlink" in result.content
        assert not list((tmp_path / "outside").iterdir())


async def test_cancellation_waits_for_issued_action_before_releasing_desktop(tmp_path):
    started = threading.Event()
    release = threading.Event()

    class BlockingDriver(Driver):
        def click(self, *args, **kwargs):
            started.set()
            assert release.wait(3)
            super().click(*args, **kwargs)

    tools = make_toolset(tmp_path, driver=BlockingDriver())
    await tools.__aenter__()
    action = asyncio.create_task(invoke(tools, action="click", x=10, y=20))
    assert await asyncio.to_thread(started.wait, 2)
    action.cancel()
    closing = asyncio.create_task(tools.__aexit__())
    await asyncio.sleep(0)
    assert not closing.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await action
    await closing
    assert len(tools.driver.calls) == 1 and not tools.active
    async with make_toolset(tmp_path):
        pass


async def test_typing_cancellation_does_not_send_remaining_chunks(tmp_path):
    started = threading.Event()
    release = threading.Event()

    class BlockingTypingDriver(Driver):
        def write(self, *args, **kwargs):
            started.set()
            assert release.wait(3)
            super().write(*args, **kwargs)

    async with make_toolset(tmp_path, driver=BlockingTypingDriver()) as tools:
        action = asyncio.create_task(invoke(tools, action="type", text="a" * 100))
        assert await asyncio.to_thread(started.wait, 2)
        action.cancel()
        await asyncio.sleep(0)
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await action
        assert len(tools.driver.calls) == 1
        assert tools.driver.calls[0][1] == ("a" * 32,)
