from __future__ import annotations

import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr, ValidationError

from harness.core import ToolCall
from harness.tools.browser import BrowserConfig, BrowserToolset

PAGE = b"""<!doctype html><html><title>Browser fixture</title><body>
<h1>Local fixture</h1><label>Name <input id="name" name="name"></label>
<button id="submit" onclick="document.querySelector('#result').textContent='Hello '+document.querySelector('#name').value">Submit</button>
<p id="result">Ready</p><a href="/next">Next page</a>
<a id="download" href="/download">Download file</a>
<label>Upload <input id="upload" type="file" onchange="document.querySelector('#result').textContent=this.files[0].name"></label>
<select id="choice" onchange="document.querySelector('#result').textContent=this.value"><option value="one">One</option><option value="two">Two</option></select>
</body></html>"""


@pytest.fixture
def website():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/download":
                body = b"downloaded fixture"
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", 'attachment; filename="fixture.txt"')
            else:
                body = (
                    PAGE
                    if self.path != "/next"
                    else b"<html><body><h1>Next document</h1></body></html>"
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


async def invoke(toolset: BrowserToolset, action: str, **arguments):
    tool = toolset.tools[0]
    return await tool(
        ToolCall(id="call", name="browser", arguments={"action": action, **arguments})
    )


def content(result):
    assert not result.is_error, result.content
    return json.loads(result.content)


async def test_navigation_refs_typing_select_and_back(tmp_path, website):
    async with BrowserToolset(BrowserConfig(), cwd=tmp_path) as tools:
        initial = content(await invoke(tools, "navigate", url=website))
        assert initial["title"] == "Browser fixture"
        assert "Local fixture" in initial["snapshot"]
        field = next(item["ref"] for item in initial["elements"] if item["name"].strip() == "Name")
        updated = content(await invoke(tools, "type", ref=field, text="Ada"))
        assert (await invoke(tools, "type", ref=field, text="stale")).is_error
        button = next(item["ref"] for item in updated["elements"] if item["name"] == "Submit")
        submitted = content(await invoke(tools, "click", ref=button))
        assert "Hello Ada" in submitted["snapshot"]
        selected = content(await invoke(tools, "select", selector="#choice", value="two"))
        assert "two" in selected["snapshot"]
        next_page = content(await invoke(tools, "click", selector="a[href='/next']"))
        assert "Next document" in next_page["snapshot"]
        assert "Local fixture" in content(await invoke(tools, "back"))["snapshot"]
        read_only = await tools.tools[1](
            ToolCall(id="snapshot", name="browser_snapshot", arguments={})
        )
        assert "Local fixture" in content(read_only)["snapshot"]
    assert tools.context is None
    assert (await invoke(tools, "snapshot")).is_error


async def test_upload_download_and_screenshot_are_workspace_artifacts(tmp_path, website):
    (tmp_path / "upload.txt").write_text("uploaded fixture")
    async with BrowserToolset(BrowserConfig(), cwd=tmp_path) as tools:
        content(await invoke(tools, "navigate", url=website))
        uploaded = content(await invoke(tools, "upload", selector="#upload", paths=["upload.txt"]))
        assert "upload.txt" in uploaded["snapshot"]
        clicked = content(await invoke(tools, "click", selector="#download"))
        assert clicked["downloads"]
        download = content(
            await invoke(tools, "download", download_id=clicked["downloads"][0]["download_id"])
        )
        assert (tmp_path / download["path"]).read_text() == "downloaded fixture"
        result = await invoke(tools, "screenshot")
        screenshot = content(result)
        assert len(result.attachments) == 1
        raw = base64.b64decode(result.attachments[0].data or "")
        assert raw.startswith(b"\x89PNG\r\n\x1a\n")
        assert (tmp_path / screenshot["path"]).read_bytes() == raw


async def test_browser_blocks_local_file_navigation_and_private_upload_aliases(tmp_path, website):
    root = tmp_path / "workspace"
    root.mkdir()
    private = root / ".harness"
    private.mkdir()
    (private / "secret.txt").write_text("PRIVATE_SENTINEL")
    (root / "alias").symlink_to(private, target_is_directory=True)
    (tmp_path / "outside.txt").write_text("outside")
    async with BrowserToolset(BrowserConfig(), cwd=root) as tools:
        for url in ("file:///etc/passwd", "javascript:alert(1)", "data:text/html,private"):
            assert (await invoke(tools, "navigate", url=url)).is_error
        content(await invoke(tools, "navigate", url=website))
        for path in (".harness/secret.txt", "alias/secret.txt", "../outside.txt"):
            result = await invoke(tools, "upload", selector="#upload", paths=[path])
            assert result.is_error and "PRIVATE_SENTINEL" not in result.content


async def test_tabs_are_isolated_and_limited(tmp_path, website):
    async with BrowserToolset(BrowserConfig(max_tabs=2), cwd=tmp_path) as first:
        one = content(await invoke(first, "navigate", url=website))
        two = content(await invoke(first, "new_tab", url=website + "/next"))
        assert one["tab_id"] != two["tab_id"]
        assert (await invoke(first, "new_tab")).is_error
        assert len(content(await invoke(first, "tabs"))["tabs"]) == 2
        content(await invoke(first, "close_tab", tab_id=two["tab_id"]))
        async with BrowserToolset(BrowserConfig(), cwd=tmp_path) as second:
            assert (await invoke(second, "snapshot", tab_id=one["tab_id"])).is_error
            assert content(await invoke(second, "tabs"))["tabs"][0]["url"] == "about:blank"


async def test_cancellation_closes_owned_browser_context(tmp_path, monkeypatch):
    tools = BrowserToolset(BrowserConfig(), cwd=tmp_path)
    entered = asyncio.Event()
    held = asyncio.Event()

    async def use_browser():
        async with tools:
            entered.set()
            await held.wait()

    task = asyncio.create_task(use_browser())
    await asyncio.wait_for(entered.wait(), 10)
    assert tools.context is not None and tools.browser is not None
    browser = tools.browser
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert tools.context is None and not browser.is_connected()


async def test_cdp_creates_private_context_and_never_uses_existing_pages(tmp_path, monkeypatch):
    from harness.tools.browser import toolset as module

    page = MagicMock()
    context = MagicMock()
    context.route = AsyncMock()
    context.new_page = AsyncMock(return_value=page)
    context.close = AsyncMock()
    browser = SimpleNamespace(
        new_context=AsyncMock(return_value=context),
        close=AsyncMock(),
        contexts=["private-other-user-context"],
    )
    driver = SimpleNamespace(
        chromium=SimpleNamespace(
            connect_over_cdp=AsyncMock(return_value=browser), launch=AsyncMock()
        ),
        stop=AsyncMock(),
    )
    monkeypatch.setattr(
        module, "async_playwright", lambda: SimpleNamespace(start=AsyncMock(return_value=driver))
    )
    config = BrowserConfig(
        backend="cdp", cdp_url=SecretStr("wss://browser.example.test/devtools?token=secret")
    )
    async with BrowserToolset(config, cwd=tmp_path):
        assert config.cdp_url is not None
        driver.chromium.connect_over_cdp.assert_awaited_once_with(
            config.cdp_url.get_secret_value(), timeout=30000
        )
        browser.new_context.assert_awaited_once()
        driver.chromium.launch.assert_not_awaited()
    context.close.assert_awaited_once()
    browser.close.assert_awaited_once()
    driver.stop.assert_awaited_once()


async def test_cdp_failure_does_not_launch_local_browser(tmp_path, monkeypatch):
    from harness.tools.browser import toolset as module

    driver = SimpleNamespace(
        chromium=SimpleNamespace(
            connect_over_cdp=AsyncMock(side_effect=RuntimeError("offline")), launch=AsyncMock()
        ),
        stop=AsyncMock(),
    )
    monkeypatch.setattr(
        module, "async_playwright", lambda: SimpleNamespace(start=AsyncMock(return_value=driver))
    )
    with pytest.raises(RuntimeError, match="offline"):
        async with BrowserToolset(
            BrowserConfig(backend="cdp", cdp_url=SecretStr("http://example.test")), cwd=tmp_path
        ):
            pytest.fail("CDP failure must not initialize")
    driver.chromium.launch.assert_not_awaited()
    driver.stop.assert_awaited_once()


@pytest.mark.parametrize("directory", ["../outside", ".harness/browser"])
async def test_artifact_scope_rejected_before_browser_launch(tmp_path, directory):
    with pytest.raises(ValueError):
        async with BrowserToolset(BrowserConfig(artifact_directory=directory), cwd=tmp_path):
            pytest.fail("invalid artifact scope")


def test_config_requires_explicit_cdp_endpoint():
    with pytest.raises(ValidationError):
        BrowserConfig(backend="cdp")
    config = BrowserConfig(backend="cdp", cdp_url=SecretStr("https://example.test?token=secret"))
    assert "token=secret" not in repr(config)
