"""Tests for PlaywrightSearchTool (DDG HTML backend). Uses httpx.MockTransport."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from harness.core import ToolCall, ToolResult
from harness.tools.web import PlaywrightSearchTool


def _call(query: str, **extra: object) -> ToolCall:
    return ToolCall(id="c1", name="web_search", arguments={"query": query, **extra})


_DDG_HTML = b"""
<html><body>
<div class="serp__results">
  <div class="result results_links web-result">
    <div class="result__body">
      <a class="result__a" href="/l/?uddg=https://docs.python.org/3.12/">Python 3.12 Release Notes</a>
      <span class="result__url">docs.python.org/3.12/whatsnew/3.12.html</span>
      <a class="result__snippet">New features including improved error messages.</a>
    </div>
  </div>
  <div class="result results_links web-result">
    <div class="result__body">
      <a class="result__a" href="/l/?uddg=https://realpython.com/python312">Real Python Guide</a>
      <span class="result__url">realpython.com/python312</span>
      <a class="result__snippet">Practical walkthrough of what changed.</a>
    </div>
  </div>
</div>
</body></html>
"""


async def _run(
    handler,
    query: str = "Python 3.12",
    **call_extra,
) -> ToolResult:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = PlaywrightSearchTool(client=client)
        return await tool(_call(query, **call_extra))


@pytest.mark.asyncio
class TestPlaywrightSearchTool:
    async def test_returns_results_with_title_snippet_url(self) -> None:
        def handler(_r):
            return httpx.Response(200, content=_DDG_HTML, headers={"content-type": "text/html"})

        result = await _run(handler)
        assert result.is_error is False
        assert "Python 3.12 Release Notes" in result.content
        assert "docs.python.org" in result.content
        assert "New features" in result.content

    async def test_metadata_has_backend_playwright_chromium(self) -> None:
        def handler(_r):
            return httpx.Response(200, content=_DDG_HTML, headers={"content-type": "text/html"})

        result = await _run(handler)
        assert result.metadata is not None
        assert result.metadata["backend"] == "playwright-chromium"
        assert result.metadata["result_count"] == 2

    async def test_empty_query_returns_error(self) -> None:
        def handler(_r):
            return httpx.Response(200, content=_DDG_HTML, headers={"content-type": "text/html"})

        result = await _run(handler, query="")
        assert result.is_error is True
        assert "query" in result.content

    async def test_no_results_returns_non_error(self) -> None:
        empty_html = b"<html><body><div class='serp__results'></div></body></html>"

        def handler(_r):
            return httpx.Response(200, content=empty_html, headers={"content-type": "text/html"})

        result = await _run(handler, query="xyzzy")
        assert result.is_error is False
        assert "no results" in result.content

    async def test_http_error_returns_error(self) -> None:
        def handler(_r):
            return httpx.Response(503, content=b"Service Unavailable")

        result = await _run(handler)
        assert result.is_error is True
        assert "503" in result.content

    async def test_max_results_respected(self) -> None:
        def handler(_r):
            return httpx.Response(200, content=_DDG_HTML, headers={"content-type": "text/html"})

        result = await _run(handler, max_results=1)
        assert result.metadata is not None
        assert result.metadata["result_count"] == 1

    async def test_max_results_capped_at_20(self) -> None:
        captured: list[int] = []

        async def fake_search(self_: object, q: str, n: int) -> list[dict]:
            captured.append(n)
            return []

        tool = PlaywrightSearchTool()
        with patch.object(PlaywrightSearchTool, "_search", fake_search):
            await tool(_call("query", max_results=999))
        assert captured[0] == 20

    async def test_approval_is_auto(self) -> None:
        assert PlaywrightSearchTool().approval == "auto"

    async def test_name_is_web_search(self) -> None:
        assert PlaywrightSearchTool().name == "web_search"

    async def test_numbered_results_format(self) -> None:
        def handler(_r):
            return httpx.Response(200, content=_DDG_HTML, headers={"content-type": "text/html"})

        result = await _run(handler)
        assert "1." in result.content
        assert "2." in result.content

    async def test_missing_browser_returns_helpful_error(self) -> None:
        tool = PlaywrightSearchTool()
        with patch.object(
            PlaywrightSearchTool,
            "_search",
            AsyncMock(side_effect=Exception("Executable doesn't exist; run playwright install")),
        ):
            result: ToolResult = await tool(_call("test"))
        assert result.is_error is True
