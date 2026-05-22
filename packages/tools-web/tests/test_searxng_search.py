"""Tests for SearXNGSearchTool. Uses httpx.MockTransport — no network I/O."""

from __future__ import annotations

import json

import httpx
import pytest

from harness.core import ToolCall, ToolResult
from harness.tools.web import SearXNGSearchTool


def _call(query: str, **extra: object) -> ToolCall:
    return ToolCall(id="c1", name="web_search", arguments={"query": query, **extra})


def _json_response(results: list[dict]) -> bytes:
    return json.dumps({"results": results, "query": "test"}).encode()


_SAMPLE_RESULTS = [
    {
        "title": "Python 3.12 Release Notes",
        "content": "New features in Python 3.12 including improved error messages.",
        "url": "https://docs.python.org/3.12/whatsnew/3.12.html",
        "engine": "google",
    },
    {
        "title": "Real Python — Python 3.12 Guide",
        "content": "A practical guide to what changed in 3.12.",
        "url": "https://realpython.com/python312",
        "engine": "bing",
    },
]


async def _run(
    handler,
    query: str = "Python 3.12",
    base_url: str = "http://searxng.test",
    **call_extra,
) -> ToolResult:
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = SearXNGSearchTool(base_url=base_url, client=client)
        return await tool(_call(query, **call_extra))


@pytest.mark.asyncio
class TestSearXNGSearchTool:
    async def test_returns_results_with_title_url_snippet(self) -> None:
        def handler(_r):
            return httpx.Response(
                200,
                content=_json_response(_SAMPLE_RESULTS),
                headers={"content-type": "application/json"},
            )

        result = await _run(handler)
        assert result.is_error is False
        assert "Python 3.12 Release Notes" in result.content
        assert "https://docs.python.org" in result.content
        assert "New features" in result.content

    async def test_engine_name_shown_in_results(self) -> None:
        def handler(_r):
            return httpx.Response(
                200,
                content=_json_response(_SAMPLE_RESULTS),
                headers={"content-type": "application/json"},
            )

        result = await _run(handler)
        assert "[google]" in result.content

    async def test_result_count_in_metadata(self) -> None:
        def handler(_r):
            return httpx.Response(
                200,
                content=_json_response(_SAMPLE_RESULTS),
                headers={"content-type": "application/json"},
            )

        result = await _run(handler)
        assert result.metadata is not None
        assert result.metadata["result_count"] == 2
        assert result.metadata["backend"] == "searxng"

    async def test_empty_results_returns_non_error_message(self) -> None:
        def handler(_r):
            return httpx.Response(
                200, content=_json_response([]), headers={"content-type": "application/json"}
            )

        result = await _run(handler, query="xyzzy_no_results")
        assert result.is_error is False
        assert "no results" in result.content

    async def test_empty_query_returns_error(self) -> None:
        def handler(_r):
            return httpx.Response(
                200, content=_json_response([]), headers={"content-type": "application/json"}
            )

        result = await _run(handler, query="   ")
        assert result.is_error is True
        assert "query" in result.content

    async def test_403_returns_helpful_error(self) -> None:
        def handler(_r):
            return httpx.Response(403, content=b"Forbidden")

        result = await _run(handler)
        assert result.is_error is True
        assert "403" in result.content or "JSON format" in result.content

    async def test_non_json_response_returns_error(self) -> None:
        def handler(_r):
            return httpx.Response(
                200, content=b"<html>not json</html>", headers={"content-type": "text/html"}
            )

        result = await _run(handler)
        assert result.is_error is True
        assert "JSON" in result.content

    async def test_max_results_respected(self) -> None:
        many = [
            {
                "title": f"Result {i}",
                "content": f"Snippet {i}",
                "url": f"https://example.com/{i}",
                "engine": "ddg",
            }
            for i in range(10)
        ]

        def handler(_r):
            return httpx.Response(
                200, content=_json_response(many), headers={"content-type": "application/json"}
            )

        result = await _run(handler, max_results=3)
        # Only 3 results should be rendered (numbered 1, 2, 3 but not 4)
        assert "4." not in result.content

    async def test_numbered_format(self) -> None:
        def handler(_r):
            return httpx.Response(
                200,
                content=_json_response(_SAMPLE_RESULTS),
                headers={"content-type": "application/json"},
            )

        result = await _run(handler)
        assert "1." in result.content
        assert "2." in result.content

    async def test_approval_is_auto(self) -> None:
        assert SearXNGSearchTool().approval == "auto"

    async def test_name_is_web_search(self) -> None:
        assert SearXNGSearchTool().name == "web_search"

    async def test_custom_base_url_forwarded(self) -> None:
        seen_urls: list[str] = []

        def handler(r: httpx.Request):
            seen_urls.append(str(r.url))
            return httpx.Response(
                200,
                content=_json_response(_SAMPLE_RESULTS),
                headers={"content-type": "application/json"},
            )

        await _run(handler, base_url="http://mysearxng:9090")
        assert seen_urls and "mysearxng:9090" in seen_urls[0]
