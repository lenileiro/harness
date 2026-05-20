"""Tests for FetchUrlTool. Uses httpx.MockTransport — no network I/O."""

from __future__ import annotations

import httpx
import pytest

from harness.core import ToolCall
from harness.tools.web import FetchUrlTool


def _call(url: str, **extra: object) -> ToolCall:
    return ToolCall(id="c1", name="fetch_url", arguments={"url": url, **extra})


async def _run(handler, **tool_kwargs):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        tool = FetchUrlTool(client=client, **tool_kwargs)
        return await tool(_call("https://example.test/page"))


@pytest.mark.asyncio
class TestFetchUrl:
    async def test_text_response(self) -> None:
        result = await _run(
            lambda _r: httpx.Response(
                200, headers={"content-type": "text/plain"}, content=b"hello world"
            )
        )
        assert result.is_error is False
        assert "hello world" in result.content
        assert "status: 200" in result.content
        assert "text/plain" in result.content

    async def test_json_response_allowed(self) -> None:
        result = await _run(
            lambda _r: httpx.Response(
                200, headers={"content-type": "application/json"}, content=b'{"a":1}'
            )
        )
        assert result.is_error is False
        assert '{"a":1}' in result.content

    async def test_non_2xx_is_error(self) -> None:
        result = await _run(
            lambda _r: httpx.Response(
                404, headers={"content-type": "text/plain"}, content=b"missing"
            )
        )
        assert result.is_error is True
        assert "HTTP 404" in result.content

    async def test_disallowed_mime_refused(self) -> None:
        result = await _run(
            lambda _r: httpx.Response(
                200, headers={"content-type": "image/png"}, content=b"\x89PNG"
            )
        )
        assert result.is_error is True
        assert "allow-list" in result.content

    async def test_oversized_body_refused(self) -> None:
        big = b"x" * 4096
        result = await _run(
            lambda _r: httpx.Response(200, headers={"content-type": "text/plain"}, content=big),
            max_bytes=1024,
        )
        assert result.is_error is True
        assert "too large" in result.content

    async def test_non_http_scheme_refused(self) -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"x"))
        ) as client:
            tool = FetchUrlTool(client=client)
            result = await tool(_call("file:///etc/passwd"))
        assert result.is_error is True
        assert "scheme" in result.content

    async def test_missing_host_refused(self) -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=b"x"))
        ) as client:
            tool = FetchUrlTool(client=client)
            result = await tool(_call("https:///nope"))
        assert result.is_error is True
        assert "host" in result.content

    async def test_connect_error_returns_typed_error(self) -> None:
        def boom(_r: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("nope")

        result = await _run(boom)
        assert result.is_error is True
        assert "connection error" in result.content

    async def test_default_approval_is_prompt(self) -> None:
        tool = FetchUrlTool()
        assert tool.approval == "prompt"
