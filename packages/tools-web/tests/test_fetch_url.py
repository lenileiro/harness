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

    async def test_large_allowed_body_returns_truncated_visible_content(self) -> None:
        body = ("0123456789" * 20).encode()
        result = await _run(
            lambda _r: httpx.Response(200, headers={"content-type": "text/plain"}, content=body),
            max_bytes=1024,
            max_output_chars=25,
        )
        assert result.is_error is False
        assert "0123456789012345678901234" in result.content
        assert "01234567890123456789012345" not in result.content
        assert "[truncated: showing first 25 of 200 characters" in result.content
        assert result.metadata is not None
        assert result.metadata["bytes"] == 200
        assert result.metadata["characters"] == 200
        assert result.metadata["returned_characters"] == 25
        assert result.metadata["truncated"] is True

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

    async def test_default_approval_is_auto_read_only(self) -> None:
        tool = FetchUrlTool()
        assert tool.approval == "auto"
        assert tool.effect_scope == "read_only"


@pytest.fixture(autouse=True)
def offline_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """All HTTP tests use mock transport; never issue real DNS lookups."""
    monkeypatch.setattr(
        "harness.tools.web.socket.getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )


@pytest.mark.parametrize(
    "destination",
    [
        "http://127.0.0.1/private",
        "http://[::1]/private",
        "http://10.0.0.1/private",
        "http://169.254.169.254/metadata",
        "http://localhost/private",
        "file:///tmp/private",
    ],
)
async def test_redirect_targets_are_validated_before_request(destination: str) -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": destination})

    result = await _run(handler)
    assert result.is_error
    assert len(seen) == 1


async def test_redirect_to_private_dns_answer_is_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "harness.tools.web.socket.getaddrinfo",
        lambda host, *args: [
            (2, 1, 6, "", ("10.0.0.1" if host == "private.test" else "93.184.216.34", 443))
        ],
    )
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "https://private.test/path"})

    result = await _run(handler)
    assert result.is_error
    assert len(seen) == 1


async def test_public_relative_redirects_succeed() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.path == "/page":
            return httpx.Response(302, headers={"location": "/final"})
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="final body")

    result = await _run(handler)
    assert not result.is_error
    assert "final body" in result.content
    assert seen == ["https://example.test/page", "https://example.test/final"]


async def test_redirect_loop_is_bounded() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(302, headers={"location": "/page"})

    result = await _run(handler, max_redirects=2)
    assert result.is_error
    assert "too many redirects" in result.content
    assert len(seen) == 3


async def test_oversized_stream_stops_reading_and_closes_response() -> None:
    class Body(httpx.AsyncByteStream):
        consumed = 0
        closed = False

        async def __aiter__(self):
            for _ in range(10):
                self.consumed += 1
                yield b"x" * 600

        async def aclose(self):
            self.closed = True

    body = Body()
    result = await _run(
        lambda request: httpx.Response(200, headers={"content-type": "text/plain"}, stream=body),
        max_bytes=1024,
    )
    assert result.is_error
    assert body.consumed == 2
    assert body.closed


async def test_cross_origin_redirect_does_not_forward_authorization() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("authorization"))
        if len(seen) == 1:
            return httpx.Response(302, headers={"location": "https://another.test/page"})
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="done")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), auth=("user", "secret")
    ) as client:
        result = await FetchUrlTool(client=client)(_call("https://example.test/page"))
    assert not result.is_error
    assert seen[0] is not None
    assert seen[1] is None
