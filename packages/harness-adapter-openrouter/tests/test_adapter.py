"""OpenRouterAdapter tests using httpx.MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from harness.adapters.openrouter import OpenRouterAdapter
from harness.core import (
    ConfigurationError,
    Done,
    Event,
    InternalError,
    Message,
    ModelUnavailableError,
    NetworkError,
    RateLimitError,
    TextDelta,
    ToolCallEvent,
)


def make_sse(*events: dict | str) -> bytes:
    lines = []
    for ev in events:
        body = ev if isinstance(ev, str) else json.dumps(ev)
        lines.append(f"data: {body}")
        lines.append("")
    return ("\n".join(lines) + "\n").encode()


def text_chunk(text: str) -> dict:
    return {"choices": [{"delta": {"content": text}, "index": 0}]}


def tool_chunk(
    *,
    index: int = 0,
    call_id: str | None = None,
    name: str | None = None,
    args_fragment: str | None = None,
) -> dict:
    delta: dict = {"index": index}
    if call_id is not None:
        delta["id"] = call_id
    func: dict = {}
    if name is not None:
        func["name"] = name
    if args_fragment is not None:
        func["arguments"] = args_fragment
    if func:
        delta["function"] = func
    return {"choices": [{"delta": {"tool_calls": [delta]}, "index": 0}]}


async def collect(it) -> list[Event]:
    out: list[Event] = []
    async for e in it:
        out.append(e)
    return out


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_missing_api_key_raises_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with pytest.raises(ConfigurationError):
            OpenRouterAdapter()

    def test_env_key_is_picked_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        adapter = OpenRouterAdapter()
        assert adapter.api_key == "test-key"

    def test_explicit_key_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-key")
        adapter = OpenRouterAdapter(api_key="explicit")
        assert adapter.api_key == "explicit"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStream:
    async def test_text_response(self) -> None:
        body = make_sse(text_chunk("hi "), text_chunk("there"), "[DONE]")
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=body))
        ) as client:
            adapter = OpenRouterAdapter(api_key="k", client=client)
            events = await collect(
                adapter.stream(
                    model="anthropic/claude-3.5-sonnet",
                    messages=[Message(role="user", content="hi")],
                )
            )
        assert [e.text for e in events if isinstance(e, TextDelta)] == ["hi ", "there"]
        done = events[-1]
        assert isinstance(done, Done)
        assert done.final_message is not None
        assert done.final_message.content == "hi there"

    async def test_tool_call_accumulation(self) -> None:
        body = make_sse(
            tool_chunk(call_id="c1", name="ping"),
            tool_chunk(args_fragment='{"host":"x.com"}'),
            "[DONE]",
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _r: httpx.Response(200, content=body))
        ) as client:
            adapter = OpenRouterAdapter(api_key="k", client=client)
            events = await collect(
                adapter.stream(
                    model="openai/gpt-4o", messages=[Message(role="user", content="ping x")]
                )
            )
        tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_events) == 1
        assert tool_events[0].call.name == "ping"
        assert tool_events[0].call.arguments == {"host": "x.com"}


# ---------------------------------------------------------------------------
# Headers and wire format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWireAndHeaders:
    async def test_includes_openrouter_headers(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            captured["url"] = str(request.url)
            return httpx.Response(200, content=make_sse(text_chunk("ok"), "[DONE]"))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenRouterAdapter(
                api_key="my-key",
                http_referer="https://example.test",
                x_title="MyApp",
                client=client,
            )
            await collect(
                adapter.stream(
                    model="anthropic/claude-3.5-sonnet",
                    messages=[Message(role="user", content="hi")],
                )
            )

        assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
        assert captured["headers"]["authorization"] == "Bearer my-key"
        assert captured["headers"]["http-referer"] == "https://example.test"
        assert captured["headers"]["x-title"] == "MyApp"

    async def test_optional_headers_omitted_when_none(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["headers"] = dict(request.headers)
            return httpx.Response(200, content=make_sse(text_chunk("ok"), "[DONE]"))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenRouterAdapter(api_key="k", http_referer=None, x_title=None, client=client)
            await collect(adapter.stream(model="x", messages=[Message(role="user", content="hi")]))
        assert "http-referer" not in captured["headers"]
        assert "x-title" not in captured["headers"]


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestErrorMapping:
    @pytest.mark.parametrize(
        ("status", "expected"),
        [
            (401, ConfigurationError),
            (402, ConfigurationError),
            (404, ModelUnavailableError),
            (429, RateLimitError),
            (500, InternalError),
            (503, InternalError),
        ],
    )
    async def test_http_status_maps_to_typed_error(
        self, status: int, expected: type[Exception]
    ) -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _r: httpx.Response(status, content=b"err"))
        ) as client:
            adapter = OpenRouterAdapter(api_key="k", client=client)
            with pytest.raises(expected):
                await collect(
                    adapter.stream(model="x", messages=[Message(role="user", content="hi")])
                )

    async def test_connect_error_maps_to_network(self) -> None:
        def raise_conn(_r: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("nope")

        async with httpx.AsyncClient(transport=httpx.MockTransport(raise_conn)) as client:
            adapter = OpenRouterAdapter(api_key="k", client=client)
            with pytest.raises(NetworkError):
                await collect(
                    adapter.stream(model="x", messages=[Message(role="user", content="hi")])
                )
