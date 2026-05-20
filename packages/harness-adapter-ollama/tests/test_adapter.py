"""OllamaAdapter tests using httpx.MockTransport to simulate responses."""

from __future__ import annotations

import json

import httpx
import pytest

from harness.adapters.ollama import OllamaAdapter, _merge_tool_call
from harness.core import (
    Done,
    Event,
    InternalError,
    Message,
    ModelUnavailableError,
    NetworkError,
    RateLimitError,
    TextDelta,
    ToolCall,
    ToolCallEvent,
)


def make_sse(*events: dict | str) -> bytes:
    """Build an SSE byte stream from a sequence of JSON chunks (or [DONE])."""
    lines = []
    for ev in events:
        body = ev if isinstance(ev, str) else json.dumps(ev)
        lines.append(f"data: {body}")
        lines.append("")  # SSE blank-line separator
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
# _merge_tool_call (pure unit)
# ---------------------------------------------------------------------------


class TestMergeToolCall:
    def test_first_chunk_sets_id_and_name(self) -> None:
        acc: dict[int, dict[str, str]] = {}
        _merge_tool_call(acc, {"index": 0, "id": "c1", "function": {"name": "ping"}})
        assert acc[0] == {"id": "c1", "name": "ping", "args_json": ""}

    def test_argument_fragments_concatenate(self) -> None:
        acc: dict[int, dict[str, str]] = {}
        _merge_tool_call(acc, {"index": 0, "id": "c1", "function": {"name": "ping"}})
        _merge_tool_call(acc, {"index": 0, "function": {"arguments": '{"host":'}})
        _merge_tool_call(acc, {"index": 0, "function": {"arguments": '"x.com"}'}})
        assert acc[0]["args_json"] == '{"host":"x.com"}'

    def test_multiple_indices_track_separately(self) -> None:
        acc: dict[int, dict[str, str]] = {}
        _merge_tool_call(acc, {"index": 0, "id": "a", "function": {"name": "x"}})
        _merge_tool_call(acc, {"index": 1, "id": "b", "function": {"name": "y"}})
        assert set(acc) == {0, 1}
        assert acc[0]["name"] == "x"
        assert acc[1]["name"] == "y"


# ---------------------------------------------------------------------------
# Streaming text response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStreamText:
    async def test_emits_deltas_and_final_done(self) -> None:
        body = make_sse(text_chunk("hello "), text_chunk("world"), "[DONE]")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _req: httpx.Response(200, content=body))
        ) as client:
            adapter = OllamaAdapter(client=client)
            events = await collect(
                adapter.stream(model="llama3.2", messages=[Message(role="user", content="hi")])
            )

        deltas = [e for e in events if isinstance(e, TextDelta)]
        assert [d.text for d in deltas] == ["hello ", "world"]

        done = events[-1]
        assert isinstance(done, Done)
        assert done.final_message is not None
        assert done.final_message.role == "assistant"
        assert done.final_message.content == "hello world"
        assert done.final_message.tool_calls is None

    async def test_skips_empty_and_non_data_lines(self) -> None:
        body = (
            b"\n: comment\ndata:\n\ndata: "
            + json.dumps(text_chunk("ok")).encode()
            + b"\n\ndata: [DONE]\n\n"
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _req: httpx.Response(200, content=body))
        ) as client:
            adapter = OllamaAdapter(client=client)
            events = await collect(
                adapter.stream(model="llama3.2", messages=[Message(role="user", content="hi")])
            )
        deltas = [e for e in events if isinstance(e, TextDelta)]
        assert [d.text for d in deltas] == ["ok"]


# ---------------------------------------------------------------------------
# Tool-call streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStreamToolCalls:
    async def test_aggregates_argument_fragments(self) -> None:
        body = make_sse(
            tool_chunk(index=0, call_id="c1", name="ping"),
            tool_chunk(index=0, args_fragment='{"host":'),
            tool_chunk(index=0, args_fragment='"x.com"}'),
            "[DONE]",
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _req: httpx.Response(200, content=body))
        ) as client:
            adapter = OllamaAdapter(client=client)
            events = await collect(
                adapter.stream(model="llama3.2", messages=[Message(role="user", content="hi")])
            )

        # Exactly one ToolCallEvent, plus the Done.
        tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert len(tool_events) == 1
        assert tool_events[0].call.name == "ping"
        assert tool_events[0].call.arguments == {"host": "x.com"}

        done = next(e for e in events if isinstance(e, Done))
        assert done.final_message is not None
        assert done.final_message.tool_calls is not None
        assert done.final_message.tool_calls[0].id == "c1"

    async def test_multiple_parallel_tool_calls(self) -> None:
        body = make_sse(
            tool_chunk(index=0, call_id="c1", name="a", args_fragment="{}"),
            tool_chunk(index=1, call_id="c2", name="b", args_fragment="{}"),
            "[DONE]",
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _req: httpx.Response(200, content=body))
        ) as client:
            adapter = OllamaAdapter(client=client)
            events = await collect(
                adapter.stream(model="llama3.2", messages=[Message(role="user", content="hi")])
            )
        tool_events = [e for e in events if isinstance(e, ToolCallEvent)]
        assert [t.call.id for t in tool_events] == ["c1", "c2"]
        assert [t.call.name for t in tool_events] == ["a", "b"]


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestErrorMapping:
    async def test_404_maps_to_model_unavailable(self) -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _req: httpx.Response(404, content=b"model not found")
            )
        ) as client:
            adapter = OllamaAdapter(client=client)
            with pytest.raises(ModelUnavailableError):
                await collect(
                    adapter.stream(model="missing", messages=[Message(role="user", content="hi")])
                )

    async def test_429_maps_to_rate_limit(self) -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _req: httpx.Response(429, content=b"slow down"))
        ) as client:
            adapter = OllamaAdapter(client=client)
            with pytest.raises(RateLimitError):
                await collect(
                    adapter.stream(model="llama3.2", messages=[Message(role="user", content="hi")])
                )

    async def test_500_maps_to_internal(self) -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _req: httpx.Response(500, content=b"boom"))
        ) as client:
            adapter = OllamaAdapter(client=client)
            with pytest.raises(InternalError):
                await collect(
                    adapter.stream(model="llama3.2", messages=[Message(role="user", content="hi")])
                )

    async def test_connection_refused_maps_to_network(self) -> None:
        def raise_conn(_req: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        async with httpx.AsyncClient(transport=httpx.MockTransport(raise_conn)) as client:
            adapter = OllamaAdapter(client=client)
            with pytest.raises(NetworkError):
                await collect(
                    adapter.stream(model="llama3.2", messages=[Message(role="user", content="hi")])
                )


# ---------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWireFormat:
    async def test_request_payload_has_expected_shape(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=make_sse(text_chunk("ok"), "[DONE]"))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OllamaAdapter(client=client, base_url="http://example:11434")
            await collect(
                adapter.stream(
                    model="llama3.2",
                    messages=[
                        Message(role="system", content="be concise"),
                        Message(role="user", content="hi"),
                    ],
                    tools=[
                        {
                            "type": "function",
                            "function": {
                                "name": "ping",
                                "description": "",
                                "parameters": {"type": "object", "properties": {}},
                            },
                        }
                    ],
                    temperature=0.2,
                    max_tokens=64,
                )
            )

        assert captured["url"] == "http://example:11434/v1/chat/completions"
        assert captured["headers"]["authorization"] == "Bearer ollama"
        body = captured["body"]
        assert body["model"] == "llama3.2"
        assert body["stream"] is True
        assert body["temperature"] == 0.2
        assert body["max_tokens"] == 64
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        assert body["tools"][0]["function"]["name"] == "ping"

    async def test_assistant_with_tool_calls_serializes(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=make_sse(text_chunk("ok"), "[DONE]"))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OllamaAdapter(client=client)
            messages = [
                Message(role="user", content="what time"),
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=[ToolCall(id="c1", name="now", arguments={"tz": "UTC"})],
                ),
                Message(role="tool", tool_call_id="c1", name="now", content="2026-05-21T00:00:00Z"),
            ]
            await collect(adapter.stream(model="llama3.2", messages=messages))

        wire = captured["body"]["messages"]
        assert wire[1]["tool_calls"][0]["function"]["name"] == "now"
        assert json.loads(wire[1]["tool_calls"][0]["function"]["arguments"]) == {"tz": "UTC"}
        assert wire[2]["tool_call_id"] == "c1"
        assert wire[2]["role"] == "tool"
