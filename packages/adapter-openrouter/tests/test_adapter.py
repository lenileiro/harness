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
    ModelSelectedEvent,
    ModelUnavailableError,
    NetworkError,
    RateLimitError,
    TextDelta,
    ToolCall,
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

    def test_model_fallbacks_can_come_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(
            "HARNESS_OPENROUTER_MODEL_FALLBACKS",
            "openai/gpt-4.1-mini, google/gemini-2.5-flash-lite",
        )
        adapter = OpenRouterAdapter(api_key="k")

        assert adapter.model_fallbacks == [
            "openai/gpt-4.1-mini",
            "google/gemini-2.5-flash-lite",
        ]

    def test_qwen_coder_is_filtered_from_configured_fallbacks_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(
            "HARNESS_OPENROUTER_MODEL_FALLBACKS",
            "qwen/qwen3-coder, openai/gpt-4.1-mini",
        )
        adapter = OpenRouterAdapter(api_key="k")

        assert adapter.model_fallbacks == ["openai/gpt-4.1-mini"]


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

    async def test_rate_limited_model_uses_explicit_fallback_before_failing(self) -> None:
        seen_models: list[str] = []
        seen_paths: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_paths.append(request.url.path)
            body = json.loads(request.content)
            seen_models.append(body["model"])
            if body["model"] == "google/gemma-4-31b-it":
                return httpx.Response(429, json={"error": {"message": "upstream rate limit"}})
            return httpx.Response(200, content=make_sse(text_chunk("ok"), "[DONE]"))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenRouterAdapter(
                api_key="k",
                client=client,
                model_fallbacks=["openai/gpt-4.1-mini"],
                auto_model_fallback=False,
            )
            events = await collect(
                adapter.stream(
                    model="google/gemma-4-31b-it",
                    messages=[Message(role="user", content="hi")],
                )
            )

        assert seen_models == ["google/gemma-4-31b-it", "openai/gpt-4.1-mini"]
        assert seen_paths == ["/api/v1/chat/completions", "/api/v1/chat/completions"]
        selected = [e for e in events if isinstance(e, ModelSelectedEvent)]
        assert len(selected) == 1
        assert selected[0].requested_model == "google/gemma-4-31b-it"
        assert selected[0].model == "openai/gpt-4.1-mini"
        assert selected[0].fallback is True
        assert isinstance(events[-1], Done)
        assert events[-1].final_message is not None
        assert events[-1].final_message.content == "ok"

    async def test_timeout_model_uses_explicit_fallback_before_failing(self) -> None:
        seen_models: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen_models.append(body["model"])
            if body["model"] == "google/gemma-4-31b-it":
                raise httpx.ReadTimeout("upstream stalled", request=request)
            return httpx.Response(200, content=make_sse(text_chunk("ok"), "[DONE]"))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenRouterAdapter(
                api_key="k",
                client=client,
                model_fallbacks=["openai/gpt-4.1-mini"],
                auto_model_fallback=False,
            )
            events = await collect(
                adapter.stream(
                    model="google/gemma-4-31b-it",
                    messages=[Message(role="user", content="hi")],
                )
            )

        assert seen_models == ["google/gemma-4-31b-it", "openai/gpt-4.1-mini"]
        selected = [e for e in events if isinstance(e, ModelSelectedEvent)]
        assert len(selected) == 1
        assert selected[0].model == "openai/gpt-4.1-mini"
        assert selected[0].fallback is True
        assert isinstance(events[-1], Done)
        assert events[-1].final_message is not None
        assert events[-1].final_message.content == "ok"

    async def test_invalid_model_id_uses_explicit_fallback_before_failing(self) -> None:
        seen_models: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            seen_models.append(body["model"])
            if body["model"] == "harness/definitely-unavailable-model":
                return httpx.Response(
                    400,
                    json={
                        "error": {
                            "message": (
                                "harness/definitely-unavailable-model is not a valid model ID"
                            ),
                            "code": 400,
                        }
                    },
                )
            return httpx.Response(200, content=make_sse(text_chunk("ok"), "[DONE]"))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenRouterAdapter(
                api_key="k",
                client=client,
                model_fallbacks=["google/gemma-4-31b-it"],
                auto_model_fallback=False,
            )
            events = await collect(
                adapter.stream(
                    model="harness/definitely-unavailable-model",
                    messages=[Message(role="user", content="hi")],
                )
            )

        assert seen_models == [
            "harness/definitely-unavailable-model",
            "google/gemma-4-31b-it",
        ]
        selected = [e for e in events if isinstance(e, ModelSelectedEvent)]
        assert len(selected) == 1
        assert selected[0].requested_model == "harness/definitely-unavailable-model"
        assert selected[0].model == "google/gemma-4-31b-it"
        assert selected[0].fallback is True
        assert isinstance(events[-1], Done)
        assert events[-1].final_message is not None
        assert events[-1].final_message.content == "ok"

    async def test_rate_limited_model_discovers_tool_capable_fallback(self) -> None:
        seen_models: list[str] = []
        model_queries: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                model_queries.append(request.url.params.get("supported_parameters"))
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": "expensive/model",
                                "context_length": 4096,
                                "pricing": {"prompt": "10", "completion": "10"},
                            },
                            {
                                "id": "cheap/tool-model",
                                "context_length": 128000,
                                "pricing": {"prompt": "0.1", "completion": "0.2"},
                            },
                        ]
                    },
                )
            body = json.loads(request.content)
            seen_models.append(body["model"])
            if body["model"] == "google/gemma-4-31b-it":
                return httpx.Response(429, json={"error": {"message": "upstream rate limit"}})
            return httpx.Response(200, content=make_sse(text_chunk("ok"), "[DONE]"))

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenRouterAdapter(
                api_key="k",
                client=client,
                model_fallbacks=[],
                model_fallback_limit=1,
            )
            events = await collect(
                adapter.stream(
                    model="google/gemma-4-31b-it",
                    messages=[Message(role="user", content="hi")],
                    tools=tools,
                )
            )

        assert seen_models == ["google/gemma-4-31b-it", "cheap/tool-model"]
        assert model_queries == ["tools"]
        assert isinstance(events[-1], Done)

    async def test_auto_discovery_skips_qwen_coder_fallback_by_default(self) -> None:
        seen_models: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/models"):
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": "qwen/qwen3-coder",
                                "context_length": 256000,
                                "pricing": {"prompt": "0.01", "completion": "0.02"},
                            },
                            {
                                "id": "openai/gpt-4.1-mini",
                                "context_length": 128000,
                                "pricing": {"prompt": "0.1", "completion": "0.2"},
                            },
                        ]
                    },
                )
            body = json.loads(request.content)
            seen_models.append(body["model"])
            if body["model"] == "google/gemma-4-31b-it":
                return httpx.Response(429, json={"error": {"message": "upstream rate limit"}})
            return httpx.Response(200, content=make_sse(text_chunk("ok"), "[DONE]"))

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a file.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenRouterAdapter(
                api_key="k",
                client=client,
                model_fallbacks=[],
                model_fallback_limit=1,
            )
            events = await collect(
                adapter.stream(
                    model="google/gemma-4-31b-it",
                    messages=[Message(role="user", content="hi")],
                    tools=tools,
                )
            )

        assert seen_models == ["google/gemma-4-31b-it", "openai/gpt-4.1-mini"]
        selected = [event for event in events if isinstance(event, ModelSelectedEvent)]
        assert selected[-1].model == "openai/gpt-4.1-mini"
        assert selected[-1].fallback is True


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

    async def test_required_tool_choice_is_forwarded(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=make_sse(text_chunk("ok"), "[DONE]"))

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenRouterAdapter(api_key="k", client=client)
            await collect(
                adapter.stream(
                    model="openai/gpt-4o",
                    messages=[Message(role="user", content="weather")],
                    tools=tools,
                    tool_choice="required",
                )
            )

        assert captured["body"]["tools"] == tools
        assert captured["body"]["tool_choice"] == "required"

    async def test_system_messages_are_sent_before_transcript_turns(self) -> None:
        captured: dict = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["body"] = json.loads(request.content)
            return httpx.Response(200, content=make_sse(text_chunk("ok"), "[DONE]"))

        messages = [
            Message(role="system", content="base instructions"),
            Message(role="user", content="first user turn"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="call_1", name="read_file", arguments={"path": "x"})],
            ),
            Message(role="tool", tool_call_id="call_1", name="read_file", content="file body"),
            Message(role="system", content="[Compacted context summary]\nolder work"),
            Message(role="user", content="next user turn"),
        ]

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenRouterAdapter(api_key="k", client=client)
            await collect(adapter.stream(model="x", messages=messages))

        sent = captured["body"]["messages"]
        assert [item["role"] for item in sent] == [
            "system",
            "system",
            "user",
            "assistant",
            "tool",
            "user",
        ]
        assert [item["content"] for item in sent[:2]] == [
            "base instructions",
            "[Compacted context summary]\nolder work",
        ]
        assert [item["content"] for item in sent[2:]] == [
            "first user turn",
            "",
            "file body",
            "next user turn",
        ]


# ---------------------------------------------------------------------------
# Model capability preflight
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestModelCapabilityPreflight:
    async def test_model_supports_tools_from_models_endpoint(self) -> None:
        requests: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(str(request.url))
            if request.url.params.get("supported_parameters") == "tools":
                return httpx.Response(
                    200,
                    json={
                        "data": [
                            {
                                "id": "google/gemma-4-26b-a4b-it",
                                "supported_parameters": ["tools", "tool_choice"],
                            }
                        ]
                    },
                )
            return httpx.Response(200, json={"data": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenRouterAdapter(api_key="k", client=client)
            supported = await adapter.model_supports_tools("openrouter/google/gemma-4-26b-a4b-it")

        assert supported is True
        assert len(requests) == 1

    async def test_model_supports_tools_returns_false_for_known_non_tool_model(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.params.get("supported_parameters") == "tools":
                return httpx.Response(
                    200,
                    json={"data": [{"id": "google/gemma-4-26b-a4b-it"}]},
                )
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "google/gemma-4-26b-a4b-it"},
                        {"id": "google/gemma-3-4b-it"},
                    ]
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenRouterAdapter(api_key="k", client=client)
            supported = await adapter.model_supports_tools("google/gemma-3-4b-it")

        assert supported is False

    async def test_model_supports_tools_returns_none_for_unknown_model(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"data": []})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenRouterAdapter(api_key="k", client=client)
            supported = await adapter.model_supports_tools("provider/new-model")

        assert supported is None


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

    async def test_provider_wrapped_rate_limit_maps_to_rate_limit(self) -> None:
        body = {"error": {"message": "google/gemma is temporarily rate-limited upstream"}}
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _r: httpx.Response(400, json=body))
        ) as client:
            adapter = OpenRouterAdapter(
                api_key="k",
                client=client,
                auto_model_fallback=False,
            )
            with pytest.raises(RateLimitError):
                await collect(
                    adapter.stream(model="x", messages=[Message(role="user", content="hi")])
                )

    async def test_tool_support_endpoint_error_maps_to_model_unavailable(self) -> None:
        body = {"error": {"message": "No endpoints found that support tool use for this model."}}
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _r: httpx.Response(400, json=body))
        ) as client:
            adapter = OpenRouterAdapter(api_key="k", client=client)
            with pytest.raises(ModelUnavailableError):
                await collect(
                    adapter.stream(
                        model="google/gemma-3-4b-it",
                        messages=[Message(role="user", content="hi")],
                        tools=[
                            {
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "description": "Read a file.",
                                    "parameters": {"type": "object", "properties": {}},
                                },
                            }
                        ],
                    )
                )
