"""Ollama/OpenRouter must preserve the shared parser contract over HTTP."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from harness.adapters.ollama import OllamaAdapter
from harness.adapters.openrouter import OpenRouterAdapter
from harness.core import InternalError, Message, NetworkError, RateLimitError
from harness.core.events import Done, ToolCallEvent


@pytest.fixture(params=["ollama", "openrouter"])
def provider(request: pytest.FixtureRequest) -> str:
    return request.param


def _adapter(provider: str, client: httpx.AsyncClient) -> OllamaAdapter | OpenRouterAdapter:
    if provider == "ollama":
        return OllamaAdapter(client=client)
    return OpenRouterAdapter(
        api_key="offline", client=client, model_fallbacks=[], auto_model_fallback=False
    )


def _sse(*chunks: Any) -> bytes:
    return "".join(
        "data: " + (chunk if isinstance(chunk, str) else json.dumps(chunk)) + "\n\n"
        for chunk in chunks
    ).encode()


async def test_adapter_requests_and_reports_usage(provider: str) -> None:
    payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=_sse(
                {"choices": [{"delta": {"content": "done"}, "finish_reason": "stop"}]},
                {
                    "choices": [],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
                },
                "[DONE]",
            ),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        events = [
            event
            async for event in _adapter(provider, client).stream(
                model="test",
                messages=[Message(role="user", content="test")],
            )
        ]
    assert payloads[0]["stream_options"] == {"include_usage": True}
    done = events[-1]
    assert isinstance(done, Done)
    assert done.usage is not None and done.usage.total_tokens == 15


@pytest.mark.parametrize(
    ("body", "error_type"),
    [
        (_sse({"choices": [{"delta": {"content": "partial"}}]}), NetworkError),
        (_sse({"error": {"code": 429, "message": "offline rate limit"}}), RateLimitError),
        (
            _sse(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"name": "example", "arguments": '{"x":'},
                                    }
                                ]
                            }
                        }
                    ]
                },
                "[DONE]",
            ),
            InternalError,
        ),
    ],
)
async def test_adapter_never_completes_or_emits_tools_for_bad_stream(
    provider: str, body: bytes, error_type: type[Exception]
) -> None:
    seen = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=body))
    ) as client:
        with pytest.raises(error_type):
            async for event in _adapter(provider, client).stream(
                model="test", messages=[Message(role="user", content="test")]
            ):
                seen.append(event)
    assert not any(isinstance(event, (Done, ToolCallEvent)) for event in seen)
