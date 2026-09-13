"""Shared OpenAI-compatible stream contract, without HTTP or live models."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from harness.core import (
    ConfigurationError,
    InternalError,
    ModelUnavailableError,
    NetworkError,
    RateLimitError,
    TimeoutError,
)
from harness.core._openai import parse_sse_stream
from harness.core.events import Done, Event, ToolCallEvent


async def _lines(*chunks: Any) -> AsyncIterator[str]:
    for chunk in chunks:
        yield "data: " + (chunk if isinstance(chunk, str) else json.dumps(chunk))


def _tool(index: int, arguments: str, name: str = "example") -> dict[str, Any]:
    return {
        "choices": [
            {
                "delta": {
                    "tool_calls": [
                        {
                            "index": index,
                            "id": f"call-{index}",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ]
                }
            }
        ]
    }


@pytest.mark.parametrize(
    "ending", ["[DONE]", {"choices": [{"delta": {}, "finish_reason": "stop"}]}]
)
async def test_successful_completion_and_usage(ending: Any) -> None:
    events = [
        event
        async for event in parse_sse_stream(
            _lines(
                {"choices": [{"delta": {"content": "answer"}}]},
                {
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                        "prompt_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 10},
                    },
                },
                ending,
            )
        )
    ]
    done = events[-1]
    assert isinstance(done, Done)
    assert done.final_message is not None and done.final_message.content == "answer"
    assert done.usage is not None
    assert done.usage.prompt_tokens == 100
    assert done.usage.completion_tokens == 20
    assert done.usage.total_tokens == 120
    assert done.usage.cache_read_input_tokens == 40
    assert done.usage.cache_creation_input_tokens == 10


async def test_missing_usage_remains_unknown() -> None:
    events = [
        event
        async for event in parse_sse_stream(
            _lines(
                {"choices": [{"delta": {"content": "answer"}}]},
                "[DONE]",
            )
        )
    ]
    assert isinstance(events[-1], Done) and events[-1].usage is None


@pytest.mark.parametrize(
    "chunks",
    [
        [],
        [{"choices": [{"delta": {"content": "partial"}}]}],
        [_tool(0, '{"path":')],
    ],
)
async def test_premature_eof_never_completes_or_emits_tools(chunks: list[Any]) -> None:
    seen: list[Event] = []
    with pytest.raises(NetworkError, match="before completion"):
        async for event in parse_sse_stream(_lines(*chunks)):
            seen.append(event)
    assert not any(isinstance(event, (Done, ToolCallEvent)) for event in seen)


@pytest.mark.parametrize("bad_arguments", ['{"path":', "[]", "null", '"text"'])
async def test_entire_tool_batch_validates_before_any_call_is_emitted(bad_arguments: str) -> None:
    seen = []
    with pytest.raises(InternalError):
        async for event in parse_sse_stream(
            _lines(_tool(0, "{}"), _tool(1, bad_arguments), "[DONE]")
        ):
            seen.append(event)
    assert not any(isinstance(event, (Done, ToolCallEvent)) for event in seen)


@pytest.mark.parametrize("reason", ["length", "content_filter", "error"])
async def test_unsuccessful_finish_reason_is_an_error(reason: str) -> None:
    with pytest.raises(InternalError, match="ended with"):
        _ = [
            event
            async for event in parse_sse_stream(
                _lines(
                    {"choices": [{"delta": {}, "finish_reason": reason}]},
                    "[DONE]",
                )
            )
        ]


@pytest.mark.parametrize(
    "chunk", ["not JSON", "[]", {"choices": "bad"}, {"choices": [None]}, "[DONE]"]
)
async def test_invalid_stream_shape_is_typed_error(chunk: Any) -> None:
    with pytest.raises(InternalError):
        _ = [event async for event in parse_sse_stream(_lines(chunk, "[DONE]"))]


@pytest.mark.parametrize(
    ("code", "error_type"),
    [
        (401, ConfigurationError),
        (404, ModelUnavailableError),
        (429, RateLimitError),
        (504, TimeoutError),
        (500, InternalError),
    ],
)
async def test_in_band_provider_error_is_classified(code: int, error_type: type[Exception]) -> None:
    with pytest.raises(error_type, match="sentinel"):
        _ = [
            event
            async for event in parse_sse_stream(
                _lines(
                    {"error": {"code": code, "message": "sentinel"}},
                    "[DONE]",
                )
            )
        ]


async def test_nullable_optional_usage_fields_do_not_fail_valid_response() -> None:
    events = [
        event
        async for event in parse_sse_stream(
            _lines(
                {"choices": [{"delta": {"content": "answer"}}]},
                {
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": None,
                        "prompt_tokens_details": {
                            "cached_tokens": None,
                            "cache_write_tokens": None,
                        },
                    }
                },
                "[DONE]",
            )
        )
    ]
    done = events[-1]
    assert isinstance(done, Done)
    assert done.usage is not None
    assert done.usage.total_tokens == 5
    assert done.usage.cache_read_input_tokens == 0


async def test_non_token_usage_metadata_remains_unknown() -> None:
    events = [
        event
        async for event in parse_sse_stream(
            _lines(
                {"choices": [{"delta": {"content": "answer"}}]},
                {"usage": {"cost": 0.01}},
                "[DONE]",
            )
        )
    ]
    assert isinstance(events[-1], Done) and events[-1].usage is None
