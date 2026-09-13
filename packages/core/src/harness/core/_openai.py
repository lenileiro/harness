"""Private OpenAI-compatible wire helpers shared by adapters.

Both the `adapter-ollama` and `adapter-openrouter` packages talk to
OpenAI-style chat-completions APIs. The SSE parsing, tool-call fragment
accumulation, and Message-to-wire conversion are identical; this module is
where they live.

The helpers are pure (no HTTP) so the `core` package stays free of httpx
and similar deps. Adapters do their own transport and pipe the resulting
lines through `parse_sse_stream`.

Leading underscore = private. Not part of the public `harness.core` API;
depend on this from inside the harness ecosystem only.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from harness.core.errors import (
    ConfigurationError,
    HarnessError,
    InternalError,
    ModelUnavailableError,
    NetworkError,
    RateLimitError,
    TimeoutError,
)
from harness.core.events import Done, Event, TextDelta, ToolCallEvent
from harness.core.schemas import MediaAttachment, Message, ToolCall, Usage


def _media_part(media: MediaAttachment) -> dict[str, Any]:
    if media.kind == "image":
        return {"type": "image_url", "image_url": {"url": media.data_uri()}}
    if media.kind == "audio":
        formats = {
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
        }
        if media.data is None or media.mime_type not in formats:
            raise ConfigurationError("Chat Completions audio input requires inline WAV or MP3 data")
        return {
            "type": "input_audio",
            "input_audio": {"data": media.data, "format": formats[media.mime_type]},
        }
    if media.data is None:
        raise ConfigurationError("Chat Completions file input requires inline data")
    return {
        "type": "file",
        "file": {"filename": media.name or "attachment", "file_data": media.data_uri()},
    }


def messages_to_wire(messages: list[Message]) -> list[dict[str, Any]]:
    """Keep tool results textual and send their media in the next user block.

    A whole contiguous tool-result group is emitted before the media block, so
    parallel tool calls always retain their required result ordering.
    """
    out: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for message in messages:
        if message.role != "tool" and pending:
            out.append({"role": "user", "content": pending})
            pending = []
        if message.role == "tool" and message.attachments:
            out.append(message_to_wire(message.model_copy(update={"attachments": []})))
            pending.append(
                {
                    "type": "text",
                    "text": f"Media returned by tool {message.name} ({message.tool_call_id}):",
                }
            )
            pending.extend(_media_part(item) for item in message.attachments if item.model_visible)
        else:
            out.append(message_to_wire(message))
    if pending:
        out.append({"role": "user", "content": pending})
    return out


def message_to_wire(m: Message) -> dict[str, Any]:
    """Convert a harness Message to the OpenAI chat-completions wire shape."""
    out: dict[str, Any] = {"role": m.role}
    if m.content is not None:
        out["content"] = m.content
    else:
        out["content"] = ""
    if m.attachments:
        if m.role != "user":
            raise ConfigurationError(
                "Media requires a user message; use messages_to_wire for tool media"
            )
        out["content"] = [
            *([{"type": "text", "text": m.content}] if m.content else []),
            *[_media_part(item) for item in m.attachments if item.model_visible],
        ]
    if m.tool_calls:
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in m.tool_calls
        ]
    if m.tool_call_id:
        out["tool_call_id"] = m.tool_call_id
    if m.name:
        out["name"] = m.name
    return out


def merge_tool_call_delta(acc: dict[int, dict[str, str]], delta: dict[str, Any]) -> None:
    """Merge a streaming tool_call delta into the per-index accumulator.

    OpenAI streams tool calls as a sequence of partial chunks indexed by
    `index`. The first chunk carries `id` and `function.name`; subsequent
    chunks append to `function.arguments`. Adapters can re-use this in
    non-streaming paths too — the accumulator becomes the source of truth.
    """
    idx = delta.get("index", 0)
    if not isinstance(idx, int) or isinstance(idx, bool) or idx < 0:
        raise InternalError("invalid tool-call index in model stream")
    bucket = acc.setdefault(idx, {"id": "", "name": "", "args_json": ""})
    delta_id = delta.get("id")
    if delta_id is not None:
        if not isinstance(delta_id, str):
            raise InternalError("invalid tool-call id in model stream")
        if delta_id:
            bucket["id"] = delta_id
    func = delta.get("function") or {}
    if not isinstance(func, dict):
        raise InternalError("invalid tool-call function in model stream")
    for field, target in (("name", "name"), ("arguments", "args_json")):
        value = func.get(field)
        if value is None:
            continue
        if not isinstance(value, str):
            raise InternalError(f"invalid tool-call {field} in model stream")
        if field == "arguments":
            bucket[target] += value
        elif value:
            bucket[target] = value


def _stream_error(error: Any) -> HarnessError:
    if not isinstance(error, dict):
        return InternalError(f"provider stream error: {error}")
    code = str(error.get("code", ""))
    message = f"provider stream error {code}: {error.get('message', 'unknown error')}"
    if code in {"401", "402", "403"}:
        return ConfigurationError(message)
    if code == "404":
        return ModelUnavailableError(message)
    if code == "429":
        return RateLimitError(message)
    if code in {"408", "504"}:
        return TimeoutError(message)
    return InternalError(message)


def _stream_usage(raw: Any) -> Usage | None:
    if raw is None or raw == {}:
        return None
    if not isinstance(raw, dict):
        raise InternalError("invalid usage in model stream")
    if not any(
        raw.get(name) is not None for name in ("prompt_tokens", "completion_tokens", "total_tokens")
    ):
        return None
    details = raw.get("prompt_tokens_details") or {}
    if not isinstance(details, dict):
        raise InternalError("invalid prompt token details in model stream")
    values = {
        "prompt_tokens": raw.get("prompt_tokens", 0),
        "completion_tokens": raw.get("completion_tokens", 0),
        "cache_creation_input_tokens": raw.get(
            "cache_creation_input_tokens", details.get("cache_write_tokens", 0)
        ),
        "cache_read_input_tokens": raw.get(
            "cache_read_input_tokens", details.get("cached_tokens", 0)
        ),
    }
    for name, value in values.items():
        if value is None:
            values[name] = 0
            continue
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise InternalError(f"invalid {name} in model stream usage")
    values["total_tokens"] = raw.get(
        "total_tokens", values["prompt_tokens"] + values["completion_tokens"]
    )
    if values["total_tokens"] is None:
        values["total_tokens"] = values["prompt_tokens"] + values["completion_tokens"]
    total = values["total_tokens"]
    if not isinstance(total, int) or isinstance(total, bool) or total < 0:
        raise InternalError("invalid total_tokens in model stream usage")
    return Usage(**values)


async def parse_sse_stream(lines: AsyncIterator[str]) -> AsyncIterator[Event]:
    """Parse OpenAI-compatible SSE lines into the normalized Event stream.

    Consumes raw text lines (already split on '\\n'), filters to `data:`
    payloads, accumulates text + tool calls, and finishes with a single
    `Done` event whose `final_message` is the assembled assistant turn.

    Adapters are responsible for HTTP transport, error mapping, and turning
    their body byte stream into a line iterator before calling this.
    """
    content_chunks: list[str] = []
    tool_accum: dict[int, dict[str, str]] = {}
    completed = False
    saw_choice = False
    usage: Usage | None = None

    async for raw in lines:
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        body = line[len("data:") :].strip()
        if body == "[DONE]":
            completed = True
            break
        if not body:
            continue
        try:
            chunk = json.loads(body)
        except json.JSONDecodeError as exc:
            raise InternalError("malformed JSON in model stream") from exc
        if not isinstance(chunk, dict):
            raise InternalError("expected an object in model stream")
        if chunk.get("error") is not None:
            raise _stream_error(chunk["error"])
        if chunk.get("usage") is not None:
            usage = _stream_usage(chunk["usage"]) or usage
        choices = chunk.get("choices", [])
        if not isinstance(choices, list):
            raise InternalError("invalid choices in model stream")
        for choice in choices:
            if not isinstance(choice, dict):
                raise InternalError("invalid choice in model stream")
            if choice.get("index", 0) != 0:
                continue
            saw_choice = True
            if choice.get("error") is not None:
                raise _stream_error(choice["error"])
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                if finish_reason not in {"stop", "tool_calls"}:
                    raise InternalError(f"model stream ended with {finish_reason!r}")
                completed = True
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                raise InternalError("invalid delta in model stream")
            text = delta.get("content")
            if text is not None and not isinstance(text, str):
                raise InternalError("invalid text in model stream")
            if text:
                content_chunks.append(text)
                yield TextDelta(text=text)
            tool_deltas = delta.get("tool_calls") or []
            if not isinstance(tool_deltas, list):
                raise InternalError("invalid tool-call deltas in model stream")
            for tc_delta in tool_deltas:
                if not isinstance(tc_delta, dict):
                    raise InternalError("invalid tool-call delta in model stream")
                merge_tool_call_delta(tool_accum, tc_delta)

    if not completed:
        raise NetworkError("model stream ended before completion")
    if not saw_choice:
        raise InternalError("model stream completed without an assistant choice")

    # Validate the entire batch before exposing any tool call for execution.
    final_tool_calls: list[ToolCall] = []
    for idx in sorted(tool_accum):
        agg = tool_accum[idx]
        try:
            args = json.loads(agg["args_json"]) if agg["args_json"] else {}
        except json.JSONDecodeError as exc:
            raise InternalError("malformed tool-call arguments in model stream") from exc
        if not isinstance(args, dict) or not agg["name"].strip():
            raise InternalError("tool calls require a name and JSON object arguments")
        final_tool_calls.append(
            ToolCall(id=agg["id"] or f"call_{idx}", name=agg["name"], arguments=args)
        )
    for call in final_tool_calls:
        yield ToolCallEvent(call=call)

    yield Done(
        final_message=Message(
            role="assistant",
            content="".join(content_chunks) if content_chunks else None,
            tool_calls=final_tool_calls or None,
        ),
        usage=usage,
    )


__all__ = ["merge_tool_call_delta", "message_to_wire", "parse_sse_stream"]
