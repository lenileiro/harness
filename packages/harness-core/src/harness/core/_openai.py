"""Private OpenAI-compatible wire helpers shared by adapters.

Both `harness-adapter-ollama` and `harness-adapter-openrouter` talk to
OpenAI-style chat-completions APIs. The SSE parsing, tool-call fragment
accumulation, and Message-to-wire conversion are identical; this module is
where they live.

The helpers are pure (no HTTP) so harness-core stays free of httpx and
similar deps. Adapters do their own transport and pipe the resulting lines
through `parse_openai_sse_stream`.

Leading underscore = private. Not part of the public harness.core API; depend
on this from inside the harness ecosystem only.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from harness.core.events import Done, Event, TextDelta, ToolCallEvent
from harness.core.schemas import Message, ToolCall


def message_to_wire(m: Message) -> dict[str, Any]:
    """Convert a harness Message to the OpenAI chat-completions wire shape."""
    out: dict[str, Any] = {"role": m.role}
    if m.content is not None:
        out["content"] = m.content
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
    idx = int(delta.get("index", 0))
    bucket = acc.setdefault(idx, {"id": "", "name": "", "args_json": ""})

    delta_id = delta.get("id")
    if delta_id:
        bucket["id"] = delta_id

    func = delta.get("function") or {}
    name = func.get("name")
    if name:
        bucket["name"] = name
    arg_fragment = func.get("arguments")
    if arg_fragment:
        bucket["args_json"] += arg_fragment


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

    async for raw in lines:
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        body = line[len("data:") :].strip()
        if body == "[DONE]":
            break
        if not body:
            continue
        try:
            chunk = json.loads(body)
        except json.JSONDecodeError:
            # Ignore non-JSON noise rather than failing the whole turn.
            continue

        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if text:
                content_chunks.append(text)
                yield TextDelta(text=text)
            for tc_delta in delta.get("tool_calls") or []:
                merge_tool_call_delta(tool_accum, tc_delta)

    final_tool_calls: list[ToolCall] = []
    for idx in sorted(tool_accum):
        agg = tool_accum[idx]
        try:
            args = json.loads(agg["args_json"]) if agg["args_json"] else {}
        except json.JSONDecodeError:
            args = {}
        call = ToolCall(
            id=agg["id"] or f"call_{idx}",
            name=agg["name"],
            arguments=args,
        )
        final_tool_calls.append(call)
        yield ToolCallEvent(call=call)

    yield Done(
        final_message=Message(
            role="assistant",
            content="".join(content_chunks) if content_chunks else None,
            tool_calls=final_tool_calls or None,
        )
    )


__all__ = ["merge_tool_call_delta", "message_to_wire", "parse_sse_stream"]
