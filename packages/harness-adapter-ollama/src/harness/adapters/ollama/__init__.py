"""Ollama adapter for Harness.

Talks to Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`). Using
the OpenAI shape means this adapter shares its parsing logic with
harness-adapter-openrouter and other OpenAI-compatible providers.

Streams via SSE. Maps HTTP status / network errors onto the harness.core
error hierarchy so FailoverPolicy can classify them.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from harness.core import (
    Capabilities,
    Done,
    Event,
    InternalError,
    Message,
    ModelUnavailableError,
    NetworkError,
    RateLimitError,
    TextDelta,
    TimeoutError,
    ToolCall,
    ToolCallEvent,
)

__version__ = "0.0.0"


DEFAULT_BASE_URL = "http://localhost:11434"


def _msg_to_wire(m: Message) -> dict[str, Any]:
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


class OllamaAdapter:
    """Streaming adapter for a local Ollama daemon.

    Args:
        base_url: Ollama server URL. Defaults to OLLAMA_HOST env or localhost:11434.
        api_key: Ollama ignores auth; any non-empty string works for the header.
        timeout: Request timeout in seconds for the streaming connection.
        client: Optional pre-built httpx.AsyncClient (lets tests inject a
                MockTransport). The adapter does NOT take ownership — caller
                closes it.
    """

    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str = "ollama",
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._injected_client = client

    # ------------------------------------------------------------------ #
    # Adapter Protocol                                                    #
    # ------------------------------------------------------------------ #

    def stream(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **_kwargs: Any,
    ) -> AsyncIterator[Event]:
        return self._stream(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=True)

    async def cancel(self, session_id: str) -> None:
        # The httpx context manager handles teardown on cancellation; nothing
        # we can do for an in-flight server-side completion.
        return None

    # ------------------------------------------------------------------ #
    # Streaming implementation                                            #
    # ------------------------------------------------------------------ #

    async def _stream(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> AsyncIterator[Event]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_msg_to_wire(m) for m in messages],
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        url = f"{self.base_url}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        owns_client = self._injected_client is None
        client = self._injected_client or httpx.AsyncClient(timeout=self.timeout)

        try:
            try:
                async with client.stream("POST", url, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        await self._raise_for_status(response)

                    content_chunks: list[str] = []
                    tool_accum: dict[int, dict[str, str]] = {}

                    async for raw in response.aiter_lines():
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
                                _merge_tool_call(tool_accum, tc_delta)

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

                    final_message = Message(
                        role="assistant",
                        content="".join(content_chunks) if content_chunks else None,
                        tool_calls=final_tool_calls or None,
                    )
                    yield Done(final_message=final_message)
            except httpx.ConnectError as exc:
                raise NetworkError(
                    f"could not connect to Ollama at {self.base_url}: {exc}"
                ) from exc
            except httpx.TimeoutException as exc:
                raise TimeoutError(f"Ollama request timed out: {exc}") from exc
            except httpx.HTTPError as exc:
                raise InternalError(f"Ollama HTTP error: {exc}") from exc
        finally:
            if owns_client:
                await client.aclose()

    async def _raise_for_status(self, response: httpx.Response) -> None:
        body = await response.aread()
        text = body.decode("utf-8", errors="replace") if body else ""
        status = response.status_code
        if status == 404:
            raise ModelUnavailableError(
                f"Ollama returned 404 — model likely not pulled. Body: {text}"
            )
        if status == 429:
            raise RateLimitError(f"Ollama rate-limited (429). Body: {text}")
        if 500 <= status < 600:
            raise InternalError(f"Ollama HTTP {status}. Body: {text}")
        raise InternalError(f"Ollama HTTP {status}. Body: {text}")


def _merge_tool_call(acc: dict[int, dict[str, str]], delta: dict[str, Any]) -> None:
    """Merge a streaming tool_call delta into the per-index accumulator.

    OpenAI streams tool calls as a sequence of partial chunks indexed by `index`.
    The first chunk carries `id` and `function.name`; subsequent chunks append
    to `function.arguments`.
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


__all__ = ["DEFAULT_BASE_URL", "OllamaAdapter", "__version__"]
