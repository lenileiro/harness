"""Ollama adapter for Harness.

Talks to Ollama's OpenAI-compatible endpoint (`/v1/chat/completions`). The
SSE parsing and tool-call accumulation live in `harness.core._openai` so they
stay in sync with the OpenRouter adapter.

Maps HTTP status / network errors onto the harness.core error hierarchy so
FailoverPolicy can classify them.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from harness.core import (
    Capabilities,
    Event,
    InternalError,
    Message,
    ModelUnavailableError,
    NetworkError,
    RateLimitError,
    TimeoutError,
)
from harness.core._openai import message_to_wire, parse_sse_stream

__version__ = "0.0.0"


DEFAULT_BASE_URL = "http://localhost:11434"


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
        **kwargs: Any,
    ) -> AsyncIterator[Event]:
        tool_choice = kwargs.get("tool_choice")
        return self._stream(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            tool_choice=tool_choice,
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
        tool_choice: str | None = None,
    ) -> AsyncIterator[Event]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message_to_wire(m) for m in messages],
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
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
                    async for event in parse_sse_stream(response.aiter_lines()):
                        yield event
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
        raise InternalError(f"Ollama HTTP {status}. Body: {text}")


__all__ = ["DEFAULT_BASE_URL", "OllamaAdapter", "__version__"]
