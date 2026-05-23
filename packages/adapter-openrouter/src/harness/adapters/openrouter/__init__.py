"""OpenRouter adapter for Harness.

OpenRouter (https://openrouter.ai/) exposes an OpenAI-compatible chat
completions API across many model providers. SSE parsing and message-wire
conversion live in `harness.core._openai` (shared with the Ollama adapter).

Differences from the Ollama adapter:
- Requires `OPENROUTER_API_KEY` in env (or passed explicitly).
- Sends optional `HTTP-Referer` and `X-Title` headers for OpenRouter's
  analytics — both are configurable.
- 401 (auth) maps to ConfigurationError (terminal, not retryable).
- 402 (out of credits) maps to ConfigurationError as well — failing over
  to a different provider with the same key won't help.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from harness.core import (
    Capabilities,
    ConfigurationError,
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


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterAdapter:
    """Streaming adapter for OpenRouter's OpenAI-compatible API.

    Args:
        api_key: OpenRouter API key. Falls back to $OPENROUTER_API_KEY.
                 Required; ConfigurationError raised at construction otherwise.
        base_url: Override the API base. Defaults to https://openrouter.ai/api/v1.
        http_referer: Sent as `HTTP-Referer` for OpenRouter analytics. Optional.
        x_title: Sent as `X-Title`. Optional.
        timeout: Streaming request timeout in seconds.
        client: Optional pre-built httpx.AsyncClient (lets tests inject a
                MockTransport).
    """

    name = "openrouter"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        http_referer: str | None = "https://github.com/lenileiro/harness",
        x_title: str | None = "harness",
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise ConfigurationError(
                "OpenRouter API key missing: pass api_key= or set OPENROUTER_API_KEY"
            )
        self.api_key = key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.http_referer = http_referer
        self.x_title = x_title
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
        return self._stream(
            model=model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=kwargs.get("response_format"),
            seed=kwargs.get("seed"),
        )

    async def capabilities(self) -> Capabilities:
        # OpenRouter routes to many models; tool-use availability depends on
        # the chosen model. We report True and let the model gate it.
        return Capabilities(streaming=True, tool_use=True)

    async def cancel(self, session_id: str) -> None:
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
        response_format: dict[str, Any] | str | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[Event]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message_to_wire(m) for m in messages],
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if seed is not None:
            payload["seed"] = seed

        url = f"{self.base_url}/chat/completions"
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

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
                    f"could not connect to OpenRouter at {self.base_url}: {exc}"
                ) from exc
            except httpx.TimeoutException as exc:
                raise TimeoutError(f"OpenRouter request timed out: {exc}") from exc
            except httpx.HTTPError as exc:
                raise InternalError(f"OpenRouter HTTP error: {exc}") from exc
        finally:
            if owns_client:
                await client.aclose()

    async def _raise_for_status(self, response: httpx.Response) -> None:
        body = await response.aread()
        text = body.decode("utf-8", errors="replace") if body else ""
        status = response.status_code
        if status == 401:
            raise ConfigurationError(f"OpenRouter 401 (auth failed). Body: {text}")
        if status == 402:
            raise ConfigurationError(f"OpenRouter 402 (out of credits). Body: {text}")
        if status == 404:
            raise ModelUnavailableError(f"OpenRouter 404 (model not found). Body: {text}")
        if status == 429:
            raise RateLimitError(f"OpenRouter rate-limited (429). Body: {text}")
        raise InternalError(f"OpenRouter HTTP {status}. Body: {text}")


__all__ = ["DEFAULT_BASE_URL", "OpenRouterAdapter", "__version__"]
