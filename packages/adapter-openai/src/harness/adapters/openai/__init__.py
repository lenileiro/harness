"""OpenAI adapter for Harness.

OpenAI exposes an OpenAI-compatible chat completions API. SSE parsing and
message-wire conversion live in `harness.core._openai` (shared with the other
OpenAI-compatible adapters).
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
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


DEFAULT_BASE_URL = "https://api.openai.com/v1"


def inspect_codex_openai_auth() -> dict[str, str | bool] | None:
    """Return minimal Codex auth metadata without exposing secrets."""

    auth_path = Path.home() / ".codex" / "auth.json"
    if not auth_path.exists():
        return None
    try:
        raw = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    auth_mode = raw.get("auth_mode")
    value = raw.get("OPENAI_API_KEY")
    return {
        "auth_mode": auth_mode if isinstance(auth_mode, str) else "unknown",
        "has_openai_api_key": bool(isinstance(value, str) and value.strip()),
    }


def load_codex_openai_api_key() -> str | None:
    """Best-effort fallback for Codex installations using API-key auth.

    Codex stores login state in ``~/.codex/auth.json``. When configured in API-key
    mode, that file includes a top-level ``OPENAI_API_KEY`` value. We intentionally
    do not treat the ChatGPT OAuth ``tokens.access_token`` as an API substitute:
    it is not guaranteed to have model scopes and should not make provider
    readiness look healthier than it is.
    """

    meta = inspect_codex_openai_auth()
    if meta is None:
        return None
    try:
        raw = json.loads((Path.home() / ".codex" / "auth.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    value = raw.get("OPENAI_API_KEY")
    if not isinstance(value, str):
        return None
    key = value.strip()
    return key or None


class OpenAIAdapter:
    """Streaming adapter for OpenAI's chat completions API."""

    name = "openai"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        key = api_key or os.environ.get("OPENAI_API_KEY") or load_codex_openai_api_key()
        if not key:
            codex_auth = inspect_codex_openai_auth()
            if codex_auth and codex_auth.get("auth_mode") == "chatgpt":
                raise ConfigurationError(
                    "OpenAI API key missing: current Codex auth is ChatGPT OAuth without OPENAI_API_KEY, which is not sufficient for OpenAI model calls"
                )
            raise ConfigurationError(
                "OpenAI API key missing: pass api_key=, set OPENAI_API_KEY, or configure ~/.codex/auth.json with OPENAI_API_KEY"
            )
        self.api_key = key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self._injected_client = client

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
        return Capabilities(streaming=True, tool_use=True)

    async def cancel(self, session_id: str) -> None:
        return None

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
                    f"could not connect to OpenAI at {self.base_url}: {exc}"
                ) from exc
            except httpx.TimeoutException as exc:
                raise TimeoutError(f"OpenAI request timed out: {exc}") from exc
            except httpx.HTTPError as exc:
                raise InternalError(f"OpenAI HTTP error: {exc}") from exc
        finally:
            if owns_client:
                await client.aclose()

    async def _raise_for_status(self, response: httpx.Response) -> None:
        body = await response.aread()
        text = body.decode("utf-8", errors="replace") if body else ""
        status = response.status_code
        if status == 401:
            raise ConfigurationError(f"OpenAI 401 (auth failed). Body: {text}")
        if status == 404:
            raise ModelUnavailableError(f"OpenAI 404 (model not found). Body: {text}")
        if status == 429:
            raise RateLimitError(f"OpenAI rate-limited (429). Body: {text}")
        raise InternalError(f"OpenAI HTTP {status}. Body: {text}")


__all__ = [
    "DEFAULT_BASE_URL",
    "OpenAIAdapter",
    "__version__",
    "inspect_codex_openai_auth",
    "load_codex_openai_api_key",
]
