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
    ModelSelectedEvent,
    ModelUnavailableError,
    NetworkError,
    RateLimitError,
    TimeoutError,
)
from harness.core._openai import message_to_wire, parse_sse_stream

__version__ = "0.0.0"


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EXCLUDED_MODEL_FALLBACKS = ("qwen/qwen3-coder",)


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
        model_fallbacks: str | list[str] | tuple[str, ...] | None = None,
        excluded_model_fallbacks: str | list[str] | tuple[str, ...] | None = None,
        auto_model_fallback: bool | None = None,
        model_fallback_limit: int | None = None,
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
        self._tool_support_cache: dict[str, bool | None] = {}
        fallback_source = (
            model_fallbacks
            if model_fallbacks is not None
            else os.environ.get("HARNESS_OPENROUTER_MODEL_FALLBACKS", "")
        )
        excluded_source = (
            excluded_model_fallbacks
            if excluded_model_fallbacks is not None
            else os.environ.get(
                "HARNESS_OPENROUTER_EXCLUDED_MODEL_FALLBACKS",
                ",".join(DEFAULT_EXCLUDED_MODEL_FALLBACKS),
            )
        )
        self.excluded_model_fallbacks = {
            _normalize_model_id(model) for model in _parse_model_fallbacks(excluded_source)
        }
        self.model_fallbacks = [
            model
            for model in _parse_model_fallbacks(fallback_source)
            if _normalize_model_id(model) not in self.excluded_model_fallbacks
        ]
        self.auto_model_fallback = (
            _env_flag("HARNESS_OPENROUTER_AUTO_MODEL_FALLBACK", default=True)
            if auto_model_fallback is None
            else auto_model_fallback
        )
        if model_fallback_limit is None:
            raw_limit = os.environ.get("HARNESS_OPENROUTER_MODEL_FALLBACK_LIMIT", "").strip()
            try:
                model_fallback_limit = int(raw_limit) if raw_limit else 2
            except ValueError:
                model_fallback_limit = 2
        self.model_fallback_limit = max(0, model_fallback_limit)

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
            tool_choice=kwargs.get("tool_choice"),
            response_format=kwargs.get("response_format"),
            seed=kwargs.get("seed"),
        )

    async def capabilities(self) -> Capabilities:
        # OpenRouter routes to many models; tool-use availability depends on
        # the chosen model. `model_supports_tools()` performs the model-specific
        # preflight when a runtime is about to send tool schemas.
        return Capabilities(streaming=True, tool_use=True)

    async def model_supports_tools(self, model: str) -> bool | None:
        """Return model-specific tool support when OpenRouter can prove it.

        `None` means the model was not present in the public model catalog, so
        the chat request should proceed and let the provider return the exact
        model-availability error.
        """
        normalized = _normalize_model_id(model)
        if normalized in self._tool_support_cache:
            return self._tool_support_cache[normalized]

        tool_model_ids = await self._fetch_model_ids(supported_parameters="tools")
        if normalized in tool_model_ids:
            self._tool_support_cache[normalized] = True
            return True

        all_model_ids = await self._fetch_model_ids()
        if normalized in all_model_ids:
            self._tool_support_cache[normalized] = False
            return False

        self._tool_support_cache[normalized] = None
        return None

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
        tool_choice: str | None = None,
        response_format: dict[str, Any] | str | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[Event]:
        candidate_models = _unique_models([model, *self.model_fallbacks])
        discovered_auto_models = False
        index = 0
        while index < len(candidate_models):
            candidate = candidate_models[index]
            yielded_any = False
            selection_emitted = False
            try:
                async for event in self._stream_once(
                    model=candidate,
                    messages=messages,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    seed=seed,
                ):
                    yielded_any = True
                    if not selection_emitted:
                        yield ModelSelectedEvent(
                            provider=self.name,
                            requested_model=model,
                            model=candidate,
                            fallback=candidate != model,
                            attempt=index,
                        )
                        selection_emitted = True
                    yield event
                return
            except (RateLimitError, ModelUnavailableError, TimeoutError, NetworkError) as exc:
                if yielded_any:
                    raise
                if index + 1 >= len(candidate_models) and not discovered_auto_models:
                    discovered_auto_models = True
                    try:
                        candidate_models = _unique_models(
                            [
                                *candidate_models,
                                *await self._discover_model_fallbacks(
                                    primary_model=model,
                                    tools_required=bool(tools),
                                ),
                            ]
                        )
                    except Exception:
                        raise exc from None
                if index + 1 >= len(candidate_models):
                    raise
                index += 1

    async def _stream_once(
        self,
        *,
        model: str,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        temperature: float | None,
        max_tokens: int | None,
        tool_choice: str | None = None,
        response_format: dict[str, Any] | str | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[Event]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [message_to_wire(m) for m in _system_messages_first(messages)],
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
            except httpx.RemoteProtocolError as exc:
                raise NetworkError(f"OpenRouter connection dropped: {exc}") from exc
            except httpx.HTTPError as exc:
                raise InternalError(f"OpenRouter HTTP error: {exc}") from exc
        finally:
            if owns_client:
                await client.aclose()

    async def _fetch_model_ids(self, *, supported_parameters: str | None = None) -> set[str]:
        return {
            _normalize_model_id(str(item["id"]))
            for item in await self._fetch_model_records(supported_parameters=supported_parameters)
            if isinstance(item.get("id"), str) and str(item.get("id")).strip()
        }

    async def _fetch_model_records(
        self, *, supported_parameters: str | None = None
    ) -> list[dict[str, Any]]:
        url = f"{self.base_url}/models"
        params = {"supported_parameters": supported_parameters} if supported_parameters else None
        headers: dict[str, str] = {"Authorization": f"Bearer {self.api_key}"}
        if self.http_referer:
            headers["HTTP-Referer"] = self.http_referer
        if self.x_title:
            headers["X-Title"] = self.x_title

        owns_client = self._injected_client is None
        client = self._injected_client or httpx.AsyncClient(timeout=self.timeout)
        try:
            try:
                response = await client.get(url, params=params, headers=headers)
                if response.status_code != 200:
                    await self._raise_for_status(response)
                payload = response.json()
            except httpx.ConnectError as exc:
                raise NetworkError(
                    f"could not connect to OpenRouter at {self.base_url}: {exc}"
                ) from exc
            except httpx.TimeoutException as exc:
                raise TimeoutError(f"OpenRouter models request timed out: {exc}") from exc
            except httpx.HTTPError as exc:
                raise InternalError(f"OpenRouter models HTTP error: {exc}") from exc
            except ValueError as exc:
                raise InternalError(
                    f"OpenRouter models response was not valid JSON: {exc}"
                ) from exc
        finally:
            if owns_client:
                await client.aclose()

        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise InternalError("OpenRouter models response did not contain a data list")
        records: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            raw_id = item.get("id")
            if isinstance(raw_id, str) and raw_id.strip():
                records.append(item)
        return records

    async def _discover_model_fallbacks(
        self, *, primary_model: str, tools_required: bool
    ) -> list[str]:
        if not self.auto_model_fallback or self.model_fallback_limit <= 0:
            return []
        records = await self._fetch_model_records(
            supported_parameters="tools" if tools_required else None
        )
        primary = _normalize_model_id(primary_model)
        ranked = sorted(records, key=_model_record_rank)
        fallback_ids: list[str] = []
        seen = {primary}
        for record in ranked:
            raw_id = record.get("id")
            if not isinstance(raw_id, str) or not raw_id.strip():
                continue
            normalized = _normalize_model_id(raw_id)
            if normalized in seen or normalized in self.excluded_model_fallbacks:
                continue
            seen.add(normalized)
            fallback_ids.append(raw_id.strip())
            if len(fallback_ids) >= self.model_fallback_limit:
                break
        return fallback_ids

    async def _raise_for_status(self, response: httpx.Response) -> None:
        body = await response.aread()
        text = body.decode("utf-8", errors="replace") if body else ""
        status = response.status_code
        lowered = text.lower()
        if "no endpoints found" in lowered and "tool" in lowered:
            raise ModelUnavailableError(
                f"OpenRouter has no endpoint for this model that supports tool use. Body: {text}"
            )
        if (
            status == 429
            or "rate-limited upstream" in lowered
            or "rate limited upstream" in lowered
        ):
            raise RateLimitError(f"OpenRouter rate-limited ({status}). Body: {text}")
        if status == 401:
            raise ConfigurationError(f"OpenRouter 401 (auth failed). Body: {text}")
        if status == 402:
            raise ConfigurationError(f"OpenRouter 402 (out of credits). Body: {text}")
        if status == 404:
            raise ModelUnavailableError(f"OpenRouter 404 (model not found). Body: {text}")
        if (
            status == 400
            and "model" in lowered
            and (
                "not a valid" in lowered
                or "invalid" in lowered
                or "not found" in lowered
                or "does not exist" in lowered
            )
        ):
            raise ModelUnavailableError(
                f"OpenRouter rejected the selected model ({status}). Body: {text}"
            )
        raise InternalError(f"OpenRouter HTTP {status}. Body: {text}")


def _parse_model_fallbacks(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return []
    raw_values = value.split(",") if isinstance(value, str) else list(value)
    return _unique_models(str(item).strip() for item in raw_values if str(item).strip())


def _unique_models(models: Any) -> list[str]:
    unique: list[str] = []
    seen: set[str] = set()
    for raw_model in models:
        model = str(raw_model or "").strip()
        if not model:
            continue
        normalized = _normalize_model_id(model)
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(model)
    return unique


def _env_flag(name: str, *, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _float_field(value: Any) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return float("inf")


def _model_record_rank(record: dict[str, Any]) -> tuple[float, float, str]:
    pricing = record.get("pricing") if isinstance(record.get("pricing"), dict) else {}
    prompt_price = _float_field(pricing.get("prompt") if isinstance(pricing, dict) else None)
    completion_price = _float_field(
        pricing.get("completion") if isinstance(pricing, dict) else None
    )
    context = _float_field(record.get("context_length"))
    if context == float("inf"):
        context = 0.0
    model_id = str(record.get("id") or "")
    return (prompt_price + completion_price, -context, model_id)


def _normalize_model_id(model: str) -> str:
    normalized = model.strip().lower()
    return normalized.removeprefix("openrouter/")


def _system_messages_first(messages: list[Message]) -> list[Message]:
    """Return messages with all system blocks before provider transcript turns.

    Some OpenRouter providers reject OpenAI-compatible payloads when a synthetic
    system block appears after user/assistant/tool messages. The runtime can
    create those blocks during compaction, so the adapter normalizes the wire
    order while preserving every non-system turn exactly as the model saw it.
    """

    system_messages = [message for message in messages if message.role == "system"]
    if not system_messages:
        return messages
    non_system_messages = [message for message in messages if message.role != "system"]
    return [*system_messages, *non_system_messages]


__all__ = ["DEFAULT_BASE_URL", "OpenRouterAdapter", "__version__"]
