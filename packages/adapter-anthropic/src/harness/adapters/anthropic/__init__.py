"""Anthropic Messages API adapter for Harness.

Calls api.anthropic.com using the official Python SDK. Handles Anthropic's
wire format differences from OpenAI:
  - System messages are extracted and passed as the `system` parameter.
  - Tool definitions use `input_schema` instead of `parameters`.
  - Assistant messages with tool calls use content blocks (type=tool_use).
  - Tool results use content blocks (type=tool_result) on a user message.
  - Consecutive tool-result messages are merged into a single user turn.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from typing import Any

from harness.core import (
    Capabilities,
    ConfigurationError,
    Event,
    InternalError,
    ModelUnavailableError,
    NetworkError,
    RateLimitError,
    TimeoutError,
)
from harness.core.events import Done, TextDelta, ToolCallEvent
from harness.core.schemas import Message, ToolCall, Usage

__version__ = "0.0.0"

DEFAULT_BASE_URL = "https://api.anthropic.com"
DEFAULT_MAX_TOKENS = 8096


class AnthropicAdapter:
    """Streaming adapter for Anthropic's Messages API.

    Args:
        api_key: Anthropic API key. Falls back to $ANTHROPIC_API_KEY.
        base_url: Override the API base URL.
        timeout: Streaming request timeout in seconds.
        default_max_tokens: Max tokens when not specified by caller.
    """

    name = "anthropic"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ConfigurationError(
                "Anthropic API key missing: pass api_key= or set ANTHROPIC_API_KEY"
            )
        try:
            import anthropic as _sdk
        except ImportError as exc:
            raise ConfigurationError(
                "anthropic package not installed: pip install anthropic"
            ) from exc

        self._client = _sdk.AsyncAnthropic(
            api_key=key,
            base_url=base_url or DEFAULT_BASE_URL,
            timeout=timeout,
        )
        self._default_max_tokens = default_max_tokens

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
            max_tokens=max_tokens or self._default_max_tokens,
        )

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=True)

    async def cancel(self, session_id: str) -> None:
        return None

    # ------------------------------------------------------------------ #
    # Wire conversion                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _convert_messages(
        messages: list[Message],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Extract system prompt and convert messages to Anthropic format.

        Returns (system_text_or_None, list_of_anthropic_message_dicts).
        Consecutive tool-result messages (role='tool') are merged into a
        single user turn as Anthropic requires.
        """
        system_parts: list[str] = []
        wire: list[dict[str, Any]] = []

        i = 0
        while i < len(messages):
            msg = messages[i]

            if msg.role == "system":
                if msg.content:
                    system_parts.append(msg.content)
                i += 1
                continue

            if msg.role == "tool":
                blocks: list[dict[str, Any]] = []
                while i < len(messages) and messages[i].role == "tool":
                    t = messages[i]
                    blocks.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": t.tool_call_id or "",
                            "content": t.content or "",
                        }
                    )
                    i += 1
                wire.append({"role": "user", "content": blocks})
                continue

            if msg.role == "assistant":
                blocks = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        blocks.append(
                            {
                                "type": "tool_use",
                                "id": tc.id,
                                "name": tc.name,
                                "input": tc.arguments,
                            }
                        )
                wire.append(
                    {
                        "role": "assistant",
                        "content": blocks if blocks else "",
                    }
                )
                i += 1
                continue

            wire.append({"role": "user", "content": msg.content or ""})
            i += 1

        system = "\n\n".join(system_parts) if system_parts else None
        return system, wire

    @staticmethod
    def _convert_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert OpenAI-style tool defs to Anthropic format.

        OpenAI: {"type": "function", "function": {"name": ..., "parameters": ...}}
        Anthropic: {"name": ..., "description": ..., "input_schema": ...}
        """
        result = []
        for t in tools:
            fn = t.get("function", t)
            result.append(
                {
                    "name": fn["name"],
                    "description": fn.get("description", ""),
                    "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        return result

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
        max_tokens: int,
    ) -> AsyncIterator[Event]:
        import anthropic as _sdk

        system, wire_messages = self._convert_messages(messages)
        wire_tools = self._convert_tools(tools) if tools else []

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": wire_messages,
        }
        if system:
            kwargs["system"] = system
        if wire_tools:
            kwargs["tools"] = wire_tools
        if temperature is not None:
            kwargs["temperature"] = temperature

        try:
            # Stream text deltas live; collect full message for tool calls + usage.
            async with self._client.messages.stream(**kwargs) as s:
                async for chunk in s.text_stream:
                    yield TextDelta(text=chunk)
                final = await s.get_final_message()

            tool_calls: list[ToolCall] = []
            for block in final.content:
                if block.type == "tool_use":
                    tc = ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=dict(block.input) if block.input else {},
                    )
                    tool_calls.append(tc)
                    yield ToolCallEvent(call=tc)

            text_content = next((b.text for b in final.content if b.type == "text"), None)
            usage = (
                Usage(
                    prompt_tokens=final.usage.input_tokens,
                    completion_tokens=final.usage.output_tokens,
                )
                if final.usage
                else None
            )
            final_msg = Message(
                role="assistant",
                content=text_content or None,
                tool_calls=tool_calls or None,
            )
            yield Done(final_message=final_msg, usage=usage)

        except _sdk.AuthenticationError as exc:
            raise ConfigurationError(f"Anthropic auth failed: {exc}") from exc
        except _sdk.NotFoundError as exc:
            raise ModelUnavailableError(f"Anthropic model not found: {exc}") from exc
        except _sdk.RateLimitError as exc:
            raise RateLimitError(f"Anthropic rate limited: {exc}") from exc
        except _sdk.APITimeoutError as exc:
            raise TimeoutError(f"Anthropic request timed out: {exc}") from exc
        except _sdk.APIConnectionError as exc:
            raise NetworkError(f"Anthropic connection error: {exc}") from exc
        except _sdk.APIStatusError as exc:
            raise InternalError(f"Anthropic API {exc.status_code}: {exc.message}") from exc


__all__ = ["DEFAULT_BASE_URL", "DEFAULT_MAX_TOKENS", "AnthropicAdapter", "__version__"]
