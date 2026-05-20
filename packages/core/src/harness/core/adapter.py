"""Provider adapter Protocol.

Sibling packages (`adapter-ollama`, `adapter-openrouter`, ...) implement
this. The runtime never imports them directly — adapters are injected.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, Protocol, runtime_checkable

from harness.core.events import Event
from harness.core.schemas import Capabilities, Message


@runtime_checkable
class Adapter(Protocol):
    """Streaming chat-completion adapter for a single provider.

    Adapters are stateless across runs (the runtime owns session state). They
    translate provider-specific wire formats (HTTP/SSE, OpenAI-compat chunks,
    Ollama's NDJSON, ...) into the normalized `Event` stream.
    """

    name: str
    """Stable provider identifier, e.g. `"openrouter"` or `"ollama"`."""

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
        """Stream a single model turn as Events.

        Implementations are async generators — that's why this is declared as
        a plain `def` returning `AsyncIterator[Event]` (an async generator
        function call site returns an iterator, not a coroutine).

        `tools` is the OpenAI-format tools array (one dict per tool, ready to
        send on the wire). Adapters that don't support tool use should ignore
        it and report `Capabilities.tool_use = False`.

        Implementations must:
          - Convert provider chunks to `TextDelta`, `ToolCallEvent`s.
          - Emit a single terminal `Done` with the assembled `final_message`.
          - Raise typed errors from `harness.core.errors` on failure (the
            runtime classifies them via FailoverPolicy).
        """
        ...

    async def capabilities(self) -> Capabilities:
        """Describe what this adapter can do."""
        ...

    async def cancel(self, session_id: str) -> None:
        """Best-effort cancel of any in-flight stream for `session_id`.

        Adapters without a cancellation mechanism can no-op.
        """
        ...


__all__ = ["Adapter"]
