"""Shared mocks and fixtures for harness-core tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from datetime import datetime
from typing import Any

import pytest

from harness.core import (
    ApprovalDecision,
    Capabilities,
    Done,
    Event,
    InternalError,
    Message,
    Session,
    SessionStatus,
    ToolCall,
)

# ---------------------------------------------------------------------------
# MockAdapter
# ---------------------------------------------------------------------------


class MockAdapter:
    """Adapter that replays a programmed event script per `stream()` call.

    Each call consumes one script from `scripts`. If `error` is set, that
    exception is raised instead — useful for testing failover. Records every
    call's inputs in `calls` for assertions.
    """

    def __init__(
        self,
        name: str,
        *,
        scripts: list[list[Event]] | None = None,
        error: BaseException | None = None,
        capabilities: Capabilities | None = None,
    ) -> None:
        self.name = name
        self.scripts: list[list[Event]] = list(scripts or [])
        self.error = error
        self._capabilities = capabilities or Capabilities(streaming=True, tool_use=True)
        self.calls: list[dict[str, Any]] = []
        self.cancelled: list[str] = []

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
        self.calls.append(
            {
                "model": model,
                "messages": list(messages),
                "tools": tools,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "kwargs": kwargs,
            }
        )
        return self._stream()

    async def _stream(self) -> AsyncIterator[Event]:
        if self.error is not None:
            raise self.error
        if not self.scripts:
            raise InternalError(f"MockAdapter {self.name!r}: out of scripts")
        for event in self.scripts.pop(0):
            yield event

    async def capabilities(self) -> Capabilities:
        return self._capabilities

    async def cancel(self, session_id: str) -> None:
        self.cancelled.append(session_id)


# ---------------------------------------------------------------------------
# MockTool
# ---------------------------------------------------------------------------


class MockTool:
    """Tool whose behavior is driven by a `responder` callable.

    The responder receives the tool's kwargs and returns either:
      - a `str` (the tool's content), or
      - an Exception instance (raised inside the tool)
    """

    def __init__(
        self,
        *,
        name: str = "echo",
        description: str = "Echo back the input text.",
        approval: ApprovalDecision = "auto",
        responder: Callable[..., str | BaseException] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self.parameters_schema: dict[str, Any] = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }
        self.approval: ApprovalDecision = approval
        self.responder = responder or (lambda **kw: str(kw.get("text", "")))
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        result = self.responder(**kwargs)
        if isinstance(result, BaseException):
            raise result
        return result


# ---------------------------------------------------------------------------
# MockStorage
# ---------------------------------------------------------------------------


class MockStorage:
    """Minimal in-memory Storage implementation for runtime tests.

    A more featureful version ships in harness-storage-memory; this one is
    just enough to exercise the runtime contract.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    async def get(self, session_id: str) -> Session | None:
        stored = self._sessions.get(session_id)
        return stored.model_copy(deep=True) if stored else None

    async def save(self, session: Session) -> None:
        self._sessions[session.id] = session.model_copy(deep=True)

    async def list(
        self,
        *,
        limit: int = 50,
        before: datetime | None = None,
        status: SessionStatus | None = None,
    ) -> list[Session]:
        items = sorted(self._sessions.values(), key=lambda s: s.updated_at, reverse=True)
        if before is not None:
            items = [s for s in items if s.updated_at < before]
        if status is not None:
            items = [s for s in items if s.status == status]
        return [s.model_copy(deep=True) for s in items[:limit]]

    async def delete(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def text_turn(text: str) -> list[Event]:
    """A script that emits one text delta and a Done."""
    from harness.core import TextDelta

    return [
        TextDelta(text=text),
        Done(final_message=Message(role="assistant", content=text)),
    ]


def tool_call_turn(*, call_id: str, name: str, arguments: dict[str, Any]) -> list[Event]:
    """A script that emits a single tool_call and Done.

    The Done's final_message has tool_calls populated — the runtime dispatches
    based on that.
    """
    from harness.core import ToolCallEvent

    call = ToolCall(id=call_id, name=name, arguments=arguments)
    return [
        ToolCallEvent(call=call),
        Done(
            final_message=Message(
                role="assistant",
                content=None,
                tool_calls=[call],
            )
        ),
    ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def storage() -> MockStorage:
    return MockStorage()
