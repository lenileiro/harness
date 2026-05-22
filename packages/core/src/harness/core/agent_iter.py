"""agent.iter() — expose the agent loop as typed step objects.

Usage::

    async with agent.iter(request) as run:
        async for step in run:
            match step:
                case ToolCallStep(tool_call=call):
                    print(f"tool: {call.name}({call.arguments})")
                case FinalResponseStep(text=text):
                    print(f"answer: {text}")

Steps map to internal events as follows:

- :class:`ModelRequestStep`  ← :class:`~harness.core.events.ModelRequestEvent`
- :class:`ToolCallStep`      ← :class:`~harness.core.events.ToolCallEvent`
- :class:`ToolResultStep`    ← :class:`~harness.core.events.ToolResultEvent`
- :class:`FinalResponseStep` ← :class:`~harness.core.events.Done`
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterator
from typing import TYPE_CHECKING

from pydantic import BaseModel

from harness.core.events import Done, ModelRequestEvent, ToolCallEvent, ToolResultEvent
from harness.core.schemas import Message, ToolCall, ToolResult

if TYPE_CHECKING:
    from harness.core.runtime import Agent
    from harness.core.schemas import RunRequest


# ---------------------------------------------------------------------------
# Step types
# ---------------------------------------------------------------------------


class ModelRequestStep(BaseModel):
    """The runtime is about to call the LLM with these messages."""

    messages: list[Message]


class ToolCallStep(BaseModel):
    """The model has requested a tool call."""

    tool_call: ToolCall


class ToolResultStep(BaseModel):
    """A tool has returned a result."""

    tool_result: ToolResult


class FinalResponseStep(BaseModel):
    """The LLM produced a final text response (no tool calls)."""

    text: str


AgentRunStep = ModelRequestStep | ToolCallStep | ToolResultStep | FinalResponseStep


# ---------------------------------------------------------------------------
# AgentRun context manager
# ---------------------------------------------------------------------------


class AgentRun:
    """Async context manager returned by :meth:`Agent.iter`.

    Wraps the agent's internal event stream and translates raw events into
    typed :data:`AgentRunStep` objects. The context manager owns the generator
    lifecycle — exiting via ``break`` or exception closes the generator cleanly.
    """

    def __init__(self, agent: Agent, request: RunRequest) -> None:
        self._agent = agent
        self._request = request
        self._gen: AsyncGenerator[AgentRunStep, None] | None = None

    async def __aenter__(self) -> AgentRun:
        self._gen = self._iterate()
        return self

    async def __aexit__(self, *_: object) -> None:
        if self._gen is not None:
            await self._gen.aclose()
            self._gen = None

    def __aiter__(self) -> AsyncIterator[AgentRunStep]:
        if self._gen is None:
            raise RuntimeError("use `async with agent.iter(request) as run:` before iterating")
        return self._gen

    async def _iterate(self) -> AsyncGenerator[AgentRunStep, None]:
        async for event in self._agent._run(self._request):
            if isinstance(event, ModelRequestEvent):
                yield ModelRequestStep(messages=event.messages)
            elif isinstance(event, ToolCallEvent):
                yield ToolCallStep(tool_call=event.call)
            elif isinstance(event, ToolResultEvent):
                yield ToolResultStep(tool_result=event.result)
            elif isinstance(event, Done) and event.final_message is not None:
                yield FinalResponseStep(text=event.final_message.content or "")


__all__ = [
    "AgentRun",
    "AgentRunStep",
    "FinalResponseStep",
    "ModelRequestStep",
    "ToolCallStep",
    "ToolResultStep",
]
