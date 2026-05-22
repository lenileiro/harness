"""Agent handoff tool — delegate control to another Agent mid-run.

Inspired by OpenAI Swarm's handoff pattern. A tool raises
:class:`~harness.core.errors.Handoff` and the runtime catches it, emits a
:class:`~harness.core.events.HandoffEvent`, then replays the current session
through the target agent.

Example::

    from harness.core.handoff import HandoffTool

    router_agent = Agent(adapters=..., tools=registry, ...)
    specialist = Agent(adapters=..., tools=specialist_registry, ...)

    registry.register(HandoffTool(specialist, name="to_specialist"))

    async for event in router_agent.run(request):
        ...  # router decides to hand off; specialist continues

The handoff is invisible to the session: the target agent sees the same
``Session`` history and picks up where the router left off.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from harness.core.errors import Handoff
from harness.core.schemas import ToolCall, ToolResult

if TYPE_CHECKING:
    from harness.core.runtime import Agent


class HandoffTool:
    """Tool that triggers a handoff to another :class:`~harness.core.runtime.Agent`.

    Register this in any agent's :class:`~harness.core.tools.ToolRegistry` to
    let the model explicitly hand off control.

    Args:
        target: The agent that will take over.
        name:   Tool name exposed to the model (default: ``"handoff"``).
        description: Tool description (default: generic).
    """

    parameters_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why control is being handed off to the specialist agent.",
            }
        },
        "required": ["reason"],
    }
    approval = "auto"

    def __init__(
        self,
        target: Agent,
        *,
        name: str = "handoff",
        description: str = "Hand off the current task to a specialist agent.",
    ) -> None:
        self._target = target
        self.name = name
        self.description = description

    async def __call__(self, call: ToolCall) -> ToolResult:
        reason = call.arguments.get("reason", "")
        raise Handoff(target=self._target, reason=reason)


__all__ = ["HandoffTool"]
