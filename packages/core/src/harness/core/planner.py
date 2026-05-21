"""Planner Protocol, NoOpPlanner, and LLMPlanner.

`NoOpPlanner` collapses any goal into a single-step plan — used as the
default when no planner is configured.

`LLMPlanner` calls the model once before the ReAct loop to generate a
concrete multi-step plan. Falls back to `NoOpPlanner` on any error.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from harness.core.schemas import Message


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    """Human-readable description of what this step accomplishes."""


class Plan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    steps: list[PlanStep] = Field(default_factory=list)


class PlanContext(BaseModel):
    """Everything a Planner gets to look at."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    messages: list[Message]
    available_tools: list[str]


@runtime_checkable
class Planner(Protocol):
    async def plan(self, goal: str, context: PlanContext) -> Plan: ...


class NoOpPlanner:
    """A planner that wraps the user's goal in a single step.

    Used as the default when no planner is configured.
    """

    async def plan(self, goal: str, context: PlanContext) -> Plan:
        return Plan(steps=[PlanStep(description=goal)])


_PLANNER_SYSTEM = (
    "You are a planning assistant. Break the user's goal into 2-5 concrete, "
    "sequential steps. Respond with ONLY a JSON object on a single line: "
    '{"steps": [{"description": "..."}, ...]}'
    "\nDo not add any prose. Do not wrap in markdown fences."
)


class LLMPlanner:
    """Calls the LLM once to generate a multi-step plan before execution.

    If the LLM returns invalid JSON or an empty steps list, falls back to
    `NoOpPlanner` so the agent always makes progress.
    """

    def __init__(self, *, adapter: object, model: str) -> None:
        self._adapter = adapter
        self._model = model

    async def plan(self, goal: str, context: PlanContext) -> Plan:
        from harness.core.events import Done

        messages = [
            Message(role="system", content=_PLANNER_SYSTEM),
            Message(role="user", content=goal),
        ]
        try:
            raw = ""
            async for event in self._adapter.stream(  # type: ignore[attr-defined]
                model=self._model, messages=messages, tools=None
            ):
                if isinstance(event, Done):
                    if event.final_message and event.final_message.content:
                        raw = event.final_message.content
                    break
            parsed = json.loads(raw)
            steps = [PlanStep(description=s["description"]) for s in parsed.get("steps", [])]
            if steps:
                return Plan(steps=steps)
        except Exception:
            pass
        return await NoOpPlanner().plan(goal, context)


__all__ = ["LLMPlanner", "NoOpPlanner", "Plan", "PlanContext", "PlanStep", "Planner"]
