"""Planner Protocol and stub.

In v1 the runtime always uses `NoOpPlanner`, which collapses any goal into a
single-step plan. The contract exists now so the v3 implementation drops in
without touching the Agent.
"""

from __future__ import annotations

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

    Used as the default in v1; the agent's ReAct loop is the planner in
    practice. Real planners arrive in v3.
    """

    async def plan(self, goal: str, context: PlanContext) -> Plan:
        return Plan(steps=[PlanStep(description=goal)])


__all__ = ["NoOpPlanner", "Plan", "PlanContext", "PlanStep", "Planner"]
