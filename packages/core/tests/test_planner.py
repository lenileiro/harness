from __future__ import annotations

import pytest

from harness.core import Message, NoOpPlanner, PlanContext


@pytest.mark.asyncio
class TestNoOpPlanner:
    async def test_returns_single_step_wrapping_goal(self) -> None:
        planner = NoOpPlanner()
        ctx = PlanContext(session_id="s1", messages=[], available_tools=[])
        plan = await planner.plan("fix the bug", ctx)
        assert len(plan.steps) == 1
        assert plan.steps[0].description == "fix the bug"

    async def test_ignores_context_in_v1(self) -> None:
        planner = NoOpPlanner()
        ctx = PlanContext(
            session_id="s1",
            messages=[Message(role="user", content="prior turn")],
            available_tools=["echo", "write"],
        )
        plan = await planner.plan("new goal", ctx)
        assert plan.steps[0].description == "new goal"
