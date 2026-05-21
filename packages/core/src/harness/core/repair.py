"""RepairOrchestrator — bounded retry budgets with structured repair directives.

After each tool call, `RepairOrchestrator.assess()` returns a `RepairDirective`
that tells the runtime how to proceed:

  - continue            — success or within-budget failure, model decides next step
  - verify_before_continue — succeeded but prediction mismatch (medium+) — verify outcome
  - escalate            — retry budget exhausted, stop and surface to user

Retry budgets are bounded by effect_scope (external calls get only 1 retry;
read-only calls get 3). When the budget is exhausted the directive becomes
"escalate" which causes the runtime to emit an ErrorEvent.

The orchestrator is stateful per-run (tracks consecutive failure counts per
tool name) and should be created fresh for each Agent.run() call.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from harness.core.prediction import PredictionOutcome
from harness.core.schemas import EffectScope, ToolResult

RepairMode = Literal["continue", "verify_before_continue", "escalate"]


class RepairDirective(BaseModel):
    """Structured repair decision returned after each tool call."""

    model_config = ConfigDict(extra="forbid")

    mode: RepairMode
    tool_name: str
    effect_scope: EffectScope | None
    consecutive_failures: int
    retry_budget_remaining: int
    reason: str


# Retry budget by effect_scope (max consecutive failures before escalation)
_BUDGETS: dict[EffectScope | None, int] = {
    "read_only": 3,
    "session_ephemeral": 3,
    "task_durable": 2,
    "agent_orchestration": 2,
    "workspace_durable": 2,
    "external_side_effect": 1,
    "routed": 1,
    None: 2,
}

_MISMATCH_ESCALATION_SEVERITIES = frozenset({"medium", "high", "critical"})


class RepairOrchestrator:
    """Per-run stateful repair budget tracker.

    Create one instance per Agent.run() call — it tracks consecutive failure
    counts across all tool calls within that run.
    """

    def __init__(self) -> None:
        self._failure_counts: dict[str, int] = {}

    def assess(
        self,
        *,
        tool_name: str,
        effect_scope: EffectScope | None,
        result: ToolResult,
        outcome: PredictionOutcome | None,
    ) -> RepairDirective:
        budget = _BUDGETS.get(effect_scope, 2)

        if not result.is_error:
            # Success: reset failure streak for this tool
            self._failure_counts.pop(tool_name, None)

            # Check for prediction mismatch on a successful call (unexpected outcome)
            if (
                outcome is not None
                and not outcome.matched
                and outcome.severity in _MISMATCH_ESCALATION_SEVERITIES
            ):
                return RepairDirective(
                    mode="verify_before_continue",
                    tool_name=tool_name,
                    effect_scope=effect_scope,
                    consecutive_failures=0,
                    retry_budget_remaining=budget,
                    reason=(
                        f"tool succeeded but prediction mismatch (severity={outcome.severity}) "
                        "— verify outcome before continuing"
                    ),
                )
            return RepairDirective(
                mode="continue",
                tool_name=tool_name,
                effect_scope=effect_scope,
                consecutive_failures=0,
                retry_budget_remaining=budget,
                reason="tool succeeded",
            )

        # Failure path: increment streak
        failures = self._failure_counts.get(tool_name, 0) + 1
        self._failure_counts[tool_name] = failures
        remaining = budget - failures

        if remaining <= 0:
            return RepairDirective(
                mode="escalate",
                tool_name=tool_name,
                effect_scope=effect_scope,
                consecutive_failures=failures,
                retry_budget_remaining=0,
                reason=(
                    f"repair budget exhausted for '{tool_name}' after {failures} consecutive "
                    f"failure(s) — escalating to human review"
                ),
            )

        return RepairDirective(
            mode="continue",
            tool_name=tool_name,
            effect_scope=effect_scope,
            consecutive_failures=failures,
            retry_budget_remaining=remaining,
            reason=f"tool failed ({failures}/{budget} failures used)",
        )


__all__ = [
    "RepairDirective",
    "RepairMode",
    "RepairOrchestrator",
]
