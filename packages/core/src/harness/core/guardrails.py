"""Guardrail protocol for Agent — run safety checks in parallel or blocking mode.

Guardrails inspect the current message history and return a verdict. When
``mode="blocking"`` the check must pass before the LLM is called. When
``mode="parallel"`` (default) the LLM starts immediately while the check runs;
if the guardrail trips, the response stream is cancelled.

Example::

    class NoSecretsGuardrail:
        name = "no_secrets"
        mode = "parallel"

        async def __call__(self, messages: list[Message]) -> GuardrailResult:
            last = messages[-1].content or ""
            if "API_KEY" in last or "password" in last.lower():
                return GuardrailResult(tripped=True, reason="possible secret in prompt")
            return GuardrailResult(tripped=False)

    agent = Agent(..., guardrails=[NoSecretsGuardrail()])
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel

from harness.core.schemas import Message

GuardrailMode = Literal["parallel", "blocking"]


class GuardrailResult(BaseModel):
    """Verdict returned by a Guardrail check."""

    tripped: bool
    reason: str = ""


@runtime_checkable
class Guardrail(Protocol):
    """Duck-type protocol for guardrail implementations."""

    name: str
    mode: GuardrailMode

    async def __call__(self, messages: list[Message]) -> GuardrailResult: ...


__all__ = ["Guardrail", "GuardrailMode", "GuardrailResult"]
