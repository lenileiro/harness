"""Demo: agent handoff — triage router hands off to a specialist.

A router agent has one tool: `escalate_to_specialist`. When called, it
raises Handoff and the specialist takes over with domain-specific context.

Architecture:
  router (gemma4)  →  HandoffTool  →  specialist (gemma4 + extra system prompt)

Run: uv run python demos/demo_agent_handoff.py
"""

from __future__ import annotations

import asyncio

from harness.adapters.ollama import OllamaAdapter
from harness.core import (
    Agent,
    Done,
    FailoverPolicy,
    HandoffEvent,
    RunRequest,
    TextDelta,
    ToolRegistry,
)
from harness.core.handoff import HandoffTool
from harness.storage.memory import InMemoryStorage

MODEL = "gemma4:latest"


def make_agent(system_prompt: str, tools=None) -> Agent:
    adapter = OllamaAdapter()
    storage = InMemoryStorage()
    registry = ToolRegistry()
    for t in tools or []:
        registry.register(t)
    failover = FailoverPolicy(chain=["ollama"], max_attempts=1)
    return Agent(
        adapters={"ollama": adapter},
        tools=registry,
        storage=storage,
        failover=failover,
        default_model=MODEL,
        system_prompt=system_prompt,
    )


async def run_scenario(prompt: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"User: {prompt}")
    print("─" * 60)

    # Specialist: deep Python knowledge
    specialist = make_agent(
        system_prompt=(
            "You are a Python expert with 20 years of experience. "
            "Give concrete, precise answers with code examples where relevant."
        )
    )

    # Router: decides whether to handle or escalate
    router = make_agent(
        system_prompt=(
            "You are a triage agent. For any question about Python programming, "
            "immediately call the escalate_to_specialist tool. "
            "For everything else, answer directly."
        ),
        tools=[HandoffTool(specialist, name="escalate_to_specialist")],
    )

    role = "router"
    async for event in router.run(RunRequest(prompt=prompt)):
        if isinstance(event, HandoffEvent):
            print(f"\n[HandoffEvent] → escalating to specialist (reason: {event.reason!r})")
            role = "specialist"
        elif isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, Done):
            print(f"\n[Done — answered by: {role}]")


async def main() -> None:
    print(f"Model: {MODEL}")

    # Python question → should trigger handoff
    await run_scenario("What's the difference between __slots__ and __dict__ in Python?")

    # General question → router answers directly (no handoff)
    await run_scenario("What is the capital of Australia?")


if __name__ == "__main__":
    asyncio.run(main())
