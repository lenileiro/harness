"""Autonomous agent demo — give it any prompt, it figures out the tools.

No domain-specific code required. The agent has shell + filesystem tools
and solves tasks by itself from plain English.

Run:
    uv run python demos/demo_autonomous.py
    uv run python demos/demo_autonomous.py "your custom prompt here"
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from harness.adapters.ollama import OllamaAdapter
from harness.core import (
    Agent,
    AutoApprove,
    Done,
    ErrorEvent,
    FailoverPolicy,
    RunRequest,
    TextDelta,
    ToolCallEvent,
    ToolRegistry,
    ToolResultEvent,
)
from harness.storage.memory import InMemoryStorage
from harness.tools.fs import ListDirTool, ReadFileTool, WriteFileTool
from harness.tools.shell import ShellTool

MODEL = "gemma4:latest"

DEMO_PROMPTS = [
    "Count the total number of Python source files in this project and tell me which top-level package has the most.",
    "Write a Python script that prints the first 15 Fibonacci numbers. Save it to /tmp/fib_demo.py then run it and show me the output.",
    "What is today's date and the current UTC time? Use a shell command to find out.",
]


def make_agent(cwd: Path) -> Agent:
    registry = ToolRegistry()
    registry.register(ShellTool(cwd=cwd, default_timeout=30.0))
    registry.register(ReadFileTool(cwd=cwd))
    registry.register(WriteFileTool(cwd=cwd))
    registry.register(ListDirTool(cwd=cwd))

    return Agent(
        adapters={"ollama": OllamaAdapter()},
        tools=registry,
        storage=InMemoryStorage(),  # type: ignore[arg-type]
        failover=FailoverPolicy(chain=["ollama"], max_attempts=1),
        default_model=MODEL,
        approval_handler=AutoApprove(),
        system_prompt=(
            "You are a capable assistant with shell and filesystem access. "
            "When given a task, use your tools to complete it. "
            "Think step by step: first plan what commands to run, then run them, "
            "then report the result clearly.\n\n"
            "IMPORTANT shell hygiene rules:\n"
            "- Always exclude .venv, __pycache__, node_modules, .git from find/glob commands.\n"
            "  Example: find . -name '*.py' -not -path './.venv/*' -not -path './__pycache__/*'\n"
            "- Use pipes to count/sort/summarise large output: | wc -l, | sort | uniq -c | sort -rn\n"
            "- Never run long-running background processes."
        ),
    )


async def run_prompt(agent: Agent, prompt: str) -> None:
    print(f"\n{'━' * 70}")
    print(f"  TASK: {prompt}")
    print(f"{'━' * 70}\n")

    async for event in agent.run(RunRequest(prompt=prompt)):
        if isinstance(event, ToolCallEvent):
            args = event.call.arguments
            display = next(iter(args.values()), "") if args else ""
            if len(str(display)) > 80:
                display = str(display)[:77] + "..."
            print(f"\n  [tool: {event.call.name}] {display}")
        elif isinstance(event, ToolResultEvent):
            content = event.result.content
            lines = content.strip().splitlines()
            if len(lines) > 10:
                content = "\n".join(lines[:10]) + f"\n  ... ({len(lines) - 10} more lines)"
            print(f"  [result]\n{content}\n")
        elif isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, Done):
            print()
        elif isinstance(event, ErrorEvent):
            print(f"\n  [error] {event.error}")


async def main() -> None:
    cwd = Path.cwd()
    print(f"Model : {MODEL}")
    print(f"CWD   : {cwd}")
    print("Tools : shell, read_file, write_file, list_dir")

    prompts = sys.argv[1:] if len(sys.argv) > 1 else DEMO_PROMPTS

    agent = make_agent(cwd)
    for prompt in prompts:
        await run_prompt(agent, prompt)


if __name__ == "__main__":
    asyncio.run(main())
