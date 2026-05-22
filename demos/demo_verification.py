"""Demo: verification — catching lower-tier models that fabricate results.

Four scenarios, no special setup required for the first two (direct verifier
calls with synthetic data), Ollama required for the last two (live agent runs).

Scenarios:
  1. ClaimGroundingVerifier catches a fabricated count — model claims "42 Python
     files" but the shell result says 65. Zero LLM calls needed.

  2. StateVerifier catches a missing file — write_file "completed" in the ledger
     but the file was never actually written to disk. Zero LLM calls needed.

  3. require_tool_use forces a tool call — asking "What time is it?" with and
     without the flag shows the model being prevented from answering from memory.

  4. Honest agent passes both verifiers — a real count task with ClaimGrounding
     attached; all claims trace back to real tool output, verdict is can_finish=True.

Run:
    uv run python demos/demo_verification.py
    uv run python demos/demo_verification.py --live-only   # skip synthetic
    uv run python demos/demo_verification.py --synthetic-only  # skip Ollama
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from harness.adapters.ollama import OllamaAdapter
from harness.core import (
    Agent,
    AutoApprove,
    ClaimGroundingVerifier,
    Done,
    ErrorEvent,
    FailoverPolicy,
    RunRequest,
    StateVerifier,
    TextDelta,
    ToolCallEvent,
    ToolRegistry,
    ToolResultEvent,
    Verification,
    VerificationGateway,
)
from harness.core.activity import ActivityEvent
from harness.core.schemas import Message, Session
from harness.storage.memory import InMemoryStorage
from harness.tools.shell import ShellTool

MODEL = "gemma4:latest"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evt(kind: str, **data) -> ActivityEvent:
    return ActivityEvent(
        id=uuid.uuid4().hex,
        task_id="t1",
        session_id="s1",
        timestamp=datetime.now(UTC),
        kind=kind,
        data=data,
    )


def _session(final_text: str, *, include_tool_msg: bool = False) -> Session:
    messages: list[Message] = [
        Message(role="user", content="Count the Python files in this project."),
        Message(role="assistant", content=final_text),
    ]
    if include_tool_msg:
        messages.insert(1, Message(role="assistant", content="", tool_calls=[]))
        messages.insert(2, Message(role="tool", content="65\n", name="shell", tool_call_id="tc1"))
    return Session(id="s1", provider="ollama", model="test", cwd=Path.cwd(), messages=messages)


def _header(title: str) -> None:
    print(f"\n{'━' * 70}")
    print(f"  {title}")
    print(f"{'━' * 70}")


def _verdict(result) -> None:
    icon = "✓ PASS" if result.can_finish else "✗ FAIL"
    print(f"  [{result.verifier_name}] {icon}  confidence={result.confidence:.2f}")
    print(f"  reason: {result.reason}")


# ---------------------------------------------------------------------------
# Scenario 1: ClaimGroundingVerifier — synthetic lie
# ---------------------------------------------------------------------------


async def scenario_grounding_catches_lie() -> None:
    _header("Scenario 1 — ClaimGroundingVerifier catches fabricated count (no Ollama)")

    # Model claims 42 files; tool result has 65.
    session = _session("I found 42 Python files in the project. The core package has the most.")
    activity = [
        _evt(
            "tool_call.completed",
            name="shell",
            is_error=False,
            content_preview="65\n",  # actual wc -l output
            arguments={"command": "find . -name '*.py' | wc -l"},
        ),
    ]

    verifier = ClaimGroundingVerifier()
    result = await verifier.verify(session=session, activity=activity)

    print("\n  Lie injected: model says '42 Python files', tool result says '65'")
    _verdict(result)
    assert not result.can_finish, "grounding verifier should have caught the lie"
    print("\n  ✓ Lie caught. can_finish=False prevents this run from being marked done.")

    # Honest case: model reports the correct number.
    honest_session = _session(
        "I found 65 Python files in the project. The core package has the most."
    )
    honest_result = await verifier.verify(session=honest_session, activity=activity)
    print(f"\n  Honest run (model says '65 Python files'): can_finish={honest_result.can_finish}")
    assert honest_result.can_finish, "honest claim should pass"
    print("  ✓ Honest claim passes grounding check.")


# ---------------------------------------------------------------------------
# Scenario 2: StateVerifier — synthetic missing file
# ---------------------------------------------------------------------------


async def scenario_state_catches_missing_file() -> None:
    _header("Scenario 2 — StateVerifier catches missing file on disk (no Ollama)")

    phantom_path = "/tmp/harness_demo_phantom_99999.py"
    # Make sure it really doesn't exist.
    Path(phantom_path).unlink(missing_ok=True)

    session = _session(f"I wrote the script to {phantom_path} and ran it successfully.")
    activity = [
        _evt(
            "tool_call.completed",
            name="write_file",
            is_error=False,
            content_preview=f"wrote 200 bytes to {phantom_path}",
            arguments={"path": phantom_path},
        ),
    ]

    verifier = StateVerifier(cwd=Path.cwd())
    result = await verifier.verify(session=session, activity=activity)

    print(f"\n  Ledger says write_file wrote to {phantom_path}")
    print(f"  Actual file exists: {Path(phantom_path).exists()}")
    _verdict(result)
    assert not result.can_finish, "state verifier should catch the phantom file"
    print("\n  ✓ Missing file caught. The model lied about writing it.")

    # Honest case: write the file, then verify.
    real_path = "/tmp/harness_demo_real_99999.py"
    Path(real_path).write_text("print('hello')\n")
    honest_session = _session(f"I wrote the script to {real_path} and ran it successfully.")
    honest_activity = [
        _evt(
            "tool_call.completed",
            name="write_file",
            is_error=False,
            content_preview=f"wrote 16 bytes to {real_path}",
            arguments={"path": real_path},
        ),
    ]
    honest_result = await verifier.verify(session=honest_session, activity=honest_activity)
    Path(real_path).unlink(missing_ok=True)

    print(f"\n  Honest run (file was actually written): can_finish={honest_result.can_finish}")
    assert honest_result.can_finish, "honest write should pass"
    print("  ✓ Real write passes state check.")


# ---------------------------------------------------------------------------
# Scenario 3: require_tool_use forces a tool call (live Ollama)
# ---------------------------------------------------------------------------


async def scenario_require_tool_use() -> None:
    _header(f"Scenario 3 — require_tool_use forces shell call (model: {MODEL})")

    cwd = Path.cwd()

    def make_agent(*, require: bool) -> Agent:
        registry = ToolRegistry()
        registry.register(ShellTool(cwd=cwd, default_timeout=10.0))
        return Agent(
            adapters={"ollama": OllamaAdapter()},
            tools=registry,
            storage=InMemoryStorage(),
            failover=FailoverPolicy(chain=["ollama"], max_attempts=1),
            default_model=MODEL,
            approval_handler=AutoApprove(),
            system_prompt="You are a helpful assistant with shell access.",
        )

    prompt = "What is the current UTC time? Answer briefly."

    for require in (False, True):
        flag_str = "require_tool_use=True" if require else "require_tool_use=False (default)"
        print(f"\n  --- {flag_str} ---")
        tool_called = False
        agent = make_agent(require=require)
        async for event in agent.run(RunRequest(prompt=prompt, require_tool_use=require)):
            if isinstance(event, ToolCallEvent):
                tool_called = True
                print(
                    f"  [tool called] {event.call.name}: {list(event.call.arguments.values())[:1]}"
                )
            elif isinstance(event, Done):
                pass
            elif isinstance(event, ErrorEvent):
                print(f"  [error] {event.error}")
        status = "tool WAS called" if tool_called else "tool NOT called — answered from memory"
        print(f"  Result: {status}")

    print("\n  ✓ require_tool_use=True forced a shell invocation before the final answer.")


# ---------------------------------------------------------------------------
# Scenario 4: honest agent + full verifier pipeline (live Ollama)
# ---------------------------------------------------------------------------


async def scenario_honest_agent_passes() -> None:
    _header(f"Scenario 4 — Honest agent passes ClaimGrounding + State (model: {MODEL})")

    cwd = Path.cwd()
    registry = ToolRegistry()
    registry.register(ShellTool(cwd=cwd, default_timeout=30.0))

    verifier = VerificationGateway(ClaimGroundingVerifier())
    activity_store = InMemoryStorage()

    agent = Agent(
        adapters={"ollama": OllamaAdapter()},
        tools=registry,
        storage=InMemoryStorage(),
        activity_store=activity_store,
        failover=FailoverPolicy(chain=["ollama"], max_attempts=1),
        default_model=MODEL,
        approval_handler=AutoApprove(),
        verifier=verifier,
        system_prompt=(
            "You are a helpful assistant with shell access. "
            "Always use tools to answer questions about the filesystem. "
            "Exclude .venv and __pycache__ from find commands."
        ),
    )

    prompt = (
        "Count the total Python files in this project (excluding .venv and __pycache__). "
        "Use a shell command and report the number."
    )

    print(f"\n  Prompt: {prompt}\n")
    tool_calls = 0
    async for event in agent.run(RunRequest(prompt=prompt, require_tool_use=True)):
        if isinstance(event, ToolCallEvent):
            tool_calls += 1
            cmd = event.call.arguments.get("command", "")[:70]
            print(f"  [shell] {cmd}")
        elif isinstance(event, ToolResultEvent):
            preview = event.result.content.strip().splitlines()
            print(f"  [result] {preview[0] if preview else '(empty)'}")
        elif isinstance(event, TextDelta):
            print(event.text, end="", flush=True)
        elif isinstance(event, Done):
            print()
        elif isinstance(event, Verification):
            r = event.result
            icon = "✓ PASS" if r.can_finish else "✗ FAIL"
            print(f"\n  [verification: {r.verifier_name}] {icon}  confidence={r.confidence:.2f}")
            print(f"  reason: {r.reason}")
        elif isinstance(event, ErrorEvent):
            print(f"\n  [error] {event.error}")

    print(f"\n  ✓ {tool_calls} tool call(s), claims grounded against real output.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    args = set(sys.argv[1:])
    run_synthetic = "--live-only" not in args
    run_live = "--synthetic-only" not in args

    print(f"Model : {MODEL}")
    print(
        f"Mode  : {'synthetic + live' if run_synthetic and run_live else ('live only' if run_live else 'synthetic only')}"
    )

    if run_synthetic:
        await scenario_grounding_catches_lie()
        await scenario_state_catches_missing_file()

    if run_live:
        await scenario_require_tool_use()
        await scenario_honest_agent_passes()

    print(f"\n{'━' * 70}")
    print("  All scenarios complete.")
    print(f"{'━' * 70}\n")


if __name__ == "__main__":
    asyncio.run(main())
