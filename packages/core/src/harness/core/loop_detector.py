"""L4 — Trajectory regulation: detect degenerate tool-call patterns.

Spec borrowed from the LifeHarness paper (Peking U., 2026). Their L4 layer
"monitors post-execution dynamics, detects degenerate patterns such as
repetition, stagnation, and invalid retries." We focus on the two patterns
that show up reliably in our eval traces:

  tool_repeat   — the agent calls the same (tool, args) signature N times
                  in a row with identical output. The next call is unlikely
                  to change anything; intervene with a repair directive.

  no_progress   — the agent has produced no file edits (no Write / Edit /
                  shell write) across the last K turns despite emitting
                  tool calls. Suggests it's spinning in read-only loops.

Both checks are pure: given a window of recent tool calls + results, the
detector returns either ``None`` (no intervention) or a
``LoopFinding(pattern, directive)``. The runtime is responsible for
appending the directive to the next user-role message and emitting the
``trajectory.regulated`` activity event.

L4 is intentionally cheap — it runs after every tool result. The expensive
work (LLM-driven critique) stays in the repair / critic path.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

from harness.core.schemas import ToolCall, ToolResult

LoopPattern = Literal["tool_repeat", "no_progress"]

# Tools whose execution is considered "progress." A run that ONLY touches
# read-only tools across the no-progress window is probably stuck.
_MUTATING_TOOL_HINTS: frozenset[str] = frozenset(
    {
        "write_file",
        "edit_file",
        "edit",
        "write",
        "apply_patch",
        "shell",
        "run_shell",
        "verify_work",
        "phase",
    }
)


def _hash_call(name: str, arguments: dict | str | None) -> str:
    """Stable fingerprint for (tool_name, arguments).

    Identical arguments must hash identically even when the dict was
    constructed in a different key order — that's the whole point.
    """
    try:
        body = json.dumps(arguments or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        body = repr(arguments)
    return hashlib.sha1(f"{name}\x00{body}".encode()).hexdigest()[:16]


@dataclass
class LoopFinding:
    """What the detector wants the runtime to do next."""

    pattern: LoopPattern
    tool: str | None
    repeats: int
    directive: str

    def as_event_data(self) -> dict[str, object]:
        return {
            "pattern": self.pattern,
            "tool": self.tool,
            "repeats": self.repeats,
            "directive": self.directive,
        }


@dataclass
class LoopDetector:
    """Sliding-window tracker over recent (tool, args) signatures.

    Construct one per Agent run (the Agent itself reuses the instance). The
    detector keeps the last ``window`` tool-call fingerprints and inspects
    them after every new call.

    Args:
        repeat_threshold: how many identical consecutive (tool, args) calls
            trip the ``tool_repeat`` pattern. 3 is the empirically right
            number: 2 happens routinely (retry on transient error), 3
            almost never happens for productive work.
        no_progress_threshold: how many tool calls without a mutating tool
            in a row before ``no_progress`` fires. Default 6 — a model that
            reads files six times without writing is almost certainly stuck.
        window: max recent calls to retain. Larger than both thresholds so
            we can distinguish "3 in a row" from "3 within 10 turns."
        mutating_tools: override the set of tools considered progress.
            When None, ``_MUTATING_TOOL_HINTS`` is used.
    """

    repeat_threshold: int = 3
    no_progress_threshold: int = 6
    window: int = 12
    mutating_tools: frozenset[str] | None = None
    _signatures: deque[tuple[str, str]] = field(default_factory=deque, init=False, repr=False)
    """Each entry is (tool_name, signature_hash)."""
    _emitted_for: set[str] = field(default_factory=set, init=False, repr=False)
    """Signatures we've already nagged about; prevents repeat-emission spam."""

    def __post_init__(self) -> None:
        if self.repeat_threshold < 2:
            raise ValueError("repeat_threshold must be >= 2")
        if self.no_progress_threshold < 2:
            raise ValueError("no_progress_threshold must be >= 2")
        if self.window < self.repeat_threshold or self.window < self.no_progress_threshold:
            raise ValueError("window must be >= both thresholds")
        if self.mutating_tools is None:
            self.mutating_tools = _MUTATING_TOOL_HINTS

    def observe(self, call: ToolCall, result: ToolResult | None = None) -> LoopFinding | None:
        """Record the call and return a finding if a pattern just tripped.

        The detector ignores the result content today (the LifeHarness paper
        does the same — repetition alone is enough signal). We keep
        ``result`` in the signature for future extensions (e.g., suppress
        warning when the result content actually changed).
        """
        sig = _hash_call(call.name, call.arguments if isinstance(call.arguments, dict) else None)
        self._signatures.append((call.name, sig))
        while len(self._signatures) > self.window:
            self._signatures.popleft()

        # Pattern 1: tool_repeat — last N entries identical.
        if len(self._signatures) >= self.repeat_threshold:
            recent = list(self._signatures)[-self.repeat_threshold :]
            first_sig = recent[0][1]
            if all(s == first_sig for _, s in recent):
                if first_sig in self._emitted_for:
                    return None
                self._emitted_for.add(first_sig)
                return LoopFinding(
                    pattern="tool_repeat",
                    tool=call.name,
                    repeats=self.repeat_threshold,
                    directive=(
                        f"You have called {call.name!r} with identical arguments "
                        f"{self.repeat_threshold} times in a row. The output is not "
                        f"changing. Try a different approach: inspect the result more "
                        f"carefully, call a different tool, or if the work is actually "
                        f"complete, call verify_work and then return."
                    ),
                )

        # Pattern 2: no_progress — no mutating tool in last K calls.
        if len(self._signatures) >= self.no_progress_threshold:
            recent = list(self._signatures)[-self.no_progress_threshold :]
            assert self.mutating_tools is not None  # post_init guarantees this
            if not any(_is_mutating(name, self.mutating_tools) for name, _ in recent):
                key = f"no_progress::{recent[-1][1]}"
                if key in self._emitted_for:
                    return None
                self._emitted_for.add(key)
                return LoopFinding(
                    pattern="no_progress",
                    tool=None,
                    repeats=self.no_progress_threshold,
                    directive=(
                        f"You have made {self.no_progress_threshold} tool calls without "
                        f"writing any files or running any commands that change state. "
                        f"If you have enough information, take action — write the fix, "
                        f"run the tests. If you are blocked, say so and stop."
                    ),
                )

        return None

    def reset(self) -> None:
        """Drop window state. Useful when the agent enters a new phase."""
        self._signatures.clear()
        self._emitted_for.clear()


def _is_mutating(tool_name: str, mutating: Iterable[str]) -> bool:
    """Match the registered tool name against the mutating-tool hint set.

    We accept exact match and a "contains" fallback so common name variants
    (``fs_write_file``, ``shell.run``) are caught without needing every
    package to keep a hardcoded list in sync.
    """
    name = tool_name.lower()
    return any(name == hint or hint in name for hint in mutating)


__all__ = ["LoopDetector", "LoopFinding", "LoopPattern"]
