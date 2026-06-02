"""L4 — Trajectory regulation: detect degenerate tool-call patterns.

Spec borrowed from the LifeHarness paper (Peking U., 2026). Their L4 layer
"monitors post-execution dynamics, detects degenerate patterns such as
repetition, stagnation, and invalid retries." We focus on the two patterns
that show up reliably in our eval traces:

  tool_repeat   — the agent calls the same (tool, args) signature N times
                  in a row with identical output. The next call is unlikely
                  to change anything; intervene with a repair directive.

  failed_tool_retry — the agent keeps receiving the same tool failure symptom
                  without changing state. The next retry is unlikely to help.

  no_progress   — the agent has produced no file edits, concrete probes, or
                  verification across a sustained window despite emitting
                  tool calls. Suggests it may be spinning in read-only loops.

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

LoopPattern = Literal["tool_repeat", "failed_tool_retry", "stale_edit_retry", "no_progress"]

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
            in a row before ``no_progress`` fires. Default 10 — long-horizon
            work in unfamiliar repositories often needs several distinct
            read/search calls before an edit is justified.
        window: max recent calls to retain. Larger than both thresholds so
            we can distinguish "3 in a row" from "3 within 10 turns."
        mutating_tools: override the set of tools considered progress.
            When None, ``_MUTATING_TOOL_HINTS`` is used.
    """

    repeat_threshold: int = 3
    no_progress_threshold: int = 10
    stale_edit_threshold: int = 3
    window: int = 12
    mutating_tools: frozenset[str] | None = None
    _signatures: deque[tuple[str, str, str | None, str | None, bool]] = field(
        default_factory=deque, init=False, repr=False
    )
    """Each entry is (tool_name, signature_hash, stale_edit_path, failure_hash, made_progress)."""
    _emitted_for: set[str] = field(default_factory=set, init=False, repr=False)
    """Signatures we've already nagged about; prevents repeat-emission spam."""

    def __post_init__(self) -> None:
        if self.repeat_threshold < 2:
            raise ValueError("repeat_threshold must be >= 2")
        if self.no_progress_threshold < 2:
            raise ValueError("no_progress_threshold must be >= 2")
        if self.stale_edit_threshold < 2:
            raise ValueError("stale_edit_threshold must be >= 2")
        if self.window < self.repeat_threshold or self.window < self.no_progress_threshold:
            raise ValueError("window must be >= both thresholds")
        if self.window < self.stale_edit_threshold:
            raise ValueError("window must be >= stale_edit_threshold")
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
        stale_edit_path = _stale_edit_path(call, result)
        failure_sig = _failed_tool_signature(result)
        made_progress = _made_progress(call.name, result, self.mutating_tools or frozenset())
        self._signatures.append((call.name, sig, stale_edit_path, failure_sig, made_progress))
        while len(self._signatures) > self.window:
            self._signatures.popleft()

        # Pattern 1: tool_repeat — last N entries identical.
        if len(self._signatures) >= self.repeat_threshold:
            recent = list(self._signatures)[-self.repeat_threshold :]
            first_sig = recent[0][1]
            if all(s == first_sig for _, s, _, _, _ in recent):
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

        # Pattern 2: failed_tool_retry — repeated failures with the same symptom
        # and no intervening state change. This catches loops where the model
        # keeps varying shell quoting or verify_work wrappers around the same
        # underlying failure.
        if failure_sig is not None and len(self._signatures) >= self.repeat_threshold:
            recent = list(self._signatures)[-self.repeat_threshold :]
            if all(failed == failure_sig and not progress for _, _, _, failed, progress in recent):
                key = f"failed_tool_retry::{failure_sig}"
                if key in self._emitted_for:
                    return None
                self._emitted_for.add(key)
                return LoopFinding(
                    pattern="failed_tool_retry",
                    tool=call.name,
                    repeats=self.repeat_threshold,
                    directive=(
                        f"Your last {self.repeat_threshold} tool attempts failed with the "
                        "same underlying error. Stop rerunning variants of the same check. "
                        "Treat the failure output as evidence and continue autonomously with "
                        "a materially different next action: change the implementation, use a "
                        "different tool, inspect different project evidence, or choose a "
                        "materially different verification command."
                    ),
                )

        # Pattern 3: stale_edit_retry — repeated failed edit_file attempts against
        # the same path with stale context, even when interleaved with rereads.
        if stale_edit_path is not None and len(self._signatures) >= self.stale_edit_threshold:
            stale_matches = [
                path
                for _, _, path, _, _ in self._signatures
                if path is not None and path == stale_edit_path
            ]
            if len(stale_matches) >= self.stale_edit_threshold:
                key = f"stale_edit_retry::{stale_edit_path}"
                if key in self._emitted_for:
                    return None
                self._emitted_for.add(key)
                return LoopFinding(
                    pattern="stale_edit_retry",
                    tool=call.name,
                    repeats=len(stale_matches),
                    directive=(
                        f"Your last edit attempts for {stale_edit_path!r} are failing "
                        "because the old text is not present. Stop retrying the same "
                        "patch. Use the freshly read file content to choose an exact "
                        "existing snippet, or make a smaller targeted edit. After the "
                        "change applies, call verify_work."
                    ),
                )

        # Pattern 4: no_progress — no successful progress-making tool in last K
        # calls. Failed shell and failed verify_work calls are evidence, but they
        # are not progress by themselves.
        if len(self._signatures) >= self.no_progress_threshold:
            recent = list(self._signatures)[-self.no_progress_threshold :]
            if not any(progress for _, _, _, _, progress in recent):
                key = "no_progress"
                if key in self._emitted_for:
                    return None
                self._emitted_for.add(key)
                return LoopFinding(
                    pattern="no_progress",
                    tool=None,
                    repeats=self.no_progress_threshold,
                    directive=(
                        f"You have made {self.no_progress_threshold} tool calls without "
                        "writing files, changing state, or verifying behavior. Before "
                        "continuing broad inspection, state the current hypothesis and "
                        "what result you expect from the next action. Then take a "
                        "concrete next step: run a focused verification/probe, inspect "
                        "a specific missing dependency, or edit only if the evidence is "
                        "strong enough. If the available evidence proves one path is "
                        "blocked, treat that as evidence and choose a materially "
                        "different autonomous path."
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


def _made_progress(
    tool_name: str,
    result: ToolResult | None,
    mutating: Iterable[str],
) -> bool:
    if not _is_mutating(tool_name, mutating):
        return False
    return not (result is not None and result.is_error)


def _failed_tool_signature(result: ToolResult | None) -> str | None:
    if result is None or not result.is_error:
        return None
    content = " ".join(str(result.content or "").split()).lower()
    if not content:
        return None
    for marker in ("traceback", "usage:", "stderr:"):
        index = content.rfind(marker)
        if index >= 0:
            content = content[index:]
            break
    content = content[-300:]
    return hashlib.sha1(content.encode()).hexdigest()[:16]


def _stale_edit_path(call: ToolCall, result: ToolResult | None) -> str | None:
    if result is None or not result.is_error:
        return None
    name = call.name.lower()
    if "edit_file" not in name and name != "edit":
        return None
    content = (result.content or "").lower()
    if "old" not in content or "not found" not in content:
        return None
    if not isinstance(call.arguments, dict):
        return None
    path = call.arguments.get("path")
    return path if isinstance(path, str) and path else None


__all__ = ["LoopDetector", "LoopFinding", "LoopPattern"]
