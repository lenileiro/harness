"""Defense ledger — summarize which anti-lying defenses fired in a run.

Scans an `ActivityEvent` ledger and counts the high-signal events the harness
emits: verifier verdicts (pass/block per verifier), repair attempts, critic
invocations, tool calls grouped by name, and stall events.

Callers invoke ``build_ledger(activity)`` at end-of-run and render the result
so users can tell what the harness actually did, not just what the agent
claimed. The ledger is observability, not control flow — failures here must
never crash the run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from harness.core.activity import (
    AGENT_RUN_STALLED,
    REPAIR_DIRECTIVE_ISSUED,
    TOOL_CALL_COMPLETED,
    VERIFICATION_COMPLETED,
    ActivityEvent,
)


@dataclass
class DefenseLedger:
    """Aggregate counts of defenses that fired during a single agent run."""

    verifier_passes: Counter[str] = field(default_factory=Counter)
    verifier_blocks: Counter[str] = field(default_factory=Counter)
    repair_attempts: int = 0
    critic_invocations: int = 0
    tool_calls: Counter[str] = field(default_factory=Counter)
    tool_errors: Counter[str] = field(default_factory=Counter)
    stalled: bool = False

    def is_empty(self) -> bool:
        return (
            not self.verifier_passes
            and not self.verifier_blocks
            and self.repair_attempts == 0
            and self.critic_invocations == 0
            and not self.tool_calls
            and not self.stalled
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "verifier_passes": dict(self.verifier_passes),
            "verifier_blocks": dict(self.verifier_blocks),
            "repair_attempts": self.repair_attempts,
            "critic_invocations": self.critic_invocations,
            "tool_calls": dict(self.tool_calls),
            "tool_errors": dict(self.tool_errors),
            "stalled": self.stalled,
        }


def build_ledger(activity: list[ActivityEvent]) -> DefenseLedger:
    """Scan the activity ledger and tally defense-firing events."""
    ledger = DefenseLedger()
    for ev in activity:
        if ev.kind == VERIFICATION_COMPLETED:
            name = str(ev.data.get("verifier_name", "unknown"))
            if ev.data.get("can_finish"):
                ledger.verifier_passes[name] += 1
            else:
                ledger.verifier_blocks[name] += 1
        elif ev.kind == REPAIR_DIRECTIVE_ISSUED:
            ledger.repair_attempts += 1
            if ev.data.get("critic"):
                ledger.critic_invocations += 1
        elif ev.kind == TOOL_CALL_COMPLETED:
            tool_name = str(ev.data.get("name", "unknown"))
            ledger.tool_calls[tool_name] += 1
            if ev.data.get("is_error"):
                ledger.tool_errors[tool_name] += 1
        elif ev.kind == AGENT_RUN_STALLED:
            ledger.stalled = True
    return ledger


def format_ledger(ledger: DefenseLedger) -> str:
    """Compact single-block text representation suitable for CLI output."""
    if ledger.is_empty():
        return "defense ledger: (no events recorded)"

    lines: list[str] = ["defense ledger:"]
    if ledger.verifier_passes or ledger.verifier_blocks:
        verdicts: list[str] = []
        for name in sorted(set(ledger.verifier_passes) | set(ledger.verifier_blocks)):
            p = ledger.verifier_passes.get(name, 0)
            b = ledger.verifier_blocks.get(name, 0)
            verdicts.append(f"{name}={p}✓/{b}✗")
        lines.append("  verifiers: " + ", ".join(verdicts))
    if ledger.repair_attempts:
        critic_part = (
            f" (critic in {ledger.critic_invocations}/{ledger.repair_attempts})"
            if ledger.critic_invocations
            else ""
        )
        lines.append(f"  repair attempts: {ledger.repair_attempts}{critic_part}")
    if ledger.tool_calls:
        # Show the top 6 tools to keep the output compact.
        top = ledger.tool_calls.most_common(6)
        rendered = []
        for name, count in top:
            errs = ledger.tool_errors.get(name, 0)
            rendered.append(f"{name} x{count}" + (f" (err x{errs})" if errs else ""))
        lines.append("  tools: " + ", ".join(rendered))
    if ledger.stalled:
        lines.append("  stalled: yes")
    return "\n".join(lines)


__all__ = ["DefenseLedger", "build_ledger", "format_ledger"]
