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

import re
from collections import Counter
from dataclasses import dataclass, field

from harness.core.activity import (
    ACTION_CANONICALIZED,
    AGENT_RUN_STALLED,
    ENV_CONTRACT_INJECTED,
    PROCEDURAL_TIP_INJECTED,
    REPAIR_DIRECTIVE_ISSUED,
    TOOL_CALL_COMPLETED,
    TRAJECTORY_REGULATED,
    USAGE_RECORDED,
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
    # LifeHarness L1..L4 — counts how often each layer fired during the run.
    # L1 (contracts injected at run start) and L2 (tips injected) typically
    # show 0..1 because both only act once per run. L3 (action canonical
    # rewrites) and L4 (trajectory regulation interventions) can climb.
    contracts_injected: int = 0
    tips_injected: int = 0
    actions_canonicalized: int = 0
    trajectory_regulations: int = 0
    # Prompt-cache accounting (Anthropic-style cache_read / cache_creation
    # tokens). Aggregated across all turns of the run. The hit ratio is
    # surfaced in `format_ledger` when at least one cache-aware turn fired.
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    prompt_tokens_total: int = 0

    def is_empty(self) -> bool:
        return (
            not self.verifier_passes
            and not self.verifier_blocks
            and self.repair_attempts == 0
            and self.critic_invocations == 0
            and not self.tool_calls
            and not self.stalled
            and self.contracts_injected == 0
            and self.tips_injected == 0
            and self.actions_canonicalized == 0
            and self.trajectory_regulations == 0
            and self.cache_creation_tokens == 0
            and self.cache_read_tokens == 0
        )

    @property
    def cache_hit_ratio(self) -> float | None:
        """Fraction of input tokens served from cache. None if no cache data."""
        cache_eligible = self.cache_read_tokens + self.cache_creation_tokens
        if cache_eligible == 0:
            return None
        return self.cache_read_tokens / cache_eligible

    def to_dict(self) -> dict[str, object]:
        return {
            "verifier_passes": dict(self.verifier_passes),
            "verifier_blocks": dict(self.verifier_blocks),
            "repair_attempts": self.repair_attempts,
            "critic_invocations": self.critic_invocations,
            "tool_calls": dict(self.tool_calls),
            "tool_errors": dict(self.tool_errors),
            "stalled": self.stalled,
            "contracts_injected": self.contracts_injected,
            "tips_injected": self.tips_injected,
            "actions_canonicalized": self.actions_canonicalized,
            "trajectory_regulations": self.trajectory_regulations,
            "cache_creation_tokens": self.cache_creation_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "prompt_tokens_total": self.prompt_tokens_total,
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
        elif ev.kind == ENV_CONTRACT_INJECTED:
            ledger.contracts_injected += int(ev.data.get("count", 0) or 0)
        elif ev.kind == PROCEDURAL_TIP_INJECTED:
            ledger.tips_injected += int(ev.data.get("count", 0) or 0)
        elif ev.kind == ACTION_CANONICALIZED:
            ledger.actions_canonicalized += 1
        elif ev.kind == TRAJECTORY_REGULATED:
            ledger.trajectory_regulations += 1
        elif ev.kind == USAGE_RECORDED:
            ledger.prompt_tokens_total += int(ev.data.get("prompt_tokens", 0) or 0)
            ledger.cache_creation_tokens += int(ev.data.get("cache_creation_input_tokens", 0) or 0)
            ledger.cache_read_tokens += int(ev.data.get("cache_read_input_tokens", 0) or 0)
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
    # LifeHarness layers — only list those that actually fired.
    life_parts: list[str] = []
    if ledger.contracts_injected:
        life_parts.append(f"L1 contracts x{ledger.contracts_injected}")
    if ledger.tips_injected:
        life_parts.append(f"L2 tips x{ledger.tips_injected}")
    if ledger.actions_canonicalized:
        life_parts.append(f"L3 canonical x{ledger.actions_canonicalized}")
    if ledger.trajectory_regulations:
        life_parts.append(f"L4 regulate x{ledger.trajectory_regulations}")
    if life_parts:
        lines.append("  lifeharness: " + ", ".join(life_parts))
    if ledger.cache_read_tokens or ledger.cache_creation_tokens:
        ratio = ledger.cache_hit_ratio
        ratio_str = f"{ratio:.1%}" if ratio is not None else "n/a"
        lines.append(
            f"  cache: {ledger.cache_read_tokens} read / "
            f"{ledger.cache_creation_tokens} written (hit ratio {ratio_str})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parsing & correlation
# ---------------------------------------------------------------------------


_VERIFIER_RE = re.compile(r"(\w+)=(\d+)✓/(\d+)✗")
_REPAIR_RE = re.compile(r"repair attempts:\s*(\d+)(?:\s*\(critic in (\d+)/\d+\))?")
_STALLED_RE = re.compile(r"stalled:\s*yes", re.IGNORECASE)


def parse_ledger_text(text: str) -> DefenseLedger | None:
    """Reverse of ``format_ledger`` — extract structured form from a transcript.

    The defense ledger is printed at end-of-run by the CLI. Tools that consume
    captured transcripts (e.g. the eval framework) parse it back into a
    ``DefenseLedger`` to correlate firings with downstream outcomes.

    Returns None if no ledger block is found.
    """
    if not text or "defense ledger:" not in text:
        return None
    # Slice from the marker to a blank line / end-of-block.
    start = text.index("defense ledger:")
    block = text[start:]
    # Stop at the first non-indented non-empty line that isn't the header,
    # or 8 lines down — whichever first.
    lines = block.splitlines()[:8]

    ledger = DefenseLedger()
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if not (line.startswith(" ") or line.startswith("\t")):
            break  # left the indented block
        if stripped.startswith("verifiers:"):
            for m in _VERIFIER_RE.finditer(stripped):
                name = m.group(1)
                passes = int(m.group(2))
                blocks = int(m.group(3))
                if passes:
                    ledger.verifier_passes[name] = passes
                if blocks:
                    ledger.verifier_blocks[name] = blocks
        elif stripped.startswith("repair attempts:"):
            rm = _REPAIR_RE.search(stripped)
            if rm:
                ledger.repair_attempts = int(rm.group(1))
                if rm.group(2):
                    ledger.critic_invocations = int(rm.group(2))
        elif _STALLED_RE.search(stripped):
            ledger.stalled = True
        # tools: line is parsed below if needed — skipping for the correlation
        # use-case since we already track defenses, not raw tool calls, there.

    return None if ledger.is_empty() else ledger


@dataclass
class DefenseStat:
    """How a single defense correlates with PASS/FAIL outcomes."""

    name: str
    block_pass: int = 0  # blocked AND trial passed (defense helped or was OK)
    block_fail: int = 0  # blocked AND trial failed (defense fired but didn't save)
    silent_pass: int = 0  # never blocked AND trial passed (defense not needed)
    silent_fail: int = 0  # never blocked AND trial failed (defense missed the problem)

    @property
    def total(self) -> int:
        return self.block_pass + self.block_fail + self.silent_pass + self.silent_fail

    @property
    def block_pass_rate(self) -> float | None:
        fires = self.block_pass + self.block_fail
        return None if fires == 0 else self.block_pass / fires

    def verdict(self) -> str:
        """Human-readable categorical verdict.

        - "helps": when it fires, the trial mostly still passes (defense
          caught something correctable AND agent recovered).
        - "hurts": when it fires, the trial mostly fails (defense correlated
          with failure — either firing wrongly or pushing the agent into a
          worse state via repair).
        - "neutral": between the two.
        - "n/a": never fired.
        """
        rate = self.block_pass_rate
        if rate is None:
            return "n/a"
        fires = self.block_pass + self.block_fail
        if fires < 3:
            return "n/a (small N)"
        if rate >= 0.66:
            return "helps"
        if rate <= 0.33:
            return "hurts"
        return "neutral"


def correlate_defenses(
    trials: list[tuple[DefenseLedger | None, bool]],
) -> list[DefenseStat]:
    """For each defense seen across trials, tally block/silent vs pass/fail.

    ``trials`` is a list of (ledger, trial_passed). Ledger may be None for
    trials where parsing failed; those count as "silent" for every defense.
    """
    all_names: set[str] = set()
    for ledger, _ in trials:
        if ledger is None:
            continue
        all_names |= set(ledger.verifier_passes)
        all_names |= set(ledger.verifier_blocks)
    # Always include critic as a "defense" so we can see whether it correlates.
    has_critic_data = any(
        ledger is not None and ledger.critic_invocations > 0 for ledger, _ in trials
    )
    if has_critic_data:
        all_names.add("critic")

    stats: dict[str, DefenseStat] = {n: DefenseStat(name=n) for n in all_names}
    for ledger, passed in trials:
        for name in all_names:
            if ledger is None:
                # Trial happened but ledger unparseable — treat as silent for everything.
                if passed:
                    stats[name].silent_pass += 1
                else:
                    stats[name].silent_fail += 1
                continue
            if name == "critic":
                fired = ledger.critic_invocations > 0
            else:
                fired = ledger.verifier_blocks.get(name, 0) > 0
            if fired and passed:
                stats[name].block_pass += 1
            elif fired and not passed:
                stats[name].block_fail += 1
            elif not fired and passed:
                stats[name].silent_pass += 1
            else:
                stats[name].silent_fail += 1

    # Sort: hurts first (highest priority to investigate), then by name.
    order = {"hurts": 0, "neutral": 1, "helps": 2, "n/a (small N)": 3, "n/a": 4}
    return sorted(
        stats.values(),
        key=lambda s: (order.get(s.verdict(), 5), s.name),
    )


__all__ = [
    "DefenseLedger",
    "DefenseStat",
    "build_ledger",
    "correlate_defenses",
    "format_ledger",
    "parse_ledger_text",
]
