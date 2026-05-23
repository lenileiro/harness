"""Tiered denylist for shell-style tool calls.

Both ``ShellTool`` and ``VerifyWorkTool`` accept an arbitrary command string
and pipe it to ``/bin/sh``. That's a real arson tool unless we cap blast
radius. The model's ``cwd`` is just the *starting* directory — a shell can
``cd /`` and operate anywhere the user can.

This module owns the structural denylist, modeled on Claude Code's auto-mode
classifier:

  - ``hard``  — unconditional. Cannot be overridden by user intent or config.
                Universal-destruction patterns, privilege escalation, credential
                destruction, security-control disabling.
  - ``soft``  — blocked by default. The intent is "user can override via config
                if they really mean it"; for now we still refuse, but the tier
                tells callers it's recoverable in principle.

It's not a substitute for a real OS sandbox (bubblewrap / seatbelt / Docker).
It's a "don't let the LLM accidentally rm -rf when an arson tool exists in
its surface" backstop.
"""

from __future__ import annotations

import re
from typing import Literal

DenyTier = Literal["hard", "soft"]


# Each entry is (pattern, tier, reason). Patterns are matched against the
# command via re.search (not anchored) so they catch the dangerous fragment
# wherever it appears in a longer command line.
_DENYLIST: tuple[tuple[re.Pattern[str], DenyTier, str], ...] = (
    # ── HARD: universal destruction ────────────────────────────────────────
    # rm -rf at root or bare home. Subpaths like ~/.cache stay allowed —
    # those are deliberate cleanup the user/agent owns.
    (
        re.compile(
            r"\brm\s+(?:-[a-zA-Z]*[rRfF][a-zA-Z]*\s+)+"
            r"(?:/(?=\s|$|\*)|~(?=\s|$)|\$HOME(?=\s|$)|/\*)"
        ),
        "hard",
        "rm -rf at filesystem root or bare home (irreversible)",
    ),
    # rm -rf inside credential directories — destroys user secrets.
    (
        re.compile(
            r"\brm\s+(?:-[a-zA-Z]*[rRfF][a-zA-Z]*\s+)+(?:~|\$HOME)/"
            r"(?:\.ssh|\.aws|\.config/gcloud|\.gnupg)\b"
        ),
        "hard",
        "rm -rf inside a credentials directory",
    ),
    # ── HARD: privilege escalation ─────────────────────────────────────────
    (
        re.compile(r"(?:^|[\s;&|])sudo\b"),
        "hard",
        "sudo (privilege escalation)",
    ),
    (
        re.compile(r"(?:^|[\s;&|])su\s"),
        "hard",
        "su (user switching)",
    ),
    # ── HARD: security-control disabling ───────────────────────────────────
    (
        re.compile(r"\b(?:ufw|iptables|firewall-cmd)\s+(?:disable|flush|stop|--flush)\b"),
        "hard",
        "firewall disable/flush",
    ),
    (
        re.compile(r"\bspctl\s+--master-disable\b"),
        "hard",
        "macOS Gatekeeper disable",
    ),
    # ── HARD: credential write ─────────────────────────────────────────────
    (
        re.compile(r"(?:>{1,2}|tee)\s+~/\.ssh/|(?:>{1,2}|tee)\s+\$HOME/\.ssh/"),
        "hard",
        "write to ~/.ssh (overwrites SSH config/keys)",
    ),
    # ── HARD: storage / device destruction ─────────────────────────────────
    (
        re.compile(r"\bdd\b[^;&|]*\bof=/dev/"),
        "hard",
        "dd writing to a block device (data destruction)",
    ),
    (
        re.compile(r"\bmkfs\.\S+\s+/dev/"),
        "hard",
        "mkfs (formatting a device)",
    ),
    # ── HARD: fork bomb ────────────────────────────────────────────────────
    (
        re.compile(r":\(\)\s*{[^}]*:\|:[^}]*}"),
        "hard",
        "fork bomb",
    ),
    # ── HARD: chmod that loosens security on system dirs ───────────────────
    (
        re.compile(r"\bchmod\s+(?:0?7[5-7]{2}|a\+w|\+w)\s+/(?:etc|usr|bin|sbin|var)"),
        "hard",
        "chmod loosens permissions on a system directory",
    ),
    # ── SOFT: package publishing ───────────────────────────────────────────
    # Real engineers do publish from dev machines; flagging keeps an agent
    # from accidentally shipping a package mid-debug.
    (
        re.compile(r"\bnpm\s+publish\b"),
        "soft",
        "npm publish (releases a package — explicit user intent expected)",
    ),
    (
        re.compile(r"\b(?:gem|cargo|poetry|twine|maturin)\s+(?:push|publish|upload)\b"),
        "soft",
        "package publish/upload (releases an artifact)",
    ),
    (
        re.compile(r"\bpypi-cli\s+upload\b|\bdocker\s+push\b"),
        "soft",
        "registry push (publishes an artifact)",
    ),
    # ── SOFT: history rewrite on remotes ───────────────────────────────────
    (
        re.compile(r"\bgit\s+push\b[^;&|]*(?:--force\b|-f\b|--force-with-lease\b)"),
        "soft",
        "git push --force (rewrites remote history)",
    ),
    # ── SOFT: pipe-to-shell ────────────────────────────────────────────────
    # Real installers use this pattern (rustup, nvm, etc.), so it's soft.
    (
        re.compile(r"(?:curl|wget|fetch)\s[^|]*\|\s*(?:sh|bash|zsh|ksh|python|perl|ruby|node)\b"),
        "soft",
        "curl/wget piped to shell (remote code execution from untrusted source)",
    ),
    # ── SOFT: writes to system locations ───────────────────────────────────
    # Setup scripts can legitimately need these; not unconditional.
    (
        re.compile(r"(?:>{1,2}|tee)\s+/(?:etc|usr|bin|sbin|boot|System|Library)/"),
        "soft",
        "write to system directory (admin-style write)",
    ),
)


def check_dangerous_command(command: str) -> tuple[DenyTier, str] | None:
    """Return ``(tier, reason)`` if `command` matches the denylist, else None.

    Callers refuse the call and surface ``reason`` to the agent. The ``tier``
    is intended for downstream policy:
      - ``"hard"``: never overridable. Callers MUST refuse.
      - ``"soft"``: overridable in principle by user config / explicit intent.
        Today we still refuse — the tier tells you the verdict is reversible
        as a future policy decision, not that it's already overridden.

    This check is deliberately not configurable from a tool argument, so the
    model can't suppress it by adding a flag.
    """
    if not command:
        return None
    for pattern, tier, reason in _DENYLIST:
        if pattern.search(command):
            return (tier, reason)
    return None


__all__ = ["DenyTier", "check_dangerous_command"]
