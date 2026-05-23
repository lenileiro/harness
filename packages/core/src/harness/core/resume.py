"""Resume contract for cross-session agent continuity.

Pattern borrowed from Anthropic's `Effective harnesses for long-running
agents`: a discrete-shift agent reads a structured ``resume.json`` on
boot, derives state from it, and updates a single ``progress.txt`` at the
end of every session. The two together let a fresh agent pick up where
the previous one stopped without replaying the entire conversation.

We diverge from the post in three ways:

  • JSON-only: we don't keep a separate ``progress.txt``. The single
    ``resume.json`` carries everything (feature list, single in-flight
    feature, phase plan if any, last-session id, notes). The post argues
    JSON beats Markdown because models edit it less recklessly.
  • Phase-aware: when a session has phases declared, the in-flight
    feature inherits the phase plan. Resume becomes "continue phase X".
  • Optional: resume injection is opt-in via the CLI. Sessions that
    don't care about cross-session continuity just don't write a
    ``resume.json``; the runtime skips injection.

Storage location: ``.harness/resume.json`` relative to the working
directory. The file is hand-editable; we never auto-generate phase
plans without explicit caller intent.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.core.telemetry import get_logger

logger = get_logger("harness.resume")

DEFAULT_RESUME_PATH = Path(".harness") / "resume.json"
"""Where the resume contract lives, relative to ``cwd``."""


@dataclass
class FeatureItem:
    """One feature on the resume contract's roadmap.

    Args:
        name: short kebab-case identifier (e.g., ``"add-power-method"``).
        description: one-paragraph summary of what shipping this means.
        status: ``"pending"`` / ``"in_progress"`` / ``"done"``.
        phases: optional ordered phase plan inherited by the agent if
            this feature is the current in-flight one.
        notes: free-form notes from prior sessions; appended to over time.
        session_id: the session that last touched this feature (when known).
    """

    name: str
    description: str = ""
    status: str = "pending"
    phases: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    session_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FeatureItem:
        return cls(
            name=str(data.get("name", "")).strip(),
            description=str(data.get("description") or ""),
            status=str(data.get("status") or "pending"),
            phases=[str(p) for p in (data.get("phases") or []) if str(p).strip()],
            notes=[str(n) for n in (data.get("notes") or []) if str(n).strip()],
            session_id=data.get("session_id"),
        )


@dataclass
class ResumeContract:
    """Cross-session state for a long-running agent workspace.

    Single-feature-per-session is a design rule, not a hard constraint:
    ``current`` names the one feature the agent should focus on this
    session. Hand-edit the file to advance ``current`` to the next
    feature. The agent updates ``status``/``notes`` on the in-flight
    feature as it works; everything else stays editor-owned.
    """

    current: str | None = None
    """Name of the feature the agent should focus on this session."""

    features: list[FeatureItem] = field(default_factory=list)
    """Ordered roadmap. Agent never reorders this; only the human does."""

    notes: list[str] = field(default_factory=list)
    """Workspace-wide notes outside any specific feature."""

    def feature(self, name: str) -> FeatureItem | None:
        for f in self.features:
            if f.name == name:
                return f
        return None

    def current_feature(self) -> FeatureItem | None:
        return self.feature(self.current) if self.current else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "current": self.current,
            "features": [f.as_dict() for f in self.features],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResumeContract:
        return cls(
            current=data.get("current"),
            features=[
                FeatureItem.from_dict(f)
                for f in (data.get("features") or [])
                if isinstance(f, dict)
            ],
            notes=[str(n) for n in (data.get("notes") or []) if str(n).strip()],
        )

    # ------------------------------------------------------------------ #
    # Disk I/O                                                            #
    # ------------------------------------------------------------------ #

    @classmethod
    def load(cls, path: Path) -> ResumeContract | None:
        """Read a resume contract from disk. Returns None when missing or
        malformed; the runtime treats both as 'no resume context'."""
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("resume.parse_failed", path=str(path), error=str(exc))
            return None
        if not isinstance(raw, dict):
            logger.warning("resume.bad_shape", path=str(path), got=type(raw).__name__)
            return None
        return cls.from_dict(raw)

    def save(self, path: Path) -> None:
        """Atomic-ish write: tmp file then rename."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)

    # ------------------------------------------------------------------ #
    # Rendering for the system prompt                                     #
    # ------------------------------------------------------------------ #

    def render_for_prompt(self) -> str | None:
        """Render the contract into a system-message block.

        Returns None when there's no current feature — there's nothing
        useful to inject. The agent should still rely on `.harness/`
        memory and the activity log as primary state; this block is
        coarse orientation, not authoritative state.
        """
        cur = self.current_feature()
        if cur is None:
            return None
        lines: list[str] = ["[harness:resume] cross-session state for this run:"]
        lines.append(f"  current feature: {cur.name} (status={cur.status})")
        if cur.description:
            lines.append(f"  description: {cur.description}")
        if cur.phases:
            lines.append(f"  phases: {', '.join(cur.phases)}")
        if cur.notes:
            lines.append("  prior-session notes:")
            for n in cur.notes[-5:]:  # cap to last 5 to keep the block tight
                lines.append(f"    - {n}")
        if cur.session_id:
            lines.append(f"  last session: {cur.session_id}")
        # Other roadmap entries appear as a single roster line so the
        # agent knows what comes after but isn't tempted to wander.
        others = [f for f in self.features if f.name != cur.name]
        if others:
            roster = ", ".join(f"{f.name}({f.status})" for f in others)
            lines.append(f"  roadmap (do not touch this session): {roster}")
        return "\n".join(lines)


__all__ = ["DEFAULT_RESUME_PATH", "FeatureItem", "ResumeContract"]
