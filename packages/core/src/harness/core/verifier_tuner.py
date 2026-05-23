"""ACON-style failure-driven prompt tuner.

Pattern borrowed from *ACON: Optimizing Context Compression for Long-
horizon LLM Agents* (Kang et al., arXiv 2510.00615). ACON's loop:

  1. Run an agent in two configurations (full context vs compressed).
  2. Collect pairs where the compressed version regressed.
  3. Hand the divergence to an LLM along with the current guideline.
  4. The LLM proposes a natural-language delta to the guideline.
  5. The delta is reviewed; if approved, the guideline is versioned.

Our analog: instead of compression guidelines, we tune *verifier and
critic prompts*. The eval harness already produces the paired traces
(defended vs bare; failed vs passed); we just didn't have a loop that
mines them. This module supplies the loop.

The output is **always advisory** — a `ProposedDelta` with a candidate
new prompt and a rationale. We don't auto-apply. The reviewer picks
yes/no and the prompt store records the version chain.

Storage: `.harness/tuned-prompts/<key>.json` carries the version chain
for one tunable prompt:

    {
      "key": "minimal_fix_verifier",
      "versions": [
        {"version": 1, "text": "...", "rationale": "seed", "created_at": "..."},
        {"version": 2, "text": "...", "rationale": "...", "created_at": "..."}
      ]
    }

The runtime reads `versions[-1].text` at load time. Rollback = delete
the latest entry. No automatic mutation of the file by the runtime.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.core.telemetry import get_logger

logger = get_logger("harness.verifier_tuner")


DEFAULT_TUNED_DIR = Path(".harness") / "tuned-prompts"


@dataclass
class PromptVersion:
    version: int
    text: str
    rationale: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class TunablePrompt:
    """Versioned prompt store, file-backed.

    Args:
        key: identifier of the prompt being tuned. Usually a verifier
            class name in snake_case (e.g. ``"minimal_fix_verifier"``).
        versions: ordered list. ``versions[-1]`` is the current text.
    """

    key: str
    versions: list[PromptVersion] = field(default_factory=list)

    @property
    def current(self) -> PromptVersion | None:
        return self.versions[-1] if self.versions else None

    def add_version(self, text: str, rationale: str = "") -> PromptVersion:
        next_version = (self.versions[-1].version + 1) if self.versions else 1
        entry = PromptVersion(version=next_version, text=text, rationale=rationale)
        self.versions.append(entry)
        return entry

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "versions": [asdict(v) for v in self.versions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TunablePrompt:
        versions = [
            PromptVersion(
                version=int(v.get("version", 0)),
                text=str(v.get("text") or ""),
                rationale=str(v.get("rationale") or ""),
                created_at=float(v.get("created_at", time.time())),
            )
            for v in (data.get("versions") or [])
            if isinstance(v, dict)
        ]
        return cls(key=str(data.get("key") or ""), versions=versions)

    @classmethod
    def load(cls, path: Path) -> TunablePrompt | None:
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("tuner.parse_failed", path=str(path), error=str(exc))
            return None
        if not isinstance(raw, dict):
            return None
        return cls.from_dict(raw)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.as_dict(), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)


# ---------------------------------------------------------------------------
# Trajectory pair mining
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryPair:
    """A single (defended-win, bare-win-or-loss) comparison.

    Each entry is a short transcript excerpt + the verifier verdict + the
    final outcome. The tuner LLM reads these side-by-side and proposes
    why the prompt under tuning helped or hurt.
    """

    fixture: str
    defended_excerpt: str
    defended_outcome: str
    """e.g. "PASS overall=5/5" / "FAIL scope=1/5"."""
    bare_excerpt: str
    bare_outcome: str
    differing_dimension: str | None = None
    """The dimension where the two trials diverged most (if known)."""


@dataclass
class TuneRequest:
    """Inputs to the LLM tuner for a single verifier/critic prompt.

    Args:
        prompt_key: the TunablePrompt key being tuned.
        current_prompt: the verifier's current prompt text.
        pairs: trajectory pairs distilled from prior evals.
        notes: free-form reviewer hints (e.g. "scope creep on F04").
    """

    prompt_key: str
    current_prompt: str
    pairs: list[TrajectoryPair]
    notes: str = ""


@dataclass
class ProposedDelta:
    """The tuner's proposal for one tunable prompt.

    The reviewer reads ``rationale`` to decide whether to accept the
    proposal. ``new_prompt`` is the full replacement text. We deliberately
    don't return a patch / diff: small targeted edits to a prompt cause
    more confusion than they're worth, and the tuner LLM produces better
    proposals when allowed to rewrite holistically.
    """

    prompt_key: str
    new_prompt: str
    rationale: str
    raw_response: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Prompt rendering + parsing
# ---------------------------------------------------------------------------

TUNER_SYSTEM = (
    "You are a prompt-tuning assistant. You will be shown the current "
    "prompt for a verifier/critic in an LLM agent harness, along with "
    "paired trajectories from an A/B eval (defended vs bare). Your job "
    "is to propose a revised prompt that would have helped the harness "
    "catch the failure mode visible in the pairs.\n\n"
    "Rules:\n"
    "  • Keep the prompt the same length or shorter. Verbose prompts "
    "    regress on instruction-following.\n"
    "  • Make at most TWO substantive changes. Targeted edits beat "
    "    holistic rewrites.\n"
    "  • Keep all variable-substitution placeholders (`{...}`) intact.\n"
    "  • Don't reference specific fixtures by name. The prompt has to "
    "    generalize across tasks.\n\n"
    "Return ONLY a JSON object:\n"
    '  {"new_prompt": "...", "rationale": "<one paragraph>"}'
)


def render_tune_prompt(request: TuneRequest) -> str:
    """Build the user message handed to the tuner LLM."""
    pair_lines: list[str] = []
    for i, p in enumerate(request.pairs, 1):
        pair_lines.append(
            f"\n--- PAIR {i}: fixture={p.fixture}, divergence={p.differing_dimension or 'unknown'} ---\n"
            f"[DEFENDED] {p.defended_outcome}\n{p.defended_excerpt.strip()[:2000]}\n"
            f"[BARE]     {p.bare_outcome}\n{p.bare_excerpt.strip()[:2000]}\n"
        )
    notes_block = f"\nReviewer notes: {request.notes.strip()}\n" if request.notes.strip() else ""
    return (
        f"PROMPT KEY: {request.prompt_key}\n"
        f"\n--- CURRENT PROMPT ---\n{request.current_prompt.strip()}\n"
        f"\n--- PAIRED TRAJECTORIES ---{''.join(pair_lines)}"
        f"{notes_block}"
        "\nPropose a revised prompt."
    )


def parse_proposal(response: str, *, prompt_key: str) -> ProposedDelta | None:
    """Coerce the tuner LLM's JSON response into a ProposedDelta.

    Returns None when the response is unparseable. The caller logs a
    warning and lets the user retry rather than guessing.
    """
    body = response.strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        logger.warning("tuner.bad_json", preview=body[:200])
        return None
    if not isinstance(payload, dict):
        return None
    new_prompt = str(payload.get("new_prompt") or "").strip()
    rationale = str(payload.get("rationale") or "").strip()
    if not new_prompt:
        return None
    return ProposedDelta(
        prompt_key=prompt_key,
        new_prompt=new_prompt,
        rationale=rationale,
        raw_response=response,
    )


__all__ = [
    "DEFAULT_TUNED_DIR",
    "TUNER_SYSTEM",
    "PromptVersion",
    "ProposedDelta",
    "TrajectoryPair",
    "TunablePrompt",
    "TuneRequest",
    "parse_proposal",
    "render_tune_prompt",
]
