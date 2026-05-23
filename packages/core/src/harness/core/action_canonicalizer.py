"""L3 — Action realization: validate and canonicalize model tool calls.

Spec borrowed from the LifeHarness paper (Peking U., 2026). Their L3 layer
"validates and canonicalizes the model-generated actions" before they hit
the environment — typo repair, missing-field defaulting, alias resolution.
The paper's setting (AL-World, 2010-vintage rigid syntax) hits this hard.
Modern code-tool APIs hit it less, but two failure modes do recur in our
eval traces and are worth catching:

  unknown_name   — the model emitted a tool name that doesn't exist in the
                   registry. Usually a near-miss (`read` vs `read_file`,
                   `Write` vs `write_file`, `bash` vs `shell`). We try a
                   high-confidence closest-match repair; if confidence is
                   below threshold we return the original (so the runtime
                   surfaces the existing "unknown tool" error).

  alias          — the model used a known alias for a registered tool name
                   (curated table below). Cheaper than fuzzy matching and
                   higher recall on the common cases.

The canonicalizer is intentionally narrow: we don't repair arguments here
(pydantic schema validation already happens in the tool implementations),
we don't emit warnings for valid tool names, and we never invent a tool.
The activity event ``action.canonicalized`` records every successful
rewrite so the defense ledger can attribute outcomes.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from dataclasses import dataclass

# Hand-curated aliases. Keys are the (lowercased) name the model emitted;
# values are the canonical name expected to live in the registry. Misses
# fall through to fuzzy matching.
#
# Convention: prefer additions over fuzzy matching when the same misnomer
# shows up twice in eval traces. Aliases are O(1) lookups and don't risk
# silent collisions with newly-added tools.
_BUILTIN_ALIASES: dict[str, str] = {
    "read": "read_file",
    "readfile": "read_file",
    "read_files": "read_file",
    "cat": "read_file",
    "write": "write_file",
    "writefile": "write_file",
    "create_file": "write_file",
    "edit": "edit_file",
    "editfile": "edit_file",
    "update_file": "edit_file",
    "modify_file": "edit_file",
    "bash": "shell",
    "sh": "shell",
    "run": "shell",
    "exec": "shell",
    "execute": "shell",
    "run_command": "shell",
    "run_shell": "shell",
    "verify": "verify_work",
    "done": "verify_work",
    "finish": "verify_work",
    "web": "web_search",
    "search": "web_search",
    "google": "web_search",
}


@dataclass(frozen=True)
class CanonicalizationResult:
    """What L3 decided about a single tool name."""

    original: str
    canonical: str
    reason: str
    """Human-readable reason, recorded in the activity ledger."""
    confidence: float
    """0.0 to 1.0. 1.0 means alias-table hit; lower means fuzzy match."""

    @property
    def changed(self) -> bool:
        return self.original != self.canonical


def canonicalize_tool_name(
    name: str,
    known: Iterable[str],
    *,
    aliases: dict[str, str] | None = None,
    fuzzy_threshold: float = 0.78,
) -> CanonicalizationResult:
    """Return the registry-resolved tool name for an LLM-emitted name.

    Resolution order:

      1. Exact match against the registry → return unchanged.
      2. Alias-table lookup against ``aliases`` (defaults to ``_BUILTIN_ALIASES``)
         where the alias's target also exists in the registry.
      3. Closest-match against the registry via ``difflib.get_close_matches``
         with cutoff ``fuzzy_threshold``. Only accepted when the match score
         is high enough that we're not gambling on a wrong tool.
      4. Fall through unchanged. The runtime's existing "unknown tool"
         error path then fires.

    Args:
        name: the tool name the model wrote in its tool_calls payload.
        known: iterable of registered tool names. Order doesn't matter.
        aliases: optional override of the alias table. Useful in tests.
        fuzzy_threshold: cutoff for fuzzy matches. Default 0.78 is tuned
            to accept ``read`` → ``read_file`` (~0.83) and reject
            ``frobnicate`` → ``read_file`` (~0.4).
    """
    known_set = {k for k in known}
    if name in known_set:
        return CanonicalizationResult(name, name, "exact_match", 1.0)

    # Case-insensitive exact match — models often capitalize tool names
    # (e.g. "Read" / "Edit") in their tool_calls. Cheap to check, high
    # recall, no risk of false-positive resolution because the lower-
    # cased target either exists in the registry or it doesn't.
    lower = name.lower()
    case_hits = [k for k in known_set if k.lower() == lower]
    if case_hits:
        return CanonicalizationResult(name, case_hits[0], "case_insensitive_match", 0.98)

    table = aliases if aliases is not None else _BUILTIN_ALIASES
    target = table.get(lower)
    if target and target in known_set:
        return CanonicalizationResult(name, target, f"alias:{lower}->{target}", 0.95)

    # Fuzzy match — last resort. difflib's ratio is a reasonable proxy
    # for "this is the same name with a typo" without pulling in
    # heavier string-distance deps.
    matches = difflib.get_close_matches(
        lower, [k.lower() for k in known_set], n=1, cutoff=fuzzy_threshold
    )
    if matches:
        # Map the lower-cased match back to the original case in `known`.
        match_lower = matches[0]
        canonical = next(k for k in known_set if k.lower() == match_lower)
        ratio = difflib.SequenceMatcher(a=lower, b=match_lower).ratio()
        return CanonicalizationResult(name, canonical, f"fuzzy_match:ratio={ratio:.2f}", ratio)

    return CanonicalizationResult(name, name, "no_match", 0.0)


__all__ = ["CanonicalizationResult", "canonicalize_tool_name"]
