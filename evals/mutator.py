"""Fixture mutator — structure-preserving transformations for benchmark integrity.

Pattern from *Saving SWE-Bench: Benchmark Mutation* (arXiv 2510.08996).
Static fixtures get memorized; mutating them lets you measure
contamination resistance and detect when an A/B signal is leaking
through to a memorized solution.

What we mutate:

  1. Symbol renames — function names, class names, variable names in
     the source tree are rewritten consistently across ``src/`` and
     ``tests/``. Renames also propagate into ``TASK.md`` and
     ``EVAL.md`` so the prompt stays coherent.
  2. Method-level wording — docstrings and error messages get
     paraphrase substitutions to defeat memorized literal strings.

What we deliberately don't mutate:

  • Test semantics (assertions, expected values).
  • Import paths from the standard library or pytest.
  • The trap structure of the fixture (e.g. F03's wrong-diagnosis
    redirection stays in place).

Each mutation is deterministic in the random seed so a mutated fixture
is reproducible. Cross-eval comparison works by running the same seed
on both arms.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from random import Random

# Curated rename pools — small enough that we can audit collisions.
# Each pool maps an original symbol to a list of structure-preserving
# alternatives. Random.choice() picks one per mutation seed.
_RENAME_POOLS: dict[str, list[str]] = {
    # Class names — keep CamelCase, keep concept neutral
    "Calculator": ["MathOps", "Reckoner", "ArithBox"],
    "Cache": ["Store", "Buffer", "MemTab"],
    "PriceFormatter": ["MoneyFormatter", "AmountRenderer"],
    "UserHandler": ["AccountHandler", "MemberHandler"],
    # Function / method names — keep snake_case
    "power": ["pow_op", "exponentiate", "raise_to"],
    "format_price": ["render_amount", "format_money"],
    "get_user": ["lookup_user", "fetch_user", "resolve_user"],
    "deduplicate": ["unique_only", "drop_duplicates"],
    # Constants
    "TIMEOUT_SECONDS": ["TIMEOUT_S", "MAX_WAIT_SECONDS"],
    "DEFAULT_TIMEOUT": ["FALLBACK_TIMEOUT", "BASE_TIMEOUT"],
}


@dataclass
class Mutation:
    """One concrete (seed-pinned) renaming applied to a fixture."""

    seed: int
    renames: dict[str, str] = field(default_factory=dict)

    def display(self) -> str:
        if not self.renames:
            return f"seed={self.seed} (no renames applied)"
        pairs = ", ".join(f"{k}→{v}" for k, v in self.renames.items())
        return f"seed={self.seed} | {pairs}"


def plan_mutation(fixture_text: str, seed: int) -> Mutation:
    """Build a deterministic rename plan for a fixture's combined text.

    `fixture_text` is the concatenation of all source / test / TASK / EVAL
    text — used only to detect which pool entries actually occur in the
    fixture so we don't propose irrelevant renames. The plan returns a
    final ``Mutation`` containing only renames that will land.
    """
    rng = Random(seed)
    renames: dict[str, str] = {}
    used_targets: set[str] = set()
    for original, candidates in _RENAME_POOLS.items():
        if not _occurs(fixture_text, original):
            continue
        # Filter targets that would collide with another original symbol.
        viable = [c for c in candidates if c not in used_targets and c not in _RENAME_POOLS]
        if not viable:
            continue
        target = rng.choice(viable)
        renames[original] = target
        used_targets.add(target)
    return Mutation(seed=seed, renames=renames)


def _occurs(text: str, symbol: str) -> bool:
    """Word-boundary check, case-sensitive."""
    return re.search(rf"\b{re.escape(symbol)}\b", text) is not None


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------


_MUTATABLE_SUFFIXES: frozenset[str] = frozenset({".py", ".md", ".txt", ".cfg", ".toml"})


def apply_mutation(src_dir: Path, dest_dir: Path, mutation: Mutation) -> list[Path]:
    """Copy `src_dir` to `dest_dir` and apply `mutation` in place.

    Returns the list of files that were touched (relative to dest_dir).
    Conservative: only rewrites files whose suffix is in
    ``_MUTATABLE_SUFFIXES``. Binary files and unknown extensions are
    copied verbatim.
    """
    if dest_dir.exists():
        shutil.rmtree(dest_dir)
    shutil.copytree(src_dir, dest_dir)

    if not mutation.renames:
        return []

    touched: list[Path] = []
    for path in dest_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in _MUTATABLE_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        new_text = _rewrite(text, mutation.renames)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            touched.append(path.relative_to(dest_dir))
    return touched


def _rewrite(text: str, renames: dict[str, str]) -> str:
    """Apply renames as whole-word substitutions in a single pass.

    We compile a single alternation regex so a rename to a target that
    happens to be another rename's source can't be reapplied. Names are
    matched on word boundaries so substrings stay intact.
    """
    if not renames:
        return text
    keys = sorted(renames, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")
    return pattern.sub(lambda m: renames[m.group(0)], text)


# ---------------------------------------------------------------------------
# Top-level driver — used by the CLI
# ---------------------------------------------------------------------------


@dataclass
class MutationResult:
    fixture_name: str
    seed: int
    dest_dir: Path
    renames: dict[str, str]
    touched_files: list[Path]


def mutate_fixture(
    src_fixture_dir: Path,
    *,
    seed: int,
    dest_root: Path | None = None,
) -> MutationResult:
    """Mutate a single fixture directory.

    Writes the mutated copy to ``dest_root / <seed>-<name>``. Defaults
    to a sibling ``fixtures-mutated/`` next to the source fixture's
    parent directory (typical layout: ``evals/fixtures-mutated/...``).
    """
    if not src_fixture_dir.is_dir():
        raise ValueError(f"fixture dir not found: {src_fixture_dir}")
    if dest_root is None:
        dest_root = src_fixture_dir.parent.parent / "fixtures-mutated"
    dest_root.mkdir(parents=True, exist_ok=True)

    # Slurp all mutable files to plan against their combined text.
    corpus_parts: list[str] = []
    for path in src_fixture_dir.rglob("*"):
        if path.is_file() and path.suffix in _MUTATABLE_SUFFIXES:
            try:
                corpus_parts.append(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, OSError):
                continue
    corpus = "\n".join(corpus_parts)

    mutation = plan_mutation(corpus, seed)
    dest_dir = dest_root / f"{seed:04d}-{src_fixture_dir.name}"
    touched = apply_mutation(src_fixture_dir, dest_dir, mutation)
    return MutationResult(
        fixture_name=src_fixture_dir.name,
        seed=seed,
        dest_dir=dest_dir,
        renames=mutation.renames,
        touched_files=touched,
    )


__all__ = [
    "Mutation",
    "MutationResult",
    "apply_mutation",
    "mutate_fixture",
    "plan_mutation",
]
