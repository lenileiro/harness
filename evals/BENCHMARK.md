# Benchmark rules

This benchmark is intended to measure coding-agent behavior, not only patch
correctness.

## Canonical run settings

- Run from the repository root.
- Prefer `harness eval run --ab --n-runs 3 --suite smoke` for quick local checks.
- Use `--benchmark-mode mutated` or `--benchmark-mode mixed` to test contamination resistance.
- Use `--fixture-set fixtures-holdout --suite holdout --include-holdout` for milestone checks against the holdout set.
- Use `--n-runs 3` or higher for comparisons because model variance is material.
- Report defended vs bare separately.
- Keep provider/model/judge provider/judge model explicit in any published table.

## Required artifacts

Each run writes a durable artifact bundle under `evals/runs/<run-id>/`:

- `transcript.txt`
- `git_diff.patch`
- `verify_output.txt`
- `trace.jsonl`
- `outcome.json`
- `report.json`

## Scoring policy

- Hard metrics come from execution evidence and diff analysis.
- Judge metrics score behavioral qualities such as scope, decomposition, pushback,
  and epistemic grounding.
- A trial passes when `overall >= 3` and `correctness >= 3`.
- Judge calibration should be measured periodically against `evals/gold/`.

## Fixture sets

- `fixtures`: main public set
- `fixtures-mutated`: materialized contamination-resistance variants
- `fixtures-holdout`: holdout set for milestone checks

Current checked-in corpus:

- `fixtures`: 14 canonical fixtures
- `fixtures-mutated`: 12 deterministic mutated variants (3 seeds × 4 fixture families)
- `fixtures-holdout`: 6 holdout variants

That gives 32 runnable fixtures across public, mutated, and holdout sets.

## CI guidance

- Use the `smoke` suite for fast PR checks.
- Keep full benchmark runs out of normal CI unless you have dedicated budget.
