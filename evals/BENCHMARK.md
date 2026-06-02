# Benchmark rules

This benchmark is intended to measure coding-agent behavior, not only patch
correctness.

## Canonical run settings

- Run from the repository root.
- Prefer `harness eval run --ab --n-runs 3 --suite smoke` for quick local checks.
- Use `--benchmark-mode mutated` or `--benchmark-mode mixed` to test contamination resistance.
- Use `--fixture-set fixtures-holdout --suite holdout --include-holdout` for milestone checks against the holdout set.
- Include at least one external DeepSWE task in goal-level validation. This is
  the long-form real-world gate: the agent must plan, mutate a real repository,
  run public repository checks through `verify_work`, refuse to finish on failed
  evidence, and then be graded by the hidden verifier after it stops.
- Treat DeepSWE `oracle` or `solution/solve.*` runs as verifier calibration only,
  not benchmark scores. Real LLM trials must run without `solution/`, hidden
  `tests/`, or unrestricted web access to the source repository.
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
- Keep DeepSWE out of normal CI. Run it manually or in a dedicated nightly job
  with explicit provider credentials and Docker capacity.

Current DeepSWE-style external run results live in
[evals/DEEPSWE_RESULTS.md](DEEPSWE_RESULTS.md).

## External DeepSWE goal test

Use a DeepSWE checkout as the external benchmark source and a clean target clone
of the task repository at the task's base commit. The task's hidden verifier is
evaluator-only: run it after the agent finishes to grade the captured patch, not
as the agent's `verify_work` command or as a tool-visible script.

Required evidence for a valid run:

- the target repository starts clean at the task base commit
- the agent workspace does not contain DeepSWE `solution/` or hidden `tests/`
  files before the verifier phase
- read-only public web access is allowed for autonomous research, tool setup,
  documentation, and error investigation
- task-scoped policy blocks hidden benchmark artifacts, solution-bearing pages,
  source-repository lookup, and source-repository web search during the agent
  phase
- the shell prints a PID or live session id while the run is active
- the transcript includes concrete tool use, predictions, and at least one
  `verify_work` call for any state-changing attempt
- failed verifier output blocks completion and is fed back into repair
- the target repository is reset after capturing the log, diff, and verifier
  output

Example shape, with secrets supplied by the caller's environment:

```bash
uv run python evals/deepswe_runner.py /path/to/deep-swe/tasks/<task> \
  --model openai/gpt-5.4-nano \
  --max-steps 45 \
  --source-change-retries 1 \
  --verification-retries 1
```
