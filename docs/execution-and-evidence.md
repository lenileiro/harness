# Execution and evidence

## Run a planned mission

`mission run` executes features with the existing Agent runtime and then runs
the mission's assertion commands independently of the worker's response. Start
with an approved mission created through `mission create`, `mission plan`, and
`mission approve`. Inspect its assertion IDs and attach an executable command
to each assertion before running it:

```bash
uv run harness mission show-contract --mission "$MISSION_ID" --json
uv run harness mission set-assertion-command \
  --mission "$MISSION_ID" --assertion "$ASSERTION_ID" \
  --command '["python", "-m", "pytest", "tests/test_feature.py", "-q"]' \
  --timeout 60
uv run harness mission run --mission "$MISSION_ID" \
  --provider "$PROVIDER" --model "$MODEL" \
  --max-features 5 --max-worker-steps 20 --timeout 300
```

Commands are JSON argv arrays; a shell is used only when explicitly included
in the array. Use `--cwd` to select another workspace. Missing commands block
execution before a worker starts. Each assertion must cover features within one
milestone; split assertions that span milestones before using the real runner.
Failed checks and timeouts produce persisted
evidence and prevent milestone completion. Worker sessions and handoffs are
saved, so running the same mission again can resume interrupted features.
Inspect `mission list-runs`, `mission list-handoffs`, and `mission list-findings`
for the outcome and evidence locations.

The runner enforces feature, worker-step, and runtime limits. Mission token
budgets use provider usage when available and estimates otherwise; a response
already in progress can exceed its estimate, after which further requests are
blocked. The planner and legacy `mission launch` workflow remain separate from
this bounded feature runner.

`mission execute-burst --simulate` and `mission execute-milestone --simulate`
retain deterministic orchestration for tests. `--auto-complete` remains an alias.
Simulation is recorded on the mission and cannot subsequently be presented as
real execution by `mission run`.

## Schedule workers and retry notifications

```bash
uv run harness scheduler add-mission --mission "$MISSION_ID" \
  --every 30m --execute --provider "$PROVIDER" --model "$MODEL"
uv run harness scheduler start
uv run harness scheduler list-deliveries --json
uv run harness scheduler retry-deliveries
```

Without `--execute`, a scheduled mission uses the existing handoff mode (or
explicit simulation configuration). Scheduled workers put tool approval
requests in the inbox by default. Add `--yes` when automatic approval is intended.
Manual `mission run` retains interactive approval by default.

Schedulers claim each due occurrence under an OS lock and preserve pauses made
during execution. Failed notifications are retried from a separate persistent
outbox, with backoff, without rerunning the job. `retry-deliveries --force`
ignores the backoff deadline. Delivery is at least once: a crash after sending
but before saving the acknowledgement can produce a duplicate. A claimed job
interrupted by process death remains `running`; inspect its state and use
`scheduler run-now JOB_ID` for explicit recovery.

## Measure an experiment

Experiment plans accept checks, evaluation commands, and a measurement command
whose stdout is a nonempty JSON object of finite numeric values, for example
`{"latency_ms": 12.5, "accuracy": 0.94}`. Nonzero commands, empty plans, invalid
measurements, timeouts, and interruptions do not count as passed evidence.
Command output and partial interruption results are preserved. Rerun with
`research experiment run --plan PLAN_ID` after addressing a failure or interruption.
Uncatchable process termination can leave a `running` record; inspect its logs
and explicitly rerun the plan rather than treating that record as success.

```bash
uv run harness research plan-experiment --hypothesis "$HYPOTHESIS_ID" \
  --plan "Measure the proposed change" --target-files src/component.py \
  --checks 'python -m pytest tests/test_component.py -q' \
  --measurement-command 'python benchmarks/component.py --json' \
  --minimize latency_ms --maximize accuracy --baseline "$BASELINE_EXPERIMENT_ID"
uv run harness research experiment run --plan "$PLAN_ID" --timeout 120
uv run harness research experiment show "$EXPERIMENT_ID"
uv run harness research experiment compare "$BASELINE_EXPERIMENT_ID" "$EXPERIMENT_ID"
```

First run a baseline plan without `--baseline` against the baseline workspace.
Then run a candidate plan with that experiment ID after making the change.
`show` includes measurements and the workspace fingerprint; `compare` includes
deltas and the declared improvement direction. Checks-only plans remain useful
for correctness changes that have no numeric goal.

## Promote with current evidence

Link experiments when creating a candidate with repeated `--source-experiment`
options. Candidates linked only through hypotheses use the latest unambiguous
execution of each relevant plan. Draft and branch preparation are available
without passed execution evidence. Commit, push, and PR creation require:

- Nonempty passed commands matching the current experiment plan.
- Explicit relative target paths covered by the selected plans.
- A fingerprint matching the current workspace contents and file modes.
- For declared metric goals, a passed measured baseline, no goal regressions,
  and at least one improvement.

The gate runs again before each mutation, including after commit hooks. PR
preparation commits declared source paths; generated `.harness` drafts stay
local. Autonomous research records completed promotion stages, avoids duplicate
child artifacts, and lets other eligible work advance when a candidate is
blocked.

Fingerprints cover Git-tracked and nonignored files (or ordinary files when Git
is unavailable), excluding generated `.harness`, virtualenv, and cache state.
Checks that modify fingerprinted inputs cannot certify them in the same run.
Evidence does not attest to external services, ignored dependencies, or
deliberately detached processes; use controlled inputs for reproducible
measurements.

## Benchmark process isolation

Coding fixture agents run in an OS filesystem sandbox with a public workspace
and staged Harness runtime. Fixture grading commands and evaluator paths remain
with the parent evaluator and are used after the agent finishes. Workspace
copies preserve symlinks so they cannot copy private target contents into the
public workspace. Agent processes receive a temporary home and no evaluator
experience roots; host home-based CLI login state is not inherited.

macOS requires `sandbox-exec`. Linux requires `bubblewrap` and permission to
create its namespaces; the runner fails closed if isolation cannot start. CI
installs Bubblewrap and the default test run now includes `evals/tests`.
Local integration tests passed on macOS. Linux/arm64 smoke attempts used an
ephemeral generic Docker container: default security denied namespace creation,
and container-only `seccomp=unconfined` (including a separate `SYS_ADMIN` probe)
still denied `/proc` mounting. These launches failed closed; a successful Linux
sandbox launch remains unverified. Ordinary child process groups are cleaned up
on completion and timeout. The sandbox is a filesystem boundary, not a guarantee
of network isolation or of terminating deliberately detached macOS processes.
