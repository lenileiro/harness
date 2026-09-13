# Harness implementation audit — 2026-09-13

Audited `main` at `099a8e2` using three independent exploration agents plus a coordinating review. The areas were runtime/orchestration, CLI/mission/scheduler workflows, and adapters/storage/evaluation integrity. The coordinator independently reran all three agents' offline reproduction scripts and investigated research progression and promotion evidence.

The main opportunity is to make execution, evidence, and persisted outcomes agree. Harness already exposes many useful commands, but several paths can report success without completed work, lose recovery state, or fail under concurrent use. More command families would provide less immediate value than finishing these foundations.

This records the exploration baseline before implementation. The subsequent implementation addresses the findings below; usage and remaining execution limits are documented in [Execution and evidence](execution-and-evidence.md). Audit reproductions used fake adapters, mock HTTP transports, synthetic evaluator sentinels, and temporary workspaces. No live model calls, external notifications, pushes, or deployments were performed.

**Implementation validation**

The implementation was verified locally on Python 3.12/macOS: the full default
suite passed 2,220 tests. After the final workspace-copy boundary fix, all 158
evaluator tests passed, including three newly added boundary cases. Ruff lint,
formatting, Pyright, `git diff --check`, and eval asset validation passed. Tests
use synthetic providers and local fixtures; live provider behavior was not
exercised. Linux Docker sandbox attempts failed closed under namespace/proc
mount restrictions, so successful Linux startup remains unverified. The usage
guide records recovery, evidence scope, and delivery limitations.

**Validation baseline**

| Check | Result |
| --- | --- |
| `uv sync --locked --python 3.12 --all-extras --all-groups` | Passed |
| `uv run --no-sync ruff check .` | Passed |
| `uv run --no-sync ruff format --check .` | Passed; 501 files |
| `uv run --no-sync pyright` | Passed; zero errors/warnings |
| `uv run --no-sync harness --help` | Passed |
| `uv run --no-sync harness eval validate` | Passed |
| `uv run --no-sync pytest -q` | 1,921 passed; 1 failed; 137 seconds |
| `uv run --no-sync pytest -q evals/tests` | 144 passed; 15 seconds |

The package test failure is `test_remote_shell_tool_guides_missing_toolchain_setup` at `packages/cli/tests/test_external_workspace.py:2214`. It runs real `go test ./...` and assumes Go is absent. This machine has `/opt/homebrew/bin/go`, so the output reports a missing Go module instead. This is an environment-dependent test, not evidence that missing-tool guidance itself is broken.

**Fixes to prioritize**

P1 means address before relying on the affected boundary or unattended workflow. P2 means a concrete correctness or completeness issue worth scheduling after the highest-risk failures. Unless specifically qualified, the behavior below was reproduced offline; a reproduction establishes the code path, not its frequency in production.

1. **P1 — Reused agents retain another session's memory-tool binding.** `runtime.py:545` registers `NotesTool` and `PruneLedgerTool` only once, retaining the first `Session` object. Resuming that session and adding a second note leaves only the first note in storage; a new session can list both notes from the old object. Rebind runtime-owned tools to the current session on each run, preserving custom tool ownership and handling concurrent reuse explicitly. Acceptance: notes and pruning persist across resume, and separate sessions cannot read or alter one another's scratchpads. Sources: `packages/core/src/harness/core/runtime.py:545`, `packages/core/src/harness/core/tools_memory.py:104`.

2. **P1 — Worker exceptions disappear while the parent job becomes done.** A fake worker raising `RuntimeError` leaves its child task `in_progress`, emits no failure event, and ends with the root `done`. `gather(return_exceptions=True)` discards worker failures, while root completion is unconditional. Record worker/factory/judge failures, settle or requeue claimed work, and derive the parent outcome from child outcomes. Acceptance: exceptions remain observable and unfinished children prevent a successful root. Sources: `packages/core/src/harness/core/orchestrator.py:556`, `packages/core/src/harness/core/orchestrator.py:419`.

3. **P1 — Closing a worker event stream does not stop its workers.** After consuming a worker-start event and closing `_run_workers()`, releasing a blocked fake worker still lets it complete its task. Add cancellation and awaiting of all owned tasks in `finally`, including closure of nested generators. Acceptance: early close, consumer failure, and cancellation return only after workers stop; interrupted claims retain a recoverable state. Source: `packages/core/src/harness/core/orchestrator.py:553`.

4. **P1 — SQLite task transactions conflict on a shared connection.** Five simultaneous creates yield one task and four `cannot start a transaction within a transaction` errors. Concurrent claims also fail. The initialization lock does not protect the full `BEGIN IMMEDIATE` transaction. Serialize transactions and prevent unrelated writes/commits from interleaving, or give transactions separate connections. Acceptance: concurrent creates succeed with unique references, and concurrent claims assign each task once, including interleaving with session/activity writes. Sources: `packages/storage-sqlite/src/harness/storage/sqlite/__init__.py:255`, `packages/storage-sqlite/src/harness/storage/sqlite/__init__.py:380`.

5. **P1 — Repository-backed benchmark workspaces expose evaluator assets.** The copy exclusion list omits `evals/gold`, and the agent environment exposes the original checkout through `HARNESS_EVAL_PROJECT_ROOT`. A synthetic repository proved a dummy gold sentinel was copied while fixture directories were excluded. No actual hidden answers were read, and this does not establish contamination of any historical score. Use an allowlist for agent-visible files, remove evaluator source paths, and enforce filesystem/process isolation for shell-enabled benchmarks. Acceptance: synthetic evaluator sentinels are inaccessible through every offered tool while public task files remain usable. Sources: `evals/runner.py:242`, `evals/runner.py:79`.

6. **P1 — URL redirects bypass the tool's private-address restriction.** The initial host is checked, then HTTP redirects are followed automatically without checking destinations. A mocked public URL redirected to `127.0.0.1`, and the private sentinel body was returned successfully. Validate every redirect destination before requesting it; bound redirect chains and preserve scheme/address checks. If promising DNS-rebinding protection, bind the request to the validated resolution. Acceptance: prohibited redirect destinations never reach the transport, while permitted redirects still work. Source: `packages/tools-web/src/harness/tools/web/__init__.py:195`.

7. **P1 — Truncated or error model streams become successful turns.** The shared Ollama/OpenRouter SSE parser emits `Done` at EOF, ignores error envelopes, and converts malformed tool arguments into `{}`. Synthetic truncated text, provider-error envelopes, and partial JSON arguments all reproduced this. Track protocol completion and error states; reject incomplete calls rather than substituting arguments. Acceptance: incomplete/error responses cannot report success or dispatch malformed calls, while valid parallel tool streams retain their behavior. Sources: `packages/core/src/harness/core/_openai.py:89`, `packages/core/src/harness/core/_openai.py:116`.

8. **P1 — Scheduler locks do not identify an occurrence that has already run.** Two scheduler threads can snapshot the same due reminder. After the first completes and releases its lock, the second dispatches the now-completed job; the probe produced two successful records. Recheck status/due time under the lock and atomically claim an occurrence. Keep explicit manual replay distinct. Acceptance: competing watchers execute each scheduled occurrence once, even when the second acquires the lock after completion. Sources: `packages/core/src/harness/core/scheduler_runtime.py:236`, `packages/core/src/harness/core/scheduler_runtime.py:348`.

9. **P1 — Scheduler completion overwrites a concurrent pause.** Pausing a recurring job during dispatch changes its stored state to `paused`; completion replaces it with the stale pre-dispatch `active` object. Synchronize state updates or merge completion into the latest control state using a version check. Acceptance: pause survives both successful and failed runs while execution history is recorded. Sources: `packages/core/src/harness/core/scheduler_runtime.py:318`, `packages/core/src/harness/core/scheduler_store.py:140`.

10. **P1 — Reminder delivery can fail after being permanently marked complete.** One-shot completion is persisted before the completion hook sends the reminder. A throwing delivery hook leaves a completed job/run with no next run, and its exception escapes. Store delivery state independently and retry it durably; contain notification errors so other jobs continue. Acceptance: a bridge outage leaves delivery pending, restart retries it, and successful delivery has a separate acknowledgement. Sources: `packages/core/src/harness/core/scheduler_runtime.py:316`, `packages/cli/src/harness/cli/gateway_hooks.py:69`.

11. **P1 — High-priority research work repeatedly creates hypotheses without advancing.** Opportunities remain eligible after generating a hypothesis. A high-priority opportunity scores 60; its medium-risk hypothesis scores 50. Four burst steps reproduced four hypotheses for the same opportunity and zero plans. Add explicit lifecycle state and links to the next stage; make transitions idempotent and select actionable work. Inspect unknowns and successfully prepared promotions for the same repeated-selection pattern. Acceptance: repeated bursts advance the same item rather than duplicating children, and blocked high-priority work does not starve eligible work. Sources: `packages/core/src/harness/core/research_scheduler.py:56`, `packages/core/src/harness/core/autonomy.py:229`.

12. **P1 — Dependencies across mission milestones never become ready.** Planning accepts dependencies across a whole mission, but execution resolves them only against the current milestone's features. A validated feature in milestone 1 still blocks its dependent in milestone 2. Resolve dependencies across all mission features while limiting dispatch candidates to the selected milestone. Acceptance: completed prerequisites in prior milestones unlock their dependents; missing prerequisites and cycles fail clearly. Sources: `packages/core/src/harness/core/mission_planner.py:293`, `packages/core/src/harness/core/mission_runtime.py:226`.

13. **P2 — A throwing start hook leaks the scheduler lock.** `on_job_started` runs after lock acquisition but before its cleanup `try/finally`. The probe left a lock and no run record; retry was skipped as already running. Move every post-acquisition operation under cleanup and record hook failures. Acceptance: any hook exception releases the lock and leaves an observable outcome. Source: `packages/core/src/harness/core/scheduler_runtime.py:258`.

14. **P2 — Selecting a future milestone dispatches a different milestone.** Requesting milestone 2 while milestone 1 is pending returns `completed`, `steps_run=0`, but silently changes milestone 1's feature to `handoff`. The generic dispatcher runs before the requested milestone is checked. Validate selection before mutation or pass the target into dispatch. Acceptance: selecting a future/completed milestone never mutates unrelated work or falsely reports the selected work complete. Source: `packages/core/src/harness/core/mission_runtime.py:407`.

15. **P2 — Public run outcomes disagree with runtime failures.** An always-rejecting verifier with no repairs left emits `can_finish=False` but persists session `done`. Separately, `agent.iter()` drops error/verification events: a network failure yields only a model-request step and exits normally while storage says failed. Persist a typed terminal outcome and expose it consistently through both APIs. The CLI's one-shot path separately inspects verification, so the first observation alone is not proof of a CLI exit-code bug. Acceptance: failure, cancellation, exhausted verification, and successful repair have consistent events and stored outcomes. Sources: `packages/core/src/harness/core/runtime.py:824`, `packages/core/src/harness/core/runtime.py:924`, `packages/core/src/harness/core/agent_iter.py:100`.

16. **P2 — Flow checkpoints lose routing and join progress.** Checkpoints save state plus one step name, omitting completed steps, pending work, and router decisions. A routed flow resumes as `[choose]` instead of `[choose, finish]`; a joined flow resumes as `[a, b]` instead of `[a, b, merge]`. Version checkpoints to include the full execution state, captured after successor scheduling. Acceptance: routers, multiple starts, fan-out, and joins resume to equivalent final state without repeating completed effects. Sources: `packages/core/src/harness/core/flow.py:268`, `packages/core/src/harness/core/flow.py:315`.

17. **P2 — Legacy SQLite initialization fails before its migration.** Schema setup indexes `sessions(task_id)` before the migration adds that column to older tables. A synthetic legacy database fails with `no such column: task_id`. Migrate columns before dependent indexes and publish the connection only after successful initialization. Acceptance: older sessions survive migration and reopening is idempotent. Sources: `packages/storage-sqlite/src/harness/storage/sqlite/__init__.py:62`, `packages/storage-sqlite/src/harness/storage/sqlite/__init__.py:162`.

18. **P2 — Experiment results are incomplete as execution evidence.** An empty plan reports `passed` with zero commands. A timeout raises before experiment/result JSON is persisted. A failed command produces a failed result but `research experiment run` exits 0. Reject empty executable plans or mark them inconclusive; persist timeout/failure and partial logs; return a nonzero CLI status for failed execution. Acceptance: empty, failing, and timed-out plans cannot appear successful to automation, and attempted runs remain inspectable. Sources: `packages/core/src/harness/core/experiment_runner.py:67`, `packages/core/src/harness/core/experiment_runner.py:71`, `packages/cli/src/harness/cli/research_commands.py:1969`.

19. **P2 — Ollama/OpenRouter usage is discarded even when supplied.** A valid synthetic response reporting 100 prompt and 20 completion tokens ends with `usage=None`. The runtime already records usage, and dynamic workflows already have token limits; the missing piece is reliable adapter input to those features. Normalize reported token/cache counts and request usage where supported. Acceptance: supplied usage reaches `Done`, the activity ledger, and workflow totals; unavailable usage remains explicitly unknown. Sources: `packages/core/src/harness/core/_openai.py:104`, `packages/core/src/harness/core/runtime.py:1434`, `packages/cli/src/harness/cli/workflow_commands.py:2161`.

20. **P2 — Default CI excludes evaluator tests and a toolchain test depends on the host.** `testpaths=["packages"]` means CI's plain pytest command omits 15 evaluator test modules (144 collected cases when run separately). Asset validation does not replace these behavior tests. Include `evals/tests` in CI/default collection and make the missing-Go test control its environment or fake command results. Acceptance: a failing evaluator test fails CI, and the toolchain-guidance test behaves identically with Go installed or absent. Sources: `pyproject.toml:98`, `.github/workflows/ci.yml:50`, `packages/cli/tests/test_external_workspace.py:2214`.

**Feature opportunities grounded in these gaps**

| Priority | Feature | Smallest useful implementation and acceptance |
| --- | --- | --- |
| First | Real mission workers and independent assertion execution | Connect mission handoffs to an executor using the existing `Agent` runtime, then run declared verification before marking features validated. Preserve simulation as an explicit test mode. A requested artifact must exist and pass its check before a mission can complete. |
| First | Measured experiment comparisons and evidence-gated promotion | Attach baseline/candidate measurements, command outcomes, artifact references, and a workspace revision/fingerprint to experiments. Require current evidence and bounded scope before mutation-capable promotion. Missing or stale evidence must block promotion; failed checks must remain visible. |
| Next | Durable run outcomes and recovery across APIs | Expose one terminal result for sessions, workers, flows, and missions, with reason, verification result, and recoverable state. Build complete checkpoints and cancellation ownership into it. After interruption, a resumed run should continue pending work and agree with its stored status. |
| Next | Shared adapter conformance suite and usage parity | Run reusable synthetic stream contracts against adapters: normal termination, error envelopes, truncation, parallel tools, cancellation, and usage. Feed accurate accounting into existing workflow budgets before extending budget controls across mission roles and child agents. |
| Next | Durable outbound delivery queue | Persist reminders/notifications as pending deliveries with attempts, backoff, and acknowledgements. Keep job execution and delivery results distinct. Restart after an outage must recover unsent items; use transport idempotency where available and document remaining duplicate-delivery ambiguity. |

The mission feature is a substantial missing capability, not an accidental regression: `mission_runtime.py:264` records a worker run/handoff without invoking a worker, `:439` supplies synthetic completion text under `--auto-complete`, and `mission_validator.py:154` checks feature statuses rather than executing assertions. An offline two-milestone run reached `completed` with both features `validated` and no target files created. Existing tests deliberately cover deterministic orchestration; the change should make execution mode explicit rather than silently reinterpret those tests.

Promotion has a similar distinction between preparing artifacts and proving a change. `_review_promotion_artifacts` checks populated metadata and PR sections. A candidate with no source publications or hypotheses passed that review in a direct probe. `compare_experiment_results` currently compares status, duration, and command counts. Those are useful scaffolds, but do not establish that a change improved the expected metric. Sources: `packages/core/src/harness/core/autonomy.py:100`, `packages/core/src/harness/core/experiment_runner.py:118`.

**Suggested implementation order**

1. Restore trust in boundaries and outcomes: session-memory binding, evaluator isolation, redirect validation, stream correctness, worker error propagation/cancellation, and SQLite transaction isolation.
2. Make unattended work durable: scheduler occurrence claims and control-state updates, hook cleanup, delivery retries, research-stage progression, and mission dependency resolution.
3. Finish the execution product: real mission workers and assertion runners, complete recovery checkpoints, and measured promotion gates.
4. Expand continuous verification: evaluator tests in CI, hermetic environment tests, shared adapter contracts, and usage parity. Individual regression tests should accompany each preceding fix rather than wait for this final group.

**Reproduction artifacts**

The following temporary artifacts remain available on this machine. They contain synthetic inputs and observed outputs, not benchmark answers. Their assertions describe current defects, so some will fail after fixes are implemented.

- `/tmp/harness-runtime-audit/repro.py` and `results.json`: memory binding, worker crash/closure, checkpoint routing/joins, verification status, iterator errors.
- `/tmp/harness-workflow-audit/reproduce.py` and `output.jsonl`: mission simulation/dependencies/selection, scheduler duplicate occurrence/pause/hooks/delivery.
- `/tmp/harness-eval-adapter-audit/repro.py` and `results.json`: legacy/concurrent SQLite, synthetic evaluator isolation, mocked redirect restrictions, stream errors/usage, CI collection.
- `/tmp/harness-audit-pytest.log`, `/tmp/harness-audit-eval-tests.log`, `/tmp/harness-audit-ruff.log`, `/tmp/harness-audit-format.log`, `/tmp/harness-audit-pyright.log`: baseline check output.

Run reproduction scripts from the repository with `.venv/bin/python <script>`. The runtime script additionally accepts an output JSON path. Root research/experiment probes used temporary `ResearchStore` instances: four burst steps for a high-priority opportunity; an empty plan; a `sleep 0.2` check with a 0.01-second timeout; and a CLI plan containing `exit 1`.
