# DeepSWE external eval results

Snapshot date: 2026-06-03

This page tracks Harness runs against local
[DeepSWE](https://deepswe.datacurve.ai/blog)-style tasks: short behavioral
prompts, real repositories, hidden handwritten verifiers, and evaluator-only
reference solutions.

This is not a public leaderboard yet. It is a single-model Harness stress test
focused on whether the agent can explore, implement, verify, and refuse false
completion without seeing the hidden verifier or reference solution.

## Bottom line

Current representative snapshot:

| Measure | Result |
|---|---:|
| Model | `openai/gpt-5.4-nano` through OpenRouter |
| Representative tasks | 7 |
| Repositories | 7 |
| Languages | Go, Python, TypeScript, Rust |
| Model hidden-verifier rewards | 0/7 |
| Harness false success reports | 0/7 |
| Leak scans for hidden tests / solution / DeepSWE repo lookups | 7/7 clean |
| Saved reference calibrations | 1 pass, 1 fail, 5 not rerun in this snapshot |

The model did not solve any representative task to hidden-verifier reward `1`.
That is not a Harness pass-rate claim; it is a model outcome under this
experimental Harness configuration. The Harness result is better: it did not
report success when isolated grading, latest verification evidence, or
autonomous setup discipline did not support success.

## Methodology

Each DeepSWE task supplies:

- `task.toml`: repository URL, pinned base commit, language, container image,
  and resource limits.
- `instruction.md`: the only task prompt the agent sees.
- `environment/`: fallback build context if the prebuilt image is unavailable.
- `tests/`: hidden verifier, evaluator-only.
- `solution/`: reference solution, evaluator-only.

Harness run policy:

- The agent works in a clean external workspace, not in the Harness repo.
- The agent may use shell, repository inspection, public web research, and
  normal setup tools.
- Hidden `tests/`, `solution/`, and DeepSWE source artifacts are not mounted
  into the agent workspace.
- The agent must discover and prepare the repository environment itself from
  public task metadata and repository-visible evidence.
- `verify_work` must run real in-repository checks after source changes.
- Hidden verifier and reference solution are only used after the agent stops,
  by the evaluator.
- Completion requires a captured patch and isolated hidden-verifier reward `1`.

The public DeepSWE writeup emphasizes contamination-free tasks, broad repository
coverage, short behavior-focused prompts, and behavioral verifiers. This local
Harness eval borrows that shape but reports only local Harness evidence.

## Current snapshot

| Task | Language | Category | Run root | Model reward | Harness outcome | Leak scan | Reference calibration | Main signal |
|---|---|---|---|---:|---|---|---|---|
| `anko-default-function-arguments` | Go | Feature | `evals/results/deepswe/anko-default-function-arguments-1f08cee2` | 0 | Failed | Clean | Not rerun | Latest run reached passing public no-network verification, then removed behavior-specific tests; Harness semantic coverage rejected completion and now exports that verifier reason in run metadata. |
| `mashumaro-flattened-dataclass-fields` | Python | Feature | `evals/results/deepswe/mashumaro-flattened-dataclass-fields-461c6ca7` | 0 | Failed | Clean | Not rerun | Latest run produced only helper signature changes plus failing flatten tests; Harness rejected failed/stale verification and the final setup/permission handoff. |
| `aiomonitor-task-snapshots-diff` | Python | Feature | `evals/results/deepswe/aiomonitor-task-snapshots-diff-d19773de` | 0 | Failed | Clean | Not rerun | Agent stopped around host dependency/setup failure instead of fully preparing the environment. |
| `arktype-json-schema-refs-dependencies` | TypeScript | Feature | `evals/results/deepswe/arktype-json-schema-refs-dependencies-0804cefd` | 0 | Failed | Clean | Not rerun | Source changes were not followed by passing in-repository verification. |
| `happy-dom-abort-pending-body-reads` | TypeScript | Bugfix | `evals/results/deepswe/happy-dom-abort-pending-body-reads-62a432f7` | 0 | Failed | Clean | Not rerun | Agent asked to continue rather than autonomously finishing from available tools. |
| `prometheus-typed-label-sorting` | Go | Bugfix | `evals/results/deepswe/prometheus-typed-label-sorting-21c8e783` | 0 | Failed | Clean | Pass (`reward=1`) | Agent produced incomplete typed ordering; Harness rejected completion. |
| `fd-deterministic-multi-key-sorting` | Rust | Feature | `evals/results/deepswe/fd-deterministic-multi-key-sorting-88caa394` | 0 | Failed | Clean | Failed locally (`reward=0`) | Agent passed `cargo test`, but added a new dependency that no-network isolated grading could not fetch. |

## Harness findings

The current Harness behavior is moving in the right direction:

- It blocks final success when the latest verification is stale, failed, or
  contradicted by later edits.
- It blocks final success when the agent asks the user to provide setup,
  tooling, source, or approval that the Harness can pursue autonomously.
- It rejects weak verification evidence such as hidden failure behind shell
  fallbacks.
- It captures `model.patch` / `final.diff` for independent grading.
- It prints the live shell session / PID while long external runs execute.
- It keeps hidden verifier and reference solution out of the agent workspace.
- It can force no-network verification in the declared task image before any
  hidden verifier is allowed to run.

Issues exposed and fixed during this DeepSWE phase:

- `5997e422` - generalized Harness workflows and verification.
- `b438972d` - fixed external workspace setup capture paths.
- `ca85d858` - covered nested Docker verifier evidence.
- `d97d600c` - avoided false failure matches in verification output.
- `56eb2e54` - rejected git inspection as verify work.
- `998edddf` - required public offline DeepSWE verification before hidden
  grading.
- `b0286b4b` - exported final completion-verifier rejection metadata so
  semantic coverage failures are visible in `outcome.json` instead of appearing
  only as generic exit-code failures.
- Current change - clarified no-network verifier repair guidance so the final
  `verify_work` command itself runs the declared Docker image instead of a
  misleading shell-Docker-plus-local-verify sequence.

## Failure patterns

The failures are useful because they separate model behavior from Harness
defense behavior:

- **Environment autonomy is still a hard test.** Several runs failed because
  the model did not fully prepare or switch into the correct target environment.
- **Passing public tests is not enough.** The fd run passed `cargo test`, but
  hidden isolated grading failed before behavior checks because the patch added
  a dependency that was unavailable without network.
- **Partial implementations look plausible.** Prometheus and fd both produced
  meaningful patches, but hidden grading caught incomplete edge behavior or
  packaging assumptions.
- **Single-attempt repair can thrash.** The latest Anko run reached passing
  public no-network tests, then behavior-specific tests exposed that the
  generated parser was still untouched. The model removed those tests and
  deleted `parser.go.y` instead of repairing the generated artifact.
- **Container verification wording matters.** The latest Mashumaro run showed
  that ambiguous no-network repair text can push the model into running Docker
  with `shell` and then following it with local-only `verify_work`, which the
  Harness correctly rejects but should guide more directly.
- **Tracked source deletion is now a completion blocker by default.** External
  workspace verification records `deleted_source_paths` and refuses completion
  when tracked implementation files disappear without an explicit policy opt-out.
- **The Harness should stay behavior-first.** The correct response is not to
  add task-specific regexes or weather-style special cases. The runner should
  give the model tools, require evidence, and independently grade the result.

## Limitations

- This snapshot uses one model only: `openai/gpt-5.4-nano`.
- Runs were collected while the Harness was being improved, so this is not a
  clean N-run statistical comparison.
- Only two reference calibrations are saved in this snapshot. Prometheus passed;
  fd failed locally and should be investigated before using that task as a
  calibrated score.
- The table does not include cost, input tokens, output tokens, or wall-clock
  percentiles yet.
- The current result is a Harness development audit, not a DataCurve leaderboard
  submission.

## Reproduce

Example run shape:

```bash
uv run python evals/deepswe_runner.py /path/to/deep-swe/tasks/<task-id> \
  --env-file .env \
  --model openai/gpt-5.4-nano \
  --max-steps 70 \
  --max-output-tokens 4096 \
  --pass-timeout-seconds 180 \
  --run-timeout-seconds 2400 \
  --idle-timeout 120 \
  --turn-timeout 180
```

Do not pass hidden `tests/` or `solution/` paths to the agent. Use them only
after the agent stops, as evaluator-only grading artifacts.

## Next work

- Rerun the same task set after the current Harness fixes are committed.
- Run at least 3 trials per task before publishing model-comparison numbers.
- Save reference calibration artifacts for every reported task.
- Add an automated results-table generator from `outcome.json`, reward files,
  leak scans, and reference calibration runs.
- Add cost, token, runtime, and tool-use summaries once the pass/fail data is
  stable.
- Add isolated candidate/tournament workflows so failed assumptions can fork a
  fresh attempt instead of letting one context thrash after contradictory
  evidence.
