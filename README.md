# Harness

Harness is a Python agent runtime and benchmark harness for tool-using LLMs.
It is built as a `uv` workspace with a Typer CLI, pluggable model adapters,
durable sessions and tasks, layered runtime defenses, and an execution-backed
eval stack for coding-agent behavior.

At a high level, Harness gives you:

- a resumable agent runtime with tools, approvals, and storage
- a CLI for running, chatting with, and inspecting agents
- structural defenses around the base model/tool loop
- persistent workspace memory, contracts, tips, experience, and resume state
- a behavioral eval harness for defended-vs-bare A/B testing

## What Harness Is For

Harness is designed for the part of agent systems that lives outside the model:
session management, tool orchestration, approval policy, verification, failure
recovery, and evals.

That means the project is useful in two modes:

1. As a runtime for real agent tasks in a workspace.
2. As an experimentation surface for improving the code around the model, not
   just swapping the model itself.

The repo currently includes adapters for OpenRouter, Ollama, and Anthropic, a
shared runtime core, storage backends, built-in tools, and a CLI that ties the
system together.

## Workspace Layout

This repository is a `uv` workspace. The root package depends on every
workspace member so `uv sync` installs the whole stack in editable mode.

```text
packages/
├── adapter-anthropic/    # Anthropic adapter
├── adapter-ollama/       # Ollama adapter
├── adapter-openrouter/   # OpenRouter adapter
├── cli/                  # Typer + Rich CLI, installs `harness`
├── core/                 # Runtime loop, verifiers, critics, contracts, tips
├── storage-memory/       # In-memory storage backend
├── storage-sqlite/       # SQLite storage backend
├── tasks/                # Durable task model and activity log
├── tools-fs/             # Filesystem tools
├── tools-shell/          # Shell execution tools
└── tools-web/            # HTTP/web tools
```

Supporting benchmark assets live under `evals/`.

Key internal module boundaries after the reorg:

```text
packages/cli/src/harness/cli/
├── __main__.py                    # CLI bootstrap and command registration
├── approvals_evidence_commands.py # approvals + evidence command family
├── builtin_tools.py               # built-in tool provider registration
├── chat_commands.py               # interactive chat / REPL flow
├── common.py                      # shared CLI helpers
├── config.py                      # CLI config loading and models
├── evals.py                       # eval command family
├── experience_commands.py         # procedures + curator commands
├── introspection.py               # providers + tools command family
├── lab_commands.py                # multi-agent lab command family
├── lifecycle_commands.py          # phase / contracts / tips / resume
├── markdown_render.py             # markdown + mermaid rendering
├── plugins.py                     # plugin discovery and precedence
├── render.py                      # Rich rendering helpers
├── review_commands.py             # diff-aware review entrypoint
├── run_commands.py                # one-shot run flow
├── runtime_agent.py               # runtime agent assembly
├── runtime_helpers.py             # verifier / critic / storage helpers
├── sessions_commands.py           # session command family
├── tasks_commands.py              # task command family
├── tune_commands.py               # prompt tuning command family
└── workspace_commands.py          # init / goal / memory command family

evals/
├── artifacts.py                   # artifact persistence helpers
├── calibration.py                 # judge calibration helpers
├── discovery.py                   # fixture discovery + metadata loading
├── failure_analyzer.py            # artifact-to-adjustment analysis
├── hard_checks.py                 # semantic hard behavior contracts
├── judge.py                       # optional LLM-as-judge scoring
├── runner.py                      # eval orchestration entrypoint
└── types.py                       # shared eval schemas

packages/core/src/harness/core/
├── domain_profiles.py             # task/domain policy presets
├── experience.py                  # public experience compatibility surface
├── experience_curator.py          # archival maintenance for procedures
├── experience_providers.py        # static + artifact + procedure retrieval
├── extensions.py                  # provider/plugin extension protocols
├── plugin_loader.py               # plugin manifest and loader
├── procedural_skill.py            # thin compatibility entrypoint
├── procedures.py                  # writable procedure artifacts
├── result_schemas.py              # typed machine-readable outputs
├── tool_entry.py                  # declarative tool entry model
├── tips_mining.py                 # tip extraction from failures
├── tips_models.py                 # tip and experience data models
├── tips_providers.py              # compatibility shim over experience
├── verification.py                # compatibility surface
├── verification_behavioral.py     # prompt/diff-sensitive verifiers
├── verification_guards.py         # guardrail helpers
├── verification_judges.py         # LLM/rule judge verifiers
└── verification_structural.py     # deterministic structural verifiers
```

## Installation

Harness targets Python 3.11+.

```bash
uv sync
```

If you want the `harness` command on your path outside `uv run`, install the
CLI package as an editable tool:

```bash
uv tool install --editable packages/cli
```

Otherwise use:

```bash
uv run harness --help
```

To install local Git hooks:

```bash
uv run pre-commit install --hook-type pre-commit --hook-type pre-push
```

The configured hooks run format, lint, and type checks before commit/push.

## Quick Start

### Run a one-shot task

```bash
uv run harness run \
  --provider openrouter \
  --model google/gemma-4-26b-a4b-it \
  --yes \
  "summarize the repository layout"
```

### Start a resumable chat session

```bash
uv run harness chat \
  --provider ollama \
  --model gemma4:latest
```

### Initialize workspace-local state

```bash
uv run harness init
```

This creates `.harness/harness.db` in the current workspace so later commands
can reuse local sessions, memory, tasks, and related state.

### Ask the agent to plan before acting

```bash
uv run harness run \
  --provider openrouter \
  --model google/gemma-4-26b-a4b-it \
  --goal \
  --yes \
  "refactor the approval flow and update the tests"
```

## Core CLI Surface

The main entrypoint is:

```bash
uv run harness --help
```

Current top-level commands include:

- `run`: single prompt execution
- `chat`: interactive REPL
- `review`: diff-aware read-only code review
- `goal`: planner-first execution
- `init`: create workspace-local storage
- `sessions`: inspect and resume saved sessions
- `plugins`: inspect discovered plugin providers
- `providers`: inspect provider configuration
- `tools`: inspect built-in tools
- `tasks`: durable task management
- `approvals`: inspect and resolve queued tool approvals
- `evidence`: inspect the tool-call evidence ledger
- `lab`: planner/worker/reporter multi-agent workflow
- `memory`: persistent workspace memory
- `experience`: manage writable procedures and curation
- `eval`: run and inspect behavioral evals
- `phase`: external phase tracking for multi-step tasks
- `tips`: procedural skill tips
- `tune`: prompt-tuning support for verifiers and critics
- `resume`: cross-session roadmap contract
- `contracts`: environment contracts loaded into runs

The CLI help is the source of truth for exact arguments and subcommands.

## Runtime Model

Harness runs a tool-using agent loop with durable state around it. The runtime
is not just “model + tools”; it also layers policy and verification around the
loop.

Important runtime concepts:

- Sessions: saved transcripts and activity that can be resumed later.
- Tasks: durable work items that can be linked to sessions.
- Approvals: per-tool approval policy, with optional durable inboxing.
- Evidence: a ledger of what tools ran and what happened.
- Memory: persistent facts injected into later runs.
- Contracts: hard environment rules loaded from `.harness/contracts/`.
- Tips: soft procedural hints loaded from `.harness/tips.jsonl`.
- Experience: artifact-backed lessons recovered from prior eval runs.
- Procedures: reusable writable guidance stored under `.harness/procedures/`.
- Resume state: a workspace roadmap file injected at run start.

## Running Agents

The most important command is `harness run`.

```bash
uv run harness run --help
```

Key options:

- `--provider`, `--model`, `--base-url`: model routing
- `--cwd`: tool working directory
- `--session`: reuse or create a named session
- `--task`: attach the run to a durable task
- `--yes`: auto-approve tools
- `--inbox`: queue approval requests instead of prompting
- `--max-steps`, `--max-output-tokens`, `--max-repair`: execution limits
- `--goal`: plan first, then execute
- `--domain`: task/use-case policy preset such as `coding` or `code-review`
- `--require-tools`: forbid answer-only responses
- `--auto-compact`: summarize old history when context is tight
- `--predict`: record consequence predictions before tool execution

## Code Review

Harness also supports a read-only review flow over the current git diff.

```bash
uv run harness review \
  --provider openrouter \
  --model google/gemma-4-26b-a4b-it \
  --base origin/main
```

This command:

- loads the current `git diff`
- runs the agent in the `code-review` domain profile
- restricts the tool set to read-oriented inspection tools
- asks for structured review findings instead of a freeform essay

Use `--json` when you want machine-readable output for CI or downstream tools.

### GitHub Actions example

A reference workflow lives at:

```text
examples/github-actions/review-pr.yml
```

It shows how to:

- run `harness review` on pull requests
- upload the raw JSON and rendered markdown as artifacts
- post or update a sticky pull request comment

Expected repository configuration:

- `OPENROUTER_API_KEY` or `ANTHROPIC_API_KEY` in repository secrets
- optional `HARNESS_PROVIDER` and `HARNESS_MODEL` in repository variables

The reference workflow intentionally skips forked PRs. Secrets are typically
not available there, and `code-review` should not silently fall back to an
unauthenticated run.

## Defense Profiles

Harness can run the same model/tool loop with different levels of structural
defense.

Current `--profile` values:

- `bare`: no defense chain, no critic; closest to raw model + tools
- `adaptive`: default; chooses a lighter or stricter path from task shape
- `diagnostic`: emphasizes diagnosis-alignment and repair quality
- `minimal`: light structural checks
- `strict`: the full verifier chain

These profiles are what the eval harness compares in defended-vs-bare A/B runs.

## Verification and Critics

Harness can verify whether the agent actually did the work it claims to have
done.

Verification modes include:

- `grounding`
- `state`
- `rule`
- `shell`
- `llm`
- `auto`
- `none`

Example:

```bash
uv run harness run \
  --provider ollama \
  --model gemma4:latest \
  --verify auto \
  --yes \
  "fix the failing test and leave the rest alone"
```

You can also attach a critic:

- `--critic llm`
- `--critic llm+search`
- `--critic none`

The verifier layer is where much of the harness behavior lives: tests-first,
verify-before-done, file-scope checks, diagnosis alignment, prompt-surface
revert logic, loop detection, and related safeguards.

## Sessions, Tasks, and Memory

### Sessions

Use sessions when you want continuity across invocations.

```bash
uv run harness sessions --help
```

Typical use:

```bash
uv run harness run --session fix-auth --yes "start debugging auth failures"
uv run harness sessions list
uv run harness resume --help
```

### Tasks

Tasks are durable work items with their own activity log.

```bash
uv run harness tasks --help
```

The task CLI supports creation, listing, inspection, updates, linking, and
deletion.

### Memory

Workspace memories are injected into every run.

```bash
uv run harness memory save --kind project_fact "use uv, not pip"
uv run harness memory list
uv run harness memory search "uv"
```

Supported memory operations:

- `save`
- `list`
- `search`
- `rm`

## Contracts, Tips, and Resume State

Harness separates hard and soft context:

- Contracts are hard rules loaded from `.harness/contracts/` and
  `~/.harness/contracts/`.
- Tips are procedural hints loaded from `.harness/tips.jsonl` and
  `~/.harness/tips.jsonl`.
- Experience can also be recovered from saved eval artifacts, where analyzed
  `harness_adjustments.json` records are turned back into reusable guidance for
  defended runs.
- Resume state is a structured roadmap file at `.harness/resume.json`.

These layers are intended to make the outer runtime more informative and more
stable without modifying the base model itself.

Useful commands:

```bash
uv run harness contracts --help
uv run harness tips --help
uv run harness resume --help
uv run harness experience --help
```

The tips CLI currently supports:

- `list`
- `add`
- `test`
- `mine`

The experience CLI supports writable procedure artifacts and curation:

- `procedures add`
- `procedures list`
- `curate`

## Evals

Harness includes a behavioral eval stack under `evals/`.

Use it to compare defended and bare agent behavior, generate mutated fixtures,
calibrate judge outputs, and track saved benchmark runs.

```bash
uv run harness eval --help
```

Current subcommands:

- `list`
- `mutate`
- `calibrate`
- `history`
- `adjustments`
- `export-adjustments`
- `validate`
- `run`

### Example eval runs

```bash
# Run the smoke suite with defended-vs-bare A/B and save artifacts
uv run harness eval run --suite smoke --ab --n-runs 3

# Run with JSON output
uv run harness eval run --suite smoke --ab --json-out

# Run mutation-based variants
uv run harness eval run --suite smoke --benchmark-mode mutated --mutation-seeds 7,8

# Inspect analyzer output from saved eval runs
uv run harness eval adjustments evals/runs --limit 20

# Export a consolidated adjustment corpus
uv run harness eval export-adjustments adjustments.jsonl --root evals/runs

# Validate fixtures, suites, and gold labels
uv run harness eval validate
```

### How eval scoring works

The eval harness uses two layers of scoring:

1. Hard metrics from execution evidence and fixture-specific behavior contracts.
2. Optional LLM-judge scores for qualities like scope discipline, decomposition,
   pushback, and epistemic grounding.

The hard layer is behavior-first, not exact patch-text matching. A correct fix
should not fail just because of harmless whitespace noise; the contracts are
intended to check what changed and why, not only whether a diff matches a
single string shape.

Each saved eval run also persists artifact-backed adjustment hints such as:

- what kind of failure happened
- which verifier or guardrail would have helped
- what corrective behavior should be reinforced later

Those adjustments are written into `harness_adjustments.json` beside the other
run artifacts and can be inspected or exported with the eval CLI.

For benchmark rules and asset layout, see [evals/BENCHMARK.md](evals/BENCHMARK.md).

## Development Workflow

### Format, lint, type check, test

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

These are the same quality gates enforced in CI.

### Auto-format

```bash
uv run ruff format .
uv run ruff check --fix .
```

### Run a focused test slice

```bash
uv run pytest packages/core/tests/test_verification.py -q
uv run pytest evals/tests/test_execution.py -q
```

## Design Principles

This repo is increasingly oriented around a simple claim:

> The code around the model matters.

That means Harness focuses on:

- explicit runtime state instead of opaque conversations
- execution evidence instead of answer-only scoring
- defended-vs-bare comparisons instead of ungrounded claims
- reproducible artifacts instead of benchmark anecdotes
- environment-layer improvements, not only model swaps

If you are working on coding agents, evals, or runtime defenses, that is the
part of the stack this repository is trying to make concrete.
