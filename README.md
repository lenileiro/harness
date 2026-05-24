# Harness

Harness is a Python agent runtime and benchmark harness for tool-using LLMs.
It is built as a `uv` workspace with a Typer CLI, pluggable model adapters,
durable sessions and tasks, layered runtime defenses, and an execution-backed
eval stack for coding-agent behavior.

At a high level, Harness gives you:

- a resumable agent runtime with tools, approvals, and storage
- a CLI for running, chatting with, and inspecting agents
- structural defenses around the base model/tool loop
- persistent workspace memory, contracts, tips, and resume state
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
- `goal`: planner-first execution
- `init`: create workspace-local storage
- `sessions`: inspect and resume saved sessions
- `providers`: inspect provider configuration
- `tools`: inspect built-in tools
- `tasks`: durable task management
- `approvals`: inspect and resolve queued tool approvals
- `evidence`: inspect the tool-call evidence ledger
- `lab`: planner/worker/reporter multi-agent workflow
- `memory`: persistent workspace memory
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
- `--require-tools`: forbid answer-only responses
- `--auto-compact`: summarize old history when context is tight
- `--predict`: record consequence predictions before tool execution

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
- Resume state is a structured roadmap file at `.harness/resume.json`.

These layers are intended to make the outer runtime more informative and more
stable without modifying the base model itself.

Useful commands:

```bash
uv run harness contracts --help
uv run harness tips --help
uv run harness resume --help
```

The tips CLI currently supports:

- `list`
- `add`
- `test`
- `mine`

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

For benchmark rules and asset layout, see [evals/BENCHMARK.md](evals/BENCHMARK.md).

## Development Workflow

### Format, lint, type check, test

```bash
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

### Auto-format

```bash
uv run ruff format .
uv run ruff check --fix .
```

### Run a focused test slice

```bash
uv run pytest packages/core/tests/test_verification.py -q
uv run pytest evals/tests/test_runner.py -q
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
