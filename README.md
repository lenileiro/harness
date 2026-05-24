# Harness

> Python runtime/orchestration harness for LLM agents over **OpenRouter** and **Ollama**, with a ReAct tool-use loop, pluggable storage, per-tool approval, policy-driven provider failover, persistent memory, session forking, and post-run verification.

**Status:** functional. Install with `uv sync`, then run `harness --help`.

## What it is

A Python equivalent of [`jido_harness`](https://github.com/agentjido/jido_harness)'s adapter contract, extended with a runtime layer that handles:

- Long-running sessions persisted across CLI invocations
- ReAct-style agent loop (think → tool call → observe → repeat)
- Streaming responses from any OpenAI-compatible provider
- Per-tool approval policies (`auto` / `prompt` / `deny`)
- Configurable failover chains across providers
- Persistent memory injected into every run
- Workspace-local storage via `harness init`
- Session forking to branch from a prior conversation
- Post-run verification that catches agents lying about work done
- Multi-step goal planning via LLM before the ReAct loop

## Layout

This is a [uv](https://docs.astral.sh/uv/) workspace with nine packages:

```
packages/
├── core/                  # Protocols, schemas, runtime, ReAct loop
├── storage-memory/        # In-memory Session storage
├── storage-sqlite/        # SQLite (aiosqlite) Session storage
├── adapter-openrouter/    # OpenRouter HTTP/SSE adapter
├── adapter-ollama/        # Ollama HTTP/SSE adapter
├── tools-fs/              # read / write / edit / list / glob
├── tools-shell/           # subprocess exec
├── tools-web/             # http fetch
└── cli/                   # Typer + Rich CLI (installs `harness` binary)
```

All packages share the `harness.*` namespace at the import level (e.g. `from harness.core import Agent`).

---

## Installation

```bash
# 1. Install workspace dependencies
uv sync

# 2. Install git hooks (runs format checks before commit/push)
uv run pre-commit install --hook-type pre-commit --hook-type pre-push

# 3. Install the harness binary globally (accessible without activating the venv)
uv tool install --editable packages/cli

# Confirm it works
harness version
```

Or if you prefer not to install globally, prefix every command with `uv run`:

```bash
uv run harness version
uv run harness --help
```

---

## Usage

### Run a one-shot prompt

```bash
# Against Ollama (local)
harness run --provider ollama --model gemma4:latest --yes "list the files in /tmp"

# Against OpenRouter
harness run --provider openrouter --model openai/gpt-4o --yes "summarize this project"

# With a named session (resumable later)
harness run --provider ollama --model gemma4:latest --session my-session --yes "start a task"
```

### Interactive chat

```bash
harness chat --provider ollama --model gemma4:latest
```

Keeps a session alive across turns. `Ctrl+D` to exit; resume later with `harness sessions resume`.

### Multi-step goal planning

```bash
# The LLM generates a plan first, then executes each step
harness goal --provider ollama --model gemma4:latest --yes \
  "refactor the approval handler to support batch approvals"

# Equivalent using --goal flag on run
harness run --provider ollama --model gemma4:latest --yes --goal \
  "refactor the approval handler to support batch approvals"
```

### Workspace-local storage

```bash
# Initialise a .harness/harness.db in the current directory
harness init

# All subsequent harness commands in this directory auto-use .harness/harness.db
harness run --provider ollama --model gemma4:latest --yes "hello"
harness sessions list
```

### Session management

```bash
# List all saved sessions
harness sessions list

# Show full transcript of a session
harness sessions show <session-id>

# Resume a session with a new prompt
harness sessions resume <session-id> --yes "continue where we left off"

# Fork a session — branch from a prior conversation's message history
harness sessions fork <session-id>

# Fork and immediately run a new prompt in the fork
harness sessions fork <session-id> --yes "try a different approach"

# Delete a session
harness sessions rm <session-id>
```

### Persistent memory

Memory entries are injected as a system message at the start of every run. The agent always knows what you've told it.

```bash
# Save facts the agent should always know
harness memory save --kind project_fact "this project uses uv, not pip"
harness memory save --kind user_preference "prefer concise responses"
harness memory save --kind project_context "we are refactoring the auth module"

# Memory kinds: user_preference | user_fact | project_fact | project_context
harness memory list
harness memory list --kind project_fact
harness memory search "uv"
harness memory rm mem_<id>
```

### Post-run verification

Harness can verify whether the agent actually accomplished the goal — catching cases where an LLM verbally claims work is done without calling any tools.

```bash
# Rule-based: fast, checks tool errors only — blind to verbal lies
harness run --provider ollama --model gemma4:latest --yes --verify rule \
  "create a file called output.txt"

# LLM judge: one extra adapter call to evaluate the outcome
harness run --provider ollama --model gemma4:latest --yes --verify llm \
  "create a file called output.txt"

# Auto (recommended): routes based on what tools were called
#   - no tools / mutating tools (write_file, edit_file, shell) → LLM judge
#   - read-only tools only → rule verifier (fast, no extra cost)
harness run --provider ollama --model gemma4:latest --yes --verify auto \
  "create a file called output.txt"
```

The LLM judge retries up to 3 times on transient failures (network errors, malformed JSON) with exponential backoff before giving up.

Verification output appears as `✓ verify` or `✗ verify` with a reason and confidence score.

---

## Testing

### Unit and integration tests

```bash
# Full test suite (451+ tests, ~9s)
uv run pytest

# A single package
uv run pytest packages/core/

# Specific test file
uv run pytest packages/core/tests/test_verifier_routing.py -v

# With coverage
uv run pytest --cov=harness --cov-report=term-missing
```

### Full quality gate

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

### Live testing against Ollama

Requires [Ollama](https://ollama.com) running locally with a model pulled (e.g. `ollama pull gemma4:latest`).

```bash
# Basic smoke test
harness run --provider ollama --model gemma4:latest --in-memory --yes "What is 2+2?"

# Filesystem tools
harness run --provider ollama --model gemma4:latest --in-memory --yes \
  "Use the shell tool to run: echo hello > /tmp/test.txt, then confirm it worked"

# Verify the agent actually did the work (not just claimed to)
harness run --provider ollama --model gemma4:latest --in-memory --yes --verify auto \
  "Use the shell tool to write 'hello' into /tmp/harness-test.txt"

# Memory smoke test
harness memory save --kind project_fact "uses uv workspace" --in-memory
harness run --provider ollama --model gemma4:latest --in-memory --yes \
  "what do you remember about this project?"

# Workspace init smoke test
mkdir /tmp/test-workspace && cd /tmp/test-workspace
harness init
harness run --provider ollama --model gemma4:latest --yes "hello"
harness sessions list   # shows session in .harness/harness.db

# Session fork smoke test
harness run --provider ollama --model gemma4:latest --session base-sess --yes "remember the number 42"
harness sessions fork base-sess --yes "what number did we discuss?"

# Goal mode smoke test
harness goal --provider ollama --model gemma4:latest --in-memory --yes \
  "summarize this project in 3 sentences"   # emits 2-5 StepStarted events
```

---

## Provider failover

```bash
# Try Ollama first, fall back to OpenRouter if it fails
harness run --failover ollama,openrouter --model gemma4:latest --yes "hello"
```

---

## Approval policies

By default, high-risk tools like `shell` require interactive approval. Use `--yes` to auto-approve all tools, or `--inbox` to queue approvals for later review:

```bash
# Approve all interactively (prompted per tool call)
harness run --provider ollama --model gemma4:latest "do something risky"

# Auto-approve everything
harness run --provider ollama --model gemma4:latest --yes "do something"

# Queue approvals in the durable inbox (non-interactive, human-in-the-loop later)
harness run --provider ollama --model gemma4:latest --inbox "do something"
harness approvals list
harness approvals approve <approval-id>
```

---

## License

Private / unlicensed. Name and license TBD before any publication.
