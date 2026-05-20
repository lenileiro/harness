# Harness

> Python runtime/orchestration harness for LLM agents over **OpenRouter** and **Ollama**, with a ReAct tool-use loop, pluggable storage, per-tool approval, and policy-driven provider failover.

**Status:** scaffolding phase. Nothing is runnable yet. See [`~/.claude/todo.md`](../../.claude/todo.md) for the build plan.

## What it is

A Python equivalent of [`jido_harness`](https://github.com/agentjido/jido_harness)'s adapter contract, extended with a runtime layer that handles:

- Long-running sessions persisted across CLI invocations
- ReAct-style agent loop (think → tool call → observe → repeat)
- Streaming responses from any OpenAI-compatible provider
- Per-tool approval policies (`auto` / `prompt` / `deny`)
- Configurable failover chains across providers

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

All packages share the `harness.*` namespace at the import level (e.g.
`from harness.core import Agent`). The directory and distribution names
drop the `harness-` prefix to keep the workspace tidy.

All packages live under the `harness.*` namespace (PEP 420 implicit namespace packages).

## Development

```bash
# Install workspace + dev tools
uv sync

# Run the full quality gate
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest

# Auto-format
uv run ruff format .
uv run ruff check --fix .
```

## License

Private / unlicensed. Name and license TBD before any publication.
