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
├── harness-core/                 # Protocols, schemas, runtime, ReAct loop
├── harness-storage-memory/       # In-memory Session storage
├── harness-storage-sqlite/       # SQLite (aiosqlite) Session storage
├── harness-adapter-openrouter/   # OpenRouter HTTP/SSE adapter
├── harness-adapter-ollama/       # Ollama HTTP/SSE adapter
├── harness-tools-fs/             # read / write / edit / list / glob
├── harness-tools-shell/          # subprocess exec
├── harness-tools-web/            # http fetch
└── harness-cli/                  # Typer + Rich CLI
```

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
