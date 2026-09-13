# adapter-claude

Harness model provider backed by the locally installed, authenticated
**Claude Code CLI** (`claude`).

This is the right adapter when your Claude credential is a Claude.ai
subscription login rather than an API key: the CLI understands that OAuth login
state, while the stored token is not a general Anthropic API key. Use the
[`anthropic`](../adapter-anthropic) adapter when you have `ANTHROPIC_API_KEY`.

| Provider    | Transport                | Credential              |
| ----------- | ------------------------ | ----------------------- |
| `claude`    | local `claude` CLI       | `claude auth login`     |
| `anthropic` | `api.anthropic.com` SDK  | `ANTHROPIC_API_KEY`     |

## Claude is a model here, not an agent

Claude Code is launched with its native tools, MCP servers, skills, custom
commands and session persistence disabled. It only *proposes* tool calls, as a
structured decision validated against the schemas Harness supplied. Harness
dispatches every call through its own `ToolRegistry`, approval policy,
verifiers and critics — so the defended loop, the evidence trail and the eval
stack are unchanged.

A proposal naming a tool outside the supplied registry is rejected rather than
executed.

## Usage

```bash
claude auth login          # once
harness providers list     # expect: claude  ready
harness run --provider claude --model sonnet "summarize this repo"
```

Configure it in `config.toml`:

```toml
[default]
provider = "claude"
model = "sonnet"

[provider.claude]
timeout = 600.0         # absolute per-turn budget, seconds
idle_timeout = 120.0    # max gap between visible events, seconds
effort = "medium"       # low | medium | high | xhigh | max
max_budget_usd = 0.50   # optional per-turn spend ceiling
```

`HARNESS_MODEL_TURN_TIMEOUT` and `HARNESS_MODEL_STREAM_IDLE_TIMEOUT` override
the two timeouts. `model` accepts CLI aliases (`sonnet`, `opus`) or full ids
(`claude-sonnet-5`); an `anthropic/` or `claude/` prefix is stripped.

## Behaviour

- **Streaming.** The CLI always runs in `stream-json` mode. With no tools in
  play the adapter emits real `TextDelta` events. With tools, one structured
  decision is returned per turn — which may carry several calls for parallel
  dispatch — and the streamed deltas spell out that JSON, so they are counted
  as progress against `idle_timeout` but never enter the transcript.
- **Process ownership.** Each turn owns a new process group, torn down with
  `SIGTERM` then `SIGKILL`, so Claude Code's helper processes cannot outlive it.
- **Unsupported.** `temperature` (no CLI equivalent) and model-visible media
  are rejected with `ConfigurationError`. `max_tokens` maps to
  `CLAUDE_CODE_MAX_OUTPUT_TOKENS`.
- **Spend.** `ClaudeAdapter.cost_usd` accumulates the CLI's reported
  `total_cost_usd`, since a subscription login has no invoice to inspect later.

## CI

The `live_provider` input of `mission-autonomy` and `research-autonomy` accepts
`claude`, but a subscription login cannot exist on a GitHub-hosted runner. Those
workflows fail fast with a clear message unless they run on a self-hosted runner
where `claude auth login` has been completed. Use `openrouter` or `anthropic`
otherwise.

## Tests

```bash
uv run pytest packages/adapter-claude

# opt-in, makes real model calls - see docs/claude-cli-validation.md
HARNESS_CLAUDE_TEST_BINARY="$(command -v claude)" \
  uv run pytest packages/cli/tests/test_claude_cli_live.py -q -s
```
