# Live Claude CLI validation

The opt-in live test uses an installed, authenticated `claude -p` command as the
model in an actual Harness agent run. Claude proposes structured tool calls or a
final answer; Harness dispatches each call and returns the real tool result on the
next model turn.

That bridge now ships as the `claude` provider
([`packages/adapter-claude`](../packages/adapter-claude/README.md)), and this test
exercises it directly. The separate `anthropic` provider continues to use the
Anthropic SDK with an `ANTHROPIC_API_KEY`; the `claude` provider instead reuses a
Claude Code subscription login through the CLI.

On a POSIX host with Claude Code installed and authenticated, run:

```bash
HARNESS_CLAUDE_TEST_BINARY="$(command -v claude)" \
  uv run pytest packages/cli/tests/test_claude_cli_live.py -q -s
```

`HARNESS_CLAUDE_TEST_MODEL` optionally selects the model; the test defaults to
`sonnet`. Running this command makes live model calls. Without the explicit
binary setting, the ordinary suite skips the test.

The fixture creates randomized paths and a randomized release label, then asks
for the active release and its source without supplying the answer or its path.
The test checks successful Harness directory listing and file reads, an answer
containing the exact label and source, a completed persisted session, unchanged
workspace files, and no clarification request. It uses the shared autonomous
context policy with clarification disabled by default.

Claude runs in an empty sibling directory with its native tools, MCP, skills and
session persistence disabled — the adapter passes those flags itself, so the same
isolation applies to ordinary `harness run` and `harness chat` invocations. Only
the Harness directory-listing and file-reading tools are exposed. Each turn owns a
process group that is torn down with `SIGTERM` then `SIGKILL`, so Claude Code's
helper processes cannot outlive it; the fixture also limits model turns, elapsed
time and reported spend. The printed
`CLAUDE_LIVE_RESULT` includes actual tool calls, the final answer, reported cost
and a path to the detailed temporary report.

This checks local context discovery through the Harness loop. Remote web
research, delegated workers, write approvals and Anthropic SDK compatibility have
separate regression coverage; a passing discovery fixture does not establish
their live acceptance or complete Hermes parity.

A run on 2026-09-13 against the shipped adapter passed in **22.06 seconds** using
Claude Code **2.1.270**: five model invocations produced four successful Harness
calls (directory listing, the discovered guide, the catalog and its selected
record) and an answer carrying both the randomized label and the exact source
path. Reported total cost was **$0.06270**.

The earlier run below used the superseded test-only bridge and is kept for
comparison. It passed in **21.97 seconds**, using Claude Code **2.1.270**. The requested `sonnet` alias resolved to `claude-sonnet-5`; the CLI
also reported Haiku helper usage. Five model invocations produced four successful
Harness calls: directory listing, reading the discovered guide, reading the
catalog and reading its selected record. Reported total cost was **$0.05708**.
The answer contained the correct randomized label and exact source path, along
with stray closing markup (`</answer></invoke>`). The test preserves the raw
answer; it verifies discovery and evidence, not prose formatting.

A separate restricted `claude -p` audit reported **149 passing existing tests**
and a successful CLI help check. It completed in 451.82 seconds and reported
**$4.633842** in usage. It identified two issues subsequently reproduced
with regression tests: legacy spawned agents rebuilt host tools under managed
execution, and local `run/chat --session` used the launch directory instead of
the saved workspace. Spawned agents now borrow the effective parent tools and
approval policy; CLI resumes resolve the saved local workspace before agent
construction. A third observation concerned intentional local memory defaults:
new tests verify automatic local writes and configured `prompt`/`deny` behavior,
and the [memory guide](skills-memory-mcp.md) documents the stricter gateway policy.
