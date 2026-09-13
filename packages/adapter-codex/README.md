# Codex adapter

`CodexAdapter(mode="app-server")` exposes Harness tools through Codex's real
bidirectional app-server protocol. The existing `mode="exec"` default retains
native Codex CLI behavior and does not support caller-provided tools.

Configure the CLI with:

```toml
[provider.codex]
mode = "app-server"
```

The bridge uses the existing Codex login/provider configuration without copying
credentials. Each model turn has a private stdio process and ephemeral thread
in an empty temporary directory. Both thread and turn select no native execution
environments. Native plugins, hooks, shell, browser, skills, apps and other action
features are disabled and checked against effective configuration; inherited
native MCP servers are explicitly disabled before thread creation. The model
receives Harness tools in the `harness` namespace. Approval, execution, evidence,
secret redaction and workspace/backend restrictions stay in Harness.

Tool requests pause the stream with an ordinary `ToolCallEvent`/`Done` pair.
Harness runs the tool, supplies its sanitized `ToolResult`, and the next stream
replies to the pending RPC before continuing the same Codex turn. Changed models,
tool schemas, instructions, compacted context and resumed approvals rebuild a
new thread from the current Harness transcript; tool effects are not replayed.
All terminal, paused and cancelled runs close the owned process.

Standalone adapter callers must pass `session_id` to `stream`, call
`submit_tool_result(session_id, result)` after each tool, and call
`await adapter.end_run(session_id)` in `finally`. `Agent` performs this lifecycle.
Concurrent streams must have distinct session IDs.

Inline image and audio attachments are preserved, including tool result media.
Remote media URLs and generic file attachments fail with an explicit diagnostic;
load supported local media with `MediaAttachment.from_file`. Attachments marked
`model_visible=False` stay out of model requests. Individual models may support
fewer input modalities. Codex exposes no `temperature` or `max_tokens` parameter;
both modes reject those settings instead of silently discarding them.

The experimental protocol and isolation settings were verified with Codex
**0.154.0**, generated version-matched schemas, and its
[official app-server documentation](https://learn.chatgpt.com/docs/app-server).
Incompatible configuration/protocol versions fail closed. The actual binary test
uses a loopback fake Responses provider, a disposable fake login, inline images,
and an inherited MCP sentinel; it verifies the tool RPC round trip, absent native
tools, and no sentinel execution. No real model/provider is called:

```sh
HARNESS_CODEX_TEST_BINARY="$(command -v codex)" \
  uv run --no-sync pytest packages/adapter-codex/tests/test_app_server_live.py
```

Process-group cleanup was exercised on macOS. Native Windows subprocess tree
ownership is not yet covered by this adapter's integration tests.
