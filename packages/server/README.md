# Harness server

An authenticated HTTP API, browser client, and optional official-SDK MCP server backed by the existing Agent and SQLite runtime. Each submitted run has a durable ID, caller-owned session, replayable events, and explicit terminal status.

```python
from harness.server import HarnessService, ServerAuth, create_app

# build_agent(context) returns a fresh configured harness.core.Agent.
service = HarnessService(database, workspace, build_agent, exposed_tools=["read_file"])
app = create_app(service, auth=ServerAuth(token_envs={"alice": "HARNESS_ALICE_TOKEN"}))
```

The builder receives `RunContext(owner, workspace, storage, run_id, session_id, model)`. It must bind filesystem/managed tools to that workspace and identity. A shared filesystem or shared remote credentials are not a tenant sandbox. The CLI builder uses its remote filesystem boundary and excludes native execution and managed shared credentials.

## Start and connect

Set a random token of at least 32 characters in an environment variable; pass its variable name, never its value, on the command line:

```sh
harness serve --workspace . --adapter openai --model gpt-4.1 \
  --token-env local:HARNESS_SERVER_TOKEN --expose-tool read_file
```

Open `http://127.0.0.1:8765/` and enter the token. The browser client supports conversations, search, streamed activity, cancellation/resumption, approval review, batches, attachments, and JSONL export. Tokens remain in window memory and are cleared on disconnect/reload. Only the static login shell is public. The API accepts bearer authentication; cookies and query parameters never authenticate callers.

`--host` defaults to loopback. Remote binding requires `--allow-remote`, `--tls-cert`, and `--tls-key`; browser origins and HTTP hosts are explicitly constrained. `--workers` controls concurrent jobs inside one process. A database lease rejects a second server process; do not use Uvicorn process workers for this service.

Install native desktop support with `uv sync --package cli --extra desktop`, then run:

```sh
harness desktop --url http://127.0.0.1:8765 --token-env HARNESS_SERVER_TOKEN
```

This opens a private pywebview window against an existing server. Its environment token is injected only into the configured origin. It does not launch a second backend. Native GUI prerequisites vary by OS; see [pywebview installation](https://pywebview.flowrl.com/guide/installation.html). The wrapper is unit-tested; the browser client is exercised in Chromium. Native window launch is not part of headless CI.

## HTTP API v1

All data routes require `Authorization: Bearer ...`. POST bodies use JSON. Caller identity comes from the server token mapping and cannot be supplied in a request body.

| Route | Behavior |
|---|---|
| `POST /v1/runs` | Queue `{prompt, session_id?, model?, max_steps?, attachments?}`; returns 202 with run/session IDs |
| `GET /v1/runs` | Latest 50 caller-owned runs; optional `session_id` filter |
| `GET /v1/runs/{id}` | Durable status |
| `GET /v1/runs/{id}/events` | SSE; resume with `Last-Event-ID` or `after` cursor |
| `POST /v1/runs/{id}/cancel` | Cancel a queued/running job |
| `POST /v1/runs/{id}/resume` | Create a new linked run from saved conversation; optional `{prompt}` |
| `GET /v1/sessions` | Caller-owned sessions; `q`, `limit` (1–50), `offset` |
| `GET /v1/sessions/{id}/messages` | Full stored messages |
| `GET /v1/sessions/{id}/export` | Version 1 trajectory JSONL |
| `GET /v1/approvals` | Caller-owned pending actions |
| `POST /v1/approvals/{id}/resolve` | Immutable decision `{granted: true/false}` |
| `GET /v1/questions` | Owned pending, answered, skipped or expired forms awaiting continuation |
| `POST /v1/questions/{id}/answer` | Save partial or complete `{answers: {q0: "text"}}` without resuming |
| `POST /v1/questions/{id}/cancel` | Skip unanswered questions while retaining saved answers |
| `POST /v1/batches` | Atomically queue `{runs: [...]}` (1–100 independent requests) |
| `GET /v1/batches` | Latest 50 caller-owned batches |
| `GET /v1/batches/{id}` | Batch and individual run statuses |
| `POST /v1/batches/{id}/cancel` | Cancel remaining work |
| `POST /v1/tool-runs` | Queue explicit `{name, arguments}` through Agent gates |

Run states are `queued`, `running`, `completed`, `paused`, `failed`, `cancelled`, or `interrupted`. Completion is determined after the Agent iterator and verification finish, not by the first model `done` event. An SSE disconnect leaves execution running. Restart retains queued work; previously running work becomes `interrupted` and requires explicit inspection/resumption. One session may have only one queued/running request.

Tool RPC injects an explicit tool-call adapter into a real Agent. It does not call raw tools. Only `exposed_tools` are permitted, existing denies win, and durable/external effects require inbox approval. Resolve an approval, then explicitly resume its paused run. Claimed uncertain approvals are never automatically executed again. A denied tool action cannot be resumed as successful.

Attachments use the versioned core `MediaAttachment` schema and are validated before enqueue. The HTTP body cap is 24 MiB (base64 counts toward the cap). The browser limits selected files to 8 MiB each; provider capability checks may reject unsupported media. Remote attachment URLs are not fetched automatically by the browser transcript.

A JSONL export is one complete session trajectory per line: `format="harness.trajectory"`, `version=1`, `session`, `messages`, `runs`, and `events`. Tool call IDs and corresponding tool-result messages remain together. Interrupted/pending state remains explicit. Diagnostic strings are redacted; attachment payloads remain intact. Exports are data artifacts, not model-training endpoints.

## MCP

`--mcp` mounts Streamable HTTP at `/mcp/`. The outer HTTP authentication/Host/Origin boundary also protects MCP, and each MCP call uses that request's authenticated identity. Default exposed methods are `run_status`, `sessions`, and `messages`. Repeat `--mcp-operation` to explicitly expose `submit`, `cancel`, `resume`, `approvals`, `resolve_approval`, `questions`, `answer_question`, `cancel_question`, or `tool_run`. `tool_run` still requires the separate tool-name exposure allowlist and Harness approval/evidence gates.

The implementation uses the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk/tree/v1.30.0), [Starlette lifespan](https://www.starlette.dev/lifespan/), and streaming responses. Native desktop integration uses the documented [pywebview API](https://pywebview.flowrl.com/api/), including `run_js` to preserve the web client's strict CSP.

## Durable delegation

`delegate`, `delegate_status`, `delegate_cancel`, `delegate_resume`, and `delegate_artifact` are constructor-bound Tool implementations. Model invocation passes through Agent's normal dispatch/approval/evidence pipeline. The HTTP server exposes these to models only when their names are explicitly allowlisted. Direct user submission is also available at `POST /v1/delegations` with `parent_session_id`, `prompt`, optional `inputs`, and budget options. `GET /v1/delegations`, `GET /v1/delegations/{handle}`, `POST .../cancel`, `POST .../resume`, and `GET .../artifacts?path=...` enforce the authenticated owner. The web client has a delegated-jobs view.

Every child has a new session and private workspace under the database's `jobs/` directory. Context arrives through its prompt and explicitly selected input files. Copying pins every path component, refuses symlinks, dot components, special files, and traversal; inputs are capped at 32 files, 4 MiB per file, and 8 MiB combined. Platforms lacking directory-relative, no-follow file opens support prompt-only delegation. Child changes never merge automatically. Selected artifacts are returned as attachments or explicit downloads.

The trusted builder must honor `RunContext.workspace` and bind every filesystem/execution capability to it. Harness's CLI server builder applies its filesystem boundary and omits host-native execution. A custom builder that exposes unconstrained shell or host paths defeats filesystem isolation.

Default shared parent limits are 8 children, depth 2, 80 model calls, 81,920 requested output tokens, and 600 seconds of reserved attempt time. Each attempt defaults to 10 model calls, at most 1,024 requested output tokens per call, and a hard elapsed timeout of 120 seconds including adapter construction. Reservations commit atomically before enqueue and remain charged after interruption, failure, or cancellation. Provider retries and repair calls share the model-call allocation. The output-token limit is passed to providers; it is not a guarantee of provider billing or tokenizer equivalence. Explicit resume reserves another attempt. The ordinary run API cannot append an unbudgeted request to a delegated session.

```sh
harness delegate --workspace . --adapter openai submit "Inspect these modules" --input src/module.py
harness delegate --workspace . submit "Independent task" --queue-only
harness delegate --workspace . status JOB_ID
harness delegate --workspace . work
harness delegate --workspace . cancel JOB_ID
harness delegate --workspace . approvals JOB_ID
harness delegate --workspace . resolve JOB_ID APPROVAL_ID --decision grant
harness delegate --workspace . questions JOB_ID
harness delegate --workspace . answer JOB_ID QUESTION_ID '{"q0":"Report","q1":"Team"}'
harness delegate --workspace . resume JOB_ID
harness delegate --workspace . artifact JOB_ID result.txt --output reviewed-result.txt
```

The local database defaults to `.harness/delegation.db`. Submit/resume run in the foreground unless `--queue-only` is used. Administrative status/cancel/approval/question commands do not start model calls and can coexist with an active worker. Expose `clarify` for child questions; status includes saved partial answers. Use `answer` or `skip-question`, then resume separately. An unanswered question blocks resume before another budget reservation. Cancellation is durable; an active worker acknowledges it before the run becomes cancelled. Ctrl-C preserves interrupted state. A crashed worker's previously running jobs become interrupted on the next worker startup and are not replayed automatically. See the [question workflow guide](../../docs/clarification.md).

Local `harness delegate` commands allow a read-only discovery preset by default:
`read_file`, `list_dir`, `glob`, `web_search`, `fetch_url`, `recall_memory`,
`search_sessions`, and `conversation_window`. Files remain limited to the child's
workspace, and recall remains limited to its own workspace and identity. Repeated
`--expose-tool` flags replace this preset with the exact named set. Managed local
children also receive scoped recall tools; their write actions still require
review. HTTP serving and standalone `LocalDelegationToolset` constructors retain
their explicit, empty-by-default tool allowlists. To expose read-only fact lookup
on the server, use `harness serve --expose-tool recall_memory`.

Standalone applications can bind child tools to an ordinary Agent after session bootstrap:

```python
from harness.server import LocalDelegationToolset

async with LocalDelegationToolset(database, workspace, build_child_agent) as children:
    agent.session_tool_factory = children.bind
    async for event in agent.run(request):
        handle(event)
```

The toolset preserves the parent's existing storage and memory scope. Normal context exit drains queued/active jobs; exceptional exit interrupts active jobs and keeps queued work durable. Its `DelegationLimits` can be supplied at construction. Child approval inboxes remain separate from the parent's storage; use the child handle's approval records and resolve/resume commands.

## Web preferences, tool inspection and schedules

Authenticated users can open **Preferences**, **Tools**, and **Schedules** in the web/desktop UI. Preferences belong to the bearer-token identity and persist in SQLite. Only provider/model/timezone preferences are editable; credentials, shell access and host service configuration are not browser settings.

- `GET /v1/configuration` exposes explicitly selected provider IDs, their configured model defaults, and supported preference/schedule behavior. It never serializes raw provider configuration, endpoint headers, secret environment references or credential values.
- `GET|POST /v1/preferences` reads/replaces the current identity's `{provider, model, timezone}` defaults. New runs/batches/schedules freeze those defaults. Existing sessions retain their provider/model unless an explicit request selects another enabled provider/model. Unknown providers fail before queueing and are checked again before execution. Operator config determines the allowlist.
- `GET /v1/tools` returns tool names, descriptions, parameter schemas, effect scope, exposure and effective approval floor. Unexposed/denied tools remain denied; exposed side effects still require approval/evidence. No endpoint changes exposure policy. Runtime-only tools may report that their schema is not yet available.
- `GET|POST /v1/schedules` lists/creates owned prompt schedules. Creation accepts `{title, prompt, at|every|cron, timezone?, provider?, model?, max_steps?}`; exactly one timing mode is required. `GET /v1/schedules/ID` includes occurrence history and latest run attempts. `POST /v1/schedules/ID/pause`, `/resume`, and `/cancel` take JSON `{}`.

Schedules reuse Harness's duration, cron and timezone parser, including rejection of ambiguous/nonexistent local wall times unless an explicit offset is supplied. Recurring intervals must be 60 seconds through 366 days; each identity may keep up to 100 active/paused/error schedules. A new one-time schedule must be in the future. The UI displays the next occurrence in the selected timezone.

Occurrence claiming, schedule advancement and run insertion happen in one SQLite transaction. A restart cannot lose a queued occurrence between those steps or duplicate the same occurrence. Missed recurring times coalesce into one queued occurrence; they are not replayed as a backlog. Each occurrence uses a fresh conversation. An earlier occurrence's queued/running/approval-paused session blocks overlap; an explicit successful resume unblocks future occurrences. Run effects use the same Harness approval/evidence pipeline as interactive work.

Pause stops future occurrences while current work continues. Resume schedules the next recurrence from now (an overdue one-time schedule becomes due immediately). Cancel stops future occurrences and cancels associated queued/running work. A one-time schedule marked `completed` means its occurrence was queued; the separate run state reports whether the actual work completed, failed or needs approval. The UI always shows that latest run state.

The scheduler runs while the owning Harness service process is running; it is not an OS scheduler or a browser timer. CLI workers using the same database must use matching operator config/provider flags so persisted selected providers remain allowed. The CLI catalog deliberately disables local computer/browser/execution tools, matching the restricted server builder.

## Questions in the web and desktop UI

Agents gather context and research autonomously by default. For an explicitly interactive
workflow, expose `clarify` when starting the server (`harness serve --expose-tool clarify`).
The model can then ask up to five questions together, with up to four choices each,
multiple selections, or open text. **Questions** has its own navigation item and count.
A paused conversation links to its questions or approvals according to what is pending.

Choose answers or enter your own text; free text is available even when choices are
provided. For multiple selections, an additional text answer is saved alongside the
selected choices. **Save answers** supports a partial form. Accepted answers become
read-only and survive disconnects, reloads, and server restarts. Unsaved drafts stay only
in the current window's memory and are cleared on disconnect/reload.

After all answers are saved, select **Continue conversation**. An outstanding approval
can still block resuming; the UI keeps the accepted answers and provides a link to
**Approvals**. Saving an answer never grants permission for an action. **Skip unanswered
questions** retains existing answers and allows continuation without the remaining
answers. Expired questions likewise allow continuation with any answers already saved.
Cancelling the run itself is separate from skipping its questions.

The UI uses the owner-scoped `/v1/questions` list and its `/{id}/answer` and `/{id}/cancel`
operations, followed by the original paused run's `/resume`. It renders question and
answer text as text, never HTML. Browser acceptance checks use a local fake adapter,
real Harness Agent and SQLite service, including partial saves, multi-select/custom
answers, reload persistence, approval-blocked continuation, skip/expiry, and narrow
mobile layouts; no live provider is needed.
