# Authenticated A2A serving

Harness serves the official A2A JSON-RPC transport using
[`a2a-sdk` 1.1.2](https://pypi.org/project/a2a-sdk/1.1.2/) and the
[official Python SDK](https://github.com/a2aproject/a2a-python). Its protobuf
request handler uses the same persistent run queue, caller identity, tool
exposure policy, approval inbox, and evidence gates as the HTTP API.

```sh
harness serve --workspace . --token-env alice:HARNESS_API_TOKEN \
  --a2a-url http://127.0.0.1:8765/a2a
```

`--a2a-url` enables POST `/a2a` and GET `/.well-known/agent-card.json`.
Both require the caller's bearer token. The card contains the operator's
explicit endpoint URL; request Host headers cannot rewrite it. Remote listening
still requires the server's explicit remote-binding and TLS flags. Browser
Origin, allowed-host, and bearer checks cover both routes.

Use the `A2A-Version: 1.0` header with `SendMessage`,
`SendStreamingMessage`, `GetTask`, `ListTasks`, `SubscribeToTask`, and
`CancelTask`. The SDK also handles A2A 0.3 `message/send`, `message/stream`,
`tasks/get`, `tasks/cancel`, and `tasks/resubscribe` requests with its own
compatibility converter and the 0.3 version header. This document covers the
inbound server. Outbound peer tools are configured separately; this endpoint
does not implement OAuth discovery/login.

Each accepted message needs a stable `messageId`. Queue insertion and the
owner-scoped message/task mapping commit in one transaction. Retrying that ID
with the same message observes its existing task; reusing it with different
content is rejected. A failed response, disconnected stream, or server restart
never implicitly repeats a task. In-flight work interrupted by a server restart
is reported as failed with an interruption explanation. Queued work remains
durable. No automatic retry is made for uncertain effects.

`configuration.returnImmediately=true` returns the queued task. The default
unary request waits until completion or human input is required. Streaming
requests and subscriptions emit task snapshots as durable status changes, with
the final answer artifact only after Harness marks the run completed. They do
not expose unverified partial answers as completed artifacts. Disconnecting a
subscriber leaves the run active. Reconnect using `SubscribeToTask` or `GetTask`.
Every task includes `metadata.harness_run_id` for the full HTTP event stream.

Human approval produces `TASK_STATE_INPUT_REQUIRED`. Review and resolve the
approval through the existing authenticated UI or `/v1/approvals`, then send a
new message ID with the task's `taskId` and `contextId`. This continues the same
task while recording a new Harness run. Resuming through the HTTP/UI route also
updates that task handle. Completed, failed, or cancelled tasks cannot be
restarted by reusing their task ID; omit `taskId` and retain `contextId` to
explicitly request new work in the same owned conversation.

When `clarify` is explicitly exposed, the model can also pause with
`TASK_STATE_INPUT_REQUIRED` to ask questions. `GetTask` and streaming task
snapshots include the actual unanswered questions, choices, expiry, and
`metadata.harness_question_id`; an input pause is not automatically an approval
request. Reply with `SendMessage`, the same `taskId`/`contextId`, a fresh stable
`messageId`, and text answering the next question. For several answers, put a
JSON object in the text part, for example
`{"q0":["Report","Custom output"],"q1":"Engineering team"}`. The 0.3
`message/send` path uses the same answer and execution pipeline.

Partial answers return the existing input-required task. Answer normalization
and the owner/message receipt commit together, so concurrent requests and
retries after restart cannot apply one reply to the next unanswered question.
Changed content under an accepted message ID is rejected. Once complete, the
answer is saved before continuing through the ordinary Harness queue. If an
approval or interrupted request prevents enqueue, resolve the blocker and retry
the same message ID: the saved receipt completes enqueue without answering
again or launching duplicate work. Answers never grant tool approvals.
Expired or explicitly skipped questions continue with their saved partial
answers and an expiry/cancellation marker. Use the HTTP/UI question cancel
operation to skip questions; A2A `CancelTask` cancels the whole run. Configure
callback subscriptions separately through their CRUD operations before replying
to a clarification; inline callback changes on answer messages are rejected
before answers are recorded.

Cancellation stops queued/running execution. Cancelling an approval-paused run
atomically denies pending approvals, revokes unclaimed grants, and marks the run
cancelled. Cancelled runs cannot be resumed. If an effect was already claimed
and its outcome is unknown, cancellation reports a conflict instead of claiming
to undo it; inspect the recorded execution and target state.

Text and typed image/audio/file parts use the shared validated attachment
schema: bounded inline bytes or HTTP(S) references, never arbitrary local paths.
The selected model still needs the corresponding media capability. Structured
data parts, arbitrary message extensions, reference
tasks, and separate tenant parameters are explicitly unsupported. The bearer
token determines ownership; metadata cannot select another user or workspace.

History is opt-in with `historyLength` from 0 to 100 and contains conversational
user/assistant content. Full tool call/result pairs remain available in owned
Harness session exports. `ListTasks` supports owned context/status/timestamp
filters, page sizes 1–100, opaque page tokens, and optional artifacts. Requests
are capped at 24 MiB before SDK parsing.

## Signed push notifications

Callbacks are opt-in. The operator grants exact owner/HTTPS URL pairs in a local
JSON file and supplies signing credentials through named environment variables:

```json
[
  {
    "owner": "alice",
    "url": "https://receiver.example/hooks/a2a",
    "secret_env": "A2A_CALLBACK_SIGNING_SECRET",
    "bearer_env": "A2A_CALLBACK_RECEIVER_TOKEN"
  }
]
```

Add `--a2a-callbacks ./callback-grants.json` to `harness serve --a2a-url ...`.
The signing secret must contain at least 32 characters; `bearer_env` is optional.
No caller bearer token or other environment variables are forwarded. Inline
credentials are rejected. The agent card advertises push notifications only
when callbacks are enabled. A grant permits disclosure of that owner's task
status and final answer to the exact endpoint; only that owner can subscribe
their own tasks. Limits are 100 operator grants and 100 active subscriptions per
owner. Callback URLs cannot include credentials, query strings, or fragments.

With `A2A-Version: 1.0`, use `CreateTaskPushNotificationConfig`,
`GetTaskPushNotificationConfig`, `ListTaskPushNotificationConfigs`, and
`DeleteTaskPushNotificationConfig`. A config includes `taskId`, `url`, an optional
stable `id`, and an optional correlation `token`. Alternatively include
`configuration.taskPushNotificationConfig` in `SendMessage`: the subscription
and executable task commit atomically. The message's callback configuration is
part of its idempotency identity. Reusing an existing subscription ID requires
identical settings; after deletion, choose a new ID.

The registration's negotiated version is persisted. Version 1.0 notifications
use the official `StreamResponse` JSON shape with a `task` snapshot; version 0.3
notifications use the SDK's direct `Task` shape (`kind: task`, lowercase status).
Each POST carries its corresponding `A2A-Version`. Legacy clients can register
with `message/send` plus `configuration.pushNotificationConfig`, or use
`tasks/pushNotificationConfig/set`, `/get`, `/list`, and `/delete`. CRUD requests
must use the registration's version, and list results are scoped to that version;
changing versions requires a new subscription ID. Payload bytes and version
remain fixed across retries and restarts. The SDK 1.1.2 compatibility adapter
reports handler validation errors on 0.3 requests as `-32603` while preserving
the explanation.
Subscribers receive recorded status transitions, including those preceding a
late subscription, in order per subscription. Final artifacts appear only on
completed states. Continuations and HTTP resumes remain attached to the same
owned task. Artifacts exceeding the 24 MiB callback limit are omitted with an
explanation and remain accessible through the owned task/session API.

Every POST includes `X-A2A-Signature`: the lowercase hexadecimal HMAC-SHA256 of
the **exact raw request body**, using the configured secret, without a prefix.
Bodies use sorted keys and UTF-8 JSON with the default JSON spacing and unescaped
Unicode (`json.dumps(payload, sort_keys=True, ensure_ascii=False)`), so receivers
that parse and recanonicalize using the pinned convention also verify correctly.
This matches [the pinned Hermes signing implementation](https://github.com/NousResearch/hermes-agent/blob/939e45c91d751fadd94dcd1b873ac3cb44846213/plugins/platforms/a2a/security.py).
Verify the signature with a constant-time comparison before parsing/using the
body. The stable `X-A2A-Delivery-ID` is also inside the signed task's
`metadata.harness_delivery_id`; persist that signed ID to reject duplicate
processing. The optional correlation token uses `X-A2A-Notification-Token`.
Configured receiver credentials use `Authorization: Bearer ...`. JWT/JWKS
authentication and automated secret rotation are not implemented.

Before each attempt, all resolved addresses must be public unicast addresses.
Harness connects to a selected numeric address while retaining the original
Host header and TLS hostname verification through the
[HTTPCore SNI extension](https://www.encode.io/httpcore/extensions/). This pins
the validated peer for the request; redirects and environment proxies are
disabled. Private, mixed public/private, and multicast DNS results fail closed.
HTTPS transport is still bounded to 15 seconds including DNS and no response
body is consumed.

Delivery records, payloads, event cursors, and attempt counts survive restart.
HTTP 2xx acknowledges delivery; 429, 5xx, connection errors, and lost
acknowledgements retry with bounded backoff, up to five attempts. Other statuses,
including redirects, fail permanently. Restart recovers interrupted attempts
with the same signed payload and delivery ID. This is **at least once** delivery
with bounded retries: duplicate notifications are possible after an uncertain
POST, and retries can eventually fail. It never repeats agent execution. A
failed delivery permits later transitions to proceed. Removing the operator
grant prevents future sends; deleting a subscription cancels pending work,
while an already claimed/in-flight POST may still reach the receiver.

Authenticated `GET /v1/a2a/callback-deliveries` shows only the caller's most recent
100 delivery states, attempt counts, and safe errors, plus a generic worker
health error. It excludes signing secrets, bearer values, and payloads.

The offline acceptance suite runs the official SDK client and JSON-RPC/SSE
routes against an ASGI app, fake model adapters, real Harness Agents, and real
SQLite. It checks ownership, 0.3 compatibility, media, transaction rollback,
concurrent retries, restarts, approval continuation, cancellation, and reconnects.
Callback tests use HTTPX MockTransport with numeric address/SNI assertions,
exact signatures and Unicode canonicalization, both callback wire versions,
foreign-owner rejection, atomic rollback, ordered retry,
lost acknowledgements, recovery, deletion, and private-address rejection.
