# Questions and continuation

Harness agents gather context and research autonomously by default. They inspect
the repo, documentation, scoped history and memory, use available research tools,
and proceed with supported reversible assumptions. Clarification is disabled by
default; ordinary implementation choices do not require a human response. See
[autonomy policy](autonomy-policy.md).

For explicitly interactive workflows, the optional `clarify` tool lets an agent
ask for essential unavailable information without losing its place. It supports
one to five questions per request, up to four choices per
question, multiple selections and free text. A custom answer is always accepted
when choices are offered. The first choice is displayed as recommended.

Questions and action approvals have separate stores and controls. An answer is
input to the conversation; it cannot grant a tool permission. A completed answer
may still leave the conversation waiting for an outstanding action approval.

## Local chat and CLI

Local managed agents register the question tool only when configured explicitly:

```toml
[clarification]
enabled = true
```

The default is `false`. The terminal
shows each unanswered question and its ID. In an active chat:

```text
/questions
/answer QUESTION_ID 1
/answer QUESTION_ID {"q1":"Engineering team"}
/skip-question QUESTION_ID
```

Plain text after the ID answers the next unanswered question. Choice numbers are
one-based; multiple selections can use comma-separated numbers or labels. A JSON
object answers specific questions by their zero-based keys (`q0` through `q4`).
Accepted partial answers remain saved. Chat continues the original session once
the form is complete or skipped.

For a paused noninteractive run, use the same workspace and database:

```sh
harness clarify list --cwd /path/to/project
harness clarify show QUESTION_ID --cwd /path/to/project
harness clarify answer QUESTION_ID 'Report' --session SESSION_ID --cwd /path/to/project
harness clarify cancel QUESTION_ID --session SESSION_ID --cwd /path/to/project
harness sessions resume SESSION_ID "Continue with the recorded answers" --cwd /path/to/project
```

Supply the same `--db` when the original session used a custom database. These
commands inspect local workspace questions only; channel and server identities
use their own authenticated controls. The answer/cancel commands save input and
do not start a model call. Resume is explicit.

## Channels, HTTP and MCP

Channel conversations accept `/questions`, `/answer ID TEXT_OR_JSON` and
`/skip-question ID`. Questions are bound to the original workspace, transport,
user, conversation and runtime session. A user in another thread cannot answer
them. Continuation retains the original provider/model and remote tool policy.
Pending questions prevent model switches, transcript mutation and handoff until
the original session has consumed the answer or skip result.

When enabled, Telegram, Discord, Slack and Google Chat also render native
choices. Controls come from the persisted question record and use opaque tokens
bound to the original owner, thread and sent message. Partial selections survive
restart. Custom text and the slash controls remain available; a button never
grants an action approval or creates a question on its own.

The HTTP service exposes `clarify` only when the operator selects it:

```sh
harness serve --expose-tool clarify
```

The authenticated web/desktop client has a **Questions** view with choice inputs,
custom text, saved partial answers and a separate **Continue conversation**
button. API clients use:

| Operation | Request |
| --- | --- |
| List owned questions | `GET /v1/questions` |
| Save answers | `POST /v1/questions/ID/answer`, body `{"answers":{"q0":"Report"}}` |
| Skip unanswered questions | `POST /v1/questions/ID/cancel` |
| Continue the original run | `POST /v1/runs/RESUME_RUN_ID/resume`, body `{}` |

The list includes `resume_run_id`. Answers can be partial, but accepted values
cannot be replaced with different values. Resuming an unanswered form returns
409 before queueing another run. Another identity receives no question details.
MCP clients can use the same operations when explicitly exposed as
`questions`, `answer_question` and `cancel_question`; their normal MCP identity
and run-resume controls apply.

## Delegated jobs

Children keep separate questions. Job status and the `delegate_status` model
tool include their pending forms. For the local delegation CLI:

```sh
harness delegate --workspace . --expose-tool clarify submit "Prepare the report"
harness delegate --workspace . questions JOB_ID
harness delegate --workspace . answer JOB_ID QUESTION_ID '{"q0":"Report","q1":"Team"}'
harness delegate --workspace . skip-question JOB_ID QUESTION_ID
harness delegate --workspace . --expose-tool clarify resume JOB_ID
```

Keep the same `--database` and `--owner` if specified originally. A question
must belong to the selected job. Inspection and answers do not build an adapter,
reserve another attempt or resolve child approvals. Resume checks unanswered
questions before charging the parent's shared budget.
The local CLI normally provides a read-only discovery preset. Explicit
`--expose-tool` flags replace that preset, so include any read/research tools the
interactive child should use as well as `clarify`.

## Recovery

Questions expire after 15 minutes by default. Expiry or skipping retains accepted
answers and produces an explicit `timed_out` or `cancelled` result for any later
continuation. Run cancellation is separate and invalidates its unapplied questions.

SQLite retains the original tool-call ID, form, answers and consumption state.
Continuation updates that original tool result before marking the question
consumed; a restart does not add a duplicate answer message or silently repeat
the question. Missing or mismatched question state blocks continuation. Forking
an unresolved session is refused because question records belong to one session.
Database waits run off the event loop so answering does not block unrelated API
work.

Acceptance includes real Agent/SQLite pause, partial answer, expiry, cancellation,
restart and ownership tests; CLI and delegated-child workflows; authenticated
HTTP/MCP fixtures; and actual Chromium checks at desktop and mobile sizes.
