# Remote A2A agents

Harness can serve an authenticated A2A endpoint and call explicitly configured
remote agents. Inbound serving, owned tasks, streaming and callback configuration
are documented in [the server guide](../packages/server/A2A.md). Outbound tools
support JSON-RPC A2A 1.0 and the 0.3 compatibility protocol.

```toml
[a2a]
enabled = true
max_parallel = 3
max_context_turns = 20
max_response_bytes = 1048576
poll_interval = 0.5

[a2a.peers.researcher]
url = "https://research.example.org/a2a"
token_env = "RESEARCH_AGENT_TOKEN"
capabilities = ["research", "summarize"]
protocol = "1.0"
timeout = 120

[a2a.peers.reviewer]
url = "http://127.0.0.1:8765/a2a"
token_env = "REVIEW_AGENT_TOKEN"
capabilities = ["review", "research"]
protocol = "0.3"
```

Credentials are environment-variable references, including values stored through
`harness auth set` in the selected profile. URLs require HTTPS except for explicit
loopback HTTP. Discovery reads `/.well-known/agent-card.json`, with the older
`/.well-known/agent.json` alias tried only after a 404. An advertised JSON-RPC
endpoint must remain on the configured origin. Redirects are not followed and
credentials are never forwarded to another origin. Discovery does not add peers
or expand the operator's capability labels.

The local CLI's managed agent exposes these tools:

| Tool | Behavior |
| --- | --- |
| `a2a_list` | List configured names/capabilities and the current owner's saved conversations. |
| `a2a_discover` | Inspect one configured Agent Card. |
| `a2a_call` | Send a bounded prompt; an optional `context_id` continues an owned conversation. |
| `a2a_history` | Read saved prompts, results and states for that conversation. |
| `a2a_status` | Refresh a known remote task without resubmitting its prompt. |
| `a2a_cancel` | Request remote cancellation; completed effects can remain. |
| `a2a_orchestrate` | Send a prompt to peers matching a configured capability, or `*`. |

Sending, orchestration and cancellation use the normal external-action approval
boundary. These integrations are not injected into remote gateway runtimes or
restricted child agents. Read-only runtimes expose only the read tools. Remote
Agent Card descriptions and returned text remain tool data, not instructions
that can change local approval policy.

Orchestration has three modes: `all` returns every result; `first` returns when a
peer completes successfully; `best` chooses the longest successful text. This
last rule matches the pinned reference's selection heuristic and is not a
quality assessment. Concurrency is bounded by `max_parallel` (at most six).
`first` stops waiting for the other peers, but a submitted remote job can keep
running. Its saved conversation remains available through `a2a_list` and
`a2a_status`; use an approved `a2a_cancel` to request cancellation explicitly.

Conversation history is stored in the profile's private
`integrations/a2a.sqlite3`, scoped to workspace and authenticated memory owner.
A unique send claim is committed before network submission. Replaying the same
session/tool-call ID cannot silently submit twice, including after restart.
Changing a peer's URL or credentials prevents reusing its previous remote
contexts; the old local history remains readable by its owner.

Working tasks retain their remote handle when polling times out or local work is
cancelled. A lost submission response is recorded as `uncertain`. An abrupt
process exit can leave a `sending` claim. Neither state means the remote work
failed, and neither is automatically resubmitted. Inspect the peer before
starting another request. A new conversation turn clears any previous completed
task handle before submission, so a lost new response cannot be confused with
an old completed task. `input_required` replies continue their existing task.

Responses and stored text are bounded and credential-redacted. Tool output is
text/JSON; remote binary artifacts, gRPC and REST transports are not implemented.
The server can still accept supported native media as described in its guide.
Tests exercise actual Harness/official-SDK A2A 1.0 and 0.3 servers, restart and
ownership boundaries, approval pause/resume, lost-response deduplication,
cancellation, discovery authentication and bounded multi-peer execution. No
external agent or paid provider was contacted.
