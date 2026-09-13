# Skills, recall, MCP, and scheduled prompts

Harness now supports portable skills, scoped durable memory, conversation search,
managed MCP clients, and approval-aware scheduled prompts. These implement the
reusable capability layer of the [Hermes parity implementation](feature-parity.md).
That inventory links the channel, media, execution and desktop implementations.

## Setup and diagnostics

```sh
harness setup --provider ollama --model YOUR_MODEL
harness doctor --cwd /path/to/workspace
harness doctor --json
```

Setup writes the default TOML configuration with private file permissions. An
existing file requires `--force`; that updates provider/model while preserving
other settings and comments. Provider credentials remain in their existing
environment or credential store. Diagnostics report credential presence, config
errors, dependencies, writable paths, skills, and MCP configuration without
making model calls. `doctor --connect` explicitly initializes configured MCP
servers. Ollama availability and API credentials are not reported as verified
merely because configuration exists.

## Portable skills

```sh
harness skills create release-check --description 'Review release metadata and changes'
harness skills install /path/to/release-check
harness skills list
harness skills show release-check
harness skills validate
```

Edit `.harness/skills/release-check/SKILL.md` to author the instructions. Packages
use the [Agent Skills format](https://agentskills.io/specification), including
YAML frontmatter with `name` and `description`, and optional `scripts`,
`references`, and `assets` directories. Installation copies an explicitly chosen
local package; it does not execute scripts or overwrite an installed skill.

Discovery precedence is configured `skills.paths`, `.harness/skills`,
`.agents/skills`, then `~/.harness/skills`. The first valid package with a given
name wins. Invalid and shadowed packages appear in `skills list`; invalid
packages make `skills validate` fail. Use `--cwd` and `--config` to inspect the
same workspace and configuration as a run.

```toml
[skills]
enabled = true
paths = ["/path/to/shared/skills"]
```

The runtime initially includes names and descriptions. The `skill_read` tool
loads a skill or a relative supporting text file on demand. In chat, `/skill`
lists packages and `/skill release-check` explicitly activates one. Activations
are saved with the conversation and restored after resume and context pruning.
The ten most recently activated names are retained, with up to 128 KiB of whole
skill bodies injected; omitted bodies can be read again. Individual text reads
are limited to 128 KiB, installation to 16 MiB/1,000 files, and file reads cannot
traverse symlinks or escape the skill directory. `allowed-tools` metadata is
guidance and never grants tool permission. Existing tips and procedures still
load through their original interfaces.

Configured context budgets reserve space for instructions, skills, memory and
tool schemas before pruning history. If those inputs alone exceed the budget,
the run fails with recovery instructions. Token counts are approximate, and
protected transcript blocks retain the existing soft-budget semantics.

## Durable memory and conversation recall

```sh
harness recall add 'Use uv for this workspace' --kind project_fact
harness recall list
harness recall get MEMORY_ID
harness recall update MEMORY_ID 'Run tests with uv run pytest'
harness recall delete MEMORY_ID
harness recall search 'release verification'
harness recall window SESSION_ID --reference MESSAGE_REFERENCE --before 2 --after 2
```

`recall search` searches conversation text and returns session IDs, excerpts,
stable message references and update times. `recall window` and the model's
`conversation_window` tool retrieve bounded adjacent context in the same scope.
It uses SQLite FTS5, refreshes the index on save, removes it on
session deletion, and backfills old conversations transactionally. The model has
`recall_memory` (list/search/get), `memory` (add/list/search/get/update/delete),
and `search_sessions` tools. `recall_memory`, `search_sessions` and
`conversation_window` are read-only, so conversational agents can gather scoped
context without requesting a write approval. Memory mutations retain the
existing approval policy. Managed children receive these read-only tools for
their own workspace and identity; they cannot retrieve a parent's private memory
or conversation just because the parent created them.

For local CLI runs, scoped memory writes are automatic by default. Set
`memory = "prompt"` or `memory = "deny"` in `[approval]` to change that behavior.
Remote conversational agents use the stricter gateway policy, which pauses
memory mutations for approval. The read-only `recall_memory` tool cannot add,
update or delete facts under either policy.

Local `run --session` and `chat --session` use the saved workspace when `--cwd`
is omitted, including when launched from another directory. They retain the
already-selected database. A conflicting explicit `--cwd`, unavailable saved
workspace or nonlocal session identity stops before model and tool construction;
resuming does not reassign a conversation to another workspace or user.

Scope is bound by the caller, never supplied in model tool arguments. Local
commands use the resolved workspace. Gateway sessions also bind the authenticated
transport/user identity, so one user's memories and transcript results cannot
be retrieved by another user. Resuming a session under a conflicting identity
fails. Persistent fact injection is limited to 20 entries and 16,000 characters;
session notes remain a separate scratchpad. SQLite now saves notes and phases.

Legacy memory is assigned a workspace only when a linked session or the
workspace-local database location establishes ownership. Ambiguous global
records remain in storage through the legacy SDK interface; they are not
guessed into a new user's scope. Existing `memory` CLI commands now use local
workspace scope. Use the same `--db` and `--cwd` when inspecting custom databases.

## MCP clients

```toml
[mcp.servers.local]
transport = "stdio"
command = "/absolute/path/to/python"
args = ["/absolute/path/to/server.py"]
include_tools = ["lookup"]
env_from = { SERVICE_TOKEN = "MY_SERVICE_TOKEN" }

[mcp.servers.remote]
transport = "streamable-http"
url = "https://example.com/mcp"
bearer_token_env = "MY_MCP_TOKEN"
expose_resources = true
expose_prompts = true
```

```sh
harness mcp list
harness mcp check --cwd /path/to/workspace
```

Servers initialize before an agent turn, including before replaying approved
actions. Discovered tools are registered for the run and removed when it ends;
connections and owned subprocesses close on success, error, and cancellation.
Tool names include server namespaces. Inclusion/exclusion lists, resource and
prompt discovery, structured/text results, and explicit error results are
supported. Credentials are resolved from named environment references. Stdio
inherits the SDK's minimal OS environment plus explicit variables; HTTP does
not follow redirects or inherit proxy credentials.

Tools require approval by default. Server-provided read-only hints cannot grant
permission. Transport loss never triggers automatic repetition of a possibly
executed action. OAuth browser login with PKCE, serialized refresh and private
profile storage is available through `harness mcp login SERVER`; binary image,
audio and file results use the shared attachment schema. MCP serving is a
separate authenticated [server surface](../packages/server/README.md).
Codex must use `mode = "app-server"` to dispatch Harness MCP tools through its
approval loop; legacy exec mode cannot. See the [MCP package guide](../packages/tools-mcp/README.md)
for full configuration and SDK usage.

## Remote approvals and scheduled prompts

Conversational gateway agents and scheduled prompt agents pause mutations and
send the exact requested action to the
bound conversation. Reply `approvals`, `approve APPROVAL_ID`, or
`deny APPROVAL_ID`. Identity, thread, expiry, and resolution state are checked.
Concurrent and duplicate replies cannot execute the same approved call twice.
The approval expires after 15 minutes. The original provider/model/session are
retained across gateway restart.

An execution claim is persisted before replay. If the process stops after the
claim and before saving the outcome, the result is uncertain and Harness stops
automatic replay. Inspect the external system before manual recovery; this is
not an exactly-once guarantee for arbitrary external side effects. Gateway
native Codex tools and ephemeral inline delegation are unavailable until they
can use this durable approval path. Remote filesystem tools cannot read, list,
or change `.harness` private state, including through symlinks. Remote host
`shell` and `verify_work` execution are unavailable until an isolated execution
backend is provided; approved local scheduled jobs retain those tools. These
application boundaries are not an OS sandbox: custom plugins and configured
MCP servers remain trusted integrations with their own host/service access.
Legacy workflow/research control commands remain privileged host operations for
trusted operators. They do not run through these conversational approval and
filesystem boundaries; the gateway is not a sandbox for mutually untrusted users.

```sh
harness scheduler add-prompt --prompt 'Review the latest project changes' \
  --provider ollama --model YOUR_MODEL --every 1h --cwd /path/to/workspace
harness scheduler list --cwd /path/to/workspace
harness scheduler start --cwd /path/to/workspace
```

Schedules accept `--at`, `--every`, or five-field `--cron`, with `--timezone`
selecting an IANA timezone (UTC by default). Ambiguous/nonexistent local `--at`
times require an explicit offset; cron handles DST through UTC occurrence scans. Prompt jobs keep
a durable session and fixed provider/model. Without a target, they use the local
approval inbox; inspect/grant with `harness approvals` and continue with
`harness sessions resume`. Supply `--transport`, `--user`, and `--thread`
together to bind a remote result/approval destination. Only channels with an
installed outbound implementation can deliver messages. Waiting approvals stop
new model work on later ticks. Each run records result text and approval IDs;
remote results enter the existing durable notification outbox.

The scheduler runs in the foreground or under the [service manager](profiles-and-maintenance.md).
External model calls, real accounts,
and message delivery require separate live verification; automated tests use
real local storage/processes and fake providers/transports.
