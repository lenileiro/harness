# Hermes Agent parity implementation

Reference: [Nous Research Hermes Agent v0.21.2 / v2026.9.11](https://github.com/NousResearch/hermes-agent/tree/939e45c91d751fadd94dcd1b873ac3cb44846213), commit `939e45c91d751fadd94dcd1b873ac3cb44846213`. Reviewed and implemented on 2026-09-13 against Harness `099a8e2` plus the working implementation.

Parity means working user workflows, persistence, recovery, configuration and acceptance evidence. A transport label, generic protocol, plugin filename or available external service is not an implemented integration. This inventory separates implemented code from live account and operating-system acceptance. It does not claim a percentage or full conformance to every upstream plugin feature.

## Milestone 1: reliable existing workflows

Implemented and regression-tested:

- `/model` changes the actual model, recreates model-bound helpers and survives restart. Configured failover providers can select their own model.
- Run/chat/resume load consistent skills, memory, experience and contracts. Forked sessions retain conversation context.
- SQLite migrations, concurrent updates, durable notes/phases, scoped search, approval claims and lifecycle cleanup are covered by tests.
- Missions execute real workers and verification; promotion records point to actual measured evidence. Scheduler delivery failures remain retryable rather than marking work successful without delivery.
- Context compaction preserves the original on failure, empty output or larger summaries. Owned adapter summaries have a separate session and close their stream/process. Native-agent adapters without a tool-free interface do not summarize implicitly.
- Remote conversational mutations use the persisted approval ledger, exact user/thread/runtime ownership, expiry and replay claims. Host workflow administration remains a separate operator permission.
- Agents gather context and research autonomously. Shared instructions prefer scoped evidence and reversible assumptions; clarification is disabled by default. Generated context remains separate from the user's task and survives resume without adding false verification requirements.

See [execution and evidence](execution-and-evidence.md) and [approval policy](autonomy-policy.md).

## Milestone 2: reusable capability layer

| Capability | Implemented behavior | Evidence and limits |
| --- | --- | --- |
| Curated memory | Scoped model-facing CRUD, CLI inspection, bounded prompt injection and durable session notes. | Workspace/profile/remote-user boundary regressions. |
| Conversation recall | SQLite FTS5, stable message references, adjacent message windows and owner-scoped CLI/model search. | Migration/update/delete and exact ownership tests; missing references fail explicitly. |
| Autonomous discovery | Local source inspection, scoped read-only memory/history and caller-configured web research available to independent agents. | Human clarification is opt-in; [autonomy policy](autonomy-policy.md) explains assumptions, missing dependencies and approval boundaries. |
| Portable skills | SKILL.md discovery, precedence, activation, bounded resources and immutable pinned Git installation. | [Skills guide](skills-memory-mcp.md), [lifecycle and provenance](skill-lifecycle.md). |
| Learning and skill evolution | Real-run evidence hashes, reviewed proposals, new skill creation, version history, update/remove/rollback and reuse in independent sessions. | Failed outcomes retained; activation does not execute bundled scripts. |
| MCP client | Official SDK stdio/Streamable HTTP, namespaced tools, filters, resources/prompts, native image/audio content and profile-scoped OAuth/PKCE. | Real local SDK server, owned subprocess and HTTP OAuth discovery/exchange/refresh fixtures. Live third-party servers unverified. |
| Account providers | Named OpenAI-compatible endpoints, independent credentials/capabilities, explicit device login and serialized refresh rotation. | [Accounts and services](accounts-and-services.md); named-provider account flows use local protocol fixtures. |

## Milestone 3: connected interaction workflows

| Capability | Implemented behavior | Evidence and limits |
| --- | --- | --- |
| Channels | Native authenticated transport adapters, scoped inbox/outbox, deduplication, rate limits, retries and scheduler delivery. | The exact platform inventory and per-channel limits are in [channel parity](channel-parity.md). Accounts remain externally unverified. |
| Native interaction | Persisted single-action approval buttons and bounded run feedback; optional question controls only for explicitly interactive workflows. | [Approval cards](channel-approval-cards.md), [run feedback](channel-progress.md), [questions](clarification.md). Ownership, callback replay and progress recovery use local protocol fixtures. |
| Gateway controls | `/model`, `/retry`, `/undo`, `/compress`, `/usage`, `/whoami`, destination-bound `/handoff` and `/continue`. | [Handoff](cross-channel-handoff.md) copies explicit transcript context without inheriting memory or approvals. |
| Profiles/personas | Isolated config, workspace, state, credentials, skills and SOUL.md; selected profile propagates through CLI commands. | Common SDK credential stores default inside the profile; explicit credential paths can override them. This is identity management, not an OS sandbox. |
| Routines | General prompt jobs, at/every/cron, IANA timezones, DST validation, pause/resume, durable approval and delivery recovery. | Timezone/storage/CLI fixtures; unattended live delivery requires running channel workers. |
| Services | Install/start/status/logs/stop/restart/supervise, duplicate prevention, bounded crash restart and owned-process cleanup. | Inspectable systemd/launchd/Windows boot manifests; host boot registration was not changed. |
| Maintenance | Config-preserving setup/doctor, consistent SQLite backup, bounded verified restore to a new directory, explicit clean-tree update. | [Profiles and maintenance](profiles-and-maintenance.md); no self-update or deployment was run. |
| Terminal | Multiline editing, searchable durable history, paste handling, attachments, cancel/steer and retained managed sessions. | Offline public CLI and lifecycle fixtures. |

## Milestone 4: execution, tools and media

| Capability | Implemented behavior | Evidence and limits |
| --- | --- | --- |
| Local processes | Backend-consistent shell/files/process/verify, interactive processes, deadlines and descendant cleanup. | POSIX controlling PTYs; Windows Git Bash pipes and Job Objects. Native Windows acceptance is in CI; ConPTY is explicitly unsupported. |
| Managed backends | Local, Docker, SSH, Singularity, Modal, Daytona and Vercel Sandbox, explicit bounded input/output transfer and owned-resource teardown. | [Execution package](../packages/tools-execution/README.md). Actual disposable Docker test passed; SSH/Singularity/cloud accounts use protocol/SDK fixtures. No host fallback. |
| Browser | Owned Playwright or CDP sessions, DOM references, actions, tabs, screenshots, upload/download and cancellation cleanup. | [Browser package](../packages/tools-browser/README.md), actual local Chromium fixtures. |
| Computer | Explicitly enabled screenshot, pointer, keyboard and scrolling controls; cross-process exclusive ownership and native image artifacts. | [Computer package](../packages/tools-computer/README.md). Fake-driver tests; primary-monitor/ASCII boundaries and native GUI acceptance are documented. |
| Media | Versioned image/audio/file content, persistence/resume, provider capability checks, vision conversion, image generation, transcription and speech. | [Media package](../packages/tools-media/README.md). HTTP fixtures and artifacts; generation providers unverified live. |
| Codex tools | Official app-server dynamic tools route through the normal Harness loop, with approval, redaction, evidence and result submission. | [Codex bridge](../packages/adapter-codex/README.md). Installed Codex binary tested against a loopback provider; native MCP/fs/shell inheritance disabled. Legacy exec mode remains explicit and cannot serve remote tools. |
| Claude Code provider | An authenticated Claude Code subscription login drives the runtime as a model: native tools, MCP, skills and session persistence are disabled, Claude proposes structured tool calls, and Harness dispatches every one through its own registry, approvals and verifiers. | [Claude bridge](../packages/adapter-claude/README.md). Installed Claude Code binary tested live through the real adapter; proposals naming tools outside the supplied registry are rejected. Per-turn process groups are terminated on timeout. Media and `temperature` are unsupported on this path. |
| Delegation/orchestration | Durable child handles, private workspaces, explicit file transfer, budgets, cancellation, approvals and resumed results; batch and model tool dispatch use the same run service. | [Delegation/API](../packages/server/README.md), local subprocess/storage fixtures. Legacy spawned roles inherit the effective parent tools and policy, including backend replacements and wrappers; they cannot recreate host tools outside the configured execution context. |

## Milestone 5: complete product surfaces and integrations

| Capability | Implemented behavior | Evidence and limits |
| --- | --- | --- |
| Agent HTTP/MCP service | Authenticated owned runs/sessions, streaming, cancellation, restart recovery, batch operations and explicitly exposed tools. | [Server API](../packages/server/README.md); real loopback HTTP/MCP fixtures. |
| Web/desktop client | Chat, sessions, runs, approvals, optional questions, batches, delegation, preferences, tool schemas and durable scheduled prompts; optional native desktop shell. | Actual Chromium checks at desktop/mobile sizes; upload/navigation race regressions run against the actual client JavaScript. Native WebView installation remains platform-specific. |
| Data/training workflows | Validated JSONL archives, OpenAI/TRL/ShareGPT formats, schema validation, bounded pair-preserving compression, deduplication, sampling/splits and durable batches. | [Datasets and batches](../packages/cli/DATASETS_AND_BATCHES.md). No paid training run submitted. |
| Migration | Inspectable content-hashed OpenClaw import plan, current SQLite/WAL and legacy sources, selected static credentials, scoped memory/persona/skills and atomic new profile creation. | [Migration](../packages/cli/OPENCLAW_MIGRATION.md). OAuth credentials are not falsely converted to API keys. |
| Portal service bundle | Shared account-backed Firecrawl search/extraction, FAL generation, audio endpoints and Browser Use lifecycle. | [Account bundle](accounts-and-services.md), offline vendor protocol fixtures. |
| Honcho | Scoped workspace/peer/session export of selected conversation text, local reconciliation ledger and dialectic queries. | Explicit export approvals, uncertain-write deduplication and restart tests; inference quality/live service acceptance remain external. |
| Home Assistant | Allowlisted entity/service discovery, state reads and approved service calls through one explicit instance/account. | [Home Assistant package](../packages/tools-homeassistant/README.md), HTTP fixtures and real Agent approval test; no devices operated. |
| A2A federation | Official SDK agent serving, owned durable tasks, streaming and signed durable callbacks; approved outbound peer calls, persistent conversations, cancellation and bounded all/first/best orchestration. | [Peer setup](a2a-peers.md), [server protocol](../packages/server/A2A.md); actual local SDK 1.0/0.3 roundtrips, optional question continuation, callback retries and approval/recovery tests. |
| Platform packaging | Workspace manifests, optional dependencies and Linux/macOS/Windows contract CI configuration. | Linux/macOS/Windows are separate acceptance targets; a configured CI job is not evidence it has passed remotely. |

Channel attachment and signed A2A callback implementations are integrated. Matrix encryption uses real SDK cryptography in offline two-device fixtures; see [encryption setup](../packages/cli/MATRIX_E2EE.md). Exact protocol and platform limits remain in the linked inventories. Human clarification is deliberately optional in line with the requested autonomous workflow.

## Verification record

The final combined Python suite passed **3,035 tests**, with five explicitly gated integrations skipped, in 248.50 seconds. The owned Docker and installed-Codex loopback checks passed separately (**2 tests**), as did the live Claude test below; the two native Windows checks require their target OS. The actual web client passed **11 Node state regressions**. Ruff lint and formatting, Pyright, diff whitespace checks and evaluation asset validation passed. All **20 workspace packages** built both wheels and source distributions, with core and CLI artifacts rebuilt after the final audit fixes.

The opt-in [live Claude CLI test](claude-cli-validation.md) passed on the final implementation using installed Claude Code **2.1.270**: five `claude -p` model invocations produced four successful Harness discovery calls and the correct randomized answer and source, with no clarification request. The test took **21.97 seconds** and the CLI reported **$0.05708**. This test is skipped without explicit live-call configuration. A separate Claude audit led to fixes for spawned backend inheritance and cross-directory CLI resume; 25 new regressions cover those paths and configured memory approvals.

Local fixtures use real SQLite, subprocesses, sockets, TLS listeners and Chromium where applicable. Remote channel and cloud contracts use bounded fake services. The authorized Claude check exercised a real model through a test-only bridge; these results do not establish live account or operating-system acceptance beyond the environments described here.

## Remaining acceptance and reference differences

- Live platform conformance for the implemented media, cards, progress updates and encryption flows.
- Real account permissions, bot subscriptions, public webhook TLS routing and provider/cloud entitlements.
- Native Windows process ownership, Linux/macOS desktop permission behavior, WebView packaging and boot-service registration on their actual target OS.
- Third-party OAuth/MCP and model/media compatibility beyond the exercised SDK/HTTP contracts.

All five planned milestone areas now have working implementations and local
checks. Known implementation differences, including platform-specific
reaction/menu/media limits, remain explicit in the linked channel and integration
inventories. Those differences are separate from live account or OS acceptance;
local milestone completion is not certification of every upstream plugin feature.
