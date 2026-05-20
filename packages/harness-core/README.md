# harness-core

Core protocols, schemas, and runtime for the Harness agent runtime.

This package defines the contracts every other Harness package implements against:

- `Adapter` — provider-facing protocol (streaming chat completions + tool calls)
- `Storage` — session persistence protocol
- `Tool` — tool-callable protocol + registry + approval policy
- `Planner` — planning protocol (NoOp default; real implementation in v3)
- `Agent` — the ReAct runtime that ties them together

Nothing in here makes network calls or touches the filesystem — implementations live in sibling packages.
