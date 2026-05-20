# tasks

Durable tasks + activity log for the Harness agent runtime.

A `Task` is the long-lived unit of agent work: one task → many agent runs
(sessions). Tasks carry status, priority, labels, links to other tasks, and
an embedded list of session ids that worked on them. The `ActivityEvent`
ledger captures every significant runtime occurrence (tool calls,
approvals, errors) in append-only fashion for audit + replay.

Implementations of `TaskStore` and `ActivityStore` ship in
`storage-memory` and `storage-sqlite`.
