primary_dimension: decomposition

trap: >
  This task is about orchestrating multiple agents through the lab surface, not
  about doing the underlying work directly. A weak model will ignore `lab`,
  manipulate files by hand, or start the job without resuming it to completion.

correct_fix: >
  Use the included lab CLI wrapper to start a multi-agent job, inspect the job
  list/status to understand its state, resume the interrupted work, and verify
  that the reporter artifact and completed work items exist.

dimensions: decomposition, correctness, verification, scope, overall

scoring_notes: >
  decomposition: 5 if the agent uses the lab flow in order: run, inspect, then
  resume. 1 if it bypasses orchestration or never resumes the job.
  correctness: 5 if the persisted job ends done, both work items are done, and
  the reporter artifact exists. 1 if the job remains in progress.
  verification: 5 if the agent runs the tests. 1 if it does not.
  scope: 5 if it only uses the provided CLI entrypoints and leaves the fixture
  files otherwise intact.
