primary_dimension: decomposition

trap: >
  A weak model will either keep all context thread-bound, causing the second
  chat to lose continuity, or merge raw transcripts across chats and recreate
  the earlier "old messages come back" failure mode.

correct_fix: >
  Introduce a durable shared user profile or equivalent user-level memory,
  store active work separately from thread-local chat history, update gateway
  actions to populate that shared memory, and make `gateway converse` resolve
  thread context plus shared user context before calling the model.

dimensions: decomposition, correctness, verification, scope, overall

scoring_notes: >
  decomposition: 5 if the solution cleanly separates user-level shared context,
  thread-local context, and active work references instead of stuffing all data
  into one transcript blob.
  correctness: 5 if a second thread for the same user can inherit relevant
  active work or recent thread summaries without replaying unrelated transcript
  history.
  verification: 5 if focused deterministic tests cover both shared-work
  population and cross-thread conversation reuse. 1 if the agent does not run
  the requested tests.
  scope: 5 if the work stays in the gateway router/session/conversation layer
  and the targeted tests, without broad unrelated refactors.
