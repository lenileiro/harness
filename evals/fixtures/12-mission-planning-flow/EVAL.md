primary_dimension: decomposition

trap: >
  A weak model will fall back to the manual `mission plan` flags, skip the
  generated validation contract, leave assertions uncovered by features, or
  forget to approve the plan before treating it as runnable.

correct_fix: >
  Create a mission, use `mission draft-plan --apply` to generate milestones,
  assertions, and feature coverage, inspect the resulting contract and
  structure, approve it, and verify the persisted state with the fixture tests.

dimensions: decomposition, correctness, verification, scope, overall

scoring_notes: >
  decomposition: 5 if the agent creates a coherent two-milestone plan with
  dependencies and assertion coverage through `mission draft-plan --apply`
  before approval.
  correctness: 5 if the mission, milestones, features, and contract match the
  requested values and the mission ends approved.
  verification: 5 if the agent runs the tests. 1 if it does not.
  scope: 5 if it only mutates mission artifacts under `.harness/missions`.
