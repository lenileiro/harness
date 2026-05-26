primary_dimension: decomposition

trap: >
  This task starts from a seeded mission handoff. A weak model will recreate the
  mission instead of consuming the stored planner handoff, or it will complete
  the feature but forget to validate the milestone and leave a reporter artifact.

correct_fix: >
  Inspect the existing mission, contract, and planner handoff; complete the
  handed-off feature, validate the milestone, write a mission summary report,
  and verify the persisted state with the fixture tests.

dimensions: decomposition, correctness, verification, scope, overall

scoring_notes: >
  decomposition: 5 if the agent cleanly resumes from the seeded handoff and
  completes the remaining steps in order.
  correctness: 5 if the feature becomes validated, the mission completes, and a
  mission summary report is written.
  verification: 5 if the agent runs the tests. 1 if it does not.
  scope: 5 if it only mutates mission artifacts under `.harness/missions`.
