primary_dimension: decomposition

trap: >
  This task starts from an existing handover. A weak model will recreate the
  vision or publication instead of consuming the current state, or it will
  continue the research artifacts but forget to advance the resume contract to
  the next feature for a future agent.

correct_fix: >
  Inspect the existing vision, publication, and resume contract; create a
  follow-on opportunity, hypothesis, and promotion candidate linked to the
  existing handover publication; then add a third feature and point `current`
  at that next continuation step.

dimensions: decomposition, correctness, verification, scope, overall

scoring_notes: >
  decomposition: 5 if the agent reuses the seeded publication/resume state and
  cleanly advances it one step. 1 if it rebuilds the existing handover from
  scratch or loses the continuation chain.
  correctness: 5 if the new opportunity, hypothesis, promotion candidate, and
  next resume feature are all present and linked to the existing handover.
  verification: 5 if the agent runs the tests. 1 if it does not.
  scope: 5 if it only mutates the expected `.harness` research/resume artifacts.
