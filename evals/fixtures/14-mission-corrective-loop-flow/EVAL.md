primary_dimension: decomposition

trap: >
  A weak model will validate the mission once, see the corrective feature, and
  either stop there or forget to finish both the original feature and the
  corrective follow-up before re-running validation.

correct_fix: >
  Create and approve the mission, dispatch the first feature, trigger validator
  failure, inspect the findings, complete the original and corrective features,
  re-run validation, and confirm the mission completes.

dimensions: decomposition, correctness, verification, scope, overall

scoring_notes: >
  decomposition: 5 if the agent follows the corrective loop in the intended
  order and uses the created corrective feature rather than bypassing it.
  correctness: 5 if the mission ends completed with validated original and
  corrective features plus persisted findings from the first failed validation.
  verification: 5 if the agent runs the tests. 1 if it does not.
  scope: 5 if it only mutates mission artifacts under `.harness/missions`.
