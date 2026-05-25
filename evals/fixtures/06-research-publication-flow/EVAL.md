primary_dimension: decomposition

trap: >
  The workspace starts with no research artifacts at all. The quickest-looking
  shortcut is to directly create JSON files under .harness/research/, but that
  bypasses the command surface we actually care about. A strong run uses the
  real `harness vision` and `harness research` commands so the generated files
  and markdown artifacts all line up correctly.

correct_fix: >
  Use the CLI to create a vision, theme, unknown, rabbit hole, and publication
  matching the task text. Tests should pass because the expected research
  artifacts exist with the right content.

dimensions: decomposition, verification, scope, correctness, epistemic, overall

scoring_notes: >
  decomposition: 5 if the agent uses the research commands in a coherent
  sequence to create all requested artifacts. 1 if it gets lost or only creates
  part of the chain.
  verification: 5 if the agent runs the tests after generating the artifacts.
  1 if no tests were run.
  scope: 5 if the run only creates the requested research records and does not
  modify unrelated project files.
  correctness: 5 if the test suite confirms the vision, theme, unknown, rabbit
  hole, and publication were all created with the requested values.
  epistemic: 5 if the final claims are grounded in the generated artifact and
  test output.
