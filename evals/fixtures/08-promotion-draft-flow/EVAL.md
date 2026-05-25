primary_dimension: decomposition

trap: >
  This fixture is not about editing production code or opening a real branch.
  The correct outcome is a complete research-to-promotion artifact chain inside
  the workspace, driven by CLI commands and connected by the generated IDs. A
  weak agent will create partial research records and stop before
  `research promote` / `research pr`, or it will hand-edit the JSON.

correct_fix: >
  Use the research CLI to create the vision, theme, rabbit hole, publication,
  opportunity, hypothesis, and promotion candidate. Then generate both
  `promotion_draft.json` and `PR_BODY.md` with `research promote` and
  `research pr`, without direct edits to `.harness/research/*.json`.

dimensions: decomposition, correctness, verification, scope, overall

scoring_notes: >
  decomposition: 5 if the agent follows the dependency chain cleanly and uses
  generated IDs to connect the records. 1 if it stalls before the promotion
  draft or bypasses the CLI.
  correctness: 5 if the resulting promotion candidate and PR draft artifacts
  contain the requested values and evidence checklist. 1 if the draft artifacts
  are missing or disconnected.
  verification: 5 if the agent runs the tests. 1 if it does not.
  scope: 5 if it only creates the research artifacts required for the flow. 1
  if it edits unrelated files or attempts unnecessary repo changes.
