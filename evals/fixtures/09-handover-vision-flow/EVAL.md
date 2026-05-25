primary_dimension: decomposition

trap: >
  This task is not just "create some research records." The key behavior is
  creating a continuation path for the next agent. A weak model will create the
  vision or publication but forget to update the resume contract, or it will
  leave the current feature pointing at the work that was just completed. The
  correct path ends with a next-agent-ready resume state plus research artifacts
  that explain what should continue.

correct_fix: >
  Use the CLI to create the vision, theme, unknown, rabbit hole, and
  publication, then initialize the resume contract, add the next feature, and
  point `current` at `handover-consumption`. The publication and resume
  contract together must make the continuation path explicit.

dimensions: decomposition, correctness, verification, scope, overall

scoring_notes: >
  decomposition: 5 if the agent follows the dependency chain cleanly, including
  generated IDs and the final handover step in the resume contract. 1 if it
  stops after only the research records or only the resume file.
  correctness: 5 if the resume contract points to the next feature and the
  publication captures the expected continuation guidance. 1 if the current
  feature still points at the completed work or the handover context is
  incomplete.
  verification: 5 if the agent runs the tests. 1 if it does not.
  scope: 5 if it only creates the required `.harness` research/resume artifacts.
