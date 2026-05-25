primary_dimension: correctness

trap: >
  This fixture is about real workspace plugin loading, not just writing a TOML
  file. A weak agent will stop after creating a manifest or only run
  `plugins list`. The correct path is to create a loadable Python provider,
  validate it through the CLI, and leave behind a provider that can actually be
  imported and queried by the test suite.

correct_fix: >
  Add `.harness/plugins/workspace-experience.toml` with the required manifest
  values and implement `workspace_plugin.py` using `from
  harness.core.tips_models import Tip` and the exact query behavior described
  in TASK.md. Run `harness plugins validate --kind experience` and then the
  tests.

dimensions: correctness, decomposition, verification, scope, overall

scoring_notes: >
  correctness: 5 if the provider is loadable and returns the expected tip for a
  plugin query. 1 if the manifest is invalid or the provider cannot be loaded.
  decomposition: 5 if the agent understands the manifest + provider +
  validation sequence and completes it without wandering. 1 if it gets stuck in
  help loops or only edits one side of the plugin.
  verification: 5 if the agent runs the CLI validation and the tests. 1 if it
  skips them.
  scope: 5 if only the two workspace plugin files are added. 1 if unrelated
  workspace files are created.
