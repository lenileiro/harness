# Orchestrate a multi-agent job and resume it to completion

This fixture includes a local Harness CLI wrapper at `./fake-harness`. Use that
entrypoint for all commands in this task.

Use CLI entrypoints only. Do not hand-edit the SQLite database or fixture files
directly.

Use these commands:

- `./fake-harness lab run`
- `./fake-harness lab list`
- `./fake-harness lab status`
- `./fake-harness lab resume`

Do all of the following:

1. Start a multi-agent job with `./fake-harness lab run`.
2. Inspect the stored jobs with `./fake-harness lab list`.
3. Inspect the created job with `./fake-harness lab status`.
4. Resume the same job with `./fake-harness lab resume`.
5. Run the tests when done.

Use these exact command shapes:

- `./fake-harness lab run "Plan, execute, and summarize the seeded work" --cwd . --db lab.db --no-judge`
- `./fake-harness lab list --db lab.db`
- `./fake-harness lab status <job-id> --db lab.db`
- `./fake-harness lab resume <job-id> --db lab.db --no-judge`

Important:

- The job id is created by the run step. Discover it from `lab list` or the
  persisted database state before calling `lab status` / `lab resume`.
- Do not bypass the orchestration flow by creating the reporter artifact
  yourself. The lab resume step should leave the final report behind.

Run the tests when done.
