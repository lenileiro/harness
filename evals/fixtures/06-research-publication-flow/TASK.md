# Use the research CLI to seed a publication

Use the built-in Harness research commands in this workspace to create a minimal
research record for a plugin investigation.

Use these CLI entrypoints, not direct file edits:

- `harness vision update`
- `harness research add-theme`
- `harness research create-unknown`
- `harness research open`
- `harness research publish`

Do all of the following:

1. Set the current vision title to `Autonomous Harness Research` and the
   summary to `Build a compounding research loop for Harness.`
2. Add a theme titled `Plugin reliability`.
3. Create an unknown under that theme asking: `How should workspace plugins be validated?`
4. Open a rabbit hole titled `Workspace plugin import flow`.
5. Publish one research publication titled `Workspace plugin validation findings`.

You will need the generated theme ID and rabbit-hole ID for later steps. Read
the generated JSON files under `.harness/research/...` to get those IDs between
commands.

Do not hand-write `.harness/research/*.json` files directly. Use the CLI.

Run the tests when done.
