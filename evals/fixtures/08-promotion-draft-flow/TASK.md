# Build a promotion candidate and PR draft with the research CLI

Use the built-in Harness research commands in this workspace to create a small
promotion pipeline artifact set for a documentation-oriented improvement.

Use these CLI entrypoints, not direct file edits:

- `harness vision update`
- `harness research add-theme`
- `harness research open`
- `harness research publish`
- `harness research create-opportunity`
- `harness research hypothesize`
- `harness research refine`
- `harness research promote`
- `harness research pr`

Do all of the following:

1. Set the current vision title to `Promotion Workflow Hardening` and the
   summary to `Turn research findings into bounded promotion artifacts.`
2. Add a theme titled `Promotion reliability`.
3. Open one rabbit hole titled `Promotion evidence checklist`.
4. Publish one publication titled `Promotion evidence findings`.
5. Create one opportunity titled `Tighten promotion drafts`.
6. Create one hypothesis for that opportunity.
7. Refine one promotion candidate titled `Promotion draft evidence section`.
8. Generate its promotion draft with:
   - `harness research promote --candidate <id> --no-create-branch`
9. Generate its PR payload with:
   - `harness research pr --candidate <id> --no-push --no-open`

Use these exact content values:

- rabbit hole question:
  `What evidence should every promotion draft include?`
- publication claim:
  `Promotion drafts should include explicit evidence and validation sections.`
- opportunity summary:
  `Promotion candidates should consistently capture evidence, validation, and file scope.`
- hypothesis claim:
  `Explicit evidence sections will make promotion drafts easier to review.`
- candidate expected metric:
  `promotion drafts include evidence checklist coverage`

You will need generated IDs between steps. Read the JSON files under
`.harness/research/...` to get those IDs as needed.

Do not hand-write `.harness/research/*.json` files directly. Use the CLI.

Run the tests when done.
