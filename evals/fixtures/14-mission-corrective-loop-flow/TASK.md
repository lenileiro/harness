# Drive a mission through a corrective validation loop

Use the built-in Harness CLI in this workspace to create and recover a mission
that fails validation on the first pass.

Use CLI entrypoints only. Do not hand-edit `.harness/missions/*.json` files
directly.

Use these commands:

- `harness mission create`
- `harness mission plan`
- `harness mission approve`
- `harness mission execute-next`
- `harness mission validate-milestone`
- `harness mission list-findings`
- `harness mission complete-feature`
- `harness mission summarize`

Do all of the following:

1. Create a mission titled `Mission Corrective Loop`.
2. Plan and approve a single-milestone mission with one feature and one assertion.
3. Dispatch the feature.
4. Validate the milestone immediately so it fails and creates a corrective feature.
5. Inspect the findings.
6. Complete the original feature.
7. Discover and complete the corrective feature.
8. Validate the milestone again so the mission finishes.
9. Write a mission summary report.
10. Run the tests when done.

Use these exact values where applicable:

- mission goal:
  `Exercise the mission validator by creating and resolving corrective follow-up work.`
- contract summary:
  `Mission validation should create corrective work when feature coverage is incomplete.`
- milestone:
  `m1|Corrective milestone|Recover from a failed mission validation pass.`
- assertion:
  `a1|Corrective loop works|The mission should fail validation once, create corrective work, and then finish cleanly.|contract|Inspect findings, corrective features, and final validation state.`
- feature:
  `f1|m1|Implement original slice|Create the original mission slice before corrective work exists.|worker|app/original.py||a1`
- original feature completed work:
  `Implemented the original mission slice after the validator exposed the missing completion state.`
- original feature next recommendation:
  `Dispatch and complete the corrective follow-up feature next.`
- corrective feature completed work:
  `Implemented the corrective follow-up slice so the validator can close the milestone.`
- corrective feature next recommendation:
  `Re-run mission validation to close the milestone.`

Important:

- Discover the generated mission id from `.harness/missions/missions/...`.
- Discover the original feature id from the `execute-next` output or the stored
  feature JSON.
- Discover the corrective feature id from the first `validate-milestone` output
  or the stored feature JSON.
- Do not recreate the mission after validation fails. Recover the existing one.

Run the tests when done.
