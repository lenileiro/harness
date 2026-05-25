# Continue from an existing handover and leave the next one ready

This workspace already contains a handover created by a previous agent.
Continue from that state instead of recreating it.

Use CLI entrypoints only. Do not hand-edit `.harness/*.json` files directly.

Use these commands:

- `harness vision show`
- `harness resume show`
- `harness research search`
- `harness research show-publication`
- `harness research create-opportunity`
- `harness research hypothesize`
- `harness research refine`
- `harness resume add-feature`
- `harness resume set-current`

Do all of the following:

1. Inspect the existing vision and resume contract.
2. Inspect the existing publication that defines the current handover.
3. Create one opportunity titled `Handover verification checklist`.
4. Create one hypothesis for that opportunity.
5. Refine one promotion candidate titled `Handover verification handoff`.
6. Add a third resume feature named `handover-verification`.
7. Switch the resume contract so the current feature becomes
   `handover-verification`.

Use these exact values where applicable:

- opportunity summary:
  `The next agent should verify that a handover names the next feature, the source publication, and the continuation path.`
- hypothesis claim:
  `A verification checklist will make handovers safer for the next agent to consume.`
- hypothesis expected win:
  `handover artifacts become easier for follow-on agents to verify`
- hypothesis risk level:
  `low`
- hypothesis change mode:
  `improve`
- candidate summary:
  `Promote a verification-oriented continuation step after the handover is consumed.`
- candidate expected metric:
  `handover artifacts include verification checklist coverage`
- candidate validation plan:
  `Run the fixture tests to confirm the continuation artifacts and resume contract are linked.`
- third feature description:
  `Verify the handover contract before the next continuation step.`
- third feature phases:
  `inspect,verify,continue`

Important:

- Reuse the existing publication as the source publication for the promotion
  candidate.
- Do not recreate the vision, theme, unknown, rabbit hole, or existing
  publication.
- Leave the previous features intact and only advance the roadmap to the new
  `handover-verification` feature.

You will need the generated publication ID and the newly created opportunity /
hypothesis IDs between steps. Read the generated JSON files under
`.harness/research/...` to get those IDs as needed.

Run the tests when done.
