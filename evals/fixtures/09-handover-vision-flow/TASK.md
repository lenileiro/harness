# Create a structured handover for the next agent

Use the built-in Harness CLI in this workspace to create a small research
handover that another agent could continue without reading the original
transcript.

Use CLI entrypoints only. Do not hand-edit `.harness/*.json` files directly.

Use these commands:

- `harness vision update`
- `harness research add-theme`
- `harness research create-unknown`
- `harness research open`
- `harness research publish`
- `harness resume init`
- `harness resume add-feature`
- `harness resume set-current`

For the resume contract, use the commands in this shape:

- `harness resume init --feature handover-checklist --description "..."`
- `harness resume add-feature handover-consumption --description "..." --phases review,continue,verify`
- `harness resume set-current handover-consumption`

Important: `harness resume init` creates a default `first-feature` when you do
not pass `--feature`. Do not rely on that default for this task.

Do all of the following:

1. Set the current vision title to `Autonomous Handover System` and the summary
   to `Leave structured continuation state for the next agent.`
2. Add a theme titled `Agent continuity`.
3. Create an unknown under that theme asking:
   `What context must a handover always preserve?`
4. Open one rabbit hole titled `Handover artifact checklist`.
5. Publish one publication titled `Handover checklist findings`.
6. Create a resume contract with current feature `handover-checklist`.
7. Add a second feature named `handover-consumption`.
8. Switch the resume contract so the current feature becomes
   `handover-consumption`.

Use these exact values where applicable:

- theme description:
  `Ensure one agent can leave durable next-step context for another.`
- unknown why-it-matters:
  `Future agents need enough structure to continue work without replaying history.`
- rabbit hole question:
  `What information should every agent handover preserve for the next agent?`
- rabbit hole scope:
  `Review resume, research, and continuation artifacts for minimum useful state.`
- publication summary:
  `A handover should preserve the next mission, the open question, and the continuation path.`
- publication claim:
  `Every handover should name the next feature, the active question, and where to continue the work.`
- publication recommendation:
  `Use the resume contract to point the next agent at a single current feature.`
- publication open question:
  `Should handover artifacts also include a verification checklist for the next agent?`
- resume init description:
  `Capture the checklist for what a durable handover must contain.`
- second feature description:
  `Consume the handover and continue the next step without replaying prior context.`
- second feature phases:
  `review,continue,verify`

You will need the generated theme ID and rabbit-hole ID between steps. Read the
generated JSON files under `.harness/...` to get those IDs as needed.

Run the tests when done.
