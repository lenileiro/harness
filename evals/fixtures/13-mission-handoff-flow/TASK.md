# Resume a mission from an existing handoff and complete it

This workspace already contains a mission that has been planned, approved, and
dispatched to a worker. Continue from that state instead of recreating it.

Use CLI entrypoints only. Do not hand-edit `.harness/missions/*.json` files
directly.

Use these commands:

- `harness mission show`
- `harness mission show-contract`
- `harness mission list-handoffs`
- `harness mission show-handoff`
- `harness mission complete-feature`
- `harness mission validate-milestone`
- `harness mission summarize`
- `harness mission list-reports`

Do all of the following:

1. Inspect the existing mission and validation contract.
2. Inspect the existing planner handoff for the current feature.
3. Complete the handed-off feature.
4. Validate the milestone so the mission can finish.
5. Write a mission summary report and inspect the stored reports.
6. Run the tests when done.

Use these exact values where applicable:

- completed work:
  `Implemented the mission handoff consumption flow and recorded the worker completion state.`
- remaining work:
  `No remaining feature work.`
- known issue:
  `Validator pass still required.`
- next recommendation:
  `Validate the milestone and publish the mission summary report.`
- confidence:
  `0.95`

Important:

- Reuse the existing mission, feature, and handoff ids from `.harness/missions`.
- Do not create a new mission or a new feature.
- The task is only complete when the mission finishes and a report exists under
  `.harness/missions/reports`.

Run the tests when done.
