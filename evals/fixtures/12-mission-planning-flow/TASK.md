# Plan and approve a mission before any implementation begins

Use the built-in Harness CLI in this workspace to create and plan a first-class
mission.

Use CLI entrypoints only. Do not hand-edit `.harness/missions/*.json` files
directly.

Use these commands:

- `harness mission create`
- `harness mission draft-plan --apply`
- `harness mission show-contract`
- `harness mission list-milestones`
- `harness mission list-features`
- `harness mission approve`

Do all of the following:

1. Create a mission titled `Mission Planning System`.
2. Draft and apply a plan with two milestones, three assertions, and three features.
3. Inspect the generated contract, milestones, and features.
4. Approve the mission.
5. Run the tests when done.

Use these exact values where applicable:

- mission goal:
  `Create a mission plan that defines milestones, assertions, and feature coverage before implementation.`
- planner model:
  `gpt-planner`
- worker model:
  `gpt-worker`
- validator model:
  `gpt-validator`
- reporter model:
  `gpt-reporter`
- budget tokens:
  `9000`
- budget runtime minutes:
  `120`
- contract summary:
  `Mission assertions must be declared before execution and covered by planned feature work.`

Milestones:

- `m1|Mission schema|Define the mission objects and storage shape.`
- `m2|Mission runtime|Dispatch feature work and validate milestone completion.`

Assertions:

- `a1|Mission object exists|The mission plan should define a durable mission object and store layout.|contract|Inspect the stored mission and milestone artifacts.`
- `a2|Runtime dispatch works|The mission runtime should be able to dispatch feature work from the approved plan.|behavior|Inspect the planned runtime feature coverage.`
- `a3|Validation gates milestones|Milestones should only complete after validator coverage is satisfied.|behavior|Inspect the planned validator feature coverage.`

Features:

- `f1|m1|Define mission schema|Add the durable mission, milestone, feature, contract, and handoff objects.|planner|packages/core/src/harness/core/mission_models.py,packages/core/src/harness/core/mission_store.py||a1`
- `f2|m2|Implement mission runtime|Dispatch ready feature work and persist mission runs and handoffs.|worker|packages/core/src/harness/core/mission_runtime.py|f1|a2`
- `f3|m2|Implement mission validator|Validate milestone assertions and create corrective follow-up work when needed.|validator|packages/core/src/harness/core/mission_validator.py|f2|a3`

Important:

- Discover the generated mission id from `.harness/missions/missions/...` before
  calling `mission draft-plan`, `show-contract`, or `approve`.
- Use `mission draft-plan --apply` rather than the manual `mission plan` flag set.
- The generated draft should use the exact milestone, assertion, and feature values listed above.
- Do not skip the contract inspection step.
- The mission should end in the `approved` state, not `draft` or `planned`.

Run the tests when done.
