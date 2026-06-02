
## Skill routing

When the user's request matches an available skill, ALWAYS invoke it using the Skill
tool as your FIRST action. Do NOT answer directly, do NOT use other tools first.
The skill has specialized workflows that produce better results than ad-hoc answers.

Key routing rules:
- Product ideas, "is this worth building", brainstorming → invoke office-hours
- Bugs, errors, "why is this broken", 500 errors → invoke investigate
- Ship, deploy, push, create PR → invoke ship
- QA, test the site, find bugs → invoke qa
- Code review, check my diff → invoke review
- Update docs after shipping → invoke document-release
- Weekly retro → invoke retro
- Design system, brand → invoke design-consultation
- Visual audit, design polish → invoke design-review
- Architecture review → invoke plan-eng-review
- Save progress, checkpoint, resume → invoke checkpoint
- Code quality, health check → invoke health

## Implementation rules

- Do not route behavior with brittle user-text prefix or keyword heuristics when the choice affects runtime mode, tool scope, or safety posture.
- For behavior/mode selection, prefer explicit user controls or a dedicated classifier over ad-hoc string matching.
- Do not introduce `_MARKERS`-style keyword tuples for intent routing. If a behavior needs routing, model it as explicit state, an explicit command, or a dedicated classifier with tests.
