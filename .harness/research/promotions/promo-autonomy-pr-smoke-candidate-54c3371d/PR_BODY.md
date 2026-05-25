## Promotion Candidate
- candidate: `promo-autonomy-pr-smoke-candidate-54c3371d`
- branch: `research/promo-autonomy-pr-smoke-candidate-54c3371d`
- base: `main`
- risk: `low`

## Summary
Use generated promotion artifacts as the bounded PR diff.

## Change Intent
- mode: `improve`
- subsystem: `research`
- risk: `low`

Smoke-test the GitHub PR path locally without touching product code.

Expected outcome: A draft PR exists from a temp-clone branch.

## Target Files
- `.harness/research/promotions/promo-autonomy-pr-smoke-candidate-54c3371d/promotion_candidate.json`
- `.harness/research/promotions/promo-autonomy-pr-smoke-candidate-54c3371d/promotion_draft.json`
- `.harness/research/promotions/promo-autonomy-pr-smoke-candidate-54c3371d/PR_BODY.md`

## Source Hypotheses
- `hyp-generated-promotion-artifacts-sh-0be40eec`

## Expected Metric
A draft PR is opened successfully on GitHub.

## Validation Plan
Verify the PR exists and contains the generated PR body.

## Evidence Checklist
- [ ] Targeted tests passed
- [ ] Relevant eval slices passed
- [ ] No unintended file-scope expansion
- [ ] Risk remains acceptable for review
