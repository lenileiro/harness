## Promotion Candidate
- candidate: `promo-validate-repaired-research-pr-co-51ac4a22`
- branch: `research/promo-validate-repaired-research-pr-co-51ac4a22`
- base: `main`
- risk: `low`

## Summary
Open a draft PR directly through harness research pr.

## Change Intent
- mode: `improve`
- subsystem: `research`
- risk: `low`

We need the native PR path to be reliable.

Expected outcome: Draft PR opens without raw gh fallback.

## Target Files
- `.harness/research/promotions`

## Source Hypotheses
- `hyp-the-repaired-research-pr-command-cb3074e4`

## Expected Metric
PR URL returned by harness research pr

## Validation Plan
Run the direct research promotion flow.

## Evidence Checklist
- [ ] Targeted tests passed
- [ ] Relevant eval slices passed
- [ ] No unintended file-scope expansion
- [ ] Risk remains acceptable for review
