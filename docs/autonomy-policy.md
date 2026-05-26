# Autonomy Policy

Harness supports autonomous research and bounded promotion. It does not assume autonomous merge.

## Principles

1. Inspiration is broad
- Agents may learn from repo artifacts, peer outputs, papers, web sources, and trends.

2. Acceptance is strict
- Promotion requires explicit evidence, not just a plausible idea.

3. Research and promotion are different lanes
- Most research should end as publication, archive, or a narrowed unknown.

4. Scope must stay bounded
- Promotion candidates define target files, expected metrics, and validation plans.

5. Failure must remain useful
- Rejected or superseded work is archived, not erased.

## Allowed autonomous outputs

- Rabbit holes
- Publications
- Citations
- Observations
- Opportunities
- Hypotheses
- Experiment plans
- Experiment results
- Promotion candidates
- Branch / commit / PR payload preparation

## Promotion expectations

Autonomous promotion should include:
- source publications or hypotheses
- change intent
- target files
- expected metric
- validation plan
- PR body with rationale and evidence checklist

## Out of scope

- autonomous merge
- unbounded file edits during promotion
- silent deletion of failed research
- direct jump from inspiration to production change without intermediate evidence

## Scheduler policy

Autonomous CI or cron execution should use bounded scheduler settings.

Recommended defaults:
- low or medium max risk
- explicit max step budget
- no branch / commit / push / PR side effects unless intentionally enabled

Harness supports this through:
- `harness research schedule-once`
- `harness research list-runs`
- `harness mission schedule-once`
- `harness mission summarize`
- `harness mission create-opportunity`
- `harness mission create-candidate`
- `[research_scheduler]` config defaults in TOML
- `[mission_scheduler]` config defaults in TOML

When mission validation finds blocking issues, preferred follow-up is:
- convert findings into explicit research opportunities
- convert validated mission features into bounded promotion candidates
- preserve mission linkage so later agents can continue work without rereading transcripts

## Live CI mode

Secret-backed live autonomy should remain explicitly gated.

Recommended policy:
- deterministic `schedule-once` runs on push / PR / cron by default
- live provider-backed evals only on manual dispatch
- required provider secrets must be present or the job should fail fast
- live mode should upload artifacts for later review instead of mutating the repo by default
- if humans review work through GitHub PRs, live mode may post a concise summary comment only when explicitly enabled

Mission autonomy follows the same deterministic-first principle.

Recommended mission policy:
- deterministic `mission schedule-once` runs on push / PR / cron by default
- mission CI should seed a bounded plan with explicit assertions before execution
- live provider-backed mission evals should remain manual and explicitly gated
- workflow summaries should include mission status, stop reason, and next actions
- uploaded artifacts should include both the scheduled run record and the mission summary report

## Mutation CI mode

Mutation-capable autonomy should be a separate lane from both deterministic and
live-eval modes.

Recommended policy:
- manual dispatch only
- dedicated protected environment such as `autonomy-mutations`
- explicit write permissions only for the mutation job
- default all mutation flags to off
- require bounded `max_steps` and low/medium risk
- upload artifacts for review even when mutation is enabled
- allow PR comments only behind an explicit dispatch flag or a supplied PR number
