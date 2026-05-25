# Research Architecture

Harness treats autonomous improvement as a tiered research system rather than a single self-editing loop.

## Layers

1. Foundation
- Stable runtime, domains, verifiers, tools, and eval baselines.

2. Research
- Rabbit holes, unknowns, section maps, observations, opportunities, hypotheses, and experiment plans.

3. Publication
- Durable publications and citations that let later agents reuse or challenge prior work.

4. Refinement
- Promotion candidates that distill promising findings into bounded, reviewable changes.

5. Promotion
- Branch, commit, and PR payload generation with explicit file scope and evidence expectations.

## Storage

Research state lives under `.harness/research/` and includes:
- `vision/`
- `themes/`
- `unknowns/`
- `rabbitholes/`
- `publications/`
- `citations/`
- `inspiration/`
- `section-maps/`
- `observations/`
- `opportunities/`
- `hypotheses/`
- `experiment-plans/`
- `experiments/`
- `promotions/`
- `archive/`

Each durable artifact is persisted as JSON and, where useful, a human-readable Markdown companion.

## Flow

1. Define or update the current vision.
2. Add themes and unknowns.
3. Explore through rabbit holes, observations, and opportunities.
4. Turn promising directions into hypotheses and experiment plans.
5. Run experiments and publish what is learned.
6. Refine strong results into promotion candidates.
7. Prepare bounded branches, commits, and PR payloads.
8. Archive or reject weak directions so they are not retried blindly.

## Safety

- Inspiration may come from repo analysis, peer artifacts, papers, web sources, or trend notes.
- Promotion must still be evidence-backed.
- Target file scope is explicit on promotion candidates.
- Archive and resurrection are first-class so failure becomes reusable memory.
