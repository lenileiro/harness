from __future__ import annotations

from harness.core.citations import Citation
from harness.core.experiment_plans import ExperimentPlan
from harness.core.experiments import CommandResult, Experiment, ExperimentResult
from harness.core.hypotheses import Hypothesis
from harness.core.observations import Observation
from harness.core.opportunities import Opportunity
from harness.core.promotion_candidates import PromotionCandidate
from harness.core.publications import ResearchAsset
from harness.core.research_models import (
    ChangeIntent,
    Publication,
    RabbitHole,
    Theme,
    Unknown,
    Vision,
)
from harness.core.section_maps import SectionMap


def test_rabbit_hole_round_trip_preserves_change_intent() -> None:
    rabbit_hole = RabbitHole(
        id="rh-demo",
        title="Verifier routing",
        question="Can we route verifiers more selectively?",
        scope="Check routing rules and eval impact.",
        theme="autonomous-improvement",
        related_sections=("verification", "runtime"),
        tags=("routing", "evals"),
        change_intent=ChangeIntent(
            mode="improve",
            subsystem="verification",
            rationale="Current routing is too broad.",
            expected_outcome="Fewer false positives.",
            risk="low",
        ),
    )

    loaded = RabbitHole.from_dict(rabbit_hole.to_dict())

    assert loaded.id == rabbit_hole.id
    assert loaded.related_sections == ("verification", "runtime")
    assert loaded.change_intent is not None
    assert loaded.change_intent.mode == "improve"


def test_publication_round_trip_preserves_lists() -> None:
    publication = Publication(
        id="pub-demo",
        rabbit_hole_id="rh-demo",
        title="Routing findings",
        summary="Selective routing reduces verifier noise.",
        claims=("Routing on task shape helps.",),
        supporting_evidence=("eval f03 improved",),
        recommendations=("Refine the router.",),
        sources=("docs/router.md",),
        citations=("pub-older",),
        confidence=2.0,
        status="promising",
    )

    loaded = Publication.from_dict(publication.to_dict())

    assert loaded.claims == ("Routing on task shape helps.",)
    assert loaded.sources == ("docs/router.md",)
    assert loaded.status == "promising"


def test_vision_theme_and_unknown_round_trip() -> None:
    vision = Vision(
        id="current",
        title="Autonomous research harness",
        summary="Turn Harness into a compounding research and promotion system.",
        themes=("autonomous-improvement", "research-memory"),
        success_metrics=("high-signal autonomous PRs",),
    )
    theme = Theme(
        id="theme-auto",
        vision_id="current",
        title="Autonomous improvement",
        description="Study how agents can improve the harness safely.",
        priority="high",
    )
    unknown = Unknown(
        id="unknown-001",
        theme_id="theme-auto",
        question="Which change classes are safe for autonomous PR creation?",
        why_it_matters="The promotion lane needs a clear initial safety envelope.",
        current_belief="Parser and prompt conformance are safer than core runtime refactors.",
        confidence=0.6,
        related_sections=("research", "runtime"),
    )

    assert Vision.from_dict(vision.to_dict()).themes == (
        "autonomous-improvement",
        "research-memory",
    )
    assert Theme.from_dict(theme.to_dict()).priority == "high"
    loaded_unknown = Unknown.from_dict(unknown.to_dict())
    assert loaded_unknown.related_sections == ("research", "runtime")
    assert loaded_unknown.confidence == 0.6


def test_section_map_and_observation_round_trip() -> None:
    section_map = SectionMap(
        id="section-runtime",
        section="runtime",
        files=("runtime.py", "run_commands.py"),
        interfaces=("Agent.run",),
        weaknesses=("too many responsibilities",),
    )
    observation = Observation(
        id="obs-demo",
        title="Repo-first research works better",
        summary="Research domain over-explored when web tools were enabled.",
        source_type="eval",
        source_ref="post-roadmap-research-live-v4",
        related_sections=("research", "domain_profiles"),
        theme="autonomous-improvement",
    )

    assert SectionMap.from_dict(section_map.to_dict()).section == "runtime"
    loaded_observation = Observation.from_dict(observation.to_dict())
    assert loaded_observation.source_type == "eval"
    assert loaded_observation.related_sections == ("research", "domain_profiles")


def test_opportunity_round_trip_preserves_links() -> None:
    opportunity = Opportunity(
        id="opp-demo",
        title="Research/runtime interaction",
        summary="Research completion issues point at runtime and domain policy together.",
        related_sections=("research", "runtime"),
        origin_observations=("obs-demo",),
        change_modes=("improve", "build_on"),
        theme="autonomous-improvement",
        priority="high",
    )

    loaded = Opportunity.from_dict(opportunity.to_dict())

    assert loaded.related_sections == ("research", "runtime")
    assert loaded.origin_observations == ("obs-demo",)
    assert loaded.priority == "high"


def test_hypothesis_and_experiment_plan_round_trip() -> None:
    hypothesis = Hypothesis(
        id="hyp-demo",
        opportunity_id="opp-demo",
        claim="Repo-first research plus loop detection improves completion.",
        expected_win="More completed research runs.",
        risk_level="low",
        change_mode="improve",
    )
    plan = ExperimentPlan(
        id="plan-demo",
        hypothesis_id="hyp-demo",
        plan="Restrict tools and compare live research runs.",
        target_files=("domain_profiles.py", "research_commands.py"),
        checks=("pytest", "pyright"),
        eval_slices=("research-smoke",),
    )

    assert Hypothesis.from_dict(hypothesis.to_dict()).change_mode == "improve"
    loaded_plan = ExperimentPlan.from_dict(plan.to_dict())
    assert loaded_plan.eval_slices == ("research-smoke",)


def test_promotion_candidate_round_trip() -> None:
    candidate = PromotionCandidate(
        id="promo-demo",
        title="Research completion fix",
        summary="Promote the repo-first research and loop-detection change.",
        source_publications=("pub-demo",),
        source_hypotheses=("hyp-demo",),
        target_files=("domain_profiles.py", "research_commands.py"),
        expected_metric="research smoke pass rate",
        validation_plan="Run pytest, pyright, and research smoke.",
        risk_level="low",
        change_intent=ChangeIntent(
            mode="improve",
            subsystem="research",
            rationale="The current path over-explores and times out.",
            expected_outcome="More final structured answers.",
        ),
    )

    loaded = PromotionCandidate.from_dict(candidate.to_dict())

    assert loaded.source_publications == ("pub-demo",)
    assert loaded.change_intent is not None
    assert loaded.change_intent.subsystem == "research"


def test_citation_round_trip() -> None:
    citation = Citation(
        id="cite-demo",
        source_publication_id="pub-new",
        target_publication_id="pub-old",
        relationship="builds_on",
        note="Extends prior work with tighter evidence.",
    )

    loaded = Citation.from_dict(citation.to_dict())

    assert loaded.relationship == "builds_on"
    assert loaded.target_publication_id == "pub-old"


def test_research_asset_round_trip() -> None:
    asset = ResearchAsset(
        id="asset-demo",
        kind="eval_report",
        path="evals/runs/demo/report.json",
        description="Targeted eval result for a research branch.",
    )

    loaded = ResearchAsset.from_dict(asset.to_dict())

    assert loaded.kind == "eval_report"
    assert loaded.path.endswith("report.json")


def test_experiment_and_result_round_trip() -> None:
    experiment = Experiment(
        id="exp-demo",
        plan_id="plan-demo",
        branch="main",
        worktree="/tmp/worktree",
    )
    result = ExperimentResult(
        experiment_id="exp-demo",
        status="passed",
        command_results=(
            CommandResult(
                kind="check",
                command="python -V",
                exit_code=0,
                duration_seconds=0.1,
            ),
        ),
        started_at="2026-05-25T12:00:00+00:00",
        finished_at="2026-05-25T12:00:01+00:00",
        duration_seconds=1.0,
    )

    assert Experiment.from_dict(experiment.to_dict()).branch == "main"
    loaded_result = ExperimentResult.from_dict(result.to_dict())
    assert loaded_result.command_results[0].command == "python -V"
