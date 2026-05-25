from __future__ import annotations

from pathlib import Path

from harness.core.citations import Citation
from harness.core.experiment_plans import ExperimentPlan
from harness.core.experiments import CommandResult, Experiment, ExperimentResult
from harness.core.hypotheses import Hypothesis
from harness.core.inspiration import ExternalSource, InspirationNote
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
from harness.core.research_store import ResearchStore, default_research_root
from harness.core.section_maps import SectionMap


def test_research_store_writes_rabbit_hole_and_publication_files(tmp_path: Path) -> None:
    store = ResearchStore(root=default_research_root(tmp_path))
    rabbit_hole = RabbitHole(
        id=store.new_id("rh", "Verifier routing"),
        title="Verifier routing",
        question="Can verifier routing be improved?",
        scope="Check current verification routing and eval impact.",
        theme="verification",
        change_intent=ChangeIntent(
            mode="improve",
            subsystem="verification",
            rationale="Routing is too broad.",
            expected_outcome="Less verifier noise.",
        ),
    )

    rabbit_path = store.add_rabbit_hole(rabbit_hole)
    publication = Publication(
        id=store.new_id("pub", "Verifier routing findings"),
        rabbit_hole_id=rabbit_hole.id,
        title="Verifier routing findings",
        summary="Scoped routing reduced noise in targeted checks.",
        claims=("Scoped routing helps.",),
    )
    publication_path = store.add_publication(publication)

    assert (rabbit_path / "rabbit_hole.json").is_file()
    assert (rabbit_path / "RABBITHOLE.md").is_file()
    assert (publication_path / "publication.json").is_file()
    assert (publication_path / "PUBLICATION.md").is_file()


def test_research_store_searches_across_kinds(tmp_path: Path) -> None:
    store = ResearchStore(root=default_research_root(tmp_path))
    rabbit_hole = RabbitHole(
        id="rh-demo",
        title="Verifier routing rabbit hole",
        question="Investigate routing quality.",
        scope="verification internals",
        theme="verification",
    )
    store.add_rabbit_hole(rabbit_hole)
    publication = Publication(
        id="pub-demo",
        rabbit_hole_id="rh-demo",
        title="Routing publication",
        summary="Shows that selective routing helps review and research domains.",
        claims=("Selective routing helps.",),
    )
    store.add_publication(publication)

    hits = store.search("routing", limit=10)

    assert [hit.kind for hit in hits] == ["rabbit_hole", "publication"]


def test_research_store_persists_vision_theme_and_unknown(tmp_path: Path) -> None:
    store = ResearchStore(root=default_research_root(tmp_path))
    vision_path = store.update_vision(
        Vision(
            id="current",
            title="Autonomous research harness",
            summary="Turn Harness into a compounding research and promotion system.",
            themes=("autonomous-improvement",),
            success_metrics=("autonomous PR quality",),
        )
    )
    theme_path = store.add_theme(
        Theme(
            id="theme-auto",
            vision_id="current",
            title="Autonomous improvement",
            description="Study safe self-improvement loops.",
            priority="high",
        )
    )
    unknown_path = store.add_unknown(
        Unknown(
            id="unknown-001",
            theme_id="theme-auto",
            question="Which changes are safe for autonomous PR creation?",
            why_it_matters="Promotion needs a strict first safety envelope.",
            confidence=0.5,
        )
    )

    assert (vision_path / "vision.json").is_file()
    assert (theme_path / "theme.json").is_file()
    assert (unknown_path / "unknown.json").is_file()
    assert store.load_vision().title == "Autonomous research harness"
    assert [item.id for item in store.list_themes(vision_id="current")] == ["theme-auto"]
    assert [item.id for item in store.list_unknowns(theme_id="theme-auto")] == ["unknown-001"]


def test_research_store_persists_section_maps_and_observations(tmp_path: Path) -> None:
    store = ResearchStore(root=default_research_root(tmp_path))
    section_path = store.add_section_map(
        SectionMap(
            id="section-runtime",
            section="runtime",
            files=("runtime.py",),
            interfaces=("Agent.run",),
        )
    )
    observation_path = store.add_observation(
        Observation(
            id="obs-runtime",
            title="Runtime is a leverage point",
            summary="Routing and loop handling interact here.",
            source_type="repo",
            related_sections=("runtime", "verification"),
        )
    )

    found = store.find_section_map("runtime")

    assert section_path.joinpath("section_map.json").is_file()
    assert observation_path.joinpath("observation.json").is_file()
    assert found is not None
    assert found.id == "section-runtime"


def test_research_store_lists_and_finds_related_opportunities(tmp_path: Path) -> None:
    store = ResearchStore(root=default_research_root(tmp_path))
    store.add_opportunity(
        Opportunity(
            id="opp-runtime",
            title="Runtime and research policy",
            summary="Research completion depends on runtime profile and tool scope.",
            related_sections=("runtime", "research"),
            origin_observations=("obs-runtime",),
            change_modes=("improve",),
            theme="autonomous-improvement",
        )
    )

    opportunities = store.list_opportunities(theme="autonomous-improvement")
    related = store.related_opportunities("runtime")

    assert [item.id for item in opportunities] == ["opp-runtime"]
    assert [item.id for item in related] == ["opp-runtime"]


def test_research_store_lists_hypotheses_and_experiment_plans(tmp_path: Path) -> None:
    store = ResearchStore(root=default_research_root(tmp_path))
    store.add_opportunity(
        Opportunity(
            id="opp-runtime",
            title="Runtime and research policy",
            summary="Research completion depends on runtime profile and tool scope.",
        )
    )
    store.add_hypothesis(
        Hypothesis(
            id="hyp-runtime",
            opportunity_id="opp-runtime",
            claim="Repo-first research plus loop detection improves completion.",
            expected_win="More completed runs.",
            risk_level="low",
            change_mode="improve",
        )
    )
    store.add_experiment_plan(
        ExperimentPlan(
            id="plan-runtime",
            hypothesis_id="hyp-runtime",
            plan="Restrict tools and compare research live runs.",
            target_files=("domain_profiles.py",),
            checks=("pytest",),
        )
    )

    hypotheses = store.list_hypotheses(opportunity_id="opp-runtime")
    plans = store.list_experiment_plans(hypothesis_id="hyp-runtime")

    assert [item.id for item in hypotheses] == ["hyp-runtime"]
    assert [item.id for item in plans] == ["plan-runtime"]


def test_research_store_search_finds_new_research_artifact_kinds(tmp_path: Path) -> None:
    store = ResearchStore(root=default_research_root(tmp_path))
    store.update_vision(
        Vision(
            id="current",
            title="Autonomous research harness",
            summary="Turn Harness into a compounding research and promotion system.",
        )
    )
    store.add_theme(
        Theme(
            id="theme-runtime",
            vision_id="current",
            title="Runtime leverage",
            description="Study how runtime behavior shapes research outcomes.",
        )
    )
    store.add_unknown(
        Unknown(
            id="unknown-runtime",
            theme_id="theme-runtime",
            question="How much does runtime policy affect research completion?",
            why_it_matters="Research reliability depends on it.",
        )
    )
    store.add_section_map(
        SectionMap(
            id="section-runtime",
            section="runtime",
            opportunities=("repo-first research handling",),
        )
    )
    store.add_observation(
        Observation(
            id="obs-runtime",
            title="Runtime affects research completion",
            summary="Loop control and finalization intersect here.",
            source_type="repo",
        )
    )
    store.add_opportunity(
        Opportunity(
            id="opp-runtime",
            title="Runtime and research policy",
            summary="Research completion depends on runtime profile and tool scope.",
        )
    )
    store.add_hypothesis(
        Hypothesis(
            id="hyp-runtime",
            opportunity_id="opp-runtime",
            claim="Repo-first research plus loop detection improves completion.",
            expected_win="More completed runs.",
            risk_level="low",
            change_mode="improve",
        )
    )
    store.add_experiment_plan(
        ExperimentPlan(
            id="plan-runtime",
            hypothesis_id="hyp-runtime",
            plan="Restrict tools and compare research live runs.",
        )
    )

    hits = store.search("runtime", limit=10)
    kinds = {hit.kind for hit in hits}

    assert {
        "theme",
        "unknown",
        "section_map",
        "observation",
        "opportunity",
        "hypothesis",
        "experiment_plan",
    } <= kinds


def test_research_store_lists_promotion_candidates(tmp_path: Path) -> None:
    store = ResearchStore(root=default_research_root(tmp_path))
    store.add_promotion_candidate(
        PromotionCandidate(
            id="promo-runtime",
            title="Research completion fix",
            summary="Promote the repo-first research and loop-detection change.",
            source_hypotheses=("hyp-runtime",),
            expected_metric="research smoke pass rate",
            validation_plan="Run pytest and research smoke.",
            risk_level="low",
        )
    )

    candidates = store.list_promotion_candidates()

    assert [item.id for item in candidates] == ["promo-runtime"]


def test_research_store_persists_experiment_and_result(tmp_path: Path) -> None:
    store = ResearchStore(root=default_research_root(tmp_path))
    experiment = Experiment(
        id="exp-runtime",
        plan_id="plan-runtime",
        branch="main",
        worktree=str(tmp_path),
    )
    result = ExperimentResult(
        experiment_id="exp-runtime",
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

    store.add_experiment(experiment, result)
    experiments = store.list_experiments()

    assert [item.id for item in experiments] == ["exp-runtime"]
    assert store.load_experiment_result("exp-runtime").status == "passed"


def test_research_store_archives_and_resurrects_item(tmp_path: Path) -> None:
    store = ResearchStore(root=default_research_root(tmp_path))
    store.add_hypothesis(
        Hypothesis(
            id="hyp-runtime",
            opportunity_id="opp-runtime",
            claim="Repo-first research plus loop detection improves completion.",
            expected_win="More completed runs.",
            risk_level="low",
            change_mode="improve",
        )
    )

    archive_path = store.archive_item(
        kind="hypothesis",
        item_id="hyp-runtime",
        reason="Superseded by a stronger hypothesis.",
    )
    archive_entry = store.load_archive_item(archive_path.name)

    assert archive_entry.kind == "hypothesis"
    assert archive_entry.original_id == "hyp-runtime"
    assert not (store.hypotheses_dir / "hyp-runtime").exists()

    restored = store.resurrect_archive_item(archive_path.name)

    assert restored == store.hypotheses_dir / "hyp-runtime"
    assert restored.exists()
    assert not archive_path.exists()


def test_research_store_persists_inspiration_notes(tmp_path: Path) -> None:
    store = ResearchStore(root=default_research_root(tmp_path))
    target = store.add_inspiration_note(
        InspirationNote(
            id="insp-demo",
            title="Interesting paper",
            summary="A promising technique for autonomous improvement.",
            source=ExternalSource(kind="paper", ref="arXiv:1234.5678", title="Interesting paper"),
            related_themes=("autonomous-improvement",),
            related_sections=("research",),
        )
    )

    assert (target / "inspiration.json").is_file()
    assert store.load_inspiration_note("insp-demo").source.kind == "paper"
    assert store.list_inspiration_notes(source_kind="paper")[0].id == "insp-demo"


def test_research_store_persists_citations(tmp_path: Path) -> None:
    store = ResearchStore(root=default_research_root(tmp_path))
    store.add_publication(
        Publication(
            id="pub-a",
            rabbit_hole_id="rh-a",
            title="A",
            summary="A summary",
        )
    )
    store.add_publication(
        Publication(
            id="pub-b",
            rabbit_hole_id="rh-b",
            title="B",
            summary="B summary",
        )
    )
    target = store.add_citation(
        Citation(
            id="cite-a-b",
            source_publication_id="pub-a",
            target_publication_id="pub-b",
            relationship="builds_on",
        )
    )

    assert (target / "citation.json").is_file()
    assert store.load_citation("cite-a-b").relationship == "builds_on"
    assert store.list_citations()[0].id == "cite-a-b"
    assert store.search("builds_on", kinds=("citation",))[0].kind == "citation"


def test_research_store_persists_research_assets(tmp_path: Path) -> None:
    store = ResearchStore(root=default_research_root(tmp_path))
    target = store.add_research_asset(
        ResearchAsset(
            id="asset-demo",
            kind="eval_report",
            path="evals/runs/demo/report.json",
            description="Targeted eval result for a research branch.",
        )
    )

    assert (target / "asset.json").is_file()
    assert store.load_research_asset("asset-demo").kind == "eval_report"
    assert store.list_research_assets()[0].id == "asset-demo"
