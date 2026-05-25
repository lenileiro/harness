from __future__ import annotations

import json
import shlex
import sys
from pathlib import Path

from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.cli.config import HarnessConfig
from harness.core.research_store import ResearchStore, default_research_root


def test_vision_update_show_theme_and_unknown_commands(tmp_path: Path) -> None:
    runner = CliRunner()

    updated = runner.invoke(
        cli_main.app,
        [
            "vision",
            "update",
            "--title",
            "Autonomous research harness",
            "--summary",
            "Turn Harness into a compounding research and promotion system.",
            "--theme",
            "autonomous-improvement",
            "--success-metric",
            "high-signal autonomous PRs",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert updated.exit_code == 0, updated.stdout

    shown = runner.invoke(
        cli_main.app,
        ["vision", "show", "--cwd", str(tmp_path)],
    )
    assert shown.exit_code == 0, shown.stdout
    assert "Autonomous research harness" in shown.stdout
    assert "high-signal autonomous PRs" in shown.stdout

    added_theme = runner.invoke(
        cli_main.app,
        [
            "research",
            "add-theme",
            "--title",
            "Autonomous improvement",
            "--description",
            "Study how agents can improve the harness safely.",
            "--priority",
            "high",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert added_theme.exit_code == 0, added_theme.stdout
    theme_dir = next((tmp_path / ".harness" / "research" / "themes").iterdir())
    theme_id = theme_dir.name

    listed_themes = runner.invoke(
        cli_main.app,
        ["research", "list-themes", "--cwd", str(tmp_path)],
    )
    assert listed_themes.exit_code == 0, listed_themes.stdout
    assert theme_id in listed_themes.stdout
    assert "high" in listed_themes.stdout

    created_unknown = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-unknown",
            "--theme-id",
            theme_id,
            "--question",
            "Which change classes are safe for autonomous PRs?",
            "--why-it-matters",
            "Promotion needs a strict first safety envelope.",
            "--confidence",
            "0.5",
            "--related-sections",
            "research,runtime",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created_unknown.exit_code == 0, created_unknown.stdout

    listed_unknowns = runner.invoke(
        cli_main.app,
        ["research", "list-unknowns", "--cwd", str(tmp_path)],
        terminal_width=160,
    )
    assert listed_unknowns.exit_code == 0, listed_unknowns.stdout
    assert listed_unknowns.stdout.strip()

    searched_unknown = runner.invoke(
        cli_main.app,
        ["research", "search", "safety", "--kind", "unknown", "--cwd", str(tmp_path)],
    )
    assert searched_unknown.exit_code == 0, searched_unknown.stdout
    assert "unknown" in searched_unknown.stdout

    searched = runner.invoke(
        cli_main.app,
        ["research", "search", "autonomous", "--kind", "theme", "--cwd", str(tmp_path)],
    )
    assert searched.exit_code == 0, searched.stdout
    assert "theme" in searched.stdout


def test_research_open_publish_and_search(tmp_path: Path) -> None:
    runner = CliRunner()

    opened = runner.invoke(
        cli_main.app,
        [
            "research",
            "open",
            "--title",
            "Verifier routing",
            "--question",
            "Can verifier routing be improved?",
            "--scope",
            "Check routing and eval impact.",
            "--theme",
            "verification",
            "--related-sections",
            "verification,runtime",
            "--cwd",
            str(tmp_path),
            "--mode",
            "improve",
            "--subsystem",
            "verification",
            "--rationale",
            "Routing is too broad.",
            "--expected-outcome",
            "Less verifier noise.",
        ],
    )
    assert opened.exit_code == 0, opened.stdout
    rabbit_root = tmp_path / ".harness" / "research" / "rabbitholes"
    rabbit_dirs = list(rabbit_root.iterdir())
    assert len(rabbit_dirs) == 1

    rabbit_id = rabbit_dirs[0].name
    published = runner.invoke(
        cli_main.app,
        [
            "research",
            "publish",
            "--rabbit-hole",
            rabbit_id,
            "--title",
            "Verifier routing findings",
            "--summary",
            "Scoped routing helps.",
            "--claim",
            "Scoped routing reduces verifier noise.",
            "--evidence",
            "Targeted evals showed fewer false positives.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert published.exit_code == 0, published.stdout
    publication_id = next((tmp_path / ".harness" / "research" / "publications").iterdir()).name
    cited = runner.invoke(
        cli_main.app,
        [
            "research",
            "cite",
            "--source-publication",
            publication_id,
            "--target-publication",
            publication_id,
            "--relationship",
            "reuses",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert cited.exit_code == 0, cited.stdout

    searched = runner.invoke(
        cli_main.app,
        ["research", "search", "routing", "--cwd", str(tmp_path)],
    )
    assert searched.exit_code == 0, searched.stdout
    assert "rabbit_hole" in searched.stdout
    assert "publication" in searched.stdout

    shown = runner.invoke(
        cli_main.app,
        ["research", "show-publication", publication_id, "--cwd", str(tmp_path)],
    )
    assert shown.exit_code == 0, shown.stdout
    assert "Scoped routing helps." in shown.stdout


def test_research_map_section_add_observation_and_show(tmp_path: Path) -> None:
    runner = CliRunner()

    mapped = runner.invoke(
        cli_main.app,
        [
            "research",
            "map-section",
            "--section",
            "runtime",
            "--files",
            "packages/core/src/harness/core/runtime.py,packages/cli/src/harness/cli/run_commands.py",
            "--interfaces",
            "Agent.run,RunRequest",
            "--weaknesses",
            "mixed responsibilities",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert mapped.exit_code == 0, mapped.stdout

    observed = runner.invoke(
        cli_main.app,
        [
            "research",
            "add-observation",
            "--title",
            "Runtime is a leverage point",
            "--summary",
            "Routing, looping, and final-answer behavior intersect here.",
            "--source-type",
            "repo",
            "--related-sections",
            "runtime,verification",
            "--theme",
            "autonomous-improvement",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert observed.exit_code == 0, observed.stdout

    shown = runner.invoke(
        cli_main.app,
        ["research", "show-section", "runtime", "--cwd", str(tmp_path)],
    )
    assert shown.exit_code == 0, shown.stdout
    assert "runtime" in shown.stdout
    assert "mixed responsibilities" in shown.stdout


def test_research_create_opportunity_list_and_related(tmp_path: Path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Runtime and research policy",
            "--summary",
            "Research completion depends on runtime profile and repo-first tool scope.",
            "--related-sections",
            "runtime,research",
            "--origin-observations",
            "obs-runtime",
            "--change-modes",
            "improve,build_on",
            "--theme",
            "autonomous-improvement",
            "--priority",
            "high",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout

    listed = runner.invoke(
        cli_main.app,
        ["research", "list-opportunities", "--cwd", str(tmp_path)],
    )
    assert listed.exit_code == 0, listed.stdout
    assert "autonomous-improvement" in listed.stdout
    assert "high" in listed.stdout

    related = runner.invoke(
        cli_main.app,
        ["research", "related", "runtime", "--cwd", str(tmp_path)],
    )
    assert related.exit_code == 0, related.stdout
    assert "opp-runtime-and-research-policy" in related.stdout


def test_research_hypothesize_and_plan_experiment(tmp_path: Path) -> None:
    runner = CliRunner()
    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Runtime and research policy",
            "--summary",
            "Research completion depends on runtime profile and repo-first tool scope.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    opp_dir = next((tmp_path / ".harness" / "research" / "opportunities").iterdir())
    opportunity_id = opp_dir.name

    hypothesized = runner.invoke(
        cli_main.app,
        [
            "research",
            "hypothesize",
            "--opportunity",
            opportunity_id,
            "--claim",
            "Repo-first research plus loop detection improves completion.",
            "--expected-win",
            "More completed research runs.",
            "--risk-level",
            "low",
            "--change-mode",
            "improve",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert hypothesized.exit_code == 0, hypothesized.stdout
    hypothesis_dir = next((tmp_path / ".harness" / "research" / "hypotheses").iterdir())
    hypothesis_id = hypothesis_dir.name

    planned = runner.invoke(
        cli_main.app,
        [
            "research",
            "plan-experiment",
            "--hypothesis",
            hypothesis_id,
            "--plan",
            "Restrict tools and compare live research runs.",
            "--target-files",
            "packages/core/src/harness/core/domain_profiles.py,packages/cli/src/harness/cli/research_commands.py",
            "--checks",
            "pytest,pyright",
            "--eval-slices",
            "research-smoke",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert planned.exit_code == 0, planned.stdout
    plan_root = tmp_path / ".harness" / "research" / "experiment-plans"
    assert any((path / "experiment_plan.json").is_file() for path in plan_root.iterdir())

    searched = runner.invoke(
        cli_main.app,
        ["research", "search", "completion", "--kind", "hypothesis", "--cwd", str(tmp_path)],
    )
    assert searched.exit_code == 0, searched.stdout
    assert "hypothesis" in searched.stdout


def test_research_refine_and_list_candidates(tmp_path: Path) -> None:
    runner = CliRunner()
    opened = runner.invoke(
        cli_main.app,
        [
            "research",
            "open",
            "--title",
            "Research completion",
            "--question",
            "Why does research fail to finalize?",
            "--scope",
            "Research domain and runtime interactions.",
            "--theme",
            "autonomous-improvement",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert opened.exit_code == 0, opened.stdout
    rabbit_hole_id = next((tmp_path / ".harness" / "research" / "rabbitholes").iterdir()).name

    published = runner.invoke(
        cli_main.app,
        [
            "research",
            "publish",
            "--rabbit-hole",
            rabbit_hole_id,
            "--title",
            "Research completion findings",
            "--summary",
            "Repo-first research and loop detection help.",
            "--claim",
            "Repo-first research reduces wandering.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert published.exit_code == 0, published.stdout
    publication_id = next((tmp_path / ".harness" / "research" / "publications").iterdir()).name

    refined = runner.invoke(
        cli_main.app,
        [
            "research",
            "refine",
            "--title",
            "Research completion fix",
            "--summary",
            "Promote the repo-first research and loop-detection change.",
            "--source-publication",
            publication_id,
            "--target-files",
            "packages/core/src/harness/core/domain_profiles.py,packages/cli/src/harness/cli/research_commands.py",
            "--expected-metric",
            "research smoke pass rate",
            "--validation-plan",
            "Run pytest, pyright, and research smoke.",
            "--risk-level",
            "low",
            "--cwd",
            str(tmp_path),
            "--mode",
            "improve",
            "--subsystem",
            "research",
            "--rationale",
            "Current research runs over-explore before finalizing.",
            "--expected-outcome",
            "More final structured answers.",
        ],
    )
    assert refined.exit_code == 0, refined.stdout

    listed = runner.invoke(
        cli_main.app,
        ["research", "list-candidates", "--cwd", str(tmp_path)],
    )
    assert listed.exit_code == 0, listed.stdout
    assert "low" in listed.stdout


def test_research_create_candidate_alias(tmp_path: Path) -> None:
    runner = CliRunner()
    opened = runner.invoke(
        cli_main.app,
        [
            "research",
            "open",
            "--title",
            "Alias candidate",
            "--question",
            "How should promotion candidate creation be exposed?",
            "--scope",
            "Make candidate creation discoverable.",
            "--theme",
            "autonomous-improvement",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert opened.exit_code == 0, opened.stdout
    rabbit_hole_id = next((tmp_path / ".harness" / "research" / "rabbitholes").iterdir()).name

    published = runner.invoke(
        cli_main.app,
        [
            "research",
            "publish",
            "--rabbit-hole",
            rabbit_hole_id,
            "--title",
            "Alias candidate findings",
            "--summary",
            "Agents search for create-candidate semantics.",
            "--claim",
            "A create-candidate alias should improve discoverability.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert published.exit_code == 0, published.stdout
    publication_id = next((tmp_path / ".harness" / "research" / "publications").iterdir()).name

    created_candidate = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-candidate",
            "--title",
            "Alias candidate",
            "--summary",
            "Create a promotion candidate through the alias.",
            "--source-publication",
            publication_id,
            "--target-files",
            ".harness/research/promotions/alias/promotion_candidate.json",
            "--expected-metric",
            "alias works",
            "--validation-plan",
            "inspect candidate",
            "--risk-level",
            "low",
            "--cwd",
            str(tmp_path),
            "--mode",
            "improve",
            "--subsystem",
            "research",
            "--rationale",
            "Alias should improve command discoverability.",
            "--expected-outcome",
            "Promotion candidate exists.",
        ],
    )
    assert created_candidate.exit_code == 0, created_candidate.stdout
    assert "Created promotion candidate" in created_candidate.stdout


def test_research_candidate_create_subcommand_and_empty_hint(tmp_path: Path) -> None:
    runner = CliRunner()

    empty_list = runner.invoke(
        cli_main.app,
        ["research", "list-candidates", "--cwd", str(tmp_path)],
    )
    assert empty_list.exit_code == 0, empty_list.stdout
    assert "create-candidate" in empty_list.stdout
    assert "candidate create" in empty_list.stdout

    opened = runner.invoke(
        cli_main.app,
        [
            "research",
            "open",
            "--title",
            "Candidate subcommand",
            "--question",
            "How should candidate creation be exposed under the candidate namespace?",
            "--scope",
            "Create a candidate through the nested subcommand.",
            "--theme",
            "autonomous-improvement",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert opened.exit_code == 0, opened.stdout
    rabbit_hole_id = next((tmp_path / ".harness" / "research" / "rabbitholes").iterdir()).name

    published = runner.invoke(
        cli_main.app,
        [
            "research",
            "publish",
            "--rabbit-hole",
            rabbit_hole_id,
            "--title",
            "Candidate subcommand findings",
            "--summary",
            "The nested candidate create command should work.",
            "--claim",
            "The candidate create subcommand improves discoverability.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert published.exit_code == 0, published.stdout
    publication_id = next((tmp_path / ".harness" / "research" / "publications").iterdir()).name

    created_candidate = runner.invoke(
        cli_main.app,
        [
            "research",
            "candidate",
            "create",
            "--title",
            "Nested candidate",
            "--summary",
            "Create a promotion candidate through the nested command.",
            "--source-publication",
            publication_id,
            "--target-files",
            ".harness/research/promotions/nested/promotion_candidate.json",
            "--expected-metric",
            "nested alias works",
            "--validation-plan",
            "inspect nested candidate",
            "--risk-level",
            "low",
            "--cwd",
            str(tmp_path),
            "--mode",
            "improve",
            "--subsystem",
            "research",
            "--rationale",
            "Nested alias should improve command discoverability.",
            "--expected-outcome",
            "Nested promotion candidate exists.",
        ],
    )
    assert created_candidate.exit_code == 0, created_candidate.stdout
    assert "Created promotion candidate" in created_candidate.stdout


def test_research_archive_reject_and_resurrect(tmp_path: Path) -> None:
    runner = CliRunner()
    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Runtime and research policy",
            "--summary",
            "Research completion depends on runtime profile and repo-first tool scope.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    opportunity_id = next((tmp_path / ".harness" / "research" / "opportunities").iterdir()).name

    hypothesized = runner.invoke(
        cli_main.app,
        [
            "research",
            "hypothesize",
            "--opportunity",
            opportunity_id,
            "--claim",
            "Repo-first research plus loop detection improves completion.",
            "--expected-win",
            "More completed research runs.",
            "--risk-level",
            "low",
            "--change-mode",
            "improve",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert hypothesized.exit_code == 0, hypothesized.stdout
    hypothesis_id = next((tmp_path / ".harness" / "research" / "hypotheses").iterdir()).name

    archived = runner.invoke(
        cli_main.app,
        [
            "research",
            "archive",
            "--kind",
            "hypothesis",
            "--id",
            hypothesis_id,
            "--reason",
            "Superseded by a stronger hypothesis.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert archived.exit_code == 0, archived.stdout
    assert not (tmp_path / ".harness" / "research" / "hypotheses" / hypothesis_id).exists()

    listed = runner.invoke(
        cli_main.app,
        ["research", "list-archive", "--cwd", str(tmp_path)],
    )
    assert listed.exit_code == 0, listed.stdout
    assert "hypothesis" in listed.stdout

    archive_id = next((tmp_path / ".harness" / "research" / "archive").iterdir()).name
    resurrected = runner.invoke(
        cli_main.app,
        ["research", "resurrect", archive_id, "--cwd", str(tmp_path)],
    )
    assert resurrected.exit_code == 0, resurrected.stdout
    assert (tmp_path / ".harness" / "research" / "hypotheses" / hypothesis_id).exists()

    rejected = runner.invoke(
        cli_main.app,
        [
            "research",
            "reject",
            "--kind",
            "hypothesis",
            "--id",
            hypothesis_id,
            "--reason",
            "No longer worth pursuing.",
            "--note",
            "Lower-signal than the replacement.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert rejected.exit_code == 0, rejected.stdout
    assert not (tmp_path / ".harness" / "research" / "hypotheses" / hypothesis_id).exists()


def test_research_promote_and_pr_commands(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    opened = runner.invoke(
        cli_main.app,
        [
            "research",
            "open",
            "--title",
            "Research completion",
            "--question",
            "Why does research fail to finalize?",
            "--scope",
            "Research domain and runtime interactions.",
            "--theme",
            "autonomous-improvement",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert opened.exit_code == 0, opened.stdout
    rabbit_hole_id = next((tmp_path / ".harness" / "research" / "rabbitholes").iterdir()).name
    published = runner.invoke(
        cli_main.app,
        [
            "research",
            "publish",
            "--rabbit-hole",
            rabbit_hole_id,
            "--title",
            "Research completion findings",
            "--summary",
            "Repo-first research plus loop detection improves completion.",
            "--claim",
            "Repo-first research helps finalization.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert published.exit_code == 0, published.stdout
    publication_id = next((tmp_path / ".harness" / "research" / "publications").iterdir()).name
    refined = runner.invoke(
        cli_main.app,
        [
            "research",
            "refine",
            "--title",
            "Research completion fix",
            "--summary",
            "Promote the repo-first research and loop-detection change.",
            "--source-publication",
            publication_id,
            "--target-files",
            "packages/core/src/harness/core/domain_profiles.py,packages/cli/src/harness/cli/research_commands.py",
            "--expected-metric",
            "research smoke pass rate",
            "--validation-plan",
            "Run pytest, pyright, and research smoke.",
            "--cwd",
            str(tmp_path),
            "--mode",
            "improve",
            "--subsystem",
            "research",
            "--rationale",
            "Current research runs over-explore before finalizing.",
            "--expected-outcome",
            "More final structured answers.",
        ],
    )
    assert refined.exit_code == 0, refined.stdout
    candidate_id = next((tmp_path / ".harness" / "research" / "promotions").iterdir()).name

    seen: dict[str, object] = {}

    def fake_ensure_branch(*, cwd: Path, branch_name: str, base_branch: str) -> None:
        seen["branch"] = (cwd, branch_name, base_branch)

    def fake_commit_paths(*, cwd: Path, message: str, paths: tuple[str, ...]) -> None:
        seen["commit"] = (cwd, message, paths)

    def fake_push_branch(*, cwd: Path, branch_name: str, remote: str = "origin") -> None:
        seen["push"] = (cwd, branch_name, remote)

    def fake_create_pull_request(
        *,
        cwd: Path,
        title: str,
        body_path: Path,
        base_branch: str,
        head_branch: str,
        draft: bool,
    ) -> None:
        seen["pr"] = (cwd, title, body_path, base_branch, head_branch, draft)

    monkeypatch.setattr("harness.cli.promotion_commands.ensure_branch", fake_ensure_branch)
    monkeypatch.setattr("harness.cli.promotion_commands.commit_paths", fake_commit_paths)
    monkeypatch.setattr("harness.cli.promotion_commands.push_branch", fake_push_branch)
    monkeypatch.setattr(
        "harness.cli.promotion_commands.create_pull_request", fake_create_pull_request
    )

    promoted = runner.invoke(
        cli_main.app,
        [
            "research",
            "promote",
            "--candidate",
            candidate_id,
            "--commit",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert promoted.exit_code == 0, promoted.stdout
    assert "commit" in seen
    assert (
        tmp_path / ".harness" / "research" / "promotions" / candidate_id / "PR_BODY.md"
    ).is_file()
    shown_candidate = runner.invoke(
        cli_main.app,
        ["research", "show-candidate", candidate_id, "--cwd", str(tmp_path)],
    )
    assert shown_candidate.exit_code == 0, shown_candidate.stdout
    assert "Research completion fix" in shown_candidate.stdout
    shown_candidate_alias = runner.invoke(
        cli_main.app,
        ["research", "candidate", "show", candidate_id, "--cwd", str(tmp_path)],
    )
    assert shown_candidate_alias.exit_code == 0, shown_candidate_alias.stdout

    pr_result = runner.invoke(
        cli_main.app,
        [
            "research",
            "pr",
            "--candidate",
            candidate_id,
            "--push",
            "--open",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert pr_result.exit_code == 0, pr_result.stdout
    assert "branch" in seen
    assert "commit" in seen
    assert "push" in seen
    assert "pr" in seen
    commit_call = seen["commit"]
    assert isinstance(commit_call, tuple)
    assert any("promotion_draft.json" in path for path in commit_call[2])
    assert any("PR_BODY.md" in path for path in commit_call[2])


def test_research_ingest_scout_roles_and_portfolio(tmp_path: Path) -> None:
    runner = CliRunner()

    ingested = runner.invoke(
        cli_main.app,
        [
            "research",
            "ingest-web",
            "--title",
            "Trend note",
            "--url",
            "https://example.com/trend",
            "--summary",
            "Tool registries and research memories are trending.",
            "--themes",
            "autonomous-improvement,research-memory",
            "--related-sections",
            "tools,research",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert ingested.exit_code == 0, ingested.stdout

    scouted = runner.invoke(
        cli_main.app,
        ["research", "scout", "--cwd", str(tmp_path)],
    )
    assert scouted.exit_code == 0, scouted.stdout
    assert "web" in scouted.stdout

    roles = runner.invoke(
        cli_main.app,
        ["research", "roles"],
    )
    assert roles.exit_code == 0, roles.stdout
    assert "section-investigator" in roles.stdout
    assert "promotion-agent" in roles.stdout

    queue = runner.invoke(
        cli_main.app,
        ["research", "queue", "--cwd", str(tmp_path)],
    )
    assert queue.exit_code == 0, queue.stdout
    assert "Research queue is empty" in queue.stdout

    portfolio = runner.invoke(
        cli_main.app,
        ["research", "portfolio", "--cwd", str(tmp_path)],
    )
    assert portfolio.exit_code == 0, portfolio.stdout
    assert "inspiration_notes" in portfolio.stdout

    rebalance = runner.invoke(
        cli_main.app,
        ["research", "rebalance", "--cwd", str(tmp_path)],
    )
    assert rebalance.exit_code == 0, rebalance.stdout
    assert "open_unknowns" in rebalance.stdout


def test_research_experiment_run_show_and_compare(tmp_path: Path) -> None:
    runner = CliRunner()
    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Runtime and research policy",
            "--summary",
            "Research completion depends on runtime profile and repo-first tool scope.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    opportunity_id = next((tmp_path / ".harness" / "research" / "opportunities").iterdir()).name
    hypothesized = runner.invoke(
        cli_main.app,
        [
            "research",
            "hypothesize",
            "--opportunity",
            opportunity_id,
            "--claim",
            "Repo-first research plus loop detection improves completion.",
            "--expected-win",
            "More completed research runs.",
            "--risk-level",
            "low",
            "--change-mode",
            "improve",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert hypothesized.exit_code == 0, hypothesized.stdout
    hypothesis_id = next((tmp_path / ".harness" / "research" / "hypotheses").iterdir()).name
    ok_cmd = f'{shlex.quote(sys.executable)} -c "print(\\"ok\\")"'
    fail_cmd = f'{shlex.quote(sys.executable)} -c "import sys; sys.exit(1)"'
    planned_ok = runner.invoke(
        cli_main.app,
        [
            "research",
            "plan-experiment",
            "--hypothesis",
            hypothesis_id,
            "--plan",
            "Run one successful check.",
            "--checks",
            ok_cmd,
            "--cwd",
            str(tmp_path),
        ],
    )
    assert planned_ok.exit_code == 0, planned_ok.stdout
    planned_fail = runner.invoke(
        cli_main.app,
        [
            "research",
            "plan-experiment",
            "--hypothesis",
            hypothesis_id,
            "--plan",
            "Run one failing check.",
            "--checks",
            fail_cmd,
            "--cwd",
            str(tmp_path),
        ],
    )
    assert planned_fail.exit_code == 0, planned_fail.stdout

    plan_dirs = sorted((tmp_path / ".harness" / "research" / "experiment-plans").iterdir())
    ok_plan_id = plan_dirs[0].name
    fail_plan_id = plan_dirs[1].name

    ran_ok = runner.invoke(
        cli_main.app,
        ["research", "experiment", "run", "--plan", ok_plan_id, "--cwd", str(tmp_path)],
    )
    assert ran_ok.exit_code == 0, ran_ok.stdout
    ran_fail = runner.invoke(
        cli_main.app,
        ["research", "experiment", "run", "--plan", fail_plan_id, "--cwd", str(tmp_path)],
    )
    assert ran_fail.exit_code == 0, ran_fail.stdout

    store = ResearchStore(root=default_research_root(tmp_path))
    experiment_dirs = sorted((tmp_path / ".harness" / "research" / "experiments").iterdir())
    ok_exp_id = ""
    fail_exp_id = ""
    for entry in experiment_dirs:
        result = store.load_experiment_result(entry.name)
        if result.status == "passed":
            ok_exp_id = entry.name
        elif result.status == "failed":
            fail_exp_id = entry.name
    assert ok_exp_id
    assert fail_exp_id

    shown = runner.invoke(
        cli_main.app,
        ["research", "experiment", "show", ok_exp_id, "--cwd", str(tmp_path)],
    )
    assert shown.exit_code == 0, shown.stdout
    assert "passed" in shown.stdout

    compared = runner.invoke(
        cli_main.app,
        ["research", "experiment", "compare", ok_exp_id, fail_exp_id, "--cwd", str(tmp_path)],
    )
    assert compared.exit_code == 0, compared.stdout
    assert "status" in compared.stdout


def test_research_execute_next_seeds_continuation_for_unknown(tmp_path: Path) -> None:
    runner = CliRunner()

    added_theme = runner.invoke(
        cli_main.app,
        [
            "research",
            "add-theme",
            "--title",
            "Autonomous improvement",
            "--description",
            "Study how agents can improve the harness safely.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert added_theme.exit_code == 0, added_theme.stdout
    theme_id = next((tmp_path / ".harness" / "research" / "themes").iterdir()).name

    created_unknown = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-unknown",
            "--theme-id",
            theme_id,
            "--question",
            "What work should the next agent take?",
            "--why-it-matters",
            "The autonomous loop needs a queue item to inspect.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created_unknown.exit_code == 0, created_unknown.stdout

    executed = runner.invoke(
        cli_main.app,
        ["research", "execute-next", "--cwd", str(tmp_path)],
    )
    assert executed.exit_code == 0, executed.stdout
    assert "executed" in executed.stdout
    assert "bounded rabbit hole" in executed.stdout

    rabbit_hole_dir = next((tmp_path / ".harness" / "research" / "rabbitholes").iterdir())
    publication_dir = next((tmp_path / ".harness" / "research" / "publications").iterdir())
    rabbit_hole_json = (rabbit_hole_dir / "rabbit_hole.json").read_text(encoding="utf-8")
    publication_json = (publication_dir / "publication.json").read_text(encoding="utf-8")
    assert "What work should the next agent take?" in rabbit_hole_json
    assert "Continuation plan for What work should the next agent take?" in publication_json


def test_research_execute_next_promotes_top_candidate(tmp_path: Path) -> None:
    runner = CliRunner()

    opened = runner.invoke(
        cli_main.app,
        [
            "research",
            "open",
            "--title",
            "Promotion workflow",
            "--question",
            "How should promotion artifacts be prepared?",
            "--scope",
            "Focus on promotion candidate drafting.",
            "--theme",
            "autonomous-improvement",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert opened.exit_code == 0, opened.stdout
    rabbit_hole_id = next((tmp_path / ".harness" / "research" / "rabbitholes").iterdir()).name

    published = runner.invoke(
        cli_main.app,
        [
            "research",
            "publish",
            "--rabbit-hole",
            rabbit_hole_id,
            "--title",
            "Promotion workflow findings",
            "--summary",
            "Promotion candidates should produce PR-ready artifacts.",
            "--claim",
            "Promotion candidates should write promotion_draft.json and PR_BODY.md.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert published.exit_code == 0, published.stdout
    publication_id = next((tmp_path / ".harness" / "research" / "publications").iterdir()).name

    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Tighten promotion drafts",
            "--summary",
            "Promotion artifacts should be easy to open as PRs.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    opportunity_id = next((tmp_path / ".harness" / "research" / "opportunities").iterdir()).name

    hypothesized = runner.invoke(
        cli_main.app,
        [
            "research",
            "hypothesize",
            "--opportunity",
            opportunity_id,
            "--claim",
            "Explicit promotion artifacts improve reviewer trust.",
            "--expected-win",
            "Clearer autonomous promotion output.",
            "--risk-level",
            "low",
            "--change-mode",
            "improve",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert hypothesized.exit_code == 0, hypothesized.stdout
    hypothesis_id = next((tmp_path / ".harness" / "research" / "hypotheses").iterdir()).name

    refined = runner.invoke(
        cli_main.app,
        [
            "research",
            "refine",
            "--title",
            "Promotion draft evidence section",
            "--summary",
            "Prepare promotion artifacts for the next bounded step.",
            "--source-publication",
            publication_id,
            "--source-hypothesis",
            hypothesis_id,
            "--target-files",
            "README.md,packages/cli/src/harness/cli/research_commands.py",
            "--expected-metric",
            "promotion draft exists",
            "--validation-plan",
            "Inspect generated artifacts.",
            "--risk-level",
            "low",
            "--mode",
            "improve",
            "--subsystem",
            "research",
            "--rationale",
            "Explicit promotion artifacts improve reviewer trust.",
            "--expected-outcome",
            "Generate promotable PR artifacts with clear evidence.",
            "--risk",
            "low",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert refined.exit_code == 0, refined.stdout
    candidate_id = next((tmp_path / ".harness" / "research" / "promotions").iterdir()).name

    executed = runner.invoke(
        cli_main.app,
        ["research", "execute-next", "--cwd", str(tmp_path)],
    )
    assert executed.exit_code == 0, executed.stdout
    assert "executed" in executed.stdout
    assert candidate_id in executed.stdout

    candidate_dir = tmp_path / ".harness" / "research" / "promotions" / candidate_id
    assert (candidate_dir / "promotion_draft.json").is_file()
    assert (candidate_dir / "PR_BODY.md").is_file()
    assert (candidate_dir / "promotion_review.json").is_file()
    review_json = (candidate_dir / "promotion_review.json").read_text(encoding="utf-8")
    assert '"status": "passed"' in review_json


def test_research_execute_next_defers_weak_candidate_after_review(tmp_path: Path) -> None:
    runner = CliRunner()

    opened = runner.invoke(
        cli_main.app,
        [
            "research",
            "open",
            "--title",
            "Weak promotion workflow",
            "--question",
            "How should weak promotion candidates be stopped?",
            "--scope",
            "Focus on autonomous review before PR drafting.",
            "--theme",
            "autonomous-improvement",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert opened.exit_code == 0, opened.stdout
    rabbit_hole_id = next((tmp_path / ".harness" / "research" / "rabbitholes").iterdir()).name

    published = runner.invoke(
        cli_main.app,
        [
            "research",
            "publish",
            "--rabbit-hole",
            rabbit_hole_id,
            "--title",
            "Weak promotion findings",
            "--summary",
            "The runner should defer candidates that lack review metadata.",
            "--claim",
            "Missing validation metadata should block autonomous promotion.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert published.exit_code == 0, published.stdout
    publication_id = next((tmp_path / ".harness" / "research" / "publications").iterdir()).name

    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Stop weak candidates",
            "--summary",
            "Candidates missing evidence fields should defer safely.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    opportunity_id = next((tmp_path / ".harness" / "research" / "opportunities").iterdir()).name

    hypothesized = runner.invoke(
        cli_main.app,
        [
            "research",
            "hypothesize",
            "--opportunity",
            opportunity_id,
            "--claim",
            "Candidates need review metadata before promotion.",
            "--expected-win",
            "Unsafe promotion attempts are deferred.",
            "--risk-level",
            "low",
            "--change-mode",
            "improve",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert hypothesized.exit_code == 0, hypothesized.stdout
    hypothesis_id = next((tmp_path / ".harness" / "research" / "hypotheses").iterdir()).name

    refined = runner.invoke(
        cli_main.app,
        [
            "research",
            "refine",
            "--title",
            "Underspecified candidate",
            "--summary",
            "Intentionally omit the evidence fields so review must defer it.",
            "--source-publication",
            publication_id,
            "--source-hypothesis",
            hypothesis_id,
            "--risk-level",
            "low",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert refined.exit_code == 0, refined.stdout
    candidate_id = next((tmp_path / ".harness" / "research" / "promotions").iterdir()).name

    executed = runner.invoke(
        cli_main.app,
        ["research", "execute-next", "--cwd", str(tmp_path)],
    )
    assert executed.exit_code == 0, executed.stdout
    assert "deferred" in executed.stdout
    assert "missing an expected metric" in executed.stdout
    assert "missing a validation plan" in executed.stdout
    assert candidate_id in executed.stdout
    assert "Created follow-up opportunity" in executed.stdout

    candidate_dir = tmp_path / ".harness" / "research" / "promotions" / candidate_id
    assert (candidate_dir / "promotion_draft.json").is_file()
    assert (candidate_dir / "PR_BODY.md").is_file()
    review_json = (candidate_dir / "promotion_review.json").read_text(encoding="utf-8")
    assert '"status": "failed"' in review_json
    assert "candidate is missing an expected metric" in review_json
    opportunity_json = ""
    for entry in (tmp_path / ".harness" / "research" / "opportunities").iterdir():
        payload = (entry / "opportunity.json").read_text(encoding="utf-8")
        if "Revise Underspecified candidate" in payload:
            opportunity_json = payload
            break
    assert "Revise Underspecified candidate" in opportunity_json
    assert "Resolve autonomous promotion review findings" in opportunity_json

    follow_up = runner.invoke(
        cli_main.app,
        ["research", "execute-next", "--cwd", str(tmp_path)],
    )
    assert follow_up.exit_code == 0, follow_up.stdout
    assert "bounded hypothesis" in follow_up.stdout


def test_research_execute_next_narrows_opportunity_to_hypothesis(tmp_path: Path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Verifier routing improvements",
            "--summary",
            "Reduce verifier noise by tightening routing decisions.",
            "--change-modes",
            "improve",
            "--priority",
            "high",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    opportunity_id = next((tmp_path / ".harness" / "research" / "opportunities").iterdir()).name

    executed = runner.invoke(
        cli_main.app,
        ["research", "execute-next", "--cwd", str(tmp_path)],
    )
    assert executed.exit_code == 0, executed.stdout
    assert "executed" in executed.stdout
    assert "bounded hypothesis" in executed.stdout
    assert opportunity_id in executed.stdout

    hypothesis_dir = next((tmp_path / ".harness" / "research" / "hypotheses").iterdir())
    hypothesis_json = (hypothesis_dir / "hypothesis.json").read_text(encoding="utf-8")
    assert "Verifier routing improvements" in hypothesis_json
    assert "Reduce verifier noise by tightening routing decisions." in hypothesis_json
    assert '"change_mode": "improve"' in hypothesis_json


def test_research_execute_next_turns_hypothesis_into_plan(tmp_path: Path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Research completion policy",
            "--summary",
            "Tighten repo-first completion policy for research runs.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    opportunity_id = next((tmp_path / ".harness" / "research" / "opportunities").iterdir()).name

    hypothesized = runner.invoke(
        cli_main.app,
        [
            "research",
            "hypothesize",
            "--opportunity",
            opportunity_id,
            "--claim",
            "Repo-first completion rules improve research reliability.",
            "--expected-win",
            "More deterministic research outcomes.",
            "--risk-level",
            "low",
            "--change-mode",
            "improve",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert hypothesized.exit_code == 0, hypothesized.stdout
    hypothesis_id = next((tmp_path / ".harness" / "research" / "hypotheses").iterdir()).name

    executed = runner.invoke(
        cli_main.app,
        ["research", "execute-next", "--cwd", str(tmp_path)],
    )
    assert executed.exit_code == 0, executed.stdout
    assert "executed" in executed.stdout
    assert "experiment plan" in executed.stdout
    assert hypothesis_id in executed.stdout

    plan_dir = next((tmp_path / ".harness" / "research" / "experiment-plans").iterdir())
    plan_json = (plan_dir / "experiment_plan.json").read_text(encoding="utf-8")
    assert hypothesis_id in plan_json
    assert (
        "Test the hypothesis: Repo-first completion rules improve research reliability."
        in plan_json
    )
    assert '"checks": [' in plan_json
    assert '"pytest"' in plan_json
    assert '"workflow-smoke"' in plan_json


def test_research_execute_next_runs_experiment_plan(tmp_path: Path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Experiment execution policy",
            "--summary",
            "Run one bounded experiment plan automatically.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    opportunity_id = next((tmp_path / ".harness" / "research" / "opportunities").iterdir()).name

    hypothesized = runner.invoke(
        cli_main.app,
        [
            "research",
            "hypothesize",
            "--opportunity",
            opportunity_id,
            "--claim",
            "One bounded experiment run is safe enough for autonomous execution.",
            "--expected-win",
            "The autonomous loop can validate a testable step.",
            "--risk-level",
            "low",
            "--change-mode",
            "improve",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert hypothesized.exit_code == 0, hypothesized.stdout
    hypothesis_id = next((tmp_path / ".harness" / "research" / "hypotheses").iterdir()).name

    ok_cmd = f'{shlex.quote(sys.executable)} -c "print(\\"ok\\")"'
    planned = runner.invoke(
        cli_main.app,
        [
            "research",
            "plan-experiment",
            "--hypothesis",
            hypothesis_id,
            "--plan",
            "Run a single passing check.",
            "--checks",
            ok_cmd,
            "--cwd",
            str(tmp_path),
        ],
    )
    assert planned.exit_code == 0, planned.stdout
    plan_id = next((tmp_path / ".harness" / "research" / "experiment-plans").iterdir()).name

    executed = runner.invoke(
        cli_main.app,
        ["research", "execute-next", "--cwd", str(tmp_path)],
    )
    assert executed.exit_code == 0, executed.stdout
    assert "executed" in executed.stdout
    assert "experiment plan" in executed.stdout
    assert plan_id in executed.stdout

    experiment_id = next((tmp_path / ".harness" / "research" / "experiments").iterdir()).name
    result_json = (
        tmp_path / ".harness" / "research" / "experiments" / experiment_id / "result.json"
    ).read_text(encoding="utf-8")
    assert '"status": "passed"' in result_json
    result_payload = json.loads(result_json)
    assert result_payload["command_results"][0]["command"] == ok_cmd


def test_research_execute_next_turns_passed_experiment_into_candidate(tmp_path: Path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Promotion from experiment evidence",
            "--summary",
            "A passed experiment should become a promotion candidate.",
            "--related-sections",
            "research,autonomy",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    opportunity_id = next((tmp_path / ".harness" / "research" / "opportunities").iterdir()).name

    hypothesized = runner.invoke(
        cli_main.app,
        [
            "research",
            "hypothesize",
            "--opportunity",
            opportunity_id,
            "--claim",
            "Passed experiments provide enough evidence to draft a promotion candidate.",
            "--expected-win",
            "The loop can advance from evidence to promotion artifacts.",
            "--risk-level",
            "low",
            "--change-mode",
            "improve",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert hypothesized.exit_code == 0, hypothesized.stdout
    hypothesis_id = next((tmp_path / ".harness" / "research" / "hypotheses").iterdir()).name

    ok_cmd = f'{shlex.quote(sys.executable)} -c "print(\\"ok\\")"'
    planned = runner.invoke(
        cli_main.app,
        [
            "research",
            "plan-experiment",
            "--hypothesis",
            hypothesis_id,
            "--plan",
            "Run a passing experiment that can be promoted.",
            "--target-files",
            "packages/core/src/harness/core/autonomy.py",
            "--checks",
            ok_cmd,
            "--cwd",
            str(tmp_path),
        ],
    )
    assert planned.exit_code == 0, planned.stdout

    first_execution = runner.invoke(
        cli_main.app,
        ["research", "execute-next", "--cwd", str(tmp_path)],
    )
    assert first_execution.exit_code == 0, first_execution.stdout
    assert "experiment plan" in first_execution.stdout

    second_execution = runner.invoke(
        cli_main.app,
        ["research", "execute-next", "--cwd", str(tmp_path)],
    )
    assert second_execution.exit_code == 0, second_execution.stdout
    assert "promotion candidate" in second_execution.stdout

    candidate_dir = next((tmp_path / ".harness" / "research" / "promotions").iterdir())
    candidate_json = (candidate_dir / "promotion_candidate.json").read_text(encoding="utf-8")
    assert "Promotion from experiment evidence promotion candidate" in candidate_json
    assert hypothesis_id in candidate_json
    assert "packages/core/src/harness/core/autonomy.py" in candidate_json
    assert '"mode": "improve"' in candidate_json
    assert '"subsystem": "research"' in candidate_json


def test_research_execute_burst_runs_until_max_steps(tmp_path: Path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Burst execution",
            "--summary",
            "Advance several bounded steps automatically.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout

    executed = runner.invoke(
        cli_main.app,
        ["research", "execute-burst", "--max-steps", "3", "--cwd", str(tmp_path)],
    )
    assert executed.exit_code == 0, executed.stdout
    assert "paused" in executed.stdout
    assert "stop_reason=max_steps" in executed.stdout
    assert "bounded hypothesis" in executed.stdout
    assert "experiment plan" in executed.stdout
    assert "recorded a failed result" in executed.stdout


def test_research_execute_burst_stops_on_deferred_review(tmp_path: Path) -> None:
    runner = CliRunner()

    opened = runner.invoke(
        cli_main.app,
        [
            "research",
            "open",
            "--title",
            "Burst weak promotion workflow",
            "--question",
            "How should burst mode stop on weak promotion candidates?",
            "--scope",
            "Focus on autonomous review before PR drafting.",
            "--theme",
            "autonomous-improvement",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert opened.exit_code == 0, opened.stdout
    rabbit_hole_id = next((tmp_path / ".harness" / "research" / "rabbitholes").iterdir()).name

    published = runner.invoke(
        cli_main.app,
        [
            "research",
            "publish",
            "--rabbit-hole",
            rabbit_hole_id,
            "--title",
            "Burst weak promotion findings",
            "--summary",
            "Burst mode should stop when promotion review fails.",
            "--claim",
            "Missing validation metadata should block autonomous promotion.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert published.exit_code == 0, published.stdout
    publication_id = next((tmp_path / ".harness" / "research" / "publications").iterdir()).name

    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Burst stop weak candidates",
            "--summary",
            "Candidates missing evidence fields should defer safely.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    opportunity_id = next((tmp_path / ".harness" / "research" / "opportunities").iterdir()).name

    hypothesized = runner.invoke(
        cli_main.app,
        [
            "research",
            "hypothesize",
            "--opportunity",
            opportunity_id,
            "--claim",
            "Candidates need review metadata before promotion.",
            "--expected-win",
            "Unsafe promotion attempts are deferred.",
            "--risk-level",
            "low",
            "--change-mode",
            "improve",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert hypothesized.exit_code == 0, hypothesized.stdout
    hypothesis_id = next((tmp_path / ".harness" / "research" / "hypotheses").iterdir()).name

    refined = runner.invoke(
        cli_main.app,
        [
            "research",
            "refine",
            "--title",
            "Burst underspecified candidate",
            "--summary",
            "Intentionally omit the evidence fields so review must defer it.",
            "--source-publication",
            publication_id,
            "--source-hypothesis",
            hypothesis_id,
            "--risk-level",
            "low",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert refined.exit_code == 0, refined.stdout

    executed = runner.invoke(
        cli_main.app,
        ["research", "execute-burst", "--max-steps", "2", "--cwd", str(tmp_path)],
    )
    assert executed.exit_code == 0, executed.stdout
    assert "paused" in executed.stdout
    assert "stop_reason=deferred" in executed.stdout
    assert "failed autonomous review" in executed.stdout
    assert "Created follow-up opportunity" in executed.stdout


def test_research_schedule_once_persists_run_record(tmp_path: Path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Scheduled burst execution",
            "--summary",
            "Persist an unattended burst run.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout

    scheduled = runner.invoke(
        cli_main.app,
        ["research", "schedule-once", "--max-steps", "2", "--cwd", str(tmp_path)],
    )
    assert scheduled.exit_code == 0, scheduled.stdout
    assert "record_dir=" in scheduled.stdout

    runs_dir = tmp_path / ".harness" / "research" / "autonomy-runs"
    run_dir = next(runs_dir.iterdir())
    run_json = (run_dir / "run.json").read_text(encoding="utf-8")
    assert '"mode": "burst"' in run_json
    assert '"steps_run": 2' in run_json
    assert (run_dir / "RUN.md").is_file()


def test_research_list_runs_shows_persisted_records(tmp_path: Path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Burst listing",
            "--summary",
            "Persist a run that can be listed later.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout

    scheduled = runner.invoke(
        cli_main.app,
        ["research", "schedule-once", "--max-steps", "1", "--cwd", str(tmp_path)],
    )
    assert scheduled.exit_code == 0, scheduled.stdout

    listed = runner.invoke(
        cli_main.app,
        ["research", "list-runs", "--cwd", str(tmp_path)],
    )
    assert listed.exit_code == 0, listed.stdout
    assert "burst" in listed.stdout
    assert "paused" in listed.stdout


def test_research_show_run_renders_persisted_record(tmp_path: Path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Show run",
            "--summary",
            "Persist a run that can be rendered later.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout

    scheduled = runner.invoke(
        cli_main.app,
        ["research", "schedule-once", "--max-steps", "1", "--cwd", str(tmp_path)],
    )
    assert scheduled.exit_code == 0, scheduled.stdout

    run_dir = next((tmp_path / ".harness" / "research" / "autonomy-runs").iterdir())
    shown = runner.invoke(
        cli_main.app,
        ["research", "show-run", run_dir.name, "--cwd", str(tmp_path)],
    )
    assert shown.exit_code == 0, shown.stdout
    assert run_dir.name in shown.stdout
    assert "mode=burst" in shown.stdout
    assert "1. executed" in shown.stdout

    shown_json = runner.invoke(
        cli_main.app,
        ["research", "show-run", run_dir.name, "--cwd", str(tmp_path), "--json"],
    )
    assert shown_json.exit_code == 0, shown_json.stdout
    assert '"mode": "burst"' in shown_json.stdout


def test_research_schedule_once_uses_scheduler_config_defaults(tmp_path: Path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Configured schedule",
            "--summary",
            "Use scheduler defaults from config.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [research_scheduler]
        max_steps = 1
        max_risk = "low"
        base_branch = "develop"
        """,
        encoding="utf-8",
    )

    scheduled = runner.invoke(
        cli_main.app,
        [
            "research",
            "schedule-once",
            "--config",
            str(config_path),
            "--cwd",
            str(tmp_path),
        ],
    )
    assert scheduled.exit_code == 0, scheduled.stdout

    run_dir = next((tmp_path / ".harness" / "research" / "autonomy-runs").iterdir())
    run_json = (run_dir / "run.json").read_text(encoding="utf-8")
    assert '"steps_run": 1' in run_json


def test_research_schedule_once_json_output_is_parseable(tmp_path: Path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "JSON schedule",
            "--summary",
            "Emit parseable JSON even when record paths are long.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [research_scheduler]
        max_steps = 1
        max_risk = "low"
        """,
        encoding="utf-8",
    )

    scheduled = runner.invoke(
        cli_main.app,
        [
            "research",
            "schedule-once",
            "--config",
            str(config_path),
            "--cwd",
            str(tmp_path),
            "--json",
        ],
    )
    assert scheduled.exit_code == 0, scheduled.stdout
    payload = json.loads(scheduled.stdout)
    assert payload["result"]["steps_run"] == 1
    assert payload["record_dir"]


def test_research_schedule_once_can_invoke_mutation_hooks_from_config(
    tmp_path: Path, monkeypatch
) -> None:
    runner = CliRunner()

    opened = runner.invoke(
        cli_main.app,
        [
            "research",
            "open",
            "--title",
            "Mutation schedule",
            "--question",
            "How should scheduled mutation mode consume a promotable candidate?",
            "--scope",
            "Seed a bounded promotion candidate.",
            "--theme",
            "autonomous-improvement",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert opened.exit_code == 0, opened.stdout
    rabbit_hole_id = next((tmp_path / ".harness" / "research" / "rabbitholes").iterdir()).name

    published = runner.invoke(
        cli_main.app,
        [
            "research",
            "publish",
            "--rabbit-hole",
            rabbit_hole_id,
            "--title",
            "Mutation schedule findings",
            "--summary",
            "A seeded candidate should exercise the mutation hooks.",
            "--claim",
            "Config-driven schedule-once should honor branch creation.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert published.exit_code == 0, published.stdout
    publication_id = next((tmp_path / ".harness" / "research" / "publications").iterdir()).name

    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Mutation seed opportunity",
            "--summary",
            "Seed a promotable item for schedule-once.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    opportunity_id = next((tmp_path / ".harness" / "research" / "opportunities").iterdir()).name

    hypothesized = runner.invoke(
        cli_main.app,
        [
            "research",
            "hypothesize",
            "--opportunity",
            opportunity_id,
            "--claim",
            "A seeded promotion candidate should let schedule-once exercise branch creation.",
            "--expected-win",
            "Mutation policy can be tested deterministically.",
            "--risk-level",
            "low",
            "--change-mode",
            "improve",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert hypothesized.exit_code == 0, hypothesized.stdout
    hypothesis_id = next((tmp_path / ".harness" / "research" / "hypotheses").iterdir()).name

    refined = runner.invoke(
        cli_main.app,
        [
            "research",
            "refine",
            "--title",
            "Mutation seeded candidate",
            "--summary",
            "Seed a promotable candidate for mutation-mode schedule-once.",
            "--source-publication",
            publication_id,
            "--source-hypothesis",
            hypothesis_id,
            "--target-files",
            "README.md",
            "--expected-metric",
            "promotion draft exists",
            "--validation-plan",
            "Inspect generated artifacts.",
            "--risk-level",
            "low",
            "--mode",
            "improve",
            "--subsystem",
            "research",
            "--rationale",
            "Exercise config-driven branch creation safely.",
            "--expected-outcome",
            "Mutation-mode schedule-once invokes ensure_branch.",
            "--risk",
            "low",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert refined.exit_code == 0, refined.stdout

    seen: dict[str, object] = {}

    def fake_ensure_branch(*, cwd: Path, branch_name: str, base_branch: str) -> None:
        seen["branch"] = (cwd, branch_name, base_branch)

    monkeypatch.setattr("harness.core.autonomy.ensure_branch", fake_ensure_branch)

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [research_scheduler]
        max_steps = 1
        max_risk = "low"
        base_branch = "develop"
        create_branch = true
        commit = false
        push = false
        open_pr = false
        draft_pr = true
        """,
        encoding="utf-8",
    )

    scheduled = runner.invoke(
        cli_main.app,
        [
            "research",
            "schedule-once",
            "--config",
            str(config_path),
            "--cwd",
            str(tmp_path),
        ],
    )
    assert scheduled.exit_code == 0, scheduled.stdout
    assert "record_dir=" in scheduled.stdout
    assert "branch" in seen
    branch_call = seen["branch"]
    assert isinstance(branch_call, tuple)
    assert branch_call[2] == "develop"


def test_research_run_subcommand_uses_research_domain(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_run_once(**kwargs: object) -> str:
        captured.update(kwargs)
        return (
            '{"summary":"Two key tradeoffs.","findings":["SQLite is simpler."],'
            '"open_questions":["How much concurrency is needed?"],'
            '"sources":[{"title":"SQLite docs","url":"https://sqlite.org","excerpt":"Reliable embedded DB."}]}'
        )

    monkeypatch.setattr(
        "harness.cli.research_commands._run_async",
        lambda awaitable: __import__("asyncio").run(awaitable),
    )
    monkeypatch.setattr(
        "harness.cli.research_commands._load_cli_config", lambda _path: HarnessConfig()
    )
    monkeypatch.setattr(
        "harness.cli.research_commands._resolve_chain", lambda **_kwargs: ["ollama"]
    )
    monkeypatch.setattr("harness.cli.run_commands.run_once", fake_run_once)

    result = CliRunner().invoke(
        cli_main.app,
        [
            "research",
            "run",
            "compare sqlite and postgres",
            "--cwd",
            str(tmp_path),
            "--in-memory",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert captured["domain"] == "research"
    assert "Two key tradeoffs." in result.stdout
