from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.core.experiment_plans import ExperimentPlan
from harness.core.experiment_runner import run_experiment_plan
from harness.core.promotion_candidates import PromotionCandidate
from harness.core.research.store import ResearchStore, default_research_root


def _candidate(cwd: Path, *, evidence: bool) -> tuple[ResearchStore, PromotionCandidate]:
    (cwd / "app.py").write_text("VALUE = 2\n")
    store = ResearchStore(root=default_research_root(cwd))
    candidate = PromotionCandidate(
        id="promo", title="Update source", summary="Change VALUE", target_files=("app.py",)
    )
    if evidence:
        code = 'from pathlib import Path; assert Path("app.py").read_text() == "VALUE = 2\\n"'
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"
        plan = ExperimentPlan(
            id="plan",
            hypothesis_id="hyp",
            plan="Verify source",
            target_files=("app.py",),
            checks=(command,),
        )
        store.add_experiment_plan(plan)
        experiment, result = run_experiment_plan(store=store, plan=plan, cwd=cwd)
        assert result.status == "passed"
        candidate = replace(candidate, source_experiments=(experiment.id,))
    store.add_promotion_candidate(candidate)
    return store, candidate


@pytest.mark.parametrize("command", ["promote", "pr"])
def test_draft_preparation_needs_no_execution_evidence(tmp_path: Path, command: str) -> None:
    store, candidate = _candidate(tmp_path, evidence=False)
    flags = ["--no-create-branch"] if command == "promote" else []
    result = CliRunner().invoke(
        cli_main.app,
        ["research", command, "--candidate", candidate.id, *flags, "--cwd", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert (store.promotion_candidates_dir / candidate.id / "PR_BODY.md").exists()


@pytest.mark.parametrize(
    ("command", "flags"), [("promote", ["--commit"]), ("pr", ["--push", "--open"])]
)
def test_missing_evidence_blocks_external_actions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: str, flags: list[str]
) -> None:
    _, candidate = _candidate(tmp_path, evidence=False)
    invoked = []
    for name in ("ensure_branch", "commit_paths", "push_branch", "create_pull_request"):
        monkeypatch.setattr(
            f"harness.cli.promotion_commands.{name}", lambda **kwargs: invoked.append(kwargs)
        )
    result = CliRunner().invoke(
        cli_main.app,
        ["research", command, "--candidate", candidate.id, *flags, "--cwd", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "no linked experiment evidence" in result.output
    assert invoked == []


@pytest.mark.parametrize(
    ("mutating_stage", "blocked_stage"),
    [
        ("ensure_branch", "commit_paths"),
        ("commit_paths", "push_branch"),
        ("push_branch", "create_pull_request"),
    ],
)
def test_rechecks_evidence_after_each_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutating_stage: str, blocked_stage: str
) -> None:
    _, candidate = _candidate(tmp_path, evidence=True)
    invoked: list[str] = []

    def stage(name: str):
        def invoke(**kwargs):
            invoked.append(name)
            if name == mutating_stage:
                (tmp_path / "app.py").write_text("VALUE = 3\n")

        return invoke

    for name in ("ensure_branch", "commit_paths", "push_branch", "create_pull_request"):
        monkeypatch.setattr(f"harness.cli.promotion_commands.{name}", stage(name))
    result = CliRunner().invoke(
        cli_main.app,
        ["research", "pr", "--candidate", candidate.id, "--push", "--open", "--cwd", str(tmp_path)],
    )
    assert result.exit_code != 0
    assert "stale" in result.output
    assert mutating_stage in invoked
    assert blocked_stage not in invoked


def test_push_commits_validated_source_while_ignored_drafts_stay_local(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=tmp_path, check=True, capture_output=True, text=True
        ).stdout

    # All git writes are confined to this temporary fixture. Push and PR are mocks.
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    git("init", "-b", "main")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    git("config", "core.hooksPath", "/dev/null")
    (tmp_path / ".gitignore").write_text(".harness/\n")
    (tmp_path / "app.py").write_text("VALUE = 1\n")
    git("add", ".gitignore", "app.py")
    git("commit", "-m", "base")
    store, candidate = _candidate(tmp_path, evidence=True)
    pushed = []
    monkeypatch.setattr(
        "harness.cli.promotion_commands.push_branch", lambda **kwargs: pushed.append(kwargs)
    )
    monkeypatch.setattr(
        "harness.cli.promotion_commands.create_pull_request",
        lambda **kwargs: "https://example.invalid/pr/1",
    )
    result = CliRunner().invoke(
        cli_main.app,
        ["research", "pr", "--candidate", candidate.id, "--push", "--open", "--cwd", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert pushed
    assert git("show", "--pretty=format:", "--name-only", "HEAD").strip() == "app.py"
    assert ".harness" not in git("ls-files")
    assert (store.promotion_candidates_dir / candidate.id / "PR_BODY.md").exists()
