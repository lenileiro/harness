from __future__ import annotations

import sys

import pytest
from evals import runner
from evals.types import FixtureMeta


def test_preparing_git_baseline_does_not_follow_fixture_metadata_links(tmp_path, monkeypatch):
    outside = tmp_path / "private-ignore"
    outside.write_text("private original")
    private_git = tmp_path / "private-git"
    private_git.mkdir()
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / ".gitignore").symlink_to(outside)
    (fixture / ".git").symlink_to(private_git, target_is_directory=True)
    (fixture / "public.py").write_text("pass")
    monkeypatch.setattr(
        runner, "_agent_cmd", lambda *args, **kwargs: [sys.executable, "-c", "pass"]
    )
    outcome = runner.run_fixture(
        FixtureMeta(
            name="task",
            path=fixture,
            task_text="Use public files",
            eval_md="",
            verify_command="true",
        ),
        provider="mock",
        model="mock",
    )
    assert outside.read_text() == "private original"
    assert list(private_git.iterdir()) == []
    assert outcome.agent_exit_code == 0, outcome.transcript


@pytest.mark.parametrize("overlay", [False, True])
def test_workspace_copy_excludes_git_indirection(tmp_path, monkeypatch, overlay):
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    source = fixture / "workspace" if overlay else fixture
    source.mkdir(exist_ok=True)
    (source / ".git").write_text(f"gitdir: {tmp_path / 'private'}\n")
    (source / "public.py").write_text("pass")
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(runner, "_project_root", lambda: project)
    work = tmp_path / "work"
    runner._prepare_workspace_for_run(
        FixtureMeta(name="task", path=fixture, task_text="Use public files", eval_md=""),
        work,
    )
    assert not (work / ".git").exists()
    assert (work / "public.py").is_file()
