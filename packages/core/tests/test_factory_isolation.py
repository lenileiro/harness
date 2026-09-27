from __future__ import annotations

import os
import subprocess
import textwrap
from pathlib import Path

import pytest

from harness.core.factory.isolation import (
    BRANCH_PREFIX,
    IsolationError,
    branch_has_commits,
    create_worktree,
    default_worktree_root,
    list_factory_worktrees,
    reap_stale_worktrees,
    remove_worktree,
    shift_workspace,
    verify_isolation,
    worktree_env,
)


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway git repo with one commit on main."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(["init", "-b", "main"], root)
    _git(["config", "user.email", "t@example.com"], root)
    _git(["config", "user.name", "T"], root)
    (root / "pkg").mkdir()
    (root / "pkg" / "mod.py").write_text("VALUE = 1\n")
    _git(["add", "-A"], root)
    _git(["commit", "-m", "init"], root)
    return root


def _fake_sync(*, points_at: Path | None = None):
    """Build a .venv whose editable .pth points wherever we say.

    ``points_at=None`` means "point into the worktree", i.e. correct isolation.
    Anything else simulates the silent-green bug.
    """

    def sync(worktree: Path) -> None:
        site = worktree / ".venv" / "lib" / "python3.12" / "site-packages"
        site.mkdir(parents=True)
        target = points_at if points_at is not None else (worktree / "pkg")
        (site / "_editable_impl_pkg.pth").write_text(f"{target}\n")

    return sync


# --------------------------------------------------------------- verify ---


def test_verify_isolation_accepts_a_worktree_local_venv(repo, tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    _fake_sync()(wt)
    verify_isolation(wt)  # does not raise


def test_verify_isolation_rejects_a_pth_pointing_at_another_checkout(repo, tmp_path):
    """The bug this module exists to prevent, caught rather than silently passing."""
    wt = tmp_path / "wt"
    wt.mkdir()
    _fake_sync(points_at=repo / "pkg")(wt)
    with pytest.raises(IsolationError, match="points outside the worktree"):
        verify_isolation(wt)


def test_verify_isolation_rejects_a_missing_venv(tmp_path):
    wt = tmp_path / "wt"
    wt.mkdir()
    with pytest.raises(IsolationError, match=r"no \.venv"):
        verify_isolation(wt)


def test_verify_isolation_rejects_a_venv_with_no_editable_installs(tmp_path):
    wt = tmp_path / "wt"
    (wt / ".venv" / "lib" / "python3.12" / "site-packages").mkdir(parents=True)
    with pytest.raises(IsolationError, match="not installed"):
        verify_isolation(wt)


# --------------------------------------------------------------- create ---


def test_create_worktree_makes_a_branch_and_verifies_isolation(repo, tmp_path):
    wt = create_worktree(repo, name="shift-1", root=tmp_path / "wts", sync=_fake_sync())
    assert wt.branch == f"{BRANCH_PREFIX}shift-1"
    assert (wt.path / "pkg" / "mod.py").exists()
    assert wt.venv.is_dir()
    assert list_factory_worktrees(repo, tmp_path / "wts") == [wt.path.resolve()]


def test_create_worktree_rolls_back_when_the_venv_is_not_isolated(repo, tmp_path):
    """A bad sync must leave no worktree and no branch behind."""
    with pytest.raises(IsolationError):
        create_worktree(
            repo,
            name="shift-bad",
            root=tmp_path / "wts",
            sync=_fake_sync(points_at=repo / "pkg"),
        )
    assert list_factory_worktrees(repo, tmp_path / "wts") == []
    branches = subprocess.run(
        ["git", "branch", "--list", f"{BRANCH_PREFIX}shift-bad"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert branches.strip() == ""


def test_create_worktree_rolls_back_when_sync_raises(repo, tmp_path):
    def boom(_: Path) -> None:
        raise IsolationError("uv sync failed")

    with pytest.raises(IsolationError):
        create_worktree(repo, name="shift-2", root=tmp_path / "wts", sync=boom)
    assert list_factory_worktrees(repo, tmp_path / "wts") == []


def test_create_worktree_refuses_to_reuse_an_existing_path(repo, tmp_path):
    root = tmp_path / "wts"
    create_worktree(repo, name="shift-3", root=root, sync=_fake_sync())
    with pytest.raises(IsolationError, match="already exists"):
        create_worktree(repo, name="shift-3", root=root, sync=_fake_sync())


# ------------------------------------------------------------ teardown ---


def test_shift_workspace_cleans_up_even_when_the_body_raises(repo, tmp_path):
    root = tmp_path / "wts"
    with (
        pytest.raises(RuntimeError, match="shift exploded"),
        shift_workspace(repo, name="shift-5", root=root, sync=_fake_sync()),
    ):
        raise RuntimeError("shift exploded")
    assert list_factory_worktrees(repo, root) == []


# ----------------------------------------------------------------- reap ---


def test_reap_removes_orphans_older_than_the_window_and_spares_fresh_ones(repo, tmp_path):
    """A SIGKILL bypasses cleanup, so orphans are expected, not exceptional."""
    root = tmp_path / "wts"
    old = create_worktree(repo, name="old", root=root, sync=_fake_sync())
    fresh = create_worktree(repo, name="fresh", root=root, sync=_fake_sync())

    # back-date `old` so the two are unambiguously on opposite sides of the window
    stale = os.stat(fresh.path).st_mtime - 7200
    os.utime(old.path, (stale, stale))

    reaped = reap_stale_worktrees(repo, root=root, stale_after_seconds=3600)

    assert reaped == (old.path.resolve(),)
    assert not old.path.exists()
    assert fresh.path.exists()
    assert list_factory_worktrees(repo, root) == [fresh.path.resolve()]


def test_reap_keeps_the_branch_of_an_orphan_that_produced_work(repo, tmp_path):
    """A killed shift may still have committed something worth reviewing."""
    root = tmp_path / "wts"
    wt = create_worktree(repo, name="orphan-with-work", root=root, sync=_fake_sync())
    (wt.path / "pkg" / "mod.py").write_text("VALUE = 9\n")
    _git(["add", "-A"], wt.path)
    _git(["commit", "-m", "factory: fix"], wt.path)

    reap_stale_worktrees(repo, root=root, stale_after_seconds=0)
    branches = subprocess.run(
        ["git", "branch", "--list", wt.branch],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert wt.branch in branches


def test_reap_drops_the_branch_of_an_orphan_that_produced_nothing(repo, tmp_path):
    """Otherwise every OOM-killed shift leaves a dead branch behind."""
    root = tmp_path / "wts"
    wt = create_worktree(repo, name="orphan-empty", root=root, sync=_fake_sync())
    reap_stale_worktrees(repo, root=root, stale_after_seconds=0)
    branches = subprocess.run(
        ["git", "branch", "--list", wt.branch],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert branches.strip() == ""


# ------------------------------------------------------------------ env ---


def test_worktree_env_clears_uv_project_environment(repo, tmp_path, monkeypatch):
    """A leaked UV_PROJECT_ENVIRONMENT would reintroduce the whole bug."""
    monkeypatch.setenv("UV_PROJECT_ENVIRONMENT", "/some/other/.venv")
    wt = create_worktree(repo, name="shift-6", root=tmp_path / "wts", sync=_fake_sync())
    env = worktree_env(wt)
    assert "UV_PROJECT_ENVIRONMENT" not in env
    assert env["VIRTUAL_ENV"] == str(wt.venv)


def test_default_worktree_root_is_outside_the_repo(repo):
    root = default_worktree_root(repo)
    assert not root.is_relative_to(repo.resolve())


# ----------------------------------------------------------------- live ---


@pytest.mark.skipif(
    not os.environ.get("HARNESS_FACTORY_LIVE_WORKTREE"),
    reason="requires a real `uv sync` in a fresh worktree (minutes, network/cache)",
)
def test_live_worktree_runs_its_own_source(tmp_path):
    """The acceptance experiment.

    Append a marker to the worktree's copy of a source file and prove a test run
    *inside* that worktree sees it. This is the exact check that caught the
    shared-venv bug: with a shared venv the marker is invisible and everything
    still reports green.
    """
    repo = Path(__file__).resolve().parents[3]
    target = "packages/tools-fs/src/harness/tools/fs/__init__.py"

    with shift_workspace(repo, name="live-check", root=tmp_path / "wts") as wt:
        src = wt.path / target
        src.write_text(src.read_text() + "\nMARKER_FROM_WORKTREE = True\n")

        probe = textwrap.dedent(
            """
            import harness.tools.fs as m
            print("FILE:", m.__file__)
            print("MARKER:", getattr(m, "MARKER_FROM_WORKTREE", False))
            """
        )
        proc = subprocess.run(
            [str(wt.venv / "bin" / "python"), "-c", probe],
            cwd=wt.path,
            capture_output=True,
            text=True,
            check=True,
            env=worktree_env(wt),
        )
        assert "MARKER: True" in proc.stdout, proc.stdout
        assert str(wt.path) in proc.stdout, (
            f"worktree imported source from outside itself:\n{proc.stdout}"
        )


def test_remove_worktree_drops_a_branch_that_produced_nothing(repo, tmp_path):
    """A shift that found no work must not leave a dead branch behind.

    Keeping every branch would litter the repo with one empty branch per quiet
    night, which is exactly what happened the first time this was dogfooded.
    """
    root = tmp_path / "wts"
    wt = create_worktree(repo, name="empty-shift", root=root, sync=_fake_sync())
    remove_worktree(repo, wt)
    branches = subprocess.run(
        ["git", "branch", "--list", wt.branch],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert branches.strip() == ""


def test_remove_worktree_keeps_a_branch_that_has_commits(repo, tmp_path):
    """The branch is the shift's output and may already be under review."""
    root = tmp_path / "wts"
    wt = create_worktree(repo, name="productive-shift", root=root, sync=_fake_sync())
    (wt.path / "pkg" / "mod.py").write_text("VALUE = 2\n")
    _git(["add", "-A"], wt.path)
    _git(["commit", "-m", "factory: fix"], wt.path)

    assert branch_has_commits(repo, wt) is True
    remove_worktree(repo, wt)
    branches = subprocess.run(
        ["git", "branch", "--list", wt.branch],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert wt.branch in branches


def test_keep_branch_false_forces_deletion_even_with_commits(repo, tmp_path):
    root = tmp_path / "wts"
    wt = create_worktree(repo, name="forced", root=root, sync=_fake_sync())
    (wt.path / "pkg" / "mod.py").write_text("VALUE = 3\n")
    _git(["add", "-A"], wt.path)
    _git(["commit", "-m", "factory: fix"], wt.path)

    remove_worktree(repo, wt, keep_branch=False)
    branches = subprocess.run(
        ["git", "branch", "--list", wt.branch],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert branches.strip() == ""
