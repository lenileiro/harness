"""Isolated workspaces for factory shifts.

A shift must never run against the user's live checkout: an overnight run would
collide with uncommitted work, and a crash would leave the working tree in an
unknown state. Each shift therefore gets its own ``git worktree``.

The non-obvious part, and the reason this module exists rather than being three
lines inside ``shift.py``:

    A worktree MUST NOT share the main checkout's virtualenv.

This workspace installs its members as editable packages, which uv implements as
``.pth`` files containing **absolute** paths::

    .venv/lib/python3.12/site-packages/_editable_impl_tools_fs.pth
        -> /Users/leiro/workspace/harness/packages/tools-fs/src

Point a worktree at that venv (via ``UV_PROJECT_ENVIRONMENT`` or otherwise) and
pytest collects the *worktree's* tests while importing the *main checkout's*
source. Nothing errors. The suite passes. And every consequence is silently
wrong: mutants planted in the worktree are never imported so they all survive,
and a ``done_command`` can never honestly go red-to-green because the agent's
edits are invisible to the code under test.

Because that failure is invisible, :func:`verify_isolation` asserts it away at
runtime rather than trusting the setup to be correct.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "IsolationError",
    "ShiftWorktree",
    "branch_has_commits",
    "create_worktree",
    "default_worktree_root",
    "list_factory_worktrees",
    "reap_stale_worktrees",
    "remove_worktree",
    "shift_workspace",
    "uv_sync",
    "verify_isolation",
]

BRANCH_PREFIX = "factory/"
_STALE_AFTER_SECONDS = 24 * 60 * 60


class IsolationError(RuntimeError):
    """A shift workspace could not be created, verified, or torn down."""


@dataclass(frozen=True, slots=True)
class ShiftWorktree:
    path: Path
    branch: str
    base: str = "main"

    @property
    def venv(self) -> Path:
        return self.path / ".venv"


def default_worktree_root(repo: Path) -> Path:
    """Where shift worktrees live.

    Deliberately outside the repository. A worktree nested inside its own repo
    invites recursive scanning, confuses tooling that walks the tree, and makes
    ``git clean`` dangerous.
    """
    return Path.home() / ".harness" / "factory-worktrees" / repo.resolve().name


def _git(args: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=False)


def _git_checked(args: list[str], *, cwd: Path) -> str:
    proc = _git(args, cwd=cwd)
    if proc.returncode != 0:
        raise IsolationError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def uv_sync(worktree: Path) -> None:
    """Give the worktree its own virtualenv.

    ``--frozen`` resolves nothing and installs exactly what uv.lock pins, which
    is what CI does and what makes a shift reproducible.
    """
    proc = subprocess.run(
        ["uv", "sync", "--frozen", "--all-packages", "--all-groups"],
        cwd=worktree,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise IsolationError(f"uv sync failed in {worktree}: {proc.stderr.strip()[:500]}")


def verify_isolation(worktree: Path) -> None:
    """Fail loudly if the worktree would import the main checkout's source.

    This is the guard for the silent-green failure described in the module
    docstring. It is cheap, it runs on every shift, and it is the difference
    between a wrong answer and no answer.
    """
    venv = worktree / ".venv"
    if not venv.is_dir():
        raise IsolationError(f"{worktree} has no .venv of its own")

    pth_files = list(venv.glob("lib/python*/site-packages/_editable_impl_*.pth"))
    if not pth_files:
        raise IsolationError(f"{venv} has no editable .pth files; the workspace was not installed")

    worktree = worktree.resolve()
    for pth in pth_files:
        for line in pth.read_text().splitlines():
            target = line.strip()
            if not target or target.startswith(("import ", "#")):
                continue
            resolved = Path(target).resolve()
            if not resolved.is_relative_to(worktree):
                raise IsolationError(
                    f"{pth.name} points outside the worktree: {resolved}. "
                    f"The worktree is sharing another checkout's venv, so tests "
                    f"would run against the wrong source."
                )


def list_factory_worktrees(repo: Path, root: Path | None = None) -> list[Path]:
    """Every registered worktree that lives under the factory root."""
    root = (root or default_worktree_root(repo)).resolve()
    out: list[Path] = []
    for line in _git_checked(["worktree", "list", "--porcelain"], cwd=repo).splitlines():
        if line.startswith("worktree "):
            path = Path(line[len("worktree ") :]).resolve()
            if path != repo.resolve() and path.is_relative_to(root):
                out.append(path)
    return out


def _branch_for_worktree(repo: Path, path: Path) -> str | None:
    """The branch checked out in a worktree, read before it is destroyed."""
    current: Path | None = None
    for line in _git_checked(["worktree", "list", "--porcelain"], cwd=repo).splitlines():
        if line.startswith("worktree "):
            current = Path(line[len("worktree ") :]).resolve()
        elif line.startswith("branch ") and current == path.resolve():
            return line[len("branch refs/heads/") :].strip()
    return None


def reap_stale_worktrees(
    repo: Path,
    *,
    root: Path | None = None,
    stale_after_seconds: float = _STALE_AFTER_SECONDS,
    now: float | None = None,
    base: str = "main",
) -> tuple[Path, ...]:
    """Remove worktrees left behind by a shift that was killed.

    A ``SIGKILL`` bypasses every cleanup path, and this repo has had processes
    killed by the OS for memory pressure, so orphans are expected rather than
    exceptional. Called at shift start, before anything else.

    A branch carrying commits is kept -- a killed shift may still have produced
    work worth reviewing. An empty one is dropped, by the same reasoning as
    :func:`remove_worktree`: otherwise every killed shift leaves a dead branch.
    """
    now = time.time() if now is None else now
    reaped: list[Path] = []
    for path in list_factory_worktrees(repo, root):
        try:
            age = now - path.stat().st_mtime
        except OSError:
            age = stale_after_seconds + 1
        if age <= stale_after_seconds:
            continue
        branch = _branch_for_worktree(repo, path)
        _git(["worktree", "remove", "--force", str(path)], cwd=repo)
        shutil.rmtree(path, ignore_errors=True)
        if branch is not None:
            stub = ShiftWorktree(path=path, branch=branch, base=base)
            if not branch_has_commits(repo, stub):
                _git(["branch", "-D", branch], cwd=repo)
        reaped.append(path)
    _git(["worktree", "prune"], cwd=repo)
    return tuple(reaped)


def create_worktree(
    repo: Path,
    *,
    name: str,
    base: str = "main",
    root: Path | None = None,
    sync: Callable[[Path], None] | None = None,
) -> ShiftWorktree:
    """Create a worktree with its own virtualenv, and prove it is isolated."""
    root = root or default_worktree_root(repo)
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    branch = f"{BRANCH_PREFIX}{name}"

    if path.exists():
        raise IsolationError(f"{path} already exists; reap it before reusing the name")

    _git_checked(["worktree", "add", "-b", branch, str(path), base], cwd=repo)
    try:
        (sync or uv_sync)(path)
        verify_isolation(path)
    except Exception:
        _git(["worktree", "remove", "--force", str(path)], cwd=repo)
        _git(["branch", "-D", branch], cwd=repo)
        raise
    return ShiftWorktree(path=path, branch=branch, base=base)


def branch_has_commits(repo: Path, worktree: ShiftWorktree) -> bool:
    """Did this shift actually produce anything?"""
    proc = _git(["rev-list", "--count", f"{worktree.base}..{worktree.branch}"], cwd=repo)
    if proc.returncode != 0:
        return True  # cannot tell -> keep it, losing work is worse than litter
    return proc.stdout.strip() not in ("", "0")


def remove_worktree(
    repo: Path, worktree: ShiftWorktree, *, keep_branch: bool | None = None
) -> None:
    """Tear the worktree down.

    A branch carrying commits is the shift's output and may already be under
    review, so it is kept. An empty branch is litter: a shift that found no work
    would otherwise leave a dead branch behind every night. ``keep_branch``
    forces the decision either way.
    """
    empty = not branch_has_commits(repo, worktree)
    _git(["worktree", "remove", "--force", str(worktree.path)], cwd=repo)
    shutil.rmtree(worktree.path, ignore_errors=True)
    if keep_branch is False or (keep_branch is None and empty):
        _git(["branch", "-D", worktree.branch], cwd=repo)
    _git(["worktree", "prune"], cwd=repo)


@contextmanager
def shift_workspace(
    repo: Path,
    *,
    name: str,
    base: str = "main",
    root: Path | None = None,
    sync: Callable[[Path], None] | None = None,
    keep_branch: bool | None = None,
) -> Iterator[ShiftWorktree]:
    """Reap orphans, create a verified workspace, tear it down afterwards."""
    reap_stale_worktrees(repo, root=root)
    worktree = create_worktree(repo, name=name, base=base, root=root, sync=sync)
    try:
        yield worktree
    finally:
        remove_worktree(repo, worktree, keep_branch=keep_branch)


def worktree_env(worktree: ShiftWorktree) -> dict[str, str]:
    """Environment for commands run inside the worktree.

    ``UV_PROJECT_ENVIRONMENT`` is explicitly cleared. If it leaks in from the
    parent environment it would point uv at another checkout's venv and
    reintroduce exactly the silent-green failure this module exists to prevent.
    """
    env = dict(os.environ)
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    env["VIRTUAL_ENV"] = str(worktree.venv)
    return env
