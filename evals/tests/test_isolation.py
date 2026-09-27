from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from evals import isolation, runner
from evals.isolation import prepare_agent_process, run_agent_process


def require_sandbox():
    available = (sys.platform == "darwin" and shutil.which("sandbox-exec")) or (
        sys.platform.startswith("linux") and shutil.which("bwrap")
    )
    if not available:
        pytest.skip("Host lacks OS sandbox; separate unit test verifies fail-closed behavior")


def test_repo_copy_uses_public_allowlist(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "evals/gold").mkdir(parents=True)
    (project / "evals/gold/sentinel").write_text("private")
    (project / "hidden-solutions").mkdir()
    (project / "hidden-solutions/sentinel").write_text("private")
    (project / "packages/demo/src").mkdir(parents=True)
    (project / "packages/demo/src/public.py").write_text("VALUE=1")
    monkeypatch.setattr(runner, "_project_root", lambda: project)
    runner._copy_repo_for_run(tmp_path / "work")
    assert not (tmp_path / "work/evals").exists()
    assert not (tmp_path / "work/hidden-solutions").exists()
    assert (tmp_path / "work/packages/demo/src/public.py").exists()


def test_workspace_copies_preserve_links_without_copying_private_contents(tmp_path, monkeypatch):
    project = tmp_path / "project"
    (project / "packages/demo").mkdir(parents=True)
    hidden = project / "hidden"
    hidden.write_text("evaluator only")
    (project / "packages/demo/escape").symlink_to(hidden)
    monkeypatch.setattr(runner, "_project_root", lambda: project)
    runner._copy_repo_for_run(tmp_path / "repo-work")
    assert (tmp_path / "repo-work/packages/demo/escape").is_symlink()

    fixture = tmp_path / "fixture"
    fixture.mkdir()
    (fixture / "escape").symlink_to(hidden)
    runner._copy_fixture_for_run(fixture, tmp_path / "fixture-work")
    assert (tmp_path / "fixture-work/escape").is_symlink()


def test_agent_process_cannot_read_evaluator_or_parent_files(tmp_path: Path):
    require_sandbox()
    project = tmp_path / "project"
    project.mkdir()
    sentinel = project / "sentinel"
    sentinel.write_text("evaluator only")
    work = tmp_path / "work"
    work.mkdir()
    (work / "escape").symlink_to(sentinel)
    script = (
        "from pathlib import Path; import json, os, subprocess, sys\n"
        "result=[]\n"
        f"for name in [{str(sentinel)!r}, 'escape']:\n"
        " try: Path(name).read_text(); result.append('read')\n"
        " except OSError: result.append('denied')\n"
        "Path('public.txt').write_text('allowed')\n"
        f"child=subprocess.run([sys.executable, '-c', {('from pathlib import Path; Path(' + repr(str(sentinel)) + ').read_text()')!r}], capture_output=True)\n"
        "result.append('denied' if child.returncode else 'read')\n"
        "print(json.dumps(result))\n"
    )
    command, env = prepare_agent_process(
        [sys.executable, "-c", script], work=work, project_root=project
    )
    completed = subprocess.run(command, env=env, cwd=work, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == ["denied", "denied", "denied"]
    assert (work / "public.txt").read_text() == "allowed"
    assert "HARNESS_EVAL_PROJECT_ROOT" not in env


def test_grading_command_is_not_given_to_agent(tmp_path, monkeypatch):
    fixture_dir = tmp_path / "evals/fixtures/task"
    fixture_dir.mkdir(parents=True)
    (fixture_dir / "TASK.md").write_text("Fix the public module.")
    (fixture_dir / "EVAL.md").write_text("primary_dimension: verification")
    (fixture_dir / "fixture.yaml").write_text("verify_command: echo evaluator-sentinel\n")
    (fixture_dir / "module.py").write_text("pass")
    captured = {}

    def command(*args, **kwargs):
        captured.update(kwargs)
        return ["/bin/sh", "-c", "echo done"]

    monkeypatch.setattr(runner, "_agent_cmd", command)
    outcome = runner.run_fixture(
        runner.discover_fixtures(tmp_path / "evals")[0], provider="mock", model="mock"
    )
    assert captured["verify_command"] is None
    assert "evaluator-sentinel" not in outcome.transcript
    assert "evaluator-sentinel" in outcome.test_output


def test_staged_harness_cli_imports_without_evaluator_access(tmp_path):
    require_sandbox()
    project = Path(__file__).resolve().parents[2]
    work = tmp_path / "work"
    work.mkdir()
    command, env = prepare_agent_process(
        [sys.executable, "-m", "harness.cli", "--help"],
        work=work,
        project_root=project,
    )
    completed = run_agent_process(command, env=env, cwd=work, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert "Usage" in completed.stdout
    source_script = "import harness.core.runtime; print(harness.core.runtime.__file__)"
    prepared, env = prepare_agent_process(
        [sys.executable, "-c", source_script],
        work=work,
        project_root=project,
    )
    completed = run_agent_process(prepared, env=env, cwd=work, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert "agent-runtime-" in completed.stdout
    assert str(project / "packages") not in completed.stdout


def test_public_plugin_fixture_uses_staged_cli_without_evaluator_root(tmp_path):
    require_sandbox()
    project = Path(__file__).resolve().parents[2]
    work = tmp_path / "work"
    runner._copy_fixture_for_run(project / "evals/fixtures/07-plugin-experience-flow", work)
    script = (
        "import runpy, os; "
        "assert 'HARNESS_EVAL_PROJECT_ROOT' not in os.environ; "
        "helpers=runpy.run_path('tests/test_plugin_flow.py'); "
        "result=helpers['_run_harness']('--help'); "
        "assert result.returncode == 0, result.stderr; "
        "assert 'Usage' in result.stdout; print('public CLI works')"
    )
    command, env = prepare_agent_process(
        [sys.executable, "-c", script], work=work, project_root=project
    )
    completed = run_agent_process(command, env=env, cwd=work, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "public CLI works"


def test_staged_source_symlinks_cannot_smuggle_evaluator_files(tmp_path):
    require_sandbox()
    project = tmp_path / "project"
    source = project / "packages/demo/src"
    source.mkdir(parents=True)
    sentinel = project / "private-sentinel"
    sentinel.write_text("private")
    (source / "escape").symlink_to(sentinel)
    work = tmp_path / "work"
    work.mkdir()
    script = (
        "from pathlib import Path; import os\n"
        "try: (Path(os.environ['PYTHONPATH'])/'escape').read_text()\n"
        "except OSError: print('denied')\n"
        "else: raise RuntimeError('evaluator file was copied into runtime')\n"
    )
    command, env = prepare_agent_process(
        [sys.executable, "-c", script], work=work, project_root=project
    )
    completed = run_agent_process(command, env=env, cwd=work, timeout=10)
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "denied"


def test_missing_sandbox_fails_closed(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setattr(isolation.sys, "platform", "linux")
    monkeypatch.setattr(isolation.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match=r"require.*bubblewrap"):
        prepare_agent_process(
            [sys.executable, "-c", "print('unsafe')"], work=work, project_root=tmp_path
        )


def test_linux_uses_namespaces_and_narrow_mounts(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(isolation.sys, "platform", "linux")
    original_which = shutil.which
    monkeypatch.setattr(
        isolation.shutil,
        "which",
        lambda name: "/usr/bin/bwrap" if name == "bwrap" else original_which(name),
    )
    command, env = prepare_agent_process(
        [sys.executable, "-c", "pass"], work=work, project_root=project
    )
    assert command[0] == "/usr/bin/bwrap"
    assert "--unshare-pid" in command and "--unshare-user" in command
    assert "--die-with-parent" in command
    mounts = [
        command[index + 1]
        for index, token in enumerate(command)
        if token in {"--ro-bind", "--bind"}
    ]
    assert "/" not in mounts and str(project) not in mounts and str(project.parent) not in mounts
    assert str(work) in mounts
    assert "HARNESS_EVAL_PROJECT_ROOT" not in env


@pytest.mark.parametrize("error", [ProcessLookupError, PermissionError])
@pytest.mark.parametrize("timeout", [True, False])
def test_process_cleanup_tolerates_a_vanished_process_group(tmp_path, monkeypatch, error, timeout):
    """Teardown must not fail when the group is already gone.

    The timeout path kills and reaps the leader, then `finally` signals the group
    a second time: Linux reports ESRCH there, macOS reports EPERM.
    """

    def _vanished(pgid: int, sig: int) -> None:
        raise error(1, "Operation not permitted")

    monkeypatch.setattr(isolation.os, "killpg", _vanished)
    # Short-lived on the timeout path: with killpg stubbed out nothing reaps the
    # child, so the follow-up communicate() waits for it to exit on its own.
    script = "import time; time.sleep(2)" if timeout else "print('finished')"
    command = [sys.executable, "-c", script]
    if timeout:
        with pytest.raises(subprocess.TimeoutExpired):
            run_agent_process(command, env=os.environ.copy(), cwd=tmp_path, timeout=0.5)
    else:
        completed = run_agent_process(command, env=os.environ.copy(), cwd=tmp_path, timeout=10)
        assert completed.returncode == 0


@pytest.mark.parametrize("timeout", [True, False])
def test_process_cleanup_terminates_background_descendants(tmp_path, timeout):
    pidfile = tmp_path / "child-pid"
    child = "import time; time.sleep(60)"
    script = (
        "import subprocess, sys, time; from pathlib import Path; "
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); "
        f"Path({str(pidfile)!r}).write_text(str(p.pid)); "
        + ("time.sleep(60)" if timeout else "print('finished')")
    )
    command = [sys.executable, "-c", script]
    if timeout:
        with pytest.raises(subprocess.TimeoutExpired):
            run_agent_process(command, env=os.environ.copy(), cwd=tmp_path, timeout=0.5)
    else:
        completed = run_agent_process(command, env=os.environ.copy(), cwd=tmp_path, timeout=5)
        assert completed.returncode == 0
    child_pid = int(pidfile.read_text())
    # A killed child may briefly remain a zombie until its parent is reaped.
    result = subprocess.run(
        ["ps", "-p", str(child_pid), "-o", "stat="], capture_output=True, text=True
    )
    assert not result.stdout.strip() or result.stdout.strip().startswith("Z"), result.stdout


class TestOutermostBinds:
    """bwrap creates a mount point per bind, so nested binds must be dropped.

    A virtualenv's `bin/python` is a symlink into the interpreter directory,
    which is itself bound read-only. Binding the symlink separately makes bwrap
    try to create a mount point inside that read-only mount, and it dies with
    "Can't create file at ...: No such file or directory". Binding the ancestor
    already exposes the child, so the nested entry is pure downside.
    """

    def test_drops_a_child_of_a_bound_directory(self):
        paths = {Path("/usr"), Path("/usr/bin/python3")}
        assert isolation._outermost(paths) == [Path("/usr")]

    def test_drops_the_venv_interpreter_symlink(self):
        """The exact shape that broke CI on ubuntu-24.04."""

        venv = Path("/home/runner/work/harness/harness/.venv")
        interpreter = Path("/home/runner/.local/share/uv/python/cpython-3.11")
        paths = {venv, venv / "bin/python", interpreter, interpreter / "bin/python3.11"}
        assert isolation._outermost(paths) == [interpreter, venv]

    def test_keeps_siblings(self):
        paths = {Path("/usr"), Path("/etc"), Path("/opt/homebrew")}
        assert isolation._outermost(paths) == [Path("/etc"), Path("/opt/homebrew"), Path("/usr")]

    def test_keeps_a_symlink_alias_that_is_not_a_subpath(self):
        """On macOS /etc and /private/etc alias each other but neither nests."""

        paths = {Path("/etc"), Path("/private/etc")}
        assert isolation._outermost(paths) == [Path("/etc"), Path("/private/etc")]

    def test_is_stable_regardless_of_set_ordering(self):
        deep = Path("/a/b/c/d")
        assert isolation._outermost({deep, Path("/a"), Path("/a/b")}) == [Path("/a")]
