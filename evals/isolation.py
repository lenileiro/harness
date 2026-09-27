"""Process isolation for agent runs; evaluator commands stay outside this boundary.

Only public runtime sources, installed dependencies, system tools, and the task
workspace are readable. A missing OS sandbox is an execution error, never an
implicit permission to run a benchmark agent on the host.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import suppress
from pathlib import Path


def prepare_agent_process(
    command: list[str],
    *,
    work: Path,
    project_root: Path,
    config_path: Path | None = None,
) -> tuple[list[str], dict[str, str]]:
    root = Path(tempfile.mkdtemp(prefix="agent-runtime-", dir=work.parent))
    sources: list[str] = []
    for package in sorted((project_root / "packages").glob("*/src")):
        if not package.resolve().is_relative_to((project_root / "packages").resolve()):
            raise RuntimeError("Public runtime source directory escapes the package tree")
        destination = root / package.parent.name / "src"
        # Preserve links: following one here could copy evaluator-only files into
        # the readable runtime. The OS boundary will reject external targets.
        shutil.copytree(
            package,
            destination,
            symlinks=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        sources.append(str(destination))
    home = Path(tempfile.mkdtemp(prefix="agent-home-", dir=work.parent))
    temporary = Path(tempfile.mkdtemp(prefix="agent-tmp-", dir=work.parent))
    env = os.environ.copy()
    for name in tuple(env):
        if name.startswith("HARNESS_EVAL_"):
            env.pop(name)
    for name in (
        "HARNESS_EVAL_PROJECT_ROOT",
        "HARNESS_EXPERIENCE_ROOTS",
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
    ):
        env.pop(name, None)
    env.update(
        {
            "HOME": str(home),
            "TMPDIR": str(temporary),
            "PYTHONPATH": os.pathsep.join(sources),
            "PYTHONNOUSERSITE": "1",
            "HARNESS_EVAL_WORKSPACE": str(work),
            "HARNESS_EXPERIENCE_ROOTS": "",
        }
    )
    if config_path is not None:
        public_config = root / "config.toml"
        shutil.copyfile(config_path, public_config)
        command = [str(public_config) if arg == str(config_path) else arg for arg in command]
    executable = Path(shutil.which(command[0]) or command[0]).absolute()
    # Include both lexical paths and targets for virtualenv/system symlinks.
    readable = {
        Path(path)
        for path in (
            "/usr",
            "/bin",
            "/sbin",
            "/lib",
            "/lib64",
            "/etc",
            "/System",
            "/Library",
            "/opt/homebrew",
            "/dev",
        )
        if Path(path).exists()
    }
    readable.update(
        {Path(sys.prefix), Path(sys.base_prefix), root, executable, executable.resolve()}
    )
    readable.update(path.resolve() for path in tuple(readable))
    writable = {work.resolve(), home.resolve(), temporary.resolve()}
    if sys.platform == "darwin" and shutil.which("sandbox-exec"):
        rules = [
            "(version 1)",
            "(allow default)",
            "(deny file-read*)",
            "(deny file-write*)",
            "(allow file-read-metadata)",
            # dyld opens the root directory on current macOS. Permit that
            # directory alone, never a subpath rule for the whole disk.
            '(allow file-read* (literal "/"))',
            '(allow file-write* (literal "/dev/null"))',
        ]
        for path in sorted(readable | writable):
            selector = "subpath" if path.is_dir() else "literal"
            rules.append(f"(allow file-read* ({selector} {json.dumps(str(path))}))")
        for path in sorted(writable):
            rules.append(f"(allow file-write* (subpath {json.dumps(str(path))}))")
        profile = root / "agent.sb"
        profile.write_text("\n".join(rules), encoding="utf-8")
        return ["/usr/bin/sandbox-exec", "-f", str(profile), *command], env
    if sys.platform.startswith("linux") and (bubblewrap := shutil.which("bwrap")):
        args = [
            bubblewrap,
            "--die-with-parent",
            "--new-session",
            "--unshare-pid",
            "--unshare-user",
            "--unshare-ipc",
            "--unshare-uts",
            "--proc",
            "/proc",
            "--dev",
            "/dev",
        ]
        for path in _outermost(readable):
            if path != Path("/dev"):
                args += ["--ro-bind", str(path), str(path)]
        for path in _outermost(writable):
            args += ["--bind", str(path), str(path)]
        args += ["--chdir", str(work), "--", *command]
        return args, env
    raise RuntimeError("Benchmark agents require sandbox-exec on macOS or bubblewrap on Linux.")


def _outermost(paths: set[Path]) -> list[Path]:
    """Drop paths already covered by an ancestor that is also being bound.

    bwrap has to create a mount point for every bind. A nested bind is
    redundant -- binding the ancestor already exposes the child -- and it
    actively breaks when the child resolves through a symlink into another
    read-only mount, which is how a virtualenv's `bin/python` is laid out:

        bwrap: Can't create file at /.../.venv/bin/python: No such file or directory

    Sorting puts every ancestor before its descendants, so one pass suffices.
    """

    kept: list[Path] = []
    for path in sorted(paths):
        if any(path.is_relative_to(parent) for parent in kept):
            continue
        kept.append(path)
    return kept


def run_agent_process(
    command: list[str],
    *,
    env: dict[str, str],
    cwd: Path,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    """Collect output and terminate ordinary descendants before grading begins.

    The OS sandbox remains inherited by subprocesses. A dedicated session lets
    timeout and normal-exit cleanup terminate background tools as well as the
    CLI, instead of killing only the direct subprocess.
    """
    process = subprocess.Popen(
        command,
        env=env,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    def kill_group() -> None:
        # Best-effort: this runs again in `finally` after the timeout path has
        # already killed and reaped the leader. A group that is gone reports
        # ESRCH on Linux but EPERM on macOS, and neither is actionable here.
        with suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)

    try:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            kill_group()
            stdout, stderr = process.communicate()
            raise subprocess.TimeoutExpired(command, timeout, output=stdout, stderr=stderr) from exc
        return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    finally:
        kill_group()
        process.wait()
