"""Run a web-backed external-workspace scenario through Harness."""

from __future__ import annotations

import sys

if __package__ in (None, "") and sys.path:
    _script_dir = sys.path[0]
    if _script_dir.endswith("/evals"):
        sys.path.pop(0)
        sys.path.insert(0, _script_dir.rsplit("/", 1)[0])

import argparse
import asyncio
import gzip
import importlib.util
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib.request import Request, urlopen
from uuid import uuid4

from evals.external_scenario_checks import (
    announce_scenario_start,
    command_output_text,
    git_diff_with_untracked,
    independent_check_failure,
    load_dotenv,
    untracked_scratch_paths,
)
from harness.cli.external_workspace import run_harness_on_external_environment

PROJECT_RELEASE_API_URL = "https://api.github.com/repos/curl/curl/releases/latest"
OFFICIAL_SOURCE_PREFIXES = ("https://curl.se/", "https://github.com/curl/curl/")


@dataclass(frozen=True)
class LocalCommandResult:
    stdout: str
    stderr: str
    return_code: int


class LocalEnvironment:
    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_sec: float | int | None = None,
    ) -> LocalCommandResult:
        working_dir = Path(cwd) if cwd else self.workdir

        def run() -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                command,
                cwd=working_dir,
                shell=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=float(timeout_sec or 30),
                check=False,
            )

        try:
            completed = await asyncio.to_thread(run)
        except subprocess.TimeoutExpired as exc:
            return LocalCommandResult(
                stdout=command_output_text(exc.stdout),
                stderr=(
                    command_output_text(exc.stderr) + f"\ncommand timed out after {timeout_sec}s"
                ),
                return_code=124,
            )
        return LocalCommandResult(
            stdout=completed.stdout,
            stderr=completed.stderr,
            return_code=completed.returncode,
        )


def _write(path: Path, content: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def create_workspace(root: Path) -> Path:
    workspace = root / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    _write(
        workspace / "current_release.py",
        """
CURRENT_RELEASE = "unknown"
SOURCE_URL = ""


def current_release():
    return CURRENT_RELEASE


def source_url():
    return SOURCE_URL
""".lstrip(),
    )
    _write(
        workspace / "tests" / "test_current_release.py",
        """
import current_release


def test_exports_release_helpers():
    assert isinstance(current_release.current_release(), str)
    assert isinstance(current_release.source_url(), str)
""".lstrip(),
    )
    _write(
        workspace / "README.md",
        """
# Current Release Fixture

Keep `current_release.py` aligned with the latest stable cURL release from
official public web sources.
""".lstrip(),
    )

    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    subprocess.run(
        ["git", "config", "user.email", "harness@example.test"], cwd=workspace, check=True
    )
    subprocess.run(["git", "config", "user.name", "Harness Eval"], cwd=workspace, check=True)
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial fixture"], cwd=workspace, check=True)
    return workspace


def latest_project_release() -> str:
    req = Request(PROJECT_RELEASE_API_URL, headers={"User-Agent": "harness-eval/0.1"})
    with urlopen(req, timeout=30) as response:
        body = response.read()
        if response.headers.get("content-encoding") == "gzip":
            body = gzip.decompress(body)
        text = body.decode("utf-8", errors="replace")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError("could not parse latest release metadata") from exc
    release_label = str(payload.get("tag_name") or payload.get("name") or "")
    match = re.search(
        r"(?<!\d)(\d+(?:[._]\d+){1,}(?:[-+][0-9A-Za-z_.-]+)?)(?!\d)",
        release_label,
    )
    if match:
        return match.group(1).replace("_", ".")
    raise RuntimeError("could not parse latest release from project metadata")


def import_current_release(workspace: Path):
    module_path = workspace / "current_release.py"
    spec = importlib.util.spec_from_file_location("current_release_eval", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load current_release.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def event_log_has_successful_web_tool(run_root: Path) -> bool:
    events_path = run_root / "harness" / "harness-events.jsonl"
    if not events_path.exists():
        return False
    for raw_line in events_path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "tool_result":
            continue
        result = event.get("result") if isinstance(event.get("result"), dict) else {}
        if result.get("name") in {"web_search", "fetch_url"} and result.get("is_error") is not True:
            return True
    return False


def independent_check(workspace: Path, run_root: Path) -> dict[str, object]:
    expected_error = ""
    expected = ""
    try:
        expected = latest_project_release()
    except Exception as exc:
        expected_error = f"{type(exc).__name__}: {exc}"

    module_error = ""
    actual = ""
    source = ""
    try:
        module = import_current_release(workspace)
        actual = str(module.current_release())
        source = str(module.source_url())
    except Exception as exc:
        module_error = f"{type(exc).__name__}: {exc}"

    pytest_result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=workspace,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=60,
    )
    status_result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=workspace,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=30,
    )
    status_paths = {
        line[3:]
        for line in status_result.stdout.splitlines()
        if len(line) > 3 and (line[:2].strip() or line.startswith("?? "))
    }
    focused_test_paths = {
        path
        for path in status_paths
        if path == "tests/test_current_release.py"
        or (path.startswith("tests/test_current_release") and path.endswith(".py"))
    }
    diff_paths = ["current_release.py", "tests"]
    diff_text = git_diff_with_untracked(
        workspace,
        diff_paths,
        status_text=status_result.stdout,
    )
    leftover_scratch_paths = untracked_scratch_paths(
        status_result.stdout,
        allowed_untracked=focused_test_paths,
    )
    return {
        "expected_release": expected,
        "expected_error": expected_error,
        "actual_release": actual,
        "source_url": source,
        "module_error": module_error,
        "release_matches": bool(expected) and actual == expected,
        "source_is_official": source.startswith(OFFICIAL_SOURCE_PREFIXES),
        "used_successful_web_tool": event_log_has_successful_web_tool(run_root),
        "pytest_return_code": pytest_result.returncode,
        "pytest_stdout": pytest_result.stdout[-1000:],
        "pytest_stderr": pytest_result.stderr[-1000:],
        "diff": diff_text,
        "status": status_result.stdout,
        "changed_source": "current_release.py" in status_result.stdout,
        "changed_tests": bool(focused_test_paths),
        "changed_test_paths": sorted(focused_test_paths),
        "leftover_scratch_paths": leftover_scratch_paths,
    }


async def run(args: argparse.Namespace) -> int:
    load_dotenv(Path(args.env_file))
    run_root = Path(args.results_root) / f"external-web-release-{uuid4().hex[:8]}"
    run_root.mkdir(parents=True, exist_ok=True)
    workspace = create_workspace(run_root)
    announce_scenario_start(run_root=run_root, workspace=workspace)
    logs_dir = run_root / "harness"
    context = SimpleNamespace(n_agent_steps=0, metadata={})
    error = ""
    try:
        await run_harness_on_external_environment(
            instruction=(
                "Update current_release.py with the latest stable cURL release from "
                "official public web sources. Add or update a focused project test for "
                "the release value and source URL, then verify the work."
            ),
            environment=LocalEnvironment(workspace),
            context=context,
            logs_dir=str(logs_dir),
            model_name=args.model,
            max_steps=args.max_steps,
            max_output_tokens=args.max_output_tokens,
            source_change_retries=args.source_change_retries,
            verification_retries=args.verification_retries,
            pass_timeout_seconds=args.pass_timeout_seconds,
            model_stream_idle_timeout_seconds=args.idle_timeout,
            model_turn_timeout_seconds=args.turn_timeout,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"

    try:
        check = independent_check(workspace, run_root)
    except Exception as exc:
        error = error or f"{type(exc).__name__}: {exc}"
        check = independent_check_failure(exc)
    passed = (
        not error
        and check.get("release_matches") is True
        and check.get("source_is_official") is True
        and check.get("used_successful_web_tool") is True
        and check.get("pytest_return_code") == 0
        and check.get("changed_source") is True
        and check.get("changed_tests") is True
        and not check.get("leftover_scratch_paths")
    )
    outcome = {
        "status": "passed" if passed else "failed",
        "error": error,
        "run_root": str(run_root),
        "workspace": str(workspace),
        "context_metadata": context.metadata,
        "independent_check": check,
    }
    (run_root / "outcome.json").write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    (run_root / "final.diff").write_text(str(check.get("diff", "")), encoding="utf-8")
    print(json.dumps(outcome, indent=2))
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="evals/results/live-scenarios")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--model", default="openai/gpt-5.4-nano")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--source-change-retries", type=int, default=1)
    parser.add_argument("--verification-retries", type=int, default=1)
    parser.add_argument("--pass-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--idle-timeout", type=float, default=120.0)
    parser.add_argument("--turn-timeout", type=float, default=180.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
