"""Shared checks for external-workspace eval scenarios."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any


def command_output_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ[key] = value


def is_generated_artifact(path: str) -> bool:
    parts = [part.lower() for part in path.strip("/").split("/") if part]
    return any(_path_part_is_generated_cache(part) for part in parts)


def _path_part_is_generated_cache(part: str) -> bool:
    return (
        part in {"cache", ".cache"}
        or part.endswith("_cache")
        or part.endswith("-cache")
        or part.endswith("cache__")
    )


def untracked_scratch_paths(
    status_text: str,
    *,
    allowed_untracked: set[str] | None = None,
) -> list[str]:
    allowed = allowed_untracked or set()
    paths: list[str] = []
    for line in status_text.splitlines():
        if not line.startswith("?? "):
            continue
        path = line[3:]
        if path in allowed:
            continue
        if is_generated_artifact(PurePosixPath(path).as_posix()):
            continue
        paths.append(path)
    return paths


def git_diff_with_untracked(
    workspace: Path,
    paths: list[str],
    *,
    status_text: str | None = None,
) -> str:
    diff_result = subprocess.run(
        ["git", "diff", "--", *paths],
        cwd=workspace,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
        timeout=30,
    )
    diff_parts = [diff_result.stdout] if diff_result.stdout else []
    status = status_text
    if status is None:
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
        status = status_result.stdout

    for line in status.splitlines():
        if not line.startswith("?? "):
            continue
        path = PurePosixPath(line[3:]).as_posix()
        if not _path_matches_any(path, paths) or is_generated_artifact(path):
            continue
        untracked_path = workspace / path
        if not untracked_path.is_file():
            continue
        content = untracked_path.read_text(encoding="utf-8", errors="replace")
        diff_parts.append(_new_file_diff(path, content))
    return "".join(diff_parts)


def _path_matches_any(path: str, pathspecs: list[str]) -> bool:
    for raw_pathspec in pathspecs:
        pathspec = PurePosixPath(raw_pathspec).as_posix().rstrip("/")
        if path == pathspec or path.startswith(f"{pathspec}/"):
            return True
    return False


def _new_file_diff(path: str, content: str) -> str:
    lines = content.splitlines()
    header = (
        f"diff --git a/{path} b/{path}\n"
        "new file mode 100644\n"
        "index 0000000..0000000\n"
        "--- /dev/null\n"
        f"+++ b/{path}\n"
        f"@@ -0,0 +1,{len(lines)} @@\n"
    )
    body = "".join(f"+{line}\n" for line in lines)
    return header + body


def announce_scenario_start(*, run_root: Path, workspace: Path) -> None:
    """Print enough process metadata to observe a long-running scenario."""

    print(f"SCENARIO_RUN_PID={os.getpid()}", flush=True)
    print(f"RUN_ROOT={run_root}", flush=True)
    print(f"WORKSPACE={workspace}", flush=True)


def independent_check_failure(exc: BaseException) -> dict[str, Any]:
    return {
        "independent_check_error": f"{type(exc).__name__}: {exc}",
    }
