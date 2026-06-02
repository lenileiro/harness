"""Shared checks for external-workspace eval scenarios."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath
from typing import Any


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
    name = parts[-1] if parts else ""
    return (
        "__pycache__" in parts
        or ".pytest_cache" in parts
        or ".mypy_cache" in parts
        or ".ruff_cache" in parts
        or ".tox" in parts
        or name.endswith((".pyc", ".pyo"))
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


def announce_scenario_start(*, run_root: Path, workspace: Path) -> None:
    """Print enough process metadata to observe a long-running scenario."""

    print(f"SCENARIO_RUN_PID={os.getpid()}", flush=True)
    print(f"RUN_ROOT={run_root}", flush=True)
    print(f"WORKSPACE={workspace}", flush=True)


def independent_check_failure(exc: BaseException) -> dict[str, Any]:
    return {
        "independent_check_error": f"{type(exc).__name__}: {exc}",
    }
