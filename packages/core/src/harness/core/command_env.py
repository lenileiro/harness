from __future__ import annotations

import os
from pathlib import Path

_ENV_ALLOWLIST = (
    "HOME",
    "TMPDIR",
    "TEMP",
    "TMP",
    "LANG",
    "LC_ALL",
    "TERM",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)


def _path_is_inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def clean_command_env(cwd: Path) -> dict[str, str]:
    """Return an environment for validating the target workspace, not Harness."""

    env: dict[str, str] = {}
    for key in _ENV_ALLOWLIST:
        value = os.environ.get(key)
        if value:
            env[key] = value

    workspace_venv = cwd / ".venv"
    harness_venv = os.environ.get("VIRTUAL_ENV", "").strip()
    harness_venv_path = Path(harness_venv).resolve() if harness_venv else None
    path_entries: list[str] = []
    for raw_entry in os.environ.get("PATH", os.defpath).split(os.pathsep):
        if not raw_entry:
            continue
        entry = Path(raw_entry)
        if harness_venv_path and _path_is_inside(entry, harness_venv_path):
            continue
        path_entries.append(raw_entry)

    workspace_bin = workspace_venv / ("Scripts" if os.name == "nt" else "bin")
    if workspace_bin.is_dir():
        path_entries.insert(0, str(workspace_bin))
        env["VIRTUAL_ENV"] = str(workspace_venv)

    env["PATH"] = os.pathsep.join(dict.fromkeys(path_entries)) or os.defpath
    return env


__all__ = ["clean_command_env"]
