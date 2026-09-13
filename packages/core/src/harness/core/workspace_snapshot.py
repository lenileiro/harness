"""Content fingerprints for execution evidence, excluding generated agent state."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from pathlib import Path


def workspace_fingerprint(cwd: Path) -> str:
    cwd = cwd.resolve()
    try:
        paths = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=cwd,
            capture_output=True,
            check=False,
        )
    except OSError:
        paths = None
    excluded = {".git", ".harness", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache"}
    if paths is not None and paths.returncode == 0:
        names = {os.fsdecode(raw) for raw in paths.stdout.split(b"\0") if raw}
    else:
        names = set()
        for parent, directories, files in os.walk(cwd):
            directories[:] = [name for name in directories if name not in excluded]
            names.update(str((Path(parent) / name).relative_to(cwd)) for name in files)
    digest = hashlib.sha256()
    for name in sorted(names):
        path = cwd / name
        if any(part in excluded for part in Path(name).parts):
            continue
        digest.update(os.fsencode(name) + b"\0")
        if path.is_symlink():
            digest.update(b"symlink\0")
            content = os.fsencode(os.readlink(path))
        elif path.is_file():
            mode = stat.S_IMODE(path.stat().st_mode)
            digest.update(f"file:{mode:o}\0".encode())
            content = path.read_bytes()
        else:
            digest.update(b"missing\0")
            content = b"<missing>"
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()
