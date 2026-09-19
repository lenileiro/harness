"""Shared JSON-on-disk plumbing for the workspace stores.

ResearchStore, MissionStore and (shortly) FactoryStore all persist small JSON
records under a ``.harness/<name>/`` root. They share four things and nothing
else: the root convention, id minting, atomic writes, and a cross-process
execution lock. Their record types and query surfaces have no overlap at all,
so this is deliberately a *plumbing* base and not a common store interface.
Do not grow it into one.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from harness.core.slug import slugify


def default_root(name: str, cwd: Path | None = None) -> Path:
    """Resolve the on-disk root for a named store under ``.harness/``."""
    return (cwd or Path.cwd()).resolve() / ".harness" / name


def write_json(path: Path, payload: dict) -> None:
    """Write JSON atomically.

    A plain ``write_text`` truncates the target before the new bytes land, so a
    process killed mid-write leaves a corrupt record behind. These records are
    written by long unattended runs that can be OOM-killed, so the temp-file
    plus ``os.replace`` dance is load bearing, not decoration.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


class JsonStore:
    """Root, id minting and locking shared by the workspace stores."""

    def __init__(self, *, root: Path) -> None:
        self.root = root

    def new_id(self, prefix: str, title: str) -> str:
        return f"{prefix}-{slugify(title)[:32]}-{uuid4().hex[:8]}"

    @contextmanager
    def execution_lock(self) -> Iterator[None]:
        """Serialize transitions across independent processes."""
        from filelock import FileLock

        self.root.mkdir(parents=True, exist_ok=True)
        with FileLock(self.root / "execution.lock", mode=0o600):
            yield
