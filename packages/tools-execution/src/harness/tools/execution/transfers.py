"""Explicit operator-selected file transfers; no implicit workspace upload."""

from __future__ import annotations

import asyncio
import base64
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from harness.core.paths import read_regular_file

if TYPE_CHECKING:
    from harness.tools.execution.backend import ExecutionBackend


def relative_path(raw: str) -> PurePosixPath:
    path = PurePosixPath(raw)
    if (
        not raw
        or raw == "."
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in raw
        or "\x00" in raw
    ):
        raise ValueError("transfer paths must be relative workspace files without traversal")
    if any(part.casefold() == ".harness" or ":" in part for part in path.parts):
        raise ValueError("Harness private state and drive-qualified paths cannot be transferred")
    return path


def host_path(root: Path, raw: str) -> Path:
    path = relative_path(raw)
    current = root
    for part in path.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("transfer paths may not contain symlinks")
        if current.exists() and getattr(current.lstat(), "st_file_attributes", 0) & 0x400:
            raise ValueError("transfer paths may not contain Windows reparse points")
    if not current.resolve().is_relative_to(root):
        raise ValueError("transfer path escapes workspace")
    return current


class WorkspaceTransfers:
    def __init__(self, backend: ExecutionBackend) -> None:
        self.backend = backend
        self.imports: list[tuple[str, bytes]] = []
        self.exported_files: list[str] = []

    async def preflight(self) -> None:
        config = self.backend.config
        root = self.backend.host_cwd
        if config.export_paths:
            host_path(root, config.export_directory)
        normalized_exports = [relative_path(raw).as_posix() for raw in config.export_paths]
        if len(set(normalized_exports)) != len(normalized_exports):
            raise ValueError("export_paths may not contain duplicate files")
        total = 0
        for raw in config.import_paths:
            path = host_path(root, raw)
            data = await asyncio.to_thread(read_regular_file, path, max_bytes=config.max_file_bytes)
            total += len(data)
            if total > config.transfer_max_bytes:
                raise ValueError("configured imports exceed total transfer byte limit")
            self.imports.append((relative_path(raw).as_posix(), data))

    async def import_files(self) -> None:
        for path, data in self.imports:
            await self.backend.files(
                "write_bytes", {"path": path, "data": base64.b64encode(data).decode("ascii")}
            )
        self.imports.clear()

    async def export_files(self, *, context_id: str) -> None:
        config = self.backend.config
        if not config.export_paths:
            return
        root = self.backend.host_cwd
        directory = host_path(root, config.export_directory) / context_id
        staged: list[tuple[Path, bytes]] = []
        total = 0
        for raw in config.export_paths:
            path = relative_path(raw).as_posix()
            result = await self.backend.files("read_bytes", {"path": path})
            data = base64.b64decode(result["content"], validate=True)
            total += len(data)
            if total > config.transfer_max_bytes:
                raise ValueError("configured exports exceed total transfer byte limit")
            target = host_path(root, (directory / path).relative_to(root).as_posix())
            staged.append((target, data))
        for target, data in staged:
            target = host_path(root, target.relative_to(root).as_posix())
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
            self.exported_files.append(str(target))


__all__ = ["WorkspaceTransfers"]
