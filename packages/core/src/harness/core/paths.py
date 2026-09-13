"""User state namespace; a selected profile supplies HARNESS_HOME."""

from __future__ import annotations

import os
import stat
from pathlib import Path


def user_home() -> Path:
    value = os.environ.get("HARNESS_HOME")
    return Path(value).expanduser().resolve() if value else Path.home() / ".harness"


def read_regular_file(path: Path, *, max_bytes: int) -> bytes:
    """Bound reads before allocation; refuse special files and final symlinks."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or getattr(before, "st_file_attributes", 0) & getattr(
        stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400
    ):
        raise ValueError("expected a regular file without symlinks or reparse points")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as source:
        info = os.fstat(source.fileno())
        if (info.st_dev, info.st_ino) != (before.st_dev, before.st_ino):
            raise ValueError("file identity changed while opening")
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ValueError(f"expected a regular file no larger than {max_bytes} bytes")
        data = source.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"file exceeds {max_bytes} bytes")
        return data
