"""Resolve a local CLI resume without rebinding the stored owner or database."""

from __future__ import annotations

from pathlib import Path

from harness.core import Storage
from harness.core.memory import MemoryScope
from harness.core.session_search import session_scope


async def resolve_local_session_workspace(
    storage: Storage, *, session_id: str | None, cwd: Path, cwd_explicit: bool
) -> Path:
    if not session_id:
        return cwd
    session = await storage.get(session_id)
    if session is None:
        # --session also permits a caller-supplied id for a new conversation.
        return cwd
    saved = session.cwd.expanduser().resolve()
    if session_scope(session) != MemoryScope(workspace=str(saved)):
        raise ValueError(
            "This is not a local CLI session; resume it through its original authenticated entrypoint."
        )
    if cwd_explicit and cwd != saved:
        raise ValueError(
            "--cwd conflicts with the saved session workspace. "
            "Omit --cwd to use that workspace, or start a new session for another directory."
        )
    if not saved.is_dir():
        raise ValueError(
            f"Saved session workspace is unavailable: {saved}. "
            "Restore that directory before resuming, or start a new session."
        )
    return saved
