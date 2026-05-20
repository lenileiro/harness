"""Task-namespaced `ActivityEvent.kind` constants.

The runtime-general vocabulary (agent_run.*, tool_call.*, approval.*,
step.*) lives in `harness.core.activity`. This module contributes the
task-domain kinds that the `tasks` CLI / API emit.

Naming convention: `task.<verb>`.
"""

from __future__ import annotations

TASK_CREATED = "task.created"
TASK_UPDATED = "task.updated"
TASK_STATUS_CHANGED = "task.status_changed"
TASK_LINKED = "task.linked"
TASK_DELETED = "task.deleted"


__all__ = [
    "TASK_CREATED",
    "TASK_DELETED",
    "TASK_LINKED",
    "TASK_STATUS_CHANGED",
    "TASK_UPDATED",
]
