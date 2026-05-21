"""Work-queue tools for multi-agent orchestration.

These tools wrap TaskStore operations. Dependencies (store, parent_id, item_id)
are injected at construction so the LLM only needs to provide task content,
not infrastructure details.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness.core.schemas import ApprovalDecision, EffectScope, ToolCall, ToolResult
from harness.tasks.schemas import Task
from harness.tasks.store import TaskStore

_CREATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Short title for the work item."},
        "description": {
            "type": "string",
            "description": "Optional longer description of what needs to be done.",
        },
    },
    "required": ["title"],
}

_COMPLETE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "Brief summary of what was accomplished.",
        },
    },
    "required": [],
}

_LIST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {
            "type": "string",
            "enum": ["todo", "in_progress", "done", "cancelled"],
            "description": "Filter by status. Omit to list all items.",
        },
    },
    "required": [],
}


class CreateWorkItemTool:
    """Create a work item (task) in the shared work queue."""

    name = "create_work_item"
    description = (
        "Create a new work item in the shared job queue. "
        "Provide 'title' (required) and optional 'description'."
    )
    effect_scope: EffectScope = "task_durable"
    approval: ApprovalDecision = "auto"
    phases: tuple[str, ...] = ("*",)

    def __init__(self, store: TaskStore, parent_id: str, cwd: Path) -> None:
        self._store = store
        self._parent_id = parent_id
        self._cwd = cwd
        self.parameters_schema: dict[str, Any] = _CREATE_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        args: dict[str, Any] = call.arguments if isinstance(call.arguments, dict) else {}
        title = args.get("title", "").strip()
        description = args.get("description")
        if not title:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="'title' is required",
                is_error=True,
            )
        task = Task(
            ref="",
            title=title,
            description=description,
            status="todo",
            parent_id=self._parent_id,
            cwd=self._cwd,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        created = await self._store.create_task(task)
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=f"created work item {created.ref}: {created.title}",
        )


class CompleteWorkItemTool:
    """Mark the current work item as completed with a result summary."""

    name = "complete_work_item"
    description = (
        "Mark this work item as done. " "Provide 'summary' describing what was accomplished."
    )
    effect_scope: EffectScope = "task_durable"
    approval: ApprovalDecision = "auto"
    phases: tuple[str, ...] = ("*",)

    def __init__(self, store: TaskStore, item_id: str) -> None:
        self._store = store
        self._item_id = item_id
        self.parameters_schema: dict[str, Any] = _COMPLETE_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        task = await self._store.get_task(self._item_id)
        if task is None:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"work item {self._item_id!r} not found",
                is_error=True,
            )
        args: dict[str, Any] = call.arguments if isinstance(call.arguments, dict) else {}
        summary = args.get("summary", "")
        updated = task.model_copy(
            update={
                "status": "done",
                "metadata": {**task.metadata, "result_summary": summary},
                "updated_at": datetime.now(UTC),
            }
        )
        await self._store.update_task(updated)
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=f"marked {task.ref} as done",
        )


class ListWorkItemsTool:
    """List work items in the job queue."""

    name = "list_work_items"
    description = (
        "List work items in the current job queue. "
        "Optionally filter by 'status' (todo, in_progress, done)."
    )
    effect_scope: EffectScope = "read_only"
    approval: ApprovalDecision = "auto"
    phases: tuple[str, ...] = ("*",)

    def __init__(self, store: TaskStore, parent_id: str) -> None:
        self._store = store
        self._parent_id = parent_id
        self.parameters_schema: dict[str, Any] = _LIST_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        args: dict[str, Any] = call.arguments if isinstance(call.arguments, dict) else {}
        status = args.get("status")
        items = await self._store.list_tasks(parent_id=self._parent_id, status=status)
        if not items:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="no work items found",
            )
        lines = [f"{t.ref} [{t.status}] {t.title}" for t in items]
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content="\n".join(lines),
        )


__all__ = ["CompleteWorkItemTool", "CreateWorkItemTool", "ListWorkItemsTool"]
