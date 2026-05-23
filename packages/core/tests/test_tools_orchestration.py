"""Tests for CreateWorkItemTool, CompleteWorkItemTool, and ListWorkItemsTool."""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.core.schemas import ToolCall
from harness.core.tools_orchestration import (
    CompleteWorkItemTool,
    CreateWorkItemTool,
    ListWorkItemsTool,
)
from harness.storage.memory import InMemoryStorage


def _call(name: str, arguments: dict) -> ToolCall:
    return ToolCall(id="tc_test", name=name, arguments=arguments)


# ---------------------------------------------------------------------------
# CreateWorkItemTool
# ---------------------------------------------------------------------------


class TestCreateWorkItemTool:
    @pytest.mark.asyncio
    async def test_creates_work_item(self) -> None:
        store = InMemoryStorage()
        tool = CreateWorkItemTool(store, parent_id="job_1", cwd=Path("/tmp"))
        result = await tool(_call("create_work_item", {"title": "Do the thing"}))
        assert not result.is_error
        assert "Do the thing" in result.content

    @pytest.mark.asyncio
    async def test_missing_title_returns_error(self) -> None:
        store = InMemoryStorage()
        tool = CreateWorkItemTool(store, parent_id="job_1", cwd=Path("/tmp"))
        result = await tool(_call("create_work_item", {}))
        assert result.is_error
        assert "required" in result.content

    @pytest.mark.asyncio
    async def test_empty_title_returns_error(self) -> None:
        store = InMemoryStorage()
        tool = CreateWorkItemTool(store, parent_id="job_1", cwd=Path("/tmp"))
        result = await tool(_call("create_work_item", {"title": "   "}))
        assert result.is_error

    @pytest.mark.asyncio
    async def test_optional_description_stored(self) -> None:
        store = InMemoryStorage()
        tool = CreateWorkItemTool(store, parent_id="job_1", cwd=Path("/tmp"))
        result = await tool(
            _call("create_work_item", {"title": "Task A", "description": "Details here"})
        )
        assert not result.is_error
        tasks = await store.list_tasks(parent_id="job_1")
        assert tasks[0].description == "Details here"

    @pytest.mark.asyncio
    async def test_task_stored_with_todo_status(self) -> None:
        store = InMemoryStorage()
        tool = CreateWorkItemTool(store, parent_id="job_2", cwd=Path("/tmp"))
        await tool(_call("create_work_item", {"title": "New task"}))
        tasks = await store.list_tasks(parent_id="job_2")
        assert len(tasks) == 1
        assert tasks[0].status == "todo"

    def test_tool_metadata(self) -> None:
        store = InMemoryStorage()
        tool = CreateWorkItemTool(store, parent_id="x", cwd=Path("/tmp"))
        assert tool.name == "create_work_item"
        assert tool.effect_scope == "task_durable"
        assert tool.approval == "auto"
        assert "title" in tool.parameters_schema["required"]


# ---------------------------------------------------------------------------
# CompleteWorkItemTool
# ---------------------------------------------------------------------------


class TestCompleteWorkItemTool:
    @pytest.mark.asyncio
    async def test_marks_item_done(self) -> None:
        store = InMemoryStorage()
        create_tool = CreateWorkItemTool(store, parent_id="job_3", cwd=Path("/tmp"))
        await create_tool(_call("create_work_item", {"title": "Do work"}))
        tasks = await store.list_tasks(parent_id="job_3")
        item_id = tasks[0].id

        tool = CompleteWorkItemTool(store, item_id=item_id)
        result = await tool(_call("complete_work_item", {"summary": "All done"}))
        assert not result.is_error
        assert "done" in result.content

        updated = await store.get_task(item_id)
        assert updated is not None
        assert updated.status == "done"
        assert updated.metadata["result_summary"] == "All done"

    @pytest.mark.asyncio
    async def test_missing_item_returns_error(self) -> None:
        store = InMemoryStorage()
        tool = CompleteWorkItemTool(store, item_id="nonexistent_id")
        result = await tool(_call("complete_work_item", {"summary": "done"}))
        assert result.is_error
        assert "not found" in result.content

    @pytest.mark.asyncio
    async def test_empty_summary_is_valid(self) -> None:
        store = InMemoryStorage()
        create_tool = CreateWorkItemTool(store, parent_id="job_4", cwd=Path("/tmp"))
        await create_tool(_call("create_work_item", {"title": "Task"}))
        tasks = await store.list_tasks(parent_id="job_4")
        item_id = tasks[0].id

        tool = CompleteWorkItemTool(store, item_id=item_id)
        result = await tool(_call("complete_work_item", {}))
        assert not result.is_error

    def test_tool_metadata(self) -> None:
        store = InMemoryStorage()
        tool = CompleteWorkItemTool(store, item_id="x")
        assert tool.name == "complete_work_item"
        assert tool.effect_scope == "task_durable"
        assert tool.approval == "auto"


# ---------------------------------------------------------------------------
# ListWorkItemsTool
# ---------------------------------------------------------------------------


class TestListWorkItemsTool:
    @pytest.mark.asyncio
    async def test_empty_queue_returns_no_items_message(self) -> None:
        store = InMemoryStorage()
        tool = ListWorkItemsTool(store, parent_id="job_empty")
        result = await tool(_call("list_work_items", {}))
        assert not result.is_error
        assert "no work items found" in result.content

    @pytest.mark.asyncio
    async def test_lists_all_items(self) -> None:
        store = InMemoryStorage()
        create_tool = CreateWorkItemTool(store, parent_id="job_6", cwd=Path("/tmp"))
        await create_tool(_call("create_work_item", {"title": "Task 1"}))
        await create_tool(_call("create_work_item", {"title": "Task 2"}))

        list_tool = ListWorkItemsTool(store, parent_id="job_6")
        result = await list_tool(_call("list_work_items", {}))
        assert not result.is_error
        assert "Task 1" in result.content
        assert "Task 2" in result.content

    @pytest.mark.asyncio
    async def test_filters_by_status(self) -> None:
        store = InMemoryStorage()
        create_tool = CreateWorkItemTool(store, parent_id="job_7", cwd=Path("/tmp"))
        await create_tool(_call("create_work_item", {"title": "Will Be Done"}))
        await create_tool(_call("create_work_item", {"title": "Still Todo"}))

        all_tasks = await store.list_tasks(parent_id="job_7")
        # Complete whichever task is "Will Be Done"
        target = next(t for t in all_tasks if t.title == "Will Be Done")
        complete_tool = CompleteWorkItemTool(store, item_id=target.id)
        await complete_tool(_call("complete_work_item", {"summary": "done"}))

        list_tool = ListWorkItemsTool(store, parent_id="job_7")
        result = await list_tool(_call("list_work_items", {"status": "todo"}))
        assert "Still Todo" in result.content
        assert "Will Be Done" not in result.content

    def test_tool_metadata(self) -> None:
        store = InMemoryStorage()
        tool = ListWorkItemsTool(store, parent_id="x")
        assert tool.name == "list_work_items"
        assert tool.effect_scope == "read_only"
        assert tool.approval == "auto"
