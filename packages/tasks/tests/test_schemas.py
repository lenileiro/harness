"""Tests for Task / TaskLink schemas + module wiring."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from harness.tasks import ActivityEvent, ActivityStore, Task, TaskLink, TaskStore, activity


class TestTaskSchema:
    def test_defaults(self, tmp_path: Path) -> None:
        t = Task(ref="T-001", title="hello", cwd=tmp_path)
        assert t.id.startswith("task_")
        assert t.status == "backlog"
        assert t.labels == []
        assert t.links == []
        assert t.session_ids == []
        assert t.parent_id is None
        assert t.cwd == tmp_path

    def test_touch_bumps_updated_at(self, tmp_path: Path) -> None:
        t = Task(ref="T-001", title="x", cwd=tmp_path)
        before = t.updated_at
        t.touch()
        assert t.updated_at >= before

    def test_unknown_status_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            Task(ref="T-001", title="x", cwd=tmp_path, status="weird")  # type: ignore[arg-type]

    def test_unknown_relation_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TaskLink(target_ref="T-002", relation="weird")  # type: ignore[arg-type]

    def test_round_trip_json(self, tmp_path: Path) -> None:
        t = Task(
            ref="T-001",
            title="x",
            cwd=tmp_path,
            links=[TaskLink(target_ref="T-002", relation="blocks")],
            labels=["a", "b"],
        )
        round_tripped = Task.model_validate(t.model_dump(mode="json"))
        assert round_tripped.ref == t.ref
        assert round_tripped.links == t.links
        assert round_tripped.labels == t.labels


class TestActivityEvent:
    def test_minimal(self) -> None:
        e = ActivityEvent(kind="custom.thing")
        assert e.id.startswith("act_")
        assert e.task_id is None
        assert e.session_id is None
        assert e.kind == "custom.thing"
        assert e.data == {}

    def test_kind_is_open_string(self) -> None:
        # No closed enum — any string is accepted.
        e = ActivityEvent(kind="my.app.custom_event")
        assert e.kind == "my.app.custom_event"


class TestModuleWiring:
    def test_activity_constants_exposed(self) -> None:
        # task-domain kinds live on harness.tasks.activity
        assert activity.TASK_CREATED == "task.created"
        assert activity.TASK_LINKED == "task.linked"

    def test_protocols_exposed(self) -> None:
        # Re-exports from core land cleanly.
        assert TaskStore is not None
        assert ActivityStore is not None
