from __future__ import annotations

import asyncio
import os
from pathlib import Path

from harness.storage.sqlite import SQLiteStorage


async def _load_job_state(db_path: Path) -> tuple[object | None, list[object]]:
    storage = SQLiteStorage(path=db_path)
    try:
        root = await storage.get_task("task_root_eval")
        assert root is not None
        items = await storage.list_tasks(parent_id=root.id)
        return root, items
    finally:
        await storage.close()


def test_lab_orchestration_flow() -> None:
    workspace = Path(os.environ.get("HARNESS_EVAL_WORKSPACE", "."))
    db_path = workspace / "lab.db"
    assert db_path.exists(), "lab.db was not created"

    root, items = asyncio.run(_load_job_state(db_path))
    assert root is not None
    assert root.status == "done"
    assert len(items) == 2

    by_title = {item.title: item for item in items}
    assert set(by_title) == {"Draft plan", "Execute change"}
    assert by_title["Draft plan"].status == "done"
    assert by_title["Execute change"].status == "done"
    assert by_title["Draft plan"].metadata.get("result_summary") == "planner split work"
    assert by_title["Execute change"].metadata.get("result_summary") == "worker finished execution"

    report_path = workspace / "lab_report.md"
    assert report_path.exists(), "reporter artifact was not created"
    report = report_path.read_text(encoding="utf-8")
    assert "Reporter summary" in report
    assert "Draft plan" in report
    assert "Execute change" in report
