from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.tasks import Task


class FakeOrchestrator:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def run(self, prompt: str) -> AsyncIterator[object]:
        if False:
            yield None

    async def resume(self, job_id: str) -> AsyncIterator[object]:
        if False:
            yield None


def _run(args: list[str]) -> Any:
    return CliRunner().invoke(cli_main.app, args)


async def _seed_lab_db(db: Path, cwd: Path) -> str:
    from harness.storage.sqlite import SQLiteStorage

    storage = SQLiteStorage(path=db)
    try:
        job = Task(ref="T-001", title="Root job", cwd=cwd, status="in_progress")
        await storage.create_task(job)
        child = Task(
            ref="T-002",
            title="Child task",
            cwd=cwd,
            parent_id=job.id,
            status="done",
            metadata={"result_summary": "finished work"},
        )
        await storage.create_task(child)
        return job.id
    finally:
        await storage.close()


class TestLabCommands:
    def test_lab_run_wrapper(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(cli_main, "MultiAgentOrchestrator", FakeOrchestrator)
        result = _run(
            [
                "lab",
                "run",
                "do work",
                "--cwd",
                str(tmp_path),
                "--no-judge",
            ]
        )
        assert result.exit_code == 0, result.stdout
        assert "harness lab run" in result.stdout

    def test_lab_list_and_status(self, tmp_path: Path) -> None:
        import asyncio

        db = tmp_path / "lab.db"
        job_id = asyncio.run(_seed_lab_db(db, tmp_path))

        listing = _run(["lab", "list", "--db", str(db)])
        assert listing.exit_code == 0, listing.stdout
        assert "Root job" in listing.stdout
        assert "1/1 items" in listing.stdout

        status = _run(["lab", "status", job_id, "--db", str(db)])
        assert status.exit_code == 0, status.stdout
        assert "Child task" in status.stdout
        assert "finished work" in status.stdout
