from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from harness.core.scheduler_models import SchedulerJob, SchedulerRunRecord


def _slugify(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "item"


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class SchedulerStore:
    def __init__(self, *, root: Path):
        self.root = root

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def ensure_layout(self) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def new_id(self, prefix: str, title: str) -> str:
        return f"{prefix}-{_slugify(title)[:32]}-{uuid4().hex[:8]}"

    def add_job(self, job: SchedulerJob) -> Path:
        self.ensure_layout()
        target = self.jobs_dir / job.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "job.json", job.to_dict())
        lines = [
            f"# Scheduler Job {job.id}",
            "",
            f"- kind: `{job.kind}`",
            f"- status: `{job.status}`",
            f"- cwd: `{job.cwd}`",
            f"- schedule: `{job.schedule.kind}:{job.schedule.value}`",
            f"- next_run_at: `{job.next_run_at}`",
        ]
        if job.last_run_at:
            lines.append(f"- last_run_at: `{job.last_run_at}`")
        if job.last_status:
            lines.append(f"- last_status: `{job.last_status}`")
        if job.last_record_dir:
            lines.append(f"- last_record_dir: `{job.last_record_dir}`")
        if job.last_error:
            lines.extend(["", "## Last Error", job.last_error])
        (target / "JOB.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        return target

    def update_job(self, job: SchedulerJob) -> Path:
        return self.add_job(job)

    def load_job(self, job_id: str) -> SchedulerJob:
        path = self.jobs_dir / job_id / "job.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return SchedulerJob.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_jobs(
        self, *, status: str | None = None, kind: str | None = None
    ) -> list[SchedulerJob]:
        if not self.jobs_dir.exists():
            return []
        items: list[SchedulerJob] = []
        for path in sorted(self.jobs_dir.iterdir()):
            payload = path / "job.json"
            if not payload.is_file():
                continue
            job = SchedulerJob.from_dict(json.loads(payload.read_text(encoding="utf-8")))
            if status and job.status != status:
                continue
            if kind and job.kind != kind:
                continue
            items.append(job)
        return items

    def add_run_record(self, record: SchedulerRunRecord) -> Path:
        self.ensure_layout()
        target = self.runs_dir / record.id
        target.mkdir(parents=True, exist_ok=True)
        _write_json(target / "run.json", record.to_dict())
        lines = [
            f"# Scheduler Run {record.id}",
            "",
            f"- job_id: `{record.job_id}`",
            f"- kind: `{record.kind}`",
            f"- trigger: `{record.trigger}`",
            f"- status: `{record.status}`",
            f"- result_status: `{record.result_status}`",
            f"- result_stop_reason: `{record.result_stop_reason}`",
            f"- cwd: `{record.cwd}`",
            f"- started_at: `{record.started_at}`",
            f"- finished_at: `{record.finished_at}`",
            f"- record_dir: `{record.record_dir}`",
            "",
            "## Summary",
            record.summary,
        ]
        (target / "RUN.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
        return target

    def load_run_record(self, run_id: str) -> SchedulerRunRecord:
        path = self.runs_dir / run_id / "run.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return SchedulerRunRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_run_records(self, *, job_id: str | None = None) -> list[SchedulerRunRecord]:
        if not self.runs_dir.exists():
            return []
        items: list[SchedulerRunRecord] = []
        for path in sorted(self.runs_dir.iterdir()):
            payload = path / "run.json"
            if not payload.is_file():
                continue
            record = SchedulerRunRecord.from_dict(json.loads(payload.read_text(encoding="utf-8")))
            if job_id and record.job_id != job_id:
                continue
            items.append(record)
        return items

    def pause_job(self, job_id: str, *, updated_at: str) -> SchedulerJob:
        job = self.load_job(job_id)
        paused = replace(job, status="paused", updated_at=updated_at)
        self.update_job(paused)
        return paused

    def resume_job(self, job_id: str, *, next_run_at: str, updated_at: str) -> SchedulerJob:
        job = self.load_job(job_id)
        resumed = replace(job, status="active", next_run_at=next_run_at, updated_at=updated_at)
        self.update_job(resumed)
        return resumed


__all__ = ["SchedulerStore"]
