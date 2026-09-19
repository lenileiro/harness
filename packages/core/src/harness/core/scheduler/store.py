from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from filelock import FileLock, Timeout

from harness.core.scheduler.models import SchedulerDelivery, SchedulerJob, SchedulerRunRecord
from harness.core.slug import slugify


def _slugify(value: str) -> str:
    return slugify(value)


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class SchedulerStore:
    def __init__(self, *, root: Path):
        self.root = root
        self._held_locks: dict[str, FileLock] = {}

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def ensure_layout(self) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    @property
    def deliveries_dir(self) -> Path:
        return self.root / "deliveries"

    @contextmanager
    def job_state_lock(self, job_id: str) -> Iterator[None]:
        """Short state transactions; never hold this lock while executing a job."""
        target = self.jobs_dir / job_id
        target.mkdir(parents=True, exist_ok=True)
        with FileLock(target / "state.lock", mode=0o600):
            yield

    def _job_lock_path(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / "run.lock"

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
        with self.job_state_lock(job_id):
            job = self.load_job(job_id)
            paused = replace(job, status="paused", updated_at=updated_at)
            self.update_job(paused)
            return paused

    def resume_job(self, job_id: str, *, next_run_at: str, updated_at: str) -> SchedulerJob:
        with self.job_state_lock(job_id):
            job = self.load_job(job_id)
            resumed = replace(job, status="active", next_run_at=next_run_at, updated_at=updated_at)
            self.update_job(resumed)
            return resumed

    def save_delivery(self, delivery: SchedulerDelivery) -> None:
        self.deliveries_dir.mkdir(parents=True, exist_ok=True)
        _write_json(self.deliveries_dir / f"{delivery.id}.json", delivery.to_dict())

    def list_deliveries(self, *, status: str | None = None) -> list[SchedulerDelivery]:
        if not self.deliveries_dir.exists():
            return []
        items = [
            SchedulerDelivery.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.deliveries_dir.glob("*.json"))
        ]
        return [item for item in items if status is None or item.status == status]

    def acquire_delivery_lock(self, delivery_id: str) -> bool:
        return self._acquire_lock(
            f"delivery:{delivery_id}", self.deliveries_dir / f"{delivery_id}.lock"
        )

    def release_delivery_lock(self, delivery_id: str) -> None:
        self._release_lock(f"delivery:{delivery_id}")

    def acquire_job_lock(self, job_id: str) -> bool:
        return self._acquire_lock(f"job:{job_id}", self._job_lock_path(job_id))

    def release_job_lock(self, job_id: str) -> None:
        self._release_lock(f"job:{job_id}")

    def _acquire_lock(self, key: str, path: Path) -> bool:
        if key in self._held_locks:
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = FileLock(path, timeout=0, mode=0o600)
        try:
            lock.acquire()
        except Timeout:
            return False
        self._held_locks[key] = lock
        return True

    def _release_lock(self, key: str) -> None:
        lock = self._held_locks.pop(key, None)
        if lock is not None:
            lock.release()


__all__ = ["SchedulerStore"]
