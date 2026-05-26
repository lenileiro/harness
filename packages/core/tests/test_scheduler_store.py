from __future__ import annotations

from harness.core.scheduler_models import SchedulerJob, SchedulerRunRecord, ScheduleSpec
from harness.core.scheduler_store import SchedulerStore


def test_scheduler_store_round_trip_jobs_and_runs(tmp_path) -> None:
    store = SchedulerStore(root=tmp_path / ".harness" / "scheduler")
    job = SchedulerJob(
        id="sched-demo",
        kind="mission.schedule_once",
        cwd=str(tmp_path),
        status="active",
        schedule=ScheduleSpec(kind="every", value="60"),
        next_run_at="2026-05-26T12:00:00+00:00",
        payload={"mission_id": "mission-demo", "max_steps": 2},
    )
    store.add_job(job)

    loaded = store.load_job(job.id)
    assert loaded.kind == "mission.schedule_once"
    assert loaded.schedule.kind == "every"
    assert loaded.payload["mission_id"] == "mission-demo"
    assert len(store.list_jobs()) == 1

    run = SchedulerRunRecord(
        id="schedrun-demo",
        job_id=job.id,
        kind=job.kind,
        cwd=job.cwd,
        trigger="manual",
        status="completed",
        result_status="paused",
        result_stop_reason="max_steps",
        started_at="2026-05-26T12:00:00+00:00",
        finished_at="2026-05-26T12:00:05+00:00",
        record_dir=str(tmp_path / "artifact"),
        summary="mission.schedule_once -> paused (max_steps)",
    )
    store.add_run_record(run)

    loaded_run = store.load_run_record(run.id)
    assert loaded_run.summary.startswith("mission.schedule_once")
    assert len(store.list_run_records(job_id=job.id)) == 1
