from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from harness.core.extensions import LifecycleHook
from harness.core.scheduler_models import SchedulerJob, ScheduleSpec
from harness.core.scheduler_runtime import (
    compute_next_run_at,
    parse_datetime_text,
    parse_schedule_spec,
    retry_scheduler_deliveries,
    run_due_scheduler_jobs,
    run_scheduler_job,
)
from harness.core.scheduler_store import SchedulerStore


def test_parse_schedule_spec_normalizes_variants() -> None:
    at = parse_schedule_spec(at="2026-05-26T12:00:00Z")
    assert at.kind == "at"
    assert at.value.endswith("+00:00")

    every = parse_schedule_spec(every="5m")
    assert every.kind == "every"
    assert every.value == "300"

    cron = parse_schedule_spec(cron="*/15 * * * *")
    assert cron.kind == "cron"
    assert cron.value == "*/15 * * * *"


def test_compute_next_run_at_supports_interval_and_cron() -> None:
    now = datetime(2026, 5, 26, 12, 7, tzinfo=UTC)
    interval = parse_schedule_spec(every="90")
    cron = parse_schedule_spec(cron="*/15 * * * *")

    interval_next = parse_datetime_text(compute_next_run_at(schedule=interval, now=now))
    cron_next = parse_datetime_text(compute_next_run_at(schedule=cron, now=now))

    assert interval_next == datetime(2026, 5, 26, 12, 8, 30, tzinfo=UTC)
    assert cron_next == datetime(2026, 5, 26, 12, 15, tzinfo=UTC)


def test_compute_next_run_at_keeps_exact_due_cron_minute() -> None:
    now = datetime(2026, 5, 26, 12, 15, 0, tzinfo=UTC)
    cron = parse_schedule_spec(cron="15 12 * * *")

    cron_next = parse_datetime_text(compute_next_run_at(schedule=cron, now=now))

    assert cron_next == datetime(2026, 5, 26, 12, 15, 0, tzinfo=UTC)


def test_run_scheduler_job_emits_hooks(tmp_path: Path, monkeypatch) -> None:
    class RecordingHook:
        def __init__(self) -> None:
            self.events: list[tuple[str, object]] = []

        def on_scheduler_tick(self, **kwargs) -> None:
            self.events.append(("tick", kwargs))

        def on_job_started(self, **kwargs) -> None:
            self.events.append(("started", kwargs["job"].id))

        def on_job_completed(self, **kwargs) -> None:
            self.events.append(("completed", kwargs["record"].id))

        def on_gateway_message(self, **kwargs) -> None:
            self.events.append(("message", kwargs))

        def on_gateway_reply(self, **kwargs) -> None:
            self.events.append(("reply", kwargs))

        def on_approval_requested(self, **kwargs) -> None:
            self.events.append(("approval_requested", kwargs))

        def on_approval_resolved(self, **kwargs) -> None:
            self.events.append(("approval_resolved", kwargs))

    store = SchedulerStore(root=tmp_path / ".harness" / "scheduler")
    job = SchedulerJob(
        id="sched-demo-1",
        kind="research.schedule_once",
        cwd=str(tmp_path),
        status="active",
        schedule=ScheduleSpec(kind="at", value="2026-05-26T12:00:00+00:00"),
        next_run_at="2026-05-26T12:00:00+00:00",
        payload={},
        created_at="2026-05-26T11:59:00+00:00",
        updated_at="2026-05-26T11:59:00+00:00",
    )
    store.add_job(job)

    monkeypatch.setattr(
        "harness.core.scheduler_runtime._dispatch_job",
        lambda job: ("completed", "ok", str(tmp_path / ".harness" / "runs" / "demo")),
    )

    hook = RecordingHook()
    record = run_scheduler_job(store=store, job_id=job.id, hooks=(hook,))
    assert record.result_status == "completed"
    assert hook.events[0] == ("started", job.id)
    assert hook.events[1] == ("completed", record.id)


def test_run_scheduler_job_skips_if_job_lock_is_already_held(tmp_path: Path) -> None:
    store = SchedulerStore(root=tmp_path / ".harness" / "scheduler")
    job = SchedulerJob(
        id="sched-demo-locked",
        kind="research.schedule_once",
        cwd=str(tmp_path),
        status="active",
        schedule=ScheduleSpec(kind="at", value="2026-05-26T12:00:00+00:00"),
        next_run_at="2026-05-26T12:00:00+00:00",
        payload={},
        created_at="2026-05-26T11:59:00+00:00",
        updated_at="2026-05-26T11:59:00+00:00",
    )
    store.add_job(job)
    assert store.acquire_job_lock(job.id) is True

    record = run_scheduler_job(store=store, job_id=job.id)

    assert record.status == "skipped"
    assert record.result_status == "skipped"
    assert record.result_stop_reason == "already_running"
    store.release_job_lock(job.id)


def _seed_job(tmp_path: Path, *, once: bool = False) -> tuple[SchedulerStore, SchedulerJob]:
    store = SchedulerStore(root=tmp_path / "scheduler")
    job = SchedulerJob(
        id="scheduled-reminder",
        kind="reminder.once" if once else "reminder.recurring",
        cwd=str(tmp_path),
        status="active",
        next_run_at="2020-01-01T00:00:00+00:00",
        schedule=ScheduleSpec(kind="at", value="2020-01-01T00:00:00+00:00")
        if once
        else ScheduleSpec(kind="every", value="60"),
        payload={"text": "test reminder", "notify_to": "original-recipient"},
    )
    store.add_job(job)
    return store, job


def test_two_due_snapshots_dispatch_one_occurrence(tmp_path: Path, monkeypatch) -> None:
    from threading import Barrier, Event, Thread, current_thread

    store, job = _seed_job(tmp_path, once=True)
    list_jobs = store.list_jobs
    snapshots, first_finished = Barrier(2), Event()
    dispatches: list[str] = []
    results: list[int] = []
    failures: list[BaseException] = []

    def staged_list(*args, **kwargs):
        items = list_jobs(*args, **kwargs)
        snapshots.wait(timeout=5)
        if current_thread().name == "second":
            assert first_finished.wait(timeout=5)
        return items

    def dispatch(item):
        dispatches.append(item.id)
        return "completed", "sent", ""

    def tick():
        try:
            results.append(run_due_scheduler_jobs(store=store).jobs_executed)
        except BaseException as exc:
            failures.append(exc)
        finally:
            if current_thread().name == "first":
                first_finished.set()

    monkeypatch.setattr(store, "list_jobs", staged_list)
    monkeypatch.setattr("harness.core.scheduler_runtime._dispatch_job", dispatch)
    threads = [Thread(target=tick, name=name) for name in ("first", "second")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads)
    assert not failures
    assert dispatches == [job.id]
    assert sorted(results) == [0, 1]


def test_pause_during_dispatch_survives_completion_and_failure(tmp_path: Path, monkeypatch) -> None:
    for fail in (False, True):
        store, job = _seed_job(tmp_path / str(fail))

        def dispatch(item, store=store, fail=fail):
            store.pause_job(item.id, updated_at=datetime.now(UTC).isoformat())
            if fail:
                raise RuntimeError("worker failed")
            return "completed", "ok", ""

        monkeypatch.setattr("harness.core.scheduler_runtime._dispatch_job", dispatch)
        result = run_scheduler_job(store=store, job_id=job.id)
        assert result.status == ("failed" if fail else "completed")
        assert store.load_job(job.id).status == "paused"


def test_start_hook_failure_is_recorded_and_releases_lock(tmp_path: Path) -> None:
    store, job = _seed_job(tmp_path)

    class BrokenHook:
        def on_job_started(self, **kwargs):
            raise RuntimeError("start failed")

        def on_job_completed(self, **kwargs):
            pass

    result = run_scheduler_job(
        store=store, job_id=job.id, hooks=(cast(LifecycleHook, BrokenHook()),)
    )
    assert result.status == "failed"
    assert result.summary == "start failed"
    assert store.acquire_job_lock(job.id)
    store.release_job_lock(job.id)
    assert run_scheduler_job(store=store, job_id=job.id).status == "completed"


def test_delivery_retries_after_restart_without_reexecuting_job(
    tmp_path: Path, monkeypatch
) -> None:
    store, job = _seed_job(tmp_path, once=True)
    dispatched: list[str] = []
    sends: list[str] = []

    def dispatch(item):
        dispatched.append(item.id)
        return "completed", "reminder", ""

    class DeliveryHook:
        def on_job_started(self, **kwargs):
            pass

        def on_job_completed(self, **kwargs):
            sends.append(kwargs["job"].payload["notify_to"])
            if len(sends) == 1:
                raise ConnectionError("bridge offline")

    monkeypatch.setattr("harness.core.scheduler_runtime._dispatch_job", dispatch)
    result = run_scheduler_job(
        store=store, job_id=job.id, hooks=(cast(LifecycleHook, DeliveryHook()),)
    )
    assert result.status == "completed"
    assert store.load_job(job.id).status == "completed"
    pending = store.list_deliveries(status="pending")
    assert len(pending) == 1
    assert pending[0].attempts == 1
    assert pending[0].last_error == "bridge offline"

    restarted = SchedulerStore(root=store.root)
    retry_at = parse_datetime_text(pending[0].next_attempt_at) + timedelta(seconds=1)
    assert (
        retry_scheduler_deliveries(
            store=restarted, hooks=(cast(LifecycleHook, DeliveryHook()),), now=retry_at
        )
        == 1
    )
    assert (
        retry_scheduler_deliveries(
            store=restarted, hooks=(cast(LifecycleHook, DeliveryHook()),), now=retry_at
        )
        == 0
    )
    assert sends == ["original-recipient", "original-recipient"]
    assert dispatched == [job.id]
    assert restarted.list_deliveries()[0].status == "sent"
