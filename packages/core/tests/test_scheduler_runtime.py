from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from harness.core.scheduler_models import SchedulerJob, ScheduleSpec
from harness.core.scheduler_runtime import (
    compute_next_run_at,
    parse_datetime_text,
    parse_schedule_spec,
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
