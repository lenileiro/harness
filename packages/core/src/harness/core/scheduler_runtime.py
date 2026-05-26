from __future__ import annotations

import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

from harness.core.autonomy import run_scheduled_research_burst
from harness.core.extensions import LifecycleHook
from harness.core.mission_runtime import run_scheduled_mission_burst
from harness.core.mission_store import MissionStore, default_mission_root
from harness.core.research_store import ResearchStore, default_research_root
from harness.core.scheduler_models import SchedulerJob, SchedulerRunRecord, ScheduleSpec
from harness.core.scheduler_store import SchedulerStore

_JOB_KINDS = {"mission.schedule_once", "research.schedule_once"}


@dataclass(frozen=True, slots=True)
class SchedulerTickResult:
    started_at: str
    finished_at: str
    jobs_seen: int
    jobs_executed: int
    run_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "jobs_seen": self.jobs_seen,
            "jobs_executed": self.jobs_executed,
            "run_ids": list(self.run_ids),
        }


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _utcnow_text() -> str:
    return _utcnow().isoformat(timespec="seconds")


def parse_datetime_text(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_duration_seconds(value: str) -> int:
    raw = value.strip().lower()
    if not raw:
        raise ValueError("duration cannot be empty")
    if raw.isdigit():
        seconds = int(raw)
        if seconds < 1:
            raise ValueError("duration must be at least 1 second")
        return seconds
    suffix_map = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    suffix = raw[-1]
    factor = suffix_map.get(suffix)
    if factor is None or not raw[:-1].isdigit():
        raise ValueError("duration must be an integer with optional s/m/h/d suffix")
    seconds = int(raw[:-1]) * factor
    if seconds < 1:
        raise ValueError("duration must be at least 1 second")
    return seconds


def parse_schedule_spec(
    *, at: str | None = None, every: str | None = None, cron: str | None = None
) -> ScheduleSpec:
    provided = [
        (kind, value) for kind, value in (("at", at), ("every", every), ("cron", cron)) if value
    ]
    if len(provided) != 1:
        raise ValueError("exactly one of --at, --every, or --cron is required")
    kind, value = provided[0]
    if kind == "at":
        moment = parse_datetime_text(value)
        return ScheduleSpec(kind="at", value=moment.isoformat(timespec="seconds"))
    if kind == "every":
        return ScheduleSpec(kind="every", value=str(_parse_duration_seconds(value)))
    expression = str(value).strip()
    if len(expression.split()) != 5:
        raise ValueError("cron expressions must have 5 fields: minute hour day month weekday")
    return ScheduleSpec(kind="cron", value=expression)


def _cron_weekday(value: datetime) -> int:
    return (value.weekday() + 1) % 7


def _match_cron_field(field: str, value: int, *, minimum: int, maximum: int) -> bool:
    for token in field.split(","):
        part = token.strip()
        if part == "*":
            return True
        if part.startswith("*/"):
            step = int(part[2:])
            if step < 1:
                raise ValueError("cron step must be positive")
            if (value - minimum) % step == 0:
                return True
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start < minimum or end > maximum or start > end:
                raise ValueError("invalid cron range")
            if start <= value <= end:
                return True
            continue
        number = int(part)
        if number < minimum or number > maximum:
            raise ValueError("cron field value out of range")
        if value == number:
            return True
    return False


def _matches_cron(expression: str, moment: datetime) -> bool:
    minute_s, hour_s, day_s, month_s, weekday_s = expression.split()
    return (
        _match_cron_field(minute_s, moment.minute, minimum=0, maximum=59)
        and _match_cron_field(hour_s, moment.hour, minimum=0, maximum=23)
        and _match_cron_field(day_s, moment.day, minimum=1, maximum=31)
        and _match_cron_field(month_s, moment.month, minimum=1, maximum=12)
        and _match_cron_field(weekday_s, _cron_weekday(moment), minimum=0, maximum=6)
    )


def compute_next_run_at(*, schedule: ScheduleSpec, now: datetime | None = None) -> str:
    current = now or _utcnow()
    if schedule.kind == "at":
        return parse_datetime_text(schedule.value).isoformat(timespec="seconds")
    if schedule.kind == "every":
        return (current + timedelta(seconds=int(schedule.value))).isoformat(timespec="seconds")
    if schedule.kind != "cron":
        raise ValueError(f"unsupported schedule kind: {schedule.kind}")
    candidate = current.astimezone(UTC).replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(366 * 24 * 60):
        if _matches_cron(schedule.value, candidate):
            return candidate.isoformat(timespec="seconds")
        candidate += timedelta(minutes=1)
    raise ValueError("could not compute next cron run within one year")


def create_scheduler_job(
    *,
    store: SchedulerStore,
    kind: str,
    cwd: Path,
    schedule: ScheduleSpec,
    payload: dict[str, object],
    title: str,
) -> SchedulerJob:
    if kind not in _JOB_KINDS:
        raise ValueError(f"unsupported scheduler job kind: {kind}")
    created_at = _utcnow()
    if schedule.kind == "at":
        next_run_at = parse_datetime_text(schedule.value).isoformat(timespec="seconds")
    else:
        next_run_at = compute_next_run_at(schedule=schedule, now=created_at)
    timestamp = created_at.isoformat(timespec="seconds")
    return SchedulerJob(
        id=store.new_id("sched", title),
        kind=kind,
        cwd=str(cwd.resolve()),
        status="active",
        schedule=schedule,
        next_run_at=next_run_at,
        payload=payload,
        created_at=timestamp,
        updated_at=timestamp,
    )


def _dispatch_job(job: SchedulerJob) -> tuple[str, str, str]:
    working_dir = Path(job.cwd)
    if job.kind == "mission.schedule_once":
        mission_id = str(job.payload.get("mission_id", "")).strip()
        if not mission_id:
            raise ValueError("mission scheduler job is missing mission_id")
        mission_store = MissionStore(root=default_mission_root(working_dir))
        result, record_dir = run_scheduled_mission_burst(
            store=mission_store,
            cwd=working_dir,
            mission_id=mission_id,
            max_steps=int(job.payload.get("max_steps", 20)),
            auto_complete=bool(job.payload.get("auto_complete", False)),
        )
        return result.status, result.stop_reason, str(record_dir)
    if job.kind == "research.schedule_once":
        research_store = ResearchStore(root=default_research_root(working_dir))
        result, record_dir = run_scheduled_research_burst(
            store=research_store,
            cwd=working_dir,
            max_steps=int(job.payload.get("max_steps", 5)),
            max_risk=str(job.payload.get("max_risk", "medium")),
            base_branch=str(job.payload.get("base_branch", "main")),
            create_branch=bool(job.payload.get("create_branch", False)),
            commit=bool(job.payload.get("commit", False)),
            push=bool(job.payload.get("push", False)),
            open_pr=bool(job.payload.get("open_pr", False)),
            draft_pr=bool(job.payload.get("draft_pr", True)),
        )
        return result.status, result.stop_reason, str(record_dir)
    raise ValueError(f"unsupported scheduler job kind: {job.kind}")


def run_scheduler_job(
    *,
    store: SchedulerStore,
    job_id: str,
    trigger: str = "manual",
    now: datetime | None = None,
    hooks: tuple[LifecycleHook, ...] = (),
) -> SchedulerRunRecord:
    job = store.load_job(job_id)
    started = now or _utcnow()
    started_text = started.isoformat(timespec="seconds")
    job_cwd = Path(job.cwd)
    for hook in hooks:
        hook.on_job_started(cwd=job_cwd, job=job, trigger=trigger, started_at=started)
    try:
        result_status, stop_reason, record_dir = _dispatch_job(job)
        status = "completed"
        summary = f"{job.kind} -> {result_status} ({stop_reason})"
    except Exception as exc:
        finished_text = _utcnow().isoformat(timespec="seconds")
        record = SchedulerRunRecord(
            id=store.new_id("schedrun", job.kind),
            job_id=job.id,
            kind=job.kind,
            cwd=job.cwd,
            trigger=trigger,
            status="failed",
            result_status="failed",
            result_stop_reason="error",
            started_at=started_text,
            finished_at=finished_text,
            record_dir="",
            summary=str(exc),
        )
        store.add_run_record(record)
        next_run_at = ""
        status = "failed" if job.schedule.kind == "at" else "active"
        if job.schedule.kind != "at":
            next_run_at = compute_next_run_at(schedule=job.schedule, now=_utcnow())
        store.update_job(
            replace(
                job,
                status=status,
                next_run_at=next_run_at,
                updated_at=finished_text,
                last_run_at=finished_text,
                last_status="failed",
                last_error=str(exc),
                last_record_dir="",
            )
        )
        for hook in hooks:
            hook.on_job_completed(cwd=job_cwd, job=job, trigger=trigger, record=record)
        return record
    finished_text = _utcnow().isoformat(timespec="seconds")
    record = SchedulerRunRecord(
        id=store.new_id("schedrun", job.kind),
        job_id=job.id,
        kind=job.kind,
        cwd=job.cwd,
        trigger=trigger,
        status=status,
        result_status=result_status,
        result_stop_reason=stop_reason,
        started_at=started_text,
        finished_at=finished_text,
        record_dir=record_dir,
        summary=summary,
    )
    store.add_run_record(record)
    next_run_at = ""
    job_status = "completed" if job.schedule.kind == "at" else job.status
    if job.schedule.kind != "at":
        next_run_at = compute_next_run_at(schedule=job.schedule, now=_utcnow())
    store.update_job(
        replace(
            job,
            status=job_status,
            next_run_at=next_run_at,
            updated_at=finished_text,
            last_run_at=finished_text,
            last_status=result_status,
            last_error="",
            last_record_dir=record_dir,
        )
    )
    for hook in hooks:
        hook.on_job_completed(cwd=job_cwd, job=job, trigger=trigger, record=record)
    return record


def run_due_scheduler_jobs(
    *,
    store: SchedulerStore,
    now: datetime | None = None,
    hooks: tuple[LifecycleHook, ...] = (),
) -> SchedulerTickResult:
    started = now or _utcnow()
    started_text = started.isoformat(timespec="seconds")
    jobs = store.list_jobs(status="active")
    due = [
        job
        for job in jobs
        if job.next_run_at and parse_datetime_text(job.next_run_at) <= started.astimezone(UTC)
    ]
    run_ids: list[str] = []
    for job in due:
        record = run_scheduler_job(
            store=store,
            job_id=job.id,
            trigger="scheduled",
            now=started,
            hooks=hooks,
        )
        run_ids.append(record.id)
    finished = _utcnow()
    finished_text = finished.isoformat(timespec="seconds")
    result = SchedulerTickResult(
        started_at=started_text,
        finished_at=finished_text,
        jobs_seen=len(jobs),
        jobs_executed=len(due),
        run_ids=tuple(run_ids),
    )
    scheduler_cwd = store.root.parent.resolve()
    for hook in hooks:
        hook.on_scheduler_tick(
            cwd=scheduler_cwd,
            started_at=started,
            finished_at=finished,
            jobs_seen=result.jobs_seen,
            jobs_executed=result.jobs_executed,
            run_ids=result.run_ids,
        )
    return result


def run_scheduler_loop(
    *,
    store: SchedulerStore,
    poll_interval_seconds: float = 30.0,
    once: bool = False,
    max_ticks: int | None = None,
    hooks: tuple[LifecycleHook, ...] = (),
) -> SchedulerTickResult:
    ticks = 0
    result = SchedulerTickResult(
        started_at=_utcnow_text(),
        finished_at=_utcnow_text(),
        jobs_seen=0,
        jobs_executed=0,
        run_ids=(),
    )
    while True:
        result = run_due_scheduler_jobs(store=store, hooks=hooks)
        ticks += 1
        if once:
            return result
        if max_ticks is not None and ticks >= max_ticks:
            return result
        time.sleep(poll_interval_seconds)


__all__ = [
    "SchedulerTickResult",
    "compute_next_run_at",
    "create_scheduler_job",
    "parse_datetime_text",
    "parse_schedule_spec",
    "run_due_scheduler_jobs",
    "run_scheduler_job",
    "run_scheduler_loop",
]
