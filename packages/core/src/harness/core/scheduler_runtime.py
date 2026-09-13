from __future__ import annotations

import hashlib
import logging
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from harness.core.autonomy import run_scheduled_research_burst
from harness.core.extensions import LifecycleHook
from harness.core.mission_runtime import run_scheduled_mission_burst
from harness.core.mission_store import MissionStore, default_mission_root
from harness.core.research_store import ResearchStore, default_research_root
from harness.core.scheduler_models import (
    SchedulerDelivery,
    SchedulerExecutionResult,
    SchedulerJob,
    SchedulerRunRecord,
    ScheduleSpec,
)
from harness.core.scheduler_store import SchedulerStore

_JOB_KINDS = {
    "mission.schedule_once",
    "research.schedule_once",
    "reminder.once",
    "reminder.recurring",
    "prompt.run",
}

MissionJobExecutor = Callable[[SchedulerJob], tuple[str, str, str]]
PromptJobExecutor = Callable[[SchedulerJob], SchedulerExecutionResult]


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
    *,
    at: str | None = None,
    every: str | None = None,
    cron: str | None = None,
    timezone: str = "UTC",
) -> ScheduleSpec:
    try:
        zone = ZoneInfo(timezone)
    except (ValueError, ZoneInfoNotFoundError) as exc:
        raise ValueError(f"unknown schedule timezone: {timezone}") from exc
    provided = [
        (kind, value) for kind, value in (("at", at), ("every", every), ("cron", cron)) if value
    ]
    if len(provided) != 1:
        raise ValueError("exactly one of --at, --every, or --cron is required")
    kind, value = provided[0]
    if kind == "at":
        moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if moment.tzinfo is None:
            wall = moment
            moment = wall.replace(tzinfo=zone)
            if (
                moment.astimezone(UTC).astimezone(zone).replace(tzinfo=None) != wall
                or moment.utcoffset() != wall.replace(tzinfo=zone, fold=1).utcoffset()
            ):
                raise ValueError(
                    "Ambiguous or nonexistent local time; provide an explicit UTC offset"
                )
        return ScheduleSpec(
            kind="at", value=moment.astimezone(UTC).isoformat(timespec="seconds"), timezone=timezone
        )
    if kind == "every":
        return ScheduleSpec(
            kind="every", value=str(_parse_duration_seconds(value)), timezone=timezone
        )
    expression = str(value).strip()
    if len(expression.split()) != 5:
        raise ValueError("cron expressions must have 5 fields: minute hour day month weekday")
    for field_text, maximum, minimum in zip(
        expression.split(), (59, 23, 31, 12, 7), (0, 0, 1, 1, 0), strict=True
    ):
        _cron_values(field_text, minimum=minimum, maximum=maximum)
    return ScheduleSpec(kind="cron", value=expression, timezone=timezone)


def _cron_weekday(value: datetime) -> int:
    return (value.weekday() + 1) % 7


@lru_cache(maxsize=512)
def _cron_values(field: str, *, minimum: int, maximum: int) -> frozenset[int]:
    values: set[int] = set()
    for token in field.split(","):
        part, separator, step_text = token.strip().partition("/")
        step = int(step_text) if separator else 1
        if step < 1:
            raise ValueError("cron step must be positive")
        if part == "*":
            start, end = minimum, maximum
        elif "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
        else:
            start = int(part)
            end = maximum if separator else start
        if not minimum <= start <= end <= maximum:
            raise ValueError("cron field value out of range")
        values.update(range(start, end + 1, step))
    return frozenset(values)


def _match_cron_field(field: str, value: int, *, minimum: int, maximum: int) -> bool:
    return value in _cron_values(field, minimum=minimum, maximum=maximum)


def _matches_cron(expression: str, moment: datetime) -> bool:
    minute_s, hour_s, day_s, month_s, weekday_s = expression.split()
    day_matches = _match_cron_field(day_s, moment.day, minimum=1, maximum=31)
    weekdays = _cron_values(weekday_s, minimum=0, maximum=7)
    weekday_matches = _cron_weekday(moment) in {day % 7 for day in weekdays}
    calendar_matches = (
        (day_matches or weekday_matches)
        if day_s != "*" and weekday_s != "*"
        else (day_matches and weekday_matches)
    )
    return (
        _match_cron_field(minute_s, moment.minute, minimum=0, maximum=59)
        and _match_cron_field(hour_s, moment.hour, minimum=0, maximum=23)
        and calendar_matches
        and _match_cron_field(month_s, moment.month, minimum=1, maximum=12)
    )


def compute_next_run_at(*, schedule: ScheduleSpec, now: datetime | None = None) -> str:
    current = now or _utcnow()
    if schedule.kind == "at":
        return parse_datetime_text(schedule.value).isoformat(timespec="seconds")
    if schedule.kind == "every":
        return (current + timedelta(seconds=int(schedule.value))).isoformat(timespec="seconds")
    if schedule.kind != "cron":
        raise ValueError(f"unsupported schedule kind: {schedule.kind}")
    candidate = current.astimezone(UTC).replace(second=0, microsecond=0)
    zone = ZoneInfo(schedule.timezone)
    if candidate < current.astimezone(UTC):
        candidate += timedelta(minutes=1)
    for _ in range(5 * 366 * 24 * 60):
        if _matches_cron(schedule.value, candidate.astimezone(zone)):
            return candidate.isoformat(timespec="seconds")
        candidate += timedelta(minutes=1)
    raise ValueError("could not compute next cron run within five years")


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
    if job.kind in {"reminder.once", "reminder.recurring"}:
        reminder_text = str(job.payload.get("text", "")).strip() or "Reminder"
        return "completed", reminder_text, ""
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


def _hook_key(hook: LifecycleHook) -> str:
    return f"{type(hook).__module__}.{type(hook).__qualname__}"


def _notification_job(hook: LifecycleHook, job: SchedulerJob) -> SchedulerJob:
    prepare = getattr(hook, "prepare_job_notification", None)
    if not callable(prepare):
        return job
    prepared = prepare(cwd=Path(job.cwd), job=job)
    if not isinstance(prepared, SchedulerJob):
        raise TypeError("notification preparation must return a SchedulerJob")
    return prepared


def retry_scheduler_deliveries(
    *,
    store: SchedulerStore,
    hooks: tuple[LifecycleHook, ...],
    now: datetime | None = None,
    force: bool = False,
) -> int:
    """Retry durable completion hooks without re-executing their jobs.

    Delivery is at least once: a receiver should use record.id to deduplicate if
    the sender crashes after an external send but before persisting its receipt.
    """
    current = now or _utcnow()
    hook_map = {_hook_key(hook): hook for hook in hooks}
    sent = 0
    for pending in store.list_deliveries(status="pending"):
        hook = hook_map.get(pending.hook_key)
        if hook is None or not store.acquire_delivery_lock(pending.id):
            continue
        try:
            delivery = next(item for item in store.list_deliveries() if item.id == pending.id)
            if delivery.status != "pending":
                continue
            if (
                not force
                and delivery.next_attempt_at
                and parse_datetime_text(delivery.next_attempt_at) > current
            ):
                continue
            attempts = delivery.attempts + 1
            try:
                record = store.load_run_record(delivery.run_id)
                job = SchedulerJob.from_dict(delivery.job)
                if not delivery.prepared:
                    job = _notification_job(hook, job)
                    delivery = replace(delivery, job=job.to_dict(), prepared=True)
                    store.save_delivery(delivery)
                hook.on_job_completed(
                    cwd=Path(job.cwd), job=job, trigger=record.trigger, record=record
                )
            except Exception as exc:
                store.save_delivery(
                    replace(
                        delivery,
                        attempts=attempts,
                        last_error=str(exc),
                        next_attempt_at=(
                            current + timedelta(seconds=min(3600, 30 * 2 ** min(attempts - 1, 7)))
                        ).isoformat(timespec="seconds"),
                    )
                )
            else:
                store.save_delivery(
                    replace(
                        delivery,
                        status="sent",
                        attempts=attempts,
                        next_attempt_at="",
                        last_error="",
                    )
                )
                sent += 1
        finally:
            store.release_delivery_lock(pending.id)
    return sent


def run_scheduler_job(
    *,
    store: SchedulerStore,
    job_id: str,
    trigger: str = "manual",
    now: datetime | None = None,
    hooks: tuple[LifecycleHook, ...] = (),
    mission_executor: MissionJobExecutor | None = None,
    prompt_executor: PromptJobExecutor | None = None,
) -> SchedulerRunRecord:
    job = store.load_job(job_id)
    started = now or _utcnow()
    started_text = started.isoformat(timespec="seconds")

    def skipped(reason: str) -> SchedulerRunRecord:
        record = SchedulerRunRecord(
            id=store.new_id("schedrun", job.kind),
            job_id=job.id,
            kind=job.kind,
            cwd=job.cwd,
            trigger=trigger,
            status="skipped",
            result_status="skipped",
            result_stop_reason=reason,
            started_at=started_text,
            finished_at=_utcnow_text(),
            record_dir="",
            summary=f"{job.kind} skipped: {reason}",
        )
        store.add_run_record(record)
        return record

    if not store.acquire_job_lock(job_id):
        return skipped("already_running")
    try:
        # Another scheduler may have finished this occurrence since our due snapshot.
        with store.job_state_lock(job_id):
            job = store.load_job(job_id)
            if trigger == "scheduled" and (
                job.status != "active"
                or not job.next_run_at
                or parse_datetime_text(job.next_run_at) > started.astimezone(UTC)
            ):
                return skipped("not_due")
            # Persist the claim before side effects. A crash leaves an explicit
            # running job that can be inspected and retried with run-now.
            if job.status != "paused":
                store.update_job(replace(job, status="running", updated_at=started_text))
        retry_after_seconds: float | None = None
        try:
            for hook in hooks:
                hook.on_job_started(cwd=Path(job.cwd), job=job, trigger=trigger, started_at=started)
            if job.kind == "prompt.run":
                if prompt_executor is None:
                    raise ValueError("Prompt job requires a configured prompt executor")
                result = prompt_executor(job)
                retry_after_seconds = result.retry_after_seconds
                if retry_after_seconds is not None and (
                    not math.isfinite(retry_after_seconds) or retry_after_seconds <= 0
                ):
                    raise ValueError("Prompt retry delay must be finite and positive")
                result_status, stop_reason, record_dir = (
                    result.status,
                    result.stop_reason,
                    result.record_dir,
                )
                job = replace(
                    job,
                    payload={
                        **job.payload,
                        "notification_text": result.notification_text,
                        "notify_result": result.notify_result,
                    },
                )
            elif (
                job.kind == "mission.schedule_once" and job.payload.get("execution_mode") == "agent"
            ):
                if mission_executor is None:
                    raise ValueError("Agent mission job requires a configured mission executor")
                result_status, stop_reason, record_dir = mission_executor(job)
            else:
                result_status, stop_reason, record_dir = _dispatch_job(job)
            status = "completed"
            summary = f"{job.kind} -> {result_status} ({stop_reason})"
        except Exception as exc:
            result_status, stop_reason, record_dir = "failed", "error", ""
            status, summary = "failed", str(exc)
        finished = _utcnow()
        finished_text = finished.isoformat(timespec="seconds")
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
        # Persist the outbox before marking execution complete. A delivery failure
        # changes the outbox only, never the result of the completed work.
        for hook in hooks:
            key = _hook_key(hook)
            delivery_id = f"{record.id}-{hashlib.sha256(key.encode()).hexdigest()[:16]}"
            error = ""
            try:
                snapshot = _notification_job(hook, job)
            except Exception as exc:
                snapshot, error = job, str(exc)
            store.save_delivery(
                SchedulerDelivery(
                    id=delivery_id,
                    run_id=record.id,
                    hook_key=key,
                    job=snapshot.to_dict(),
                    prepared=not bool(error),
                    last_error=error,
                )
            )
        with store.job_state_lock(job_id):
            latest = store.load_job(job_id)
            next_run_at = (
                ""
                if job.schedule.kind == "at"
                else compute_next_run_at(
                    schedule=job.schedule, now=finished + timedelta(microseconds=1)
                )
            )
            final_status = status if job.schedule.kind == "at" else "active"
            if status == "completed" and retry_after_seconds is not None:
                next_run_at = (finished + timedelta(seconds=retry_after_seconds)).isoformat(
                    timespec="seconds"
                )
                final_status = "active"
            if latest.status == "paused":
                final_status = "paused"
            store.update_job(
                replace(
                    latest,
                    status=final_status,
                    next_run_at=next_run_at,
                    updated_at=finished_text,
                    last_run_at=finished_text,
                    last_status=result_status,
                    last_error=summary if status == "failed" else "",
                    last_record_dir=record_dir,
                )
            )
    finally:
        store.release_job_lock(job_id)
    retry_scheduler_deliveries(store=store, hooks=hooks)
    return record


def run_due_scheduler_jobs(
    *,
    store: SchedulerStore,
    now: datetime | None = None,
    hooks: tuple[LifecycleHook, ...] = (),
    mission_executor: MissionJobExecutor | None = None,
    prompt_executor: PromptJobExecutor | None = None,
) -> SchedulerTickResult:
    started = now or _utcnow()
    started_text = started.isoformat(timespec="seconds")
    retry_scheduler_deliveries(store=store, hooks=hooks, now=started)
    jobs = store.list_jobs(status="active")
    due = [
        job
        for job in jobs
        if job.next_run_at and parse_datetime_text(job.next_run_at) <= started.astimezone(UTC)
    ]
    run_ids: list[str] = []
    executed = 0
    for job in due:
        record = run_scheduler_job(
            store=store,
            job_id=job.id,
            trigger="scheduled",
            now=started,
            hooks=hooks,
            mission_executor=mission_executor,
            prompt_executor=prompt_executor,
        )
        run_ids.append(record.id)
        executed += record.status != "skipped"
    finished = _utcnow()
    finished_text = finished.isoformat(timespec="seconds")
    result = SchedulerTickResult(
        started_at=started_text,
        finished_at=finished_text,
        jobs_seen=len(jobs),
        jobs_executed=executed,
        run_ids=tuple(run_ids),
    )
    scheduler_cwd = store.root.parent.resolve()
    for hook in hooks:
        try:
            hook.on_scheduler_tick(
                cwd=scheduler_cwd,
                started_at=started,
                finished_at=finished,
                jobs_seen=result.jobs_seen,
                jobs_executed=result.jobs_executed,
                run_ids=result.run_ids,
            )
        except Exception:
            logging.getLogger(__name__).exception("Scheduler tick hook failed")
    return result


def run_scheduler_loop(
    *,
    store: SchedulerStore,
    poll_interval_seconds: float = 30.0,
    once: bool = False,
    max_ticks: int | None = None,
    hooks: tuple[LifecycleHook, ...] = (),
    mission_executor: MissionJobExecutor | None = None,
    prompt_executor: PromptJobExecutor | None = None,
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
        result = run_due_scheduler_jobs(
            store=store,
            hooks=hooks,
            mission_executor=mission_executor,
            prompt_executor=prompt_executor,
        )
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
    "retry_scheduler_deliveries",
    "run_due_scheduler_jobs",
    "run_scheduler_job",
    "run_scheduler_loop",
]
