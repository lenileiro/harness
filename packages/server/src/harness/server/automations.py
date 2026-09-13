"""Owner-scoped prompt schedules, dispatched atomically through Harness's queue."""

from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from harness.core.scheduler_models import ScheduleSpec
from harness.core.scheduler_runtime import (
    compute_next_run_at,
    parse_datetime_text,
    parse_schedule_spec,
)
from harness.server.models import RunSubmission, ServiceError
from harness.server.store import now

if TYPE_CHECKING:
    from harness.server.service import HarnessService


class ScheduleSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    title: str = Field(min_length=1, max_length=120)
    prompt: str = Field(min_length=1, max_length=100000)
    provider: str | None = Field(default=None, min_length=1, max_length=128)
    model: str | None = Field(default=None, min_length=1, max_length=256)
    max_steps: int = Field(default=25, ge=1, le=100)
    at: str | None = Field(default=None, max_length=128)
    every: str | None = Field(default=None, max_length=128)
    cron: str | None = Field(default=None, max_length=128)
    timezone: str | None = Field(default=None, max_length=128)


class AutomationManager:
    def __init__(self, service: HarnessService):
        self.service = service
        self._wake = asyncio.Event()
        self._calendar_lock = asyncio.Lock()

    async def _next(self, spec: ScheduleSpec, moment: datetime) -> str:
        # Existing cron evaluation is bounded to five years; keep its CPU scan
        # off the ASGI loop and serialize it so requests cannot fan out workers.
        async with self._calendar_lock:
            task = asyncio.create_task(
                asyncio.to_thread(compute_next_run_at, schedule=spec, now=moment)
            )
            try:
                return await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise
            except (ValueError, OverflowError):
                raise ServiceError(422, "Schedule has no valid upcoming occurrence") from None

    async def create(self, owner: str, submission: ScheduleSubmission) -> dict:
        self.service.scope(owner)
        moment = datetime.now(UTC)
        preferences = await self.service.preferences(owner)
        try:
            spec = parse_schedule_spec(
                at=submission.at,
                every=submission.every,
                cron=submission.cron,
                timezone=submission.timezone or preferences.timezone,
            )
        except (ValueError, OverflowError):
            raise ServiceError(
                422,
                "Choose exactly one valid at/every/cron schedule and timezone; ambiguous local times require an explicit UTC offset",
            ) from None
        if spec.kind == "every" and not 60 <= int(spec.value) <= 366 * 86400:
            raise ServiceError(422, "Recurring intervals must be between 60 seconds and 366 days")
        due = await self._next(spec, moment)
        if spec.kind == "at" and parse_datetime_text(due) <= moment:
            raise ServiceError(422, "One-time schedules must be in the future")
        request = await self.service.resolve_submission(
            owner,
            RunSubmission(
                prompt=submission.prompt,
                provider=submission.provider,
                model=submission.model,
                max_steps=submission.max_steps,
            ),
        )
        identifier = "schedule_" + uuid4().hex
        stamp = now()
        async with self.service.store.connection() as db:
            await db.execute("BEGIN IMMEDIATE")
            async with db.execute(
                "SELECT COUNT(*) FROM api_schedules WHERE owner=? AND state IN ('active','paused','error')",
                (owner,),
            ) as cursor:
                count_row = await cursor.fetchone()
                assert count_row is not None
                count = count_row[0]
            if count >= 100:
                raise ServiceError(
                    409, "An identity can own at most 100 active or paused schedules"
                )
            await db.execute(
                "INSERT INTO api_schedules VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    identifier,
                    owner,
                    submission.title,
                    request.model_dump_json(),
                    json.dumps(spec.to_dict()),
                    due,
                    "active",
                    stamp,
                    stamp,
                    None,
                ),
            )
            await db.commit()
        self._wake.set()
        return await self.get(owner, identifier)

    async def get(self, owner: str, identifier: str) -> dict:
        rows = await self.service.store.rows(
            "SELECT * FROM api_schedules WHERE id=? AND owner=?", (identifier, owner)
        )
        if not rows:
            raise ServiceError(404, "Schedule not found")
        row = rows[0]
        runs = await self.service.store.rows(
            "SELECT a.due_at,a.run_id,r.session_id FROM api_schedule_runs a JOIN api_runs r ON r.id=a.run_id WHERE a.schedule_id=? AND r.owner=? ORDER BY a.due_at DESC LIMIT 50",
            (identifier, owner),
        )
        result_runs = []
        for run in runs:
            attempts = await self.service.store.rows(
                "SELECT id FROM api_runs WHERE session_id=? AND owner=? ORDER BY created_at DESC,id DESC LIMIT 1",
                (run["session_id"], owner),
            )
            result_runs.append(
                {
                    "due_at": run["due_at"],
                    "initial_run_id": run["run_id"],
                    "run": await self.service.store.run(owner, attempts[0]["id"]),
                }
            )
        from harness.server.service import public

        return public(
            {
                **row,
                "request": json.loads(row["request"]),
                "schedule": json.loads(row["schedule"]),
                "runs": result_runs,
            }
        )

    async def list(self, owner: str) -> list[dict]:
        rows = await self.service.store.rows(
            "SELECT id FROM api_schedules WHERE owner=? ORDER BY created_at DESC,id LIMIT 200",
            (owner,),
        )
        return [await self.get(owner, row["id"]) for row in rows]

    async def change(self, owner: str, identifier: str, action: str) -> dict:
        current = await self.get(owner, identifier)
        if action not in {"pause", "resume", "cancel"}:
            raise ServiceError(422, "Unknown schedule action")
        if current["state"] == "cancelled" and action == "cancel":
            return current
        if current["state"] == "cancelled" or (
            current["state"] == "completed" and action != "cancel"
        ):
            raise ServiceError(409, "Finished schedules cannot be changed; create a new schedule")
        due = current["next_run_at"]
        state = {"pause": "paused", "resume": "active", "cancel": "cancelled"}[action]
        if action == "resume":
            spec = ScheduleSpec.from_dict(current["schedule"])
            moment = datetime.now(UTC)
            due = (
                now()
                if spec.kind == "at" and parse_datetime_text(spec.value) <= moment
                else await self._next(spec, moment + timedelta(microseconds=1))
            )
            self.service.validate_provider(
                RunSubmission.model_validate(current["request"]).provider
            )
        async with self.service.store.connection() as db:
            changed = await db.execute(
                "UPDATE api_schedules SET state=?,next_run_at=?,updated_at=?,error=NULL WHERE id=? AND owner=? AND state=? AND updated_at=?",
                (state, due, now(), identifier, owner, current["state"], current["updated_at"]),
            )
            if changed.rowcount != 1:
                raise ServiceError(409, "Schedule changed concurrently; refresh and retry")
            await db.commit()
        if action == "cancel":
            rows = await self.service.store.rows(
                "SELECT r.id FROM api_runs r WHERE r.owner=? AND r.state IN ('queued','running','paused') AND r.session_id IN (SELECT initial.session_id FROM api_schedule_runs a JOIN api_runs initial ON initial.id=a.run_id WHERE a.schedule_id=?)",
                (owner, identifier),
            )
            for row in rows:
                await self.service.cancel(owner, row["id"])
        self._wake.set()
        return await self.get(owner, identifier)

    async def tick(self, *, moment: datetime | None = None) -> list[str]:
        current = moment or datetime.now(UTC)
        rows = await self.service.store.rows(
            "SELECT * FROM api_schedules WHERE state='active' AND next_run_at<=? ORDER BY next_run_at,id LIMIT 100",
            (current.isoformat(),),
        )
        submitted: list[str] = []
        for row in rows:
            # A paused approval is unfinished work. Check latest attempts in each
            # occurrence's session so an explicitly resumed success unblocks it.
            active = await self.service.store.rows(
                "SELECT 1 FROM api_schedule_runs a JOIN api_runs initial ON initial.id=a.run_id JOIN api_runs r ON r.session_id=initial.session_id WHERE a.schedule_id=? AND r.id=(SELECT recent.id FROM api_runs recent WHERE recent.session_id=r.session_id ORDER BY recent.created_at DESC,recent.id DESC LIMIT 1) AND r.state IN ('queued','running','paused') LIMIT 1",
                (row["id"],),
            )
            if active:
                continue
            try:
                request = RunSubmission.model_validate_json(row["request"])
                self.service.validate_provider(request.provider)
                self.service._validate_submission(request)
                spec = ScheduleSpec.from_dict(json.loads(row["schedule"]))
                next_due = (
                    None
                    if spec.kind == "at"
                    else await self._next(spec, current + timedelta(microseconds=1))
                )
                result = await self.service._enqueue(
                    row["owner"],
                    [("agent", request.model_dump(), None)],
                    batch=False,
                    automation=(row["id"], row["next_run_at"], next_due),
                )
                submitted.append(result[0]["id"])
            except (ServiceError, ValueError, OverflowError) as exc:
                if isinstance(exc, ServiceError) and exc.status == 409:
                    continue
                async with self.service.store.connection() as db:
                    await db.execute(
                        "UPDATE api_schedules SET state='error',error=?,updated_at=? WHERE id=? AND state='active' AND next_run_at=?",
                        (
                            exc.detail
                            if isinstance(exc, ServiceError)
                            else "Saved schedule is invalid; inspect and recreate it",
                            now(),
                            row["id"],
                            row["next_run_at"],
                        ),
                    )
                    await db.commit()
        return submitted

    async def run(self) -> None:
        while True:
            self._wake.clear()
            await self.tick()
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=1)
