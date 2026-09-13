"""One durable, identity-scoped execution path for HTTP and MCP clients."""

from __future__ import annotations

import asyncio
import inspect
import json
import sqlite3
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import PrivateAttr

from harness.core import Agent
from harness.core.adapter import Adapter
from harness.core.events import (
    Done,
    ErrorEvent,
    Event,
    ModelRequestEvent,
    ToolCallEvent,
    ToolResultEvent,
)
from harness.core.failover import FailoverPolicy
from harness.core.memory import MemoryScope
from harness.core.schemas import (
    ApprovalDecision,
    Capabilities,
    Message,
    RunRequest,
    Session,
    ToolCall,
)
from harness.core.secret_redaction import redact_secrets
from harness.core.session_search import session_scope
from harness.core.tools import ApprovalPolicy, InboxApprovalHandler, Tool
from harness.server.a2a_store import A2ABinding
from harness.server.models import (
    TERMINAL_STATES,
    AgentBuilder,
    RunContext,
    RunSubmission,
    ServerPresentation,
    ServiceError,
    ToolSubmission,
    UserPreferences,
)
from harness.server.store import ServiceStore, now
from harness.storage.sqlite import SQLiteStorage

if TYPE_CHECKING:
    from harness.server.a2a_callbacks import CallbackManager


def identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def public(value: Any) -> Any:
    """Redact emitted diagnostics without publishing injected model instructions."""
    if isinstance(value, str):
        return redact_secrets(value)[0]
    if isinstance(value, dict):
        return {key: item if key == "attachments" else public(item) for key, item in value.items()}
    if isinstance(value, list):
        return [public(item) for item in value]
    return value


class ServerApprovalPolicy(ApprovalPolicy):
    _base: ApprovalPolicy = PrivateAttr()
    _exposed: frozenset[str] = PrivateAttr()

    def __init__(self, base: ApprovalPolicy, exposed: Sequence[str]):
        super().__init__(default="deny")
        self._base = base
        self._exposed = frozenset(exposed)

    def decide(
        self, tool: Tool, *, session_overrides: dict[str, ApprovalDecision] | None = None
    ) -> ApprovalDecision:
        if tool.name not in self._exposed:
            return "deny"
        decisions = [
            self._base.per_tool.get(tool.name),
            (session_overrides or {}).get(tool.name),
            tool.approval,
        ]
        if (
            "deny" in decisions
            or self._base.decide(tool, session_overrides=session_overrides) == "deny"
        ):
            return "deny"
        if "prompt" in decisions or getattr(tool, "effect_scope", None) not in (
            "read_only",
            "session_ephemeral",
        ):
            return "prompt"
        return "auto"


class SingleActionAdapter(Adapter):
    """Supply one explicit tool call to Agent; Agent owns every policy/evidence gate."""

    name = "api_action"

    def __init__(self, action: ToolSubmission | None):
        self.action = action
        self.sent = False

    async def capabilities(self) -> Capabilities:
        return Capabilities(tool_use=True)

    async def cancel(self, session_id: str) -> None:
        pass

    async def stream(
        self, *, model: str, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[Event]:
        if self.action is not None and not self.sent:
            self.sent = True
            call = ToolCall(
                id=identifier("call"), name=self.action.name, arguments=self.action.arguments
            )
            yield ToolCallEvent(call=call)
            yield Done(final_message=Message(role="assistant", tool_calls=[call]))
        else:
            yield Done(
                final_message=Message(
                    role="assistant",
                    content="Action processing finished; inspect the tool result and run status.",
                )
            )


class HarnessService:
    def __init__(
        self,
        database: Path,
        workspace: Path,
        agent_builder: AgentBuilder,
        *,
        exposed_tools: Sequence[str] = (),
        max_workers: int = 2,
        presentation: ServerPresentation | None = None,
    ):
        if not 1 <= max_workers <= 32:
            raise ValueError("max_workers must be between 1 and 32")
        self.database = database.expanduser().resolve()
        self.workspace = workspace.expanduser().resolve()
        self.agent_builder = agent_builder
        self.exposed_tools = tuple(exposed_tools)
        self.max_workers = max_workers
        self.presentation = presentation or ServerPresentation()
        self._automation_task: asyncio.Task[None] | None = None
        self.a2a_callbacks: CallbackManager | None = None
        self.storage = SQLiteStorage(path=self.database)
        self.store = ServiceStore(self.database)
        self._wake = asyncio.Event()
        self._schedule_lock = asyncio.Lock()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._dispatcher: asyncio.Task[None] | None = None
        self._closing = False
        from harness.server.delegation import DelegationManager

        self.delegation = DelegationManager(self)
        from harness.server.automations import AutomationManager

        self.automations = AutomationManager(self)
        from harness.server.questions import QuestionManager

        self.questions = QuestionManager(self)

    async def start(self, *, dispatch: bool = True) -> None:
        if self._dispatcher is not None or self.store.db is not None:
            raise RuntimeError("Service is already running")
        self._closing = False
        try:
            await self.store.start(acquire_lease=dispatch)
            await self.storage.list(limit=1)
            await self.questions.start()
            if dispatch:
                for row in await self.store.rows("SELECT id FROM api_runs WHERE state='running'"):
                    await self.store.finish(
                        row["id"],
                        "interrupted",
                        "Server restarted during execution; inspect effects before explicitly resuming.",
                    )
            if dispatch:
                self._dispatcher = asyncio.create_task(self._dispatch(), name="harness-api-queue")
                self._wake.set()
                self._automation_task = asyncio.create_task(
                    self.automations.run(), name="harness-api-schedules"
                )
                if self.a2a_callbacks is not None:
                    await self.a2a_callbacks.start()
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        self._closing = True
        if self.a2a_callbacks is not None:
            await self.a2a_callbacks.close()
        if self._automation_task is not None:
            self._automation_task.cancel()
            await asyncio.gather(self._automation_task, return_exceptions=True)
            self._automation_task = None
        if self._dispatcher is not None:
            self._dispatcher.cancel()
            await asyncio.gather(self._dispatcher, return_exceptions=True)
            self._dispatcher = None
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for run_id in self._tasks:
            await self.store.finish(
                run_id,
                "interrupted",
                "Server stopped during execution; inspect effects before resuming",
            )
        self._tasks.clear()
        await self.questions.close()
        await self.storage.close()
        await self.store.close()

    def scope(self, owner: str, *, workspace: Path | None = None) -> MemoryScope:
        return MemoryScope(workspace=str(workspace or self.workspace), user_id=owner)

    async def session_workspace(self, session_id: str) -> Path:
        rows = await self.store.rows(
            "SELECT workspace FROM api_session_workspaces WHERE session_id=?", (session_id,)
        )
        return Path(rows[0]["workspace"]) if rows else self.workspace

    async def _owned_session(self, owner: str, session_id: str) -> Session | None:
        rows = await self.store.rows(
            "SELECT id FROM api_sessions WHERE id=? AND owner=?", (session_id, owner)
        )
        if not rows:
            raise ServiceError(404, "Session not found")
        session = await self.storage.get(session_id)
        if session is not None and session_scope(session) != self.scope(
            owner, workspace=await self.session_workspace(session_id)
        ):
            raise ServiceError(404, "Session not found")
        return session

    def _validate_submission(self, submission: RunSubmission) -> None:
        # Use the core schema for media validation; never silently discard attachments.
        values = submission.model_dump(exclude={"session_id"})
        if not values["attachments"] and "attachments" not in RunRequest.model_fields:
            values.pop("attachments")
        RunRequest.model_validate(values)
        if submission.model is not None and not submission.model.strip():
            raise ServiceError(422, "Model must be nonempty")
        if not submission.prompt.strip() and not submission.attachments:
            raise ServiceError(422, "A prompt or attachment is required")

    def validate_provider(self, provider: str | None) -> None:
        if provider is not None and provider not in {
            item.id for item in self.presentation.providers
        }:
            raise ServiceError(403, "Provider is not enabled by this server")

    async def preferences(self, owner: str) -> UserPreferences:
        self.scope(owner)
        rows = await self.store.rows(
            "SELECT preferences FROM api_preferences WHERE owner=?", (owner,)
        )
        return (
            UserPreferences.model_validate_json(rows[0]["preferences"])
            if rows
            else UserPreferences()
        )

    async def save_preferences(self, owner: str, preferences: UserPreferences) -> UserPreferences:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

        self.scope(owner)
        self.validate_provider(preferences.provider)
        try:
            ZoneInfo(preferences.timezone)
        except (ValueError, ZoneInfoNotFoundError):
            raise ServiceError(422, "Unknown timezone") from None
        async with self.store.connection() as db:
            await db.execute(
                "INSERT INTO api_preferences VALUES (?,?,?) ON CONFLICT(owner) DO UPDATE SET preferences=excluded.preferences,updated_at=excluded.updated_at",
                (owner, preferences.model_dump_json(), now()),
            )
            await db.commit()
        return preferences

    def configuration(self) -> dict[str, Any]:
        return public(
            {
                "providers": [item.model_dump() for item in self.presentation.providers],
                "default_provider": self.presentation.default_provider,
                "default_model": self.presentation.default_model,
                "preferences_fields": ["provider", "model", "timezone"],
                "automations": {
                    "minimum_interval_seconds": 60,
                    "overlap": "wait for previous session, including approvals",
                    "missed_runs": "coalesce into one occurrence",
                },
            }
        )

    def tool_catalog(self) -> list[dict[str, Any]]:
        catalog = {tool.name: tool for tool in self.presentation.tools}
        result = []
        for name in sorted(set(catalog) | set(self.exposed_tools)):
            tool = catalog.get(name)
            row = (
                tool.model_dump()
                if tool
                else {
                    "name": name,
                    "description": "Runtime tool; schema becomes available when its session starts.",
                    "parameters_schema": None,
                    "approval": "prompt",
                    "effect_scope": "unknown",
                }
            )
            row["exposed"] = name in self.exposed_tools
            row["effective_approval"] = (
                "deny"
                if name not in self.exposed_tools or row["approval"] == "deny"
                else "prompt"
                if row["effect_scope"] not in {"read_only", "session_ephemeral"}
                else row["approval"]
            )
            result.append(public(row))
        return result

    async def resolve_submission(self, owner: str, submission: RunSubmission) -> RunSubmission:
        value = submission.model_copy(deep=True)
        if value.session_id:
            session = await self._owned_session(owner, value.session_id)
            if session is not None:
                value.provider = value.provider or session.provider
                if value.model is None:
                    option = next(
                        (item for item in self.presentation.providers if item.id == value.provider),
                        None,
                    )
                    value.model = (
                        session.model
                        if value.provider == session.provider
                        else option.default_model
                        if option
                        else None
                    )
                    if value.model is None:
                        raise ServiceError(422, "Select a model when changing providers")
                # Legacy injected builders have no provider-selection surface.
                if not self.presentation.providers and submission.provider is None:
                    value.provider = None
        else:
            preferences = await self.preferences(owner)
            value.provider = (
                value.provider or preferences.provider or self.presentation.default_provider
            )
            if value.model is None:
                option = next(
                    (item for item in self.presentation.providers if item.id == value.provider),
                    None,
                )
                value.model = (
                    (
                        preferences.model
                        if value.provider
                        == (preferences.provider or self.presentation.default_provider)
                        else None
                    )
                    or (option.default_model if option else None)
                    or (
                        self.presentation.default_model
                        if value.provider == self.presentation.default_provider
                        else None
                    )
                )
        self.validate_provider(value.provider)
        if value.provider is not None and value.model is None:
            raise ServiceError(
                422, "Select a model for this provider; no default model is configured"
            )
        self._validate_submission(value)
        return value

    async def submit(self, owner: str, submission: RunSubmission) -> dict[str, Any]:
        submission = await self.resolve_submission(owner, submission)
        rows = await self._enqueue(owner, [("agent", submission.model_dump(), None)], batch=False)
        return rows[0]

    async def submit_tool(self, owner: str, submission: ToolSubmission) -> dict[str, Any]:
        if submission.name not in self.exposed_tools:
            raise ServiceError(403, "Tool is not exposed by this server")
        rows = await self._enqueue(owner, [("tool", submission.model_dump(), None)], batch=False)
        return rows[0]

    async def submit_batch(self, owner: str, submissions: list[RunSubmission]) -> dict[str, Any]:
        if not 1 <= len(submissions) <= 100:
            raise ServiceError(422, "A batch must contain between 1 and 100 runs")
        submissions = [
            await self.resolve_submission(owner, submission) for submission in submissions
        ]
        rows = await self._enqueue(
            owner, [("agent", item.model_dump(), None) for item in submissions], batch=True
        )
        return await self.batch(owner, rows[0]["batch_id"])

    async def _enqueue(
        self,
        owner: str,
        items: list[tuple[str, dict[str, Any], str | None]],
        *,
        batch: bool,
        automation: tuple[str, str, str | None] | None = None,
        a2a: A2ABinding | None = None,
    ) -> list[dict[str, Any]]:
        self.scope(owner)  # Validate trusted identity before persistence.
        if a2a is not None and len(items) != 1:
            raise ServiceError(422, "An A2A message must enqueue exactly one run")
        if self._closing:
            raise ServiceError(503, "Server is shutting down")
        for _, request, _ in items:
            if request.get("session_id"):
                owned_session = await self._owned_session(owner, request["session_id"])
                await self.questions.check_admission(owned_session)
                delegated = await self.store.rows(
                    "SELECT id FROM api_delegations WHERE session_id=?", (request["session_id"],)
                )
                if delegated:
                    raise ServiceError(
                        409,
                        "Use the delegated job resume operation to reserve another execution budget",
                    )
                pending = await self.storage.list_approvals(
                    session_id=request["session_id"], status="pending", limit=1
                )
                if pending:
                    raise ServiceError(
                        409, "Resolve pending approvals before continuing this session"
                    )
        batch_id = identifier("batch") if batch else None
        ids = []
        try:
            async with self.store.connection() as db:
                await db.execute("BEGIN IMMEDIATE")
                reused = await a2a.existing(db, owner) if a2a is not None else None
                if reused:
                    ids.append(reused)
                else:
                    if automation is not None:
                        schedule_id, due_at, next_at = automation
                        changed = await db.execute(
                            "UPDATE api_schedules SET next_run_at=?,state=?,updated_at=? WHERE id=? AND owner=? AND state='active' AND next_run_at=?",
                            (
                                next_at,
                                "active" if next_at else "completed",
                                now(),
                                schedule_id,
                                owner,
                                due_at,
                            ),
                        )
                        if changed.rowcount != 1:
                            raise ServiceError(
                                409, "Schedule occurrence was already claimed or paused"
                            )
                    if batch_id:
                        await db.execute(
                            "INSERT INTO api_batches VALUES (?,?,?)", (batch_id, owner, now())
                        )
                    for kind, request, resumed_from in items:
                        if resumed_from:
                            async with db.execute(
                                "SELECT id,state,kind FROM api_runs WHERE id=? AND owner=?",
                                (resumed_from, owner),
                            ) as cursor:
                                previous = await cursor.fetchone()
                            if previous is None:
                                raise ServiceError(404, "Original run not found")
                            if previous[
                                "state"
                            ] == "cancelled" and not await self._unstarted_cancelled_run(
                                db, owner, previous
                            ):
                                raise ServiceError(
                                    409,
                                    "A cancelled run that started, or belongs to A2A, cannot be resumed; submit a new explicit request",
                                )
                        session_id = request.get("session_id") or identifier("sess")
                        stamp = now()
                        if not request.get("session_id"):
                            await db.execute(
                                "INSERT INTO api_sessions VALUES (?,?,?)",
                                (session_id, owner, stamp),
                            )
                        run_id = identifier("run")
                        ids.append(run_id)
                        await db.execute(
                            "INSERT INTO api_runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                run_id,
                                owner,
                                session_id,
                                "queued",
                                kind,
                                json.dumps(request),
                                batch_id,
                                resumed_from,
                                stamp,
                                stamp,
                                None,
                            ),
                        )
                        await db.execute(
                            "INSERT INTO api_events(run_id,payload,created_at) VALUES (?,?,?)",
                            (run_id, json.dumps({"type": "run_status", "state": "queued"}), stamp),
                        )
                        if a2a is not None:
                            await a2a.save(db, owner, session_id, run_id)
                        elif resumed_from:
                            # Resuming from the web/API preserves the A2A task
                            # handle observed by its original client.
                            await db.execute(
                                "INSERT OR IGNORE INTO api_a2a_task_runs SELECT ?,id FROM api_a2a_tasks WHERE owner=? AND run_id=?",
                                (run_id, owner, resumed_from),
                            )
                            await db.execute(
                                "UPDATE api_a2a_tasks SET run_id=? WHERE owner=? AND run_id=?",
                                (run_id, owner, resumed_from),
                            )
                    if automation is not None:
                        await db.execute(
                            "INSERT INTO api_schedule_runs VALUES (?,?,?)",
                            (automation[0], automation[1], ids[0]),
                        )
                await db.commit()
        except sqlite3.IntegrityError:
            raise ServiceError(
                409, "A session can have only one queued or running request"
            ) from None
        self._wake.set()
        return [await self.store.run(owner, run_id) for run_id in ids]

    async def _unstarted_cancelled_run(self, db, owner: str, row) -> bool:
        """Only an unexecuted ordinary agent request may restart after cancellation."""
        if row["kind"] != "agent":
            return False
        async with db.execute(
            "SELECT 1 FROM api_events WHERE run_id=? AND json_extract(payload,'$.type')='run_status' AND json_extract(payload,'$.state')='running' LIMIT 1",
            (row["id"],),
        ) as cursor:
            if await cursor.fetchone():
                return False
        async with db.execute(
            "SELECT 1 FROM api_a2a_task_runs r JOIN api_a2a_tasks t ON t.id=r.task_id WHERE r.run_id=? AND t.owner=? LIMIT 1",
            (row["id"], owner),
        ) as cursor:
            return await cursor.fetchone() is None

    async def resume(self, owner: str, run_id: str, prompt: str | None = None) -> dict[str, Any]:
        previous = await self.store.run(owner, run_id)
        restarting_unstarted = False
        if previous["state"] == "cancelled":
            async with self.store.connection() as db:
                restarting_unstarted = await self._unstarted_cancelled_run(db, owner, previous)
            if not restarting_unstarted:
                raise ServiceError(
                    409,
                    "A cancelled run that started, or belongs to A2A, cannot be resumed; submit a new explicit request",
                )
        delegated = await self.store.rows(
            "SELECT id FROM api_delegations WHERE session_id=? AND owner=?",
            (previous["session_id"], owner),
        )
        if delegated:
            job = await self.delegation.resume(owner, delegated[0]["id"], prompt=prompt)
            return job["run"]
        if previous["state"] not in TERMINAL_STATES:
            raise ServiceError(409, "Only a finished, paused, or interrupted run can be resumed")
        if previous["kind"] == "tool" and previous["state"] != "paused":
            raise ServiceError(
                409,
                "Tool runs can resume only after approval; inspect uncertain effects and submit a new explicit action",
            )
        if previous["kind"] == "tool":
            approvals = await self.storage.list_approvals(
                session_id=previous["session_id"], limit=100
            )
            if any(item.status == "denied" for item in approvals):
                raise ServiceError(
                    409,
                    "This action was denied; submit a new explicit action to request approval again",
                )
        request: dict[str, Any] = {"session_id": previous["session_id"]}
        if previous["kind"] == "agent":
            raw = (await self.store.rows("SELECT request FROM api_runs WHERE id=?", (run_id,)))[0]
            original = RunSubmission.model_validate_json(raw["request"])
            request = original.model_copy(
                update={
                    "session_id": previous["session_id"],
                    "attachments": original.attachments if restarting_unstarted else [],
                    "prompt": prompt
                    or (
                        original.prompt
                        if restarting_unstarted
                        else "Continue from saved state without repeating completed actions."
                    ),
                }
            ).model_dump()
            self.validate_provider(original.provider)
            self._validate_submission(RunSubmission.model_validate(request))
        return (await self._enqueue(owner, [(previous["kind"], request, run_id)], batch=False))[0]

    async def cancel(self, owner: str, run_id: str) -> dict[str, Any]:
        current = await self.store.run(owner, run_id)
        if current["state"] == "paused":
            # The API queue and approval inbox share a database. Revoke only
            # unclaimed effects, atomically with the paused run transition.
            async with self.store.connection() as db:
                await db.execute("BEGIN IMMEDIATE")
                async with db.execute(
                    "SELECT id FROM api_runs WHERE session_id=? AND state IN ('queued','running')",
                    (current["session_id"],),
                ) as cursor:
                    active = await cursor.fetchone()
                async with db.execute(
                    "SELECT id FROM approvals WHERE session_id=? AND status='granted' AND replay_claimed_at IS NOT NULL AND replayed_at IS NULL",
                    (current["session_id"],),
                ) as cursor:
                    claimed = await cursor.fetchone()
                if active or claimed:
                    raise ServiceError(
                        409,
                        "The session has resumed or an action was already claimed; inspect the latest run and effects before cancelling",
                    )
                changed = await db.execute(
                    "UPDATE api_runs SET state='cancelled',updated_at=? WHERE id=? AND state='paused'",
                    (now(), run_id),
                )
                if changed.rowcount:
                    await db.execute(
                        "UPDATE clarifications SET status='cancelled',payload=json_set(payload,'$.status','cancelled') WHERE session_id=? AND applied=0 AND status IN('pending','answered')",
                        (current["session_id"],),
                    )
                    await db.execute(
                        "UPDATE approvals SET status='denied',resolved_at=?,resolved_by=? WHERE session_id=? AND (status='pending' OR (status='granted' AND replay_claimed_at IS NULL AND replayed_at IS NULL))",
                        (now(), owner, current["session_id"]),
                    )
                    await db.execute(
                        "INSERT INTO api_events(run_id,payload,created_at) VALUES (?,?,?)",
                        (run_id, json.dumps({"type": "run_status", "state": "cancelled"}), now()),
                    )
                await db.commit()
            current = await self.store.run(owner, run_id)
        if current["state"] in TERMINAL_STATES:
            for child in await self.delegation.list(owner, current["session_id"]):
                await self.delegation.cancel(owner, child["id"], current["session_id"])
            return current
        async with self._schedule_lock:
            async with self.store.connection() as db:
                await db.execute(
                    "INSERT INTO api_run_controls(run_id,cancel_requested) VALUES (?,1) ON CONFLICT(run_id) DO UPDATE SET cancel_requested=1",
                    (run_id,),
                )
                await db.commit()
            task = self._tasks.get(run_id)
            if task is not None and not task.cancelling():
                task.cancel()
            elif task is None:
                await self.store.finish(run_id, "cancelled", queued_only=True)
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
            await self.store.finish(run_id, "cancelled")
        for child in await self.delegation.list(owner, current["session_id"]):
            await self.delegation.cancel(owner, child["id"], current["session_id"])
        self._wake.set()
        return await self.store.run(owner, run_id)

    async def runs(self, owner: str, *, session_id: str | None = None) -> list[dict[str, Any]]:
        if session_id:
            await self._owned_session(owner, session_id)
            rows = await self.store.rows(
                "SELECT id FROM api_runs WHERE owner=? AND session_id=? ORDER BY created_at DESC,id LIMIT 50",
                (owner, session_id),
            )
        else:
            rows = await self.store.rows(
                "SELECT id FROM api_runs WHERE owner=? ORDER BY created_at DESC,id LIMIT 50",
                (owner,),
            )
        return [await self.store.run(owner, row["id"]) for row in rows]

    async def batches(self, owner: str) -> list[dict[str, Any]]:
        rows = await self.store.rows(
            "SELECT id FROM api_batches WHERE owner=? ORDER BY created_at DESC,id LIMIT 50",
            (owner,),
        )
        return [await self.batch(owner, row["id"]) for row in rows]

    async def batch(self, owner: str, batch_id: str) -> dict[str, Any]:
        rows = await self.store.rows(
            "SELECT * FROM api_batches WHERE id=? AND owner=?", (batch_id, owner)
        )
        if not rows:
            raise ServiceError(404, "Batch not found")
        runs = await self.store.rows(
            "SELECT id FROM api_runs WHERE batch_id=? AND owner=? ORDER BY created_at,id",
            (batch_id, owner),
        )
        values = [await self.store.run(owner, item["id"]) for item in runs]
        return {
            **rows[0],
            "complete": all(item["state"] in TERMINAL_STATES for item in values),
            "runs": values,
        }

    async def cancel_batch(self, owner: str, batch_id: str) -> dict[str, Any]:
        batch = await self.batch(owner, batch_id)
        for run in batch["runs"]:
            await self.cancel(owner, run["id"])
        return await self.batch(owner, batch_id)

    async def _dispatch(self) -> None:
        while True:
            self._wake.clear()
            requests = await self.store.rows(
                "SELECT run_id FROM api_run_controls WHERE cancel_requested=1"
            )
            for request in requests:
                task = self._tasks.get(request["run_id"])
                if task is not None and not task.done() and not task.cancelling():
                    task.cancel()
            self._tasks = {key: task for key, task in self._tasks.items() if not task.done()}
            slots = self.max_workers - len(self._tasks)
            async with self._schedule_lock:
                if slots:
                    async with self.store.connection() as db:
                        async with db.execute(
                            "SELECT * FROM api_runs WHERE state='queued' ORDER BY created_at,id LIMIT ?",
                            (slots,),
                        ) as cursor:
                            rows = [dict(row) for row in await cursor.fetchall()]
                        claimed = []
                        for row in rows:
                            update = await db.execute(
                                "UPDATE api_runs SET state='running',updated_at=? WHERE id=? AND state='queued'",
                                (now(), row["id"]),
                            )
                            if update.rowcount:
                                claimed.append(row)
                        rows = claimed
                        await db.commit()
                    # No await between the claim and task registration: cancellation can find its task.
                    for row in rows:
                        self._tasks[row["id"]] = asyncio.create_task(
                            self._execute(row), name=f"harness-{row['id']}"
                        )
            with suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=0.1)

    async def _agent(self, row: dict[str, Any]) -> Agent:
        request = json.loads(row["request"])
        workspace = await self.session_workspace(row["session_id"])
        context = RunContext(
            row["owner"],
            workspace,
            self.storage,
            row["id"],
            row["session_id"],
            request.get("model"),
            request.get("provider"),
        )
        self.validate_provider(context.provider)
        result = self.agent_builder(context)
        agent = await result if inspect.isawaitable(result) else result
        # All tool access must stay inside Harness's approval/evidence pipeline.
        for adapter in agent.adapters.values():
            if not (await adapter.capabilities()).external_tools:
                raise ServiceError(
                    422,
                    "Native agent adapters are unavailable on the server because their tools bypass Harness approvals",
                )
        agent.storage = self.storage
        agent.question_store = self.questions.store if "clarify" in self.exposed_tools else None
        agent.question_store_factory = None
        agent.activity_store = self.storage
        agent.approval_store = self.storage
        agent.memory_store = self.storage
        agent.memory_scope = self.scope(row["owner"], workspace=workspace)
        agent.default_cwd = str(workspace)
        agent.approval_policy = ServerApprovalPolicy(agent.approval_policy, self.exposed_tools)
        agent.approval_handler = InboxApprovalHandler(approval_store=self.storage)
        agent.pause_on_approval = True
        delegation_names = {
            "delegate",
            "delegate_status",
            "delegate_cancel",
            "delegate_resume",
            "delegate_artifact",
        }.intersection(self.exposed_tools)
        if delegation_names:
            from harness.server.delegation import DelegationTool

            await self.delegation.register_parent(row["owner"], row["session_id"], workspace)
            for name in sorted(delegation_names):
                agent.tools.register(
                    DelegationTool(self.delegation, row["owner"], row["session_id"], name)
                )
        if row["kind"] == "tool":
            action = None if row["resumed_from"] else ToolSubmission.model_validate(request)
            action_adapters: dict[str, Adapter] = {"api_action": SingleActionAdapter(action)}
            agent.adapters = action_adapters
            agent.failover = FailoverPolicy(chain=["api_action"], max_attempts=1)
            agent.default_provider = "api_action"
            agent.default_model = "explicit-action"
        return agent

    async def _execute(self, row: dict[str, Any]) -> None:
        run_id = row["id"]
        try:
            current = await self.store.run(row["owner"], run_id)
            if current["cancellation_requested"] or current["state"] != "running":
                await self.store.finish(run_id, "cancelled")
                return
            await self.store.event(run_id, {"type": "run_status", "state": "running"})
            attempts = await self.store.rows(
                "SELECT * FROM api_delegation_attempts WHERE run_id=?", (run_id,)
            )
            timeout = attempts[0]["timeout_seconds"] if attempts else None
            async with asyncio.timeout(timeout):
                agent = await self._agent(row)
                if row["kind"] == "tool":
                    request = RunRequest(
                        prompt="Execute the explicitly requested action through Harness policy.",
                        session_id=row["session_id"],
                        provider="api_action",
                        model="explicit-action",
                        max_steps=2,
                    )
                else:
                    values = json.loads(row["request"])
                    values["session_id"] = row["session_id"]
                    if (
                        not values.get("attachments")
                        and "attachments" not in RunRequest.model_fields
                    ):
                        values.pop("attachments", None)
                    request = RunRequest.model_validate(values)
                if attempts:
                    from harness.server.delegation import BudgetedAdapter, ExecutionAllocation

                    request.max_tokens = attempts[0]["max_tokens"]
                    request.max_steps = attempts[0]["max_steps"]
                    allocation = ExecutionAllocation(
                        request.max_steps, int(attempts[0]["max_tokens"])
                    )
                    adapters: dict[str, Adapter] = {
                        name: BudgetedAdapter(adapter, allocation)
                        for name, adapter in agent.adapters.items()
                    }
                    agent.adapters = adapters
                failure = None
                async for event in agent.run(request):
                    if isinstance(event, ModelRequestEvent):
                        payload = {"type": "model_request", "message_count": len(event.messages)}
                    else:
                        payload = public(event.model_dump(mode="json"))
                    await self.store.event(run_id, payload)
                    if isinstance(event, ErrorEvent):
                        failure = public(event.error)
                    if (
                        row["kind"] == "tool"
                        and isinstance(event, ToolResultEvent)
                        and event.result.is_error
                    ):
                        failure = public(event.result.content)
            session = await self.storage.get(row["session_id"])
            if session is not None and session.status == "paused":
                state, failure = "paused", None
            elif failure or session is None or session.status != "done":
                state = "failed"
                failure = failure or "Agent ended without a completed session"
            else:
                state = "completed"
            await self.store.finish(run_id, state, failure)
        except asyncio.CancelledError:
            state = "interrupted" if self._closing else "cancelled"
            if not self._closing and self.questions.store is not None:
                workspace = await self.session_workspace(row["session_id"])
                await asyncio.to_thread(
                    self.questions.store.cancel_session,
                    scope=self.scope(row["owner"], workspace=workspace),
                    session_id=row["session_id"],
                )
            await self.store.finish(
                run_id,
                state,
                "Execution interrupted; inspect effects before resuming" if self._closing else None,
            )
            if not self._closing:
                for child in await self.delegation.list(row["owner"], row["session_id"]):
                    await self.delegation.cancel(row["owner"], child["id"], row["session_id"])
            raise
        except TimeoutError:
            await self.store.finish(
                run_id, "failed", "Delegated run exceeded its elapsed-time budget"
            )
        except Exception as exc:
            # Avoid credential-bearing HTTP exception reprs. Detailed error events are redacted above.
            detail = (
                exc.detail
                if isinstance(exc, ServiceError)
                else f"Execution failed ({type(exc).__name__})"
            )
            await self.store.finish(run_id, "failed", public(detail))
        finally:
            self._wake.set()

    async def events(self, owner: str, run_id: str, *, after: int = 0) -> AsyncIterator[str]:
        await self.store.run(owner, run_id)
        while True:
            rows = await self.store.rows(
                "SELECT seq,payload FROM api_events WHERE run_id=? AND seq>? ORDER BY seq LIMIT 100",
                (run_id, after),
            )
            for row in rows:
                after = row["seq"]
                yield f"id: {after}\nevent: harness\ndata: {row['payload']}\n\n"
            if not rows and (await self.store.run(owner, run_id))["state"] in TERMINAL_STATES:
                return
            if not rows:
                yield ": heartbeat\n\n"
                await asyncio.sleep(0.05)

    async def sessions(
        self, owner: str, *, query: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        if not 1 <= limit <= 50 or offset < 0:
            raise ServiceError(422, "limit must be 1..50 and offset nonnegative")
        if query:
            owned = await self.store.rows(
                "SELECT s.id,w.workspace FROM api_sessions s LEFT JOIN api_session_workspaces w ON w.session_id=s.id WHERE s.owner=?",
                (owner,),
            )
            allowed = {row["id"] for row in owned}
            workspaces = {row["workspace"] or str(self.workspace) for row in owned}
            found = []
            try:
                for workspace in workspaces:
                    found.extend(
                        await self.storage.search_sessions(
                            query, scope=self.scope(owner, workspace=Path(workspace)), limit=limit
                        )
                    )
            except ValueError as exc:
                raise ServiceError(422, str(exc)) from None
            found.sort(key=lambda item: item.updated_at, reverse=True)
            return [
                public(item.model_dump(mode="json")) for item in found if item.session_id in allowed
            ][:limit]
        rows = await self.store.rows(
            "SELECT * FROM api_sessions WHERE owner=? ORDER BY created_at DESC,id LIMIT ? OFFSET ?",
            (owner, limit, offset),
        )
        results = []
        for row in rows:
            session = await self._owned_session(owner, row["id"])
            results.append(
                {
                    **row,
                    "status": session.status if session else "pending",
                    "model": session.model if session else None,
                }
            )
        return results

    async def messages(self, owner: str, session_id: str) -> list[dict[str, Any]]:
        session = await self._owned_session(owner, session_id)
        return (
            [public(item.model_dump(mode="json")) for item in session.messages] if session else []
        )

    async def approvals(self, owner: str) -> list[dict[str, Any]]:
        rows = await self.store.rows("SELECT id FROM api_sessions WHERE owner=?", (owner,))
        results = []
        for row in rows:
            await self._owned_session(owner, row["id"])
            results.extend(
                public(item.model_dump(mode="json"))
                for item in await self.storage.list_approvals(
                    session_id=row["id"], status="pending", limit=100
                )
            )
        return sorted(results, key=lambda item: item["requested_at"], reverse=True)

    async def resolve_approval(
        self, owner: str, approval_id: str, *, granted: bool
    ) -> dict[str, Any]:
        approval = await self.storage.get_approval(approval_id)
        if approval is None:
            raise ServiceError(404, "Approval not found")
        await self._owned_session(owner, approval.session_id)
        result = await self.storage.resolve_approval(
            approval_id, status="granted" if granted else "denied", resolved_by=owner
        )
        if result is None:
            raise ServiceError(409, "Approval is already resolved")
        return public(result.model_dump(mode="json"))

    async def export(self, owner: str, session_id: str) -> str:
        session = await self._owned_session(owner, session_id)
        if session is None:
            raise ServiceError(409, "Session has not started yet")
        rows = await self.store.rows(
            "SELECT id FROM api_runs WHERE session_id=? AND owner=? ORDER BY created_at,id",
            (session_id, owner),
        )
        runs = [await self.store.run(owner, row["id"]) for row in rows]
        events = await self.store.rows(
            "SELECT e.seq,e.run_id,e.payload,e.created_at FROM api_events e JOIN api_runs r ON r.id=e.run_id WHERE r.session_id=? AND r.owner=? ORDER BY e.seq",
            (session_id, owner),
        )
        trajectory = {
            "format": "harness.trajectory",
            "version": 1,
            "session": session.model_dump(mode="json", exclude={"messages"}),
            "messages": [message.model_dump(mode="json") for message in session.messages],
            "runs": runs,
            "events": [{**event, "payload": json.loads(event["payload"])} for event in events],
        }
        return json.dumps(public(trajectory), ensure_ascii=False) + "\n"
