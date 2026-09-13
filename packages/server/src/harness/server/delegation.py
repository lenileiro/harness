"""Durable child agents with trusted parent identity and reserved execution budgets."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import stat
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from harness.core.adapter import Adapter
from harness.core.errors import ConfigurationError
from harness.core.events import Event
from harness.core.schemas import (
    ApprovalDecision,
    Capabilities,
    MediaAttachment,
    Message,
    Session,
    ToolCall,
    ToolResult,
)
from harness.core.tools import Tool
from harness.server.models import TERMINAL_STATES, AgentBuilder, RunSubmission, ServiceError
from harness.server.service import identifier, public
from harness.server.store import now

if TYPE_CHECKING:
    from harness.server.service import HarnessService


class DelegationLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    max_children: int = Field(default=8, ge=1, le=100)
    max_depth: int = Field(default=2, ge=1, le=8)
    max_total_steps: int = Field(default=80, ge=1, le=10000)
    max_total_output_tokens: int = Field(default=81920, ge=1, le=10_000_000)
    max_total_seconds: float = Field(default=600, ge=1, le=86400, allow_inf_nan=False)


class DelegateSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    prompt: str = Field(min_length=1, max_length=100_000)
    inputs: list[str] = Field(default_factory=list, max_length=32)
    model: str | None = Field(default=None, max_length=256)
    max_steps: int = Field(default=10, ge=1, le=100)
    max_tokens: int = Field(default=1024, ge=1, le=16000)
    timeout_seconds: float = Field(default=120, ge=1, le=3600, allow_inf_nan=False)


class HandleArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")
    handle: str = Field(min_length=1, max_length=128)


class ArtifactArguments(HandleArguments):
    path: str = Field(min_length=1, max_length=1024)


def read_job_file(root: Path, relative: str, *, max_bytes: int = 4 * 1024 * 1024) -> bytes:
    """Pin every path component; neither symlink swaps nor special files escape the job."""
    path = Path(relative)
    if path.is_absolute() or not path.parts or any(part.startswith(".") for part in path.parts):
        raise ServiceError(
            422, "Input/artifact paths must be relative public files without dot components"
        )
    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        raise ServiceError(
            422,
            "Safe delegation file copying is unavailable on this platform; pass task context in the prompt",
        )
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in path.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        descriptor = os.open(
            path.parts[-1], os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=directory
        )
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
                raise ServiceError(422, "Delegation files must be regular and at most 4 MiB")
            result = source.read(max_bytes + 1)
            if len(result) > max_bytes:
                raise ServiceError(422, "Delegation file exceeds the size limit")
            return result
    except OSError:
        raise ServiceError(
            422, "Delegation file is missing, a symlink, or outside the workspace"
        ) from None
    finally:
        os.close(directory)


def copy_inputs(source: Path, destination: Path, paths: list[str]) -> None:
    destination.mkdir(parents=True, mode=0o700)
    size = 0
    for relative in paths:
        data = read_job_file(source, relative)
        size += len(data)
        if size > 8 * 1024 * 1024:
            raise ServiceError(422, "Combined delegation inputs exceed 8 MiB")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with target.open("xb") as output:
            os.chmod(target, 0o600)
            output.write(data)


class ExecutionAllocation:
    def __init__(self, steps: int, max_tokens: int):
        self.remaining = steps
        self.max_tokens = max_tokens

    def claim(self) -> None:
        if self.remaining <= 0:
            raise ConfigurationError(
                "Delegated model-call budget exhausted; reserve a new attempt before resuming"
            )
        self.remaining -= 1


class BudgetedAdapter(Adapter):
    def __init__(self, adapter: Adapter, allocation: ExecutionAllocation):
        self.adapter = adapter
        self.allocation = allocation
        self.name = adapter.name

    async def capabilities(self) -> Capabilities:
        return await self.adapter.capabilities()

    async def cancel(self, session_id: str) -> None:
        await self.adapter.cancel(session_id)

    async def stream(
        self, *, model: str, messages: list[Message], **kwargs: Any
    ) -> AsyncIterator[Event]:
        self.allocation.claim()
        kwargs["max_tokens"] = min(
            kwargs.get("max_tokens") or self.allocation.max_tokens, self.allocation.max_tokens
        )
        async for event in self.adapter.stream(model=model, messages=messages, **kwargs):
            yield event


class DelegationManager:
    def __init__(self, service: HarnessService, limits: DelegationLimits | None = None):
        self.service = service
        self.limits = limits or DelegationLimits()

    async def bind(self, owner: str, session: Session) -> Sequence[Tool]:
        await self.register_parent(owner, session.id, session.cwd)
        return tuple(
            DelegationTool(self, owner, session.id, name)
            for name in (
                "delegate",
                "delegate_status",
                "delegate_cancel",
                "delegate_resume",
                "delegate_artifact",
            )
        )

    async def register_parent(self, owner: str, session_id: str, workspace: Path) -> None:
        self.service.scope(owner)
        workspace = await asyncio.to_thread(workspace.resolve)
        async with self.service.store.connection() as db:
            await db.execute("BEGIN IMMEDIATE")
            async with db.execute(
                "SELECT * FROM api_delegation_parents WHERE session_id=?", (session_id,)
            ) as cursor:
                row = await cursor.fetchone()
            if row is not None:
                if row["owner"] != owner or Path(row["workspace"]) != workspace:
                    raise ServiceError(403, "Delegation parent identity does not match")
            else:
                await db.execute(
                    "INSERT INTO api_delegation_parents VALUES (?,?,?,?,?)",
                    (session_id, owner, str(workspace), session_id, 0),
                )
                await db.execute(
                    "INSERT INTO api_delegation_budgets(root_id,limits) VALUES (?,?)",
                    (session_id, self.limits.model_dump_json()),
                )
            await db.commit()

    async def _parent(self, owner: str, session_id: str) -> dict[str, Any]:
        rows = await self.service.store.rows(
            "SELECT * FROM api_delegation_parents WHERE session_id=? AND owner=?",
            (session_id, owner),
        )
        if not rows:
            raise ServiceError(404, "Delegation parent not found")
        return rows[0]

    async def _job(
        self, owner: str, handle: str, parent_session_id: str | None = None
    ) -> dict[str, Any]:
        rows = await self.service.store.rows(
            "SELECT * FROM api_delegations WHERE id=? AND owner=?", (handle, owner)
        )
        if not rows or (
            parent_session_id is not None and rows[0]["parent_session_id"] != parent_session_id
        ):
            raise ServiceError(404, "Delegated job not found")
        return rows[0]

    async def _reserve(
        self, db: Any, parent: dict[str, Any], request: DelegateSubmission, *, new_child: bool
    ) -> None:
        async with db.execute(
            "SELECT * FROM api_delegation_budgets WHERE root_id=?", (parent["root_id"],)
        ) as cursor:
            budget = await cursor.fetchone()
        if budget is None:
            raise ServiceError(409, "Delegation budget is missing")
        limits = DelegationLimits.model_validate_json(budget["limits"])
        if parent["depth"] >= limits.max_depth:
            raise ServiceError(409, "Delegation depth budget exhausted")
        tokens = request.max_steps * request.max_tokens
        if (
            budget["children"] + int(new_child) > limits.max_children
            or budget["steps"] + request.max_steps > limits.max_total_steps
            or budget["output_tokens"] + tokens > limits.max_total_output_tokens
            or budget["seconds"] + request.timeout_seconds > limits.max_total_seconds
        ):
            raise ServiceError(
                409,
                "Delegation budget exhausted; reduce requested work or start a new parent session",
            )
        await db.execute(
            "UPDATE api_delegation_budgets SET children=children+?,steps=steps+?,output_tokens=output_tokens+?,seconds=seconds+? WHERE root_id=?",
            (int(new_child), request.max_steps, tokens, request.timeout_seconds, parent["root_id"]),
        )

    async def submit(
        self, owner: str, parent_session_id: str, request: DelegateSubmission
    ) -> dict[str, Any]:
        parent = await self._parent(owner, parent_session_id)
        handle, session_id = identifier("job"), identifier("sess")
        workspace = self.service.database.parent / "jobs" / handle
        try:
            copying = asyncio.create_task(
                asyncio.to_thread(copy_inputs, Path(parent["workspace"]), workspace, request.inputs)
            )
            try:
                await asyncio.shield(copying)
            except asyncio.CancelledError:
                await copying
                raise
            async with self.service.store.connection() as db:
                await db.execute("BEGIN IMMEDIATE")
                await self._reserve(db, parent, request, new_child=True)
                stamp = now()
                await db.execute(
                    "INSERT INTO api_sessions VALUES (?,?,?)", (session_id, owner, stamp)
                )
                await db.execute(
                    "INSERT INTO api_session_workspaces VALUES (?,?)", (session_id, str(workspace))
                )
                await db.execute(
                    "INSERT INTO api_delegations VALUES (?,?,?,?,?,?,?,?)",
                    (
                        handle,
                        owner,
                        parent_session_id,
                        session_id,
                        parent["root_id"],
                        str(workspace),
                        request.model_dump_json(),
                        stamp,
                    ),
                )
                await db.execute(
                    "INSERT INTO api_delegation_parents VALUES (?,?,?,?,?)",
                    (session_id, owner, str(workspace), parent["root_id"], parent["depth"] + 1),
                )
                await self._attempt(db, owner, handle, session_id, request)
                await db.commit()
        except BaseException:
            # A cancelled SQLite commit may have completed in the worker thread.
            # Keep inputs whenever a durable handle exists; never delete a queued job's workspace.
            persisted = await self.service.store.rows(
                "SELECT id FROM api_delegations WHERE id=?", (handle,)
            )
            if not persisted:
                await asyncio.to_thread(shutil.rmtree, workspace, ignore_errors=True)
            else:
                self.service._wake.set()
            raise
        self.service._wake.set()
        return await self.status(owner, handle, parent_session_id)

    async def _attempt(
        self,
        db: Any,
        owner: str,
        handle: str,
        session_id: str,
        request: DelegateSubmission,
        resumed_from: str | None = None,
    ) -> str:
        run_id, stamp = identifier("run"), now()
        submission = RunSubmission(
            prompt=request.prompt,
            session_id=session_id,
            model=request.model,
            max_steps=request.max_steps,
        )
        self.service._validate_submission(submission)
        await db.execute(
            "INSERT INTO api_runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                owner,
                session_id,
                "queued",
                "agent",
                submission.model_dump_json(),
                None,
                resumed_from,
                stamp,
                stamp,
                None,
            ),
        )
        await db.execute(
            "INSERT INTO api_delegation_attempts VALUES (?,?,?,?,?)",
            (run_id, handle, request.max_steps, request.max_tokens, request.timeout_seconds),
        )
        await db.execute(
            "INSERT INTO api_events(run_id,payload,created_at) VALUES (?,?,?)",
            (
                run_id,
                json.dumps({"type": "run_status", "state": "queued", "delegation_handle": handle}),
                stamp,
            ),
        )
        return run_id

    async def status(
        self, owner: str, handle: str, parent_session_id: str | None = None
    ) -> dict[str, Any]:
        job = await self._job(owner, handle, parent_session_id)
        attempts = await self.service.store.rows(
            "SELECT r.id FROM api_runs r JOIN api_delegation_attempts a ON a.run_id=r.id WHERE a.delegation_id=? ORDER BY r.created_at DESC,r.id",
            (handle,),
        )
        run = await self.service.store.run(owner, attempts[0]["id"])
        session = await self.service._owned_session(owner, job["session_id"])
        summary = (
            next(
                (
                    message.content
                    for message in reversed(session.messages)
                    if message.role == "assistant" and message.content
                ),
                None,
            )
            if session
            else None
        )
        approvals = await self.service.storage.list_approvals(
            session_id=job["session_id"], status="pending", limit=100
        )
        budget = (
            await self.service.store.rows(
                "SELECT * FROM api_delegation_budgets WHERE root_id=?", (job["root_id"],)
            )
        )[0]
        budget["limits"] = json.loads(budget["limits"])
        artifacts = []
        for root, directories, filenames in os.walk(job["workspace"], followlinks=False):
            directories[:] = [
                name
                for name in directories
                if not name.startswith(".") and not (Path(root) / name).is_symlink()
            ]
            for name in filenames:
                path = Path(root) / name
                if not name.startswith(".") and not path.is_symlink() and path.is_file():
                    artifacts.append(path.relative_to(job["workspace"]).as_posix())
                    if len(artifacts) >= 100:
                        break
            if len(artifacts) >= 100:
                break
        return public(
            {
                "id": handle,
                "parent_session_id": job["parent_session_id"],
                "session_id": job["session_id"],
                "state": run["state"],
                "run": run,
                "attempts": len(attempts),
                "summary": summary,
                "approvals": [item.model_dump(mode="json") for item in approvals],
                "questions": await self.service.questions.for_session(owner, job["session_id"]),
                "artifacts": artifacts,
                "budget": budget,
            }
        )

    async def list(self, owner: str, parent_session_id: str | None = None) -> list[dict[str, Any]]:
        rows = await self.service.store.rows(
            "SELECT id FROM api_delegations WHERE owner=? AND (? IS NULL OR parent_session_id=?) ORDER BY created_at DESC LIMIT 100",
            (owner, parent_session_id, parent_session_id),
        )
        return [await self.status(owner, row["id"], parent_session_id) for row in rows]

    async def cancel(
        self, owner: str, handle: str, parent_session_id: str | None = None
    ) -> dict[str, Any]:
        current = await self.status(owner, handle, parent_session_id)
        await self.service.cancel(owner, current["run"]["id"])
        return await self.status(owner, handle, parent_session_id)

    async def resume(
        self,
        owner: str,
        handle: str,
        parent_session_id: str | None = None,
        *,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        job = await self._job(owner, handle, parent_session_id)
        status = await self.status(owner, handle, parent_session_id)
        if status["state"] not in TERMINAL_STATES:
            raise ServiceError(409, "Delegated job is already queued or running")
        if status["approvals"]:
            raise ServiceError(409, "Resolve child approvals before resuming")
        await self.service.questions.check_admission(
            await self.service._owned_session(owner, job["session_id"])
        )
        parent = await self._parent(owner, job["parent_session_id"])
        request = DelegateSubmission.model_validate_json(job["request"])
        request.prompt = (
            prompt
            or "Continue the delegated task from saved state; inspect uncertain effects and do not repeat completed actions."
        )
        async with self.service.store.connection() as db:
            await db.execute("BEGIN IMMEDIATE")
            async with db.execute(
                "SELECT id FROM api_runs WHERE session_id=? AND state IN ('queued','running')",
                (job["session_id"],),
            ) as cursor:
                if await cursor.fetchone():
                    raise ServiceError(409, "Delegated job is already active")
            await self._reserve(db, parent, request, new_child=False)
            await self._attempt(db, owner, handle, job["session_id"], request, status["run"]["id"])
            await db.commit()
        self.service._wake.set()
        return await self.status(owner, handle, parent_session_id)

    async def artifact(
        self, owner: str, handle: str, path: str, parent_session_id: str | None = None
    ) -> MediaAttachment:
        job = await self._job(owner, handle, parent_session_id)
        data = await asyncio.to_thread(read_job_file, Path(job["workspace"]), path)
        return MediaAttachment(
            kind="file",
            mime_type="application/octet-stream",
            data=base64.b64encode(data).decode("ascii"),
            name=Path(path).name,
            model_visible=False,
        )


class DelegationTool:
    approval: ApprovalDecision = "auto"
    phases = ("*",)

    def __init__(self, manager: DelegationManager, owner: str, parent_session_id: str, name: str):
        self.manager, self.owner, self.parent_session_id, self.name = (
            manager,
            owner,
            parent_session_id,
            name,
        )
        self.effect_scope = (
            "read_only"
            if name in ("delegate_status", "delegate_artifact")
            else "agent_orchestration"
        )
        self.description = {
            "delegate": "Queue an independent child agent in a private workspace. Pass context explicitly and selected public input files. Returns a durable handle; never merges child changes.",
            "delegate_status": "Read one direct child's status, summary, pending questions and approvals, artifacts, and reserved budget.",
            "delegate_cancel": "Cancel a direct child and all its descendants. Does not undo completed effects.",
            "delegate_resume": "Explicitly resume a finished or interrupted direct child using saved state and a new budget reservation. Resolve approvals first.",
            "delegate_artifact": "Return a direct child's selected artifact as an attachment for review. Does not merge it into the parent workspace.",
        }[name]
        self.arguments = (
            DelegateSubmission
            if name == "delegate"
            else ArtifactArguments
            if name == "delegate_artifact"
            else HandleArguments
        )
        self.parameters_schema = self.arguments.model_json_schema()

    async def __call__(self, call: ToolCall) -> ToolResult:
        try:
            arguments = self.arguments.model_validate(call.arguments)
            if isinstance(arguments, DelegateSubmission):
                result = await self.manager.submit(self.owner, self.parent_session_id, arguments)
            elif isinstance(arguments, ArtifactArguments):
                attachment = await self.manager.artifact(
                    self.owner, arguments.handle, arguments.path, self.parent_session_id
                )
                return ToolResult(
                    tool_call_id=call.id,
                    name=self.name,
                    content=f"Artifact {arguments.path} from {arguments.handle}",
                    attachments=[attachment],
                )
            else:
                operation = {
                    "delegate_status": self.manager.status,
                    "delegate_cancel": self.manager.cancel,
                    "delegate_resume": self.manager.resume,
                }[self.name]
                result = await operation(self.owner, arguments.handle, self.parent_session_id)
            return ToolResult(tool_call_id=call.id, name=self.name, content=json.dumps(result))
        except (ServiceError, ValidationError) as exc:
            detail = exc.detail if isinstance(exc, ServiceError) else "Invalid delegation arguments"
            return ToolResult(tool_call_id=call.id, name=self.name, content=detail, is_error=True)


class LocalDelegationToolset:
    """Own a durable child queue for a standalone Agent's async session-tool factory."""

    def __init__(
        self,
        database: Path,
        workspace: Path,
        agent_builder: AgentBuilder,
        *,
        owner: str = "local",
        limits: DelegationLimits | None = None,
        exposed_tools: Sequence[str] = (),
        max_workers: int = 2,
        wait_on_close: bool = True,
    ):
        from harness.server.service import HarnessService

        self.owner = owner
        self.wait_on_close = wait_on_close
        self.service = HarnessService(
            database, workspace, agent_builder, exposed_tools=exposed_tools, max_workers=max_workers
        )
        self.manager = DelegationManager(self.service, limits)
        self.service.delegation = self.manager

    async def __aenter__(self) -> LocalDelegationToolset:
        await self.service.start()
        return self

    async def bind(self, session: Session) -> Sequence[Tool]:
        return await self.manager.bind(self.owner, session)

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        try:
            if exc_type is None and self.wait_on_close:
                while True:
                    rows = await self.service.store.rows(
                        "SELECT id,owner FROM api_runs WHERE state IN ('queued','running') LIMIT 1"
                    )
                    if not rows:
                        break
                    async for _ in self.service.events(rows[0]["owner"], rows[0]["id"]):
                        pass
        finally:
            await self.service.close()
