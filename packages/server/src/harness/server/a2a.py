"""Official A2A SDK transport backed by Harness's durable, owned run queue."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from collections.abc import AsyncGenerator
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from a2a.auth.user import User
from a2a.server.context import ServerCallContext
from a2a.server.events.event_queue import Event
from a2a.server.request_handlers.request_handler import RequestHandler
from a2a.server.routes import (
    ServerCallContextBuilder,
    create_agent_card_routes,
    create_jsonrpc_routes,
)
from a2a.types import a2a_pb2 as proto
from a2a.utils.errors import (
    ContentTypeNotSupportedError,
    ExtendedAgentCardNotConfiguredError,
    InvalidParamsError,
    TaskNotCancelableError,
    TaskNotFoundError,
)
from google.protobuf.json_format import MessageToDict
from google.protobuf.timestamp_pb2 import Timestamp
from pydantic import ValidationError
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from harness.core.schemas import MediaAttachment
from harness.server.a2a_callbacks import callbacks
from harness.server.a2a_questions import (
    find_receipt,
    pending_question,
    question_detail,
    record_answer,
)
from harness.server.a2a_store import A2ABinding
from harness.server.models import TERMINAL_STATES, RunSubmission, ServiceError
from harness.server.service import HarnessService, identifier, public

STATES = {
    "queued": proto.TASK_STATE_SUBMITTED,
    "running": proto.TASK_STATE_WORKING,
    "completed": proto.TASK_STATE_COMPLETED,
    "failed": proto.TASK_STATE_FAILED,
    "interrupted": proto.TASK_STATE_FAILED,
    "cancelled": proto.TASK_STATE_CANCELED,
    "paused": proto.TASK_STATE_INPUT_REQUIRED,
}


class Caller(User):
    def __init__(self, owner: str):
        self.owner = owner

    @property
    def is_authenticated(self) -> bool:
        return bool(self.owner)

    @property
    def user_name(self) -> str:
        return self.owner


class CallerContext(ServerCallContextBuilder):
    def build(self, request: Request) -> ServerCallContext:
        owner = getattr(request.state, "owner", None)
        if not isinstance(owner, str) or not owner:
            raise InvalidParamsError("Authenticated Harness caller identity is required")
        # Do not copy the HTTP headers (and their bearer token) into SDK state.
        return ServerCallContext(
            user=Caller(owner),
            state={"headers": {"a2a-version": request.headers.get("a2a-version", "0.3")}},
        )


def caller(context: ServerCallContext) -> str:
    if not context.user.is_authenticated or not context.user.user_name:
        raise InvalidParamsError("Authenticated Harness caller identity is required")
    if context.tenant:
        raise InvalidParamsError(
            "This endpoint selects caller scope through bearer authentication, not tenant parameters"
        )
    return context.user.user_name


def callback_protocol(context: ServerCallContext) -> str:
    version = context.state.get("headers", {}).get("a2a-version", "0.3")
    if version not in {"1.0", "0.3"}:
        raise InvalidParamsError("Push notifications require A2A-Version 1.0 or 0.3")
    return version


def history_size(value: int | None) -> int:
    if value is None:
        return 0
    if not 0 <= value <= 100:
        raise InvalidParamsError("historyLength must be between 0 and 100")
    return value


def parts(message: dict[str, Any]) -> list[proto.Part]:
    result = [proto.Part(text=message["content"])] if message.get("content") else []
    for value in message.get("attachments", []):
        media = MediaAttachment.model_validate(value)
        if media.url:
            result.append(
                proto.Part(url=media.url, media_type=media.mime_type, filename=media.name or "")
            )
        else:
            result.append(
                proto.Part(
                    raw=base64.b64decode(media.data or ""),
                    media_type=media.mime_type,
                    filename=media.name or "",
                )
            )
    return result


class HarnessA2AHandler(RequestHandler):
    def __init__(self, service: HarnessService):
        self.service = service

    async def mapping(self, owner: str, task_id: str) -> dict[str, Any]:
        rows = await self.service.store.rows(
            "SELECT * FROM api_a2a_tasks WHERE id=? AND owner=?", (task_id, owner)
        )
        if not rows:
            raise TaskNotFoundError()
        return rows[0]

    async def task(
        self, owner: str, task_id: str, *, history: int = 0, artifacts: bool = True
    ) -> proto.Task:
        mapping = await self.mapping(owner, task_id)
        run = await self.service.store.run(owner, mapping["run_id"])
        stamp = Timestamp()
        stamp.FromDatetime(datetime.fromisoformat(run["updated_at"]))
        status = proto.TaskStatus(state=STATES[run["state"]], timestamp=stamp)
        detail = run["error"]
        question = None
        if run["state"] == "paused":
            question = await pending_question(self.service, owner, run["session_id"])
            if question is not None:
                detail = question_detail(question)
            elif await self.service.storage.list_approvals(
                session_id=run["session_id"], status="pending", limit=1
            ):
                detail = "Human approval is required. Resolve it through Harness /v1/approvals, then send a new message with this taskId to continue."
            else:
                detail = "This task is paused. Inspect the conversation and saved activity, then send a new message with this taskId when ready to continue."
        elif run["state"] == "interrupted":
            detail = "The server stopped during this run; inspect saved state and effects before starting a new task in the same context."
        if detail:
            status.message.CopyFrom(
                proto.Message(
                    message_id=f"{run['id']}-status",
                    role=proto.ROLE_AGENT,
                    task_id=task_id,
                    context_id=run["session_id"],
                    parts=[proto.Part(text=public(detail))],
                )
            )
        task = proto.Task(
            id=task_id,
            context_id=run["session_id"],
            status=status,
            metadata={"harness_run_id": run["id"], "harness_api_version": 1},
        )
        if question is not None:
            task.metadata["harness_question_id"] = question.id
        if history:
            messages = await self.service.messages(owner, run["session_id"])
            messages = [
                message
                for message in messages
                if message["role"] in {"user", "assistant"}
                and (message.get("content") or message.get("attachments"))
            ]
            for index, message in enumerate(messages[-history:]):
                task.history.append(
                    proto.Message(
                        message_id=f"{task_id}-history-{max(0, len(messages) - history) + index}",
                        context_id=run["session_id"],
                        role=proto.ROLE_USER if message["role"] == "user" else proto.ROLE_AGENT,
                        parts=parts(message),
                    )
                )
        if artifacts and run["state"] == "completed":
            rows = await self.service.store.rows(
                "SELECT payload FROM api_events WHERE run_id=? AND json_extract(payload,'$.type')='done' ORDER BY seq DESC LIMIT 1",
                (run["id"],),
            )
            for row in rows:
                event = json.loads(row["payload"])
                if event.get("type") == "done" and event.get("final_message"):
                    output = parts(event["final_message"])
                    if output:
                        task.artifacts.append(
                            proto.Artifact(
                                artifact_id=f"{run['id']}-answer", name="answer", parts=output
                            )
                        )
                    break
        return task

    async def submit(
        self, params: proto.SendMessageRequest, context: ServerCallContext
    ) -> tuple[str, str]:
        owner = caller(context)
        message = params.message
        if (
            not params.HasField("message")
            or message.role != proto.ROLE_USER
            or not message.message_id
            or len(message.message_id) > 256
        ):
            raise InvalidParamsError(
                "A user message with a nonempty messageId (at most 256 characters) is required"
            )
        if message.extensions or message.reference_task_ids:
            raise InvalidParamsError(
                "Extensions and referenceTaskIds are not supported by this endpoint"
            )
        configuration = params.configuration
        history_size(
            configuration.history_length if configuration.HasField("history_length") else None
        )
        notification = (
            configuration.task_push_notification_config
            if configuration.HasField("task_push_notification_config")
            else None
        )
        if notification is not None:
            callback_protocol(context)
            callbacks(self.service)
        if (
            configuration.accepted_output_modes
            and "text/plain" not in configuration.accepted_output_modes
        ):
            raise ContentTypeNotSupportedError(
                "This endpoint requires acceptance of text/plain output"
            )
        # Transport preferences do not change message identity. Caller identity
        # always comes from authentication, never message metadata or contextId.
        digest = hashlib.sha256(
            message.SerializeToString(deterministic=True)
            + (
                notification.SerializeToString(deterministic=True)
                if notification is not None
                else b""
            )
            + (b"0.3" if notification is not None and callback_protocol(context) == "0.3" else b"")
        ).hexdigest()
        existing = await self.service.store.rows(
            "SELECT * FROM api_a2a_messages WHERE owner=? AND message_id=?",
            (owner, message.message_id),
        )
        if existing:
            if existing[0]["digest"] != digest:
                raise InvalidParamsError("messageId was already used for different content")
            return existing[0]["task_id"], existing[0]["run_id"]
        receipt = await find_receipt(self.service, owner, message.message_id, digest)
        if receipt is not None:
            mapping = await self.mapping(owner, receipt["task_id"])
            if not receipt["ready"] or mapping["run_id"] != receipt["run_id"]:
                return mapping["id"], mapping["run_id"]
        texts = []
        attachments = []
        try:
            if not message.parts or len(message.parts) > 32:
                raise InvalidParamsError("A message must contain between 1 and 32 parts")
            for part in message.parts:
                content = part.WhichOneof("content")
                if content == "text":
                    texts.append(part.text)
                elif content in {"raw", "url"}:
                    mime = part.media_type
                    kind = (
                        "image"
                        if mime.startswith("image/")
                        else "audio"
                        if mime.startswith("audio/")
                        else "file"
                    )
                    attachments.append(
                        MediaAttachment(
                            kind=kind,
                            mime_type=mime,
                            name=part.filename or None,
                            data=base64.b64encode(part.raw).decode() if content == "raw" else None,
                            url=part.url if content == "url" else None,
                        )
                    )
                else:
                    raise ContentTypeNotSupportedError(
                        "Use text or typed file parts; structured data parts are unsupported"
                    )
            task_id = message.task_id or identifier("task")
            session_id = message.context_id or None
            expected = None
            if message.task_id:
                mapping = await self.mapping(owner, task_id)
                if session_id and session_id != mapping["session_id"]:
                    raise InvalidParamsError("taskId and contextId do not match")
                session_id = mapping["session_id"]
                run = await self.service.store.run(owner, mapping["run_id"])
                if run["state"] != "paused":
                    retried = await self.service.store.rows(
                        "SELECT * FROM api_a2a_messages WHERE owner=? AND message_id=? AND digest=?",
                        (owner, message.message_id, digest),
                    )
                    if retried:
                        return retried[0]["task_id"], retried[0]["run_id"]
                    raise InvalidParamsError(
                        "Only input-required tasks accept continuation; use a new message without taskId for a new task in the same context"
                    )
                expected = run["id"]
                question = await pending_question(self.service, owner, session_id)
                if question is not None:
                    if attachments:
                        raise InvalidParamsError("Clarification answers must use text parts")
                    if notification is not None:
                        raise InvalidParamsError(
                            "Configure callbacks with callback CRUD before answering clarification questions"
                        )
                    receipt = await record_answer(
                        self.service,
                        owner,
                        task_id=task_id,
                        run_id=expected,
                        record=question,
                        message_id=message.message_id,
                        digest=digest,
                        text="\n".join(texts),
                    )
                    if not receipt["ready"]:
                        return receipt["task_id"], receipt["run_id"]
            submission = await self.service.resolve_submission(
                owner,
                RunSubmission(
                    prompt="\n".join(texts), attachments=attachments, session_id=session_id
                ),
            )
            serialized_notification = None
            if notification is not None:
                prepared = callbacks(self.service).prepare(owner, notification, task_id)
                serialized_notification = json.dumps(MessageToDict(prepared), sort_keys=True)
            binding = A2ABinding(
                task_id,
                message.message_id,
                digest,
                expected,
                serialized_notification,
                callback_protocol(context) if notification is not None else "1.0",
            )
            runs = await self.service._enqueue(
                owner, [("agent", submission.model_dump(), expected)], batch=False, a2a=binding
            )
            # Concurrent duplicate submissions may have created a different
            # local task ID before the transaction found the existing message.
            record = (
                await self.service.store.rows(
                    "SELECT task_id FROM api_a2a_messages WHERE owner=? AND message_id=?",
                    (owner, message.message_id),
                )
            )[0]
            return record["task_id"], runs[0]["id"]
        except ServiceError as exc:
            if exc.status == 404:
                raise TaskNotFoundError() from None
            raise InvalidParamsError(exc.detail) from None
        except (ValidationError, ValueError):
            raise InvalidParamsError("Invalid message content or attachments") from None

    async def on_message_send(
        self, params: proto.SendMessageRequest, context: ServerCallContext
    ) -> proto.Task:
        task_id, run_id = await self.submit(params, context)
        owner = caller(context)
        if not params.configuration.return_immediately:
            async for _ in self.service.events(owner, run_id):
                pass
        return await self.task(
            owner,
            task_id,
            history=history_size(
                params.configuration.history_length
                if params.configuration.HasField("history_length")
                else None
            ),
        )

    async def stream(self, owner: str, task_id: str) -> AsyncGenerator[Event]:
        # Observe durable state. Disconnecting only stops observation; it never
        # cancels or resubmits executable work.
        last = None
        while True:
            task = await self.task(owner, task_id)
            signature = task.SerializeToString(deterministic=True)
            if signature != last:
                yield task
                last = signature
            if task.status.state not in {proto.TASK_STATE_SUBMITTED, proto.TASK_STATE_WORKING}:
                return
            await asyncio.sleep(0.05)

    async def on_message_send_stream(
        self, params: proto.SendMessageRequest, context: ServerCallContext
    ) -> AsyncGenerator[Event]:
        task_id, _ = await self.submit(params, context)
        async for event in self.stream(caller(context), task_id):
            yield event

    async def on_subscribe_to_task(
        self, params: proto.SubscribeToTaskRequest, context: ServerCallContext
    ) -> AsyncGenerator[Event]:
        async for event in self.stream(caller(context), params.id):
            yield event

    async def on_get_task(
        self, params: proto.GetTaskRequest, context: ServerCallContext
    ) -> proto.Task:
        return await self.task(
            caller(context),
            params.id,
            history=history_size(
                params.history_length if params.HasField("history_length") else None
            ),
        )

    async def on_cancel_task(
        self, params: proto.CancelTaskRequest, context: ServerCallContext
    ) -> proto.Task:
        owner = caller(context)
        mapping = await self.mapping(owner, params.id)
        run = await self.service.store.run(owner, mapping["run_id"])
        if run["state"] in TERMINAL_STATES - {"paused", "cancelled"}:
            raise TaskNotCancelableError()
        try:
            await self.service.cancel(owner, run["id"])
        except ServiceError as exc:
            raise TaskNotCancelableError(exc.detail) from None
        result = await self.task(owner, params.id)
        if result.status.state != proto.TASK_STATE_CANCELED:
            raise TaskNotCancelableError("Run finished before cancellation; fetch its actual state")
        return result

    async def on_list_tasks(
        self, params: proto.ListTasksRequest, context: ServerCallContext
    ) -> proto.ListTasksResponse:
        owner = caller(context)
        limit = params.page_size if params.HasField("page_size") else 50
        history = history_size(params.history_length if params.HasField("history_length") else None)
        if not 1 <= limit <= 100:
            raise InvalidParamsError("pageSize must be between 1 and 100")
        try:
            offset = (
                int(base64.urlsafe_b64decode(params.page_token).decode())
                if params.page_token
                else 0
            )
            if offset < 0 or offset > 10_000_000 or len(params.page_token) > 64:
                raise ValueError
        except (ValueError, UnicodeDecodeError):
            raise InvalidParamsError("Invalid pageToken") from None
        clauses, values = ["t.owner=?"], [owner]
        if params.context_id:
            clauses.append("t.session_id=?")
            values.append(params.context_id)
        if params.status:
            states = [state for state, value in STATES.items() if value == params.status]
            if not states:
                raise InvalidParamsError("Unsupported status filter")
            clauses.append("r.state IN (" + ",".join("?" for _ in states) + ")")
            values.extend(states)
        if params.HasField("status_timestamp_after"):
            clauses.append("r.updated_at>?")
            values.append(
                params.status_timestamp_after.ToDatetime(tzinfo=UTC).isoformat(
                    timespec="microseconds"
                )
            )
        base = "FROM api_a2a_tasks t JOIN api_runs r ON r.id=t.run_id WHERE " + " AND ".join(
            clauses
        )
        total = (await self.service.store.rows("SELECT COUNT(*) AS n " + base, tuple(values)))[0][
            "n"
        ]
        rows = await self.service.store.rows(
            "SELECT t.id " + base + " ORDER BY r.updated_at DESC,t.id LIMIT ? OFFSET ?",
            (*values, limit, offset),
        )
        tasks = [
            await self.task(
                owner,
                row["id"],
                history=history,
                artifacts=params.include_artifacts
                if params.HasField("include_artifacts")
                else False,
            )
            for row in rows
        ]
        next_page = (
            base64.urlsafe_b64encode(str(offset + len(tasks)).encode()).decode()
            if offset + len(tasks) < total
            else ""
        )
        return proto.ListTasksResponse(
            tasks=tasks, next_page_token=next_page, page_size=limit, total_size=total
        )

    async def on_create_task_push_notification_config(
        self, params: proto.TaskPushNotificationConfig, context: ServerCallContext
    ) -> proto.TaskPushNotificationConfig:
        return await callbacks(self.service).create(
            caller(context), params, protocol=callback_protocol(context)
        )

    async def on_get_task_push_notification_config(
        self, params: proto.GetTaskPushNotificationConfigRequest, context: ServerCallContext
    ) -> proto.TaskPushNotificationConfig:
        return await callbacks(self.service).get(
            caller(context), params.task_id, params.id, protocol=callback_protocol(context)
        )

    async def on_list_task_push_notification_configs(
        self, params: proto.ListTaskPushNotificationConfigsRequest, context: ServerCallContext
    ) -> proto.ListTaskPushNotificationConfigsResponse:
        values = await callbacks(self.service).list(
            caller(context), params.task_id, protocol=callback_protocol(context)
        )
        size = params.page_size or (100 if callback_protocol(context) == "0.3" else 50)
        try:
            offset = (
                int(base64.urlsafe_b64decode(params.page_token).decode())
                if params.page_token
                else 0
            )
            if not 1 <= size <= 100 or not 0 <= offset <= 100 or len(params.page_token) > 64:
                raise ValueError
        except (ValueError, UnicodeDecodeError):
            raise InvalidParamsError("Invalid callback pagination") from None
        next_page = (
            base64.urlsafe_b64encode(str(offset + size).encode()).decode()
            if offset + size < len(values)
            else ""
        )
        return proto.ListTaskPushNotificationConfigsResponse(
            configs=values[offset : offset + size], next_page_token=next_page
        )

    async def on_delete_task_push_notification_config(
        self, params: proto.DeleteTaskPushNotificationConfigRequest, context: ServerCallContext
    ) -> None:
        await callbacks(self.service).delete(
            caller(context), params.task_id, params.id, protocol=callback_protocol(context)
        )

    async def on_get_extended_agent_card(
        self, params: proto.GetExtendedAgentCardRequest, context: ServerCallContext
    ) -> proto.AgentCard:
        raise ExtendedAgentCardNotConfiguredError()


def make_a2a_routes(service: HarnessService, endpoint_url: str) -> list[Route]:
    parsed = urlsplit(endpoint_url)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path != "/a2a"
    ):
        raise ValueError(
            "A2A endpoint URL must be an explicit HTTP(S) URL ending in /a2a without credentials or query"
        )
    card = proto.AgentCard(
        name="Harness",
        version="1",
        description="Durable agent tasks with authenticated session ownership and Harness tool approvals.",
        supported_interfaces=[
            proto.AgentInterface(
                url=endpoint_url, protocol_binding="JSONRPC", protocol_version="1.0"
            )
        ],
        capabilities=proto.AgentCapabilities(
            streaming=True,
            push_notifications=service.a2a_callbacks is not None,
            extended_agent_card=False,
        ),
        default_input_modes=["text/plain", "image/*", "audio/*", "application/octet-stream"],
        default_output_modes=["text/plain"],
        skills=[
            proto.AgentSkill(
                id="agent_run",
                name="Agent run",
                description="Submit, observe and cancel a task. Exposed tools remain subject to Harness approval and evidence gates.",
                tags=["agent", "tasks"],
            )
        ],
        security_schemes={
            "bearer": proto.SecurityScheme(
                http_auth_security_scheme=proto.HTTPAuthSecurityScheme(scheme="bearer")
            )
        },
        security_requirements=[proto.SecurityRequirement(schemes={"bearer": proto.StringList()})],
    )
    rpc = create_jsonrpc_routes(
        HarnessA2AHandler(service), "/a2a", context_builder=CallerContext(), enable_v0_3_compat=True
    )[0]
    endpoint = rpc.endpoint

    async def bounded(request: Request):
        if (
            request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
            != "application/json"
        ):
            return JSONResponse({"detail": "Use application/json"}, status_code=415)
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 24 * 1024 * 1024:
                return JSONResponse({"detail": "Request body exceeds 24 MiB"}, status_code=413)
        # Cache the bounded body on this request; SDK parsing stays authoritative.
        request._body = bytes(raw)
        return await endpoint(request)

    return [Route("/a2a", bounded, methods=["POST"]), *create_agent_card_routes(card)]
