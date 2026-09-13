"""Authenticated ASGI HTTP API; mounted MCP calls the same service methods."""

from __future__ import annotations

import base64
import json
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response, StreamingResponse
from starlette.routing import BaseRoute, Mount, Route

from harness.server.a2a_callbacks import CallbackGrant, CallbackManager, callbacks
from harness.server.auth import AuthMiddleware, ServerAuth
from harness.server.automations import ScheduleSubmission
from harness.server.delegation import DelegateSubmission
from harness.server.models import RunSubmission, ServiceError, ToolSubmission, UserPreferences
from harness.server.questions import QuestionAnswers
from harness.server.service import HarnessService


class BatchSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    runs: list[RunSubmission] = Field(min_length=1, max_length=100)


class ChildSubmission(DelegateSubmission):
    parent_session_id: str = Field(min_length=1, max_length=128)


class ResumeSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str | None = Field(default=None, max_length=100_000)


class ApprovalSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    granted: bool


async def body(request: Request) -> Any:
    if (
        request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        != "application/json"
    ):
        raise ServiceError(415, "Use application/json for request bodies")
    chunks = bytearray()
    async for chunk in request.stream():
        chunks.extend(chunk)
        if len(chunks) > 24 * 1024 * 1024:
            raise ServiceError(413, "Request body exceeds 24 MiB")
    try:
        return json.loads(chunks)
    except (ValueError, UnicodeDecodeError):
        raise ServiceError(422, "Invalid JSON request body") from None


def owner(request: Request) -> str:
    return request.state.owner


def create_app(
    service: HarnessService,
    *,
    auth: ServerAuth,
    expose_mcp: bool = False,
    mcp_operations: tuple[str, ...] = ("run_status", "sessions", "messages"),
    a2a_url: str | None = None,
    a2a_callback_grants: list[CallbackGrant] | None = None,
) -> Starlette:
    if a2a_callback_grants:
        if a2a_url is None:
            raise ValueError("A2A callback grants require an enabled A2A endpoint")
        service.a2a_callbacks = CallbackManager(service, a2a_callback_grants)
    static = Path(__file__).with_name("static")

    async def index(request: Request):
        return FileResponse(static / "index.html")

    async def javascript(request: Request):
        return FileResponse(static / "app.js", media_type="text/javascript")

    async def stylesheet(request: Request):
        return FileResponse(static / "app.css", media_type="text/css")

    async def health(request: Request):
        return JSONResponse({"service": "harness", "api_version": 1, "user_id": owner(request)})

    async def configuration(request: Request):
        return JSONResponse(service.configuration())

    async def preferences(request: Request):
        result = (
            await service.preferences(owner(request))
            if request.method == "GET"
            else await service.save_preferences(
                owner(request), UserPreferences.model_validate(await body(request))
            )
        )
        return JSONResponse(result.model_dump())

    async def tools(request: Request):
        return JSONResponse({"tools": service.tool_catalog()})

    async def callback_deliveries(request: Request):
        if service.a2a_callbacks is None:
            raise ServiceError(404, "A2A callbacks are disabled")
        return JSONResponse(
            {
                "deliveries": await callbacks(service).deliveries(owner(request)),
                "worker_error": callbacks(service).last_error,
            }
        )

    async def schedules(request: Request):
        if request.method == "GET":
            return JSONResponse({"schedules": await service.automations.list(owner(request))})
        return JSONResponse(
            await service.automations.create(
                owner(request), ScheduleSubmission.model_validate(await body(request))
            ),
            status_code=201,
        )

    async def schedule(request: Request):
        return JSONResponse(
            await service.automations.get(owner(request), request.path_params["schedule_id"])
        )

    async def schedule_action(request: Request):
        await body(request)
        return JSONResponse(
            await service.automations.change(
                owner(request), request.path_params["schedule_id"], request.path_params["action"]
            )
        )

    async def runs(request: Request):
        if request.method == "GET":
            return JSONResponse(
                {
                    "runs": await service.runs(
                        owner(request), session_id=request.query_params.get("session_id")
                    )
                }
            )
        result = await service.submit(
            owner(request), RunSubmission.model_validate(await body(request))
        )
        return JSONResponse(result, status_code=202)

    async def run_status(request: Request):
        return JSONResponse(await service.store.run(owner(request), request.path_params["run_id"]))

    async def events(request: Request):
        run_id = request.path_params["run_id"]
        await service.store.run(owner(request), run_id)
        try:
            after = int(
                request.query_params.get("after", request.headers.get("last-event-id", "0"))
            )
            if after < 0:
                raise ValueError
        except ValueError:
            raise ServiceError(422, "Event cursor must be a nonnegative integer") from None
        return StreamingResponse(
            service.events(owner(request), run_id, after=after),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no"},
        )

    async def cancel(request: Request):
        return JSONResponse(await service.cancel(owner(request), request.path_params["run_id"]))

    async def resume(request: Request):
        submission = ResumeSubmission.model_validate(await body(request))
        return JSONResponse(
            await service.resume(owner(request), request.path_params["run_id"], submission.prompt),
            status_code=202,
        )

    async def sessions(request: Request):
        try:
            limit = int(request.query_params.get("limit", "50"))
            offset = int(request.query_params.get("offset", "0"))
        except ValueError:
            raise ServiceError(422, "Invalid pagination") from None
        return JSONResponse(
            {
                "sessions": await service.sessions(
                    owner(request), query=request.query_params.get("q"), limit=limit, offset=offset
                )
            }
        )

    async def messages(request: Request):
        return JSONResponse(
            {"messages": await service.messages(owner(request), request.path_params["session_id"])}
        )

    async def export(request: Request):
        return Response(
            await service.export(owner(request), request.path_params["session_id"]),
            media_type="application/x-ndjson",
            headers={"Content-Disposition": 'attachment; filename="trajectory.jsonl"'},
        )

    async def approvals(request: Request):
        return JSONResponse({"approvals": await service.approvals(owner(request))})

    async def questions(request: Request):
        return JSONResponse({"questions": await service.questions.list(owner(request))})

    async def answer_question(request: Request):
        submission = QuestionAnswers.model_validate(await body(request))
        return JSONResponse(
            await service.questions.answer(
                owner(request), request.path_params["question_id"], submission
            )
        )

    async def cancel_question(request: Request):
        return JSONResponse(
            await service.questions.cancel(owner(request), request.path_params["question_id"])
        )

    async def resolve(request: Request):
        submission = ApprovalSubmission.model_validate(await body(request))
        return JSONResponse(
            await service.resolve_approval(
                owner(request), request.path_params["approval_id"], granted=submission.granted
            )
        )

    async def batches(request: Request):
        if request.method == "GET":
            return JSONResponse({"batches": await service.batches(owner(request))})
        submission = BatchSubmission.model_validate(await body(request))
        return JSONResponse(
            await service.submit_batch(owner(request), submission.runs), status_code=202
        )

    async def batch(request: Request):
        return JSONResponse(await service.batch(owner(request), request.path_params["batch_id"]))

    async def cancel_batch(request: Request):
        return JSONResponse(
            await service.cancel_batch(owner(request), request.path_params["batch_id"])
        )

    async def tool_run(request: Request):
        return JSONResponse(
            await service.submit_tool(
                owner(request), ToolSubmission.model_validate(await body(request))
            ),
            status_code=202,
        )

    async def delegations(request: Request):
        caller = owner(request)
        if request.method == "GET":
            return JSONResponse({"jobs": await service.delegation.list(caller)})
        submission = ChildSubmission.model_validate(await body(request))
        await service._owned_session(caller, submission.parent_session_id)
        await service.delegation.register_parent(
            caller,
            submission.parent_session_id,
            await service.session_workspace(submission.parent_session_id),
        )
        job = await service.delegation.submit(
            caller,
            submission.parent_session_id,
            DelegateSubmission.model_validate(submission.model_dump(exclude={"parent_session_id"})),
        )
        return JSONResponse(job, status_code=202)

    async def delegation_status(request: Request):
        return JSONResponse(
            await service.delegation.status(owner(request), request.path_params["handle"])
        )

    async def delegation_cancel(request: Request):
        return JSONResponse(
            await service.delegation.cancel(owner(request), request.path_params["handle"])
        )

    async def delegation_resume(request: Request):
        submission = ResumeSubmission.model_validate(await body(request))
        return JSONResponse(
            await service.delegation.resume(
                owner(request), request.path_params["handle"], prompt=submission.prompt
            ),
            status_code=202,
        )

    async def delegation_artifact(request: Request):
        path = request.query_params.get("path")
        if not path:
            raise ServiceError(422, "An artifact path is required")
        result = await service.delegation.artifact(
            owner(request), request.path_params["handle"], path
        )
        return Response(
            base64.b64decode(result.data or ""),
            media_type=result.mime_type,
            headers={
                "Content-Disposition": "attachment; filename*=UTF-8''"
                + quote(result.name or "artifact", safe="")
            },
        )

    async def service_error(request: Request, exc: Exception):
        assert isinstance(exc, ServiceError)
        return JSONResponse({"detail": exc.detail}, status_code=exc.status)

    async def validation_error(request: Request, exc: Exception):
        assert isinstance(exc, ValidationError)
        # Never echo credential-bearing input or arbitrary validator exception objects.
        return JSONResponse(
            {
                "detail": "Invalid request",
                "errors": [
                    {"loc": list(error["loc"]), "type": error["type"]}
                    for error in exc.errors(include_input=False, include_context=False)
                ],
            },
            status_code=422,
        )

    routes: list[BaseRoute] = [
        Route("/", index),
        Route("/ui/app.js", javascript),
        Route("/ui/app.css", stylesheet),
        Route("/v1/health", health),
        Route("/v1/configuration", configuration),
        Route("/v1/preferences", preferences, methods=["GET", "POST"]),
        Route("/v1/tools", tools),
        Route("/v1/a2a/callback-deliveries", callback_deliveries),
        Route("/v1/schedules", schedules, methods=["GET", "POST"]),
        Route("/v1/schedules/{schedule_id}", schedule),
        Route("/v1/schedules/{schedule_id}/{action}", schedule_action, methods=["POST"]),
        Route("/v1/runs", runs, methods=["GET", "POST"]),
        Route("/v1/runs/{run_id}", run_status),
        Route("/v1/runs/{run_id}/events", events),
        Route("/v1/runs/{run_id}/cancel", cancel, methods=["POST"]),
        Route("/v1/runs/{run_id}/resume", resume, methods=["POST"]),
        Route("/v1/sessions", sessions),
        Route("/v1/sessions/{session_id}/messages", messages),
        Route("/v1/sessions/{session_id}/export", export),
        Route("/v1/approvals", approvals),
        Route("/v1/questions", questions),
        Route("/v1/questions/{question_id}/answer", answer_question, methods=["POST"]),
        Route("/v1/questions/{question_id}/cancel", cancel_question, methods=["POST"]),
        Route("/v1/approvals/{approval_id}/resolve", resolve, methods=["POST"]),
        Route("/v1/batches", batches, methods=["GET", "POST"]),
        Route("/v1/batches/{batch_id}", batch),
        Route("/v1/batches/{batch_id}/cancel", cancel_batch, methods=["POST"]),
        Route("/v1/tool-runs", tool_run, methods=["POST"]),
        Route("/v1/delegations", delegations, methods=["GET", "POST"]),
        Route("/v1/delegations/{handle}", delegation_status),
        Route("/v1/delegations/{handle}/cancel", delegation_cancel, methods=["POST"]),
        Route("/v1/delegations/{handle}/resume", delegation_resume, methods=["POST"]),
        Route("/v1/delegations/{handle}/artifacts", delegation_artifact),
    ]
    mcp = None
    if a2a_url is not None:
        from harness.server.a2a import make_a2a_routes

        routes.extend(make_a2a_routes(service, a2a_url))
    if expose_mcp:
        from harness.server.mcp import make_mcp

        mcp = make_mcp(service, mcp_operations)
        routes.append(Mount("/mcp", app=mcp.streamable_http_app()))

    @asynccontextmanager
    async def lifespan(app: Starlette):
        async with AsyncExitStack() as stack:
            await service.start()
            stack.push_async_callback(service.close)
            if mcp is not None:
                await stack.enter_async_context(mcp.session_manager.run())
            yield

    return Starlette(
        routes=routes,
        lifespan=lifespan,
        middleware=[Middleware(AuthMiddleware, auth=auth)],
        exception_handlers={ServiceError: service_error, ValidationError: validation_error},
    )
