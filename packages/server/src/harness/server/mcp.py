"""Optional official-SDK MCP transport over the exact same authenticated service."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from harness.server.models import RunSubmission, ToolSubmission
from harness.server.questions import QuestionAnswers
from harness.server.service import HarnessService


def identity(ctx: Context) -> str:
    request = ctx.request_context.request
    if request is None or not isinstance(getattr(request.state, "owner", None), str):
        raise ValueError("Authenticated HTTP caller identity is required")
    return request.state.owner


def make_mcp(service: HarnessService, operations: tuple[str, ...]) -> FastMCP:
    # The outer AuthMiddleware enforces explicit hosts/origins for HTTP and MCP alike.
    mcp = FastMCP(
        "Harness",
        stateless_http=True,
        json_response=True,
        streamable_http_path="/",
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    async def run_status(run_id: str, ctx: Context) -> dict[str, Any]:
        return await service.store.run(identity(ctx), run_id)

    async def sessions(ctx: Context, query: str | None = None) -> list[dict[str, Any]]:
        return await service.sessions(identity(ctx), query=query)

    async def messages(session_id: str, ctx: Context) -> list[dict[str, Any]]:
        return await service.messages(identity(ctx), session_id)

    async def submit(prompt: str, ctx: Context, session_id: str | None = None) -> dict[str, Any]:
        return await service.submit(
            identity(ctx), RunSubmission(prompt=prompt, session_id=session_id)
        )

    async def cancel(run_id: str, ctx: Context) -> dict[str, Any]:
        return await service.cancel(identity(ctx), run_id)

    async def resume(run_id: str, ctx: Context) -> dict[str, Any]:
        return await service.resume(identity(ctx), run_id)

    async def approvals(ctx: Context) -> list[dict[str, Any]]:
        return await service.approvals(identity(ctx))

    async def questions(ctx: Context) -> list[dict[str, Any]]:
        return await service.questions.list(identity(ctx))

    async def answer_question(
        question_id: str, answers: dict[str, str | list[str]], ctx: Context
    ) -> dict[str, Any]:
        return await service.questions.answer(
            identity(ctx), question_id, QuestionAnswers(answers=answers)
        )

    async def cancel_question(question_id: str, ctx: Context) -> dict[str, Any]:
        return await service.questions.cancel(identity(ctx), question_id)

    async def resolve_approval(approval_id: str, granted: bool, ctx: Context) -> dict[str, Any]:
        return await service.resolve_approval(identity(ctx), approval_id, granted=granted)

    async def tool_run(name: str, arguments: dict[str, Any], ctx: Context) -> dict[str, Any]:
        return await service.submit_tool(
            identity(ctx), ToolSubmission(name=name, arguments=arguments)
        )

    functions = {
        function.__name__: function
        for function in (
            run_status,
            sessions,
            messages,
            submit,
            cancel,
            resume,
            approvals,
            questions,
            answer_question,
            cancel_question,
            resolve_approval,
            tool_run,
        )
    }
    unknown = set(operations) - functions.keys()
    if unknown:
        raise ValueError(f"Unknown MCP exposure operations: {', '.join(sorted(unknown))}")
    for name in dict.fromkeys(operations):
        mcp.add_tool(functions[name], name=name)
    return mcp
