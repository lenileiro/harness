"""Durable local delegation commands using the same service as HTTP and model tools."""

from __future__ import annotations

import asyncio
import base64
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from harness.cli.serve_commands import BuilderFactory
from harness.cli.server_presentation import server_presentation
from harness.server import DelegateSubmission, HarnessService, ServiceError
from harness.server.service import identifier

LOCAL_DISCOVERY_TOOLS = (
    "read_file",
    "list_dir",
    "glob",
    "web_search",
    "fetch_url",
    "recall_memory",
    "search_sessions",
    "conversation_window",
)


@dataclass
class Options:
    workspace: Path
    database: Path
    owner: str
    config: Path | None
    adapter: str | None
    model: str | None
    exposed_tools: list[str]


class Decision(StrEnum):
    grant = "grant"
    deny = "deny"


def make_delegate_app(builder_factory: BuilderFactory) -> typer.Typer:
    app = typer.Typer(
        no_args_is_help=True,
        help="Run isolated child agents with durable handles and reserved budgets.",
    )

    @app.callback()
    def configure(
        ctx: typer.Context,
        workspace: Annotated[Path, typer.Option("--workspace", "-C")] = Path("."),
        database: Annotated[Path | None, typer.Option("--database")] = None,
        owner: Annotated[
            str,
            typer.Option(
                "--owner",
                help="Local database identity; the operating-system user must own this database",
            ),
        ] = "local",
        config: Annotated[Path | None, typer.Option("--config")] = None,
        adapter: Annotated[str | None, typer.Option("--adapter")] = None,
        model: Annotated[str | None, typer.Option("--model")] = None,
        expose_tool: Annotated[
            list[str] | None,
            typer.Option(
                "--expose-tool",
                help=(
                    "Replace the local read-only file, web and scoped recall preset "
                    "with these explicitly permitted child tools; repeat to allow multiple tools"
                ),
            ),
        ] = None,
    ) -> None:
        cwd = workspace.expanduser().resolve()
        if not cwd.is_dir():
            raise typer.BadParameter("Workspace must be an existing directory")
        ctx.obj = Options(
            cwd,
            (database or cwd / ".harness" / "delegation.db").expanduser().resolve(),
            owner,
            config,
            adapter,
            model,
            expose_tool if expose_tool else list(LOCAL_DISCOVERY_TOOLS),
        )

    @asynccontextmanager
    async def service_for(options: Options, *, dispatch: bool):
        # Read-only status operations never construct adapters or inspect credentials.
        def lazy_builder(context):
            return builder_factory(
                options.workspace, options.config, options.adapter, options.model
            )(context)

        service = HarnessService(
            options.database,
            options.workspace,
            lazy_builder,
            exposed_tools=options.exposed_tools,
            presentation=server_presentation(
                options.workspace, options.config, options.adapter, options.model
            ).model_copy(update={"default_provider": None, "default_model": None}),
        )
        await service.start(dispatch=dispatch)
        try:
            yield service
        finally:
            await service.close()

    def emit(value):
        typer.echo(json.dumps(value, indent=2, ensure_ascii=False))

    def execute(coroutine):
        try:
            return asyncio.run(coroutine)
        except (ServiceError, ValueError, RuntimeError) as exc:
            raise typer.BadParameter(str(exc)) from None

    async def wait_job(service: HarnessService, owner: str, handle: str):
        job = await service.delegation.status(owner, handle)
        async for _ in service.events(owner, job["run"]["id"]):
            pass
        result = await service.delegation.status(owner, handle)
        emit(result)
        if result["state"] in ("failed", "cancelled", "interrupted"):
            raise typer.Exit(1)

    @app.command()
    def submit(
        ctx: typer.Context,
        prompt: Annotated[str, typer.Argument(help="Concrete task for an independent child agent")],
        parent: Annotated[
            str | None, typer.Option("--parent", help="Reuse a parent's durable shared budget")
        ] = None,
        input_file: Annotated[
            list[str] | None,
            typer.Option("--input", help="Public relative file to copy into the child workspace"),
        ] = None,
        max_steps: Annotated[int, typer.Option("--max-steps", min=1, max=100)] = 10,
        max_tokens: Annotated[int, typer.Option("--max-tokens", min=1, max=16000)] = 1024,
        timeout: Annotated[float, typer.Option("--timeout", min=1, max=3600)] = 120,
        queue_only: Annotated[
            bool,
            typer.Option("--queue-only", help="Persist only; execute later with delegate work"),
        ] = False,
    ) -> None:
        """Queue a child and wait for its result, or persist it for a worker."""
        options: Options = ctx.obj

        async def run():
            async with service_for(options, dispatch=not queue_only) as service:
                parent_id = parent or identifier("parent")
                await service.delegation.register_parent(
                    options.owner, parent_id, options.workspace
                )
                request = DelegateSubmission(
                    prompt=prompt,
                    inputs=input_file or [],
                    model=options.model,
                    max_steps=max_steps,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout,
                )
                job = await service.delegation.submit(options.owner, parent_id, request)
                if queue_only:
                    emit(job)
                else:
                    typer.echo(f"Delegated job: {job['id']} (parent {parent_id})")
                    await wait_job(service, options.owner, job["id"])

        execute(run())

    @app.command()
    def status(ctx: typer.Context, handle: Annotated[str | None, typer.Argument()] = None) -> None:
        """Read durable status without starting queued model calls."""
        options: Options = ctx.obj

        async def run():
            async with service_for(options, dispatch=False) as service:
                emit(
                    await service.delegation.status(options.owner, handle)
                    if handle
                    else await service.delegation.list(options.owner)
                )

        execute(run())

    @app.command()
    def cancel(ctx: typer.Context, handle: Annotated[str, typer.Argument()]) -> None:
        """Cancel a job and descendants; a running worker acknowledges durable cancellation."""
        options: Options = ctx.obj

        async def run():
            async with service_for(options, dispatch=False) as service:
                emit(await service.delegation.cancel(options.owner, handle))

        execute(run())

    @app.command()
    def resume(
        ctx: typer.Context,
        handle: Annotated[str, typer.Argument()],
        queue_only: Annotated[bool, typer.Option("--queue-only")] = False,
    ) -> None:
        """Reserve a new attempt and resume saved state; inspect uncertain effects first."""
        options: Options = ctx.obj

        async def run():
            async with service_for(options, dispatch=not queue_only) as service:
                job = await service.delegation.resume(options.owner, handle)
                if queue_only:
                    emit(job)
                else:
                    typer.echo(f"Resuming delegated job: {handle}")
                    await wait_job(service, options.owner, handle)

        execute(run())

    @app.command()
    def work(ctx: typer.Context) -> None:
        """Drain the durable queue in the foreground; Ctrl-C preserves interrupted work."""
        options: Options = ctx.obj

        async def run():
            async with service_for(options, dispatch=True) as service:
                while True:
                    rows = await service.store.rows(
                        "SELECT id,owner FROM api_runs WHERE state IN ('queued','running') LIMIT 1"
                    )
                    if not rows:
                        break
                    async for _ in service.events(rows[0]["owner"], rows[0]["id"]):
                        pass
                emit(await service.delegation.list(options.owner))

        execute(run())

    @app.command()
    def approvals(ctx: typer.Context, handle: Annotated[str, typer.Argument()]) -> None:
        """Review a child's pending tool arguments before granting or denying them."""
        options: Options = ctx.obj

        async def run():
            async with service_for(options, dispatch=False) as service:
                emit((await service.delegation.status(options.owner, handle))["approvals"])

        execute(run())

    @app.command()
    def questions(ctx: typer.Context, handle: Annotated[str, typer.Argument()]) -> None:
        """Inspect a child's pending questions and saved partial answers."""
        options: Options = ctx.obj

        async def run():
            async with service_for(options, dispatch=False) as service:
                emit((await service.delegation.status(options.owner, handle))["questions"])

        execute(run())

    async def owned_question(
        service: HarnessService, options: Options, handle: str, question_id: str
    ):
        job = await service.delegation.status(options.owner, handle)
        if question_id not in {item["id"] for item in job["questions"]}:
            raise ServiceError(404, "Pending question does not belong to this job")
        record, _ = await service.questions._owned(options.owner, question_id)
        return record

    @app.command()
    def answer(
        ctx: typer.Context,
        handle: Annotated[str, typer.Argument()],
        question_id: Annotated[str, typer.Argument()],
        answer_text: Annotated[
            str, typer.Argument(help="Answer the next question, or JSON q0/q1 mapping")
        ],
    ) -> None:
        """Save a child's answers without granting actions or reserving another attempt."""
        from harness.core.clarification import parse_question_answer
        from harness.server.questions import QuestionAnswers

        options: Options = ctx.obj

        async def run():
            async with service_for(options, dispatch=False) as service:
                record = await owned_question(service, options, handle, question_id)
                emit(
                    await service.questions.answer(
                        options.owner,
                        question_id,
                        QuestionAnswers(answers=parse_question_answer(record, answer_text)),
                    )
                )

        execute(run())

    @app.command("skip-question")
    def skip_question(
        ctx: typer.Context,
        handle: Annotated[str, typer.Argument()],
        question_id: Annotated[str, typer.Argument()],
    ) -> None:
        """Keep saved answers and skip the remainder; resume the child separately."""
        options: Options = ctx.obj

        async def run():
            async with service_for(options, dispatch=False) as service:
                await owned_question(service, options, handle, question_id)
                emit(await service.questions.cancel(options.owner, question_id))

        execute(run())

    @app.command()
    def resolve(
        ctx: typer.Context,
        handle: Annotated[str, typer.Argument()],
        approval_id: Annotated[str, typer.Argument()],
        decision: Annotated[Decision, typer.Option("--decision")],
    ) -> None:
        """Record an explicit child approval decision; resume separately to execute it."""
        options: Options = ctx.obj

        async def run():
            async with service_for(options, dispatch=False) as service:
                job = await service.delegation.status(options.owner, handle)
                if approval_id not in {item["id"] for item in job["approvals"]}:
                    raise ServiceError(404, "Pending approval does not belong to this job")
                emit(
                    await service.resolve_approval(
                        options.owner, approval_id, granted=decision == Decision.grant
                    )
                )

        execute(run())

    @app.command()
    def artifact(
        ctx: typer.Context,
        handle: Annotated[str, typer.Argument()],
        path: Annotated[str, typer.Argument()],
        output: Annotated[Path, typer.Option("--output")],
    ) -> None:
        """Save one explicitly selected child artifact; never overwrites an existing file."""
        options: Options = ctx.obj

        async def run():
            async with service_for(options, dispatch=False) as service:
                result = await service.delegation.artifact(options.owner, handle, path)
                destination = output.expanduser()
                with destination.open("xb") as target:
                    target.write(base64.b64decode(result.data or ""))
                typer.echo(f"Saved {destination}")

        try:
            execute(run())
        except OSError as exc:
            raise typer.BadParameter(f"Cannot save artifact: {exc.strerror}") from None

    return app


__all__ = ["make_delegate_app"]
