"""Durable batch CLI backed by the authenticated server's execution service."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from harness.cli.dataset_commands import _write
from harness.cli.serve_commands import BuilderFactory
from harness.cli.server_presentation import server_presentation
from harness.core.paths import read_regular_file
from harness.server import HarnessService, RunSubmission, ServiceError
from harness.server.models import TERMINAL_STATES
from harness.server.store import now


@dataclass
class BatchOptions:
    workspace: Path
    database: Path
    owner: str
    config: Path | None
    adapter: str | None
    model: str | None
    tools: list[str]
    workers: int


def make_batch_app(builder_factory: BuilderFactory) -> typer.Typer:
    app = typer.Typer(
        no_args_is_help=True, help="Queue, run, inspect and export durable local batches."
    )

    @app.callback()
    def configure(
        ctx: typer.Context,
        workspace: Annotated[Path, typer.Option("--workspace", "-C")] = Path("."),
        database: Annotated[Path | None, typer.Option("--database")] = None,
        owner: Annotated[
            str, typer.Option("--owner", help="Trusted local database identity.")
        ] = "local",
        config: Annotated[Path | None, typer.Option("--config")] = None,
        adapter: Annotated[str | None, typer.Option("--adapter")] = None,
        model: Annotated[str | None, typer.Option("--model")] = None,
        expose_tool: Annotated[list[str] | None, typer.Option("--expose-tool")] = None,
        workers: Annotated[int, typer.Option("--workers", min=1, max=32)] = 2,
    ) -> None:
        cwd = workspace.expanduser().resolve()
        if not cwd.is_dir():
            raise typer.BadParameter("Workspace must be an existing directory")
        ctx.obj = BatchOptions(
            cwd,
            (database or cwd / ".harness/server.db").expanduser().resolve(),
            owner,
            config,
            adapter,
            model,
            expose_tool or [],
            workers,
        )

    @asynccontextmanager
    async def service_for(options: BatchOptions, *, dispatch: bool):
        def lazy_builder(context):
            return builder_factory(
                options.workspace, options.config, options.adapter, options.model
            )(context)

        service = HarnessService(
            options.database,
            options.workspace,
            lazy_builder,
            exposed_tools=options.tools,
            max_workers=options.workers,
            presentation=server_presentation(
                options.workspace, options.config, options.adapter, options.model
            ).model_copy(update={"default_provider": None, "default_model": None}),
        )
        await service.start(dispatch=dispatch)
        try:
            yield service
        finally:
            await service.close()

    def execute(coroutine):
        try:
            return asyncio.run(coroutine)
        except (ValueError, OSError, RuntimeError) as exc:
            detail = (
                str(exc)
                if isinstance(exc, ServiceError)
                else "invalid input, unavailable database/worker, or existing output; inspect local paths"
            )
            typer.echo(f"Batch operation failed: {detail}", err=True)
            raise typer.Exit(1) from None

    def emit(value):
        typer.echo(json.dumps(value, indent=2, ensure_ascii=False))

    async def wait_batch(service: HarnessService, options: BatchOptions, handle: str):
        batch = await service.batch(options.owner, handle)
        for run in batch["runs"]:
            async for _ in service.events(options.owner, run["id"]):
                pass
        result = await service.batch(options.owner, handle)
        emit(result)
        if any(run["state"] != "completed" for run in result["runs"]):
            raise typer.Exit(1)

    @app.command()
    def submit(
        ctx: typer.Context,
        source: Annotated[
            Path,
            typer.Argument(help="JSONL RunSubmission objects; validated before queueing any row."),
        ],
        queue_only: Annotated[bool, typer.Option("--queue-only")] = False,
    ) -> None:
        options: BatchOptions = ctx.obj

        async def run():
            data = read_regular_file(source, max_bytes=24 * 1024 * 1024)
            values = [
                RunSubmission.model_validate_json(line)
                for line in data.splitlines()
                if line.strip()
            ]
            async with service_for(options, dispatch=not queue_only) as service:
                batch = await service.submit_batch(options.owner, values)
                if queue_only:
                    emit(batch)
                else:
                    typer.echo(f"Batch: {batch['id']}")
                    await wait_batch(service, options, batch["id"])

        execute(run())

    @app.command()
    def status(ctx: typer.Context, handle: Annotated[str | None, typer.Argument()] = None) -> None:
        options: BatchOptions = ctx.obj

        async def run():
            async with service_for(options, dispatch=False) as service:
                emit(
                    await service.batch(options.owner, handle)
                    if handle
                    else await service.batches(options.owner)
                )

        execute(run())

    @app.command()
    def cancel(ctx: typer.Context, handle: Annotated[str, typer.Argument()]) -> None:
        options: BatchOptions = ctx.obj

        async def run():
            async with service_for(options, dispatch=False) as service:
                emit(await service.cancel_batch(options.owner, handle))

        execute(run())

    @app.command()
    def resume(
        ctx: typer.Context,
        run_id: Annotated[
            str,
            typer.Argument(
                help="Explicit failed/interrupted/paused run to continue; completed work is never silently replayed."
            ),
        ],
        prompt: Annotated[str | None, typer.Option("--prompt")] = None,
        queue_only: Annotated[bool, typer.Option("--queue-only")] = False,
    ) -> None:
        options: BatchOptions = ctx.obj

        async def run():
            async with service_for(options, dispatch=not queue_only) as service:
                saved = await service.store.run(options.owner, run_id)
                if saved["state"] == "completed":
                    raise ServiceError(409, "Completed runs require a new explicit submission")
                resumed = await service.resume(options.owner, run_id, prompt)
                if not queue_only:
                    async for _ in service.events(options.owner, resumed["id"]):
                        pass
                    resumed = await service.store.run(options.owner, resumed["id"])
                emit(resumed)
                if not queue_only and resumed["state"] != "completed":
                    raise typer.Exit(1)

        execute(run())

    @app.command()
    def work(ctx: typer.Context) -> None:
        """Drain this database's durable queue. Interruption remains inspectable and resumable."""
        options: BatchOptions = ctx.obj

        async def run():
            started_at = now()
            async with service_for(options, dispatch=True) as service:
                while True:
                    rows = await service.store.rows(
                        "SELECT id,owner FROM api_runs WHERE state IN ('queued','running') LIMIT 1"
                    )
                    if not rows:
                        break
                    async for _ in service.events(rows[0]["owner"], rows[0]["id"]):
                        pass
                values = await service.batches(options.owner)
                emit(values)
                changed = await service.store.rows(
                    "SELECT state FROM api_runs WHERE owner=? AND updated_at>=? AND state!='completed'",
                    (options.owner, started_at),
                )
                if changed:
                    raise typer.Exit(1)

        execute(run())

    @app.command()
    def export(
        ctx: typer.Context,
        handle: Annotated[str, typer.Argument()],
        output: Annotated[Path, typer.Option("--output")],
    ) -> None:
        """Export complete per-session trajectories in this owned batch as JSONL."""
        options: BatchOptions = ctx.obj

        async def run():
            async with service_for(options, dispatch=False) as service:
                batch = await service.batch(options.owner, handle)
                if any(run["state"] not in TERMINAL_STATES for run in batch["runs"]):
                    raise ServiceError(
                        409,
                        "Batch has active work; wait or cancel before exporting a stable snapshot",
                    )
                sessions = dict.fromkeys(run["session_id"] for run in batch["runs"])
                records = [await service.export(options.owner, session) for session in sessions]
                _write(output, "".join(records).encode())
                typer.echo(f"Exported {len(records)} owned session trajectories to {output}.")

        execute(run())

    return app
