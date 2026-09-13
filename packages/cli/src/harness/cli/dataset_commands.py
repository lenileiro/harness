"""Offline dataset conversion, validation and reproducible training preparation."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Never, cast

import typer

from harness.core.memory import MemoryScope
from harness.core.paths import read_regular_file
from harness.core.secret_redaction import redact_secrets
from harness.core.trajectories import (
    DatasetFormat,
    Trajectory,
    TrajectoryError,
    compress_trajectory,
    parse_jsonl,
    prepare_dataset,
    validate_trajectory,
)
from harness.storage.sqlite import SQLiteStorage, default_db_path

MAX_DATASET_BYTES = 128 * 1024 * 1024
dataset_app = typer.Typer(
    no_args_is_help=True,
    help="Export, import, validate and prepare local trajectory datasets; never launch training.",
)


class Format(StrEnum):
    harness = "harness"
    openai = "openai"
    trl = "trl"
    sharegpt = "sharegpt"


class TrainingFormat(StrEnum):
    openai = "openai"
    trl = "trl"
    sharegpt = "sharegpt"


def _schemas(path: Path | None) -> list[dict[str, Any]] | None:
    if path is None:
        return None
    try:
        value = json.loads(read_regular_file(path, max_bytes=4 * 1024 * 1024))
    except (ValueError, OSError):
        raise TrajectoryError("tool schemas must be a readable JSON array") from None
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TrajectoryError("tool schemas must be a JSON array of function objects")
    return value


def _read(
    path: Path, source_format: Format = Format.harness, tools: Path | None = None
) -> list[Trajectory]:
    return parse_jsonl(
        read_regular_file(path, max_bytes=MAX_DATASET_BYTES),
        source_format=cast(DatasetFormat, source_format.value),
        tools=_schemas(tools),
    )


def _bytes(records: list[Any]) -> bytes:
    return (
        "\n".join(
            json.dumps(
                item.model_dump(mode="json") if isinstance(item, Trajectory) else item,
                ensure_ascii=False,
                allow_nan=False,
            )
            for item in records
        )
        + ("\n" if records else "")
    ).encode()


def _write(path: Path, data: bytes) -> None:
    """A complete owner-only file, published exclusively without overwriting."""
    fd, name = tempfile.mkstemp(prefix=".harness-dataset-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _failure(exc: Exception) -> Never:
    detail = (
        str(exc)
        if isinstance(exc, TrajectoryError)
        else "invalid input or inaccessible/existing output; check paths and file format"
    )
    typer.echo(f"Dataset operation failed: {detail}", err=True)
    raise typer.Exit(1) from None


@dataset_app.command("import")
def import_command(
    source: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Option("--output")],
    source_format: Annotated[Format, typer.Option("--format")] = Format.harness,
    tools: Annotated[
        Path | None,
        typer.Option(
            "--tools",
            help="Original public function schemas, required later for tool-call training.",
        ),
    ] = None,
) -> None:
    """Convert JSONL to versioned archives; imported records never become executable sessions."""
    try:
        records = _read(source, source_format, tools)
        _write(output, _bytes(records))
    except (ValueError, OSError) as exc:
        _failure(exc)
    typer.echo(f"Imported {len(records)} trajectories to {output}.")


@dataset_app.command("validate")
def validate_command(
    source: Annotated[Path, typer.Argument()],
    complete: Annotated[
        bool, typer.Option("--complete", help="Require every tool call to have a matching result.")
    ] = False,
) -> None:
    try:
        records = _read(source)
        for record in records:
            validate_trajectory(record, require_complete=complete)
    except (ValueError, OSError) as exc:
        _failure(exc)
    typer.echo(f"Validated {len(records)} versioned trajectories.")


@dataset_app.command("compress")
def compress_command(
    source: Annotated[Path, typer.Argument()],
    output: Annotated[Path, typer.Option("--output")],
    max_tokens: Annotated[
        int,
        typer.Option(
            "--max-tokens",
            min=1,
            help="Approximate UTF-8 JSON bytes/4 budget; no tokenizer download.",
        ),
    ],
    tool_result_chars: Annotated[int, typer.Option("--tool-result-chars", min=64)] = 2000,
) -> None:
    try:
        records = [
            compress_trajectory(record, max_tokens=max_tokens, tool_result_chars=tool_result_chars)
            for record in _read(source)
        ]
        _write(output, _bytes(records))
    except (ValueError, OSError) as exc:
        _failure(exc)
    typer.echo(
        f"Compressed {len(records)} trajectories; review metadata.compression for every omission."
    )


@dataset_app.command("prepare")
def prepare_command(
    source: Annotated[Path, typer.Argument()],
    output: Annotated[
        Path,
        typer.Option(
            "--output", help="New private directory for train.jsonl, eval.jsonl and manifest.json."
        ),
    ],
    target: Annotated[TrainingFormat, typer.Option("--format")] = TrainingFormat.trl,
    tools: Annotated[Path | None, typer.Option("--tools")] = None,
    seed: Annotated[int, typer.Option("--seed")] = 42,
    sample: Annotated[int | None, typer.Option("--sample", min=1)] = None,
    eval_fraction: Annotated[float, typer.Option("--eval-fraction", min=0, max=0.99)] = 0.1,
    max_tokens: Annotated[int | None, typer.Option("--max-tokens", min=1)] = None,
) -> None:
    """Produce training-ready public JSONL plus reproducibility manifest; no model weights change."""
    staging: Path | None = None
    reservation: os.stat_result | None = None
    try:
        records = _read(source, tools=tools)
        train, evaluation, manifest = prepare_dataset(
            records,
            target=cast(Any, target.value),
            seed=seed,
            sample=sample,
            eval_fraction=eval_fraction,
            max_tokens=max_tokens,
        )
        staging = Path(tempfile.mkdtemp(prefix=".harness-training-", dir=output.parent))
        _write(staging / "train.jsonl", _bytes(train))
        _write(staging / "eval.jsonl", _bytes(evaluation))
        _write(staging / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
        output.mkdir(mode=0o700, exist_ok=False)
        reservation = output.stat(follow_symlinks=False)
        staging.rename(output)
        staging = None
        reservation = None
    except (ValueError, OSError) as exc:
        _failure(exc)
    finally:
        if staging is not None:
            shutil.rmtree(staging)
        if reservation is not None:
            current = output.stat(follow_symlinks=False)
            if (current.st_dev, current.st_ino) == (reservation.st_dev, reservation.st_ino):
                output.rmdir()
    typer.echo(
        f"Prepared {len(train)} training and {len(evaluation)} evaluation records in {output}. No training job was started."
    )


def _redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_secrets(value)[0]
    if isinstance(value, dict):
        return {key: item if key == "attachments" else _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


@dataset_app.command("export")
def export_command(
    session_id: Annotated[str, typer.Argument()],
    output: Annotated[Path, typer.Option("--output")],
    database: Annotated[Path | None, typer.Option("--database")] = None,
    workspace: Annotated[Path, typer.Option("--workspace")] = Path("."),
    user: Annotated[
        str | None,
        typer.Option(
            "--user", help="Trusted local database identity; must match saved session scope."
        ),
    ] = None,
    tools: Annotated[Path | None, typer.Option("--tools")] = None,
) -> None:
    """Export one owned local session, preserving messages/media and tool relationships."""

    async def export() -> Trajectory:
        storage = SQLiteStorage(path=database or default_db_path())
        try:
            session = await storage.get(session_id)
            if session is None:
                raise TrajectoryError("session not found")
            saved_scope = session.metadata.get("memory_scope")
            if saved_scope is None or MemoryScope.model_validate(saved_scope) != MemoryScope(
                workspace=str(workspace), user_id=user
            ):
                raise TrajectoryError("session scope does not match the selected workspace/user")
            record = Trajectory(
                session=session.model_dump(mode="json", exclude={"messages"}),
                messages=session.messages,
                tools=_schemas(tools) or [],
                metadata={
                    "export": "local-session",
                    "redaction": "known-secret-patterns; attachments retained",
                },
            )
            record = Trajectory.model_validate(_redact(record.model_dump(mode="json")))
            validate_trajectory(record)
            return record
        finally:
            await storage.close()

    try:
        record = asyncio.run(export())
        _write(output, _bytes([record]))
    except (ValueError, OSError) as exc:
        _failure(exc)
    typer.echo(f"Exported session to {output}. Review content and attachments before sharing.")
