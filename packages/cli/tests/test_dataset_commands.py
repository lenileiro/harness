from __future__ import annotations

import asyncio
import json

import typer
from typer.testing import CliRunner

from harness.cli.dataset_commands import dataset_app
from harness.core.memory import MemoryScope
from harness.core.schemas import Message, Session
from harness.core.trajectories import parse_jsonl
from harness.storage.sqlite import SQLiteStorage


def app():
    application = typer.Typer()
    application.add_typer(dataset_app, name="dataset")
    return application


def test_import_validate_compress_prepare_real_files_and_no_overwrite(tmp_path):
    source = tmp_path / "public.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i}"},
            ]
        }
        for i in range(6)
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows))
    canonical = tmp_path / "archive.jsonl"
    runner = CliRunner()
    command = ["dataset", "import", str(source), "--format", "trl", "--output", str(canonical)]
    result = runner.invoke(app(), command)
    assert result.exit_code == 0, result.output
    assert len(parse_jsonl(canonical.read_bytes())) == 6
    assert runner.invoke(app(), command).exit_code == 1
    assert (
        runner.invoke(app(), ["dataset", "validate", str(canonical), "--complete"]).exit_code == 0
    )
    compressed = tmp_path / "compressed.jsonl"
    assert (
        runner.invoke(
            app(),
            [
                "dataset",
                "compress",
                str(canonical),
                "--output",
                str(compressed),
                "--max-tokens",
                "1000",
            ],
        ).exit_code
        == 0
    )
    output = tmp_path / "training"
    result = runner.invoke(
        app(),
        [
            "dataset",
            "prepare",
            str(compressed),
            "--output",
            str(output),
            "--format",
            "trl",
            "--seed",
            "77",
            "--eval-fraction",
            "0.3",
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads((output / "manifest.json").read_text())
    assert report["training_records"] == 4 and report["evaluation_records"] == 2
    assert report["seed"] == 77
    assert (output / "train.jsonl").stat().st_mode & 0o777 == 0o600
    assert output.stat().st_mode & 0o777 == 0o700
    before = (output / "manifest.json").read_bytes()
    assert (
        runner.invoke(
            app(), ["dataset", "prepare", str(compressed), "--output", str(output)]
        ).exit_code
        == 1
    )
    assert (output / "manifest.json").read_bytes() == before


def test_export_checks_scope_and_preserves_versioned_messages(tmp_path):
    database = tmp_path / "sessions.db"

    async def save():
        storage = SQLiteStorage(path=database)
        session = Session(
            provider="fake",
            model="test",
            cwd=tmp_path,
            metadata={
                "memory_scope": MemoryScope(workspace=str(tmp_path), user_id="alice").model_dump()
            },
            messages=[
                Message(role="user", content="Hello"),
                Message(role="assistant", content="OPENAI_API_KEY=sk-" + "a" * 30),
            ],
            status="done",
        )
        try:
            await storage.save(session)
            return session.id
        finally:
            await storage.close()

    session_id = asyncio.run(save())
    runner = CliRunner()
    output = tmp_path / "export.jsonl"
    command = [
        "dataset",
        "export",
        session_id,
        "--database",
        str(database),
        "--workspace",
        str(tmp_path),
        "--output",
        str(output),
    ]
    wrong = runner.invoke(app(), command)
    assert wrong.exit_code == 1 and not output.exists()
    result = runner.invoke(app(), [*command, "--user", "alice"])
    assert result.exit_code == 0, result.output
    record = parse_jsonl(output.read_bytes())[0]
    assert record.session["id"] == session_id
    assert record.messages[0].content == "Hello"
    assert "sk-" + "a" * 30 not in output.read_text()


def test_failed_validation_never_publishes_partial_output(tmp_path):
    source = tmp_path / "invalid.jsonl"
    source.write_text(
        '{"messages":[{"role":"user","content":"valid"},{"role":"assistant","content":"ok"}]}\n{"invalid":"private-data"}'
    )
    output = tmp_path / "no-output.jsonl"
    result = CliRunner().invoke(
        app(), ["dataset", "import", str(source), "--format", "trl", "--output", str(output)]
    )
    assert result.exit_code == 1 and "line 2" in result.output
    assert "private-data" not in result.output and not output.exists()
