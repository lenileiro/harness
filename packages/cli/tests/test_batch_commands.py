from __future__ import annotations

import json

import typer
from typer.testing import CliRunner

from harness.cli.batch_commands import make_batch_app
from harness.core import Agent, Capabilities, Done, FailoverPolicy, Message, ToolRegistry
from harness.core.trajectories import parse_jsonl


class Adapter:
    name = "fake"

    async def capabilities(self):
        return Capabilities(tool_use=True)

    async def cancel(self, session_id):
        pass

    async def stream(self, **kwargs):
        yield Done(final_message=Message(role="assistant", content="batch result"))


def application(calls):
    app = typer.Typer()

    def factory(workspace, config, provider, model):
        calls.append(workspace)

        def build(context):
            return Agent(
                adapters={"fake": Adapter()},
                tools=ToolRegistry(),
                storage=context.storage,
                failover=FailoverPolicy(chain=["fake"]),
                default_model="test",
            )

        return build

    app.add_typer(make_batch_app(factory), name="batch")
    return app


def test_durable_queue_status_work_export_and_completed_resume_refusal(tmp_path):
    calls = []
    app = application(calls)
    runner = CliRunner()
    source = tmp_path / "requests.jsonl"
    source.write_text('{"prompt":"first"}\n{"prompt":"second"}\n')
    common = ["batch", "--workspace", str(tmp_path)]
    queued = runner.invoke(app, [*common, "submit", str(source), "--queue-only"])
    assert queued.exit_code == 0, queued.output
    batch = json.loads(queued.output)
    assert len(batch["runs"]) == 2 and calls == []
    status = runner.invoke(app, [*common, "status", batch["id"]])
    assert status.exit_code == 0 and calls == []
    worked = runner.invoke(app, [*common, "work"])
    assert worked.exit_code == 0, worked.output
    assert len(calls) == 2
    output = tmp_path / "batch-export.jsonl"
    exported = runner.invoke(app, [*common, "export", batch["id"], "--output", str(output)])
    assert exported.exit_code == 0, exported.output
    records = parse_jsonl(output.read_bytes())
    assert len(records) == 2 and all(
        record.messages[-1].content == "batch result" for record in records
    )
    resumed = runner.invoke(app, [*common, "resume", batch["runs"][0]["id"], "--queue-only"])
    assert resumed.exit_code == 1 and len(calls) == 2
    other = runner.invoke(app, [*common, "--owner", "other", "status", batch["id"]])
    assert other.exit_code == 1


def test_cancel_then_explicit_resume_and_atomic_bad_input(tmp_path):
    calls = []
    app = application(calls)
    runner = CliRunner()
    source = tmp_path / "requests.jsonl"
    source.write_text('{"prompt":"do not start yet"}\n')
    common = ["batch", "--workspace", str(tmp_path)]
    result = runner.invoke(app, [*common, "submit", str(source), "--queue-only"])
    batch = json.loads(result.output)
    cancelled = runner.invoke(app, [*common, "cancel", batch["id"]])
    assert cancelled.exit_code == 0 and calls == []
    assert json.loads(cancelled.output)["runs"][0]["state"] == "cancelled"
    resumed = runner.invoke(app, [*common, "resume", batch["runs"][0]["id"]])
    assert resumed.exit_code == 0, resumed.output
    assert json.loads(resumed.output)["state"] == "completed" and len(calls) == 1
    source.write_text('{"prompt":"valid"}\n{"prompt":"private","unsupported":true}')
    invalid = runner.invoke(app, [*common, "submit", str(source), "--queue-only"])
    assert invalid.exit_code == 1 and "private" not in invalid.output
    statuses = json.loads(runner.invoke(app, [*common, "status"]).output)
    assert len(statuses) == 1 and len(calls) == 1
