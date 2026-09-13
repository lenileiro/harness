import json

import pytest
import typer
from typer.testing import CliRunner

from harness.cli.delegate_commands import make_delegate_app
from harness.core import Agent, Capabilities, Done, FailoverPolicy, Message, ToolCall, ToolRegistry


class Adapter:
    name = "fake"

    async def capabilities(self):
        return Capabilities(tool_use=True)

    async def cancel(self, session_id):
        pass

    async def stream(self, **kwargs):
        yield Done(final_message=Message(role="assistant", content="delegated result"))


def application(calls, adapter=None):
    app = typer.Typer()

    def factory(workspace, config, provider, model):
        calls.append(workspace)

        def build(context):
            return Agent(
                adapters={"fake": adapter or Adapter()},
                tools=ToolRegistry(),
                storage=context.storage,
                failover=FailoverPolicy(chain=["fake"]),
                default_model="test",
            )

        return build

    app.add_typer(make_delegate_app(factory), name="delegate")
    return app


def test_cli_queue_status_work_resume_and_artifact(tmp_path):
    calls = []
    app = application(calls)
    runner = CliRunner()
    common = ["delegate", "--workspace", str(tmp_path)]
    (tmp_path / "input.txt").write_text("input copy")
    queued = runner.invoke(
        app, [*common, "submit", "independent task", "--queue-only", "--input", "input.txt"]
    )
    assert queued.exit_code == 0, queued.output
    job = json.loads(queued.output)
    assert job["state"] == "queued" and calls == []
    status = runner.invoke(app, [*common, "status", job["id"]])
    assert status.exit_code == 0 and json.loads(status.output)["state"] == "queued"
    assert calls == []
    worked = runner.invoke(app, [*common, "work"])
    assert worked.exit_code == 0, worked.output
    assert json.loads(worked.output)[0]["state"] == "completed" and len(calls) == 1
    resumed = runner.invoke(app, [*common, "resume", job["id"], "--queue-only"])
    assert resumed.exit_code == 0 and json.loads(resumed.output)["attempts"] == 2
    assert len(calls) == 1
    cancelled = runner.invoke(app, [*common, "cancel", job["id"]])
    assert cancelled.exit_code == 0 and json.loads(cancelled.output)["state"] == "cancelled"
    saved = runner.invoke(
        app,
        [*common, "artifact", job["id"], "input.txt", "--output", str(tmp_path / "reviewed.txt")],
    )
    assert saved.exit_code == 0 and (tmp_path / "reviewed.txt").read_text() == "input copy"
    duplicate = runner.invoke(
        app,
        [*common, "artifact", job["id"], "input.txt", "--output", str(tmp_path / "reviewed.txt")],
    )
    assert duplicate.exit_code != 0


def test_foreground_submit_waits_for_actual_result(tmp_path):
    calls = []
    result = CliRunner().invoke(
        application(calls), ["delegate", "--workspace", str(tmp_path), "submit", "do the task"]
    )
    assert result.exit_code == 0, result.output
    assert "Delegated job:" in result.output and '"state": "completed"' in result.output
    assert '"summary": "delegated result"' in result.output


@pytest.mark.parametrize("override", [False, True])
def test_concrete_local_child_reads_copied_input_by_default_and_explicit_flags_replace_preset(
    tmp_path, monkeypatch, override
):
    import importlib

    from harness.cli.server_builder import server_builder

    observed = []

    class ReadingAdapter(Adapter):
        async def stream(self, **kwargs):
            prior = [message for message in kwargs["messages"] if message.name == "read_file"]
            if prior:
                observed.extend(message.content for message in prior)
                yield Done(final_message=Message(role="assistant", content="Inspected input"))
            else:
                yield Done(
                    final_message=Message(
                        role="assistant",
                        tool_calls=[
                            ToolCall(id="read", name="read_file", arguments={"path": "input.txt"})
                        ],
                    )
                )

    module = importlib.import_module("harness.cli.server_builder")
    monkeypatch.setattr(module, "_build_adapter", lambda *args, **kwargs: ReadingAdapter())
    (tmp_path / "input.txt").write_text("copied public discovery evidence")
    config = tmp_path / "config.toml"
    config.write_text('[default]\nprovider="ollama"\nmodel="fake"\n')
    app = typer.Typer()
    app.add_typer(make_delegate_app(server_builder), name="delegate")
    flags = ["--expose-tool", "list_dir"] if override else []
    response = CliRunner().invoke(
        app,
        [
            "delegate",
            "--workspace",
            str(tmp_path),
            "--config",
            str(config),
            *flags,
            "submit",
            "Read the supplied input",
            "--input",
            "input.txt",
        ],
    )
    assert response.exit_code == 0, response.output
    assert observed
    if override:
        assert "copied public discovery evidence" not in observed
        assert "denied" in observed[0].lower()
    else:
        assert observed == ["copied public discovery evidence"]


@pytest.mark.parametrize("finish_question", ["answer", "skip"])
def test_child_questions_are_owned_partial_and_resume_separately(tmp_path, finish_question):
    results = []

    class AskingAdapter(Adapter):
        async def stream(self, **kwargs):
            prior = [message for message in kwargs["messages"] if message.name == "clarify"]
            if prior:
                results.append(json.loads(prior[0].content))
                yield Done(final_message=Message(role="assistant", content="delegated result"))
            else:
                yield Done(
                    final_message=Message(
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                id="ask-child",
                                name="clarify",
                                arguments={
                                    "questions": [
                                        {"question": "Format?", "choices": ["Report", "Chart"]},
                                        {"question": "Audience?"},
                                    ]
                                },
                            )
                        ],
                    )
                )

    calls = []
    app = application(calls, AskingAdapter())
    runner = CliRunner()
    common = ["delegate", "--workspace", str(tmp_path), "--expose-tool", "clarify"]

    def invoke(*args):
        response = runner.invoke(app, [*common, *args])
        assert response.exit_code == 0, response.output
        return json.loads(response.output)

    children = [invoke("submit", prompt, "--queue-only") for prompt in ("one", "two")]
    assert all(child["state"] == "paused" for child in invoke("work"))
    questions = [invoke("questions", child["id"])[0] for child in children]
    first, other = children
    question = questions[0]
    denied = runner.invoke(app, [*common, "answer", first["id"], questions[1]["id"], "wrong"])
    assert denied.exit_code != 0 and "does not belong" in denied.output
    stranger = runner.invoke(app, [*common, "--owner", "stranger", "questions", first["id"]])
    assert stranger.exit_code != 0 and "Audience?" not in stranger.output
    partial = invoke("answer", first["id"], question["id"], "1")
    assert partial["answers"] == {"q0": "Report"} and partial["status"] == "pending"
    blocked = runner.invoke(app, [*common, "resume", first["id"], "--queue-only"])
    assert blocked.exit_code != 0 and "question" in blocked.output
    assert invoke("status", first["id"])["attempts"] == 1
    assert len(calls) == 2  # Inspection and answers never construct a model adapter.
    if finish_question == "answer":
        saved = invoke("answer", first["id"], question["id"], "Team")
        assert saved["status"] == "answered"
    else:
        saved = invoke("skip-question", first["id"], question["id"])
        assert saved["status"] == "cancelled"
    assert saved["answers"]["q0"] == "Report" and len(calls) == 2
    invoke("resume", first["id"], "--queue-only")
    invoke("work")
    final = invoke("status", first["id"])
    assert final["state"] == "completed" and final["attempts"] == 2 and final["questions"] == []
    assert invoke("status", other["id"])["state"] == "paused"
    assert len(results) == 1 and results[0]["responses"][0]["user_response"] == "Report"
    if finish_question == "skip":
        assert results[0]["cancelled"] is True
    else:
        assert results[0]["responses"][1]["user_response"] == "Team"
