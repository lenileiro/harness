from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.cli import gateway_runtime, scheduler_commands
from harness.cli.gateway_runtime import GatewayApprovalPolicy
from harness.core import (
    Agent,
    ApprovalDecision,
    Capabilities,
    Done,
    FailoverPolicy,
    InboxApprovalHandler,
    Message,
    RunRequest,
    ToolCall,
    ToolRegistry,
    ToolResult,
)
from harness.core.gateway_models import default_gateway_root
from harness.core.gateway_sessions import GatewaySessionStore
from harness.core.memory import MemoryScope
from harness.core.scheduler_models import SchedulerExecutionResult
from harness.core.scheduler_runtime import retry_scheduler_deliveries, run_scheduler_job
from harness.core.scheduler_store import SchedulerStore
from harness.storage.sqlite import SQLiteStorage


def _fake_turn(tmp_path, calls, turns):
    class Action:
        name = "scheduled_action"
        description = "Append marker"
        approval: ApprovalDecision = "auto"
        effect_scope = "workspace_durable"

        def __init__(self):
            self.parameters_schema: dict[str, Any] = {"type": "object", "properties": {}}

        async def __call__(self, call):
            calls.append(call.id)
            return ToolResult(tool_call_id=call.id, name=self.name, content="marker appended")

    class Adapter:
        name = "fake"

        async def capabilities(self):
            return Capabilities(streaming=True, tool_use=True)

        async def cancel(self, session_id):
            pass

        async def stream(self, **kwargs):
            latest_user = next(
                item.content for item in reversed(kwargs["messages"]) if item.role == "user"
            )
            if "approved" in latest_user.lower():
                yield Done(
                    final_message=Message(role="assistant", content="Scheduled marker complete.")
                )
            else:
                yield Done(
                    final_message=Message(
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                id=f"scheduled-{len(turns)}", name="scheduled_action", arguments={}
                            )
                        ],
                    )
                )

    async def run(**kwargs):
        turns.append(kwargs)
        storage = SQLiteStorage(path=tmp_path / ".harness/harness.db")
        registry = ToolRegistry()
        registry.register(Action())
        scope = MemoryScope(
            workspace=str(tmp_path),
            user_id=None
            if kwargs.get("local_only")
            else json.dumps([kwargs["transport"], kwargs["user_id"]], separators=(",", ":")),
        )
        try:
            agent = Agent(
                adapters={"fake": Adapter()},
                tools=registry,
                storage=storage,
                approval_store=storage,
                approval_handler=InboxApprovalHandler(approval_store=storage),
                approval_policy=GatewayApprovalPolicy(),
                failover=FailoverPolicy(chain=["fake"], max_attempts=1),
                default_model=kwargs["model"],
                default_cwd=str(tmp_path),
                memory_scope=scope,
                pause_on_approval=True,
            )
            text = ""
            async for event in agent.run(
                RunRequest(
                    prompt=kwargs["prompt"],
                    session_id=kwargs["session_id"],
                    model=kwargs["model"],
                    max_steps=2,
                )
            ):
                if isinstance(event, Done) and event.final_message is not None:
                    text = event.final_message.content or ""
            return text
        finally:
            await storage.close()

    return run


def _add(runner, tmp_path, *target):
    created = runner.invoke(
        cli_main.app,
        [
            "scheduler",
            "add-prompt",
            "--cwd",
            str(tmp_path),
            "--prompt",
            "Append marker",
            "--provider",
            "fake",
            "--model",
            "original-model",
            "--every",
            "5m",
            "--json",
            *target,
        ],
    )
    assert created.exit_code == 0, created.output
    return json.loads(created.stdout)


def test_scheduled_remote_prompt_queues_then_resumes_original_session_and_retries_delivery(
    tmp_path, monkeypatch
):
    runner = CliRunner()
    calls, turns, sent = [], [], []
    monkeypatch.setattr(
        gateway_runtime, "_run_gateway_chat_turn", _fake_turn(tmp_path, calls, turns)
    )
    monkeypatch.setattr(gateway_runtime, "_load_hooks", lambda _: ())

    def send(**kwargs):
        sent.append(kwargs)
        if len(sent) == 1:
            raise OSError("transport offline")

    monkeypatch.setattr("harness.cli.gateway_hooks.send_whatsapp_text_message", send)
    created = _add(
        runner, tmp_path, "--transport", "whatsapp", "--user", "owner", "--thread", "bound-thread"
    )
    job_id = created["id"]
    assert "append-marker" not in job_id
    assert created["payload"]["prompt"] == "Append marker"
    store = SchedulerStore(root=tmp_path / ".harness/scheduler")

    def run_now():
        result = runner.invoke(
            cli_main.app, ["scheduler", "run-now", job_id, "--cwd", str(tmp_path), "--json"]
        )
        assert result.exit_code == 0, result.output
        return json.loads(result.stdout)

    first = run_now()
    assert first["result_status"] == "approval_required"
    artifact = json.loads((Path(first["record_dir"]) / "result.json").read_text())
    approval_id = artifact["approval_ids"][0]
    assert calls == [] and len(turns) == 1
    assert len(store.list_deliveries(status="pending")) == 1
    assert sent[0]["to"] == "bound-thread"
    sessions = GatewaySessionStore(root=default_gateway_root(tmp_path))
    sessions.get_or_create_session(
        transport="whatsapp", user_id="unrelated", thread_id="wrong-thread"
    )
    hooks = scheduler_commands._load_hooks(tmp_path)
    assert retry_scheduler_deliveries(store=store, hooks=hooks, force=True) == 1
    assert sent[1]["to"] == "bound-thread" and sent[1]["text"] == sent[0]["text"]
    assert calls == [] and len(turns) == 1

    waiting = run_now()
    assert waiting["result_status"] == "approval_required"
    assert len(turns) == 1 and len(sent) == 2
    approved = runner.invoke(
        cli_main.app,
        [
            "gateway",
            "receive",
            "--cwd",
            str(tmp_path),
            "--transport",
            "whatsapp",
            "--user",
            "owner",
            "--thread",
            "bound-thread",
            "--message",
            f"approve {approval_id}",
            "--json",
        ],
    )
    assert approved.exit_code == 0, approved.output
    assert "Scheduled marker complete" in json.loads(approved.stdout)["reply"]["text"]
    assert len(calls) == 1
    assert turns[0]["session_id"] == turns[1]["session_id"] == created["session_id"]
    assert turns[1]["model"] == "original-model"
    next_occurrence = run_now()
    assert next_occurrence["result_status"] == "approval_required"
    assert len(turns) == 3 and len(calls) == 1


def test_local_prompt_has_local_scope_and_cli_resume_instructions(tmp_path, monkeypatch):
    runner = CliRunner()
    calls, turns = [], []
    monkeypatch.setattr(
        gateway_runtime, "_run_gateway_chat_turn", _fake_turn(tmp_path, calls, turns)
    )
    monkeypatch.setattr(scheduler_commands, "_load_hooks", lambda _: ())
    created = _add(runner, tmp_path)
    result = runner.invoke(
        cli_main.app, ["scheduler", "run-now", created["id"], "--cwd", str(tmp_path), "--json"]
    )
    assert result.exit_code == 0, result.output
    record = json.loads(result.stdout)
    artifact = json.loads((Path(record["record_dir"]) / "result.json").read_text())
    assert artifact["local_only"] is True
    assert artifact["resume_commands"] == [
        f"harness approvals grant {artifact['approval_ids'][0]}",
        f"harness sessions resume {created['session_id']}",
    ]
    assert turns[0]["local_only"] is True

    async def load_scope():
        storage = SQLiteStorage(path=tmp_path / ".harness/harness.db")
        try:
            session = await storage.get(created["session_id"])
            assert session is not None
            return session.metadata["memory_scope"]
        finally:
            await storage.close()

    assert asyncio.run(load_scope()) == {"workspace": str(tmp_path), "user_id": None}

    class ReplayAction:
        name = "scheduled_action"
        description = "Append approved marker"
        approval: ApprovalDecision = "prompt"

        def __init__(self):
            self.parameters_schema: dict[str, Any] = {"type": "object", "properties": {}}

        async def __call__(self, call):
            calls.append(call.id)
            return ToolResult(tool_call_id=call.id, name=self.name, content="marker appended")

    class FinalAdapter:
        name = "fake"

        async def capabilities(self):
            return Capabilities(streaming=True, tool_use=True)

        async def cancel(self, session_id):
            pass

        async def stream(self, **kwargs):
            yield Done(
                final_message=Message(role="assistant", content="Approved scheduled work complete.")
            )

    def builder(**kwargs):
        registry = ToolRegistry()
        registry.register(ReplayAction())
        return Agent(
            adapters={"fake": FinalAdapter()},
            tools=registry,
            storage=kwargs["storage"],
            approval_store=kwargs["approval_store"],
            failover=FailoverPolicy(chain=["fake"], max_attempts=1),
            memory_scope=MemoryScope(workspace=str(tmp_path)),
            default_cwd=str(tmp_path),
            default_model="original-model",
        )

    monkeypatch.setattr(cli_main, "_build_agent", builder)
    db_path = tmp_path / ".harness/harness.db"
    granted = runner.invoke(
        cli_main.app, ["approvals", "grant", artifact["approval_ids"][0], "--db", str(db_path)]
    )
    assert granted.exit_code == 0, granted.output
    resumed = runner.invoke(
        cli_main.app,
        ["sessions", "resume", created["session_id"], "--cwd", str(tmp_path), "--db", str(db_path)],
    )
    assert resumed.exit_code == 0, resumed.output
    assert calls == ["scheduled-1"]


def test_scheduled_prompt_requires_target_identity_and_safe_provider(tmp_path):
    runner = CliRunner()
    for flags in (["--transport", "whatsapp"], ["--provider", "codex"]):
        result = runner.invoke(
            cli_main.app,
            [
                "scheduler",
                "add-prompt",
                "--cwd",
                str(tmp_path),
                "--prompt",
                "work",
                "--every",
                "1h",
                *flags,
            ],
        )
        assert result.exit_code == 2


def test_core_prompt_dispatch_requires_injected_executor(tmp_path):
    runner = CliRunner()
    created = _add(runner, tmp_path)
    store = SchedulerStore(root=tmp_path / ".harness/scheduler")
    failed = run_scheduler_job(store=store, job_id=created["id"])
    assert failed.status == "failed"
    assert "configured prompt executor" in failed.summary
    seen = []

    def executor(job):
        seen.append(job.id)
        return SchedulerExecutionResult(
            status="completed", stop_reason="done", record_dir="", notification_text="result"
        )

    completed = run_scheduler_job(store=store, job_id=created["id"], prompt_executor=executor)
    assert completed.result_status == "completed"
    assert seen == [created["id"]]


def test_one_shot_prompt_busy_conversation_is_rescheduled(tmp_path, monkeypatch):
    runner = CliRunner()
    monkeypatch.setattr(scheduler_commands, "_load_hooks", lambda _: ())
    created = runner.invoke(
        cli_main.app,
        [
            "scheduler",
            "add-prompt",
            "--cwd",
            str(tmp_path),
            "--prompt",
            "work",
            "--provider",
            "fake",
            "--model",
            "fake",
            "--at",
            "2000-01-01T00:00:00Z",
            "--json",
        ],
    )
    assert created.exit_code == 0, created.output
    job_id = json.loads(created.stdout)["id"]
    sessions = GatewaySessionStore(root=default_gateway_root(tmp_path))
    with sessions.conversation_lock(transport="local", user_id="cli", thread_id=job_id) as acquired:
        assert acquired
        result = runner.invoke(
            cli_main.app, ["scheduler", "run-now", job_id, "--cwd", str(tmp_path), "--json"]
        )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["result_stop_reason"] == "conversation_busy"
    job = SchedulerStore(root=tmp_path / ".harness/scheduler").load_job(job_id)
    assert job.status == "active" and job.next_run_at != "2000-01-01T00:00:00+00:00"
