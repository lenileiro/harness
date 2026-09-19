from __future__ import annotations

import json
import sys
from typing import Any

from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.cli import mission_commands
from harness.core import (
    Agent,
    ApprovalDecision,
    AutoApprove,
    Capabilities,
    Done,
    FailoverPolicy,
    Message,
    ToolCall,
    ToolRegistry,
    ToolResult,
)
from harness.core.mission.models import (
    Milestone,
    Mission,
    MissionFeature,
    ValidationAssertion,
    ValidationContract,
)
from harness.core.mission.store import MissionStore, default_mission_root
from harness.core.scheduler.models import default_scheduler_root
from harness.core.scheduler.store import SchedulerStore


def _seed(tmp_path):
    store = MissionStore(root=default_mission_root(tmp_path))
    store.add_mission(
        Mission(
            id="mission",
            title="Deliver",
            goal="Create target.txt",
            status="approved",
            current_milestone_id="milestone",
        )
    )
    store.add_milestone(
        Milestone(
            id="milestone", mission_id="mission", title="Deliver", summary="Create target", order=1
        )
    )
    store.add_feature(
        MissionFeature(
            id="feature",
            mission_id="mission",
            milestone_id="milestone",
            title="Create target",
            summary="Create target.txt",
            target_files=("target.txt",),
        )
    )
    store.add_contract(
        ValidationContract(
            id="contract",
            mission_id="mission",
            summary="Target exists",
            assertions=(
                ValidationAssertion(
                    id="assertion",
                    contract_id="contract",
                    title="Target exists",
                    description="Check target",
                    kind="test",
                    verification_method="Run assertion command",
                    covered_by_features=("feature",),
                ),
            ),
        )
    )
    return store


def _fake_agent_builder(tmp_path, captured):
    class CreateTool:
        name = "create_target"
        description = "Create the test target"
        approval: ApprovalDecision = "auto"

        def __init__(self):
            self.parameters_schema: dict[str, Any] = {"type": "object", "properties": {}}

        async def __call__(self, call):
            (tmp_path / "target.txt").write_text("created")
            return ToolResult(name=self.name, tool_call_id=call.id, content="Created target")

    class FakeAdapter:
        name = "fake"
        turns = 0

        async def capabilities(self):
            return Capabilities(streaming=True, tool_use=True)

        async def cancel(self, session_id):
            pass

        async def stream(self, **kwargs):
            self.turns += 1
            if self.turns == 1:
                yield Done(
                    final_message=Message(
                        role="assistant",
                        tool_calls=[ToolCall(id="call", name="create_target", arguments={})],
                    )
                )
            else:
                yield Done(final_message=Message(role="assistant", content="Created target"))

    def build(**kwargs):
        captured.append({"model": kwargs["model"], "yes": kwargs["yes"], "inbox": kwargs["inbox"]})
        registry = ToolRegistry()
        registry.register(CreateTool())
        return Agent(
            adapters={"fake": FakeAdapter()},
            tools=registry,
            storage=kwargs["storage"],
            failover=FailoverPolicy(chain=["fake"], max_attempts=1),
            approval_handler=AutoApprove(),
            default_model=kwargs["model"],
            default_cwd=str(tmp_path),
            max_repair_attempts=0,
        )

    return build


def _configure_assertion(runner, tmp_path):
    command = [
        sys.executable,
        "-c",
        "from pathlib import Path; assert Path('target.txt').read_text() == 'created'",
    ]
    configured = runner.invoke(
        cli_main.app,
        [
            "mission",
            "set-assertion-command",
            "--mission",
            "mission",
            "--assertion",
            "assertion",
            "--command",
            json.dumps(command),
            "--cwd",
            str(tmp_path),
        ],
    )
    assert configured.exit_code == 0, configured.output


def test_real_mission_cli_runs_worker_and_verification(tmp_path, monkeypatch):
    store = _seed(tmp_path)
    captured = []
    monkeypatch.setattr(mission_commands, "_build_agent", _fake_agent_builder(tmp_path, captured))
    runner = CliRunner()
    _configure_assertion(runner, tmp_path)
    result = runner.invoke(
        cli_main.app,
        [
            "mission",
            "run",
            "--mission",
            "mission",
            "--provider",
            "ollama",
            "--model",
            "chosen-model",
            "--yes",
            "--cwd",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "completed"
    assert store.load_feature("feature").status == "validated"
    assert captured == [{"model": "chosen-model", "yes": True, "inbox": False}]


def test_scheduled_agent_mission_uses_injected_cli_factory(tmp_path, monkeypatch):
    store = _seed(tmp_path)
    captured = []
    monkeypatch.setattr(mission_commands, "_build_agent", _fake_agent_builder(tmp_path, captured))
    runner = CliRunner()
    _configure_assertion(runner, tmp_path)
    added = runner.invoke(
        cli_main.app,
        [
            "scheduler",
            "add-mission",
            "--mission",
            "mission",
            "--execute",
            "--provider",
            "ollama",
            "--model",
            "scheduled-model",
            "--yes",
            "--at",
            "2020-01-01T00:00:00Z",
            "--cwd",
            str(tmp_path),
            "--json",
        ],
    )
    assert added.exit_code == 0, added.output
    job_id = json.loads(added.output)["id"]
    tick = runner.invoke(
        cli_main.app, ["scheduler", "start", "--once", "--cwd", str(tmp_path), "--json"]
    )
    assert tick.exit_code == 0, tick.output
    assert json.loads(tick.output)["jobs_executed"] == 1
    scheduler = SchedulerStore(root=default_scheduler_root(tmp_path))
    assert scheduler.load_job(job_id).last_status == "completed"
    assert store.load_feature("feature").status == "validated"
    assert captured == [{"model": "scheduled-model", "yes": True, "inbox": False}]
    deliveries = runner.invoke(
        cli_main.app, ["scheduler", "list-deliveries", "--cwd", str(tmp_path), "--json"]
    )
    assert deliveries.exit_code == 0
    assert all(item["status"] == "sent" for item in json.loads(deliveries.output))
