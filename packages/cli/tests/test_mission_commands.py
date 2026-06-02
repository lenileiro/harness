from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.cli import mission_commands
from harness.core import ResumeContract, default_scheduler_root
from harness.core.research_store import ResearchStore, default_research_root
from harness.core.scheduler_store import SchedulerStore
from harness.storage.sqlite import SQLiteStorage


def test_mission_create_show_and_list(tmp_path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "mission",
            "create",
            "--title",
            "Mission create demo",
            "--goal",
            "Add the first mission primitives to Harness.",
            "--planner-model",
            "gpt-planner",
            "--worker-model",
            "gpt-worker",
            "--budget-tokens",
            "5000",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    assert "Created mission" in created.stdout

    mission_root = tmp_path / ".harness" / "missions" / "missions"
    mission_id = next(mission_root.iterdir()).name

    listed = runner.invoke(
        cli_main.app,
        ["mission", "list", "--cwd", str(tmp_path)],
    )
    assert listed.exit_code == 0, listed.stdout
    assert "draft" in listed.stdout

    shown = runner.invoke(
        cli_main.app,
        ["mission", "show", mission_id, "--cwd", str(tmp_path)],
    )
    assert shown.exit_code == 0, shown.stdout
    assert "Mission create demo" in shown.stdout
    assert f"id={mission_id}" in shown.stdout
    assert "budget_tokens=5000" in shown.stdout
    assert "Add the first mission primitives to Harness." in shown.stdout


def test_mission_plan_and_approve_flow(tmp_path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "mission",
            "create",
            "--title",
            "Mission planning demo",
            "--goal",
            "Turn a mission into a structured plan.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    mission_id = next((tmp_path / ".harness" / "missions" / "missions").iterdir()).name

    planned = runner.invoke(
        cli_main.app,
        [
            "mission",
            "plan",
            "--mission",
            mission_id,
            "--contract-summary",
            "Assertions define correctness before implementation.",
            "--milestone",
            "m1|Milestone 1|Ship the first validated slice.",
            "--assertion",
            "a1|Login works|Primary login flow succeeds.|behavior|Run browser validation.",
            "--feature",
            "f1|m1|Implement login flow|Add the login screen and handler.|worker|app/login.py||a1",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert planned.exit_code == 0, planned.stdout
    assert "Planned mission" in planned.stdout
    assert "milestones=1" in planned.stdout
    assert "features=1" in planned.stdout
    assert "assertions=1" in planned.stdout

    listed_milestones = runner.invoke(
        cli_main.app,
        ["mission", "list-milestones", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert listed_milestones.exit_code == 0, listed_milestones.stdout
    assert "Milestone 1" in listed_milestones.stdout

    listed_features = runner.invoke(
        cli_main.app,
        ["mission", "list-features", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert listed_features.exit_code == 0, listed_features.stdout
    feature_root = tmp_path / ".harness" / "missions" / "features"
    feature_json = next(feature_root.iterdir()) / "feature.json"
    payload = json.loads(feature_json.read_text(encoding="utf-8"))
    assert payload["title"] == "Implement login flow"

    shown_contract = runner.invoke(
        cli_main.app,
        ["mission", "show-contract", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert shown_contract.exit_code == 0, shown_contract.stdout
    assert "Assertions define correctness before implementation." in shown_contract.stdout
    assert "Login works" in shown_contract.stdout

    approved = runner.invoke(
        cli_main.app,
        ["mission", "approve", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert approved.exit_code == 0, approved.stdout
    assert "Approved mission" in approved.stdout

    shown = runner.invoke(
        cli_main.app,
        ["mission", "show", mission_id, "--cwd", str(tmp_path)],
    )
    assert shown.exit_code == 0, shown.stdout
    assert "status=approved" in shown.stdout


def test_mission_launch_bootstraps_long_running_workflow(tmp_path) -> None:
    runner = CliRunner()

    launched = runner.invoke(
        cli_main.app,
        [
            "mission",
            "launch",
            "--title",
            "Long running demo",
            "--goal",
            "Keep pushing the checkout migration forward.",
            "--feature",
            "checkout-migration",
            "--phases",
            "plan,act,verify",
            "--every",
            "45m",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert launched.exit_code == 0, launched.stdout
    assert "Launched long-running workflow" in launched.stdout
    assert "task_ref=" in launched.stdout
    assert "mission_id=" in launched.stdout
    assert "scheduler_job_id=" in launched.stdout

    resume = ResumeContract.load(tmp_path / ".harness" / "resume.json")
    assert resume is not None
    assert resume.current == "checkout-migration"
    current = resume.current_feature()
    assert current is not None
    assert current.phases == ["plan", "act", "verify"]

    mission_root = tmp_path / ".harness" / "missions" / "missions"
    mission_id = next(mission_root.iterdir()).name
    mission_json = json.loads(
        (mission_root / mission_id / "mission.json").read_text(encoding="utf-8")
    )
    assert mission_json["status"] == "approved"
    assert mission_json["current_milestone_id"]

    features_root = tmp_path / ".harness" / "missions" / "features"
    feature_json = json.loads(
        (next(features_root.iterdir()) / "feature.json").read_text(encoding="utf-8")
    )
    assert feature_json["title"] == "checkout-migration"
    assert feature_json["status"] == "pending"

    contracts_root = tmp_path / ".harness" / "missions" / "contracts"
    contract_json = json.loads(
        (next(contracts_root.iterdir()) / "contract.json").read_text(encoding="utf-8")
    )
    assert contract_json["assertions"]

    scheduler_store = SchedulerStore(root=default_scheduler_root(tmp_path))
    jobs = scheduler_store.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].payload["mission_id"] == mission_id
    assert jobs[0].schedule.kind == "every"

    contract_file = tmp_path / ".harness" / "contracts" / "long-running-demo.json"
    assert contract_file.exists()
    tip_file = tmp_path / ".harness" / "tips.jsonl"
    assert tip_file.exists()
    assert "checkout-migration" in tip_file.read_text(encoding="utf-8")

    async def verify_storage() -> None:
        storage = SQLiteStorage(path=tmp_path / ".harness" / "harness.db")
        try:
            tasks = await storage.list_tasks(limit=10)
            assert len(tasks) == 1
            task = tasks[0]
            assert task.status == "in_progress"
            assert task.metadata["mission_id"] == mission_id
            assert task.metadata["resume_feature"] == "checkout-migration"
            assert task.metadata["env_contract"] == ".harness/contracts/long-running-demo.json"
            assert task.metadata["tips_path"] == ".harness/tips.jsonl"

            memories = await storage.list_memory(limit=10)
            assert len(memories) == 1
            assert "Long-running workflow" in memories[0].text
        finally:
            await storage.close()

    import asyncio

    asyncio.run(verify_storage())


def test_mission_launch_run_now_creates_scheduler_and_mission_run_artifacts(tmp_path) -> None:
    runner = CliRunner()

    launched = runner.invoke(
        cli_main.app,
        [
            "mission",
            "launch",
            "--title",
            "Run now demo",
            "--goal",
            "Create the workflow and immediately execute the first bounded burst.",
            "--feature",
            "run-now-feature",
            "--every",
            "10m",
            "--run-now",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert launched.exit_code == 0, launched.stdout
    assert "scheduler_run_id=" in launched.stdout
    assert "scheduler_run_status=paused" in launched.stdout

    scheduler_store = SchedulerStore(root=default_scheduler_root(tmp_path))
    records = scheduler_store.list_run_records()
    assert len(records) == 1
    assert records[0].result_status == "paused"
    assert records[0].result_stop_reason == "feature_dispatched"

    mission_runs_root = tmp_path / ".harness" / "missions" / "runs"
    run_dirs = list(mission_runs_root.iterdir())
    assert len(run_dirs) >= 1
    scheduled_runs = [
        json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        for run_dir in run_dirs
        if (run_dir / "run.json").is_file()
    ]
    run_json = next(
        item for item in scheduled_runs if item.get("stop_reason") == "feature_dispatched"
    )
    assert run_json["status"] == "paused"
    assert run_json["stop_reason"] == "feature_dispatched"
    assert run_json["steps_run"] >= 1


def test_mission_plan_supports_research_refs(tmp_path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "mission",
            "create",
            "--title",
            "Mission planning refs demo",
            "--goal",
            "Turn a mission into a structured plan with research refs.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    mission_id = next((tmp_path / ".harness" / "missions" / "missions").iterdir()).name

    planned = runner.invoke(
        cli_main.app,
        [
            "mission",
            "plan",
            "--mission",
            mission_id,
            "--contract-summary",
            "Assertions define correctness before implementation.",
            "--milestone",
            "m1|Milestone 1|Ship the first validated slice.",
            "--assertion",
            "a1|Login works|Primary login flow succeeds.|behavior|Run browser validation.",
            "--feature",
            "f1|m1|Implement login flow|Add the login screen and handler.|worker|app/login.py||a1|publication-1,hypothesis-2",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert planned.exit_code == 0, planned.stdout

    feature_root = tmp_path / ".harness" / "missions" / "features"
    feature_json = next(feature_root.iterdir()) / "feature.json"
    payload = json.loads(feature_json.read_text(encoding="utf-8"))
    assert payload["research_refs"] == ["publication-1", "hypothesis-2"]


def test_mission_draft_plan_can_apply_generated_plan(tmp_path, monkeypatch) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "mission",
            "create",
            "--title",
            "Mission drafted plan demo",
            "--goal",
            "Generate a plan from a high-level goal.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    mission_id = next((tmp_path / ".harness" / "missions" / "missions").iterdir()).name

    monkeypatch.setattr(
        mission_commands,
        "_generate_mission_plan_text",
        lambda **kwargs: json.dumps(
            {
                "contract_summary": "Assertions exist before coding.",
                "milestones": [
                    {"label": "m1", "title": "Milestone 1", "summary": "Ship the first slice."}
                ],
                "assertions": [
                    {
                        "label": "a1",
                        "title": "Slice works",
                        "description": "The slice validates cleanly.",
                        "kind": "contract",
                        "verification_method": "Inspect validation output.",
                    }
                ],
                "features": [
                    {
                        "label": "f1",
                        "milestone_label": "m1",
                        "title": "Implement slice",
                        "summary": "Build the first slice.",
                        "assigned_role": "worker",
                        "target_files": ["app/slice.py"],
                        "depends_on_labels": [],
                        "assertion_labels": ["a1"],
                        "research_refs": ["publication-seed"],
                    }
                ],
            }
        ),
    )

    drafted = runner.invoke(
        cli_main.app,
        [
            "mission",
            "draft-plan",
            "--mission",
            mission_id,
            "--apply",
            "--cwd",
            str(tmp_path),
            "--json",
        ],
    )
    assert drafted.exit_code == 0, drafted.stdout
    payload = json.loads(drafted.stdout)
    assert payload["draft"]["contract_summary"] == "Assertions exist before coding."
    assert payload["applied"]["features"] == 1

    shown_contract = runner.invoke(
        cli_main.app,
        ["mission", "show-contract", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert shown_contract.exit_code == 0, shown_contract.stdout
    assert "Slice works" in shown_contract.stdout


def test_mission_draft_plan_can_read_fixture_from_env(tmp_path, monkeypatch) -> None:
    runner = CliRunner()
    created = runner.invoke(
        cli_main.app,
        [
            "mission",
            "create",
            "--title",
            "Mission env draft demo",
            "--goal",
            "Use an env-backed draft plan.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    mission_id = next((tmp_path / ".harness" / "missions" / "missions").iterdir()).name
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(
        json.dumps(
            {
                "contract_summary": "Env fixture contract summary.",
                "milestones": [
                    {"label": "m1", "title": "Env milestone", "summary": "Ship the env slice."}
                ],
                "assertions": [
                    {
                        "label": "a1",
                        "title": "Env assertion",
                        "description": "The env draft should apply.",
                        "kind": "contract",
                        "verification_method": "Inspect stored artifacts.",
                    }
                ],
                "features": [
                    {
                        "label": "f1",
                        "milestone_label": "m1",
                        "title": "Env feature",
                        "summary": "Create the env-backed feature.",
                        "assigned_role": "worker",
                        "target_files": ["app/env.py"],
                        "depends_on_labels": [],
                        "assertion_labels": ["a1"],
                        "research_refs": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HARNESS_MISSION_PLAN_DRAFT_FILE", str(draft_path))

    drafted = runner.invoke(
        cli_main.app,
        [
            "mission",
            "draft-plan",
            "--mission",
            mission_id,
            "--apply",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert drafted.exit_code == 0, drafted.stdout
    assert "Applied drafted plan" in drafted.stdout


def test_mission_execute_and_complete_flow(tmp_path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "mission",
            "create",
            "--title",
            "Mission runtime demo",
            "--goal",
            "Dispatch and complete a mission feature.",
            "--planner-model",
            "gpt-planner",
            "--worker-model",
            "gpt-worker",
            "--validator-model",
            "gpt-validator",
            "--planner-brief",
            "Plan the next bounded step.",
            "--worker-brief",
            "Implement the assigned feature and leave a handoff.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    mission_id = next((tmp_path / ".harness" / "missions" / "missions").iterdir()).name

    planned = runner.invoke(
        cli_main.app,
        [
            "mission",
            "plan",
            "--mission",
            mission_id,
            "--contract-summary",
            "Runtime assertions are declared before coding.",
            "--milestone",
            "m1|Milestone 1|Ship the first runtime slice.",
            "--assertion",
            "a1|Dispatch works|The mission runtime should dispatch the first feature.|contract|Inspect persisted run output.",
            "--feature",
            "f1|m1|Implement runtime slice|Create the first executable mission slice.|worker|app/runtime.py||a1",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert planned.exit_code == 0, planned.stdout
    approved = runner.invoke(
        cli_main.app,
        ["mission", "approve", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert approved.exit_code == 0, approved.stdout

    executed = runner.invoke(
        cli_main.app,
        ["mission", "execute-next", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert executed.exit_code == 0, executed.stdout
    assert "dispatched" in executed.stdout
    feature_id = next(
        line.split("=", 1)[1].strip()
        for line in executed.stdout.splitlines()
        if line.startswith("feature_id=")
    )
    run_id = next(
        line.split("=", 1)[1].strip()
        for line in executed.stdout.splitlines()
        if line.startswith("run_id=")
    )
    handoff_id = next(
        line.split("=", 1)[1].strip()
        for line in executed.stdout.splitlines()
        if line.startswith("handoff_id=")
    )

    listed_runs = runner.invoke(
        cli_main.app,
        ["mission", "list-runs", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert listed_runs.exit_code == 0, listed_runs.stdout

    shown_run = runner.invoke(
        cli_main.app,
        ["mission", "show-run", run_id, "--cwd", str(tmp_path)],
    )
    assert shown_run.exit_code == 0, shown_run.stdout
    assert "Dispatch" in shown_run.stdout or "Dispatched feature" in shown_run.stdout
    assert "role_model=gpt-worker" in shown_run.stdout

    listed_handoffs = runner.invoke(
        cli_main.app,
        ["mission", "list-handoffs", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert listed_handoffs.exit_code == 0, listed_handoffs.stdout

    completed = runner.invoke(
        cli_main.app,
        [
            "mission",
            "complete-feature",
            "--mission",
            mission_id,
            "--feature",
            feature_id,
            "--completed-work",
            "Implemented the first executable mission slice.",
            "--next-recommendation",
            "Advance to validation next.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert completed.exit_code == 0, completed.stdout
    assert "completed" in completed.stdout or "recorded" in completed.stdout

    shown_handoff = runner.invoke(
        cli_main.app,
        ["mission", "show-handoff", handoff_id, "--cwd", str(tmp_path)],
    )
    assert shown_handoff.exit_code == 0, shown_handoff.stdout
    assert "Prepared the execution brief" in shown_handoff.stdout
    assert "role_model=gpt-planner" in shown_handoff.stdout

    validated = runner.invoke(
        cli_main.app,
        ["mission", "validate-milestone", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert validated.exit_code == 0, validated.stdout
    assert "completed" in validated.stdout or "passed" in validated.stdout

    shown_mission = runner.invoke(
        cli_main.app,
        ["mission", "show", mission_id, "--cwd", str(tmp_path)],
    )
    assert shown_mission.exit_code == 0, shown_mission.stdout
    assert "status=completed" in shown_mission.stdout
    assert "planner_model=gpt-planner" in shown_mission.stdout


def test_mission_create_honors_role_defaults_from_config(tmp_path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "mission-role-config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[mission_roles.planner]",
                'model = "gpt-planner"',
                'brief = "Plan before coding."',
                "",
                "[mission_roles.worker]",
                'model = "gpt-worker"',
                'brief = "Implement the assigned feature."',
            ]
        ),
        encoding="utf-8",
    )

    created = runner.invoke(
        cli_main.app,
        [
            "mission",
            "create",
            "--title",
            "Mission config demo",
            "--goal",
            "Use config-backed role defaults.",
            "--config",
            str(config_path),
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    mission_id = next((tmp_path / ".harness" / "missions" / "missions").iterdir()).name

    shown = runner.invoke(
        cli_main.app,
        ["mission", "show", mission_id, "--cwd", str(tmp_path)],
    )
    assert shown.exit_code == 0, shown.stdout
    assert "planner_model=gpt-planner" in shown.stdout
    assert "worker_model=gpt-worker" in shown.stdout
    assert "brief=Plan before coding." in shown.stdout


def test_mission_validation_failure_creates_findings_and_corrective_feature(tmp_path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "mission",
            "create",
            "--title",
            "Mission validator demo",
            "--goal",
            "Block a milestone until validation passes.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    mission_id = next((tmp_path / ".harness" / "missions" / "missions").iterdir()).name

    planned = runner.invoke(
        cli_main.app,
        [
            "mission",
            "plan",
            "--mission",
            mission_id,
            "--contract-summary",
            "Validation should create findings for incomplete features.",
            "--milestone",
            "m1|Milestone 1|Ship the validator slice.",
            "--assertion",
            "a1|Validator works|The milestone validator should gate completion.|contract|Inspect validator findings.",
            "--feature",
            "f1|m1|Implement validator slice|Create the first validator-controlled feature.|worker|app/validator.py||a1",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert planned.exit_code == 0, planned.stdout
    approved = runner.invoke(
        cli_main.app,
        ["mission", "approve", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert approved.exit_code == 0, approved.stdout
    dispatched = runner.invoke(
        cli_main.app,
        ["mission", "execute-next", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert dispatched.exit_code == 0, dispatched.stdout

    validated = runner.invoke(
        cli_main.app,
        ["mission", "validate-milestone", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert validated.exit_code == 0, validated.stdout
    assert "failed" in validated.stdout
    assert "scrutiny_run_id=" in validated.stdout
    assert "behavior_run_id=" in validated.stdout
    corrective_feature_id = next(
        line.split("=", 1)[1].strip()
        for line in validated.stdout.splitlines()
        if line.startswith("corrective_feature_id=")
    )

    findings = runner.invoke(
        cli_main.app,
        ["mission", "list-findings", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert findings.exit_code == 0, findings.stdout
    finding_json = (
        next((tmp_path / ".harness" / "missions" / "findings").iterdir()) / "finding.json"
    )
    finding_payload = json.loads(finding_json.read_text(encoding="utf-8"))
    assert finding_payload["severity"] == "error"
    assert finding_payload["mission_id"] == mission_id
    assert finding_payload["source"] in {"scrutiny-validator", "behavior-validator"}

    listed_features = runner.invoke(
        cli_main.app,
        ["mission", "list-features", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert listed_features.exit_code == 0, listed_features.stdout
    feature_json = next(
        path / "feature.json"
        for path in (tmp_path / ".harness" / "missions" / "features").iterdir()
        if path.name == corrective_feature_id
    )
    payload = json.loads(feature_json.read_text(encoding="utf-8"))
    assert payload["title"].startswith("Corrective:")


def test_mission_execute_milestone_and_burst_commands(tmp_path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "mission",
            "create",
            "--title",
            "Mission burst demo",
            "--goal",
            "Drive a mission through multiple milestones.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    mission_id = next((tmp_path / ".harness" / "missions" / "missions").iterdir()).name

    planned = runner.invoke(
        cli_main.app,
        [
            "mission",
            "plan",
            "--mission",
            mission_id,
            "--contract-summary",
            "Burst execution should validate both milestones.",
            "--milestone",
            "m1|Milestone 1|Ship the first burst slice.",
            "--milestone",
            "m2|Milestone 2|Ship the second burst slice.",
            "--assertion",
            "a1|First slice works|The first milestone should validate cleanly.|contract|Inspect validator output.",
            "--assertion",
            "a2|Second slice works|The second milestone should validate cleanly.|contract|Inspect validator output.",
            "--feature",
            "f1|m1|Implement first burst slice|Create the first slice.|worker|app/one.py||a1",
            "--feature",
            "f2|m2|Implement second burst slice|Create the second slice.|worker|app/two.py||a2",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert planned.exit_code == 0, planned.stdout
    approved = runner.invoke(
        cli_main.app,
        ["mission", "approve", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert approved.exit_code == 0, approved.stdout

    milestone_run = runner.invoke(
        cli_main.app,
        [
            "mission",
            "execute-milestone",
            "--mission",
            mission_id,
            "--max-steps",
            "10",
            "--auto-complete",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert milestone_run.exit_code == 0, milestone_run.stdout
    assert "completed" in milestone_run.stdout
    assert "validation_passed" in milestone_run.stdout

    burst_run = runner.invoke(
        cli_main.app,
        [
            "mission",
            "execute-burst",
            "--mission",
            mission_id,
            "--max-steps",
            "20",
            "--auto-complete",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert burst_run.exit_code == 0, burst_run.stdout
    assert "completed" in burst_run.stdout
    shown = runner.invoke(
        cli_main.app,
        ["mission", "show", mission_id, "--cwd", str(tmp_path)],
    )
    assert shown.exit_code == 0, shown.stdout
    assert "status=completed" in shown.stdout


def test_mission_report_commands(tmp_path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "mission",
            "create",
            "--title",
            "Mission report demo",
            "--goal",
            "Create a persisted mission summary report.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    mission_id = next((tmp_path / ".harness" / "missions" / "missions").iterdir()).name

    planned = runner.invoke(
        cli_main.app,
        [
            "mission",
            "plan",
            "--mission",
            mission_id,
            "--contract-summary",
            "Mission report assertions are declared before execution.",
            "--milestone",
            "m1|Milestone 1|Ship the report slice.",
            "--assertion",
            "a1|Report slice works|The mission summary should explain blocked work.|contract|Inspect report output.",
            "--feature",
            "f1|m1|Implement report slice|Create the report feature.|worker|app/report.py||a1",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert planned.exit_code == 0, planned.stdout
    approved = runner.invoke(
        cli_main.app,
        ["mission", "approve", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert approved.exit_code == 0, approved.stdout
    dispatched = runner.invoke(
        cli_main.app,
        ["mission", "execute-next", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert dispatched.exit_code == 0, dispatched.stdout
    validated = runner.invoke(
        cli_main.app,
        ["mission", "validate-milestone", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert validated.exit_code == 0, validated.stdout
    assert "failed" in validated.stdout

    summarized = runner.invoke(
        cli_main.app,
        ["mission", "summarize", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert summarized.exit_code == 0, summarized.stdout
    assert "Wrote mission summary" in summarized.stdout
    reports_root = tmp_path / ".harness" / "missions" / "reports"
    report_id = next(reports_root.iterdir()).name

    listed = runner.invoke(
        cli_main.app,
        ["mission", "list-reports", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert listed.exit_code == 0, listed.stdout

    shown = runner.invoke(
        cli_main.app,
        ["mission", "show-report", report_id, "--cwd", str(tmp_path)],
    )
    assert shown.exit_code == 0, shown.stdout
    assert "Mission report demo" in shown.stdout or "mission_id=" in shown.stdout
    assert "Role Profiles" in shown.stdout


def test_mission_schedule_once_honors_config(tmp_path) -> None:
    runner = CliRunner()
    config_path = tmp_path / "mission-config.toml"
    config_path.write_text(
        "\n".join(
            [
                "[mission_scheduler]",
                "max_steps = 20",
                "auto_complete = true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    created = runner.invoke(
        cli_main.app,
        [
            "mission",
            "create",
            "--title",
            "Mission schedule demo",
            "--goal",
            "Run a mission through schedule-once.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    mission_id = next((tmp_path / ".harness" / "missions" / "missions").iterdir()).name

    planned = runner.invoke(
        cli_main.app,
        [
            "mission",
            "plan",
            "--mission",
            mission_id,
            "--contract-summary",
            "Schedule-once should use config defaults.",
            "--milestone",
            "m1|Milestone 1|Ship the scheduled slice.",
            "--assertion",
            "a1|Scheduled slice works|The mission should complete in one scheduled burst.|contract|Inspect schedule output.",
            "--feature",
            "f1|m1|Implement scheduled slice|Create the scheduled feature.|worker|app/schedule.py||a1",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert planned.exit_code == 0, planned.stdout
    approved = runner.invoke(
        cli_main.app,
        ["mission", "approve", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert approved.exit_code == 0, approved.stdout

    scheduled = runner.invoke(
        cli_main.app,
        [
            "mission",
            "schedule-once",
            "--mission",
            mission_id,
            "--config",
            str(config_path),
            "--cwd",
            str(tmp_path),
            "--json",
        ],
    )
    assert scheduled.exit_code == 0, scheduled.stdout
    payload = json.loads(scheduled.stdout)
    assert payload["result"]["status"] == "completed"
    record_dir = Path(payload["record_dir"])
    assert (record_dir / "run.json").is_file()


def test_mission_can_emit_research_opportunity_and_candidate(tmp_path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "mission",
            "create",
            "--title",
            "Mission research bridge demo",
            "--goal",
            "Turn mission artifacts into research artifacts.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    mission_id = next((tmp_path / ".harness" / "missions" / "missions").iterdir()).name

    planned = runner.invoke(
        cli_main.app,
        [
            "mission",
            "plan",
            "--mission",
            mission_id,
            "--contract-summary",
            "Mission validation should emit follow-up research work when blocked.",
            "--milestone",
            "m1|Milestone 1|Ship the bridge slice.",
            "--assertion",
            "a1|Bridge works|The mission bridge should create research follow-up artifacts.|contract|Inspect research output.",
            "--feature",
            "f1|m1|Implement bridge slice|Create the first bridge feature.|worker|app/bridge.py||a1",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert planned.exit_code == 0, planned.stdout
    approved = runner.invoke(
        cli_main.app,
        ["mission", "approve", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert approved.exit_code == 0, approved.stdout

    executed = runner.invoke(
        cli_main.app,
        ["mission", "execute-next", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert executed.exit_code == 0, executed.stdout
    feature_id = next(
        line.split("=", 1)[1].strip()
        for line in executed.stdout.splitlines()
        if line.startswith("feature_id=")
    )

    validated = runner.invoke(
        cli_main.app,
        ["mission", "validate-milestone", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert validated.exit_code == 0, validated.stdout
    finding_id = next(
        path.name for path in (tmp_path / ".harness" / "missions" / "findings").iterdir()
    )

    created_opportunity = runner.invoke(
        cli_main.app,
        [
            "mission",
            "create-opportunity",
            "--mission",
            mission_id,
            "--finding",
            finding_id,
            "--title",
            "Mission validator follow-up",
            "--summary",
            "Convert the validator finding into durable research follow-up work.",
            "--cwd",
            str(tmp_path),
            "--json",
        ],
    )
    assert created_opportunity.exit_code == 0, created_opportunity.stdout
    opportunity_payload = json.loads(created_opportunity.stdout)
    assert opportunity_payload["opportunity"]["mission_id"] == mission_id

    completed = runner.invoke(
        cli_main.app,
        [
            "mission",
            "complete-feature",
            "--mission",
            mission_id,
            "--feature",
            feature_id,
            "--completed-work",
            "Implemented the bridge slice.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert completed.exit_code == 0, completed.stdout

    created_candidate = runner.invoke(
        cli_main.app,
        [
            "mission",
            "create-candidate",
            "--mission",
            mission_id,
            "--feature",
            feature_id,
            "--title",
            "Mission bridge candidate",
            "--summary",
            "Promote the bridge feature into the research promotion lane.",
            "--cwd",
            str(tmp_path),
            "--json",
        ],
    )
    assert created_candidate.exit_code == 0, created_candidate.stdout
    candidate_payload = json.loads(created_candidate.stdout)
    assert candidate_payload["candidate"]["mission_id"] == mission_id
    assert candidate_payload["candidate"]["mission_feature_ids"] == [feature_id]

    shown_candidate = runner.invoke(
        cli_main.app,
        [
            "research",
            "show-candidate",
            candidate_payload["candidate"]["id"],
            "--cwd",
            str(tmp_path),
        ],
    )
    assert shown_candidate.exit_code == 0, shown_candidate.stdout
    assert "Mission bridge candidate" in shown_candidate.stdout
    assert "Mission Features" in shown_candidate.stdout
    assert feature_id in shown_candidate.stdout

    research_store = ResearchStore(root=default_research_root(tmp_path))
    opportunities = research_store.list_opportunities()
    assert len(opportunities) == 1
    assert opportunities[0].mission_id == mission_id
    candidates = research_store.list_promotion_candidates()
    assert len(candidates) == 1
    assert candidates[0].mission_feature_ids == (feature_id,)


def test_scheduler_add_mission_run_now_and_list_runs(tmp_path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "mission",
            "create",
            "--title",
            "Mission scheduler demo",
            "--goal",
            "Run a scheduled mission job.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    mission_id = next((tmp_path / ".harness" / "missions" / "missions").iterdir()).name

    planned = runner.invoke(
        cli_main.app,
        [
            "mission",
            "plan",
            "--mission",
            mission_id,
            "--contract-summary",
            "Assertions define correctness before implementation.",
            "--milestone",
            "m1|Milestone 1|Ship a single validated slice.",
            "--assertion",
            "a1|Mission runs|The first mission feature can execute.|behavior|Run the mission loop.",
            "--feature",
            "f1|m1|Implement slice|Add the first mission slice.|worker|app/demo.py||a1",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert planned.exit_code == 0, planned.stdout

    approved = runner.invoke(
        cli_main.app,
        ["mission", "approve", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert approved.exit_code == 0, approved.stdout

    added = runner.invoke(
        cli_main.app,
        [
            "scheduler",
            "add-mission",
            "--mission",
            mission_id,
            "--at",
            "2026-05-26T12:00:00Z",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert added.exit_code == 0, added.stdout

    jobs_root = tmp_path / ".harness" / "scheduler" / "jobs"
    job_id = next(jobs_root.iterdir()).name

    listed = runner.invoke(
        cli_main.app,
        ["scheduler", "list", "--cwd", str(tmp_path), "--json"],
    )
    assert listed.exit_code == 0, listed.stdout
    listed_payload = json.loads(listed.stdout)
    assert listed_payload[0]["kind"] == "mission.schedule_once"

    ran = runner.invoke(
        cli_main.app,
        ["scheduler", "run-now", job_id, "--cwd", str(tmp_path)],
    )
    assert ran.exit_code == 0, ran.stdout
    assert "mission.schedule_once" in ran.stdout

    runs = runner.invoke(
        cli_main.app,
        ["scheduler", "list-runs", "--job", job_id, "--cwd", str(tmp_path), "--json"],
    )
    assert runs.exit_code == 0, runs.stdout
    runs_payload = json.loads(runs.stdout)
    assert runs_payload[0]["job_id"] == job_id


def test_scheduler_start_once_executes_research_job_with_pause_resume(tmp_path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "research",
            "create-opportunity",
            "--title",
            "Scheduler research demo",
            "--summary",
            "Use the scheduler to advance research once.",
            "--related-sections",
            "README.md",
            "--change-modes",
            "improve",
            "--theme",
            "scheduler",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout

    due_at = (datetime.now(UTC) - timedelta(minutes=1)).isoformat(timespec="seconds")
    added = runner.invoke(
        cli_main.app,
        [
            "scheduler",
            "add-research",
            "--at",
            due_at,
            "--cwd",
            str(tmp_path),
        ],
    )
    assert added.exit_code == 0, added.stdout

    jobs_root = tmp_path / ".harness" / "scheduler" / "jobs"
    job_id = next(jobs_root.iterdir()).name

    paused = runner.invoke(
        cli_main.app,
        ["scheduler", "pause", job_id, "--cwd", str(tmp_path)],
    )
    assert paused.exit_code == 0, paused.stdout

    resumed = runner.invoke(
        cli_main.app,
        ["scheduler", "resume", job_id, "--cwd", str(tmp_path)],
    )
    assert resumed.exit_code == 0, resumed.stdout

    started = runner.invoke(
        cli_main.app,
        ["scheduler", "start", "--once", "--cwd", str(tmp_path), "--json"],
    )
    assert started.exit_code == 0, started.stdout
    payload = json.loads(started.stdout)
    assert payload["jobs_executed"] == 1

    runs = runner.invoke(
        cli_main.app,
        ["scheduler", "list-runs", "--job", job_id, "--cwd", str(tmp_path), "--json"],
    )
    assert runs.exit_code == 0, runs.stdout
    runs_payload = json.loads(runs.stdout)
    assert runs_payload[0]["job_id"] == job_id
