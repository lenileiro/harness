from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.cli import mission_commands
from harness.core.research_store import ResearchStore, default_research_root


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
