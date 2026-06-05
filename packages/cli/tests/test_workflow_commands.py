from __future__ import annotations

# pyright: reportAttributeAccessIssue=false, reportIndexIssue=false
import asyncio
import json
import re
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import pytest
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.cli import workflow_commands
from harness.core import Done, Message
from harness.core.activity import ActivityEvent
from harness.core.dynamic_workflows import (
    EvidenceRequirement,
    WorkflowNode,
    WorkflowRun,
    WorkflowStore,
    create_default_workflow,
    default_workflow_root,
    evaluate_node_evidence,
    workflow_decision,
)
from harness.core.schemas import ToolCall


async def _fake_successful_activity(**_kwargs) -> list[ActivityEvent]:
    session_id = str(_kwargs.get("session_id") or "")
    if session_id.endswith("_plan") or "_plan_" in session_id:
        return []
    if re.search(r"_verify(?:_attempt\d+)?(?:_candidate\d+)?$", session_id):
        return [
            ActivityEvent(
                session_id="s",
                kind="tool_call.completed",
                data={
                    "name": "read_file",
                    "is_error": False,
                    "arguments": {"path": "result.txt"},
                    "content_preview": "supporting evidence",
                },
            ),
            ActivityEvent(
                session_id="s",
                kind="tool_call.completed",
                data={
                    "name": "verify_work",
                    "is_error": False,
                    "arguments": {"command": "test -f result.txt"},
                    "content_preview": "PASSED result.txt exists",
                },
            ),
        ]
    if re.search(r"_work(?:_attempt\d+)?(?:_candidate\d+)?$", session_id):
        return [
            ActivityEvent(
                session_id="s",
                kind="tool_call.completed",
                data={
                    "name": "list_dir",
                    "is_error": False,
                    "arguments": {"path": "."},
                    "content_preview": "result.txt",
                    "metadata": {"entries": 1},
                },
            ),
            ActivityEvent(
                session_id="s",
                kind="tool_call.completed",
                data={
                    "name": "write_file",
                    "is_error": False,
                    "arguments": {"path": "result.txt"},
                    "content_preview": "wrote result.txt",
                },
            ),
            ActivityEvent(
                session_id="s",
                kind="tool_call.completed",
                data={
                    "name": "verify_work",
                    "is_error": False,
                    "arguments": {"command": "test -f result.txt"},
                    "content_preview": "PASSED result.txt exists",
                },
            ),
        ]
    return [
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "read_file",
                "is_error": False,
                "content_preview": "supporting evidence",
            },
        )
    ]


def test_activity_error_message_uses_agent_run_failure() -> None:
    activity = [
        ActivityEvent(
            session_id="s",
            kind="agent_run.failed",
            data={
                "error": "exceeded max_steps=8 without final answer",
                "kind": "internal",
            },
        )
    ]

    assert (
        workflow_commands._activity_error_message(activity)
        == "internal: exceeded max_steps=8 without final answer"
    )


def test_workflow_start_plan_only_writes_durable_run(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        cli_main.app,
        [
            "workflow",
            "start",
            "--cwd",
            str(tmp_path),
            "--goal",
            "Audit checkout for unsafe redirects.",
            "--plan-only",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "pending"
    assert [node["id"] for node in payload["nodes"]] == [
        "plan",
        "research",
        "work",
        "verify",
        "review",
        "refute",
        "merge",
    ]
    stored = WorkflowStore(root=default_workflow_root(tmp_path)).load_run(payload["id"])
    assert stored.goal == "Audit checkout for unsafe redirects."


def test_workflow_resume_runs_nodes_with_dependency_context(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    created = runner.invoke(
        cli_main.app,
        [
            "workflow",
            "start",
            "--cwd",
            str(tmp_path),
            "--goal",
            "Check a tiny workflow.",
            "--plan-only",
            "--json",
        ],
    )
    assert created.exit_code == 0, created.stdout
    workflow_id = json.loads(created.stdout)["id"]
    prompts: list[str] = []

    async def _fake_node(**kwargs):
        node = kwargs["node"]
        run = kwargs["run"]
        prompt = workflow_commands._node_prompt(run, node)
        prompts.append(prompt)
        if node.kind in {"review", "refute"}:
            return f"{node.id} result\nWORKFLOW_DECISION: pass"
        return f"{node.id} result"

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _fake_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _fake_successful_activity)

    result = runner.invoke(
        cli_main.app,
        [
            "workflow",
            "resume",
            workflow_id,
            "--cwd",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["status"] == "completed"
    assert "Completed with workflow evidence" in payload["final_report"]
    assert "refute result" in payload["final_report"]
    assert "plan (completed): plan result" in prompts[1]
    assert "Goal authority:" in prompts[1]
    assert all(node["status"] == "completed" for node in payload["nodes"])


def test_workflow_report_and_cancel(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()

    async def _fake_node(**kwargs):
        if kwargs["node"].kind in {"review", "refute"}:
            return f"{kwargs['node'].id} result\nWORKFLOW_DECISION: pass"
        return f"{kwargs['node'].id} result"

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _fake_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _fake_successful_activity)
    created = runner.invoke(
        cli_main.app,
        [
            "workflow",
            "start",
            "--cwd",
            str(tmp_path),
            "--goal",
            "Summarize something.",
            "--json",
        ],
    )
    assert created.exit_code == 0, created.stdout
    workflow_id = json.loads(created.stdout)["id"]

    report = runner.invoke(
        cli_main.app,
        ["workflow", "report", workflow_id, "--cwd", str(tmp_path)],
    )
    assert report.exit_code == 0, report.stdout
    assert "Completed with workflow evidence" in report.stdout
    assert "refute result" in report.stdout

    cancelled = runner.invoke(
        cli_main.app,
        ["workflow", "cancel", workflow_id, "--cwd", str(tmp_path), "--json"],
    )
    assert cancelled.exit_code == 0, cancelled.stdout
    assert json.loads(cancelled.stdout)["status"] == "cancelled"


def test_workflow_dynamic_planner_accepts_valid_plan(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()

    async def _fake_planner(**_kwargs):
        return json.dumps(
            {
                "title": "Dynamic test plan",
                "nodes": [
                    {
                        "id": "inspect",
                        "title": "Inspect",
                        "kind": "research",
                        "prompt": "Inspect the repo.",
                    },
                    {
                        "id": "change",
                        "title": "Change",
                        "kind": "work",
                        "depends_on": ["inspect"],
                        "prompt": "Make the change.",
                        "allow_mutation": True,
                    },
                ],
            }
        )

    monkeypatch.setattr(workflow_commands, "_run_dynamic_planner", _fake_planner)

    result = runner.invoke(
        cli_main.app,
        [
            "workflow",
            "start",
            "--cwd",
            str(tmp_path),
            "--goal",
            "Use a generated plan.",
            "--planner",
            "dynamic",
            "--plan-only",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["title"] == "Dynamic test plan"
    assert [node["id"] for node in payload["nodes"]] == ["inspect", "change", "merge"]
    assert payload["metadata"]["planner"] == "dynamic"


def test_workflow_dynamic_planner_falls_back_on_invalid_plan(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()

    async def _fake_planner(**_kwargs):
        return "not json"

    monkeypatch.setattr(workflow_commands, "_run_dynamic_planner", _fake_planner)

    result = runner.invoke(
        cli_main.app,
        [
            "workflow",
            "start",
            "--cwd",
            str(tmp_path),
            "--goal",
            "Use fallback plan.",
            "--planner",
            "dynamic",
            "--plan-only",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["metadata"]["planner"] == "static-fallback"
    assert "planner_fallback_reason" in payload["metadata"]
    assert [node["id"] for node in payload["nodes"]] == [
        "plan",
        "research",
        "work",
        "verify",
        "review",
        "refute",
        "merge",
    ]


def test_workflow_scheduler_runs_ready_nodes_concurrently(tmp_path: Path, monkeypatch) -> None:
    store = WorkflowStore(root=default_workflow_root(tmp_path))
    run = WorkflowRun(
        id=store.new_id("Concurrent"),
        title="Concurrent",
        goal="run independent nodes",
        nodes=(
            WorkflowNode(
                id="a",
                title="A",
                kind="research",
                role="researcher",
                prompt="a",
                expected_evidence=(EvidenceRequirement(kind="result_nonempty"),),
            ),
            WorkflowNode(
                id="b",
                title="B",
                kind="research",
                role="researcher",
                prompt="b",
                expected_evidence=(EvidenceRequirement(kind="result_nonempty"),),
            ),
            WorkflowNode(
                id="merge",
                title="Merge",
                kind="merge",
                role="merger",
                prompt="merge",
                depends_on=("a", "b"),
                expected_evidence=(EvidenceRequirement(kind="result_nonempty"),),
            ),
        ),
    )
    store.add_run(run)

    async def _fake_node(**kwargs):
        if kwargs["node"].id in {"a", "b"}:
            await asyncio.sleep(0.1)
        return f"{kwargs['node'].id} result"

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _fake_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _fake_successful_activity)

    started = time.monotonic()
    finished = asyncio.run(
        workflow_commands.run_workflow(
            cwd=tmp_path,
            store=store,
            workflow_id=run.id,
            concurrency=2,
        )
    )
    elapsed = time.monotonic() - started

    assert finished.status == "completed"
    assert elapsed < 0.2
    event_kinds = [event.kind for event in store.list_events(run.id)]
    assert "node.queued" in event_kinds
    assert "workflow.completed" in event_kinds


def test_workflow_subagent_consensus_selects_agreed_candidate() -> None:
    candidates = [
        workflow_commands.NodeCandidate(0, "same answer", [], "s0"),
        workflow_commands.NodeCandidate(1, "same answer", [], "s1"),
        workflow_commands.NodeCandidate(2, "different answer with more text", [], "s2"),
    ]

    selected, consensus = workflow_commands._choose_candidate(
        candidates,
        consensus_threshold=2,
    )

    assert selected.result == "same answer"
    assert consensus["met_threshold"] is True
    assert consensus["best_agreement"] == 2


def test_workflow_subagents_do_not_parallelize_mutating_nodes() -> None:
    mutating = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="work",
        allow_mutation=True,
    )
    read_only = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        role="researcher",
        prompt="research",
        allow_mutation=False,
    )

    assert workflow_commands._node_candidate_count(mutating, 3) == 1
    assert workflow_commands._node_candidate_count(read_only, 3) == 3


def test_workflow_review_decision_requirement_is_explicit() -> None:
    node = WorkflowNode(
        id="review",
        title="Review",
        kind="review",
        role="reviewer",
        prompt="review",
        expected_evidence=(EvidenceRequirement(kind="review_decision"),),
    )

    text = workflow_commands._evidence_requirements_text(node)

    assert "WORKFLOW_DECISION: pass" in text
    assert "WORKFLOW_DECISION: retry" in text


def test_workflow_objective_status_requirement_is_final_response_only() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="work",
        expected_evidence=(EvidenceRequirement(kind="objective_status"),),
    )

    text = workflow_commands._evidence_requirements_text(node)
    system_prompt = workflow_commands._workflow_system_prompt("base", node)

    assert '{"status":"pass"}' in text
    assert "final assistant response" in text
    assert "do not emit this flag via shell or tool output" in text
    assert "objective status gate" in system_prompt
    assert "not in a shell command or tool output" in system_prompt


def test_workflow_environment_check_guidance_uses_missing_tool_evidence() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="work",
        expected_evidence=(EvidenceRequirement(kind="environment_checked"),),
    )

    text = workflow_commands._evidence_requirements_text(node)
    system_prompt = workflow_commands._workflow_system_prompt("base", node)

    assert "missing-tool evidence" in text
    assert "research one" in text
    assert "If a tool is missing" in system_prompt
    assert "After any install or update, re-check" in system_prompt


def test_workflow_review_prompt_demands_decision_only() -> None:
    run = WorkflowRun(id="wf", title="WF", goal="goal")

    by_id = {
        node.id: node
        for node in create_default_workflow(
            workflow_id=run.id,
            title=run.title,
            goal=run.goal,
        ).nodes
    }

    assert "Output exactly one line" in by_id["review"].prompt
    assert "WORKFLOW_DECISION: pass" in by_id["review"].prompt
    assert "WORKFLOW_DECISION: retry" in by_id["review"].prompt


def test_workflow_system_prompt_scopes_shell_guidance_by_role() -> None:
    planner = WorkflowNode(id="plan", title="Plan", kind="plan", role="planner", prompt="plan")
    worker = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="work",
        expected_evidence=(EvidenceRequirement(kind="verify_work_if_state_changed"),),
    )
    verifier = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        role="verifier",
        prompt="verify",
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )

    planner_prompt = workflow_commands._workflow_system_prompt("base", planner)
    worker_prompt = workflow_commands._workflow_system_prompt("base", worker)
    verifier_prompt = workflow_commands._workflow_system_prompt("base", verifier)

    assert "do not execute commands or claim completion" in planner_prompt
    assert "using the Goal as the source of truth" in planner_prompt
    assert "do not invent shell-like tools" in planner_prompt
    assert "For shell commands use `shell`" in worker_prompt
    assert "recomputes from the source inputs" in worker_prompt
    assert "do not invent shell-like tools" in verifier_prompt


def test_workflow_review_roles_can_decide_from_existing_evidence() -> None:
    reviewer = WorkflowNode(
        id="review",
        title="Review",
        kind="review",
        role="reviewer",
        prompt="review",
    )
    refuter = WorkflowNode(
        id="refute",
        title="Refute",
        kind="refute",
        role="refuter",
        prompt="refute",
    )

    assert workflow_commands._policy_for_node(reviewer).require_tools is False
    assert workflow_commands._policy_for_node(refuter).require_tools is False
    assert workflow_commands._policy_for_node(reviewer).tool_include == set()
    assert workflow_commands._policy_for_node(refuter).tool_include == set()
    assert "WORKFLOW_DECISION: pass" in workflow_commands._policy_for_node(reviewer).instructions
    assert "WORKFLOW_DECISION: retry" in workflow_commands._policy_for_node(refuter).instructions
    assert (
        "Do not add new claims"
        in workflow_commands._policy_for_node(
            WorkflowNode(
                id="merge",
                title="Merge",
                kind="merge",
                role="merger",
                prompt="merge",
            )
        ).instructions
    )


def test_workflow_verify_policy_can_call_verify_work() -> None:
    verifier = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        role="verifier",
        prompt="verify",
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )
    worker = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="work",
        expected_evidence=(EvidenceRequirement(kind="verify_work_if_state_changed"),),
    )

    verifier_policy = workflow_commands._policy_for_node(verifier)
    verifier_prompt = workflow_commands._workflow_system_prompt("base", verifier)
    worker_prompt = workflow_commands._workflow_system_prompt("base", worker)
    evidence_text = workflow_commands._evidence_requirements_text(worker)

    assert verifier_policy.tool_include is not None
    assert "verify_work" in verifier_policy.tool_include
    assert "shell" not in verifier_policy.tool_include
    assert "call verify_work after the final mutation" in evidence_text
    assert "verify after the final mutation" in worker_prompt
    assert "clean command environment" in worker_prompt
    assert "unsupported command syntax" in worker_prompt
    assert "Do not simulate tool calls or tool outputs" in worker_prompt
    assert "do not create temporary files or redirect output" in verifier_prompt


def test_workflow_node_shell_uses_clean_target_environment(tmp_path: Path) -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="work",
        allow_mutation=True,
    )

    registry = workflow_commands._build_tools_for_node(node)(tmp_path, include={"shell"})

    assert registry.get("shell").clean_env is True
    assert registry.get("shell").pipefail is True


def test_workflow_research_node_can_probe_environment_with_shell(tmp_path: Path) -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        role="researcher",
        prompt="inspect environment",
    )

    registry = workflow_commands._build_tools_for_node(node)(tmp_path)

    assert registry.has("shell")
    assert registry.get("shell").clean_env is True
    assert registry.get("shell").pipefail is True
    assert "local clock" in registry.get("shell").description
    assert "timezone-aware clock conversions" in registry.get("shell").description
    assert not registry.has("write_file")


@pytest.mark.asyncio
async def test_workflow_research_shell_refuses_mutation(tmp_path: Path) -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        role="researcher",
        prompt="inspect environment",
    )
    registry = workflow_commands._build_tools_for_node(node)(tmp_path)
    shell = registry.get("shell")

    refused = await shell(
        ToolCall(
            id="s1",
            name="shell",
            arguments={"command": "missing-tool input.json > transformed.json"},
        )
    )
    allowed = await shell(
        ToolCall(
            id="s2",
            name="shell",
            arguments={"command": "command -v definitely-not-present-harness-tool"},
        )
    )
    version_probe = await shell(
        ToolCall(
            id="s3",
            name="shell",
            arguments={"command": "python3 --version"},
        )
    )
    shell_probe = await shell(
        ToolCall(
            id="s4",
            name="shell",
            arguments={"command": "bash -c 'echo ok && uname -a'"},
        )
    )
    timezone_probe = await shell(
        ToolCall(
            id="s4-timezone",
            name="shell",
            arguments={"command": "TZ='America/New_York' date '+%Z'"},
        )
    )
    env_mutation = await shell(
        ToolCall(
            id="s4-mutation",
            name="shell",
            arguments={"command": "PATH=/tmp rm transformed.json"},
        )
    )
    commented_probe = await shell(
        ToolCall(
            id="s5",
            name="shell",
            arguments={
                "command": (
                    "# Check whether tools are installed before work\n"
                    "which python3 || echo 'python3 not found'\n"
                    "which definitely-not-present-harness-tool || echo 'tool not found'\n"
                )
            },
        )
    )

    assert refused.is_error is True
    assert "refused read-only shell command" in refused.content
    assert not (tmp_path / "transformed.json").exists()
    assert allowed.is_error is True
    assert allowed.metadata is not None
    assert allowed.metadata["clean_env"] is True
    assert allowed.metadata["pipefail"] is True
    assert not (version_probe.metadata or {}).get("read_only_shell_refused")
    assert not (shell_probe.metadata or {}).get("read_only_shell_refused")
    assert not (timezone_probe.metadata or {}).get("read_only_shell_refused")
    assert env_mutation.is_error is True
    assert (env_mutation.metadata or {}).get("read_only_shell_refused") is True
    assert commented_probe.is_error is True
    assert not (commented_probe.metadata or {}).get("read_only_shell_refused")
    assert (commented_probe.metadata or {}).get("masked_failure_exit_status") is True
    assert "tool not found" in commented_probe.content


@pytest.mark.asyncio
async def test_verify_node_auto_checks_source_derived_report(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks"
    (tasks / "a").mkdir(parents=True)
    (tasks / "b").mkdir(parents=True)
    (tasks / "a" / "task.toml").write_text('language = "python"\n', encoding="utf-8")
    (tasks / "b" / "task.toml").write_text('language = "go"\n', encoding="utf-8")
    (tmp_path / "REPORT.md").write_text(
        "Total Task Count: 2\nPython: 1\nGo: 1\n",
        encoding="utf-8",
    )
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="write report",
        status="completed",
        metadata={
            "activity_summary": {
                "changed_paths": ["REPORT.md"],
                "read_paths": ["tasks", "tasks/*/task.toml"],
            }
        },
    )
    verify = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify report",
        depends_on=("work",),
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create report.",
        nodes=(work, verify),
    )
    candidate = workflow_commands.NodeCandidate(
        index=0,
        result="",
        activity=[],
        session_id="verify-session",
    )

    updated = await workflow_commands._auto_verify_candidate_if_needed(
        cwd=tmp_path,
        run=run,
        node=verify,
        candidate=candidate,
    )

    completed = [
        event
        for event in updated.activity
        if event.kind == "tool_call.completed" and event.data.get("name") == "verify_work"
    ]
    assert completed
    assert completed[-1].data["is_error"] is False
    assert "source-artifact checks" in completed[-1].data["content_preview"]


def test_workflow_retry_attempts_use_distinct_session_ids() -> None:
    run = WorkflowRun(id="workflow-test", title="Workflow Test", goal="test")
    node = WorkflowNode(id="work", title="Work", kind="work", prompt="work")

    assert workflow_commands._node_session_id(run, replace(node, attempts=1)) == (
        "workflow_workflow_test_work"
    )
    assert workflow_commands._node_session_id(run, replace(node, attempts=2)) == (
        "workflow_workflow_test_work_attempt2"
    )
    assert (
        workflow_commands._node_session_id(
            run,
            replace(node, attempts=2),
            candidate_index=1,
        )
        == "workflow_workflow_test_work_attempt2_candidate1"
    )


def test_failed_workflow_report_does_not_publish_failed_merge_result() -> None:
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="test",
        nodes=(
            WorkflowNode(
                id="merge",
                title="Merge",
                kind="merge",
                prompt="merge",
                status="failed",
                result="Hallucinated success report.",
                error="unsupported path claim",
            ),
        ),
    )

    report = workflow_commands._failure_report_for_run(run)

    assert "Workflow failed before a verified final report" in report
    assert "unsupported path claim" in report
    assert "Hallucinated success report" not in report


def test_failure_report_treats_retry_decision_as_unverified() -> None:
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="test",
        nodes=(
            WorkflowNode(
                id="refute",
                title="Refute",
                kind="refute",
                prompt="refute",
                status="completed",
                result="WORKFLOW_DECISION: retry",
            ),
            WorkflowNode(
                id="merge",
                title="Merge",
                kind="merge",
                prompt="merge",
                status="completed",
                result="Should not be final.",
            ),
        ),
    )

    report = workflow_commands._failure_report_for_run(run)

    assert "Workflow failed before a verified final report" in report
    assert "requested another review round" in report
    assert "Should not be final" not in report


def test_workflow_node_prompt_includes_previous_failure_history() -> None:
    node = WorkflowNode(
        id="review",
        title="Review",
        kind="review",
        role="reviewer",
        prompt="review",
        metadata={
            "failure_history": [
                {
                    "attempt": 1,
                    "error": "missing WORKFLOW_DECISION: pass|retry",
                    "result": "The test passes.",
                }
            ]
        },
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="test",
        nodes=(node,),
    )

    prompt = workflow_commands._node_prompt(run, node)

    assert "Previous failed attempts for this node" in prompt
    assert "missing WORKFLOW_DECISION" in prompt
    assert "The test passes." in prompt


def test_workflow_node_prompt_includes_retry_evidence_history() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="implement",
        metadata={
            "failure_history": [
                {
                    "attempt": 1,
                    "error": "1 tool call(s) returned errors",
                    "result": "Added helper module.",
                    "activity_summary": {
                        "state_changed": True,
                        "verify_work_passed": False,
                    },
                    "failed_tools": [
                        "shell(pytest tests/test_feature.py): exit_code: 1 assertion failed"
                    ],
                }
            ]
        },
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="test",
        nodes=(node,),
    )

    prompt = workflow_commands._node_prompt(run, node)

    assert "state_changed=True" in prompt
    assert "pytest tests/test_feature.py" in prompt
    assert "Continue from the current files" in prompt


def test_workflow_node_prompt_includes_environment_observations() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="implement",
        allow_mutation=True,
        metadata={
            "failure_history": [
                {
                    "attempt": 1,
                    "error": "environment setup was not rechecked after installation or update",
                    "result": "Tried the local tool.",
                    "activity_summary": {
                        "state_changed": True,
                        "verify_work_passed": False,
                        "environment_observations": [
                            "missing tool: sample-tool --version: command not found"
                        ],
                    },
                    "failed_tools": [
                        "shell(sample-tool --bad-flag): exit_code: 2 stderr: unknown option"
                    ],
                }
            ]
        },
    )
    run = WorkflowRun(id="workflow-test", title="Workflow Test", goal="test", nodes=(node,))

    prompt = workflow_commands._node_prompt(run, node)

    assert "environment observations" in prompt
    assert "sample-tool --version" in prompt
    assert "use web_search or fetch_url" in prompt
    assert "do not repeat failed command syntax" in prompt
    assert "Before the next mutation" in prompt


def test_workflow_node_prompt_guides_read_only_retry_after_mutation() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        role="verifier",
        prompt="verify",
        allow_mutation=False,
        metadata={
            "failure_history": [
                {
                    "attempt": 1,
                    "error": "read-only node changed workspace state",
                    "result": "Tried to verify with a temporary file.",
                    "activity_summary": {
                        "state_changed": True,
                        "verify_work_passed": False,
                    },
                    "failed_tools": [
                        "verify_work(command > expected.txt && diff output.txt expected.txt)"
                    ],
                }
            ]
        },
    )
    run = WorkflowRun(id="workflow-test", title="Workflow Test", goal="test", nodes=(node,))

    prompt = workflow_commands._node_prompt(run, node)

    assert "this role is read-only" in prompt
    assert "Do not use output redirection" in prompt
    assert "process substitution" in prompt


def test_workflow_node_prompt_guides_source_artifact_retry() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        role="verifier",
        prompt="verify",
        metadata={
            "failure_history": [
                {
                    "attempt": 1,
                    "error": (
                        "passing verify_work did not compare dependency artifacts with "
                        "source inputs; recompute from source inputs or run a broader "
                        "test command"
                    ),
                    "result": "Computed source facts only.",
                    "activity_summary": {
                        "state_changed": False,
                        "verify_work_passed": True,
                    },
                }
            ]
        },
    )
    run = WorkflowRun(id="workflow-test", title="Workflow Test", goal="test", nodes=(node,))

    prompt = workflow_commands._node_prompt(run, node)

    assert "must read both the source input file(s)" in prompt
    assert "generated artifact file(s)" in prompt
    assert "same command" in prompt


def test_workflow_node_prompt_status_retry_does_not_redo_verified_work() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="implement",
        allow_mutation=True,
        metadata={
            "failure_history": [
                {
                    "attempt": 1,
                    "error": "objective status is missing",
                    "result": "verify_work passed",
                    "activity_summary": {
                        "state_changed": True,
                        "verify_work_passed": True,
                    },
                }
            ]
        },
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="test",
        nodes=(node,),
    )

    prompt = workflow_commands._node_prompt(run, node)

    assert "produced verified evidence but omitted" in prompt
    assert "Do not redo the work" in prompt
    assert '{"status":"pass"}' in prompt


def test_workflow_node_prompt_warns_after_inspection_only_retry() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="implement",
        allow_mutation=True,
        metadata={
            "failure_history": [
                {
                    "attempt": 1,
                    "error": "required evidence missing: no state change observed",
                    "result": "Inspected files and found the likely module.",
                    "activity_summary": {
                        "state_changed": False,
                        "verify_work_passed": False,
                    },
                }
            ]
        },
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="test",
        nodes=(node,),
    )

    prompt = workflow_commands._node_prompt(run, node)

    assert "state_changed=False" in prompt
    assert "avoid repeating inspection-only steps" in prompt
    assert "make a concrete change toward the Goal" in prompt


def test_workflow_status_only_retry_needs_small_runtime_budget() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="work",
        error="objective status is missing",
        metadata={"completion_timeout_seconds": 120},
    )

    assert workflow_commands._retry_runtime_budget_seconds(node) == 5.0


def test_workflow_adds_objective_status_from_verified_tool_evidence() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="create result.txt",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="result_nonempty"),
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="no_failed_tools"),
            EvidenceRequirement(kind="files_changed"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
            EvidenceRequirement(kind="objective_status"),
        ),
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="Create result.txt.",
        nodes=(node,),
    )
    candidate = workflow_commands.NodeCandidate(
        index=0,
        result="Created result.txt and verified it exists.",
        activity=[
            ActivityEvent(
                session_id="s",
                kind="tool_call.completed",
                data={
                    "name": "write_file",
                    "is_error": False,
                    "arguments": {"path": "result.txt", "content": "ok"},
                    "content_preview": "wrote result.txt",
                },
            ),
            ActivityEvent(
                session_id="s",
                kind="tool_call.completed",
                data={
                    "name": "verify_work",
                    "is_error": False,
                    "arguments": {"command": "test -f result.txt"},
                    "content_preview": "PASSED",
                },
            ),
        ],
        session_id="s",
    )

    closed = workflow_commands._candidate_with_verified_objective_status_if_needed(
        node=node,
        run=run,
        candidate=candidate,
    )

    assert workflow_decision(closed.result) == "pass"
    ok, results = evaluate_node_evidence(
        node=node,
        result=closed.result,
        activity=closed.activity,
        run=run,
    )
    assert ok is True
    assert [item.status for item in results][-1] == "passed"


def test_workflow_does_not_add_objective_status_when_verified_evidence_fails() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="create result.txt",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="result_nonempty"),
            EvidenceRequirement(kind="no_failed_tools"),
            EvidenceRequirement(kind="files_changed"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
            EvidenceRequirement(kind="objective_status"),
        ),
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="Create result.txt.",
        nodes=(node,),
    )
    candidate = workflow_commands.NodeCandidate(
        index=0,
        result="Created result.txt.",
        activity=[
            ActivityEvent(
                session_id="s",
                kind="tool_call.completed",
                data={
                    "name": "write_file",
                    "is_error": False,
                    "arguments": {"path": "result.txt", "content": "ok"},
                    "content_preview": "wrote result.txt",
                },
            ),
            ActivityEvent(
                session_id="s",
                kind="tool_call.completed",
                data={
                    "name": "verify_work",
                    "is_error": True,
                    "arguments": {"command": "test -f result.txt"},
                    "content_preview": "FAILED (exit 1)",
                },
            ),
        ],
        session_id="s",
    )

    closed = workflow_commands._candidate_with_verified_objective_status_if_needed(
        node=node,
        run=run,
        candidate=candidate,
    )

    assert workflow_decision(closed.result) == ""


def test_budget_recovery_does_not_auto_complete_verifier_without_tool_evidence() -> None:
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="create result.txt",
        status="completed",
        result="Created result.txt and verified it exists.",
        metadata={
            "activity_summary": {
                "state_changed": True,
                "verify_work_passed": True,
            }
        },
    )
    verify = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        role="verifier",
        prompt="verify result.txt",
        depends_on=("work",),
        expected_evidence=(
            EvidenceRequirement(kind="result_nonempty"),
            EvidenceRequirement(kind="tool_called"),
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="no_state_change"),
            EvidenceRequirement(kind="no_failed_tools"),
        ),
    )
    merge = WorkflowNode(
        id="merge",
        title="Merge",
        kind="merge",
        role="merger",
        prompt="merge evidence",
        depends_on=("verify",),
        expected_evidence=(
            EvidenceRequirement(kind="result_nonempty"),
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="no_state_change"),
            EvidenceRequirement(kind="no_failed_tools"),
        ),
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="Create result.txt.",
        nodes=(work, verify, merge),
    )

    updated, completed = workflow_commands._complete_ready_deterministic_nodes(
        run=run,
        started_at=time.monotonic(),
    )

    assert completed == []
    assert {node.id: node.status for node in updated.nodes} == {
        "work": "completed",
        "verify": "pending",
        "merge": "pending",
    }


def test_workflow_node_prompt_does_not_embed_exact_content_shortcuts() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="create shell-result.txt containing exactly 'shelltruth' by shell redirection",
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="Create shell-result.txt containing exactly 'shelltruth' by shell redirection.",
        nodes=(node,),
    )

    prompt = workflow_commands._node_prompt(run, node)

    assert "Detected exact-content requirements" not in prompt
    assert "UTF-8 byte length" not in prompt


def test_workflow_node_prompt_does_not_embed_exact_output_shortcuts() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        role="verifier",
        prompt="verify output",
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="Create hello.py that prints exactly 'workflow-ok' with no trailing newline.",
        nodes=(node,),
    )

    prompt = workflow_commands._node_prompt(run, node)

    assert "Detected exact-output requirements" not in prompt
    assert "stdout must equal" not in prompt
    assert "raw stdout proves" not in prompt


def test_workflow_auto_verify_policy_does_not_infer_languages() -> None:
    source = Path(workflow_commands.__file__).read_text(encoding="utf-8")
    section = source[
        source.index("def _declared_executable_command") : source.index("def _goal_mentions_path")
    ]

    forbidden = (
        r"\.py\b",
        r"\bpython\d?\b",
        r"\bpytest\b",
        r"\bnode\b",
        r"\bnpm\b",
        r"\bcargo\b",
        r"\brust\b",
        r"\bgo\b",
    )
    for pattern in forbidden:
        assert re.search(pattern, section, flags=re.IGNORECASE) is None


def test_workflow_node_prompt_includes_tool_catalog_and_prediction_instruction() -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        role="researcher",
        prompt="research the objective",
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="answer accurately",
        nodes=(node,),
    )

    prompt = workflow_commands._node_prompt(run, node)

    assert "Available tools for this node" in prompt
    assert "shell (run a shell command" in prompt
    assert "web_search (search public web sources" in prompt
    assert "write_file" not in prompt
    assert "verify_work" not in prompt


def test_workflow_node_uses_workflow_evidence_not_run_once_verifier(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_build_agent(**kwargs):
        captured["memory_tools_enabled"] = kwargs.get("memory_tools_enabled")
        return object()

    async def _fake_run_once(**kwargs):
        captured.update(kwargs)
        kwargs["build_agent"](
            chain=["ollama"],
            base_url=None,
            model="test-model",
            storage=object(),
            cwd=tmp_path,
            config=object(),
            yes=True,
            build_tools=lambda *_args, **_kwargs: object(),
        )
        return "work result"

    monkeypatch.setattr(cli_main, "_build_agent", _fake_build_agent)
    monkeypatch.setattr(workflow_commands, "_run_once_impl", _fake_run_once)
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="work",
        allow_mutation=True,
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="test",
        nodes=(node,),
    )

    result = asyncio.run(
        workflow_commands._run_workflow_node(
            cwd=tmp_path,
            run=run,
            node=node,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
        )
    )

    assert result == "work result"
    assert captured["verify"] == "none"
    assert captured["profile"] == "bare"
    assert captured["predict"] is True
    assert captured["loop_detect"] is True
    assert captured["memory_tools_enabled"] is False


def test_workflow_read_only_node_disables_mutation_loop_directive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    async def _fake_run_once(**kwargs):
        captured.update(kwargs)
        return "research result"

    monkeypatch.setattr(workflow_commands, "_run_once_impl", _fake_run_once)
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        role="researcher",
        prompt="research",
        allow_mutation=False,
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="test",
        nodes=(node,),
    )

    result = asyncio.run(
        workflow_commands._run_workflow_node(
            cwd=tmp_path,
            run=run,
            node=node,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
        )
    )

    assert result == "research result"
    assert captured["loop_detect"] is False


def test_workflow_no_tool_node_uses_direct_text_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TextAdapter:
        async def stream(self, **_kwargs):
            yield Done(final_message=Message(role="assistant", content="direct plan"))

    async def _fail_run_once(**_kwargs):
        raise AssertionError("no-tool node should not use the full agent runtime")

    monkeypatch.setattr(workflow_commands, "_build_adapter", lambda *args, **kwargs: _TextAdapter())
    monkeypatch.setattr(workflow_commands, "_run_once_impl", _fail_run_once)
    node = WorkflowNode(
        id="plan",
        title="Plan",
        kind="plan",
        role="planner",
        prompt="plan",
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="test",
        nodes=(node,),
    )

    result = asyncio.run(
        workflow_commands._run_workflow_node(
            cwd=tmp_path,
            run=run,
            node=node,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
        )
    )

    assert result == "direct plan"


def test_workflow_verified_missing_status_retry_uses_direct_text_stream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _TextAdapter:
        async def stream(self, **kwargs):
            captured.update(kwargs)
            yield Done(final_message=Message(role="assistant", content='{"status":"pass"}'))

    async def _fail_run_once(**_kwargs):
        raise AssertionError("status-only retry should not reopen the tool loop")

    monkeypatch.setattr(workflow_commands, "_build_adapter", lambda *args, **kwargs: _TextAdapter())
    monkeypatch.setattr(workflow_commands, "_run_once_impl", _fail_run_once)
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="work",
        allow_mutation=True,
        metadata={
            "failure_history": [
                {
                    "attempt": 1,
                    "error": "objective status is missing",
                    "result": "verify_work passed",
                    "activity_summary": {
                        "state_changed": True,
                        "verify_work_passed": True,
                    },
                }
            ]
        },
    )
    run = WorkflowRun(id="workflow-test", title="Workflow Test", goal="test", nodes=(node,))

    result = asyncio.run(
        workflow_commands._run_workflow_node(
            cwd=tmp_path,
            run=run,
            node=node,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
        )
    )

    assert result == '{"status":"pass"}'
    assert "Previous evidence already exists" in captured["messages"][-1].content
    assert "Do not call tools" in captured["messages"][-1].content


def test_workflow_candidate_uses_tool_evidence_when_final_text_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="create files",
        allow_mutation=True,
        metadata={"completion_timeout_seconds": 0.01},
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="create files",
        nodes=(node,),
    )
    activity = [
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "write_file",
                "is_error": False,
                "arguments": {"path": "sales_report.py"},
                "content_preview": "wrote 100 bytes to sales_report.py",
            },
        ),
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "verify_work",
                "is_error": False,
                "arguments": {"command": "python3 sales_report.py"},
                "content_preview": "PASSED\n\nTotal Revenue: $460.00",
            },
        ),
    ]

    async def _slow_node(**_kwargs):
        await asyncio.sleep(10)
        return "late"

    async def _activity(**_kwargs):
        return activity

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _slow_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _activity)

    candidate = asyncio.run(
        workflow_commands._run_candidate(
            cwd=tmp_path,
            run=run,
            node=node,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
            candidate_index=0,
        )
    )

    assert candidate.error == ""
    assert "verify_work passed" in candidate.result
    assert "Total Revenue: $460.00" in candidate.result


def test_workflow_candidate_uses_tool_evidence_when_model_run_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="create files",
        allow_mutation=True,
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="create files",
        nodes=(node,),
    )
    activity = [
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "list_dir",
                "is_error": False,
                "arguments": {"path": "."},
                "content_preview": "README.md",
            },
        ),
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "write_file",
                "is_error": False,
                "arguments": {"path": "HARNESS_DEEPSWE_CHECK.md"},
                "content_preview": "wrote HARNESS_DEEPSWE_CHECK.md",
            },
        ),
    ]

    async def _raising_node(**_kwargs):
        raise TimeoutError("model stream produced no events for 45.0s")

    async def _activity(**_kwargs):
        return activity

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _raising_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _activity)

    candidate = asyncio.run(
        workflow_commands._run_candidate(
            cwd=tmp_path,
            run=run,
            node=node,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
            candidate_index=0,
        )
    )

    assert candidate.error == ""
    assert "write_file HARNESS_DEEPSWE_CHECK.md" in candidate.result


def test_workflow_candidate_uses_generic_plan_when_preflight_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = WorkflowNode(
        id="plan",
        title="Plan",
        kind="plan",
        role="planner",
        prompt="plan",
        metadata={"completion_timeout_seconds": 0.01},
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="create a useful script",
        nodes=(node,),
    )

    async def _slow_node(**_kwargs):
        await asyncio.sleep(10)
        return "late"

    async def _no_activity(**_kwargs):
        return []

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _slow_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _no_activity)

    candidate = asyncio.run(
        workflow_commands._run_candidate(
            cwd=tmp_path,
            run=run,
            node=node,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
            candidate_index=0,
        )
    )

    assert candidate.error == ""
    assert "Check any assumptions" in candidate.result
    assert "create a useful script" in candidate.result


def test_workflow_candidate_fails_fast_when_node_makes_no_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="create files",
        allow_mutation=True,
        metadata={"completion_timeout_seconds": 10, "idle_timeout_seconds": 0.01},
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="create files",
        nodes=(node,),
    )

    async def _slow_node(**_kwargs):
        await asyncio.sleep(10)
        return "late"

    async def _no_activity(**_kwargs):
        return []

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _slow_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _no_activity)

    candidate = asyncio.run(
        workflow_commands._run_candidate(
            cwd=tmp_path,
            run=run,
            node=node,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
            candidate_index=0,
        )
    )

    assert "no progress" in candidate.error
    assert candidate.result == ""


def test_workflow_merge_uses_completed_evidence_when_final_text_times_out(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="work",
        status="completed",
        result="Created sales_report.py and verify_work passed.",
    )
    merge = WorkflowNode(
        id="merge",
        title="Merge",
        kind="merge",
        role="merger",
        prompt="merge",
        depends_on=("work",),
        metadata={"completion_timeout_seconds": 0.01},
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="create a useful script",
        nodes=(work, merge),
    )

    async def _slow_node(**_kwargs):
        await asyncio.sleep(10)
        return "late"

    async def _no_activity(**_kwargs):
        return []

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _slow_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _no_activity)

    candidate = asyncio.run(
        workflow_commands._run_candidate(
            cwd=tmp_path,
            run=run,
            node=merge,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
            candidate_index=0,
        )
    )

    assert candidate.error == ""
    assert "Completed with workflow evidence" in candidate.result
    assert "verify_work passed" in candidate.result


def test_workflow_budget_completes_running_merge_from_verified_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = WorkflowStore(root=default_workflow_root(tmp_path))
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="work",
        status="completed",
        result="Created inventory_checker.py and verify_work passed.",
    )
    merge = WorkflowNode(
        id="merge",
        title="Merge",
        kind="merge",
        role="merger",
        prompt="merge",
        depends_on=("work",),
    )
    run = WorkflowRun(
        id="workflow-budget-merge",
        title="Workflow Budget Merge",
        goal="create an inventory checker",
        nodes=(work, merge),
    )
    store.add_run(run)

    async def _slow_node(**_kwargs):
        await asyncio.sleep(10)
        return "late"

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _slow_node)

    completed = asyncio.run(
        workflow_commands.run_workflow(
            cwd=tmp_path,
            store=store,
            workflow_id=run.id,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
            max_runtime_seconds=1,
        )
    )

    assert completed.status == "completed"
    by_id = {node.id: node for node in completed.nodes}
    assert by_id["merge"].status == "completed"
    assert "Completed with workflow evidence" in completed.final_report
    assert "verify_work passed" in completed.final_report


def test_workflow_budget_completes_running_work_from_verified_activity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = WorkflowStore(root=default_workflow_root(tmp_path))
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="work",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="result_nonempty"),
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="no_failed_tools"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
    )
    run = WorkflowRun(
        id="workflow-budget-work",
        title="Workflow Budget Work",
        goal="create an inventory checker",
        nodes=(work,),
    )
    store.add_run(run)

    async def _slow_node(**_kwargs):
        await asyncio.sleep(10)
        return "late"

    async def _verified_activity(**_kwargs):
        return [
            ActivityEvent(
                session_id="s",
                kind="tool_call.completed",
                data={
                    "name": "write_file",
                    "is_error": False,
                    "arguments": {"path": "inventory_checker.py"},
                    "content_preview": "wrote inventory_checker.py",
                },
            ),
            ActivityEvent(
                session_id="s",
                kind="tool_call.completed",
                data={
                    "name": "verify_work",
                    "is_error": False,
                    "arguments": {"command": "python3 inventory_checker.py"},
                    "content_preview": "PASSED inventory checker ran",
                },
            ),
        ]

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _slow_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _verified_activity)

    completed = asyncio.run(
        workflow_commands.run_workflow(
            cwd=tmp_path,
            store=store,
            workflow_id=run.id,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
            max_runtime_seconds=1,
        )
    )

    assert completed.status == "completed"
    by_id = {node.id: node for node in completed.nodes}
    assert by_id["work"].status == "completed"
    assert "Node produced tool evidence" in completed.final_report
    assert "verify_work passed" in completed.final_report


def test_workflow_budget_completes_running_refute_after_passing_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = WorkflowStore(root=default_workflow_root(tmp_path))
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="work",
        status="completed",
        result="Created ENV_REPORT.md and verify_work passed.",
        metadata={"activity_summary": {"state_changed": True, "verify_work_passed": True}},
    )
    verify = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        depends_on=("work",),
        status="completed",
        result="Independent verification passed.",
    )
    review = WorkflowNode(
        id="review",
        title="Review",
        kind="review",
        prompt="review",
        depends_on=("verify",),
        status="completed",
        result="Review found the evidence sufficient.\nWORKFLOW_DECISION: pass",
    )
    refute = WorkflowNode(
        id="refute",
        title="Refute",
        kind="refute",
        prompt="refute",
        depends_on=("review",),
        status="running",
        session_id="workflow-test-refute",
        metadata={"lease_expires_at": time.time() + 900},
        expected_evidence=(
            EvidenceRequirement(kind="result_nonempty"),
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="no_state_change"),
            EvidenceRequirement(kind="no_failed_tools"),
            EvidenceRequirement(kind="review_decision"),
        ),
    )
    merge = WorkflowNode(
        id="merge",
        title="Merge",
        kind="merge",
        prompt="merge",
        depends_on=("refute",),
        expected_evidence=(
            EvidenceRequirement(kind="result_nonempty"),
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="no_state_change"),
            EvidenceRequirement(kind="no_failed_tools"),
        ),
    )
    run = WorkflowRun(
        id="workflow-budget-refute",
        title="Workflow Budget Refute",
        goal="Create ENV_REPORT.md.",
        status="running",
        nodes=(work, verify, review, refute, merge),
    )
    store.add_run(run)

    async def _no_activity(**_kwargs):
        return []

    monkeypatch.setattr(workflow_commands, "_load_node_activity", _no_activity)

    completed = asyncio.run(
        workflow_commands.run_workflow(
            cwd=tmp_path,
            store=store,
            workflow_id=run.id,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
            max_runtime_seconds=0,
        )
    )

    by_id = {node.id: node for node in completed.nodes}
    assert completed.status == "completed"
    assert by_id["refute"].status == "completed"
    assert workflow_decision(by_id["refute"].result) == "pass"
    assert by_id["refute"].metadata["consensus"]["recovered_from_budget_evidence"] is True
    assert by_id["merge"].status == "completed"


def test_workflow_does_not_retry_when_runtime_budget_is_too_low(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = WorkflowStore(root=default_workflow_root(tmp_path))
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="work",
        max_attempts=2,
        metadata={"completion_timeout_seconds": 0.01},
    )
    run = WorkflowRun(
        id="workflow-no-hopeless-retry",
        title="Workflow No Hopeless Retry",
        goal="create an inventory checker",
        nodes=(work,),
    )
    store.add_run(run)

    async def _slow_node(**_kwargs):
        await asyncio.sleep(10)
        return "late"

    async def _no_activity(**_kwargs):
        return []

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _slow_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _no_activity)

    completed = asyncio.run(
        workflow_commands.run_workflow(
            cwd=tmp_path,
            store=store,
            workflow_id=run.id,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
            max_runtime_seconds=1,
        )
    )

    by_id = {node.id: node for node in completed.nodes}
    assert completed.status == "failed"
    assert by_id["work"].attempts == 1
    assert "node timed out before producing final text" in completed.final_report


def test_workflow_mutating_work_rejects_read_only_tool_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="create inventory checker",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="result_nonempty"),
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="no_failed_tools"),
            EvidenceRequirement(kind="files_changed"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="Can you build me a tiny inventory checker in Python for a shop?",
        nodes=(node,),
    )
    activity = [
        ActivityEvent(
            session_id="workflow_workflow_test_work",
            kind="tool_call.completed",
            data={
                "name": "shell",
                "is_error": False,
                "arguments": {"command": "ls -F"},
                "content_preview": "exit_code: 0 (no output)",
            },
        )
    ]

    async def _node(**_kwargs):
        return ""

    async def _activity(**_kwargs):
        return activity

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _activity)

    finished = asyncio.run(
        workflow_commands._execute_node(
            cwd=tmp_path,
            run=run,
            node=node,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
            subagents=1,
            consensus_threshold=1,
        )
    )

    assert finished.status == "failed"
    assert "no state change observed" in finished.error


def test_workflow_verify_does_not_replace_pseudo_tool_text_without_tool_evidence() -> None:
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="work",
        status="completed",
        result="Created sales_report.py and verify_work passed.",
    )
    verify = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        role="verifier",
        prompt="verify",
        depends_on=("work",),
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="create a useful script",
        nodes=(work, verify),
    )
    candidate = workflow_commands.NodeCandidate(
        index=0,
        result='<|tool_call>call:list_dir{path:"."}<tool_call|>',
        activity=[],
        session_id="s",
    )

    replaced = workflow_commands._candidate_with_deterministic_result_if_needed(
        node=verify,
        run=run,
        candidate=candidate,
    )

    assert replaced.result == candidate.result
    assert "tool_call" in replaced.result


def test_workflow_auto_verifies_declared_executable_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "sales.csv").write_text(
        "product,price,quantity\nWidget A,10.00,5\nWidget B,20.00,3\n",
        encoding="utf-8",
    )
    (tmp_path / "sales_report").write_text(
        "#!/bin/sh\nprintf '%s\\n' 'Total Revenue: $110.00'\n",
        encoding="utf-8",
    )
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="create sales report",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="result_nonempty"),
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="no_failed_tools"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="Create sales_report that reads sales.csv and verifies it runs.",
        nodes=(node,),
    )
    activity = [
        ActivityEvent(
            session_id="workflow_workflow_test_work",
            kind="tool_call.completed",
            data={
                "name": "write_file",
                "is_error": False,
                "arguments": {"path": "sales.csv"},
                "content_preview": "wrote 57 bytes to sales.csv",
            },
        ),
        ActivityEvent(
            session_id="workflow_workflow_test_work",
            kind="tool_call.completed",
            data={
                "name": "write_file",
                "is_error": False,
                "arguments": {"path": "sales_report"},
                "content_preview": "wrote sales_report",
            },
        ),
    ]

    async def _node(**_kwargs):
        return ""

    async def _activity(**_kwargs):
        return activity

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _activity)

    finished = asyncio.run(
        workflow_commands._execute_node(
            cwd=tmp_path,
            run=run,
            node=node,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
            subagents=1,
            consensus_threshold=1,
        )
    )

    assert finished.status == "completed"
    assert "verify_work passed" in finished.result
    assert "Total Revenue: $110.00" in finished.result
    evidence = finished.metadata["evidence_results"]
    assert any(
        item["requirement"]["kind"] == "verify_work_if_state_changed" and item["status"] == "passed"
        for item in evidence
    )


def test_workflow_auto_verify_prefers_declared_test_command_over_plain_script(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "inventory_checker").write_text(
        "requires interactive input\n",
        encoding="utf-8",
    )
    (tmp_path / "test_inventory").write_text(
        "#!/bin/sh\nprintf '%s\\n' 'PASSED inventory check'\n",
        encoding="utf-8",
    )
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="create inventory checker",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="result_nonempty"),
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="no_failed_tools"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="Create an inventory checker and verify it works.",
        nodes=(node,),
    )
    activity = [
        ActivityEvent(
            session_id="workflow_workflow_test_work",
            kind="tool_call.completed",
            data={
                "name": "write_file",
                "is_error": False,
                "arguments": {"path": "inventory_checker"},
                "content_preview": "wrote inventory_checker",
            },
        ),
        ActivityEvent(
            session_id="workflow_workflow_test_work",
            kind="tool_call.completed",
            data={
                "name": "write_file",
                "is_error": False,
                "arguments": {"path": "test_inventory"},
                "content_preview": "wrote test_inventory",
            },
        ),
    ]

    async def _node(**_kwargs):
        return ""

    async def _activity(**_kwargs):
        return activity

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _activity)

    finished = asyncio.run(
        workflow_commands._execute_node(
            cwd=tmp_path,
            run=run,
            node=node,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
            subagents=1,
            consensus_threshold=1,
        )
    )

    assert finished.status == "completed"
    assert "verify_work passed" in finished.result
    assert "test_inventory" in finished.result
    assert "PASSED inventory check" in finished.result


def test_workflow_auto_verify_exact_file_asserts_content_and_size(tmp_path: Path) -> None:
    (tmp_path / "answer.txt").write_bytes(b"hello")
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="Create answer.txt containing exactly `hello` with no trailing newline.",
        allow_mutation=True,
        expected_evidence=(EvidenceRequirement(kind="verify_work_if_state_changed"),),
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="Create answer.txt containing exactly `hello` with no trailing newline.",
        nodes=(node,),
    )
    write_event = ActivityEvent(
        session_id="workflow_workflow_test_work",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "answer.txt", "content": "hello"},
            "content_preview": "wrote answer.txt",
        },
    )

    command = workflow_commands._infer_auto_verify_command(
        cwd=tmp_path,
        run=run,
        node=node,
        activity=[write_event],
    )
    verify_event = ActivityEvent(
        session_id="workflow_workflow_test_work",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": command},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="answer.txt verified",
        activity=[write_event, verify_event],
        run=run,
    )

    assert "cmp -s" in command
    assert "wc -c" in command
    assert ok is True
    assert results[0].status == "passed"


def test_workflow_auto_verify_infers_runnable_script_command(tmp_path: Path) -> None:
    (tmp_path / "get_weather").write_text("#!/bin/sh\nprintf '%s\\n' ok\n", encoding="utf-8")
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="create script",
        allow_mutation=True,
        expected_evidence=(EvidenceRequirement(kind="verify_work_if_state_changed"),),
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="Create get_weather and verify it runs.",
        nodes=(node,),
    )
    write_event = ActivityEvent(
        session_id="workflow_workflow_test_work",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "get_weather"},
            "content_preview": "wrote get_weather",
        },
    )

    command = workflow_commands._infer_auto_verify_command(
        cwd=tmp_path,
        run=run,
        node=node,
        activity=[write_event],
    )

    assert command == "/bin/sh ./get_weather"


def test_workflow_auto_verify_skips_source_artifact_handoff(tmp_path: Path) -> None:
    (tmp_path / "invoices.csv").write_text("id,amount\n1,10.00\n", encoding="utf-8")
    (tmp_path / "ANALYSIS.md").write_text("total: 10.00\n", encoding="utf-8")
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="create analysis",
        allow_mutation=True,
        expected_evidence=(EvidenceRequirement(kind="verify_work_if_state_changed"),),
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="Produce ANALYSIS.md from invoices.csv.",
        nodes=(node,),
    )
    read_source = ActivityEvent(
        session_id="workflow_workflow_test_work",
        kind="tool_call.completed",
        data={
            "name": "read_file",
            "is_error": False,
            "arguments": {"path": "invoices.csv"},
            "content_preview": "id,amount\n1,10.00",
        },
    )
    write_report = ActivityEvent(
        session_id="workflow_workflow_test_work",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "ANALYSIS.md"},
            "content_preview": "wrote ANALYSIS.md",
        },
    )

    command = workflow_commands._infer_auto_verify_command(
        cwd=tmp_path,
        run=run,
        node=node,
        activity=[read_source, write_report],
    )

    assert command == ""


def test_workflow_auto_verify_recurses_source_directories_for_distributions(
    tmp_path: Path,
) -> None:
    tasks = tmp_path / "tasks"
    (tasks / "one").mkdir(parents=True)
    (tasks / "two").mkdir()
    (tasks / "three").mkdir()
    (tasks / "one" / "task.toml").write_text('language = "go"\n', encoding="utf-8")
    (tasks / "two" / "task.toml").write_text('language = "python"\n', encoding="utf-8")
    (tasks / "three" / "task.toml").write_text('language = "python"\n', encoding="utf-8")
    (tmp_path / "REPORT.md").write_text(
        "Task count: 3\nLanguage distribution:\n- Go: 1\n- Python: 2\n",
        encoding="utf-8",
    )

    command = workflow_commands._source_artifact_handoff_command(
        cwd=tmp_path,
        changed_paths=[Path("REPORT.md")],
        source_paths=[Path("tasks"), Path("tasks/*/task.toml")],
    )

    result = subprocess.run(
        command,
        cwd=tmp_path,
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )

    assert command
    assert result.returncode == 0, result.stderr
    assert "tasks:dir-count=3" in result.stdout
    assert "language:distribution" in result.stdout


def test_verify_node_requires_fresh_tool_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="Do the work.",
        allow_mutation=True,
        status="completed",
        result="Created answer.txt containing exactly hello and verify_work passed.",
        metadata={"activity_summary": {"state_changed": True, "verify_work_passed": True}},
    )
    verify = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        role="verifier",
        prompt="Verify the work.",
        depends_on=("work",),
        expected_evidence=(
            EvidenceRequirement(kind="result_nonempty"),
            EvidenceRequirement(kind="tool_called"),
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="no_state_change"),
            EvidenceRequirement(kind="no_failed_tools"),
        ),
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="Create answer.txt containing exactly `hello`.",
        nodes=(work, verify),
    )

    async def _run_candidate(**_kwargs):
        return workflow_commands.NodeCandidate(
            index=0,
            result="Read answer.txt and verified it contains exactly hello.",
            activity=[
                ActivityEvent(
                    session_id="s",
                    kind="tool_call.completed",
                    data={
                        "name": "read_file",
                        "is_error": False,
                        "arguments": {"path": "answer.txt"},
                        "content_preview": "hello",
                    },
                )
            ],
            session_id="s",
        )

    monkeypatch.setattr(workflow_commands, "_run_candidate", _run_candidate)

    finished = asyncio.run(
        workflow_commands._execute_node(
            cwd=tmp_path,
            run=run,
            node=verify,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
            subagents=1,
            consensus_threshold=1,
        )
    )

    assert finished.status == "completed"
    assert "verified it contains exactly hello" in finished.result
    assert finished.metadata["activity_summary"]["tool_calls"] == 1


def test_workflow_does_not_auto_verify_partial_non_runnable_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "sales.csv").write_text(
        "product,price,quantity\nWidget A,10.00,5\n",
        encoding="utf-8",
    )
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="create sales report",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="result_nonempty"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="Create sales_report.py that reads sales.csv and verifies it runs.",
        nodes=(node,),
    )
    activity = [
        ActivityEvent(
            session_id="workflow_workflow_test_work",
            kind="tool_call.completed",
            data={
                "name": "write_file",
                "is_error": False,
                "arguments": {"path": "sales.csv"},
                "content_preview": "wrote 38 bytes to sales.csv",
            },
        )
    ]

    async def _node(**_kwargs):
        return "created sales.csv"

    async def _activity(**_kwargs):
        return activity

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _activity)

    finished = asyncio.run(
        workflow_commands._execute_node(
            cwd=tmp_path,
            run=run,
            node=node,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
            subagents=1,
            consensus_threshold=1,
        )
    )

    assert finished.status == "failed"
    assert "state changed without a later passing verify_work call" in finished.error


def test_workflow_materializes_exact_file_when_model_stalls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt="create result.txt containing exactly 'harness-ok'",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="result_nonempty"),
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="no_failed_tools"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
        metadata={"completion_timeout_seconds": 0.01},
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal="Create result.txt containing exactly 'harness-ok' and verify it.",
        nodes=(node,),
    )

    async def _slow_node(**_kwargs):
        await asyncio.sleep(10)
        return "late"

    async def _no_activity(**_kwargs):
        return []

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _slow_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _no_activity)

    candidate = asyncio.run(
        workflow_commands._run_candidate(
            cwd=tmp_path,
            run=run,
            node=node,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
            candidate_index=0,
        )
    )

    assert candidate.error == ""
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "harness-ok"
    assert "verify_work passed" in candidate.result
    assert "result.txt" in candidate.result
    assert "harness-ok" in candidate.result
    ok, results = evaluate_node_evidence(
        node=node,
        result=candidate.result,
        activity=candidate.activity,
        run=run,
    )
    assert ok, [result.message for result in results]


def test_workflow_materializes_exact_file_after_read_only_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        role="implementer",
        prompt=(
            "Create GREETING.txt from profile.json. It should say: Hello Ada, Python count is 4."
        ),
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="result_nonempty"),
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="no_failed_tools"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
        metadata={"completion_timeout_seconds": 0.01},
    )
    run = WorkflowRun(
        id="workflow-test",
        title="Workflow Test",
        goal=(
            "Create GREETING.txt from profile.json. It should say: Hello Ada, Python count is 4."
        ),
        nodes=(node,),
    )
    activity = [
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "list_dir",
                "is_error": False,
                "arguments": {"path": "."},
                "content_preview": "profile.json",
            },
        ),
    ]

    async def _raising_node(**_kwargs):
        raise TimeoutError("model stream produced no events for 45.0s")

    async def _activity(**_kwargs):
        return activity

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _raising_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _activity)

    candidate = asyncio.run(
        workflow_commands._run_candidate(
            cwd=tmp_path,
            run=run,
            node=node,
            provider="ollama",
            model="test-model",
            max_steps=3,
            yes=True,
            candidate_index=0,
        )
    )

    assert candidate.error == ""
    assert (tmp_path / "GREETING.txt").read_text(
        encoding="utf-8"
    ) == "Hello Ada, Python count is 4."
    assert "write_file GREETING.txt" in candidate.result
    assert "verify_work passed" in candidate.result
    ok, results = evaluate_node_evidence(
        node=node,
        result=candidate.result,
        activity=candidate.activity,
        run=run,
    )
    assert ok, [result.message for result in results]


def test_activity_fallback_formats_structured_web_search_results() -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        prompt="Find current web results.",
    )
    activity = [
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "web_search",
                "is_error": False,
                "content_preview": "Results for: OpenAI...",
                "arguments": {"query": "OpenAI news"},
                "metadata": {
                    "backend": "duckduckgo",
                    "results": [
                        {
                            "title": "OpenAI News",
                            "content": "Official updates from OpenAI.",
                            "url": "https://openai.com/news/",
                        },
                        {
                            "title": "OpenAI update",
                            "content": "A second public result.",
                            "url": "https://example.com/openai",
                        },
                    ],
                },
            },
        )
    ]

    result = workflow_commands._activity_fallback_result(node, activity)

    assert "web_search results" in result
    assert "OpenAI News" in result
    assert "https://openai.com/news/" in result
    assert "A second public result" in result


def test_activity_fallback_marks_truncated_read_previews() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="Verify the report.",
    )
    activity = [
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "read_file",
                "is_error": False,
                "arguments": {"path": "ANALYSIS.md"},
                "content_preview": "Paid Amount by Region: - EU: 420.50 - US: 2",
                "content_size": 223,
            },
        ),
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "verify_work",
                "is_error": False,
                "arguments": {"command": "python3 verify_analysis.py"},
                "content_preview": "PASSED\n\nPASSED",
                "content_size": 14,
            },
        ),
    ]

    result = workflow_commands._activity_fallback_result(node, activity)

    assert "read_file ANALYSIS.md" in result
    assert "truncated preview" in result
    assert "do not infer exact missing suffix" in result
    assert "verify_work passed" in result


def test_empty_candidate_uses_activity_fallback_result() -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        prompt="Find current web results.",
    )
    run = WorkflowRun(id="wf", title="WF", goal="Find current web results", nodes=(node,))
    activity = [
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "web_search",
                "is_error": False,
                "arguments": {"query": "OpenAI news"},
                "metadata": {
                    "results": [
                        {
                            "title": "OpenAI News",
                            "content": "Official updates from OpenAI.",
                            "url": "https://openai.com/news/",
                        }
                    ]
                },
            },
        )
    ]
    candidate = workflow_commands.NodeCandidate(
        index=0,
        result="",
        activity=activity,
        session_id="s",
    )

    selected = workflow_commands._candidate_with_deterministic_result_if_needed(
        node=node,
        run=run,
        candidate=candidate,
    )

    assert "Node produced tool evidence" in selected.result
    assert "https://openai.com/news/" in selected.result


def test_merge_deterministic_result_can_use_research_evidence() -> None:
    research = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        prompt="research",
        status="completed",
        result="Research evidence with https://openai.com/news/",
    )
    answer = WorkflowNode(
        id="answer",
        title="Answer",
        kind="merge",
        prompt="answer",
        depends_on=("research",),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Answer from research",
        nodes=(research, answer),
    )

    result = workflow_commands._deterministic_node_result(node=answer, run=run)

    assert "Completed with workflow evidence" in result
    assert "https://openai.com/news/" in result


def test_merge_deterministic_result_prefers_declared_dependency_evidence() -> None:
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="work",
        status="completed",
        result="work details should not be duplicated",
    )
    verify = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        status="completed",
        result="verified result",
    )
    merge = WorkflowNode(
        id="merge",
        title="Merge",
        kind="merge",
        prompt="merge",
        depends_on=("verify",),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Answer from verify",
        nodes=(work, verify, merge),
    )

    result = workflow_commands._deterministic_node_result(node=merge, run=run)

    assert "verified result" in result
    assert "work details should not be duplicated" not in result


def test_merge_deterministic_result_does_not_claim_completion_for_raw_tool_evidence() -> None:
    research = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        prompt="research",
        status="completed",
        result=(
            "Node produced tool evidence before final text was available.\n"
            "- web_search results:\n"
            "  1. Source (https://example.com/source)"
        ),
    )
    answer = WorkflowNode(
        id="answer",
        title="Answer",
        kind="merge",
        prompt="answer",
        depends_on=("research",),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Answer from research",
        nodes=(research, answer),
    )

    result = workflow_commands._deterministic_node_result(node=answer, run=run)

    assert "could not produce a verified final answer" in result
    assert "Completed with workflow evidence" not in result
    assert "https://example.com/source" in result


def test_workflow_runtime_budget_stops_before_work(tmp_path: Path, monkeypatch) -> None:
    store = WorkflowStore(root=default_workflow_root(tmp_path))
    run = WorkflowRun(
        id=store.new_id("Budget"),
        title="Budget",
        goal="stop quickly",
        nodes=(
            WorkflowNode(
                id="a",
                title="A",
                kind="research",
                role="researcher",
                prompt="a",
                expected_evidence=(EvidenceRequirement(kind="result_nonempty"),),
            ),
        ),
    )
    store.add_run(run)

    async def _fake_node(**_kwargs):
        raise AssertionError("node should not run when budget is already exceeded")

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _fake_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _fake_successful_activity)

    finished = asyncio.run(
        workflow_commands.run_workflow(
            cwd=tmp_path,
            store=store,
            workflow_id=run.id,
            max_runtime_seconds=-1,
        )
    )

    assert finished.status == "failed"
    assert finished.final_report
    assert any(event.kind == "workflow.budget_exceeded" for event in store.list_events(run.id))


def test_workflow_runtime_budget_fails_active_node(tmp_path: Path, monkeypatch) -> None:
    store = WorkflowStore(root=default_workflow_root(tmp_path))
    run = WorkflowRun(
        id=store.new_id("Active Budget"),
        title="Active Budget",
        goal="stop active node",
        nodes=(
            WorkflowNode(
                id="a",
                title="A",
                kind="research",
                role="researcher",
                prompt="a",
                expected_evidence=(EvidenceRequirement(kind="result_nonempty"),),
            ),
        ),
    )
    store.add_run(run)

    async def _slow_node(**_kwargs):
        await asyncio.sleep(10)
        return "late"

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _slow_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _fake_successful_activity)

    finished = asyncio.run(
        workflow_commands.run_workflow(
            cwd=tmp_path,
            store=store,
            workflow_id=run.id,
            max_runtime_seconds=1,
        )
    )

    assert finished.status == "failed"
    assert finished.nodes[0].status == "failed"
    assert "runtime budget" in finished.nodes[0].error
    assert finished.final_report
    assert any(event.kind == "workflow.budget_exceeded" for event in store.list_events(run.id))


def test_workflow_resume_recovers_expired_running_node(tmp_path: Path, monkeypatch) -> None:
    store = WorkflowStore(root=default_workflow_root(tmp_path))
    run = WorkflowRun(
        id=store.new_id("Recover"),
        title="Recover",
        goal="recover stale lease",
        status="running",
        nodes=(
            WorkflowNode(
                id="a",
                title="A",
                kind="research",
                role="researcher",
                prompt="a",
                status="running",
                metadata={"lease_owner": "old", "lease_expires_at": 0},
                expected_evidence=(EvidenceRequirement(kind="result_nonempty"),),
            ),
        ),
    )
    store.add_run(run)

    async def _fake_node(**kwargs):
        return f"{kwargs['node'].id} result"

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _fake_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _fake_successful_activity)

    finished = asyncio.run(
        workflow_commands.run_workflow(
            cwd=tmp_path,
            store=store,
            workflow_id=run.id,
        )
    )

    assert finished.status == "completed"
    assert finished.nodes[0].metadata["recovered_from_stale_lease"] is True
    assert any(event.kind == "node.lease_recovered" for event in store.list_events(run.id))


def test_workflow_review_retry_resets_worker_once(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()
    calls: dict[str, int] = {}

    async def _fake_node(**kwargs):
        node = kwargs["node"]
        calls[node.id] = calls.get(node.id, 0) + 1
        if node.kind == "review":
            return "review result\nWORKFLOW_DECISION: pass"
        if node.kind == "refute" and calls[node.id] == 1:
            return "not enough evidence\nWORKFLOW_DECISION: retry"
        if node.kind == "refute":
            return "evidence is now enough\nWORKFLOW_DECISION: pass"
        return f"{node.id} result"

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _fake_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _fake_successful_activity)
    created = runner.invoke(
        cli_main.app,
        [
            "workflow",
            "start",
            "--cwd",
            str(tmp_path),
            "--goal",
            "Retry once after review.",
            "--max-review-rounds",
            "1",
            "--json",
        ],
    )

    assert created.exit_code == 0, created.stdout
    payload = json.loads(created.stdout)
    assert payload["status"] == "completed"
    assert calls["work"] == 2
    assert payload["metadata"]["review_rounds"] == 1


def test_workflow_retry_decision_fails_when_rounds_exhausted(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runner = CliRunner()

    async def _fake_node(**kwargs):
        node = kwargs["node"]
        if node.kind == "review":
            return "review result\nWORKFLOW_DECISION: pass"
        if node.kind == "refute":
            return "not enough evidence\nWORKFLOW_DECISION: retry"
        return f"{node.id} result"

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _fake_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _fake_successful_activity)
    created = runner.invoke(
        cli_main.app,
        [
            "workflow",
            "start",
            "--cwd",
            str(tmp_path),
            "--goal",
            "Retry should fail with no rounds.",
            "--max-review-rounds",
            "0",
            "--json",
        ],
    )

    assert created.exit_code == 0, created.stdout
    payload = json.loads(created.stdout)
    by_id = {node["id"]: node for node in payload["nodes"]}
    assert payload["status"] == "failed"
    assert by_id["refute"]["status"] == "failed"
    assert "max review rounds" in by_id["refute"]["error"]
    assert by_id["merge"]["status"] == "skipped"
    assert "verified final report" in payload["final_report"]


def test_workflow_events_and_graph_commands(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()

    async def _fake_node(**kwargs):
        if kwargs["node"].kind in {"review", "refute"}:
            return f"{kwargs['node'].id} result\nWORKFLOW_DECISION: pass"
        return f"{kwargs['node'].id} result"

    monkeypatch.setattr(workflow_commands, "_run_workflow_node", _fake_node)
    monkeypatch.setattr(workflow_commands, "_load_node_activity", _fake_successful_activity)
    created = runner.invoke(
        cli_main.app,
        [
            "workflow",
            "start",
            "--cwd",
            str(tmp_path),
            "--goal",
            "Observe workflow.",
            "--json",
        ],
    )
    assert created.exit_code == 0, created.stdout
    workflow_id = json.loads(created.stdout)["id"]

    events = runner.invoke(
        cli_main.app,
        ["workflow", "events", workflow_id, "--cwd", str(tmp_path), "--json"],
    )
    assert events.exit_code == 0, events.stdout
    assert any(item["kind"] == "workflow.completed" for item in json.loads(events.stdout))

    graph = runner.invoke(
        cli_main.app,
        ["workflow", "graph", workflow_id, "--cwd", str(tmp_path)],
    )
    assert graph.exit_code == 0, graph.stdout
    assert "flowchart TD" in graph.stdout
    assert "refute --> merge" in graph.stdout
