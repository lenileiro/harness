from __future__ import annotations

import re
from pathlib import Path

import pytest

from harness.core.activity import ActivityEvent
from harness.core.dynamic_workflows import (
    EvidenceRequirement,
    WorkflowNode,
    WorkflowRun,
    WorkflowStore,
    _exact_file_content_requests,
    _exact_stdout_requests,
    create_default_workflow,
    create_workflow_from_plan_spec,
    default_workflow_root,
    evaluate_node_evidence,
    render_workflow_mermaid,
    summarize_activity,
    workflow_decision,
)


def test_workflow_store_round_trips_default_workflow(tmp_path: Path) -> None:
    store = WorkflowStore(root=default_workflow_root(tmp_path))
    workflow_id = store.new_id("Checkout Audit")
    run = create_default_workflow(
        workflow_id=workflow_id,
        title="Checkout Audit",
        goal="Audit checkout for unsafe redirects.",
    )

    store.add_run(run)
    store.append_event(run.id, kind="workflow.created", message="created")

    loaded = store.load_run(run.id)
    assert loaded.id == workflow_id
    assert loaded.status == "pending"
    assert [node.id for node in loaded.nodes] == [
        "plan",
        "research",
        "work",
        "verify",
        "review",
        "refute",
        "merge",
    ]
    assert loaded.nodes[-1].depends_on == ("refute",)
    assert loaded.nodes[2].allow_mutation is True
    assert "tool_called" in {item.kind for item in loaded.nodes[3].expected_evidence}
    assert "verify_work_passed" in {item.kind for item in loaded.nodes[3].expected_evidence}
    assert loaded.nodes[3].max_attempts == 2
    assert "expected outcome with confidence" in loaded.nodes[0].prompt
    assert "Check the plan's important assumptions" in loaded.nodes[1].prompt
    assert loaded.nodes[0].metadata["completion_timeout_seconds"] == 45
    assert loaded.nodes[1].metadata["idle_timeout_seconds"] == 60
    assert loaded.nodes[2].metadata["completion_timeout_seconds"] == 240
    assert store.list_events(run.id)[0].message == "created"


def test_plan_evidence_rejects_tool_calls_and_completion_claims() -> None:
    node = WorkflowNode(
        id="plan",
        title="Plan",
        kind="plan",
        prompt="Plan the workflow.",
        expected_evidence=(EvidenceRequirement(kind="planning_only"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="create result.txt", nodes=(node,))
    mutating_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "printf '%s' truth > result.txt"},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="The execution of the workflow was successful and result.txt was created.",
        activity=[mutating_verify],
        run=run,
    )

    assert ok is False
    assert "plan node executed tools" in results[0].message

    ok, results = evaluate_node_evidence(
        node=node,
        result="The workflow has been completed successfully.",
        activity=[],
        run=run,
    )

    assert ok is False
    assert "claims completed work" in results[0].message

    ok, results = evaluate_node_evidence(
        node=node,
        result="I have created result.txt and verified the file content.",
        activity=[],
        run=run,
    )

    assert ok is False
    assert "claims completed work" in results[0].message

    ok, results = evaluate_node_evidence(
        node=node,
        result="Plan: create result.txt, then verify content and byte size.",
        activity=[],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"

    ok, results = evaluate_node_evidence(
        node=node,
        result="The task can be completed using the Python standard library.",
        activity=[],
        run=run,
    )

    assert ok is True
    assert results[0].message == "plan node stayed in planning mode"

    ok, results = evaluate_node_evidence(
        node=node,
        result=(
            "**Expected Outcome:**\n"
            "- A file named `result.txt` is created in the workspace.\n"
            "- The verification step returns a `PASSED` status."
        ),
        activity=[],
        run=run,
    )

    assert ok is True
    assert results[0].message == "plan node stayed in planning mode"


def test_tool_called_requirement_without_name_requires_successful_tool_evidence() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        expected_evidence=(EvidenceRequirement(kind="tool_called"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="verify result", nodes=(node,))
    read = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "read_file",
            "is_error": False,
            "arguments": {"path": "result.txt"},
            "content_preview": "ok",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Verified result.txt.",
        activity=[],
        run=run,
    )

    assert ok is False
    assert results[0].message == "no successful tool call observed"

    ok, results = evaluate_node_evidence(
        node=node,
        result="Verified result.txt.",
        activity=[read],
        run=run,
    )

    assert ok is True
    assert results[0].message == "at least one tool call succeeded"


def test_summarize_activity_records_read_and_changed_paths() -> None:
    activity = [
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "read_file",
                "is_error": False,
                "arguments": {"path": "source.csv"},
                "content_preview": "amount\n1",
            },
        ),
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "write_file",
                "is_error": False,
                "arguments": {"path": "REPORT.md", "content": "total 1"},
                "content_preview": "wrote REPORT.md",
            },
        ),
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "shell",
                "is_error": False,
                "arguments": {"command": "jq -r '.items[].name' source.csv > sorted_names.txt"},
                "content_preview": "exit_code: 0",
            },
        ),
    ]

    summary = summarize_activity(activity)

    assert summary["read_paths"] == ["source.csv"]
    assert summary["changed_paths"] == ["REPORT.md", "sorted_names.txt"]


def test_summarize_activity_records_shell_source_paths() -> None:
    activity = [
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "shell",
                "is_error": False,
                "arguments": {
                    "command": (
                        "find tasks -name 'task.toml' | wc -l && "
                        "grep -h 'language =' tasks/*/task.toml"
                    )
                },
                "content_preview": "exit_code: 0",
            },
        ),
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "write_file",
                "is_error": False,
                "arguments": {"path": "REPORT.md", "content": "report"},
                "content_preview": "wrote REPORT.md",
            },
        ),
    ]

    summary = summarize_activity(activity)

    assert "tasks" in summary["read_paths"]
    assert "tasks/*/task.toml" in summary["read_paths"]
    assert summary["changed_paths"] == ["REPORT.md"]


def test_verify_node_rejects_artifact_only_verify_for_source_derived_dependency() -> None:
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="work",
        status="completed",
        metadata={
            "activity_summary": {
                "state_changed": True,
                "verify_work_passed": True,
                "read_paths": ["invoices.csv"],
                "changed_paths": ["ANALYSIS.md"],
            }
        },
    )
    verify = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        depends_on=("work",),
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="verify report", nodes=(work, verify))
    artifact_only = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "grep -q 'US: 299.50' ANALYSIS.md"},
            "content_preview": "PASSED",
        },
    )
    source_based = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": "awk -F, '{print $0}' invoices.csv && grep -q 'US: 299.50' ANALYSIS.md"
            },
            "content_preview": "PASSED",
        },
    )
    source_only = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "awk -F, '{sum += $3} END {print sum}' invoices.csv"},
            "content_preview": "PASSED\n299.50",
        },
    )

    ok, results = evaluate_node_evidence(
        node=verify,
        result="Verified report.",
        activity=[artifact_only],
        run=run,
    )

    assert ok is False
    assert "did not compare dependency artifacts with source inputs" in results[0].message

    ok, results = evaluate_node_evidence(
        node=verify,
        result="Recomputed source values.",
        activity=[source_only],
        run=run,
    )

    assert ok is False
    assert "did not compare dependency artifacts with source inputs" in results[0].message

    ok, results = evaluate_node_evidence(
        node=verify,
        result="Verified report from source.",
        activity=[source_based],
        run=run,
    )

    assert ok is True
    assert results[0].message == "verify_work passed"

    presence_only = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "test -s ANALYSIS.md && test -s invoices.csv"},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=verify,
        result="Verified files exist.",
        activity=[presence_only],
        run=run,
    )

    assert ok is False
    assert "did not compare dependency artifacts with source inputs" in results[0].message


def test_verify_node_accepts_shell_redirect_artifact_compare() -> None:
    work_summary = summarize_activity(
        [
            ActivityEvent(
                session_id="s",
                kind="tool_call.completed",
                data={
                    "name": "read_file",
                    "is_error": False,
                    "arguments": {"path": "data.json"},
                    "content_preview": '{"items":[]}',
                },
            ),
            ActivityEvent(
                session_id="s",
                kind="tool_call.completed",
                data={
                    "name": "shell",
                    "is_error": False,
                    "arguments": {
                        "command": "jq -r '.items[].name' data.json | sort > sorted_names.txt"
                    },
                    "content_preview": "exit_code: 0",
                },
            ),
        ]
    )
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="work",
        status="completed",
        metadata={"activity_summary": work_summary},
    )
    verify = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        depends_on=("work",),
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="verify names", nodes=(work, verify))
    source_compare = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    'python3 -c \'import json; data=json.load(open("data.json")); '
                    'print("\\n".join(sorted(item["name"] for item in data["items"])))\' '
                    "| diff - sorted_names.txt"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=verify,
        result="Verified names from source.",
        activity=[source_compare],
        run=run,
    )

    assert work_summary["changed_paths"] == ["sorted_names.txt"]
    assert ok is True
    assert results[0].message == "verify_work passed"


def test_work_node_can_defer_source_verification_to_dedicated_verify_node() -> None:
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create report",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="files_changed"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
    )
    verify = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        depends_on=("work",),
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="Produce ANALYSIS.md.", nodes=(work, verify))
    read_source = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "read_file",
            "is_error": False,
            "arguments": {"path": "invoices.csv"},
            "content_preview": "region,amount,status\nUS,299.50,paid",
        },
    )
    write = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "ANALYSIS.md", "content": "US: 299.50"},
            "content_preview": "wrote ANALYSIS.md",
        },
    )
    weak_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "grep -q 'US: 299.50' ANALYSIS.md"},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=work,
        result="Created ANALYSIS.md; verification remains for the verify node.",
        activity=[read_source, write, weak_verify],
        run=run,
    )

    assert ok is True
    assert [item.status for item in results] == ["passed", "skipped"]
    assert results[1].message == "verification deferred to dedicated verify node"


def test_work_retry_can_defer_previous_mutation_to_dedicated_verify_node() -> None:
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create report",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="files_changed"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
        metadata={
            "failure_history": [
                {
                    "activity_summary": {
                        "state_changed": True,
                        "verify_work_passed": False,
                    },
                    "evidence_results": [
                        {
                            "requirement": {"kind": "verify_work_if_state_changed"},
                            "status": "skipped",
                        }
                    ],
                }
            ]
        },
    )
    verify = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        depends_on=("work",),
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="Produce ANALYSIS.md.", nodes=(work, verify))
    read = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "read_file",
            "is_error": False,
            "arguments": {"path": "ANALYSIS.md"},
            "content_preview": "US: 299.50",
        },
    )

    ok, results = evaluate_node_evidence(
        node=work,
        result="ANALYSIS.md exists; final verification remains for the verify node.",
        activity=[read],
        run=run,
    )

    assert ok is True
    assert [item.status for item in results] == ["passed", "skipped"]
    assert results[1].message == "verification deferred to dedicated verify node"


def test_no_state_change_rejects_mutating_verify_work_command() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="Verify result.txt.",
        expected_evidence=(EvidenceRequirement(kind="no_state_change"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="verify result.txt", nodes=(node,))
    mutating_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "printf '%s' truth > result.txt"},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="verified",
        activity=[mutating_verify],
        run=run,
    )

    assert ok is False
    assert results[0].message == "read-only node changed workspace state"

    truncating_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "truncate -s 0 result.txt"},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="verified",
        activity=[truncating_verify],
        run=run,
    )

    assert ok is False
    assert results[0].message == "read-only node changed workspace state"


def test_no_state_change_allows_quoted_python_verifier_comparisons() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="Verify report.",
        expected_evidence=(EvidenceRequirement(kind="no_state_change"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="verify report", nodes=(node,))
    verifier = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    "python3 -c 'values=[1, 2, 3]; assert len(values) > 1; assert max(values) >= 3'"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="verified",
        activity=[verifier],
        run=run,
    )

    assert ok is True
    assert results[0].message == "node did not change workspace state"


def test_no_state_change_rejects_failed_shell_redirection() -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        prompt="Inspect without mutation.",
        expected_evidence=(EvidenceRequirement(kind="no_state_change"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="inspect", nodes=(node,))
    failed_redirect = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": True,
            "arguments": {"command": "missing-json-tool input.json > transformed.json"},
            "content_preview": "exit_code: 127\ncommand not found",
            "metadata": {"exit_code": 127},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="inspection failed",
        activity=[failed_redirect],
        run=run,
    )

    assert ok is False
    assert results[0].message == "read-only node changed workspace state"


def test_no_state_change_allows_refused_read_only_shell_mutation() -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        prompt="Inspect without mutation.",
        expected_evidence=(EvidenceRequirement(kind="no_state_change"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="inspect", nodes=(node,))
    refused_redirect = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": True,
            "arguments": {"command": "missing-json-tool input.json > transformed.json"},
            "content_preview": "refused read-only shell command",
            "metadata": {"read_only_shell_refused": True},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="inspection failed",
        activity=[refused_redirect],
        run=run,
    )

    assert ok is True
    assert [result.message for result in results] == ["node did not change workspace state"]


def test_no_failed_tools_ignores_refused_read_only_shell_mutation() -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        prompt="Inspect without mutation.",
        expected_evidence=(EvidenceRequirement(kind="no_failed_tools"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="inspect", nodes=(node,))
    refused = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": True,
            "arguments": {"command": "external-check tasks/example"},
            "content_preview": "refused read-only shell command",
            "metadata": {"read_only_shell_refused": True},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Gathered enough read-only evidence without running the mutating check.",
        activity=[refused],
        run=run,
    )

    assert ok is True
    assert results[0].message == "no failed tool calls"


def test_no_state_change_allows_shell_command_noop() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Inspect environment.",
        expected_evidence=(EvidenceRequirement(kind="no_state_change"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="inspect", nodes=(node,))
    noop = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {"command": "command"},
            "content_preview": "exit_code: 0\n\n(no output)",
            "metadata": {"exit_code": 0},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="No-op shell probe ran.",
        activity=[noop],
        run=run,
    )

    assert ok is True
    assert results[0].message == "node did not change workspace state"


def test_research_can_use_successful_web_evidence_when_fetch_fails() -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        prompt="Find two current public web results.",
        expected_evidence=(
            EvidenceRequirement(kind="result_nonempty"),
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="no_state_change"),
            EvidenceRequirement(kind="no_failed_tools"),
        ),
    )
    run = WorkflowRun(id="wf", title="WF", goal="Find OpenAI results", nodes=(node,))
    search = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "web_search",
            "is_error": False,
            "arguments": {"query": "OpenAI latest news"},
            "content_preview": (
                "Results for: OpenAI latest news\n1. OpenAI News\nURL: https://openai.com/news/"
            ),
        },
    )
    failed_fetch = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "fetch_url",
            "is_error": True,
            "arguments": {"url": "https://openai.com/news/"},
            "content_preview": "HTTP 403",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="OpenAI News is listed at https://openai.com/news/.",
        activity=[search, failed_fetch],
        run=run,
    )

    assert ok is True
    assert [item.message for item in results][-1] == "no failed tool calls"

    ok, results = evaluate_node_evidence(
        node=node,
        result="I fetched https://openai.com/news/ and summarized the page.",
        activity=[search, failed_fetch],
        run=run,
    )

    assert ok is False
    assert "tool call(s) returned errors" in results[-1].message


def test_claim_grounding_does_not_treat_urls_as_absolute_paths() -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        prompt="Find public web results.",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="Find OpenAI results", nodes=(node,))
    search = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "web_search",
            "is_error": False,
            "arguments": {"query": "OpenAI latest news"},
            "content_preview": ("OpenAI News | Reuters https://www.reuters.com/technology/openai/"),
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Reuters lists OpenAI updates at https://www.reuters.com/technology/openai/.",
        activity=[search],
        run=run,
    )

    assert ok is True
    assert results[0].message == "result claims are grounded in evidence"


def test_web_result_requests_require_source_urls_from_evidence() -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        prompt="Find two public web results.",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Find two current public web results about OpenAI and summarize them.",
        nodes=(node,),
    )
    search = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "web_search",
            "is_error": False,
            "arguments": {"query": "OpenAI latest news"},
            "metadata": {
                "results": [
                    {
                        "title": "OpenAI News",
                        "content": "Official updates from OpenAI.",
                        "url": "https://openai.com/news/",
                    },
                    {
                        "title": "Reuters OpenAI",
                        "content": "Latest OpenAI reporting.",
                        "url": "https://www.reuters.com/technology/openai/",
                    },
                ]
            },
            "content_preview": "OpenAI News and Reuters OpenAI",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="OpenAI News and Reuters have current public results.",
        activity=[search],
        run=run,
    )

    assert ok is False
    assert "source URL" in results[0].message

    ok, results = evaluate_node_evidence(
        node=node,
        result=(
            "OpenAI News: https://openai.com/news/\n"
            "Reuters OpenAI: https://www.reuters.com/technology/openai/"
        ),
        activity=[search],
        run=run,
    )

    assert ok is True
    assert results[0].message == "result claims are grounded in evidence"


def test_internal_source_url_prompt_does_not_make_coding_research_citation_gated() -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        prompt=(
            "If a current or live value cannot be directly verified, report that "
            "limitation with the source URLs found instead of substituting unsupported estimates."
        ),
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create a Python script named word_count.py and verify it works.",
        nodes=(node,),
    )
    search = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "web_search",
            "is_error": False,
            "arguments": {"query": "how to check python installed"},
            "metadata": {
                "results": [
                    {
                        "title": "Python docs",
                        "content": "Python command line usage.",
                        "url": "https://docs.python.org/3/using/cmdline.html",
                    }
                ]
            },
            "content_preview": "Python command line usage.",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Workspace is empty and ready for creating the script.",
        activity=[search],
        run=run,
    )

    assert ok is True
    assert results[0].message == "result claims are grounded in evidence"


def test_requested_tests_require_test_file_and_test_run_evidence() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Complete the coding task.",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal=(
            "Create a Python analyzer, add a unittest file that verifies the math, "
            "and run the tests."
        ),
        nodes=(node,),
    )
    partial_activity = [
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "write_file",
                "is_error": False,
                "arguments": {"path": "expense_report.py", "content": "print('ok')"},
                "content_preview": "wrote expense_report.py",
            },
        ),
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "verify_work",
                "is_error": False,
                "arguments": {"command": "python3 expense_report.py"},
                "content_preview": "PASSED",
            },
        ),
    ]

    ok, results = evaluate_node_evidence(
        node=node,
        result="Created the analyzer and ran it.",
        activity=partial_activity,
        run=run,
    )

    assert ok is False
    assert results[0].message == "requested test file is not grounded in evidence"

    complete_activity = [
        partial_activity[0],
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "write_file",
                "is_error": False,
                "arguments": {
                    "path": "test_expense_report.py",
                    "content": "import unittest\n",
                },
                "content_preview": "wrote test_expense_report.py",
            },
        ),
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "verify_work",
                "is_error": False,
                "arguments": {"command": "python3 -m unittest -v test_expense_report.py"},
                "content_preview": "PASSED",
            },
        ),
    ]

    ok, results = evaluate_node_evidence(
        node=node,
        result="Created the analyzer, added test_expense_report.py, and ran unittest.",
        activity=complete_activity,
        run=run,
    )

    assert ok is True
    assert results[0].message == "result claims are grounded in evidence"


def test_workflow_store_lists_newest_first(tmp_path: Path) -> None:
    store = WorkflowStore(root=default_workflow_root(tmp_path))
    first = create_default_workflow(
        workflow_id=store.new_id("First"),
        title="First",
        goal="first",
    )
    second = create_default_workflow(
        workflow_id=store.new_id("Second"),
        title="Second",
        goal="second",
    )
    store.add_run(first)
    store.add_run(second)

    assert [item.id for item in store.list_runs()] == [second.id, first.id]


def test_dynamic_plan_spec_is_validated_and_normalized(tmp_path: Path) -> None:
    store = WorkflowStore(root=default_workflow_root(tmp_path))
    run = create_workflow_from_plan_spec(
        workflow_id=store.new_id("Dynamic"),
        title="Dynamic",
        goal="ship defended workflows",
        plan={
            "nodes": [
                {
                    "id": "Inspect",
                    "title": "Inspect code",
                    "kind": "research",
                    "prompt": "Inspect the workflow code.",
                    "expected_evidence": ["result_nonempty", "no_failed_tools"],
                },
                {
                    "id": "Patch",
                    "title": "Patch code",
                    "kind": "work",
                    "depends_on": ["Inspect"],
                    "prompt": "Patch the workflow code.",
                    "allow_mutation": True,
                    "expected_evidence": [
                        {"kind": "result_nonempty"},
                    ],
                },
                {
                    "id": "Verify Output",
                    "title": "Verify output",
                    "kind": "verify",
                    "depends_on": ["Patch"],
                    "prompt": "Verify the patch.",
                    "expected_evidence": [{"kind": "result_nonempty"}],
                },
            ]
        },
    )

    assert [node.id for node in run.nodes] == ["inspect", "patch", "verify-output", "merge"]
    assert run.nodes[-1].depends_on == ("verify-output",)
    evidence = {item.kind for item in run.nodes[1].expected_evidence}
    assert "files_changed" in evidence
    assert "environment_checked" in evidence
    assert "verify_work_if_state_changed" in evidence
    verify_evidence = {item.kind for item in run.nodes[2].expected_evidence}
    assert "verify_work_passed" in verify_evidence
    assert "tool_called" in verify_evidence


def test_dynamic_plan_rejects_cycles_and_unknown_evidence(tmp_path: Path) -> None:
    store = WorkflowStore(root=default_workflow_root(tmp_path))

    with pytest.raises(ValueError, match="cycle"):
        create_workflow_from_plan_spec(
            workflow_id=store.new_id("Cycle"),
            title="Cycle",
            goal="reject cycle",
            plan={
                "nodes": [
                    {
                        "id": "a",
                        "title": "A",
                        "kind": "research",
                        "prompt": "a",
                        "depends_on": ["b"],
                    },
                    {
                        "id": "b",
                        "title": "B",
                        "kind": "research",
                        "prompt": "b",
                        "depends_on": ["a"],
                    },
                ]
            },
        )

    with pytest.raises(ValueError, match="unknown evidence"):
        create_workflow_from_plan_spec(
            workflow_id=store.new_id("Evidence"),
            title="Evidence",
            goal="reject evidence",
            plan={
                "nodes": [
                    {
                        "id": "a",
                        "title": "A",
                        "kind": "research",
                        "prompt": "a",
                        "expected_evidence": ["made_up_evidence"],
                    }
                ]
            },
        )


def test_render_workflow_mermaid_includes_statuses() -> None:
    run = WorkflowRun(
        id="wf",
        title="Demo",
        goal="demo",
        nodes=(
            WorkflowNode(
                id="a",
                title="A",
                kind="plan",
                prompt="a",
                expected_evidence=(EvidenceRequirement(kind="result_nonempty"),),
                status="completed",
            ),
            WorkflowNode(id="b", title="B", kind="merge", prompt="b", depends_on=("a",)),
        ),
    )

    graph = render_workflow_mermaid(run)
    assert "a --> b" in graph
    assert "a: A (completed)" in graph


def test_evidence_requires_verify_work_after_state_change() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="work",
        allow_mutation=True,
        expected_evidence=(EvidenceRequirement(kind="verify_work_if_state_changed"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(node,))
    changed = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "edit_file",
            "is_error": False,
            "arguments": {"path": "app.py"},
            "content_preview": "edited",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed],
        run=run,
    )
    assert ok is False
    assert "without a later passing verify_work" in results[0].message

    verified = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "pytest"},
            "content_preview": "PASSED",
        },
    )
    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, verified],
        run=run,
    )
    assert ok is True
    assert results[0].status == "passed"


def test_exact_file_content_requires_assertive_verify_work_after_state_change() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create result.txt containing exactly 'truth'",
        allow_mutation=True,
        expected_evidence=(EvidenceRequirement(kind="verify_work_if_state_changed"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal=(
            "Create result.txt containing exactly 'truth' and verify by reading the file "
            "and checking byte size."
        ),
        nodes=(node,),
    )
    changed = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "result.txt", "content": "truth"},
            "content_preview": "wrote 5 bytes",
        },
    )
    print_only_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "cat result.txt; wc -c < result.txt"},
            "content_preview": "PASSED\n\ntruth       5",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, print_only_verify],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[0].message

    echo_only_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "echo truth result.txt"},
            "content_preview": "PASSED\n\ntruth result.txt",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, echo_only_verify],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[0].message

    content_only_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'test -f result.txt && [ "$(cat result.txt)" = "truth" ]'},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, content_only_assertion],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[0].message

    unrelated_grep_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    "echo truth; grep -qx wrong result.txt; [ $(wc -c < result.txt) -eq 5 ]"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, unrelated_grep_assertion],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[0].message

    unrelated_diff_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    "echo truth; diff result.txt result.txt; [ $(wc -c < result.txt) -eq 5 ]"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, unrelated_diff_assertion],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[0].message

    unrelated_byte_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    'test -f result.txt && [ "$(cat result.txt)" = "truth" ] '
                    "&& [ 5 -eq 5 ] && wc -c < result.txt"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, unrelated_byte_assertion],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[0].message

    echoed_substitution_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    'test "$(cat result.txt >/dev/null; echo truth)" = "truth" '
                    "&& [ $(wc -c < result.txt) -eq 5 ]"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, echoed_substitution_assertion],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[0].message

    grep_side_input_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    "grep -qx 'truth' <(printf %s truth) result.txt "
                    "&& [ $(wc -c < result.txt) -eq 5 ]"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, grep_side_input_assertion],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[0].message

    masked_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    'test "$(cat result.txt)" = "truth" || true && [ $(wc -c < result.txt) -eq 5 ]'
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, masked_assertion],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[0].message

    assertive_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    'test -f result.txt && [ "$(cat result.txt)" = "truth" ] '
                    "&& [ $(wc -c < result.txt) -eq 5 ]"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, assertive_verify],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"


def test_exact_file_content_accepts_grep_and_cmp_assertions() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create result.txt containing exactly 'truth'",
        allow_mutation=True,
        expected_evidence=(EvidenceRequirement(kind="verify_work_if_state_changed"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal=(
            "Create result.txt containing exactly 'truth' and verify by reading the file "
            "and checking byte size."
        ),
        nodes=(node,),
    )
    changed = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "result.txt", "content": "truth"},
            "content_preview": "wrote 5 bytes",
        },
    )

    grep_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": "grep -qx 'truth' result.txt && [ $(wc -c < result.txt) -eq 5 ]"
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, grep_verify],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"

    cmp_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    "cmp <(printf %s 'truth') result.txt && [ $(wc -c < result.txt) -eq 5 ]"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, cmp_verify],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"


def test_no_trailing_newline_requires_byte_size_without_verify_prompt() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create result.txt containing exactly 'truth' with no trailing newline",
        allow_mutation=True,
        expected_evidence=(EvidenceRequirement(kind="verify_work_if_state_changed"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create result.txt containing exactly 'truth' with no trailing newline.",
        nodes=(node,),
    )
    changed = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "result.txt", "content": "truth"},
            "content_preview": "wrote 5 bytes",
        },
    )
    content_only = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'test "$(cat result.txt)" = "truth"'},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, content_only],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[0].message

    with_byte_size = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": ('test "$(cat result.txt)" = "truth" && [ $(wc -c < result.txt) -eq 5 ]')
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, with_byte_size],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"

    split_content_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'test "$(cat result.txt)" = "truth"'},
            "content_preview": "PASSED",
        },
    )
    split_byte_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "test $(wc -c < result.txt) -eq 5"},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, split_content_assertion, split_byte_assertion],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"

    shell_read_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    '([ "$(< result.txt)" = "truth" ] '
                    "&& [ \"$(wc -c < result.txt | tr -d ' ')\" -eq 5 ]) || exit 1"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, shell_read_assertion],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"


def test_exact_file_content_without_byte_size_accepts_content_assertion() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create result.txt containing exactly 'truth'",
        allow_mutation=True,
        expected_evidence=(EvidenceRequirement(kind="verify_work_if_state_changed"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create result.txt containing exactly 'truth' and verify it.",
        nodes=(node,),
    )
    changed = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "result.txt", "content": "truth"},
            "content_preview": "wrote 5 bytes",
        },
    )
    content_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'test -f result.txt && [ "$(cat result.txt)" = "truth" ]'},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, content_assertion],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"

    unquoted_content_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'test "$(cat result.txt)" = truth'},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, unquoted_content_assertion],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"


def test_exact_file_content_rejects_echo_only_verify_work_evidence() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Create hello.txt containing exactly 'verified hello' and verify it.",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create hello.txt containing exactly 'verified hello' and verify it.",
        nodes=(node,),
    )
    echo_only = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "echo 'verified hello' hello.txt"},
            "content_preview": "PASSED\n\nverified hello",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='hello.txt was created with exactly "verified hello".',
        activity=[echo_only],
        run=run,
    )

    assert ok is False
    assert "requested exact content" in results[0].message


def test_research_can_discuss_exact_content_without_final_content_evidence() -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        prompt="Check assumptions before work.",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create answer.txt containing exactly `hello` with no trailing newline.",
        nodes=(node,),
    )
    checked_workspace = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "list_dir",
            "is_error": False,
            "arguments": {"path": "."},
            "content_preview": ".",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result=(
            "The research phase is complete. Use printf to write hello to answer.txt "
            "without a trailing newline, then verify the byte count."
        ),
        activity=[checked_workspace],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"


def test_acknowledged_unavailable_research_tool_does_not_count_as_state_change_or_failure() -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        prompt="Check assumptions before work.",
        expected_evidence=(
            EvidenceRequirement(kind="no_state_change"),
            EvidenceRequirement(kind="no_failed_tools"),
        ),
    )
    run = WorkflowRun(id="wf", title="WF", goal="Create answer.txt.", nodes=(node,))
    failed_shell = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": True,
            "arguments": {"command": "printf hello > answer.txt"},
            "content_preview": "unknown tool: 'shell'",
        },
    )
    list_dir = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "list_dir",
            "is_error": False,
            "arguments": {"path": "."},
            "content_preview": ".",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Could not run shell from the research role, so I used read-only evidence.",
        activity=[failed_shell, list_dir],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"
    assert results[0].message == "node did not change workspace state"
    assert results[1].status == "passed"
    assert results[1].message == "no failed tool calls"


def test_research_accepts_acknowledged_missing_path_as_negative_evidence() -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        prompt="Check whether hello_weather.py exists before work starts.",
        expected_evidence=(
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="no_failed_tools"),
            EvidenceRequirement(kind="no_state_change"),
        ),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create hello_weather.py that prints exactly sunny.",
        nodes=(node,),
    )
    missing_read = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "read_file",
            "is_error": True,
            "arguments": {"path": "hello_weather.py"},
            "content_preview": "path does not exist: hello_weather.py",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="The read_file check shows hello_weather.py does not currently exist.",
        activity=[missing_read],
        run=run,
    )

    assert ok is True
    assert [item.status for item in results] == ["passed", "passed", "passed"]


def test_research_ignores_recovered_empty_argument_tool_call() -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        prompt="Inspect the repository before work.",
        expected_evidence=(
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="no_failed_tools"),
            EvidenceRequirement(kind="no_state_change"),
        ),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Report the local README.",
        nodes=(node,),
    )
    malformed_read = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "read_file",
            "is_error": True,
            "arguments": {},
            "content_preview": "missing or invalid `path` argument",
        },
    )
    recovered_read = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "read_file",
            "is_error": False,
            "arguments": {"path": "README.md"},
            "content_preview": "DeepSWE benchmark",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="README.md describes the DeepSWE benchmark.",
        activity=[malformed_read, recovered_read],
        run=run,
    )

    assert ok is True
    assert [item.status for item in results] == ["passed", "passed", "passed"]


def test_exact_stdout_requires_program_output_after_state_change() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Create hello.py that prints exactly 'harness-ok'.",
        allow_mutation=True,
        expected_evidence=(EvidenceRequirement(kind="verify_work_if_state_changed"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create hello.py that prints exactly 'harness-ok'.",
        nodes=(node,),
    )
    changed = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "hello.py", "content": "print('harness-ok')"},
            "content_preview": "wrote hello.py",
        },
    )
    echo_only = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "echo harness-ok hello.py"},
            "content_preview": "PASSED\n\nharness-ok hello.py",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, echo_only],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    wrong_output = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "python3 hello.py"},
            "content_preview": "PASSED\n\nwrong",
            "metadata": {"stdout": "wrong\n", "exit_code": 0},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, wrong_output],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    commented_direct_run = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "python3 hello.py # || false"},
            "content_preview": "PASSED\n\nharness-ok",
            "metadata": {"stdout": "harness-ok\n", "exit_code": 0},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, commented_direct_run],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"

    failed_output = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "python3 hello.py"},
            "content_preview": "PASSED\n\nharness-ok",
            "metadata": {"stdout": "harness-ok\n", "exit_code": 1},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, failed_output],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    command_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'test "$(python3 hello.py)" = "harness-ok"'},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, command_assertion],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"

    commented_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'test "$(python3 hello.py)" = "harness-ok" # || true'},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, commented_assertion],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"

    unreachable_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'false && test "$(python3 hello.py)" = "harness-ok"'},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, unreachable_assertion],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    trailing_failure_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'test "$(python3 hello.py)" = "harness-ok" && false'},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, trailing_failure_assertion],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    grouped_failure_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'test "$(python3 hello.py)" = "harness-ok" && (false)'},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, grouped_failure_assertion],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    indirect_failure_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'test "$(python3 hello.py)" = "harness-ok" && sh -c false'},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, indirect_failure_assertion],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    backgrounded_failure_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'test "$(python3 hello.py)" = "harness-ok" & false'},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, backgrounded_failure_assertion],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    env_wrapped_failure_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'test "$(python3 hello.py)" = "harness-ok" && env false'},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, env_wrapped_failure_assertion],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    command_wrapped_failure_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'test "$(python3 hello.py)" = "harness-ok" && command false'},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, command_wrapped_failure_assertion],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    pipeline_failure_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'test "$(python3 hello.py)" = "harness-ok" && true | false'},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, pipeline_failure_assertion],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    skipped_branch_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'false && true && test "$(python3 hello.py)" = "harness-ok"'},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, skipped_branch_assertion],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    skipped_if_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": 'if false; then test "$(python3 hello.py)" = "harness-ok"; fi'
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, skipped_if_assertion],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    byte_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "./hello.py | cmp - <(printf %s 'harness-ok')"},
            "content_preview": "PASSED",
            "metadata": {"stdout": "", "exit_code": 0},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, byte_assertion],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"

    direct_output = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "python3 hello.py"},
            "content_preview": "PASSED\n\nharness-ok",
            "metadata": {"stdout": "harness-ok\n", "exit_code": 0},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, direct_output],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"

    direct_output_with_stderr_failure = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "python3 hello.py"},
            "content_preview": "PASSED\n\nharness-ok",
            "metadata": {
                "stdout": "harness-ok\n",
                "stderr": ("Traceback (most recent call last):\nAssertionError: wrong result\n"),
                "exit_code": 0,
            },
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, direct_output_with_stderr_failure],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    backgrounded_failure_output = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "python3 hello.py & false"},
            "content_preview": "PASSED\n\nharness-ok",
            "metadata": {"stdout": "harness-ok\n", "exit_code": 0},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, backgrounded_failure_output],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    trailing_failure_output = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "python3 hello.py && false"},
            "content_preview": "PASSED\n\nharness-ok",
            "metadata": {"stdout": "harness-ok\n", "exit_code": 0},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, trailing_failure_output],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message


def test_exact_stdout_no_trailing_newline_requires_raw_direct_output() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Create hello.py that prints exactly 'harness-ok' with no trailing newline.",
        allow_mutation=True,
        expected_evidence=(EvidenceRequirement(kind="verify_work_if_state_changed"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create hello.py that prints exactly 'harness-ok' with no trailing newline.",
        nodes=(node,),
    )
    changed = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "hello.py", "content": "print('harness-ok')"},
            "content_preview": "wrote hello.py",
        },
    )
    command_substitution = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'test "$(python3 hello.py)" = "harness-ok"'},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, command_substitution],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    transformed_output = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "python3 hello.py | tr -d '\\n'"},
            "content_preview": "PASSED\n\nharness-ok",
            "metadata": {"stdout": "harness-ok", "exit_code": 0},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, transformed_output],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    trailing_newline = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "python3 hello.py"},
            "content_preview": "PASSED\n\nharness-ok",
            "metadata": {"stdout": "harness-ok\n", "exit_code": 0},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, trailing_newline],
        run=run,
    )

    assert ok is False
    assert "exact requested output" in results[0].message

    byte_assertion = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "./hello.py | cmp - <(printf %s 'harness-ok')"},
            "content_preview": "PASSED",
            "metadata": {"stdout": "", "exit_code": 0},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, byte_assertion],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"

    exact_raw_output = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "python3 hello.py"},
            "content_preview": "PASSED\n\nharness-ok",
            "metadata": {"stdout": "harness-ok", "exit_code": 0},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[changed, exact_raw_output],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"


def test_exact_stdout_claim_grounding_rejects_echo_only_evidence() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Create hello.py that prints exactly 'harness-ok'.",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create hello.py that prints exactly 'harness-ok'.",
        nodes=(node,),
    )
    echo_only = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "echo harness-ok hello.py"},
            "content_preview": "PASSED\n\nharness-ok hello.py",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="hello.py prints exactly harness-ok.",
        activity=[echo_only],
        run=run,
    )

    assert ok is False
    assert "requested exact output" in results[0].message

    direct_output = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "python3 hello.py"},
            "content_preview": "PASSED\n\nharness-ok",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="hello.py prints exactly harness-ok.",
        activity=[direct_output],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"


def test_objective_status_uses_structured_flag_not_phrase_matching() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Write a Python script to get the weather.",
        expected_evidence=(EvidenceRequirement(kind="objective_status"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Write a Python script to get the weather.",
        nodes=(node,),
    )
    write = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {
                "path": "get_weather.py",
                "content": "API_KEY = 'YOUR_API_KEY'\nprint(API_KEY)\n",
            },
            "content_preview": "wrote get_weather.py",
        },
    )
    verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "python get_weather.py London"},
            "content_preview": "PASSED\n\n401 Client Error: Unauthorized for url",
            "metadata": {
                "stdout": "401 Client Error: Unauthorized for url\n",
                "exit_code": 0,
            },
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='{"status":"retry","reason":"the objective is not complete yet"}',
        activity=[write, verify],
        run=run,
    )

    assert ok is False
    assert results[0].message == "objective status is retry"

    ok, results = evaluate_node_evidence(
        node=node,
        result='{"status":"pass","reason":"verified by concrete evidence"}',
        activity=[write, verify],
        run=run,
    )

    assert ok is True
    assert results[0].message == "objective status is pass"


def test_workflow_decision_uses_final_status_object_after_evidence_json() -> None:
    assert (
        workflow_decision(
            'read_file profile.json: {"name":"Ada","languages":["Python"]}\n\n{"status":"pass"}'
        )
        == "pass"
    )


def test_objective_status_ignores_tool_output_status_flags() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Write a Python script.",
        expected_evidence=(EvidenceRequirement(kind="objective_status"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="Write a Python script.", nodes=(node,))
    shell_echo = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {"command": 'echo \'{"status":"pass"}\''},
            "content_preview": 'exit_code: 0\n\nstdout:\n{"status":"pass"}\n',
            "metadata": {"stdout": '{"status":"pass"}\n', "exit_code": 0},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="",
        activity=[shell_echo],
        run=run,
    )

    assert ok is False
    assert results[0].message == "objective status is missing"


def test_claim_grounded_rejects_pseudo_tool_syntax_without_semantic_word_match() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Write a Python script to get the weather.",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Write a Python script to get the weather.",
        nodes=(node,),
    )
    write = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "get_weather.py", "content": "print('ok')\n"},
            "content_preview": "wrote get_weather.py",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='I will continue. [web_search(query="next step")]',
        activity=[write],
        run=run,
    )

    assert ok is False
    assert "pseudo tool syntax" in results[0].message

    ok, results = evaluate_node_evidence(
        node=node,
        result='```tool_response\n[shell(command="echo should have been a tool call")]\n```',
        activity=[write],
        run=run,
    )

    assert ok is False
    assert "pseudo tool syntax" in results[0].message

    ok, results = evaluate_node_evidence(
        node=node,
        result="<|tool_call>call:glob{pattern: 'weather_tokyo.py'}<tool_call|>",
        activity=[write],
        run=run,
    )

    assert ok is False
    assert "pseudo tool syntax" in results[0].message

    ok, results = evaluate_node_evidence(
        node=node,
        result='I checked it with [shell(command="echo should have been a tool call")].',
        activity=[write],
        run=run,
    )

    assert ok is False
    assert "pseudo tool syntax" in results[0].message


def test_exact_file_content_parser_stops_quoted_content_before_method_clause() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create shell-result.txt containing exactly 'shelltruth' by shell redirection",
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create shell-result.txt containing exactly 'shelltruth' by shell redirection.",
        nodes=(node,),
    )

    assert _exact_file_content_requests(node=node, run=run) == [("shell-result.txt", "shelltruth")]


def test_exact_stdout_parser_handles_named_script_and_no_newline_clause() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt=(
            "Create a Python script named hello.py that prints exactly 'harness-ok' "
            "with no trailing newline."
        ),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal=(
            "Create a Python script named hello.py that prints exactly 'harness-ok' "
            "with no trailing newline."
        ),
        nodes=(node,),
    )

    assert _exact_stdout_requests(node=node, run=run) == [("hello.py", "harness-ok")]


def test_exact_file_content_parser_stops_quoted_content_before_no_newline_clause() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt=(
            "create shell-result.txt containing exactly 'shelltruth' with no trailing "
            "newline, then verify it"
        ),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal=(
            "Create shell-result.txt containing exactly 'shelltruth' with no trailing "
            "newline, then verify it."
        ),
        nodes=(node,),
    )

    assert _exact_file_content_requests(node=node, run=run) == [("shell-result.txt", "shelltruth")]


def test_exact_file_content_parser_accepts_named_file_content_variants() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create a file named result.txt with content exactly 'shelltruth'",
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create a file named result.txt with content exactly 'shelltruth'.",
        nodes=(node,),
    )

    assert _exact_file_content_requests(node=node, run=run) == [("result.txt", "shelltruth")]


def test_exact_file_content_parser_accepts_should_say_clause() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt=(
            "Create GREETING.txt from profile.json. It should say: Hello Ada, Python count is 4."
        ),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal=(
            "Create GREETING.txt from profile.json. It should say: Hello Ada, Python count is 4."
        ),
        nodes=(node,),
    )

    assert _exact_file_content_requests(node=node, run=run) == [
        ("GREETING.txt", "Hello Ada, Python count is 4.")
    ]


def test_shell_write_requires_and_accepts_later_assertive_verify_work() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create result.txt containing exactly 'truth'",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal=(
            "Create result.txt containing exactly 'truth' and verify by reading the file "
            "and checking byte size."
        ),
        nodes=(node,),
    )
    shell_write = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {"command": "printf %s truth > result.txt"},
            "content_preview": "exit_code: 0",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='result.txt was created with exactly "truth".',
        activity=[shell_write],
        run=run,
    )

    assert ok is False
    assert results[1].status == "failed"
    assert "passing verify_work assertion" in results[1].message

    assertive_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    'test -f result.txt && [ "$(cat result.txt)" = "truth" ] '
                    "&& [ $(wc -c < result.txt) -eq 5 ]"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='result.txt was created with exactly "truth".',
        activity=[shell_write, assertive_verify],
        run=run,
    )

    assert ok is True
    assert [item.status for item in results] == ["passed", "passed"]

    unreachable_assertive_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    'false && [ "$(cat result.txt)" = "truth" ] && [ $(wc -c < result.txt) -eq 5 ]'
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='result.txt was created with exactly "truth".',
        activity=[shell_write, unreachable_assertive_verify],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[1].message

    grouped_failure_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    '[ "$(cat result.txt)" = "truth" ] '
                    "&& [ $(wc -c < result.txt) -eq 5 ] && (false)"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='result.txt was created with exactly "truth".',
        activity=[shell_write, grouped_failure_verify],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[1].message

    indirect_failure_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    '[ "$(cat result.txt)" = "truth" ] '
                    "&& [ $(wc -c < result.txt) -eq 5 ] && sh -c false"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='result.txt was created with exactly "truth".',
        activity=[shell_write, indirect_failure_verify],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[1].message

    backgrounded_failure_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    '[ "$(cat result.txt)" = "truth" ] && [ $(wc -c < result.txt) -eq 5 ] & false'
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='result.txt was created with exactly "truth".',
        activity=[shell_write, backgrounded_failure_verify],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[1].message

    env_wrapped_failure_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    '[ "$(cat result.txt)" = "truth" ] '
                    "&& [ $(wc -c < result.txt) -eq 5 ] && env false"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='result.txt was created with exactly "truth".',
        activity=[shell_write, env_wrapped_failure_verify],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[1].message

    command_wrapped_failure_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    '[ "$(cat result.txt)" = "truth" ] '
                    "&& [ $(wc -c < result.txt) -eq 5 ] && command false"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='result.txt was created with exactly "truth".',
        activity=[shell_write, command_wrapped_failure_verify],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[1].message

    pipeline_failure_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    '[ "$(cat result.txt)" = "truth" ] '
                    "&& [ $(wc -c < result.txt) -eq 5 ] && true | false"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='result.txt was created with exactly "truth".',
        activity=[shell_write, pipeline_failure_verify],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[1].message

    skipped_branch_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    'false && true && [ "$(cat result.txt)" = "truth" ] '
                    "&& [ $(wc -c < result.txt) -eq 5 ]"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='result.txt was created with exactly "truth".',
        activity=[shell_write, skipped_branch_verify],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[1].message

    skipped_if_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    'if false; then [ "$(cat result.txt)" = "truth" ] '
                    "&& [ $(wc -c < result.txt) -eq 5 ]; fi"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='result.txt was created with exactly "truth".',
        activity=[shell_write, skipped_if_verify],
        run=run,
    )

    assert ok is False
    assert "passing verify_work assertion" in results[1].message


def test_shell_heredoc_write_with_test_content_counts_as_state_change() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create a regression test file",
        allow_mutation=True,
        expected_evidence=(EvidenceRequirement(kind="files_changed"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create tests/models/test_json_streaming.py with pytest tests.",
        nodes=(node,),
    )
    shell_write = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {
                "command": (
                    "cat << 'EOF' > tests/models/test_json_streaming.py\n"
                    "import pytest\n"
                    "def test_streaming_json():\n"
                    "    assert True\n"
                    "EOF"
                )
            },
            "content_preview": "exit_code: 0\n\n(no output)",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Created tests/models/test_json_streaming.py.",
        activity=[shell_write],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"


def test_verify_work_heredoc_stdin_does_not_count_as_state_change() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create a script and verify it with stdin",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="files_changed"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create a script and verify it using stdin input.",
        nodes=(node,),
    )
    write = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "script.py", "content": "print(input())"},
            "content_preview": "wrote script.py",
        },
    )
    verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "python3 script.py <<EOF\nTokyo\nEOF\n"},
            "content_preview": "PASSED\n\nTokyo",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Created script.py and verified it using stdin.",
        activity=[write, verify],
        run=run,
    )

    assert ok is True
    assert [item.status for item in results] == ["passed", "passed"]


def test_environment_checked_requires_probe_before_first_mutation() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create a script",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="environment_checked"),
            EvidenceRequirement(kind="files_changed"),
        ),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create a script.",
        nodes=(node,),
    )
    write = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "script.py", "content": "print('ok')"},
            "content_preview": "wrote script.py",
        },
    )
    runtime_probe = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {"command": "python3 --version"},
            "content_preview": "exit_code: 0\n\nstdout:\nPython 3.14.3\n",
        },
    )
    commented_runtime_probe = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {
                "command": (
                    "# Check whether tools are installed before work\n"
                    "which python3 || echo 'python3 not found'\n"
                )
            },
            "content_preview": "exit_code: 0\n\nstdout:\n/usr/bin/python3\n",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Created script.py.",
        activity=[write],
        run=run,
    )

    assert ok is False
    assert results[0].status == "failed"
    assert "before first mutation" in results[0].message

    ok, results = evaluate_node_evidence(
        node=node,
        result="Checked the runtime and created script.py.",
        activity=[runtime_probe, write],
        run=run,
    )

    assert ok is True
    assert [item.status for item in results] == ["passed", "passed"]

    ok, results = evaluate_node_evidence(
        node=node,
        result="Checked the runtime and created script.py.",
        activity=[commented_runtime_probe, write],
        run=run,
    )

    assert ok is True
    assert [item.status for item in results] == ["passed", "passed"]


def test_head_preview_sigpipe_is_not_relevant_failed_tool() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="inspect a directory",
        allow_mutation=True,
        expected_evidence=(EvidenceRequirement(kind="no_failed_tools"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="Inspect files.", nodes=(node,))
    preview = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": True,
            "arguments": {"command": "ls -R tasks | head -n 20"},
            "content_preview": "exit_code: 141\n\nstdout:\nfile-a\nfile-b\n",
            "metadata": {"exit_code": 141, "pipefail": True},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Inspected a short preview of tasks.",
        activity=[preview],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"

    curl_preview = preview.model_copy(
        update={
            "data": {
                **preview.data,
                "arguments": {"command": 'curl -s "https://wttr.in/Tokyo?format=j1" | head -c 500'},
                "content_preview": 'exit_code: 56\n\nstdout:\n{ "current_condition": [',
                "metadata": {"exit_code": 56, "pipefail": True},
            }
        }
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Inspected a short preview of the HTTP response.",
        activity=[curl_preview],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"

    curl_head_closed_pipe_preview = preview.model_copy(
        update={
            "data": {
                **preview.data,
                "arguments": {"command": 'curl -s "https://wttr.in/Tokyo?format=j1" | head -n 20'},
                "content_preview": 'exit_code: 23\n\nstdout:\n{ "current_condition": [',
                "metadata": {"exit_code": 23, "pipefail": True},
            }
        }
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Inspected a short preview of the HTTP response.",
        activity=[curl_head_closed_pipe_preview],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"


def test_environment_checked_requires_recheck_after_installation() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="install a missing tool and use it",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="environment_checked"),
            EvidenceRequirement(kind="no_failed_tools"),
            EvidenceRequirement(kind="files_changed"),
        ),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Install a missing tool and use it.",
        nodes=(node,),
    )
    missing_probe = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": True,
            "arguments": {"command": "command -v sample-tool"},
            "content_preview": "exit_code: 1\n\nstderr:\nsample-tool not found\n",
        },
    )
    install = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {"command": "npm install -g sample-tool"},
            "content_preview": "exit_code: 0\n",
        },
    )
    recheck = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {"command": "sample-tool --version"},
            "content_preview": "exit_code: 0\n\nstdout:\n1.0.0\n",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Installed sample-tool.",
        activity=[missing_probe, install],
        run=run,
    )

    assert ok is False
    assert results[0].status == "failed"
    assert "rechecked after installation" in results[0].message
    assert results[1].status == "passed"

    ok, results = evaluate_node_evidence(
        node=node,
        result="Installed sample-tool and confirmed sample-tool --version.",
        activity=[missing_probe, install, recheck],
        run=run,
    )

    assert ok is True
    assert [item.status for item in results] == ["passed", "passed", "passed"]


def test_environment_checked_accepts_install_and_recheck_in_one_shell_command() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="install a missing tool and check it",
        allow_mutation=True,
        expected_evidence=(EvidenceRequirement(kind="environment_checked"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Install a missing tool and check it.",
        nodes=(node,),
    )
    precheck = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": True,
            "arguments": {"command": "which sample-tool"},
            "content_preview": "exit_code: 1\n",
        },
    )
    install_and_recheck = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {"command": "npm install -g sample-tool && sample-tool --version"},
            "content_preview": "exit_code: 0\n\nstdout:\n1.0.0\n",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Installed sample-tool and checked its version.",
        activity=[precheck, install_and_recheck],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"


def test_failed_command_v_environment_probe_can_be_informational_without_output() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="check whether a tool is missing and write a report",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="no_failed_tools"),
            EvidenceRequirement(kind="environment_checked"),
            EvidenceRequirement(kind="files_changed"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create ENV_REPORT.md for a missing tool.",
        nodes=(node,),
    )
    missing_probe = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": True,
            "arguments": {"command": "command -v harness-missing-tool-20260530"},
            "content_preview": "exit_code: 1\n\n(no output)",
            "metadata": {"exit_code": 1},
        },
    )
    write = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "ENV_REPORT.md"},
            "content_preview": "wrote ENV_REPORT.md",
        },
    )
    verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "grep STATUS=missing ENV_REPORT.md"},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result=(
            "The command returned a non-zero exit code, confirming the tool is missing. "
            "Created ENV_REPORT.md and verify_work passed."
        ),
        activity=[missing_probe, write, verify],
        run=run,
    )

    assert ok is True
    assert [item.status for item in results] == ["passed", "passed", "passed", "passed"]


def test_successful_lookup_with_absent_output_becomes_environment_observation() -> None:
    event = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {"command": "which harness-missing-tool || echo NOT_FOUND"},
            "content_preview": "exit_code: 0\n\nstdout:\nNOT_FOUND\n",
            "metadata": {"exit_code": 0},
        },
    )

    summary = summarize_activity([event])

    assert summary["environment_observations"]
    assert "missing tool" in summary["environment_observations"][0]
    assert "harness-missing-tool" in summary["environment_observations"][0]


def test_failed_missing_command_can_be_environment_evidence_before_mutation() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="use a missing tool or fall back",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="no_failed_tools"),
            EvidenceRequirement(kind="environment_checked"),
            EvidenceRequirement(kind="files_changed"),
        ),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create FALLBACK.md after discovering whether the requested tool exists.",
        nodes=(node,),
    )
    missing_attempt = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": True,
            "arguments": {"command": "harness-missing-tool-20260530 --inspect"},
            "content_preview": "exit_code: 127\n\nstderr:\nzsh: command not found: harness-missing-tool-20260530",
            "metadata": {"exit_code": 127},
        },
    )
    write = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "FALLBACK.md"},
            "content_preview": "wrote FALLBACK.md",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result=(
            "The requested command is missing in this environment, so I used the "
            "available fallback and wrote FALLBACK.md."
        ),
        activity=[missing_attempt, write],
        run=run,
    )

    assert ok is True
    assert [item.status for item in results] == ["passed", "passed", "passed"]
    summary = summarize_activity([missing_attempt, write])
    assert summary["environment_observations"]
    assert "missing tool" in summary["environment_observations"][0]


def test_unsupported_shell_option_recovered_by_later_environment_path() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="inspect a file",
        expected_evidence=(EvidenceRequirement(kind="no_failed_tools"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="Inspect ANALYSIS.md.", nodes=(node,))
    unsupported = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": True,
            "arguments": {"command": "cat -A ANALYSIS.md"},
            "content_preview": (
                "exit_code: 1\n\nstderr:\ncat: illegal option -- A\n"
                "usage: cat [-belnstuv] [file ...]"
            ),
            "metadata": {"exit_code": 1},
        },
    )
    recovered = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {"command": "cat -e ANALYSIS.md"},
            "content_preview": "exit_code: 0\n\nstdout:\n# Report$",
            "metadata": {"exit_code": 0},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="cat -A is unsupported here, so I used cat -e and inspected the report.",
        activity=[unsupported, recovered],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"
    summary = summarize_activity([unsupported, recovered])
    assert "unsupported tool option" in summary["environment_observations"][0]


def test_failed_shell_probe_recovered_by_later_verify_work() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="verify report",
        expected_evidence=(EvidenceRequirement(kind="no_failed_tools"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="Verify ANALYSIS.md.", nodes=(node,))
    failed_probe = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": True,
            "arguments": {"command": "grep 'Total: 113' ANALYSIS.md"},
            "content_preview": "exit_code: 1\n\n(no output)",
            "metadata": {"exit_code": 1},
        },
    )
    later_verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "grep -q 'Total' ANALYSIS.md"},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Recovered the failed probe with verify_work.",
        activity=[failed_probe, later_verify],
        run=run,
    )

    assert ok is True
    assert results[0].message == "no failed tool calls"


def test_derived_artifact_verify_work_must_use_source_evidence() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create report",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="environment_checked"),
            EvidenceRequirement(kind="files_changed"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Produce ANALYSIS.md from invoices.csv.",
        nodes=(node,),
    )
    read_source = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "read_file",
            "is_error": False,
            "arguments": {"path": "invoices.csv"},
            "content_preview": "customer,region,amount,status\nBeacon Co,US,89.25,paid",
        },
    )
    write_report = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "ANALYSIS.md"},
            "content_preview": "wrote ANALYSIS.md",
        },
    )
    self_check = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "grep -q 'US: 300.00' ANALYSIS.md"},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Created ANALYSIS.md and verify_work passed.",
        activity=[read_source, write_report, self_check],
        run=run,
    )

    assert ok is False
    assert results[2].status == "failed"
    assert "did not compare generated artifacts with source inputs" in results[2].message

    source_only_check = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "awk -F, '{sum += $3} END {print sum}' invoices.csv"},
            "content_preview": "PASSED\n89.25",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Recomputed invoices.csv values.",
        activity=[read_source, write_report, source_only_check],
        run=run,
    )

    assert ok is False
    assert results[2].status == "failed"
    assert "did not compare generated artifacts with source inputs" in results[2].message


def test_derived_artifact_verify_work_rejects_presence_only_source_check() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create report",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="environment_checked"),
            EvidenceRequirement(kind="files_changed"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Produce ANALYSIS.md from invoices.csv.",
        nodes=(node,),
    )
    read_source = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "read_file",
            "is_error": False,
            "arguments": {"path": "invoices.csv"},
            "content_preview": "customer,region,amount,status\nBeacon Co,US,89.25,paid",
        },
    )
    write_report = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "ANALYSIS.md"},
            "content_preview": "wrote ANALYSIS.md",
        },
    )
    presence_check = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": "test -s ANALYSIS.md && test -s invoices.csv",
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Created ANALYSIS.md and verify_work passed.",
        activity=[read_source, write_report, presence_check],
        run=run,
    )

    assert ok is False
    assert results[2].status == "failed"
    assert "did not compare generated artifacts with source inputs" in results[2].message


def test_derived_artifact_verify_work_rejects_variable_token_presence_check() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create report",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="environment_checked"),
            EvidenceRequirement(kind="files_changed"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Produce ANALYSIS.md from records.json.",
        nodes=(node,),
    )
    read_source = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "read_file",
            "is_error": False,
            "arguments": {"path": "records.json"},
            "content_preview": '{"items":[{"kind":"alpha","count":2},{"kind":"beta","count":3}]}',
        },
    )
    write_report = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "ANALYSIS.md"},
            "content_preview": "wrote ANALYSIS.md",
        },
    )
    weak_check = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    "KINDS=$(jq -r '.items[].kind' records.json | sort -u)\n"
                    'for kind in $KINDS; do grep -q "$kind" ANALYSIS.md; done'
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Created ANALYSIS.md and verify_work passed.",
        activity=[read_source, write_report, weak_check],
        run=run,
    )

    assert ok is False
    assert results[2].status == "failed"
    assert "without a later passing verify_work" in results[2].message


def test_verify_node_rejects_membership_only_source_artifact_check() -> None:
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create report",
        status="completed",
        metadata={
            "activity_summary": {
                "read_paths": ["records.json"],
                "changed_paths": ["ANALYSIS.md"],
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
    run = WorkflowRun(id="wf", title="WF", goal="Verify ANALYSIS.md.", nodes=(work, verify))
    weak_check = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    "python3 -c \"import json; data=json.load(open('records.json')); "
                    "kinds=sorted({i['kind'] for i in data['items']}); "
                    "report=open('ANALYSIS.md').read(); "
                    'assert all(k in report for k in kinds)"'
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=verify,
        result="Verified report from source.",
        activity=[weak_check],
        run=run,
    )

    assert ok is False
    assert results[0].status == "failed"
    assert "did not compare dependency artifacts with source inputs" in results[0].message


def test_verify_node_rejects_every_includes_membership_only_source_artifact_check() -> None:
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create report",
        status="completed",
        metadata={
            "activity_summary": {
                "read_paths": ["records.json"],
                "changed_paths": ["ANALYSIS.md"],
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
    run = WorkflowRun(id="wf", title="WF", goal="Verify ANALYSIS.md.", nodes=(work, verify))
    weak_check = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    "node -e \"const fs = require('fs'); "
                    "const data = JSON.parse(fs.readFileSync('records.json', 'utf8')); "
                    "const kinds = [...new Set(data.items.map((item) => item.kind))]; "
                    "const report = fs.readFileSync('ANALYSIS.md', 'utf8'); "
                    'if (!kinds.every((kind) => report.includes(kind))) process.exit(1);"'
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=verify,
        result="Verified report from source.",
        activity=[weak_check],
        run=run,
    )

    assert ok is False
    assert results[0].status == "failed"
    assert "did not compare dependency artifacts with source inputs" in results[0].message


def test_verify_node_accepts_computed_key_value_source_artifact_check() -> None:
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create report",
        status="completed",
        metadata={
            "activity_summary": {
                "read_paths": ["records.json"],
                "changed_paths": ["ANALYSIS.md"],
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
    run = WorkflowRun(id="wf", title="WF", goal="Verify ANALYSIS.md.", nodes=(work, verify))
    source_compare = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    "python3 -c \"import json; data=json.load(open('records.json')); "
                    "counts={}; "
                    "[counts.update({i['kind']: counts.get(i['kind'], 0) + i['count']}) "
                    "for i in data['items']]; "
                    "report=open('ANALYSIS.md').read(); "
                    "for k, v in counts.items():\\n "
                    " expected = f'{k}: {v}'\\n "
                    ' assert expected in report"'
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=verify,
        result="Verified report from source.",
        activity=[source_compare],
        run=run,
    )

    assert ok is True
    assert results[0].message == "verify_work passed"


def test_source_artifact_presence_gate_has_no_language_specific_policy() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src" / "harness" / "core" / "dynamic_workflows.py"
    ).read_text(encoding="utf-8")
    section = source[
        source.index("def _grep_segment_uses_weak_variable_presence") : source.index(
            "def _verify_work_command_changes_state"
        )
    ]
    language_specific_terms = frozenset(
        {
            "python",
            "python3",
            "pytest",
            "pip",
            "pip3",
            "node",
            "npm",
            "npx",
            "pnpm",
            "yarn",
            "cargo",
            "go",
            "rust",
            "javascript",
            "js",
            "typescript",
            "ts",
        }
    )
    words = set(re.findall(r"[a-z0-9_+.-]+", section.lower()))
    assert words.isdisjoint(language_specific_terms)


def test_generic_verification_gates_have_no_language_specific_policy() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src" / "harness" / "core" / "dynamic_workflows.py"
    ).read_text(encoding="utf-8")
    sections = [
        source[
            source.index("_EXACT_FILE_CONTENT_NAMED_REQUEST_RE") : source.index("_NUMERIC_CLAIM_RE")
        ],
        source[
            source.index("def _command_directly_runs_path") : source.index(
                "def _stdout_no_trailing_newline_requested"
            )
        ],
        source[
            source.index("def _command_reads_path") : source.index("def _byte_size_check_requested")
        ],
    ]
    language_specific_terms = frozenset(
        {
            "python",
            "python3",
            "pytest",
            "pip",
            "pip3",
            "node",
            "npm",
            "npx",
            "pnpm",
            "yarn",
            "cargo",
            "go",
            "rust",
            "javascript",
            "js",
            "typescript",
            "ts",
        }
    )
    words = {word for section in sections for word in re.findall(r"[a-z0-9_+.-]+", section.lower())}
    assert words.isdisjoint(language_specific_terms)


def test_derived_artifact_rejects_shell_generated_output_only_check() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create report",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="environment_checked"),
            EvidenceRequirement(kind="files_changed"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Produce ANALYSIS.md from invoices.csv.",
        nodes=(node,),
    )
    activity = [
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "read_file",
                "is_error": False,
                "arguments": {"path": "invoices.csv"},
                "content_preview": "customer,region,amount,status\nBeacon Co,US,89.25,paid",
            },
        ),
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "write_file",
                "is_error": False,
                "arguments": {"path": "analyze.py"},
                "content_preview": "wrote analyze.py",
            },
        ),
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "shell",
                "is_error": False,
                "arguments": {"command": "python3 analyze.py"},
                "content_preview": "exit_code: 0",
            },
        ),
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "read_file",
                "is_error": False,
                "arguments": {"path": "ANALYSIS.md"},
                "content_preview": "US: 89.25",
            },
        ),
        ActivityEvent(
            session_id="s",
            kind="tool_call.completed",
            data={
                "name": "verify_work",
                "is_error": False,
                "arguments": {"command": "grep -q 'US: 89.25' ANALYSIS.md"},
                "content_preview": "PASSED",
            },
        ),
    ]

    ok, results = evaluate_node_evidence(
        node=node,
        result="Generated ANALYSIS.md and checked it.",
        activity=activity,
        run=run,
    )

    assert ok is False
    assert results[2].status == "failed"
    assert "did not compare generated artifacts with source inputs" in results[2].message


def test_retry_history_does_not_reuse_failed_verify_work_evidence() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="retry verification",
        allow_mutation=True,
        expected_evidence=(EvidenceRequirement(kind="verify_work_if_state_changed"),),
        metadata={
            "failure_history": [
                {
                    "activity_summary": {
                        "state_changed": True,
                        "verify_work_passed": True,
                    },
                    "evidence_results": [
                        {
                            "requirement": {"kind": "verify_work_if_state_changed"},
                            "status": "failed",
                            "message": "weak verifier",
                        }
                    ],
                }
            ]
        },
    )
    run = WorkflowRun(id="wf", title="WF", goal="verify report", nodes=(node,))

    ok, results = evaluate_node_evidence(
        node=node,
        result="No new verification.",
        activity=[],
        run=run,
    )

    assert ok is False
    assert results[0].message == "previous state change lacks a later passing verify_work call"


def test_derived_artifact_verify_work_accepts_source_based_check() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="create report",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="no_failed_tools"),
            EvidenceRequirement(kind="environment_checked"),
            EvidenceRequirement(kind="files_changed"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Produce ANALYSIS.md from invoices.csv.",
        nodes=(node,),
    )
    read_source = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "read_file",
            "is_error": False,
            "arguments": {"path": "invoices.csv"},
            "content_preview": "customer,region,amount,status\nBeacon Co,US,89.25,paid",
        },
    )
    write_report = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "ANALYSIS.md"},
            "content_preview": "wrote ANALYSIS.md",
        },
    )
    failed_self_check = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": True,
            "arguments": {"command": "grep -q 'US: 300.00' ANALYSIS.md"},
            "content_preview": "FAILED (exit 1)",
        },
    )
    source_check = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": (
                    'awk -F, \'$4=="paid"{sum+=$3} END{printf "US: %.2f", sum}\' '
                    "invoices.csv | grep -q 'US: 89.25' && grep -q 'US: 89.25' ANALYSIS.md"
                )
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Created ANALYSIS.md and verify_work passed.",
        activity=[read_source, write_report, failed_self_check, source_check],
        run=run,
    )

    assert ok is True
    assert [item.status for item in results] == ["passed", "passed", "passed", "passed"]


def test_retry_can_verify_state_changed_by_previous_attempt() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="verify existing artifact",
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="environment_checked"),
            EvidenceRequirement(kind="files_changed"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
        ),
        metadata={
            "failure_history": [
                {
                    "attempt": 1,
                    "error": "state changed without a later passing verify_work call",
                    "activity_summary": {
                        "state_changed": True,
                        "verify_work_passed": False,
                    },
                    "evidence_results": [
                        {
                            "requirement": {"kind": "environment_checked"},
                            "status": "passed",
                            "message": "environment checked around mutation",
                        }
                    ],
                }
            ]
        },
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create HARNESS_DEEPSWE_CHECK.md and verify it.",
        nodes=(node,),
    )
    verify = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {
                "command": "test -f HARNESS_DEEPSWE_CHECK.md && grep 113 HARNESS_DEEPSWE_CHECK.md"
            },
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="Verified HARNESS_DEEPSWE_CHECK.md contains 113.",
        activity=[verify],
        run=run,
    )

    assert ok is True
    assert [item.status for item in results] == ["passed", "passed", "passed"]


def test_retry_can_finalize_with_previous_verified_grounding() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt=(
            "Create GREETING.txt from profile.json. It should say: Hello Ada, Python count is 4."
        ),
        allow_mutation=True,
        expected_evidence=(
            EvidenceRequirement(kind="result_nonempty"),
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="environment_checked"),
            EvidenceRequirement(kind="files_changed"),
            EvidenceRequirement(kind="verify_work_if_state_changed"),
            EvidenceRequirement(kind="objective_status"),
        ),
        metadata={
            "failure_history": [
                {
                    "attempt": 1,
                    "error": "objective status is missing",
                    "result": (
                        "Created GREETING.txt with exact content "
                        "Hello Ada, Python count is 4. and verify_work passed."
                    ),
                    "activity_summary": {
                        "state_changed": True,
                        "verify_work_passed": True,
                    },
                    "evidence_results": [
                        {
                            "requirement": {"kind": "claim_grounded"},
                            "status": "passed",
                        },
                        {
                            "requirement": {"kind": "environment_checked"},
                            "status": "passed",
                        },
                        {
                            "requirement": {"kind": "verify_work_if_state_changed"},
                            "status": "passed",
                        },
                    ],
                }
            ]
        },
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal=(
            "Create GREETING.txt from profile.json. It should say: Hello Ada, Python count is 4."
        ),
        nodes=(node,),
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result=(
            "Previous tool evidence created GREETING.txt with exact content "
            "Hello Ada, Python count is 4. and verify_work passed.\n\n"
            '{"status":"pass"}'
        ),
        activity=[],
        run=run,
    )

    assert ok is True
    assert [item.status for item in results] == [
        "passed",
        "passed",
        "passed",
        "passed",
        "passed",
        "passed",
    ]


def test_claim_grounded_rejects_unsupported_absolute_path() -> None:
    node = WorkflowNode(
        id="review",
        title="Review",
        kind="review",
        prompt="review",
        depends_on=("verify",),
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    verify = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        status="completed",
        result="The current directory is /private/tmp/harness-realcheck.",
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(verify, node))

    ok, results = evaluate_node_evidence(
        node=node,
        result="The current directory is /private/tmp/harness-abcdef-1234.",
        activity=[],
        run=run,
    )

    assert ok is False
    assert "unsupported path claim" in results[0].message


def test_claim_grounded_ignores_slash_separated_concepts() -> None:
    node = WorkflowNode(
        id="research",
        title="Research",
        kind="research",
        prompt="research",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(node,))
    read = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "read_file",
            "is_error": False,
            "arguments": {"path": "httpx/_models.py"},
            "content_preview": "Response has json, iter_bytes, and aiter_bytes.",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result=(
            "The JSON/encoding path and iter_/aiter_ symmetry are understood; "
            "UTF-8/16/32 handling still needs implementation."
        ),
        activity=[read],
        run=run,
    )

    assert ok is True
    assert [item.status for item in results] == ["passed"]


def test_claim_grounded_rejects_empty_directory_claim_when_list_dir_has_entries() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="work",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(node,))
    listed = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "list_dir",
            "is_error": False,
            "arguments": {},
            "content_preview": ".harness/",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="The current working directory appears to be empty.",
        activity=[listed],
        run=run,
    )

    assert ok is False
    assert "directory is empty" in results[0].message


def test_claim_grounded_accepts_empty_directory_claim_from_empty_list_dir_metadata() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="work",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(node,))
    listed = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "list_dir",
            "is_error": False,
            "arguments": {"path": "."},
            "content_preview": "(empty)",
            "metadata": {"path": ".", "entries": 0, "ignored_entries": 1},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="The current workspace is empty.",
        activity=[listed],
        run=run,
    )

    assert ok is True
    assert results[0].message == "result claims are grounded in evidence"


def test_claim_grounded_rejects_incomplete_workflow_boilerplate() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="work",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(node,))
    listed = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {"command": "ls -F"},
            "content_preview": "",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="The current directory is empty. I am ready for your request.",
        activity=[listed],
        run=run,
    )

    assert ok is False
    assert "instead of completing" in results[0].message


def test_claim_grounded_requires_path_for_exact_cwd_path_goal() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Report the exact current working directory path.",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Report the exact current working directory path.",
        nodes=(node,),
    )
    listed = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {"command": "ls -F"},
            "content_preview": "",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="The current directory is empty.",
        activity=[listed],
        run=run,
    )

    assert ok is False
    assert "requested absolute path" in results[0].message


def test_claim_grounded_accepts_exact_cwd_path_from_pwd_evidence() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Report the exact current working directory path.",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Report the exact current working directory path.",
        nodes=(node,),
    )
    pwd = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {"command": "pwd"},
            "content_preview": "/private/tmp/harness-realcheck-cwd3",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="The current working directory path is `/private/tmp/harness-realcheck-cwd3`.",
        activity=[pwd],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"


def test_claim_grounded_rejects_missing_exact_first_line_in_merge() -> None:
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="work",
        status="completed",
        result='README.md exists and the first line is: "Harness fixture first line".',
    )
    merge = WorkflowNode(
        id="merge",
        title="Merge",
        kind="merge",
        prompt="Report whether README.md exists and quote exactly the first line.",
        depends_on=("work",),
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Report whether README.md exists and quote exactly the first line.",
        nodes=(work, merge),
    )

    ok, results = evaluate_node_evidence(
        node=merge,
        result="README.md exists and the first line was retrieved.",
        activity=[],
        run=run,
    )

    assert ok is False
    assert "requested exact first line" in results[0].message


def test_claim_grounded_rejects_unsupported_numeric_merge_claim() -> None:
    verify = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        status="completed",
        result=("Verified task_count 113 and languages: typescript, python, go, rust, javascript."),
    )
    merge = WorkflowNode(
        id="merge",
        title="Merge",
        kind="merge",
        prompt="merge",
        depends_on=("verify",),
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Report verified values.",
        nodes=(verify, merge),
    )

    ok, results = evaluate_node_evidence(
        node=merge,
        result="Final values: task_count 113, Python 102, Java 11, 100% accurate.",
        activity=[],
        run=run,
    )

    assert ok is False
    assert results[0].status == "failed"
    assert "unsupported numeric claim" in results[0].message


def test_claim_grounded_accepts_exact_first_line_in_merge() -> None:
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="work",
        status="completed",
        result='README.md exists and the first line is: "Harness fixture first line".',
    )
    merge = WorkflowNode(
        id="merge",
        title="Merge",
        kind="merge",
        prompt="Report whether README.md exists and quote exactly the first line.",
        depends_on=("work",),
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Report whether README.md exists and quote exactly the first line.",
        nodes=(work, merge),
    )

    ok, results = evaluate_node_evidence(
        node=merge,
        result='README.md exists. First line: "Harness fixture first line".',
        activity=[],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"


def test_claim_grounded_rejects_missing_first_line_from_read_file_evidence() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Report whether README.md exists and quote exactly the first line.",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Report whether README.md exists and quote exactly the first line.",
        nodes=(node,),
    )
    read_file = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "read_file",
            "is_error": False,
            "arguments": {"path": "README.md"},
            "content_preview": "Harness fixture first line\nSecond line",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="README.md exists and contains two lines.",
        activity=[read_file],
        run=run,
    )

    assert ok is False
    assert "Harness fixture first line" in results[0].message


def test_claim_grounded_ignores_verify_work_status_for_first_line_candidate() -> None:
    node = WorkflowNode(
        id="merge",
        title="Merge",
        kind="merge",
        prompt="Report whether README.md exists and quote exactly the first line.",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Report whether README.md exists and quote exactly the first line.",
        nodes=(node,),
    )
    verified = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "test -f README.md && head -n 1 README.md"},
            "content_preview": "PASSED\n\nHarness fixture first line",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='README.md exists. First line: "Harness fixture first line".',
        activity=[verified],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"


def test_claim_grounded_rejects_wrong_exact_file_content_write() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Create hello.txt containing exactly verified hello and verify it.",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create hello.txt containing exactly verified hello and verify it.",
        nodes=(node,),
    )
    wrote = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "hello.txt", "content": "hello"},
            "content_preview": "wrote 5 bytes to hello.txt",
        },
    )
    read = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "read_file",
            "is_error": False,
            "arguments": {"path": "hello.txt"},
            "content_preview": "hello",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='hello.txt was created with exactly "hello".',
        activity=[wrote, read],
        run=run,
    )

    assert ok is False
    assert "requested exact content" in results[0].message
    assert "verified hello" in results[0].message


def test_claim_grounded_rejects_exact_file_content_without_evidence() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Create hello.txt containing exactly verified hello and verify it.",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create hello.txt containing exactly verified hello and verify it.",
        nodes=(node,),
    )
    read = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "read_file",
            "is_error": False,
            "arguments": {"path": "hello.txt"},
            "content_preview": "hello",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='hello.txt was created with exactly "verified hello".',
        activity=[read],
        run=run,
    )

    assert ok is False
    assert "not grounded in evidence" in results[0].message


def test_claim_grounded_accepts_exact_file_content_write() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Create hello.txt containing exactly verified hello and verify it.",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create hello.txt containing exactly verified hello and verify it.",
        nodes=(node,),
    )
    wrote = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "hello.txt", "content": "verified hello"},
            "content_preview": "wrote 14 bytes to hello.txt",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='hello.txt was created with exactly "verified hello".',
        activity=[wrote],
        run=run,
    )

    assert ok is True
    assert results[0].status == "passed"


def test_claim_grounded_rejects_exact_file_content_with_trailing_newline_write() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Create hello.txt containing exactly 'verified hello' and verify it.",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create hello.txt containing exactly 'verified hello' and verify it.",
        nodes=(node,),
    )
    wrote = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "hello.txt", "content": "verified hello\n"},
            "content_preview": "wrote 15 bytes to hello.txt",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='hello.txt was created with exactly "verified hello".',
        activity=[wrote],
        run=run,
    )

    assert ok is False
    assert "not grounded in evidence" in results[0].message


def test_claim_grounded_rejects_later_edit_that_changes_exact_file_content() -> None:
    node = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="Create hello.txt containing exactly 'verified hello' and verify it.",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create hello.txt containing exactly 'verified hello' and verify it.",
        nodes=(node,),
    )
    wrote = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "write_file",
            "is_error": False,
            "arguments": {"path": "hello.txt", "content": "verified hello"},
            "content_preview": "wrote 14 bytes to hello.txt",
        },
    )
    edited = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "edit_file",
            "is_error": False,
            "arguments": {
                "path": "hello.txt",
                "old": "verified hello",
                "new": "verified hello\n",
            },
            "content_preview": "replaced 1 occurrence in hello.txt",
        },
    )
    verified = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "cat hello.txt"},
            "content_preview": "PASSED\n\nverified hello",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result='hello.txt was created with exactly "verified hello".',
        activity=[wrote, edited, verified],
        run=run,
    )

    assert ok is False
    assert "not grounded in evidence" in results[0].message


def test_claim_grounded_rejects_checksum_match_claim_with_different_values() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="verify checksum", nodes=(node,))
    actual = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {"command": "md5sum hello.txt"},
            "content_preview": "c847181f978163ab7ffef72f604ff6f5  hello.txt",
        },
    )
    expected = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "echo -n 'verified hello' | md5sum"},
            "content_preview": "PASSED\n\n270739512e8002a6105dffbd4b9a6914  -",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="The MD5 checksum comparison yielded a match.",
        activity=[actual, expected],
        run=run,
    )

    assert ok is False
    assert "differing checksums" in results[0].message


def test_claim_grounded_rejects_checksum_match_claim_with_single_value() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="verify checksum", nodes=(node,))
    checksum = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "echo 'verified hello' | md5"},
            "content_preview": "PASSED\n\n215f7cb2071a835fcfbeaaa27cf23146",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="An MD5 checksum comparison confirms that the content matches.",
        activity=[checksum],
        run=run,
    )

    assert ok is False
    assert "without comparable checksum evidence" in results[0].message


def test_review_pass_rejects_missing_exact_file_content_evidence() -> None:
    work = WorkflowNode(
        id="work",
        title="Work",
        kind="work",
        prompt="work",
        status="completed",
        result='hello.txt was created with exactly "hello".',
    )
    review = WorkflowNode(
        id="review",
        title="Review",
        kind="review",
        prompt="Create hello.txt containing exactly verified hello and verify it.",
        depends_on=("work",),
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Create hello.txt containing exactly verified hello and verify it.",
        nodes=(work, review),
    )

    ok, results = evaluate_node_evidence(
        node=review,
        result="WORKFLOW_DECISION: pass",
        activity=[],
        run=run,
    )

    assert ok is False
    assert "verified hello" in results[0].message


def test_default_workflow_requires_grounded_work_and_review_claims() -> None:
    run = create_default_workflow(workflow_id="wf", title="WF", goal="goal")
    by_id = {node.id: node for node in run.nodes}

    assert "planning_only" in {item.kind for item in by_id["plan"].expected_evidence}
    assert "no_state_change" in {item.kind for item in by_id["plan"].expected_evidence}
    assert "no_failed_tools" in {item.kind for item in by_id["plan"].expected_evidence}
    assert "claim_grounded" in {item.kind for item in by_id["work"].expected_evidence}
    assert "environment_checked" in {item.kind for item in by_id["work"].expected_evidence}
    assert "files_changed" in {item.kind for item in by_id["work"].expected_evidence}
    assert "claim_grounded" in {item.kind for item in by_id["verify"].expected_evidence}
    assert "claim_grounded" in {item.kind for item in by_id["review"].expected_evidence}
    assert "claim_grounded" in {item.kind for item in by_id["merge"].expected_evidence}
    assert "no_state_change" in {item.kind for item in by_id["verify"].expected_evidence}
    assert "no_state_change" in {item.kind for item in by_id["review"].expected_evidence}
    assert "no_state_change" in {item.kind for item in by_id["refute"].expected_evidence}
    assert "no_state_change" in {item.kind for item in by_id["merge"].expected_evidence}
    assert "no_failed_tools" in {item.kind for item in by_id["review"].expected_evidence}
    assert "no_failed_tools" in {item.kind for item in by_id["refute"].expected_evidence}
    assert "no_failed_tools" in {item.kind for item in by_id["merge"].expected_evidence}
    assert "Independently verify the result" in by_id["verify"].prompt
    assert "WORKFLOW_DECISION: pass" in by_id["review"].prompt
    assert "WORKFLOW_DECISION: retry" in by_id["refute"].prompt


def test_workflow_decision_accepts_grounded_structured_json() -> None:
    node = WorkflowNode(
        id="review",
        title="Review",
        kind="review",
        prompt="review",
        depends_on=("verify",),
        expected_evidence=(
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="review_decision"),
        ),
    )
    verify = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        status="completed",
        result="The current directory is /private/tmp/harness-realcheck.",
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(verify, node))

    ok, results = evaluate_node_evidence(
        node=node,
        result=(
            '{"review_summary":"The current directory is /private/tmp/harness-realcheck.",'
            '"is_successful":true}'
        ),
        activity=[],
        run=run,
    )

    assert ok is True
    assert [result.status for result in results] == ["passed", "passed"]


def test_workflow_decision_accepts_markdown_wrapped_line() -> None:
    assert workflow_decision("**WORKFLOW_DECISION: retry**") == "retry"


def test_refute_claims_can_use_upstream_verifier_evidence() -> None:
    verify = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        status="completed",
        result="The current working directory path is `/private/tmp/harness-realcheck`.",
    )
    review = WorkflowNode(
        id="review",
        title="Review",
        kind="review",
        prompt="review",
        status="completed",
        result="WORKFLOW_DECISION: pass",
    )
    refute = WorkflowNode(
        id="refute",
        title="Refute",
        kind="refute",
        prompt="refute",
        depends_on=("review",),
        expected_evidence=(
            EvidenceRequirement(kind="claim_grounded"),
            EvidenceRequirement(kind="review_decision"),
        ),
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(verify, review, refute))

    ok, results = evaluate_node_evidence(
        node=refute,
        result=(
            "The verifier evidence reports `/private/tmp/harness-realcheck`.\n"
            "WORKFLOW_DECISION: pass"
        ),
        activity=[],
        run=run,
    )

    assert ok is True
    assert [result.status for result in results] == ["passed", "passed"]


def test_refute_rejects_relative_cwd_path_claim_for_exact_path_goal() -> None:
    verify = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        status="completed",
        result="The current working directory path is `/private/tmp/harness-realcheck`.",
    )
    review = WorkflowNode(
        id="review",
        title="Review",
        kind="review",
        prompt="review",
        status="completed",
        result="WORKFLOW_DECISION: pass",
    )
    refute = WorkflowNode(
        id="refute",
        title="Refute",
        kind="refute",
        prompt="refute",
        depends_on=("review",),
        expected_evidence=(EvidenceRequirement(kind="claim_grounded"),),
    )
    run = WorkflowRun(
        id="wf",
        title="WF",
        goal="Report the exact current working directory path.",
        nodes=(verify, review, refute),
    )

    ok, results = evaluate_node_evidence(
        node=refute,
        result="The current working directory path is `.harness/`.\nWORKFLOW_DECISION: pass",
        activity=[],
        run=run,
    )

    assert ok is False
    assert "requested absolute path" in results[0].message


def test_verify_work_evidence_accepts_zero_failed_summary() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(node,))
    passed = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "pytest"},
            "content_preview": "2 passed, 0 failed in 0.10s",
        },
    )
    failed = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "pytest"},
            "content_preview": "1 failed, 2 passed in 0.10s",
        },
    )

    ok, results = evaluate_node_evidence(node=node, result="done", activity=[passed], run=run)
    assert ok is True
    assert results[0].status == "passed"

    ok, results = evaluate_node_evidence(node=node, result="done", activity=[failed], run=run)
    assert ok is False
    assert results[0].message == "verify_work did not pass"

    failed_metadata = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "pytest"},
            "content_preview": "PASSED",
            "metadata": {"exit_code": 1},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[failed_metadata],
        run=run,
    )

    assert ok is False
    assert results[0].message == "verify_work did not pass"


def test_verify_work_evidence_rejects_statically_failing_success_claim() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(node,))
    impossible_pass = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "pytest -q && false"},
            "content_preview": "PASSED",
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[impossible_pass],
        run=run,
    )

    assert ok is False
    assert results[0].message == "verify_work did not pass"


def test_verify_work_evidence_rejects_masked_failure_success_claim() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(node,))
    masked_failure = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "test -f missing.txt || echo 'not actually verified'"},
            "content_preview": "PASSED\n\nnot actually verified",
            "metadata": {"exit_code": 0},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[masked_failure],
        run=run,
    )

    assert ok is False
    assert results[0].message == "verify_work did not pass"


def test_verify_work_evidence_rejects_traceback_success_claim() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(node,))
    traceback_pass = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "pytest tests"},
            "content_preview": (
                "PASSED\n\nTraceback (most recent call last):\nAssertionError: wrong result"
            ),
            "metadata": {
                "stdout": ("Traceback (most recent call last):\nAssertionError: wrong result\n"),
                "exit_code": 0,
            },
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[traceback_pass],
        run=run,
    )

    assert ok is False
    assert results[0].message == "verify_work did not pass"


def test_verify_work_evidence_rejects_error_success_claim() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(node,))
    error_pass = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "pytest tests"},
            "content_preview": "PASSED\n\nERROR: expected harness-ok got wrong",
            "metadata": {
                "stdout": "ERROR: expected harness-ok got wrong\n",
                "exit_code": 0,
            },
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[error_pass],
        run=run,
    )

    assert ok is False
    assert results[0].message == "verify_work did not pass"


def test_verify_work_evidence_rejects_only_skipped_success_claim() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(node,))
    skipped_pass = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "pytest tests"},
            "content_preview": "PASSED\n\n1 skipped, 0 passed, 0 failed",
            "metadata": {
                "stdout": "1 skipped, 0 passed, 0 failed\n",
                "exit_code": 0,
            },
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[skipped_pass],
        run=run,
    )

    assert ok is False
    assert results[0].message == "verify_work did not pass"


def test_verify_work_evidence_rejects_all_skipped_success_claim() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(node,))
    skipped_pass = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "pytest tests"},
            "content_preview": "PASSED\n\n3 skipped in 0.02s",
            "metadata": {
                "stdout": "3 skipped in 0.02s\n",
                "exit_code": 0,
            },
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[skipped_pass],
        run=run,
    )

    assert ok is False
    assert results[0].message == "verify_work did not pass"


def test_verify_work_evidence_rejects_all_deselected_success_claim() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(node,))
    deselected_pass = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "pytest tests"},
            "content_preview": "PASSED\n\n3 deselected in 0.02s",
            "metadata": {
                "stdout": "3 deselected in 0.02s\n",
                "exit_code": 0,
            },
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[deselected_pass],
        run=run,
    )

    assert ok is False
    assert results[0].message == "verify_work did not pass"


def test_verify_work_evidence_rejects_zero_passing_success_claim() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(node,))
    zero_passing = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "npm test"},
            "content_preview": "PASSED\n\n0 passing (1ms)",
            "metadata": {
                "stdout": "0 passing (1ms)\n",
                "exit_code": 0,
            },
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[zero_passing],
        run=run,
    )

    assert ok is False
    assert results[0].message == "verify_work did not pass"


def test_verify_work_evidence_rejects_empty_for_loop_success_claim() -> None:
    node = WorkflowNode(
        id="verify",
        title="Verify",
        kind="verify",
        prompt="verify",
        expected_evidence=(EvidenceRequirement(kind="verify_work_passed"),),
    )
    run = WorkflowRun(id="wf", title="WF", goal="goal", nodes=(node,))
    skipped_loop = ActivityEvent(
        session_id="s",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": 'for item in ; do test -f "$item"; done'},
            "content_preview": "PASSED",
            "metadata": {"exit_code": 0},
        },
    )

    ok, results = evaluate_node_evidence(
        node=node,
        result="done",
        activity=[skipped_loop],
        run=run,
    )

    assert ok is False
    assert results[0].message == "verify_work did not pass"
