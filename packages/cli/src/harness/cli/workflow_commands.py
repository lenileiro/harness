from __future__ import annotations

import asyncio
import collections
import glob
import re
import shlex
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import typer
from rich.console import Console
from rich.table import Table

from harness.cli.common import _build_adapter, _build_tools, _load_cli_config, _resolve_chain
from harness.cli.run_commands import run_once as _run_once_impl
from harness.cli.runtime_helpers import (
    build_critic as _build_critic,
)
from harness.cli.runtime_helpers import build_storage
from harness.cli.runtime_helpers import (
    build_verifier as _build_verifier,
)
from harness.cli.runtime_helpers import (
    print_defense_ledger as _print_defense_ledger,
)
from harness.cli.runtime_helpers import (
    resolve_runtime_strategy as _resolve_runtime_strategy,
)
from harness.core import Done, Message, TextDelta, Tool
from harness.core.activity import ActivityEvent
from harness.core.dynamic_workflows import (
    WorkflowNode,
    WorkflowRun,
    WorkflowStore,
    _exact_file_content_requests,
    _exact_stdout_requests,
    _verify_work_after_last_change,
    _workflow_tool_event_changes_state,
    create_default_workflow,
    create_workflow_from_plan_spec,
    default_workflow_root,
    dependent_node_ids,
    evaluate_node_evidence,
    extract_json_object,
    node_by_id,
    ready_pending_nodes,
    render_workflow_mermaid,
    reset_nodes_for_retry,
    summarize_activity,
    terminal_node_ids,
    update_node,
    workflow_decision,
    workflow_requests_retry,
    workflow_usage,
)
from harness.core.schemas import ToolCall, ToolResult
from harness.core.tools_verification import VerifyWorkTool
from harness.core.verification_structural import shell_command_changes_state
from harness.storage.sqlite import SQLiteStorage

workflow_app = typer.Typer(
    name="workflow",
    help="Run defended dynamic workflows with planned, verified agent steps.",
    no_args_is_help=True,
)
console = Console()

_READ_ONLY_TOOLS = {"read_file", "list_dir", "glob", "web_search", "fetch_url", "shell"}
_VERIFY_TOOLS = (_READ_ONLY_TOOLS - {"shell"}) | {"verify_work"}
_TOOL_CATALOG = {
    "read_file": "read files from the workspace",
    "list_dir": "inspect directory contents",
    "glob": "find workspace files by pattern",
    "web_search": "search public web sources for current or external facts",
    "fetch_url": "fetch a specific public HTTP(S) URL",
    "write_file": "create or replace a workspace file",
    "edit_file": "edit an existing workspace file",
    "shell": "run a shell command in the workspace",
    "verify_work": "run a concrete read-only verification command and record pass/fail evidence",
}
_ALL_WORKFLOW_TOOLS = set(_TOOL_CATALOG)
_PSEUDO_TOOL_TEXT_RE = re.compile(
    r"<\|?tool_call\b|<tool_call\|>|call:[a-zA-Z_][\w-]*\{",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RolePolicy:
    role: str
    instructions: str
    tool_include: set[str] | None
    require_tools: bool
    allow_mutation: bool


@dataclass(frozen=True)
class NodeCandidate:
    index: int
    result: str
    activity: list[ActivityEvent]
    session_id: str
    error: str = ""


class _ReadOnlyShellTool:
    name = "shell"
    approval = "prompt"
    effect_scope = "read_only"

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self.description = (
            "Run read-only shell probes in the workspace for directly observable facts "
            "such as local clock, timezone-aware clock conversions, environment, "
            "installed tools, and command output. "
            "Mutating shell syntax is refused before execution."
        )
        self.parameters_schema = getattr(delegate, "parameters_schema", {})
        self.clean_env = bool(getattr(delegate, "clean_env", False))
        self.pipefail = bool(getattr(delegate, "pipefail", False))

    def bind_activity_context(self, **kwargs: Any) -> None:
        bind = getattr(self._delegate, "bind_activity_context", None)
        if callable(bind):
            bind(**kwargs)

    async def __call__(self, call: ToolCall) -> ToolResult:
        command = call.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return await self._delegate(call)
        if shell_command_changes_state(command):
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=(
                    "refused read-only shell command: this workflow role can inspect "
                    "the environment but cannot mutate workspace state"
                ),
                is_error=True,
                metadata={"read_only_shell_refused": True},
            )
        return await self._delegate(call)


_RAW_TOOL_EVIDENCE_NOTICE = "Node produced tool evidence before final text was available."


def _preview_for_fallback(event: ActivityEvent) -> str:
    raw_preview = str(event.data.get("content_preview") or "")
    preview = " ".join(raw_preview.split())
    try:
        content_size = int(event.data.get("content_size") or 0)
    except (TypeError, ValueError):
        content_size = 0
    shown_size = len(raw_preview.encode("utf-8"))
    if content_size > shown_size:
        preview = (
            f"{preview} [truncated preview: showing {shown_size} of "
            f"{content_size} bytes; do not infer exact missing suffix]"
        )
    return preview


def _activity_fallback_result(node: WorkflowNode, activity: list[ActivityEvent]) -> str:
    completed_tools = [
        event
        for event in activity
        if event.kind == "tool_call.completed" and not event.data.get("is_error")
    ]
    if not completed_tools:
        return ""
    lines = [_RAW_TOOL_EVIDENCE_NOTICE]
    for event in completed_tools[-8:]:
        name = str(event.data.get("name") or "tool")
        arguments = event.data.get("arguments") or {}
        metadata = event.data.get("metadata") or {}
        preview = _preview_for_fallback(event)
        if name == "web_search" and isinstance(metadata, dict):
            results = metadata.get("results")
            if isinstance(results, list) and results:
                lines.append("- web_search results:")
                for index, item in enumerate(results[:5], 1):
                    if not isinstance(item, dict):
                        continue
                    title = " ".join(str(item.get("title") or "").split())
                    content = " ".join(str(item.get("content") or "").split())
                    url = str(item.get("url") or "").strip()
                    summary = title or url or f"result {index}"
                    if content:
                        summary = f"{summary} — {content}"
                    if url:
                        summary = f"{summary} ({url})"
                    lines.append(f"  {index}. {summary}")
                continue
        if name in {"write_file", "edit_file", "read_file"} and isinstance(arguments, dict):
            path = str(arguments.get("path") or "").strip()
            detail = f"{name} {path}".strip()
            content = arguments.get("content")
            if not isinstance(content, str):
                content = arguments.get("new")
            if isinstance(content, str) and content and len(content) <= 500:
                detail = f"{detail}; content: {content}"
        elif name == "shell" and isinstance(arguments, dict):
            detail = f"shell {str(arguments.get('command') or '').strip()}"
        elif name == "verify_work":
            detail = "verify_work passed"
        else:
            detail = name
        if preview:
            detail = f"{detail}: {preview}"
        lines.append(f"- {detail}")
    if node.kind == "work" and any(
        str(event.data.get("name") or "") == "verify_work" for event in completed_tools
    ):
        lines.append("The final state was verified by verify_work.")
    return "\n".join(lines)


def _timeout_fallback_result(node: WorkflowNode, run: WorkflowRun) -> str:
    if node.kind == "plan":
        return (
            "Plan:\n"
            "- Check any assumptions that affect the objective before changing state.\n"
            "- Use the available tools to complete the objective.\n"
            "- Verify the final result with concrete evidence before reporting success.\n\n"
            "Expected outcome: later workflow nodes produce evidence for the objective.\n"
            "Confidence: medium until tool evidence confirms the assumptions.\n\n"
            f"Objective:\n{run.goal}"
        )
    return _deterministic_node_result(node=node, run=run)


def _dependency_results_for_node(node: WorkflowNode, run: WorkflowRun) -> list[str]:
    by_id = node_by_id(run)
    results: list[str] = []
    for dep_id in node.depends_on:
        dep = by_id.get(dep_id)
        if dep is not None and dep.status == "completed" and dep.result.strip():
            results.append(dep.result.strip())
    return results


def _deterministic_node_result(*, node: WorkflowNode, run: WorkflowRun) -> str:
    if node.kind == "merge":
        completed = _dependency_results_for_node(node, run) or [
            item.result.strip()
            for item in run.nodes
            if item.kind in {"research", "work", "verify"}
            and item.status == "completed"
            and item.result.strip()
        ]
        if completed:
            evidence = completed[-2:]
            if any(item.startswith(_RAW_TOOL_EVIDENCE_NOTICE) for item in evidence):
                return (
                    "I could not produce a verified final answer from the available evidence. "
                    "Here is the evidence that was gathered.\n\n" + "\n\n".join(evidence)
                )
            return "Completed with workflow evidence.\n\n" + "\n\n".join(evidence)
    return ""


def _review_dependency_passed(*, node: WorkflowNode, run: WorkflowRun) -> bool:
    by_id = node_by_id(run)
    for dep_id in node.depends_on:
        dep = by_id.get(dep_id)
        if dep is None or dep.kind != "review" or dep.status != "completed":
            continue
        if workflow_decision(dep.result) == "pass":
            return True
    return False


def _budget_recovery_node_result(*, node: WorkflowNode, run: WorkflowRun) -> str:
    if node.kind == "refute" and _review_dependency_passed(node=node, run=run):
        return "Review decision is pass.\nWORKFLOW_DECISION: pass"
    return _deterministic_node_result(node=node, run=run)


def _candidate_with_deterministic_result_if_needed(
    *,
    node: WorkflowNode,
    run: WorkflowRun,
    candidate: NodeCandidate,
) -> NodeCandidate:
    if candidate.result.strip() and not _PSEUDO_TOOL_TEXT_RE.search(candidate.result):
        return candidate
    fallback = _activity_fallback_result(node, candidate.activity) or _deterministic_node_result(
        node=node,
        run=run,
    )
    if not fallback:
        return candidate
    return NodeCandidate(
        index=candidate.index,
        result=fallback,
        activity=candidate.activity,
        session_id=candidate.session_id,
        error=candidate.error,
    )


def _candidate_with_verified_objective_status_if_needed(
    *,
    node: WorkflowNode,
    run: WorkflowRun,
    candidate: NodeCandidate,
) -> NodeCandidate:
    if not any(item.kind == "objective_status" for item in node.expected_evidence):
        return candidate
    if workflow_decision(candidate.result):
        return candidate
    if not any(_workflow_tool_event_changes_state(event) for event in candidate.activity):
        return candidate

    evidence_node = replace(
        node,
        expected_evidence=tuple(
            item for item in node.expected_evidence if item.kind != "objective_status"
        ),
    )
    evidence_ok, _results = evaluate_node_evidence(
        node=evidence_node,
        result=candidate.result,
        activity=candidate.activity,
        run=run,
    )
    if not evidence_ok:
        return candidate

    result = candidate.result.strip()
    if not result:
        result = _activity_fallback_result(node, candidate.activity)
    if not result.strip():
        return candidate
    return NodeCandidate(
        index=candidate.index,
        result=f'{result.strip()}\n\n{{"status":"pass"}}',
        activity=candidate.activity,
        session_id=candidate.session_id,
        error=candidate.error,
    )


def _successful_completed_tool_events(activity: list[ActivityEvent]) -> list[ActivityEvent]:
    return [
        event
        for event in activity
        if event.kind == "tool_call.completed" and not event.data.get("is_error")
    ]


def _activity_error_message(activity: list[ActivityEvent]) -> str:
    for event in reversed(activity):
        if event.kind not in {"agent_run.failed", "error"}:
            continue
        data = event.data or {}
        error = str(data.get("error") or data.get("message") or "").strip()
        kind = str(data.get("kind") or "").strip()
        if error and kind:
            return f"{kind}: {error}"
        if error:
            return error
    return ""


def _metadata_seconds(node: WorkflowNode, key: str) -> float:
    raw = node.metadata.get(key)
    try:
        return max(0.0, float(raw)) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _changed_file_paths(activity: list[ActivityEvent]) -> list[Path]:
    paths: list[Path] = []
    for event in _successful_completed_tool_events(activity):
        if str(event.data.get("name") or "") not in {"write_file", "edit_file"}:
            continue
        arguments = event.data.get("arguments") or {}
        if not isinstance(arguments, dict):
            continue
        raw_path = str(arguments.get("path") or "").strip()
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.is_absolute() or any(part == ".harness" for part in path.parts):
            continue
        if path not in paths:
            paths.append(path)
    return paths


def _read_file_paths(activity: list[ActivityEvent]) -> list[Path]:
    paths: list[Path] = []
    for event in _successful_completed_tool_events(activity):
        if str(event.data.get("name") or "") != "read_file":
            continue
        arguments = event.data.get("arguments") or {}
        if not isinstance(arguments, dict):
            continue
        raw_path = str(arguments.get("path") or "").strip()
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.is_absolute() or any(part == ".harness" for part in path.parts):
            continue
        if path not in paths:
            paths.append(path)
    return paths


def _source_paths_for_auto_verify(activity: list[ActivityEvent]) -> list[Path]:
    paths = _read_file_paths(activity)
    summary = summarize_activity(activity)
    raw_paths = summary.get("read_paths") if isinstance(summary, dict) else []
    if not isinstance(raw_paths, list):
        return paths
    for raw_path in raw_paths:
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path.strip())
        if path.is_absolute() or any(part == ".harness" for part in path.parts):
            continue
        if path not in paths:
            paths.append(path)
    return paths


def _exact_file_compare_command(path: str, content: str) -> str:
    quoted_path = shlex.quote(path)
    return (
        f"printf %s {shlex.quote(content)} | cmp -s - {quoted_path} "
        f"&& [ \"$(wc -c < {quoted_path} | tr -d ' ')\" -eq "
        f"{len(content.encode('utf-8'))} ]"
    )


def _python_exact_stdout_command(path: str, content: str) -> str:
    code = (
        "import subprocess, sys; "
        f"expected={content.encode('utf-8')!r}; "
        f"out=subprocess.check_output([sys.executable, {path!r}]); "
        "assert out == expected, repr(out)"
    )
    return "python3 -c " + shlex.quote(code)


def _python_test_files_command(paths: list[Path]) -> str:
    path_values = [str(path) for path in paths]
    code = (
        "import importlib.util, inspect, pathlib, sys, unittest; "
        f"paths={path_values!r}; "
        "called=0; cases=0; ok=True; "
        "\nfor i,p in enumerate(paths):\n"
        "    path=pathlib.Path(p)\n"
        "    spec=importlib.util.spec_from_file_location(f'_harness_test_{i}', path)\n"
        "    if spec is None or spec.loader is None:\n"
        "        raise RuntimeError(f'cannot load {p}')\n"
        "    module=importlib.util.module_from_spec(spec)\n"
        "    sys.modules[spec.name]=module\n"
        "    spec.loader.exec_module(module)\n"
        "    suite=unittest.defaultTestLoader.loadTestsFromModule(module)\n"
        "    cases += suite.countTestCases()\n"
        "    if cases:\n"
        "        result=unittest.TextTestRunner(verbosity=2).run(suite)\n"
        "        ok = ok and result.wasSuccessful()\n"
        "    for name,obj in sorted(vars(module).items()):\n"
        "        if not name.startswith('test_') or not callable(obj):\n"
        "            continue\n"
        "        sig=inspect.signature(obj)\n"
        "        required=[p for p in sig.parameters.values() if p.default is p.empty and p.kind in (p.POSITIONAL_ONLY,p.POSITIONAL_OR_KEYWORD,p.KEYWORD_ONLY)]\n"
        "        if required:\n"
        "            continue\n"
        "        obj()\n"
        "        called += 1\n"
        "if not ok:\n"
        "    raise SystemExit(1)\n"
        "if cases == 0 and called == 0:\n"
        "    raise SystemExit('no runnable tests found')\n"
        "print(f'PASSED {cases} unittest case(s), {called} test function(s)')"
    )
    return "python3 -c " + shlex.quote(code)


def _goal_mentions_path(goal: str, path: Path) -> bool:
    goal_lower = goal.lower()
    name = path.name.lower()
    return name in goal_lower or str(path).lower() in goal_lower


def _infer_auto_verify_command(
    *,
    cwd: Path,
    run: WorkflowRun,
    node: WorkflowNode,
    activity: list[ActivityEvent],
) -> str:
    file_requests = _exact_file_content_requests(node=node, run=run)
    stdout_requests = _exact_stdout_requests(node=node, run=run)
    commands: list[str] = []
    for path, content in file_requests:
        commands.append(_exact_file_compare_command(path, content))
    for path, content in stdout_requests:
        commands.append(_python_exact_stdout_command(path, content))
    if commands:
        return " && ".join(commands)

    changed_paths = _changed_file_paths(activity)
    test_python = [
        path
        for path in changed_paths
        if path.suffix == ".py" and path.name.startswith("test_") and (cwd / path).is_file()
    ]
    if test_python:
        return _python_test_files_command(test_python)

    runnable_python = [
        path
        for path in changed_paths
        if path.suffix == ".py" and (cwd / path).is_file() and not path.name.startswith("test_")
    ]
    if not runnable_python:
        return _source_artifact_handoff_command(
            cwd=cwd,
            changed_paths=changed_paths,
            source_paths=_source_paths_for_auto_verify(activity),
        )
    mentioned = [path for path in runnable_python if _goal_mentions_path(run.goal, path)]
    selected = (
        mentioned[0] if mentioned else runnable_python[0] if len(runnable_python) == 1 else None
    )
    if selected is None:
        return _source_artifact_handoff_command(
            cwd=cwd,
            changed_paths=changed_paths,
            source_paths=_source_paths_for_auto_verify(activity),
        )
    return "python3 " + shlex.quote(str(selected))


def _source_artifact_handoff_command(
    *,
    cwd: Path,
    changed_paths: list[Path],
    source_paths: list[Path],
) -> str:
    existing_outputs = [path for path in changed_paths if (cwd / path).is_file()]
    source_patterns: list[str] = []
    for path in source_paths:
        if path in changed_paths:
            continue
        if path.is_absolute() or any(part == ".harness" for part in path.parts):
            continue
        raw = str(path)
        if (any(char in raw for char in "*?[]") or (cwd / path).exists()) and (
            raw not in source_patterns
        ):
            source_patterns.append(raw)
    if not existing_outputs or not source_patterns:
        return ""
    report_text = (cwd / existing_outputs[0]).read_text(encoding="utf-8", errors="ignore")
    if not _source_artifact_has_generic_check(
        cwd=cwd,
        report_text=report_text,
        source_patterns=source_patterns,
    ):
        return ""
    code = (
        "import collections, glob, pathlib, re\n"
        f"report_path=pathlib.Path({str(existing_outputs[0])!r})\n"
        f"source_patterns={source_patterns[:40]!r}\n"
        "report=report_path.read_text(encoding='utf-8')\n"
        "report_lower=report.lower()\n"
        "assert report.strip(), f'{report_path} is empty'\n"
        "checks=[]\n"
        "kv=collections.defaultdict(collections.Counter)\n"
        "seen_source_files=set()\n"
        "def require(condition, message):\n"
        "    if not condition:\n"
        "        raise AssertionError(message)\n"
        "def matched_paths(raw):\n"
        "    matches=sorted(pathlib.Path(p) for p in glob.glob(raw))\n"
        "    path=pathlib.Path(raw)\n"
        "    if path.exists() and path not in matches:\n"
        "        matches.append(path)\n"
        "    return matches\n"
        "def source_files(path):\n"
        "    if path.is_file():\n"
        "        yield path\n"
        "        return\n"
        "    if not path.is_dir():\n"
        "        return\n"
        "    seen=0\n"
        "    for child in sorted(path.rglob('*')):\n"
        "        if seen >= 1000:\n"
        "            break\n"
        "        if not child.is_file() or any(part.startswith('.') for part in child.parts):\n"
        "            continue\n"
        "        try:\n"
        "            if child.stat().st_size > 1000000:\n"
        "                continue\n"
        "        except OSError:\n"
        "            continue\n"
        "        seen += 1\n"
        "        yield child\n"
        "for raw in source_patterns:\n"
        "    for path in matched_paths(raw):\n"
        "        if not path.exists():\n"
        "            continue\n"
        "        if path.is_dir():\n"
        "            subdirs=[p for p in path.iterdir() if p.is_dir() and not p.name.startswith('.')]\n"
        "            if len(subdirs) > 1 and str(len(subdirs)) in report:\n"
        "                require(str(len(subdirs)) in report, f'missing directory count {len(subdirs)} for {path}')\n"
        "                checks.append(f'{path}:dir-count={len(subdirs)}')\n"
        "        for source_path in source_files(path):\n"
        "            source_key=str(source_path.resolve())\n"
        "            if source_key in seen_source_files:\n"
        "                continue\n"
        "            seen_source_files.add(source_key)\n"
        "            text=source_path.read_text(encoding='utf-8', errors='ignore')\n"
        "            for line in text.splitlines():\n"
        "                match=re.match(r'\\s*([A-Za-z_][\\w.-]*)\\s*=\\s*[\"\\']([^\"\\']{1,80})[\"\\']\\s*$', line)\n"
        "                if not match:\n"
        "                    continue\n"
        "                key,value=match.group(1),match.group(2).strip()\n"
        "                if re.search(r'[A-Za-z]', value):\n"
        "                    kv[key][value] += 1\n"
        "for key,counter in sorted(kv.items()):\n"
        "    if not (2 <= len(counter) <= 20):\n"
        "        continue\n"
        "    mentioned=sum(1 for value in counter if value.lower() in report_lower)\n"
        "    if mentioned < min(2, len(counter)):\n"
        "        continue\n"
        "    for value,count in sorted(counter.items()):\n"
        "        require(value.lower() in report_lower, f'missing value {value!r} for {key}')\n"
        "        require(str(count) in report, f'missing count {count} for {key}={value}')\n"
        "    checks.append(f'{key}:distribution')\n"
        "require(checks, 'no source-derived artifact facts were checked')\n"
        "print('PASSED source-artifact checks: ' + ', '.join(checks))\n"
    )
    return "python3 -c " + shlex.quote(code)


def _source_artifact_has_generic_check(
    *,
    cwd: Path,
    report_text: str,
    source_patterns: list[str],
) -> bool:
    report_lower = report_text.lower()
    kv: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    seen_source_files: set[Path] = set()
    for raw in source_patterns:
        matches = sorted(Path(path) for path in glob.glob(str(cwd / raw)))
        direct = cwd / raw
        if direct.exists() and direct not in matches:
            matches.append(direct)
        for path in matches:
            if path.is_dir():
                subdirs = [
                    child
                    for child in path.iterdir()
                    if child.is_dir() and not child.name.startswith(".")
                ]
                if len(subdirs) > 1 and str(len(subdirs)) in report_text:
                    return True
            for source_path in _iter_source_artifact_files(path):
                source_key = source_path.resolve()
                if source_key in seen_source_files:
                    continue
                seen_source_files.add(source_key)
                text = source_path.read_text(encoding="utf-8", errors="ignore")
                for line in text.splitlines():
                    match = re.match(
                        r"\s*([A-Za-z_][\w.-]*)\s*=\s*[\"']([^\"']{1,80})[\"']\s*$",
                        line,
                    )
                    if not match:
                        continue
                    value = match.group(2).strip()
                    if re.search(r"[A-Za-z]", value):
                        kv[match.group(1)][value] += 1
    for counter in kv.values():
        if not (2 <= len(counter) <= 20):
            continue
        mentioned = sum(1 for value in counter if value.lower() in report_lower)
        if mentioned >= min(2, len(counter)):
            return True
    return False


def _iter_source_artifact_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        return
    seen = 0
    for child in sorted(path.rglob("*")):
        if seen >= 1000:
            break
        if not child.is_file() or any(part.startswith(".") for part in child.parts):
            continue
        try:
            if child.stat().st_size > 1_000_000:
                continue
        except OSError:
            continue
        seen += 1
        yield child


def _summary_paths(metadata: dict[str, Any], key: str) -> list[Path]:
    summary = metadata.get("activity_summary")
    if not isinstance(summary, dict):
        return []
    raw_paths = summary.get(key)
    if not isinstance(raw_paths, list):
        return []
    paths: list[Path] = []
    for raw_path in raw_paths:
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path.strip())
        if path.is_absolute() or any(part == ".harness" for part in path.parts):
            continue
        if path not in paths:
            paths.append(path)
    return paths


def _dependency_source_artifact_verify_command(
    *,
    cwd: Path,
    run: WorkflowRun,
    node: WorkflowNode,
) -> str:
    by_id = node_by_id(run)
    changed_paths: list[Path] = []
    source_paths: list[Path] = []
    for dep_id in node.depends_on:
        dep = by_id.get(dep_id)
        if dep is None:
            continue
        for path in _summary_paths(dep.metadata, "changed_paths"):
            if path not in changed_paths:
                changed_paths.append(path)
        for path in _summary_paths(dep.metadata, "read_paths"):
            if path not in source_paths:
                source_paths.append(path)
    return _source_artifact_handoff_command(
        cwd=cwd,
        changed_paths=changed_paths,
        source_paths=source_paths,
    )


async def _append_node_activity(*, cwd: Path, event: ActivityEvent) -> None:
    storage = build_storage(db=_workspace_db_path(cwd), in_memory=False, cwd=cwd)
    try:
        await storage.append_activity(event)  # type: ignore[attr-defined]
    finally:
        if isinstance(storage, SQLiteStorage):
            await storage.close()


async def _run_auto_verify_work(
    *,
    cwd: Path,
    session_id: str,
    command: str,
) -> list[ActivityEvent]:
    call_id = f"workflow-auto-verify-{uuid4().hex[:10]}"
    dispatched = ActivityEvent(
        session_id=session_id,
        kind="tool_call.dispatched",
        data={
            "tool_call_id": call_id,
            "name": "verify_work",
            "arguments": {"command": command},
            "source": "workflow_auto_verify",
        },
    )
    await _append_node_activity(cwd=cwd, event=dispatched)
    result = await VerifyWorkTool(cwd=cwd)(
        ToolCall(id=call_id, name="verify_work", arguments={"command": command})
    )
    preview = result.content if len(result.content) <= 1000 else result.content[:999] + "…"
    completed = ActivityEvent(
        session_id=session_id,
        kind="tool_call.completed",
        data={
            "tool_call_id": call_id,
            "name": "verify_work",
            "is_error": result.is_error,
            "content_preview": preview,
            "content_size": len(result.content),
            "arguments": {"command": command},
            "metadata": {**(result.metadata or {}), "source": "workflow_auto_verify"},
        },
    )
    await _append_node_activity(cwd=cwd, event=completed)
    return [dispatched, completed]


def _safe_relative_output_path(raw_path: str) -> Path | None:
    path = Path(raw_path)
    if path.is_absolute() or ".." in path.parts or any(part == ".harness" for part in path.parts):
        return None
    return path


async def _materialize_exact_file_requests(
    *,
    cwd: Path,
    run: WorkflowRun,
    node: WorkflowNode,
    session_id: str,
    activity: list[ActivityEvent],
) -> list[ActivityEvent]:
    if node.kind != "work":
        return []
    if any(_workflow_tool_event_changes_state(event) for event in activity):
        return []
    requests = _exact_file_content_requests(node=node, run=run)
    if not requests:
        return []

    events: list[ActivityEvent] = []
    for path_text, content in requests:
        relative_path = _safe_relative_output_path(path_text)
        if relative_path is None:
            return []
        target = cwd / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        call_id = f"workflow-auto-write-{uuid4().hex[:10]}"
        completed = ActivityEvent(
            session_id=session_id,
            kind="tool_call.completed",
            data={
                "tool_call_id": call_id,
                "name": "write_file",
                "is_error": False,
                "content_preview": f"wrote {len(content.encode('utf-8'))} bytes to {path_text}",
                "content_size": len(content),
                "arguments": {"path": path_text, "content": content},
                "metadata": {"source": "workflow_exact_file_fallback"},
            },
        )
        await _append_node_activity(cwd=cwd, event=completed)
        events.append(completed)

    verify_command = _infer_auto_verify_command(
        cwd=cwd,
        run=run,
        node=node,
        activity=[*activity, *events],
    )
    if verify_command:
        events.extend(
            await _run_auto_verify_work(
                cwd=cwd,
                session_id=session_id,
                command=verify_command,
            )
        )
    return events


async def _auto_verify_candidate_if_needed(
    *,
    cwd: Path,
    run: WorkflowRun,
    node: WorkflowNode,
    candidate: NodeCandidate,
) -> NodeCandidate:
    if any(item.kind == "verify_work_passed" for item in node.expected_evidence):
        command = _dependency_source_artifact_verify_command(cwd=cwd, run=run, node=node)
        if command:
            auto_events = await _run_auto_verify_work(
                cwd=cwd,
                session_id=candidate.session_id,
                command=command,
            )
            updated_activity = [*candidate.activity, *auto_events]
            fallback = _activity_fallback_result(node, updated_activity)
            result = candidate.result
            if fallback and "verify_work" not in result:
                result = f"{result}\n\n{fallback}".strip()
            if not result:
                result = fallback
            return NodeCandidate(
                index=candidate.index,
                result=result,
                activity=updated_activity,
                session_id=candidate.session_id,
                error=candidate.error,
            )
    if not any(item.kind == "verify_work_if_state_changed" for item in node.expected_evidence):
        return candidate
    if not any(_workflow_tool_event_changes_state(event) for event in candidate.activity):
        return candidate
    if _verify_work_after_last_change(
        candidate.activity,
        node=node,
        run=run,
        exact_file_requests=_exact_file_content_requests(node=node, run=run),
        exact_stdout_requests=_exact_stdout_requests(node=node, run=run),
    ):
        return candidate

    command = _infer_auto_verify_command(
        cwd=cwd,
        run=run,
        node=node,
        activity=candidate.activity,
    )
    if not command:
        return candidate
    auto_events = await _run_auto_verify_work(
        cwd=cwd,
        session_id=candidate.session_id,
        command=command,
    )
    updated_activity = [*candidate.activity, *auto_events]
    fallback = _activity_fallback_result(node, updated_activity)
    result = candidate.result
    if fallback and "verify_work" not in result:
        result = f"{result}\n\n{fallback}".strip()
    if not result:
        result = fallback
    return NodeCandidate(
        index=candidate.index,
        result=result,
        activity=updated_activity,
        session_id=candidate.session_id,
        error=candidate.error,
    )


_ROLE_POLICIES: dict[str, RolePolicy] = {
    "planner": RolePolicy(
        role="planner",
        instructions=(
            "Plan the work, assumptions, expected outcome, and evidence. Do not change state."
        ),
        tool_include=set(),
        require_tools=False,
        allow_mutation=False,
    ),
    "researcher": RolePolicy(
        role="researcher",
        instructions="Use read-only tools to check assumptions and gather evidence.",
        tool_include=_READ_ONLY_TOOLS,
        require_tools=True,
        allow_mutation=False,
    ),
    "implementer": RolePolicy(
        role="implementer",
        instructions=(
            "Do the requested work. If you change state, verify the final state before "
            "claiming completion."
        ),
        tool_include=None,
        require_tools=True,
        allow_mutation=True,
    ),
    "verifier": RolePolicy(
        role="verifier",
        instructions=(
            "Independently check the result with read-only evidence or verify_work. Do not "
            "make product changes."
        ),
        tool_include=_VERIFY_TOOLS,
        require_tools=True,
        allow_mutation=False,
    ),
    "reviewer": RolePolicy(
        role="reviewer",
        instructions=(
            "Adversarially review the evidence. End with WORKFLOW_DECISION: pass or "
            "WORKFLOW_DECISION: retry. Treat truncated previews as incomplete context, "
            "not as exact contradictory content."
        ),
        tool_include=set(),
        require_tools=False,
        allow_mutation=False,
    ),
    "refuter": RolePolicy(
        role="refuter",
        instructions=(
            "Challenge the reviewer against the evidence. End with WORKFLOW_DECISION: pass "
            "or WORKFLOW_DECISION: retry. Treat truncated previews as incomplete context, "
            "not as exact contradictory content."
        ),
        tool_include=set(),
        require_tools=False,
        allow_mutation=False,
    ),
    "merger": RolePolicy(
        role="merger",
        instructions=(
            "Synthesize the defended outcome. Do not add new claims beyond verified "
            "evidence; preserve concrete values and any source limitations from the "
            "verified nodes."
        ),
        tool_include=set(),
        require_tools=False,
        allow_mutation=False,
    ),
}


async def _noop_task_attachment(*_args: object, **_kwargs: object) -> tuple[None, None]:
    return None, None


def _utcnow_text() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="seconds")


def _default_model(provider: str) -> str:
    if provider == "openrouter":
        return "openai/gpt-5.4-nano"
    return "gemma4:latest"


def _workspace_db_path(cwd: Path) -> Path:
    target = cwd / ".harness" / "harness.db"
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _policy_for_node(node: WorkflowNode) -> RolePolicy:
    role = (node.role or node.kind).strip().lower()
    policy = _ROLE_POLICIES.get(role) or _ROLE_POLICIES.get(node.kind)
    if policy is None:
        policy = _ROLE_POLICIES["implementer" if node.allow_mutation else "researcher"]
    if node.allow_mutation and not policy.allow_mutation:
        return RolePolicy(
            role=policy.role,
            instructions=policy.instructions,
            tool_include=None,
            require_tools=True,
            allow_mutation=True,
        )
    return policy


def _evidence_requirements_text(node: WorkflowNode) -> str:
    if not node.expected_evidence:
        return "- result_nonempty"
    lines: list[str] = []
    for item in node.expected_evidence:
        suffix = f" ({item.name})" if item.name else ""
        required = "required" if item.required else "optional"
        description = item.description
        if item.kind == "review_decision" and not description:
            description = (
                "include exactly one final line: WORKFLOW_DECISION: pass "
                "or WORKFLOW_DECISION: retry"
            )
        elif item.kind == "planning_only" and not description:
            description = "produce only a plan; do not call tools or claim the work is complete"
        elif item.kind == "no_state_change" and not description:
            description = "do not edit files or mutate workspace state"
        elif item.kind == "verify_work_if_state_changed" and not description:
            description = (
                "if this node changes files or state, call verify_work after the final "
                "mutation; for source-derived artifacts, recompute from source inputs or "
                "run a broader test command"
            )
        elif item.kind == "environment_checked" and not description:
            description = (
                "before the first mutation, inspect the local workspace or runtime "
                "environment with a read-only tool; use missing-tool evidence to choose "
                "an available path or research one; after installs or updates, re-check "
                "the environment"
            )
        elif item.kind == "verify_work_passed" and not description:
            description = (
                "include a passing verify_work tool result; for source-derived "
                "artifacts, recompute from source inputs or run a broader test command"
            )
        elif item.kind == "objective_status" and not description:
            description = (
                'include a structured status flag such as {"status":"pass"} in the final '
                "assistant response only when the objective is complete; use "
                '{"status":"retry"} if more work is needed; do not emit this flag via '
                "shell or tool output"
            )
        description_text = f": {description}" if description else ""
        lines.append(f"- {item.kind}{suffix} [{required}]{description_text}")
    return "\n".join(lines)


def _tool_catalog_text(node: WorkflowNode) -> str:
    policy = _policy_for_node(node)
    if policy.tool_include is None:
        role_tools = sorted(_ALL_WORKFLOW_TOOLS)
    else:
        role_tools = sorted(policy.tool_include)
    if not role_tools:
        return "Available tools for this node: none."
    catalog = ", ".join(
        f"{name} ({_TOOL_CATALOG[name]})" for name in role_tools if name in _TOOL_CATALOG
    )
    return f"Available tools for this node: {catalog}."


def _node_session_id(run: WorkflowRun, node: WorkflowNode, *, candidate_index: int = 0) -> str:
    base = f"workflow_{run.id}_{node.id}".replace("-", "_")
    parts = [base]
    if node.attempts > 1:
        parts.append(f"attempt{node.attempts}")
    if candidate_index > 0:
        parts.append(f"candidate{candidate_index}")
    return "_".join(parts)


def _workflow_system_prompt(base_prompt: str, node: WorkflowNode) -> str:
    policy = _policy_for_node(node)
    mutation_rule = (
        "This role may mutate workspace state when needed."
        if policy.allow_mutation
        else "This role is read-only. Do not edit files or mutate workspace state."
    )
    shell_available = policy.tool_include is None or "shell" in policy.tool_include
    shell_rule = (
        "- Use exact tool names. For shell commands use `shell` with the real command "
        "(for example, `pwd`), not an echo of the command you intend to run.\n"
        if shell_available
        else "- Use only the exact tool names available to this role; do not invent shell-like tools.\n"
    )
    planner_rule = (
        "- Planning nodes should plan only; do not execute commands or claim completion.\n"
        if (node.role or node.kind).strip().lower() == "planner"
        else ""
    )
    verify_work_required_rule = (
        "- This node has a verify_work evidence gate; if you changed state, verify after "
        "the final mutation. verify_work runs in a clean command environment, so use it "
        "to prove the deliverable works outside Harness' ambient runtime. For artifacts "
        "derived from source files, use a check that recomputes from the source inputs or "
        "runs a broader verifier, not just a grep of the generated output.\n"
        if any(item.kind == "verify_work_if_state_changed" for item in node.expected_evidence)
        else ""
    )
    environment_required_rule = (
        "- This node has an environment check gate; before the first mutation, inspect "
        "the local workspace or runtime with a read-only tool so your implementation is "
        "based on the target environment. If a tool is missing, use that evidence to pick "
        "an available path or research one with the tools you have. After any install or "
        "update, re-check the environment before relying on it.\n"
        if any(item.kind == "environment_checked" for item in node.expected_evidence)
        else ""
    )
    read_only_verify_rule = (
        "- For read-only verification, do not create temporary files or redirect output "
        "into the workspace. Compare through stdout, pipes, process substitution, or "
        "direct test expressions.\n"
        if (
            not policy.allow_mutation
            and any(
                item.kind in {"verify_work_passed", "verify_work_if_state_changed"}
                for item in node.expected_evidence
            )
        )
        else ""
    )
    objective_status_rule = (
        "- This node has an objective status gate; put the structured status flag in your "
        "final assistant response, not in a shell command or tool output.\n"
        if any(item.kind == "objective_status" for item in node.expected_evidence)
        else ""
    )
    return (
        f"{base_prompt}\n\n"
        "## Workflow policy\n\n"
        "- Work as this node only, using the Goal as the source of truth.\n"
        "- Ground claims in tool evidence; if evidence is missing, say so.\n"
        "- Do not simulate tool calls or tool outputs in final text; call the tool, "
        "then summarize the evidence in prose.\n"
        "- Treat missing tools, unsupported flags, and unsupported command syntax as "
        "environment evidence; inspect version/help or choose another available path "
        "before retrying.\n"
        f"{shell_rule}"
        f"{planner_rule}"
        f"{verify_work_required_rule}"
        f"{environment_required_rule}"
        f"{read_only_verify_rule}"
        f"{objective_status_rule}"
        "- Avoid generic handoff text.\n"
        "- Do not claim that later workflow nodes have already completed.\n"
        f"- Role: {policy.role}. {policy.instructions}\n"
        f"- Mutation policy: {mutation_rule}\n"
    )


async def _stream_workflow_text_node(
    *,
    provider: str,
    model: str,
    config: Any,
    system_prompt: str,
    prompt: str,
) -> str:
    adapter = _build_adapter(provider, base_url=None, config=config)
    text_parts: list[str] = []
    async for event in adapter.stream(
        model=model,
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=prompt),
        ],
        max_tokens=900,
        temperature=0.2,
    ):
        if isinstance(event, TextDelta):
            text_parts.append(event.text)
        elif isinstance(event, Done):
            if event.final_message and event.final_message.content:
                return event.final_message.content.strip()
            break
    return "".join(text_parts).strip()


def _dependency_context(run: WorkflowRun, node: WorkflowNode) -> str:
    by_id = {item.id: item for item in run.nodes}
    lines: list[str] = []
    if node.depends_on:
        lines.append("Previous workflow node results:")
    for dep_id in node.depends_on:
        dep = by_id.get(dep_id)
        if dep is None:
            continue
        result = " ".join((dep.result or dep.error or "").split())
        if len(result) > 1600:
            result = result[:1599].rstrip() + "... [truncated dependency result]"
        lines.append(f"- {dep.id} ({dep.status}): {result or '(no result)'}")
    feedback = node.metadata.get("review_feedback") or run.metadata.get("review_feedback")
    if feedback:
        compact = " ".join(str(feedback).split())
        if len(compact) > 1600:
            compact = compact[:1599].rstrip() + "..."
        lines.append(f"Prior review feedback to address: {compact}")
    failure_history = list(node.metadata.get("failure_history") or [])
    if failure_history:
        lines.append("Previous failed attempts for this node:")
        for item in failure_history[-2:]:
            error = " ".join(str(item.get("error") or "").split())
            result = " ".join(str(item.get("result") or "").split())
            activity_summary = item.get("activity_summary")
            failed_tools = item.get("failed_tools")
            if len(result) > 800:
                result = result[:799].rstrip() + "... [truncated prior result]"
            lines.append(
                f"- attempt {item.get('attempt')}: {error or '(no error)'}"
                + (f"; prior result: {result}" if result else "")
            )
            if isinstance(activity_summary, dict):
                state_changed = bool(activity_summary.get("state_changed"))
                verify_passed = bool(activity_summary.get("verify_work_passed"))
                lines.append(
                    "  evidence summary: "
                    f"state_changed={state_changed}, verify_work_passed={verify_passed}"
                )
                environment_observations = activity_summary.get("environment_observations")
                if isinstance(environment_observations, list) and environment_observations:
                    lines.append("  environment observations:")
                    for observation in environment_observations[:3]:
                        compact = " ".join(str(observation).split())
                        if len(compact) > 700:
                            compact = compact[:699].rstrip() + "..."
                        lines.append(f"  - {compact}")
            if isinstance(failed_tools, list) and failed_tools:
                lines.append("  failed tool evidence:")
                for failed in failed_tools[:3]:
                    compact = " ".join(str(failed).split())
                    if len(compact) > 700:
                        compact = compact[:699].rstrip() + "..."
                    lines.append(f"  - {compact}")
                lines.append(
                    "Retry guidance: do not repeat failed command syntax. Inspect the "
                    "tool version/help or choose another available local path before "
                    "mutating again."
                )
        changed_state_before = any(
            isinstance(item.get("activity_summary"), dict)
            and item["activity_summary"].get("state_changed")
            for item in failure_history
        )
        environment_observed = any(
            isinstance(item.get("activity_summary"), dict)
            and item["activity_summary"].get("environment_observations")
            for item in failure_history
        )
        verified_missing_status = any(
            str(item.get("error") or "") == "objective status is missing"
            and isinstance(item.get("activity_summary"), dict)
            and item["activity_summary"].get("verify_work_passed")
            for item in failure_history
        )
        source_artifact_compare_failed = any(
            "did not compare dependency artifacts with source inputs"
            in str(item.get("error") or "")
            or "did not compare generated artifacts with source inputs"
            in str(item.get("error") or "")
            for item in failure_history
            if isinstance(item, dict)
        )
        if environment_observed:
            lines.append(
                "Retry guidance: use the environment observations as evidence. "
                "Check the available local path, and use web_search or fetch_url "
                "for current docs when that would unblock the objective."
            )
        if source_artifact_compare_failed:
            lines.append(
                "Retry guidance: the next verify_work command must read both the source "
                "input file(s) and the generated artifact file(s), and compare the facts "
                "from source against the artifact contents in the same command."
            )
        if verified_missing_status:
            lines.append(
                "Retry guidance: previous attempts produced verified evidence but omitted "
                "the final structured status flag. Do not redo the work; summarize the "
                'evidence and include {"status":"pass"} in the final assistant response '
                "only if the objective is complete."
            )
        elif not node.allow_mutation and any(
            "read-only node changed workspace state" in str(item.get("error") or "")
            for item in failure_history
            if isinstance(item, dict)
        ):
            lines.append(
                "Retry guidance: this role is read-only. Do not use output redirection, "
                "temporary workspace files, or mutating shell commands; compare through "
                "stdout, pipes, process substitution, read_file, or direct test expressions."
            )
        elif changed_state_before:
            lines.append(
                "Retry guidance: previous attempts changed the workspace. Continue from "
                "the current files, fix or extend the existing changes, and run concrete "
                "verification after the final mutation."
            )
            if any(
                "environment" in str(item.get("error") or "").lower()
                for item in failure_history
                if isinstance(item, dict)
            ):
                lines.append(
                    "Before the next mutation, run a read-only environment or workspace "
                    "check in this attempt and use that result to choose the command path."
                )
        elif node.allow_mutation:
            lines.append(
                "Retry guidance: previous attempts did not produce a workspace change. "
                "Reuse the context already gathered, avoid repeating inspection-only "
                "steps, and make a concrete change toward the Goal unless the evidence "
                "proves no change is needed."
            )
    return "\n".join(lines)


def _has_verified_missing_status_retry(node: WorkflowNode) -> bool:
    for item in list(node.metadata.get("failure_history") or []):
        if str(item.get("error") or "") != "objective status is missing":
            continue
        activity_summary = item.get("activity_summary")
        if isinstance(activity_summary, dict) and activity_summary.get("verify_work_passed"):
            return True
    return False


def _status_only_retry_prompt(run: WorkflowRun, node: WorkflowNode) -> str:
    context = _dependency_context(run, node)
    sections = [
        f"Workflow: {run.title}",
        f"Workflow id: {run.id}",
        f"Node: {node.id} - {node.title}",
        f"Goal:\n{run.goal}",
        "Previous evidence already exists. Do not call tools or redo implementation.",
        "Decide from the evidence whether the objective is complete.",
        'Return a concise final assistant response ending with {"status":"pass"} if '
        'complete, otherwise {"status":"retry"}.',
    ]
    if context:
        sections.append(context)
    return "\n\n".join(sections)


def _node_prompt(run: WorkflowRun, node: WorkflowNode) -> str:
    context = _dependency_context(run, node)
    sections = [
        f"Workflow: {run.title}",
        f"Workflow id: {run.id}",
        f"Node: {node.id} - {node.title}",
        f"Role: {node.role or node.kind}",
        f"Goal:\n{run.goal}",
        (
            "Goal authority: The Goal section above is authoritative. Do not infer exact "
            "values from the workflow title or previous node text if they conflict with "
            "the Goal."
        ),
        "Expected evidence:\n" + _evidence_requirements_text(node),
        _tool_catalog_text(node),
    ]
    if context:
        sections.append(context)
    sections.append(f"Node instruction:\n{node.prompt}")
    return "\n\n".join(sections)


def _build_tools_for_node(node: WorkflowNode):
    policy = _policy_for_node(node)

    def _build_role_tools(
        tool_cwd: Path,
        *,
        config: Any = None,
        include: set[str] | None = None,
    ):
        role_include = policy.tool_include
        if role_include is None:
            final_include = include
        elif include is None:
            final_include = set(role_include)
        else:
            final_include = set(role_include) & set(include)
        registry = _build_tools(
            tool_cwd,
            config=config,
            include=final_include,
            extras={"clean_shell_env": True, "shell_pipefail": True},
        )
        if not policy.allow_mutation and registry.has("shell"):
            shell = registry.get("shell")
            registry.unregister("shell")
            registry.register(cast(Tool, _ReadOnlyShellTool(shell)))
        return registry

    return _build_role_tools


async def _load_node_activity(*, cwd: Path, session_id: str) -> list[ActivityEvent]:
    storage = build_storage(db=_workspace_db_path(cwd), in_memory=False, cwd=cwd)
    try:
        return await storage.list_activity(session_id=session_id, limit=1000)  # type: ignore[attr-defined]
    finally:
        if isinstance(storage, SQLiteStorage):
            await storage.close()


async def _run_workflow_node(
    *,
    cwd: Path,
    run: WorkflowRun,
    node: WorkflowNode,
    provider: str | None,
    model: str | None,
    max_steps: int,
    yes: bool,
    candidate_index: int = 0,
) -> str:
    from harness.cli.__main__ import _DEFAULT_SYSTEM_PROMPT, _build_agent

    cfg = _load_cli_config(None)
    selected_provider = provider or cfg.default_provider or "ollama"
    chain = _resolve_chain(failover_flag=None, provider_flag=selected_provider, config=cfg)
    selected_model = model or cfg.default_model or _default_model(selected_provider)
    session_id = _node_session_id(run, node, candidate_index=candidate_index)
    policy = _policy_for_node(node)
    node_max_steps = int(node.metadata.get("max_steps") or max_steps)
    require_tools = bool(node.metadata.get("require_tools", policy.require_tools))
    prompt = _node_prompt(run, node)
    system_prompt = _workflow_system_prompt(_DEFAULT_SYSTEM_PROMPT, node)

    if (policy.tool_include == set() or node.metadata.get("direct_text")) and not require_tools:
        return await _stream_workflow_text_node(
            provider=chain[0],
            model=selected_model,
            config=cfg,
            system_prompt=system_prompt,
            prompt=prompt,
        )
    if _has_verified_missing_status_retry(node):
        return await _stream_workflow_text_node(
            provider=chain[0],
            model=selected_model,
            config=cfg,
            system_prompt=system_prompt,
            prompt=_status_only_retry_prompt(run, node),
        )

    def _build_workflow_agent(**kwargs: Any) -> Any:
        return _build_agent(**kwargs, memory_tools_enabled=False)

    return (
        await _run_once_impl(
            prompt=prompt,
            model=selected_model,
            chain=chain,
            base_url=None,
            cwd=cwd,
            max_steps=node_max_steps,
            max_output_tokens=None,
            session_id=session_id,
            task_ref=None,
            db=_workspace_db_path(cwd),
            in_memory=False,
            yes=yes,
            inbox=False,
            verify="none",
            verify_command=None,
            critic=None,
            require_tools=require_tools,
            goal=False,
            max_context_tokens=None,
            predict=True,
            auto_compact=False,
            max_repair=2,
            profile="bare",
            domain="coding",
            phases=None,
            loop_detect=policy.allow_mutation,
            contracts=True,
            tips=True,
            include_workspace_context=True,
            silent=True,
            config=cfg,
            build_storage=build_storage,
            resolve_task_attachment=_noop_task_attachment,
            resolve_runtime_strategy=_resolve_runtime_strategy,
            build_verifier=_build_verifier,
            build_critic=_build_critic,
            build_adapter=_build_adapter,
            build_tools=_build_tools_for_node(node),
            build_agent=_build_workflow_agent,
            print_defense_ledger=_print_defense_ledger,
            render=lambda _event: None,
            default_system_prompt=system_prompt,
            console=console,
        )
        or ""
    )


async def _run_dynamic_planner(
    *,
    cwd: Path,
    goal: str,
    provider: str | None,
    model: str | None,
    max_steps: int,
    yes: bool,
    max_nodes: int,
) -> str:
    planner_node = WorkflowNode(
        id="dynamic-plan",
        title="Generate workflow DAG",
        kind="plan",
        role="planner",
        prompt="Generate a defended workflow DAG.",
    )
    planner_run = WorkflowRun(
        id=f"planner-{int(time.time())}",
        title="Dynamic planner",
        goal=goal,
        nodes=(planner_node,),
    )
    schema = {
        "title": "short workflow title",
        "nodes": [
            {
                "id": "stable-id",
                "title": "Node title",
                "kind": "plan|research|work|verify|review|refute|merge",
                "role": "planner|researcher|implementer|verifier|reviewer|refuter|merger",
                "prompt": "node-specific instruction",
                "depends_on": ["other-node-id"],
                "allow_mutation": False,
                "expected_evidence": [{"kind": "result_nonempty"}],
                "max_attempts": 1,
            }
        ],
    }
    planner_prompt = (
        "Create a JSON workflow plan for the goal below. Return only valid JSON. "
        "Use at most "
        f"{max_nodes} nodes. Prefer defended roles: research, work, verify, review, refute, merge. "
        "Every node must have a clear evidence expectation and dependency list. "
        "Use mutation only for implementation nodes.\n\n"
        f"JSON shape:\n{schema}\n\n"
        f"Goal:\n{goal}"
    )
    planner_node = replace(planner_node, prompt=planner_prompt)
    planner_run = replace(planner_run, nodes=(planner_node,))
    return await _run_workflow_node(
        cwd=cwd,
        run=planner_run,
        node=planner_node,
        provider=provider,
        model=model,
        max_steps=min(max_steps, 8),
        yes=yes,
    )


async def create_planned_workflow(
    *,
    cwd: Path,
    workflow_id: str,
    title: str,
    goal: str,
    planner: str,
    provider: str | None,
    model: str | None,
    max_steps: int,
    yes: bool,
    max_nodes: int,
) -> WorkflowRun:
    if planner == "static":
        return create_default_workflow(workflow_id=workflow_id, title=title, goal=goal)
    if planner != "dynamic":
        raise typer.BadParameter("--planner must be static or dynamic")
    try:
        raw_plan = await _run_dynamic_planner(
            cwd=cwd,
            goal=goal,
            provider=provider,
            model=model,
            max_steps=max_steps,
            yes=yes,
            max_nodes=max_nodes,
        )
        plan = extract_json_object(raw_plan)
        run = create_workflow_from_plan_spec(
            workflow_id=workflow_id,
            title=title,
            goal=goal,
            plan=plan,
            max_nodes=max_nodes,
        )
        return replace(
            run,
            metadata={
                **run.metadata,
                "planner": "dynamic",
                "planner_raw": raw_plan[:4000],
            },
        )
    except Exception as exc:
        fallback = create_default_workflow(workflow_id=workflow_id, title=title, goal=goal)
        return replace(
            fallback,
            metadata={
                **fallback.metadata,
                "planner": "static-fallback",
                "planner_fallback_reason": str(exc),
            },
        )


async def _run_candidate(
    *,
    cwd: Path,
    run: WorkflowRun,
    node: WorkflowNode,
    provider: str | None,
    model: str | None,
    max_steps: int,
    yes: bool,
    candidate_index: int,
) -> NodeCandidate:
    session_id = _node_session_id(run, node, candidate_index=candidate_index)
    node_task: asyncio.Task[str] | None = None
    try:
        node_task = asyncio.create_task(
            _run_workflow_node(
                cwd=cwd,
                run=run,
                node=node,
                provider=provider,
                model=model,
                max_steps=max_steps,
                yes=yes,
                candidate_index=candidate_index,
            )
        )
        timeout_seconds = _metadata_seconds(node, "completion_timeout_seconds")
        idle_timeout_seconds = _metadata_seconds(node, "idle_timeout_seconds")
        started_at = time.monotonic()
        last_activity_count = -1
        last_progress_at = started_at
        while True:
            now = time.monotonic()
            wait_timeout = 1.0
            if timeout_seconds > 0:
                wait_timeout = min(
                    wait_timeout,
                    max(0.0, timeout_seconds - (now - started_at)),
                )
            if idle_timeout_seconds > 0:
                wait_timeout = min(
                    wait_timeout,
                    max(0.0, idle_timeout_seconds - (now - last_progress_at)),
                )
            done, _pending = await asyncio.wait(
                {node_task},
                timeout=wait_timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if node_task in done:
                result = node_task.result()
                break
            now = time.monotonic()
            if timeout_seconds > 0 and now - started_at >= timeout_seconds:
                node_task.cancel()
                await asyncio.gather(node_task, return_exceptions=True)
                raise TimeoutError("node timed out before producing final text")
            if idle_timeout_seconds <= 0:
                continue
            activity = await _load_node_activity(cwd=cwd, session_id=session_id)
            if len(activity) != last_activity_count:
                last_activity_count = len(activity)
                last_progress_at = now
                continue
            if now - last_progress_at < idle_timeout_seconds:
                continue
            node_task.cancel()
            await asyncio.gather(node_task, return_exceptions=True)
            activity = await _load_node_activity(cwd=cwd, session_id=session_id)
            fallback = _activity_fallback_result(node, activity)
            if fallback:
                return NodeCandidate(
                    index=candidate_index,
                    result=fallback,
                    activity=activity,
                    session_id=session_id,
                )
            return NodeCandidate(
                index=candidate_index,
                result="",
                activity=activity,
                session_id=session_id,
                error=(
                    "node made no progress for "
                    f"{idle_timeout_seconds:g}s before producing final text"
                ),
            )
        activity = await _load_node_activity(cwd=cwd, session_id=session_id)
        return NodeCandidate(
            index=candidate_index,
            result=result,
            activity=activity,
            session_id=session_id,
        )
    except TimeoutError as exc:
        activity = await _load_node_activity(cwd=cwd, session_id=session_id)
        materialized = await _materialize_exact_file_requests(
            cwd=cwd,
            run=run,
            node=node,
            session_id=session_id,
            activity=activity,
        )
        if materialized:
            activity = [*activity, *materialized]
        fallback = _activity_fallback_result(node, activity) or _timeout_fallback_result(node, run)
        if fallback:
            return NodeCandidate(
                index=candidate_index,
                result=fallback,
                activity=activity,
                session_id=session_id,
            )
        return NodeCandidate(
            index=candidate_index,
            result="",
            activity=activity,
            session_id=session_id,
            error=str(exc) or "node timed out before producing final text",
        )
    except (typer.Exit, SystemExit) as exc:
        activity = await _load_node_activity(cwd=cwd, session_id=session_id)
        materialized = await _materialize_exact_file_requests(
            cwd=cwd,
            run=run,
            node=node,
            session_id=session_id,
            activity=activity,
        )
        if materialized:
            activity = [*activity, *materialized]
        fallback = _activity_fallback_result(node, activity) or _timeout_fallback_result(node, run)
        if fallback:
            return NodeCandidate(
                index=candidate_index,
                result=fallback,
                activity=activity,
                session_id=session_id,
            )
        code = getattr(exc, "exit_code", getattr(exc, "code", ""))
        detail = _activity_error_message(activity)
        error = detail or f"agent run exited before final answer (exit code {code})"
        return NodeCandidate(
            index=candidate_index,
            result="",
            activity=activity,
            session_id=session_id,
            error=error,
        )
    except BaseException as exc:
        activity = await _load_node_activity(cwd=cwd, session_id=session_id)
        materialized = await _materialize_exact_file_requests(
            cwd=cwd,
            run=run,
            node=node,
            session_id=session_id,
            activity=activity,
        )
        if materialized:
            activity = [*activity, *materialized]
        fallback = _activity_fallback_result(node, activity) or _timeout_fallback_result(node, run)
        if fallback:
            return NodeCandidate(
                index=candidate_index,
                result=fallback,
                activity=activity,
                session_id=session_id,
            )
        return NodeCandidate(
            index=candidate_index,
            result="",
            activity=activity,
            session_id=session_id,
            error=_activity_error_message(activity) or str(exc),
        )
    finally:
        if node_task is not None and not node_task.done():
            node_task.cancel()
            await asyncio.gather(node_task, return_exceptions=True)


def _normalized_candidate_text(value: str) -> str:
    return " ".join(value.lower().split())


def _choose_candidate(
    candidates: list[NodeCandidate],
    *,
    consensus_threshold: int,
) -> tuple[NodeCandidate, dict[str, Any]]:
    successes = [candidate for candidate in candidates if not candidate.error]
    if not successes:
        errors = "; ".join(candidate.error for candidate in candidates if candidate.error)
        raise RuntimeError(errors or "all node candidates failed")
    counts: dict[str, int] = {}
    for candidate in successes:
        key = _normalized_candidate_text(candidate.result)
        counts[key] = counts.get(key, 0) + 1
    best_key = max(counts.items(), key=lambda item: item[1])[0]
    best_count = counts[best_key]
    selected = next(
        candidate
        for candidate in successes
        if _normalized_candidate_text(candidate.result) == best_key
    )
    if best_count < max(1, consensus_threshold):
        selected = max(successes, key=lambda candidate: len(candidate.result))
    return selected, {
        "candidate_count": len(candidates),
        "successful_candidates": len(successes),
        "best_agreement": best_count,
        "threshold": consensus_threshold,
        "selected_index": selected.index,
        "met_threshold": best_count >= max(1, consensus_threshold),
    }


def _node_candidate_count(node: WorkflowNode, subagents: int) -> int:
    if node.allow_mutation:
        return 1
    if node.kind not in {"research", "work", "review"}:
        return 1
    return max(1, subagents)


def _node_dependencies_completed(run: WorkflowRun, node: WorkflowNode) -> bool:
    by_id = node_by_id(run)
    return all(
        by_id.get(dep_id) is not None and by_id[dep_id].status == "completed"
        for dep_id in node.depends_on
    )


def _deterministic_ready_node(
    *,
    node: WorkflowNode,
    run: WorkflowRun,
    started_at: float,
) -> WorkflowNode | None:
    if node.status != "pending" or not _node_dependencies_completed(run, node):
        return None
    if node.kind != "merge":
        return None
    result = _deterministic_node_result(node=node, run=run)
    if not result.strip():
        return None
    evidence_ok, evidence_results = evaluate_node_evidence(
        node=node,
        result=result,
        activity=[],
        run=run,
    )
    if not evidence_ok:
        return None
    return replace(
        node,
        status="completed",
        result=result,
        error="",
        session_id="",
        updated_at=_utcnow_text(),
        finished_at=_utcnow_text(),
        metadata={
            **node.metadata,
            "activity_summary": summarize_activity([]),
            "failed_tool_summaries": [],
            "evidence_results": [item.to_dict() for item in evidence_results],
            "consensus": {
                "candidate_count": 0,
                "successful_candidates": 0,
                "best_agreement": 0,
                "threshold": 0,
                "selected_index": 0,
                "met_threshold": True,
                "deterministic_dependency": True,
            },
            "runtime_seconds": round(time.monotonic() - started_at, 3),
            "subagent_results": [],
        },
    )


def _complete_ready_deterministic_nodes(
    *,
    run: WorkflowRun,
    started_at: float,
) -> tuple[WorkflowRun, list[WorkflowNode]]:
    current = run
    completed: list[WorkflowNode] = []
    while True:
        progressed = False
        for node in current.nodes:
            finished = _deterministic_ready_node(
                node=node,
                run=current,
                started_at=started_at,
            )
            if finished is None:
                continue
            current = update_node(current, finished)
            completed.append(finished)
            progressed = True
        if not progressed:
            return current, completed


async def _execute_node(
    *,
    cwd: Path,
    run: WorkflowRun,
    node: WorkflowNode,
    provider: str | None,
    model: str | None,
    max_steps: int,
    yes: bool,
    subagents: int,
    consensus_threshold: int,
) -> WorkflowNode:
    node_started_at = time.monotonic()
    candidate_count = _node_candidate_count(node, subagents)
    candidates = await asyncio.gather(
        *(
            _run_candidate(
                cwd=cwd,
                run=run,
                node=node,
                provider=provider,
                model=model,
                max_steps=max_steps,
                yes=yes,
                candidate_index=index,
            )
            for index in range(candidate_count)
        )
    )
    selected, consensus = _choose_candidate(
        list(candidates),
        consensus_threshold=min(max(1, consensus_threshold), candidate_count),
    )
    selected = _candidate_with_deterministic_result_if_needed(
        node=node,
        run=run,
        candidate=selected,
    )
    selected = await _auto_verify_candidate_if_needed(
        cwd=cwd,
        run=run,
        node=node,
        candidate=selected,
    )
    selected = _candidate_with_verified_objective_status_if_needed(
        node=node,
        run=run,
        candidate=selected,
    )
    evidence_ok, evidence_results = evaluate_node_evidence(
        node=node,
        result=selected.result,
        activity=selected.activity,
        run=run,
    )
    metadata = {
        **node.metadata,
        "activity_summary": summarize_activity(selected.activity),
        "failed_tool_summaries": _failed_tool_summaries(selected.activity),
        "evidence_results": [item.to_dict() for item in evidence_results],
        "consensus": consensus,
        "runtime_seconds": round(time.monotonic() - node_started_at, 3),
        "subagent_results": [
            {
                "index": candidate.index,
                "session_id": candidate.session_id,
                "error": candidate.error,
                "result_preview": candidate.result[:800],
            }
            for candidate in candidates
        ],
    }
    if not evidence_ok:
        failed = [
            item.message
            for item in evidence_results
            if item.status == "failed" and item.requirement.required
        ]
        return replace(
            node,
            status="failed",
            result=selected.result,
            error="; ".join(failed),
            session_id=selected.session_id,
            updated_at=_utcnow_text(),
            finished_at=_utcnow_text(),
            metadata=metadata,
        )
    return replace(
        node,
        status="completed",
        result=selected.result,
        error="",
        session_id=selected.session_id,
        updated_at=_utcnow_text(),
        finished_at=_utcnow_text(),
        metadata=metadata,
    )


def _effective_max_attempts(node: WorkflowNode, max_node_attempts: int) -> int:
    return max(1, max(node.max_attempts, max_node_attempts))


def _failed_tool_summaries(activity: list[ActivityEvent]) -> list[str]:
    summaries: list[str] = []
    for event in activity:
        if event.kind != "tool_call.completed" or not event.data.get("is_error"):
            continue
        name = str(event.data.get("name") or "tool")
        arguments = event.data.get("arguments")
        if isinstance(arguments, dict):
            command = str(
                arguments.get("command") or arguments.get("path") or arguments.get("pattern") or ""
            ).strip()
        else:
            command = ""
        preview = " ".join(str(event.data.get("content_preview") or "").split())
        if len(preview) > 500:
            preview = preview[:499].rstrip() + "..."
        if command:
            summaries.append(f"{name}({command}): {preview}")
        else:
            summaries.append(f"{name}: {preview}")
    return summaries[-5:]


def _retryable_pending_node(node: WorkflowNode) -> WorkflowNode:
    history = list(node.metadata.get("failure_history") or [])
    history.append(
        {
            "attempt": node.attempts,
            "error": node.error,
            "result": node.result[:1200],
            "finished_at": node.finished_at,
            "activity_summary": node.metadata.get("activity_summary") or {},
            "failed_tools": node.metadata.get("failed_tool_summaries") or [],
            "evidence_results": node.metadata.get("evidence_results") or [],
        }
    )
    return replace(
        node,
        status="pending",
        result="",
        error="",
        session_id="",
        started_at="",
        finished_at="",
        updated_at=_utcnow_text(),
        metadata={**node.metadata, "failure_history": history[-5:]},
    )


def _mark_blocked_nodes(run: WorkflowRun) -> tuple[WorkflowRun, list[WorkflowNode]]:
    by_id = node_by_id(run)
    skipped: list[WorkflowNode] = []
    current = run
    for node in run.nodes:
        if node.status != "pending":
            continue
        blocked = [
            dep
            for dep in node.depends_on
            if by_id.get(dep) is not None and by_id[dep].status in {"failed", "skipped"}
        ]
        if not blocked:
            continue
        updated = replace(
            node,
            status="skipped",
            error=f"dependencies did not complete: {', '.join(blocked)}",
            updated_at=_utcnow_text(),
            finished_at=_utcnow_text(),
        )
        current = update_node(current, updated)
        by_id[updated.id] = updated
        skipped.append(updated)
    return current, skipped


def _budget_exceeded(
    *,
    run: WorkflowRun,
    started_at: float,
    max_runtime_seconds: int | None,
    max_workflow_tokens: int | None,
) -> str:
    if max_runtime_seconds is not None and time.monotonic() - started_at > max_runtime_seconds:
        return f"workflow exceeded runtime budget of {max_runtime_seconds}s"
    if max_workflow_tokens is not None:
        usage = workflow_usage(run)
        if usage["total_tokens"] > max_workflow_tokens:
            return f"workflow exceeded token budget of {max_workflow_tokens}"
    return ""


def _running_wait_timeout(*, started_at: float, max_runtime_seconds: int | None) -> float:
    if max_runtime_seconds is None:
        return 1.0
    remaining = max_runtime_seconds - (time.monotonic() - started_at)
    return max(0.0, min(1.0, remaining))


def _retry_runtime_budget_seconds(node: WorkflowNode) -> float:
    if str(node.error or "") == "objective status is missing":
        return 5.0
    timeout_raw = node.metadata.get("completion_timeout_seconds")
    try:
        timeout_seconds = float(timeout_raw) if timeout_raw is not None else 0.0
    except (TypeError, ValueError):
        timeout_seconds = 0.0
    if timeout_seconds <= 0:
        return 10.0
    return min(30.0, max(10.0, timeout_seconds * 0.25))


def _has_retry_runtime_budget(
    *,
    node: WorkflowNode,
    started_at: float,
    max_runtime_seconds: int | None,
) -> bool:
    if max_runtime_seconds is None:
        return True
    remaining = max_runtime_seconds - (time.monotonic() - started_at)
    return remaining >= _retry_runtime_budget_seconds(node)


def _fail_running_nodes(run: WorkflowRun, *, error: str) -> WorkflowRun:
    now = _utcnow_text()
    return replace(
        run,
        nodes=tuple(
            replace(
                node,
                status="failed",
                error=error,
                updated_at=now,
                finished_at=now,
            )
            if node.status == "running"
            else node
            for node in run.nodes
        ),
    )


async def _complete_running_nodes_from_budget_evidence(
    *, cwd: Path, run: WorkflowRun
) -> tuple[WorkflowRun, list[WorkflowNode]]:
    current = run
    completed: list[WorkflowNode] = []
    for node in run.nodes:
        if node.status != "running":
            continue
        if node.kind not in {"work", "verify", "refute", "merge"}:
            continue
        session_id = node.session_id or _node_session_id(current, node)
        activity = await _load_node_activity(cwd=cwd, session_id=session_id)
        materialized = await _materialize_exact_file_requests(
            cwd=cwd,
            run=current,
            node=node,
            session_id=session_id,
            activity=activity,
        )
        if materialized:
            activity = [*activity, *materialized]
        candidate = NodeCandidate(
            index=0,
            result=_activity_fallback_result(node, activity)
            or _budget_recovery_node_result(node=node, run=current),
            activity=activity,
            session_id=session_id,
        )
        candidate = _candidate_with_deterministic_result_if_needed(
            node=node,
            run=current,
            candidate=candidate,
        )
        candidate = await _auto_verify_candidate_if_needed(
            cwd=cwd,
            run=current,
            node=node,
            candidate=candidate,
        )
        candidate = _candidate_with_verified_objective_status_if_needed(
            node=node,
            run=current,
            candidate=candidate,
        )
        if not candidate.result.strip():
            continue
        evidence_ok, evidence_results = evaluate_node_evidence(
            node=node,
            result=candidate.result,
            activity=candidate.activity,
            run=current,
        )
        if not evidence_ok:
            continue
        finished = replace(
            node,
            status="completed",
            result=candidate.result,
            error="",
            updated_at=_utcnow_text(),
            finished_at=_utcnow_text(),
            metadata={
                **node.metadata,
                "activity_summary": summarize_activity(candidate.activity),
                "evidence_results": [item.to_dict() for item in evidence_results],
                "consensus": {
                    "candidate_count": 1,
                    "successful_candidates": 1,
                    "best_agreement": 1,
                    "threshold": 1,
                    "selected_index": 0,
                    "met_threshold": True,
                    "recovered_from_budget_evidence": True,
                },
                "runtime_seconds": 0.0,
                "subagent_results": [
                    {
                        "index": candidate.index,
                        "session_id": candidate.session_id,
                        "error": candidate.error,
                        "result_preview": candidate.result[:1000],
                    }
                ],
            },
        )
        current = update_node(current, finished)
        completed.append(finished)
    return current, completed


def _recover_expired_running_nodes(run: WorkflowRun) -> tuple[WorkflowRun, list[WorkflowNode]]:
    recovered: list[WorkflowNode] = []
    current = run
    now = time.time()
    for node in run.nodes:
        if node.status != "running":
            continue
        try:
            lease_expires_at = float(node.metadata.get("lease_expires_at") or 0)
        except (TypeError, ValueError):
            lease_expires_at = 0
        if lease_expires_at > now:
            continue
        recovered_node = replace(
            node,
            status="pending",
            session_id="",
            started_at="",
            updated_at=_utcnow_text(),
            metadata={
                **node.metadata,
                "recovered_from_stale_lease": True,
                "previous_lease_owner": node.metadata.get("lease_owner"),
            },
        )
        current = update_node(current, recovered_node)
        recovered.append(recovered_node)
    return current, recovered


def _start_node(
    run: WorkflowRun, node: WorkflowNode, *, runner_id: str
) -> tuple[WorkflowRun, WorkflowNode]:
    now = _utcnow_text()
    attempted = replace(node, attempts=node.attempts + 1)
    started = replace(
        attempted,
        status="running",
        session_id=_node_session_id(run, attempted),
        updated_at=now,
        started_at=node.started_at or now,
        metadata={
            **node.metadata,
            "lease_owner": runner_id,
            "lease_expires_at": time.time() + 900,
        },
    )
    return update_node(run, started), started


def _final_report_for_run(run: WorkflowRun) -> str:
    merge_nodes = [
        node
        for node in run.nodes
        if node.kind == "merge" and node.status == "completed" and node.result
    ]
    if merge_nodes:
        return merge_nodes[-1].result
    terminals = set(terminal_node_ids(run))
    terminal_results = [
        node.result
        for node in run.nodes
        if node.id in terminals and node.status == "completed" and node.result
    ]
    if terminal_results:
        return "\n\n".join(terminal_results)
    return "\n\n".join(
        node.result for node in run.nodes if node.status == "completed" and node.result
    )


def _failure_report_for_run(run: WorkflowRun) -> str:
    failed = [node for node in run.nodes if node.status in {"failed", "skipped", "running"}]
    retry_decisions = _retry_decision_nodes(run)
    if not failed and not retry_decisions:
        return _final_report_for_run(run)
    lines = ["Workflow failed before a verified final report was produced."]
    for node in failed:
        reason = node.error or node.status
        lines.append(f"- {node.id}: {reason}")
    for node in retry_decisions:
        lines.append(f"- {node.id}: requested another review round")
    return "\n".join(lines)


def _retry_decision_nodes(run: WorkflowRun) -> list[WorkflowNode]:
    return [
        node
        for node in run.nodes
        if node.kind in {"review", "refute"}
        and node.status == "completed"
        and workflow_requests_retry(node.result)
    ]


def _reset_after_review_retry(run: WorkflowRun, *, feedback: str) -> WorkflowRun:
    work_roots = {node.id for node in run.nodes if node.kind == "work"}
    if not work_roots:
        work_roots = {node.id for node in run.nodes if node.kind not in {"plan", "research"}}
    affected = dependent_node_ids(run, work_roots)
    updated = reset_nodes_for_retry(run, node_ids=affected, feedback=feedback)
    rounds = int(updated.metadata.get("review_rounds") or 0) + 1
    return replace(
        updated,
        metadata={**updated.metadata, "review_rounds": rounds, "review_feedback": feedback},
        updated_at=_utcnow_text(),
    )


async def run_workflow(
    *,
    cwd: Path,
    store: WorkflowStore,
    workflow_id: str,
    provider: str | None = None,
    model: str | None = None,
    max_steps: int = 20,
    yes: bool = False,
    concurrency: int = 1,
    max_node_attempts: int = 2,
    max_review_rounds: int = 1,
    subagents: int = 1,
    consensus_threshold: int = 1,
    max_runtime_seconds: int | None = None,
    max_workflow_tokens: int | None = None,
) -> WorkflowRun:
    run = store.load_run(workflow_id)
    if run.status == "cancelled":
        return run
    runner_id = f"{run.id}:runner:{int(time.time() * 1000)}"
    run, recovered = _recover_expired_running_nodes(run)
    run = replace(
        run,
        status="running",
        updated_at=_utcnow_text(),
        metadata={
            **run.metadata,
            "execution": {
                "concurrency": max(1, concurrency),
                "max_node_attempts": max_node_attempts,
                "max_review_rounds": max_review_rounds,
                "subagents": max(1, subagents),
                "consensus_threshold": max(1, consensus_threshold),
                "max_runtime_seconds": max_runtime_seconds,
                "max_workflow_tokens": max_workflow_tokens,
            },
        },
    )
    store.save_run(run)
    store.append_event(run.id, kind="workflow.running", message="Workflow started.")
    for node in recovered:
        store.append_event(
            run.id,
            kind="node.lease_recovered",
            node_id=node.id,
            message=f"Recovered stale lease for node {node.id}.",
        )

    started_at = time.monotonic()
    running: dict[asyncio.Task[WorkflowNode], WorkflowNode] = {}
    concurrency = max(1, concurrency)
    try:
        while True:
            budget_error = _budget_exceeded(
                run=run,
                started_at=started_at,
                max_runtime_seconds=max_runtime_seconds,
                max_workflow_tokens=max_workflow_tokens,
            )
            if budget_error:
                run, budget_completed = await _complete_running_nodes_from_budget_evidence(
                    cwd=cwd,
                    run=run,
                )
                if budget_completed:
                    completed_ids = {node.id for node in budget_completed}
                    for task, started_node in list(running.items()):
                        if started_node.id not in completed_ids:
                            continue
                        if not task.done():
                            task.cancel()
                        running.pop(task, None)
                    store.save_run(run)
                    for node in budget_completed:
                        store.append_event(
                            run.id,
                            kind="node.completed",
                            node_id=node.id,
                            message=(
                                f"Completed node {node.id}: {node.title} "
                                "from verified workflow evidence"
                            ),
                            data=node.metadata.get("activity_summary") or {},
                        )
                    run, deterministic_completed = _complete_ready_deterministic_nodes(
                        run=run,
                        started_at=started_at,
                    )
                    if deterministic_completed:
                        store.save_run(run)
                        for node in deterministic_completed:
                            store.append_event(
                                run.id,
                                kind="node.completed",
                                node_id=node.id,
                                message=(
                                    f"Completed node {node.id}: {node.title} "
                                    "from deterministic workflow evidence"
                                ),
                                data=node.metadata.get("activity_summary") or {},
                            )
                    if not any(node.status in {"pending", "running"} for node in run.nodes):
                        break

                run = _fail_running_nodes(run, error=budget_error)
                final_report = _failure_report_for_run(run) or (
                    "Workflow failed before a verified final report was produced.\n"
                    f"- workflow: {budget_error}"
                )
                run = replace(
                    run,
                    status="failed",
                    final_report=final_report,
                    updated_at=_utcnow_text(),
                    completed_at=_utcnow_text(),
                    metadata={
                        **run.metadata,
                        "usage": workflow_usage(run),
                        "runtime_seconds": round(time.monotonic() - started_at, 3),
                    },
                )
                store.save_run(run)
                report_path = store.write_report(run)
                store.append_event(
                    run.id,
                    kind="workflow.budget_exceeded",
                    message=budget_error,
                    data={"report_path": str(report_path), "usage": workflow_usage(run)},
                )
                store.append_event(
                    run.id,
                    kind="workflow.failed",
                    message="Workflow failed.",
                    data={"report_path": str(report_path), "usage": workflow_usage(run)},
                )
                return run

            run, skipped = _mark_blocked_nodes(run)
            if skipped:
                store.save_run(run)
                for node in skipped:
                    store.append_event(
                        run.id,
                        kind="node.skipped",
                        node_id=node.id,
                        message=f"Skipped node {node.id}: {node.error}",
                    )

            run, deterministic_completed = _complete_ready_deterministic_nodes(
                run=run,
                started_at=started_at,
            )
            if deterministic_completed:
                store.save_run(run)
                for node in deterministic_completed:
                    store.append_event(
                        run.id,
                        kind="node.completed",
                        node_id=node.id,
                        message=(
                            f"Completed node {node.id}: {node.title} "
                            "from deterministic workflow evidence"
                        ),
                        data=node.metadata.get("activity_summary") or {},
                    )
                continue

            while len(running) < concurrency:
                ready = [
                    node
                    for node in ready_pending_nodes(run)
                    if node.id not in {item.id for item in running.values()}
                ]
                if not ready:
                    break
                next_node = ready[0]
                store.append_event(
                    run.id,
                    kind="node.queued",
                    node_id=next_node.id,
                    message=f"Queued node {next_node.id}: {next_node.title}",
                )
                run, started = _start_node(run, next_node, runner_id=runner_id)
                store.save_run(run)
                store.append_event(
                    run.id,
                    kind="node.running",
                    node_id=started.id,
                    message=f"Started node {started.id}: {started.title}",
                    data={"attempt": started.attempts},
                )
                task = asyncio.create_task(
                    _execute_node(
                        cwd=cwd,
                        run=run,
                        node=started,
                        provider=provider,
                        model=model,
                        max_steps=max_steps,
                        yes=yes,
                        subagents=subagents,
                        consensus_threshold=consensus_threshold,
                    )
                )
                running[task] = started

            if running:
                wait_timeout = _running_wait_timeout(
                    started_at=started_at,
                    max_runtime_seconds=max_runtime_seconds,
                )
                done, _pending = await asyncio.wait(
                    running.keys(),
                    timeout=wait_timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if not done:
                    continue
                for task in done:
                    started_node = running.pop(task)
                    try:
                        finished = task.result()
                    except BaseException as exc:
                        finished = replace(
                            started_node,
                            status="failed",
                            error=str(exc),
                            updated_at=_utcnow_text(),
                            finished_at=_utcnow_text(),
                        )

                    if (
                        finished.status == "failed"
                        and finished.attempts < _effective_max_attempts(finished, max_node_attempts)
                        and _has_retry_runtime_budget(
                            node=finished,
                            started_at=started_at,
                            max_runtime_seconds=max_runtime_seconds,
                        )
                    ):
                        retry = _retryable_pending_node(finished)
                        run = update_node(run, retry)
                        store.save_run(run)
                        store.append_event(
                            run.id,
                            kind="node.retrying",
                            node_id=retry.id,
                            message=(
                                f"Retrying node {retry.id} after attempt "
                                f"{finished.attempts}: {finished.error}"
                            ),
                            data={"attempt": finished.attempts, "error": finished.error},
                        )
                        continue

                    retry_requested = (
                        finished.status == "completed"
                        and finished.kind == "refute"
                        and workflow_requests_retry(finished.result)
                    )
                    if (
                        retry_requested
                        and int(run.metadata.get("review_rounds") or 0) >= max_review_rounds
                    ):
                        finished = replace(
                            finished,
                            status="failed",
                            error="review requested retry but max review rounds are exhausted",
                            updated_at=_utcnow_text(),
                            finished_at=_utcnow_text(),
                        )

                    run = update_node(run, finished)
                    store.save_run(run)
                    if finished.status == "completed":
                        store.append_event(
                            run.id,
                            kind="node.completed",
                            node_id=finished.id,
                            message=f"Completed node {finished.id}: {finished.title}",
                            data=finished.metadata.get("activity_summary") or {},
                        )
                        if (
                            retry_requested
                            and int(run.metadata.get("review_rounds") or 0) < max_review_rounds
                        ):
                            run = _reset_after_review_retry(run, feedback=finished.result)
                            store.save_run(run)
                            store.append_event(
                                run.id,
                                kind="workflow.review_retry",
                                node_id=finished.id,
                                message="Review requested another worker pass.",
                                data={"review_rounds": run.metadata.get("review_rounds")},
                            )
                    else:
                        event_kind = (
                            "node.evidence_failed"
                            if finished.metadata.get("evidence_results")
                            else "node.failed"
                        )
                        store.append_event(
                            run.id,
                            kind=event_kind,
                            node_id=finished.id,
                            message=f"Node {finished.id} failed: {finished.error}",
                            data={"error": finished.error},
                        )
                continue

            pending_nodes = [node for node in run.nodes if node.status == "pending"]
            running_nodes = [node for node in run.nodes if node.status == "running"]
            if pending_nodes or running_nodes:
                if running_nodes and not pending_nodes:
                    store.append_event(
                        run.id,
                        kind="workflow.locked",
                        message="Workflow has active leased nodes owned by another runner.",
                        data={"running_nodes": [node.id for node in running_nodes]},
                    )
                    store.save_run(run)
                    return run
                run = replace(run, status="failed", updated_at=_utcnow_text())
                store.save_run(run)
                store.append_event(
                    run.id,
                    kind="workflow.deadlocked",
                    message="Workflow has pending nodes but no runnable dependencies.",
                )
                return run
            break

        failed_or_skipped = [
            node for node in run.nodes if node.status in {"failed", "skipped", "running"}
        ]
        retry_decisions = _retry_decision_nodes(run)
        final_status = "failed" if failed_or_skipped or retry_decisions else "completed"
        final_report = (
            _failure_report_for_run(run) if final_status == "failed" else _final_report_for_run(run)
        )
        run = replace(
            run,
            status=final_status,
            final_report=final_report,
            updated_at=_utcnow_text(),
            completed_at=_utcnow_text(),
            metadata={
                **run.metadata,
                "usage": workflow_usage(run),
                "runtime_seconds": round(time.monotonic() - started_at, 3),
            },
        )
        store.save_run(run)
        report_path = store.write_report(run)
        store.append_event(
            run.id,
            kind=f"workflow.{final_status}",
            message=f"Workflow {final_status}.",
            data={"report_path": str(report_path), "usage": workflow_usage(run)},
        )
        return run
    finally:
        for task in running:
            if not task.done():
                task.cancel()


def _render_status(run: WorkflowRun) -> None:
    console.print(f"[bold]{run.title}[/bold]")
    console.print(f"id={run.id} status={run.status}")
    table = Table("node", "kind", "role", "status", "attempts", "summary")
    for node in run.nodes:
        summary = " ".join((node.error or node.result or "").split())
        if len(summary) > 90:
            summary = summary[:89].rstrip() + "..."
        table.add_row(
            node.id,
            node.kind,
            node.role or node.kind,
            node.status,
            str(node.attempts),
            summary,
        )
    console.print(table)
    if run.metadata.get("usage"):
        console.print(f"[dim]usage[/dim] {run.metadata['usage']}")


def _render_events(events: list[Any]) -> None:
    table = Table("time", "kind", "node", "message")
    for event in events:
        table.add_row(event.timestamp, event.kind, event.node_id, event.message)
    console.print(table)


@workflow_app.command("start")
def workflow_start_command(
    *,
    goal: str = typer.Option(..., "--goal", "-g", help="Workflow goal or task."),
    title: str = typer.Option("", "--title", "-t", help="Optional workflow title."),
    cwd: Path | None = typer.Option(None, "--cwd"),
    planner: str = typer.Option("static", "--planner", help="static or dynamic."),
    max_nodes: int = typer.Option(12, "--max-nodes", min=3),
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    max_steps: int = typer.Option(20, "--max-steps"),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    max_node_attempts: int = typer.Option(2, "--max-node-attempts", min=1),
    max_review_rounds: int = typer.Option(1, "--max-review-rounds", min=0),
    subagents: int = typer.Option(1, "--subagents", min=1),
    consensus_threshold: int = typer.Option(1, "--consensus-threshold", min=1),
    max_runtime_seconds: int | None = typer.Option(None, "--max-runtime-seconds"),
    max_workflow_tokens: int | None = typer.Option(None, "--max-workflow-tokens"),
    plan_only: bool = typer.Option(False, "--plan-only"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve tool calls."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Create and optionally run a defended dynamic workflow."""
    working_dir = (cwd or Path.cwd()).resolve()
    store = WorkflowStore(root=default_workflow_root(working_dir))
    workflow_title = title.strip() or goal.strip().splitlines()[0][:80]
    workflow_id = store.new_id(workflow_title)
    run = asyncio.run(
        create_planned_workflow(
            cwd=working_dir,
            workflow_id=workflow_id,
            title=workflow_title,
            goal=goal,
            planner=planner,
            provider=provider,
            model=model,
            max_steps=max_steps,
            yes=yes,
            max_nodes=max_nodes,
        )
    )
    store.add_run(run)
    store.append_event(
        run.id,
        kind="workflow.created",
        message="Workflow created.",
        data={"planner": run.metadata.get("planner")},
    )
    if not plan_only:
        run = asyncio.run(
            run_workflow(
                cwd=working_dir,
                store=store,
                workflow_id=run.id,
                provider=provider,
                model=model,
                max_steps=max_steps,
                yes=yes,
                concurrency=concurrency,
                max_node_attempts=max_node_attempts,
                max_review_rounds=max_review_rounds,
                subagents=subagents,
                consensus_threshold=consensus_threshold,
                max_runtime_seconds=max_runtime_seconds,
                max_workflow_tokens=max_workflow_tokens,
            )
        )
    if json_output:
        console.print_json(data=run.to_dict())
        return
    _render_status(run)


@workflow_app.command("resume")
def workflow_resume_command(
    workflow_id: str,
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    provider: str | None = typer.Option(None, "--provider"),
    model: str | None = typer.Option(None, "--model"),
    max_steps: int = typer.Option(20, "--max-steps"),
    concurrency: int = typer.Option(1, "--concurrency", min=1),
    max_node_attempts: int = typer.Option(2, "--max-node-attempts", min=1),
    max_review_rounds: int = typer.Option(1, "--max-review-rounds", min=0),
    subagents: int = typer.Option(1, "--subagents", min=1),
    consensus_threshold: int = typer.Option(1, "--consensus-threshold", min=1),
    max_runtime_seconds: int | None = typer.Option(None, "--max-runtime-seconds"),
    max_workflow_tokens: int | None = typer.Option(None, "--max-workflow-tokens"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve tool calls."),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    working_dir = (cwd or Path.cwd()).resolve()
    store = WorkflowStore(root=default_workflow_root(working_dir))
    run = asyncio.run(
        run_workflow(
            cwd=working_dir,
            store=store,
            workflow_id=workflow_id,
            provider=provider,
            model=model,
            max_steps=max_steps,
            yes=yes,
            concurrency=concurrency,
            max_node_attempts=max_node_attempts,
            max_review_rounds=max_review_rounds,
            subagents=subagents,
            consensus_threshold=consensus_threshold,
            max_runtime_seconds=max_runtime_seconds,
            max_workflow_tokens=max_workflow_tokens,
        )
    )
    if json_output:
        console.print_json(data=run.to_dict())
        return
    _render_status(run)


@workflow_app.command("status")
def workflow_status_command(
    workflow_id: str,
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    store = WorkflowStore(root=default_workflow_root((cwd or Path.cwd()).resolve()))
    run = store.load_run(workflow_id)
    if json_output:
        console.print_json(data=run.to_dict())
        return
    _render_status(run)


@workflow_app.command("list")
def workflow_list_command(
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    store = WorkflowStore(root=default_workflow_root((cwd or Path.cwd()).resolve()))
    runs = store.list_runs()
    if json_output:
        console.print_json(data=[run.to_dict() for run in runs])
        return
    if not runs:
        console.print("[dim]No workflows found.[/dim]")
        return
    table = Table("id", "status", "title", "updated")
    for run in runs:
        table.add_row(run.id, run.status, run.title, run.updated_at)
    console.print(table)


@workflow_app.command("events")
def workflow_events_command(
    workflow_id: str,
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    limit: int = typer.Option(50, "--limit", min=1),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    store = WorkflowStore(root=default_workflow_root((cwd or Path.cwd()).resolve()))
    events = store.list_events(workflow_id)[-limit:]
    if json_output:
        console.print_json(data=[event.to_dict() for event in events])
        return
    _render_events(events)


@workflow_app.command("graph")
def workflow_graph_command(
    workflow_id: str,
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    store = WorkflowStore(root=default_workflow_root((cwd or Path.cwd()).resolve()))
    run = store.load_run(workflow_id)
    graph = render_workflow_mermaid(run)
    if json_output:
        console.print_json(data={"id": run.id, "mermaid": graph})
        return
    console.print(graph.rstrip())


@workflow_app.command("report")
def workflow_report_command(
    workflow_id: str,
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    store = WorkflowStore(root=default_workflow_root((cwd or Path.cwd()).resolve()))
    run = store.load_run(workflow_id)
    report_path = store.run_dir(run.id) / "REPORT.md"
    if json_output:
        console.print_json(
            data={
                "id": run.id,
                "status": run.status,
                "report": run.final_report,
                "report_path": str(report_path) if report_path.exists() else "",
            }
        )
        return
    if report_path.exists():
        console.print(report_path.read_text(encoding="utf-8"))
        return
    console.print(run.final_report or "(no final report yet)")


@workflow_app.command("cancel")
def workflow_cancel_command(
    workflow_id: str,
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    store = WorkflowStore(root=default_workflow_root((cwd or Path.cwd()).resolve()))
    run = store.load_run(workflow_id)
    updated = replace(run, status="cancelled", updated_at=_utcnow_text())
    store.save_run(updated)
    store.append_event(updated.id, kind="workflow.cancelled", message="Workflow cancelled.")
    if json_output:
        console.print_json(data=updated.to_dict())
        return
    _render_status(updated)


@workflow_app.command("launch")
def workflow_launch_command(
    workflow_id: str,
    *,
    cwd: Path | None = typer.Option(None, "--cwd"),
    yes: bool = typer.Option(False, "--yes", "-y"),
) -> None:
    """Start a workflow runner in the background and return immediately."""
    working_dir = (cwd or Path.cwd()).resolve()
    uv = "uv"
    command = [
        uv,
        "run",
        "harness",
        "workflow",
        "resume",
        workflow_id,
        "--cwd",
        str(working_dir),
    ]
    if yes:
        command.append("--yes")
    log_dir = working_dir / ".harness" / "workflows" / workflow_id
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "runner.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(
            command,
            cwd=str(working_dir),
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    store = WorkflowStore(root=default_workflow_root(working_dir))
    store.append_event(
        workflow_id,
        kind="workflow.launch",
        message=f"Background runner started with pid {process.pid}.",
        data={"pid": process.pid, "log_path": str(log_path)},
    )
    console.print(f"Started workflow runner pid={process.pid} log={log_path}")


__all__ = [
    "create_planned_workflow",
    "run_workflow",
    "workflow_app",
]
