from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from harness.core.activity import ActivityEvent
from harness.core.tools_verification import (
    _command_exits_before_trailing_command,
    _failure_branch_masks_exit_status,
    _output_reports_failure,
)
from harness.core.verification_structural import tool_event_changes_state

WorkflowStatus = Literal["pending", "running", "completed", "failed", "cancelled"]
WorkflowNodeStatus = Literal["pending", "running", "completed", "failed", "skipped"]
WorkflowNodeKind = Literal[
    "plan",
    "research",
    "work",
    "verify",
    "review",
    "refute",
    "merge",
]
EvidenceStatus = Literal["passed", "failed", "skipped"]

VALID_NODE_KINDS: tuple[str, ...] = (
    "plan",
    "research",
    "work",
    "verify",
    "review",
    "refute",
    "merge",
)
VALID_EVIDENCE_KINDS: tuple[str, ...] = (
    "result_nonempty",
    "tool_called",
    "no_failed_tools",
    "no_state_change",
    "planning_only",
    "claim_grounded",
    "files_changed",
    "environment_checked",
    "verify_work_passed",
    "verify_work_if_state_changed",
    "tests_passed",
    "objective_status",
    "review_decision",
    "review_passed",
    "dependency_completed",
)

_STATE_CHANGE_TOOL_NAMES = frozenset(
    {
        "write_file",
        "edit_file",
        "apply_patch",
        "write",
        "replace",
        "shell",
        "bash",
        "run_command",
        "execute",
    }
)
_VERIFY_WORK_MUTATION_RE = re.compile(
    r"(^|[;&|]\s*)(rm|mv|cp|mkdir|touch|chmod|chown|install|tee|patch|truncate)\b"
    r"|(^|[;&|]\s*)git\s+(apply|checkout|commit|merge|pull|push|rebase|reset|restore|switch)\b"
    r"|(^|[;&|]\s*)sed\s+-i\b"
    r"|(^|[;&|]\s*)perl\s+-p?i\b"
    r"|>{1,2}"
    r"|\b(?:write_text|write_bytes|unlink|rename)\s*\("
    r"|\bopen\s*\([^)]*['\"][wax+]['\"]",
    re.IGNORECASE,
)
_ENVIRONMENT_DISCOVERY_TOOL_NAMES = frozenset({"list_dir", "glob", "read_file"})
_ENVIRONMENT_SHELL_MUTATION_WORD_RE = re.compile(
    r"\b(?:install|uninstall|upgrade|update|add|remove|delete|create|write)\b",
    re.IGNORECASE,
)
_ENVIRONMENT_SETUP_WORD_RE = re.compile(r"\b(?:install|uninstall|upgrade|update)\b", re.I)
_ENVIRONMENT_MISSING_TEXT_RE = re.compile(
    r"\b(?:command not found|not found|not installed|missing|not available|"
    r"cannot find|could not find|unable to find|no such file)\b",
    re.IGNORECASE,
)
_TOOL_UNAVAILABLE_TEXT_RE = re.compile(
    r"\b(?:unknown tool|tool not available|tool unavailable|tool not permitted|"
    r"tool not allowed)\b",
    re.IGNORECASE,
)
_ENVIRONMENT_UNSUPPORTED_TEXT_RE = re.compile(
    r"\b(?:illegal option|invalid option|unrecognized option|unsupported option|"
    r"unknown option|bad option)\b",
    re.IGNORECASE,
)
_ENVIRONMENT_LOOKUP_ABSENT_TEXT_RE = re.compile(
    r"\b(?:not[-_ ]?found|not installed|no such command|missing|absent)\b",
    re.IGNORECASE,
)
_PLAN_FIRST_PERSON_EXECUTION_CLAIM_RE = re.compile(
    r"\b(?:i|we)\s+(?:have\s+|successfully\s+|already\s+)*"
    r"(?:created|verified|completed|executed|confirmed|satisfied|implemented|built|wrote|"
    r"written|added|updated)\b",
    re.IGNORECASE,
)
_PLAN_EXECUTION_CLAIM_RE = re.compile(
    r"\b(?:workflow|task|execution)\b.{0,100}\b"
    r"(?:was|were|has been|have been|is)\s+"
    r"(?:successfully\s+)?(?:completed|complete|successful|satisfied)\b"
    r"|\b(?:was|were|has been|have been)\s+"
    r"(?:successfully\s+)?"
    r"(?:created|verified|completed|executed|confirmed|satisfied|implemented|built|written|"
    r"added|updated)\b",
    re.IGNORECASE | re.S,
)
_PLAN_PROSPECTIVE_CONTEXT_RE = re.compile(
    r"\b(?:expected|expectation|will|would|should|must|needs?\s+to|can\s+be|could\s+be|"
    r"to\s+be|planned|success\s+criteria|acceptance\s+criteria)\b",
    re.IGNORECASE,
)
_PLAN_PROSPECTIVE_HEADER_RE = re.compile(
    r"^\s*(?:[#>*-]+\s*)?(?:\*\*)?\s*"
    r"(?:expected\s+outcomes?|expected\s+results?|success\s+criteria|acceptance\s+criteria|"
    r"planned\s+outcomes?|target\s+outcomes?)\b",
    re.IGNORECASE,
)
_PSEUDO_TOOL_RESULT_RE = re.compile(
    r"</?tool_code\b[^>]*>"
    r"|<\|tool_call\>.*?<tool_call\|>"
    r"|```+\s*tool_(?:response|call)\b"
    r"|^\s*(?:google_search|web_search|fetch_url)\s*\("
    r"|\[\s*(?:fetch_url|glob|google_search|list_dir|read_file|request_critique|shell|"
    r"verify_work|web_search|write_file)\s*\(",
    re.IGNORECASE | re.MULTILINE,
)
_ROLE_BY_KIND = {
    "plan": "planner",
    "research": "researcher",
    "work": "implementer",
    "verify": "verifier",
    "review": "reviewer",
    "refute": "refuter",
    "merge": "merger",
}


def default_workflow_root(cwd: Path | None = None) -> Path:
    return (cwd or Path.cwd()).resolve() / ".harness" / "workflows"


def _utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "workflow"


def normalize_node_id(value: str, *, fallback: str) -> str:
    raw = str(value or "").strip().lower()
    normalized = re.sub(r"[^a-z0-9_-]+", "-", raw).strip("-_")
    return normalized[:48] or fallback


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    kind: str
    name: str = ""
    description: str = ""
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "description": self.description,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: str | dict[str, Any]) -> EvidenceRequirement:
        if isinstance(data, str):
            return cls(kind=data.strip())
        return cls(
            kind=str(data.get("kind") or "").strip(),
            name=str(data.get("name") or "").strip(),
            description=str(data.get("description") or "").strip(),
            required=bool(data.get("required", True)),
        )


@dataclass(frozen=True, slots=True)
class EvidenceResult:
    requirement: EvidenceRequirement
    status: EvidenceStatus
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement": self.requirement.to_dict(),
            "status": self.status,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class WorkflowNode:
    id: str
    title: str
    kind: WorkflowNodeKind
    prompt: str
    depends_on: tuple[str, ...] = ()
    role: str = ""
    allow_mutation: bool = False
    expected_evidence: tuple[EvidenceRequirement, ...] = ()
    max_attempts: int = 1
    status: WorkflowNodeStatus = "pending"
    session_id: str = ""
    result: str = ""
    error: str = ""
    attempts: int = 0
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    started_at: str = ""
    finished_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "prompt": self.prompt,
            "depends_on": list(self.depends_on),
            "role": self.role or _ROLE_BY_KIND.get(self.kind, self.kind),
            "allow_mutation": self.allow_mutation,
            "expected_evidence": [item.to_dict() for item in self.expected_evidence],
            "max_attempts": self.max_attempts,
            "status": self.status,
            "session_id": self.session_id,
            "result": self.result,
            "error": self.error,
            "attempts": self.attempts,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowNode:
        kind = str(data.get("kind") or "work").strip()
        if kind not in VALID_NODE_KINDS:
            kind = "work"
        expected = data.get("expected_evidence") or data.get("evidence") or ()
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or "").strip(),
            kind=kind,  # type: ignore[arg-type]
            prompt=str(data.get("prompt") or "").strip(),
            depends_on=tuple(str(item).strip() for item in data.get("depends_on") or []),
            role=str(data.get("role") or _ROLE_BY_KIND.get(kind, kind)).strip(),
            allow_mutation=bool(data.get("allow_mutation", kind == "work")),
            expected_evidence=tuple(
                EvidenceRequirement.from_dict(item)
                for item in expected
                if isinstance(item, str | dict)
            ),
            max_attempts=max(1, int(data.get("max_attempts") or 1)),
            status=str(data.get("status") or "pending"),  # type: ignore[arg-type]
            session_id=str(data.get("session_id") or "").strip(),
            result=str(data.get("result") or ""),
            error=str(data.get("error") or ""),
            attempts=int(data.get("attempts") or 0),
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
            started_at=str(data.get("started_at") or ""),
            finished_at=str(data.get("finished_at") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True, slots=True)
class WorkflowEvent:
    id: str
    workflow_id: str
    kind: str
    message: str
    node_id: str = ""
    timestamp: str = field(default_factory=_utcnow)
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "workflow_id": self.workflow_id,
            "kind": self.kind,
            "message": self.message,
            "node_id": self.node_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowEvent:
        return cls(
            id=str(data["id"]),
            workflow_id=str(data.get("workflow_id") or "").strip(),
            kind=str(data.get("kind") or "").strip(),
            message=str(data.get("message") or "").strip(),
            node_id=str(data.get("node_id") or "").strip(),
            timestamp=str(data.get("timestamp") or _utcnow()),
            data=dict(data.get("data") or {}),
        )


@dataclass(frozen=True, slots=True)
class WorkflowRun:
    id: str
    title: str
    goal: str
    status: WorkflowStatus = "pending"
    nodes: tuple[WorkflowNode, ...] = ()
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    completed_at: str = ""
    final_report: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "goal": self.goal,
            "status": self.status,
            "nodes": [node.to_dict() for node in self.nodes],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "final_report": self.final_report,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowRun:
        return cls(
            id=str(data["id"]),
            title=str(data.get("title") or "").strip(),
            goal=str(data.get("goal") or "").strip(),
            status=str(data.get("status") or "pending"),  # type: ignore[arg-type]
            nodes=tuple(
                WorkflowNode.from_dict(item)
                for item in data.get("nodes") or []
                if isinstance(item, dict)
            ),
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
            completed_at=str(data.get("completed_at") or ""),
            final_report=str(data.get("final_report") or ""),
            metadata=dict(data.get("metadata") or {}),
        )


class WorkflowStore:
    def __init__(self, *, root: Path):
        self.root = root

    def ensure_layout(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def new_id(self, title: str) -> str:
        return f"workflow-{_slugify(title)[:32]}-{uuid4().hex[:8]}"

    def run_dir(self, workflow_id: str) -> Path:
        return self.root / workflow_id

    def add_run(self, run: WorkflowRun) -> Path:
        self.ensure_layout()
        target = self.run_dir(run.id)
        target.mkdir(parents=True, exist_ok=True)
        self.save_run(run)
        (target / "artifacts").mkdir(exist_ok=True)
        return target

    def save_run(self, run: WorkflowRun) -> None:
        target = self.run_dir(run.id)
        target.mkdir(parents=True, exist_ok=True)
        (target / "workflow.json").write_text(
            json.dumps(run.to_dict(), indent=2),
            encoding="utf-8",
        )

    def load_run(self, workflow_id: str) -> WorkflowRun:
        path = self.run_dir(workflow_id) / "workflow.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        return WorkflowRun.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_runs(self) -> list[WorkflowRun]:
        if not self.root.exists():
            return []
        runs: list[WorkflowRun] = []
        for path in sorted(self.root.iterdir()):
            payload = path / "workflow.json"
            if not payload.is_file():
                continue
            runs.append(WorkflowRun.from_dict(json.loads(payload.read_text(encoding="utf-8"))))
        return sorted(runs, key=lambda item: item.updated_at, reverse=True)

    def append_event(
        self,
        workflow_id: str,
        *,
        kind: str,
        message: str,
        node_id: str = "",
        data: dict[str, Any] | None = None,
    ) -> WorkflowEvent:
        event = WorkflowEvent(
            id=f"wev-{uuid4().hex[:10]}",
            workflow_id=workflow_id,
            kind=kind,
            message=message,
            node_id=node_id,
            data=data or {},
        )
        target = self.run_dir(workflow_id)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict()) + "\n")
        return event

    def list_events(self, workflow_id: str) -> list[WorkflowEvent]:
        path = self.run_dir(workflow_id) / "events.jsonl"
        if not path.is_file():
            return []
        events: list[WorkflowEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            events.append(WorkflowEvent.from_dict(json.loads(line)))
        return events

    def write_report(self, run: WorkflowRun) -> Path:
        target = self.run_dir(run.id)
        target.mkdir(parents=True, exist_ok=True)
        path = target / "REPORT.md"
        path.write_text(render_workflow_report(run), encoding="utf-8")
        return path


def _evidence(*requirements: str | EvidenceRequirement) -> tuple[EvidenceRequirement, ...]:
    return tuple(
        item if isinstance(item, EvidenceRequirement) else EvidenceRequirement(kind=item)
        for item in requirements
    )


def _with_evidence(
    evidence: tuple[EvidenceRequirement, ...],
    *requirements: str,
) -> tuple[EvidenceRequirement, ...]:
    existing = {item.kind for item in evidence}
    additions = tuple(
        EvidenceRequirement(kind=requirement)
        for requirement in requirements
        if requirement not in existing
    )
    return evidence + additions


def _harden_node_evidence(
    *,
    kind: str,
    allow_mutation: bool,
    evidence: tuple[EvidenceRequirement, ...],
) -> tuple[EvidenceRequirement, ...]:
    hardened = evidence
    if kind == "plan":
        hardened = _with_evidence(hardened, "planning_only")
    if kind == "work" and allow_mutation:
        hardened = _with_evidence(hardened, "environment_checked")
        hardened = _with_evidence(hardened, "files_changed")
        hardened = _with_evidence(hardened, "verify_work_if_state_changed")
    if kind == "verify":
        hardened = _with_evidence(
            hardened,
            "tool_called",
            "verify_work_passed",
            "claim_grounded",
            "no_failed_tools",
        )
    if kind in {"review", "refute"}:
        hardened = _with_evidence(hardened, "review_decision")
    if kind == "merge":
        hardened = _with_evidence(hardened, "claim_grounded", "no_failed_tools")
    if not allow_mutation:
        hardened = _with_evidence(hardened, "no_state_change")
    return hardened


def create_default_workflow(*, workflow_id: str, title: str, goal: str) -> WorkflowRun:
    nodes = (
        WorkflowNode(
            id="plan",
            title="Plan workflow",
            kind="plan",
            role="planner",
            expected_evidence=_evidence(
                "result_nonempty", "planning_only", "no_state_change", "no_failed_tools"
            ),
            max_attempts=2,
            metadata={"completion_timeout_seconds": 45},
            prompt=(
                "Plan the workflow with the available tools. Note assumptions to check, gaps "
                "to research, the expected outcome with confidence, and any child workflow "
                "that should split off. Do not claim completion.\n\n"
                f"Goal:\n{goal}"
            ),
        ),
        WorkflowNode(
            id="research",
            title="Verify assumptions and fill gaps",
            kind="research",
            role="researcher",
            depends_on=("plan",),
            expected_evidence=_evidence(
                "result_nonempty", "claim_grounded", "no_state_change", "no_failed_tools"
            ),
            max_attempts=2,
            metadata={"completion_timeout_seconds": 120, "idle_timeout_seconds": 60},
            prompt=(
                "Check the plan's important assumptions with read-only tools and fill gaps "
                "before work starts. If one path fails, try another available path before "
                "calling it blocked. Note any child workflow needed.\n\n"
                f"Goal:\n{goal}"
            ),
        ),
        WorkflowNode(
            id="work",
            title="Execute primary work",
            kind="work",
            role="implementer",
            allow_mutation=True,
            depends_on=("research",),
            expected_evidence=_evidence(
                "result_nonempty",
                "claim_grounded",
                "no_failed_tools",
                "environment_checked",
                "files_changed",
                "verify_work_if_state_changed",
            ),
            max_attempts=3,
            metadata={"completion_timeout_seconds": 240, "idle_timeout_seconds": 90},
            prompt=(
                "Do the primary work. Use tools to reduce uncertainty. If you change state, "
                "verify the final state before claiming completion.\n\n"
                f"Goal:\n{goal}"
            ),
        ),
        WorkflowNode(
            id="verify",
            title="Verify evidence",
            kind="verify",
            role="verifier",
            depends_on=("work",),
            expected_evidence=_evidence(
                "result_nonempty",
                "tool_called",
                "verify_work_passed",
                "claim_grounded",
                "no_state_change",
                "no_failed_tools",
            ),
            max_attempts=2,
            metadata={"completion_timeout_seconds": 120, "idle_timeout_seconds": 60},
            prompt=(
                "Independently verify the result against evidence. Use read-only checks or "
                "verify_work as needed. Return concrete pass/fail findings.\n\n"
                f"Goal:\n{goal}"
            ),
        ),
        WorkflowNode(
            id="review",
            title="Adversarial review",
            kind="review",
            role="reviewer",
            depends_on=("verify",),
            expected_evidence=_evidence(
                "result_nonempty",
                "claim_grounded",
                "no_state_change",
                "no_failed_tools",
                "review_decision",
            ),
            prompt=(
                "Review the evidence for false claims or missing verification. Output exactly "
                "one line: WORKFLOW_DECISION: pass or WORKFLOW_DECISION: retry.\n\n"
                f"Goal:\n{goal}"
            ),
        ),
        WorkflowNode(
            id="refute",
            title="Refute reviewer findings",
            kind="refute",
            role="refuter",
            depends_on=("review",),
            expected_evidence=_evidence(
                "result_nonempty",
                "claim_grounded",
                "no_state_change",
                "no_failed_tools",
                "review_decision",
            ),
            metadata={"can_request_retry": True},
            prompt=(
                "Challenge the reviewer using only the evidence. Output exactly one line: "
                "WORKFLOW_DECISION: pass or WORKFLOW_DECISION: retry.\n\n"
                f"Goal:\n{goal}"
            ),
        ),
        WorkflowNode(
            id="merge",
            title="Merge final answer",
            kind="merge",
            role="merger",
            depends_on=("refute",),
            expected_evidence=_evidence(
                "result_nonempty", "claim_grounded", "no_state_change", "no_failed_tools"
            ),
            metadata={"completion_timeout_seconds": 20},
            prompt=(
                "Return the defended final answer using only supported claims. Mention blockers "
                "only when the workflow proved them.\n\n"
                f"Goal:\n{goal}"
            ),
        ),
    )
    now = _utcnow()
    return WorkflowRun(
        id=workflow_id,
        title=title,
        goal=goal,
        status="pending",
        nodes=nodes,
        created_at=now,
        updated_at=now,
        metadata={"planner": "static-defended-workflow-v2", "review_rounds": 0},
    )


def _coerce_expected_evidence(raw: Any) -> tuple[EvidenceRequirement, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list | tuple):
        raise ValueError("expected_evidence must be a list")
    requirements: list[EvidenceRequirement] = []
    for item in raw:
        requirement = EvidenceRequirement.from_dict(item)
        if requirement.kind not in VALID_EVIDENCE_KINDS:
            raise ValueError(f"unknown evidence requirement: {requirement.kind!r}")
        requirements.append(requirement)
    return tuple(requirements)


def _assert_acyclic(nodes: tuple[WorkflowNode, ...]) -> tuple[WorkflowNode, ...]:
    by_id = {node.id: node for node in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("workflow node ids must be unique")
    ordered: list[WorkflowNode] = []
    pending = dict(by_id)
    while pending:
        ready = [
            node_id
            for node_id, node in pending.items()
            if all(dep in by_id and dep not in pending for dep in node.depends_on)
        ]
        if not ready:
            unresolved = ", ".join(sorted(pending))
            raise ValueError(f"workflow graph has a cycle or unknown dependency: {unresolved}")
        for node_id in sorted(ready):
            ordered.append(pending.pop(node_id))
    return tuple(ordered)


def create_workflow_from_plan_spec(
    *,
    workflow_id: str,
    title: str,
    goal: str,
    plan: dict[str, Any],
    max_nodes: int = 12,
) -> WorkflowRun:
    """Validate a model-authored workflow plan and convert it to a durable run."""

    raw_nodes = plan.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("dynamic workflow plan must contain a non-empty nodes list")
    if len(raw_nodes) > max_nodes:
        raise ValueError(f"dynamic workflow plan has {len(raw_nodes)} nodes; max is {max_nodes}")

    nodes: list[WorkflowNode] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_nodes, start=1):
        if not isinstance(item, dict):
            raise ValueError("workflow nodes must be objects")
        node_id = normalize_node_id(str(item.get("id") or ""), fallback=f"node-{index}")
        if node_id in seen:
            raise ValueError(f"duplicate workflow node id: {node_id}")
        seen.add(node_id)
        kind = str(item.get("kind") or "work").strip().lower()
        if kind not in VALID_NODE_KINDS:
            raise ValueError(f"unknown workflow node kind: {kind!r}")
        title_text = str(item.get("title") or node_id).strip()
        prompt = str(item.get("prompt") or item.get("instruction") or "").strip()
        if not prompt:
            raise ValueError(f"workflow node {node_id} is missing prompt")
        depends_on = tuple(
            normalize_node_id(str(dep), fallback="")
            for dep in item.get("depends_on") or []
            if str(dep).strip()
        )
        allow_mutation = bool(item.get("allow_mutation", kind == "work"))
        evidence = _coerce_expected_evidence(item.get("expected_evidence"))
        if not evidence:
            evidence = _evidence("result_nonempty", "claim_grounded", "no_failed_tools")
            if allow_mutation:
                evidence += _evidence("files_changed", "verify_work_if_state_changed")
            if kind in {"review", "refute"}:
                evidence += _evidence("review_decision")
        evidence = _harden_node_evidence(
            kind=kind,
            allow_mutation=allow_mutation,
            evidence=evidence,
        )
        metadata = dict(item.get("metadata") or {})
        if kind in {"verify", "merge"}:
            metadata.setdefault("completion_timeout_seconds", 20)
        nodes.append(
            WorkflowNode(
                id=node_id,
                title=title_text,
                kind=kind,  # type: ignore[arg-type]
                role=str(item.get("role") or _ROLE_BY_KIND.get(kind, kind)).strip(),
                prompt=prompt,
                depends_on=depends_on,
                allow_mutation=allow_mutation,
                expected_evidence=evidence,
                max_attempts=max(1, int(item.get("max_attempts") or 1)),
                metadata=metadata,
            )
        )

    known = {node.id for node in nodes}
    for node in nodes:
        missing = [dep for dep in node.depends_on if dep not in known]
        if missing:
            raise ValueError(f"workflow node {node.id} depends on unknown node(s): {missing}")

    if not any(node.kind == "merge" for node in nodes):
        if len(nodes) + 1 > max_nodes:
            raise ValueError("dynamic workflow plan needs a merge node within max_nodes")
        depended_on = {dep for node in nodes for dep in node.depends_on}
        terminal_ids = tuple(node.id for node in nodes if node.id not in depended_on)
        nodes.append(
            WorkflowNode(
                id="merge",
                title="Merge final answer",
                kind="merge",
                role="merger",
                depends_on=terminal_ids,
                expected_evidence=_harden_node_evidence(
                    kind="merge",
                    allow_mutation=False,
                    evidence=_evidence("result_nonempty", "claim_grounded", "no_failed_tools"),
                ),
                metadata={"completion_timeout_seconds": 20},
                prompt=(
                    "Synthesize the workflow outcome into a final answer. Include evidence, "
                    "risks, and any remaining blockers."
                ),
            )
        )

    ordered = _assert_acyclic(tuple(nodes))
    now = _utcnow()
    plan_title = str(plan.get("title") or title).strip() or title
    return WorkflowRun(
        id=workflow_id,
        title=plan_title,
        goal=goal,
        status="pending",
        nodes=ordered,
        created_at=now,
        updated_at=now,
        metadata={
            "planner": "dynamic",
            "planner_schema": "defended-workflow-plan-v1",
            "review_rounds": 0,
        },
    )


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped.strip(), flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def update_node(run: WorkflowRun, node: WorkflowNode) -> WorkflowRun:
    nodes = tuple(node if item.id == node.id else item for item in run.nodes)
    return replace(run, nodes=nodes, updated_at=_utcnow())


def node_by_id(run: WorkflowRun) -> dict[str, WorkflowNode]:
    return {node.id: node for node in run.nodes}


def terminal_node_ids(run: WorkflowRun) -> tuple[str, ...]:
    depended_on = {dep for node in run.nodes for dep in node.depends_on}
    return tuple(node.id for node in run.nodes if node.id not in depended_on)


def ready_pending_nodes(run: WorkflowRun) -> tuple[WorkflowNode, ...]:
    completed = {node.id for node in run.nodes if node.status == "completed"}
    return tuple(
        node
        for node in run.nodes
        if node.status == "pending" and all(dep in completed for dep in node.depends_on)
    )


def dependent_node_ids(run: WorkflowRun, roots: set[str]) -> set[str]:
    affected = set(roots)
    changed = True
    while changed:
        changed = False
        for node in run.nodes:
            if node.id in affected:
                continue
            if any(dep in affected for dep in node.depends_on):
                affected.add(node.id)
                changed = True
    return affected


def reset_nodes_for_retry(
    run: WorkflowRun,
    *,
    node_ids: set[str],
    feedback: str,
) -> WorkflowRun:
    updated_nodes: list[WorkflowNode] = []
    for node in run.nodes:
        if node.id not in node_ids:
            updated_nodes.append(node)
            continue
        history = list(node.metadata.get("round_history") or [])
        if node.result or node.error:
            history.append(
                {
                    "status": node.status,
                    "result": node.result,
                    "error": node.error,
                    "finished_at": node.finished_at,
                }
            )
        updated_nodes.append(
            replace(
                node,
                status="pending",
                result="",
                error="",
                session_id="",
                started_at="",
                finished_at="",
                updated_at=_utcnow(),
                metadata={
                    **node.metadata,
                    "round_history": history[-5:],
                    "review_feedback": feedback,
                },
            )
        )
    return replace(run, nodes=tuple(updated_nodes), updated_at=_utcnow())


def _json_objects_in_text(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", text):
        try:
            value, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)
    return objects


def _workflow_decision_from_payload(payload: dict[str, Any]) -> str:
    for key in ("workflow_decision", "decision", "status"):
        value = str(payload.get(key) or "").strip().lower()
        if value in {"pass", "passed", "success", "successful", "ok"}:
            return "pass"
        if value in {"retry", "fail", "failed", "failure", "blocked"}:
            return "retry"
    success = payload.get("is_successful")
    if isinstance(success, bool):
        return "pass" if success else "retry"
    return ""


def workflow_decision(text: str) -> str:
    match = re.search(
        r"^\s*(?:[*_`~\s]*)WORKFLOW_DECISION:\s*(pass|retry)(?:[*_`~\s]*)$",
        text,
        re.I | re.M,
    )
    if match:
        return match.group(1).lower()
    try:
        payload = extract_json_object(text)
    except (json.JSONDecodeError, ValueError):
        payload = {}
    decision = _workflow_decision_from_payload(payload) if payload else ""
    if decision:
        return decision
    for candidate in reversed(_json_objects_in_text(text)):
        decision = _workflow_decision_from_payload(candidate)
        if decision:
            return decision
    return ""


def workflow_requests_retry(text: str) -> bool:
    return workflow_decision(text) == "retry"


def _tool_events(activity: list[ActivityEvent]) -> list[ActivityEvent]:
    return [event for event in activity if event.kind == "tool_call.completed"]


def _tool_name(event: ActivityEvent) -> str:
    return str(event.data.get("name") or "")


def _tool_content(event: ActivityEvent) -> str:
    metadata = event.data.get("metadata")
    extra = metadata if isinstance(metadata, dict) else {}
    return "\n".join(
        str(value or "")
        for value in (
            event.data.get("content_preview"),
            extra.get("stdout"),
            extra.get("stderr"),
        )
    )


def _content_mentions_nonzero_failure(content: str) -> bool:
    return _output_reports_failure(content)


def _verify_event_passed(event: ActivityEvent) -> bool:
    if _tool_name(event) != "verify_work" or event.data.get("is_error"):
        return False
    exit_code = _tool_event_exit_code(event)
    if exit_code is not None and exit_code != 0:
        return False
    arguments = event.data.get("arguments")
    command = ""
    if isinstance(arguments, dict):
        command = str(arguments.get("command") or arguments.get("cmd") or "")
    if _shell_command_static_outcome(command) in {"failure", "exit"}:
        return False
    if _failure_branch_masks_exit_status(command):
        return False
    if _command_exits_before_trailing_command(command):
        return False
    content = _tool_content(event).lower()
    if _content_mentions_nonzero_failure(content):
        return False
    if not content.strip():
        return True
    return "passed" in content or re.search(r"\bpass\b", content) is not None


def _verify_work_passed(tool_events: list[ActivityEvent]) -> bool:
    return any(_verify_event_passed(event) for event in tool_events)


def _successful_tool_events(tool_events: list[ActivityEvent]) -> list[ActivityEvent]:
    return [event for event in tool_events if not event.data.get("is_error")]


def _result_acknowledges_missing_path_evidence(event: ActivityEvent, *, result: str) -> bool:
    if not event.data.get("is_error") or _tool_name(event) != "read_file":
        return False
    content = _tool_content(event).lower()
    if not any(
        phrase in content
        for phrase in (
            "path does not exist",
            "file does not exist",
            "does not exist",
            "no such file",
            "not found",
        )
    ):
        return False
    arguments = event.data.get("arguments")
    args = arguments if isinstance(arguments, dict) else {}
    path = str(args.get("path") or args.get("file") or args.get("filename") or "").strip()
    result_lower = result.lower()
    if path and path.lower() not in result_lower:
        return False
    return any(
        phrase in result_lower
        for phrase in (
            "does not exist",
            "do not exist",
            "not currently exist",
            "is missing",
            "is absent",
            "not found",
        )
    )


def _claim_evidence_text(
    *,
    node: WorkflowNode,
    run: WorkflowRun,
    tool_events: list[ActivityEvent],
    result: str,
) -> str:
    parts: list[str] = []
    for event in tool_events:
        if event.data.get("is_error") and not _result_acknowledges_missing_path_evidence(
            event,
            result=result,
        ):
            continue
        parts.append(_tool_name(event))
        parts.append(json.dumps(event.data.get("arguments") or {}, sort_keys=True))
        parts.append(_tool_content(event))
    if node.kind in {"review", "refute", "merge"}:
        for dep in run.nodes:
            if dep.status == "completed":
                parts.append(dep.result)
    elif node.kind == "verify":
        by_id = node_by_id(run)
        for dep_id in node.depends_on:
            dep = by_id.get(dep_id)
            if dep is not None and dep.status == "completed":
                parts.append(dep.result)
    elif node.kind == "work":
        parts.extend(_failure_history_grounded_results(node))
    return "\n".join(parts)


_ABSOLUTE_PATH_RE = re.compile(
    r"(?<![\w.:\-/])/(?!/)(?:[A-Za-z0-9._@%+=:,~-]+/)*[A-Za-z0-9._@%+=:,~-]+"
)
_COMMON_ABSOLUTE_PATH_PREFIXES = (
    "/app/",
    "/bin/",
    "/etc/",
    "/home/",
    "/mnt/",
    "/opt/",
    "/private/",
    "/root/",
    "/sbin/",
    "/srv/",
    "/tmp/",
    "/usr/",
    "/Users/",
    "/var/",
    "/workspace/",
    "/workspaces/",
)
_FILE_EXTENSION_PATH_RE = re.compile(r"/[^/\s]+\.[A-Za-z0-9]{1,12}(?:/|$)")
_INCOMPLETE_WORKFLOW_REPLY_RE = re.compile(
    r"\b("
    r"please provide (?:the )?(?:task|request)"
    r"|ready for (?:your|the) request"
    r"|how can i help(?: you)?"
    r"|what (?:would you like|do you want) me to (?:do|perform)"
    r")\b",
    re.IGNORECASE,
)
_EXACT_PATH_REQUEST_RE = re.compile(
    r"\b("
    r"exact current working directory path"
    r"|current working directory path"
    r"|current directory path"
    r"|cwd path"
    r")\b",
    re.IGNORECASE,
)
_FIRST_LINE_QUOTE_REQUEST_RE = re.compile(
    r"\b(?:quote|include|report)\b[^\n.]{0,80}\bfirst line\b"
    r"|\bfirst line\b[^\n.]{0,80}\b(?:quote|quoted|include|report)\b",
    re.IGNORECASE,
)
_FIRST_LINE_VALUE_RE = re.compile(
    r"first line(?:\s+(?:is|was))?\s*[:\-]?\s*(?:\*\*)?[\"'“”]([^\"'“”\n]+)[\"'“”]",
    re.IGNORECASE,
)
_EXACT_FILE_CONTENT_REQUEST_RE = re.compile(
    r"\b(?:create|write|make)\s+(?P<path>[A-Za-z0-9._@%+=:,~/-]+)\s+"
    r"(?:containing|that\s+contains|with(?:\s+the)?(?:\s+exact)?(?:\s+contents?|\s+content|\s+text)?)"
    r"\s+exactly\s+"
    r"(?P<content>`[^`\n]+`|\"[^\"\n]+\"|'[^'\n]+'|[^\n.;]+?)"
    r"(?=\s+(?:and|then|without|do\s+not|by|using|via)\b"
    r"|\s+with\s+(?:no|zero)\s+(?:trailing\s+|final\s+)?(?:newline|line\s+break)s?\b"
    r"|[.;,\n]|$)",
    re.IGNORECASE,
)
_EXACT_FILE_CONTENT_NAMED_REQUEST_RE = re.compile(
    r"\b(?:create|write|make)\s+(?:(?:an?|the)\s+)?"
    r"(?:(?:plain|text|json|python|node|shell|bash)\s+)?"
    r"(?:file|script|program)\s+(?:(?:named|called|at|as)\s+)?"
    r"(?P<path>[A-Za-z0-9._@%+=:,~/-]+)\s+"
    r"(?:containing|that\s+contains|with(?:\s+the)?(?:\s+exact)?(?:\s+contents?|\s+content|\s+text)?)"
    r"\s+exactly\s+"
    r"(?P<content>`[^`\n]+`|\"[^\"\n]+\"|'[^'\n]+'|[^\n.;]+?)"
    r"(?=\s+(?:and|then|without|do\s+not|by|using|via)\b"
    r"|\s+with\s+(?:no|zero)\s+(?:trailing\s+|final\s+)?(?:newline|line\s+break)s?\b"
    r"|[.;,\n]|$)",
    re.IGNORECASE,
)
_EXACT_FILE_SHOULD_SAY_REQUEST_RE = re.compile(
    r"\b(?:create|write|make)\s+(?P<path>[A-Za-z0-9._@%+=:,~/-]+)\b"
    r"[^\n]{0,160}?\b(?:it|file|contents?|text)\s+should\s+"
    r"(?:say|read|be)\s*:?\s+"
    r"(?P<content>`[^`\n]+`|\"[^\"\n]+\"|'[^'\n]+'|[^\n]+?)"
    r"(?=\s+(?:and|then|without|do\s+not|by|using|via)\b|\n|$)",
    re.IGNORECASE,
)
_EXACT_STDOUT_REQUEST_RE = re.compile(
    r"\b(?:create|write|make)\s+(?P<path>[A-Za-z0-9._@%+=:,~/-]+)\s+"
    r"(?:that\s+)?(?:prints?|outputs?|emits|writes\s+to\s+stdout)\s+exactly\s+"
    r"(?P<content>`[^`\n]+`|\"[^\"\n]+\"|'[^'\n]+'|[^\n.;]+?)"
    r"(?=\s+(?:and|then|without|do\s+not|by|using|via)\b"
    r"|\s+with\s+(?:no|zero)\s+(?:trailing\s+|final\s+)?(?:newline|line\s+break)s?\b"
    r"|[.;,\n]|$)",
    re.IGNORECASE,
)
_EXACT_STDOUT_NAMED_REQUEST_RE = re.compile(
    r"\b(?:create|write|make)\s+(?:(?:an?|the)\s+)?"
    r"(?:(?:python|node|shell|bash)\s+)?(?:script|program|file)\s+"
    r"(?:(?:named|called|at|as)\s+)?(?P<path>[A-Za-z0-9._@%+=:,~/-]+)\s+"
    r"(?:that\s+)?(?:prints?|outputs?|emits|writes\s+to\s+stdout)\s+exactly\s+"
    r"(?P<content>`[^`\n]+`|\"[^\"\n]+\"|'[^'\n]+'|[^\n.;]+?)"
    r"(?=\s+(?:and|then|without|do\s+not|by|using|via)\b"
    r"|\s+with\s+(?:no|zero)\s+(?:trailing\s+|final\s+)?(?:newline|line\s+break)s?\b"
    r"|[.;,\n]|$)",
    re.IGNORECASE,
)
_BYTE_SIZE_REQUEST_RE = re.compile(r"\b(?:byte\s+size|file\s+size|checking\s+size|bytes?)\b", re.I)
_NO_TRAILING_NEWLINE_RE = re.compile(
    r"\b(?:no|zero)\s+(?:trailing\s+|final\s+)?(?:newline|line\s+break)s?\b",
    re.I,
)
_MD5_VALUE_RE = re.compile(r"\b[a-fA-F0-9]{32}\b")
_WEB_RESULT_REQUEST_RE = re.compile(
    r"\b(?:public\s+)?web\s+results?\b|\bsource\s+urls?\b|\burls?\s+for\b",
    re.IGNORECASE,
)
_TWO_RESULT_REQUEST_RE = re.compile(
    r"\b(?:two|2)\b.{0,60}\b(?:public\s+)?web\s+results?\b",
    re.IGNORECASE | re.S,
)
_TEST_FILE_REQUEST_RE = re.compile(
    r"\b(?:add|create|write|include)\b.{0,120}\b(?:unit\s*test|unittest|test\s+file|tests?)\b"
    r"|\b(?:unit\s*test|unittest|test\s+file)\b",
    re.IGNORECASE | re.S,
)
_TEST_RUN_REQUEST_RE = re.compile(
    r"\b(?:run|execute)\s+(?:the\s+)?(?:unit\s+)?tests?\b" r"|\btests?\s+(?:pass|passed|passing)\b",
    re.IGNORECASE,
)
_TEST_ARTIFACT_EVIDENCE_RE = re.compile(
    r"(?<![\w./-])(?:[\w./-]*(?:test|spec)[\w./-]*\."
    r"(?:py|js|jsx|ts|tsx|rs|go|ex|exs|rb|java|kt|php)|tests?/[^\s\"'`]+)",
    re.IGNORECASE,
)
_TEST_RUN_EVIDENCE_RE = re.compile(
    r"\b(?:pytest|unittest|npm\s+(?:run\s+)?test|pnpm\s+test|yarn\s+test|"
    r"bun\s+test|deno\s+test|cargo\s+test|go\s+test|mix\s+test|mvn\s+test|"
    r"gradle\s+test|python[0-9.]*\s+-m\s+unittest)\b",
    re.IGNORECASE,
)
_NUMERIC_CLAIM_RE = re.compile(r"(?<![\w.-])\d+(?:\.\d+)?%?(?![\w.-])")
_URL_RE = re.compile(r"https?://[^\s)>\]}\"']+")


def _path_claims(text: str) -> list[str]:
    claims: list[str] = []
    for match in _ABSOLUTE_PATH_RE.finditer(text):
        value = match.group(0).rstrip(".,;:)]}\"'")
        if (
            value
            and value != "/"
            and _looks_like_filesystem_absolute_path(value)
            and value not in claims
        ):
            claims.append(value)
    return claims


def _looks_like_filesystem_absolute_path(value: str) -> bool:
    if value.startswith(_COMMON_ABSOLUTE_PATH_PREFIXES):
        return True
    parts = [part for part in value.split("/") if part]
    if len(parts) >= 2 and _FILE_EXTENSION_PATH_RE.search(value):
        return True
    return len(parts) >= 3 and any("." in part for part in parts)


def _urls_from_text(text: str) -> list[str]:
    urls: list[str] = []
    for match in _URL_RE.finditer(text):
        url = match.group(0).rstrip(".,;:")
        if url and url not in urls:
            urls.append(url)
    return urls


def _web_result_url_requirement(*, node: WorkflowNode, run: WorkflowRun) -> int:
    explicit = node.metadata.get("require_source_urls")
    if explicit is not None:
        if not explicit:
            return 0
        return 2 if _TWO_RESULT_REQUEST_RE.search(run.goal) else 1
    text = run.goal
    if not _WEB_RESULT_REQUEST_RE.search(text):
        return 0
    return 2 if _TWO_RESULT_REQUEST_RE.search(text) else 1


def _web_search_evidence_urls(tool_events: list[ActivityEvent]) -> list[str]:
    urls: list[str] = []
    for event in _successful_tool_events(tool_events):
        if _tool_name(event) not in {"web_search", "fetch_url"}:
            continue
        metadata = event.data.get("metadata")
        if isinstance(metadata, dict):
            results = metadata.get("results")
            if isinstance(results, list):
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    url = str(item.get("url") or "").strip()
                    if url and url not in urls:
                        urls.append(url)
            url = str(metadata.get("url") or "").strip()
            if url and url not in urls:
                urls.append(url)
        for url in _urls_from_text(_tool_content(event)):
            if url not in urls:
                urls.append(url)
    return urls


def _web_result_url_failure(
    *,
    node: WorkflowNode,
    run: WorkflowRun,
    result: str,
    tool_events: list[ActivityEvent],
) -> str:
    required = _web_result_url_requirement(node=node, run=run)
    if not required:
        return ""
    evidence_urls = _web_search_evidence_urls(tool_events)
    if node.kind in {"review", "refute", "merge"}:
        for dep in run.nodes:
            if dep.status != "completed":
                continue
            for url in _urls_from_text(dep.result):
                if url not in evidence_urls:
                    evidence_urls.append(url)
    if not evidence_urls:
        return ""
    included = [url for url in evidence_urls if url in result]
    if len(included) < min(required, len(evidence_urls)):
        return "result does not include source URL(s) from web_search evidence"
    return ""


def _requested_test_evidence_failure(
    *,
    node: WorkflowNode,
    run: WorkflowRun,
    evidence_text: str,
) -> str:
    if node.kind == "research":
        return ""
    request_text = f"{run.goal}\n{node.prompt}"
    if _TEST_FILE_REQUEST_RE.search(request_text) and not _TEST_ARTIFACT_EVIDENCE_RE.search(
        evidence_text
    ):
        return "requested test file is not grounded in evidence"
    if _TEST_RUN_REQUEST_RE.search(request_text) and not _TEST_RUN_EVIDENCE_RE.search(
        evidence_text
    ):
        return "requested test run is not grounded in evidence"
    return ""


def _numeric_claim_values(text: str) -> list[str]:
    values: list[str] = []
    for line in text.splitlines():
        cleaned = re.sub(r"^\s*\d+\.\s+", "", line)
        for match in _NUMERIC_CLAIM_RE.finditer(cleaned):
            value = match.group(0)
            digits = re.sub(r"\D", "", value)
            if len(digits) < 2 and not value.endswith("%") and "." not in value:
                continue
            if value not in values:
                values.append(value)
    return values


def _unsupported_numeric_claim_failure(
    *,
    node: WorkflowNode,
    result: str,
    evidence_text: str,
) -> str:
    if node.kind not in {"review", "refute", "merge"}:
        return ""
    evidence_values = set(_numeric_claim_values(evidence_text))
    for value in _numeric_claim_values(result):
        if value not in evidence_values:
            return f"unsupported numeric claim: {value}"
    return ""


def _list_dir_has_entries(event: ActivityEvent) -> bool:
    if _tool_name(event) != "list_dir" or event.data.get("is_error"):
        return False
    metadata = event.data.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("entries"), int):
        return metadata["entries"] > 0
    content = _tool_content(event)
    entries = [
        line.strip()
        for line in content.splitlines()
        if line.strip()
        and line.strip().lower() not in {"(empty)", "(no matches)"}
        and "no files" not in line.lower()
    ]
    return bool(entries)


def _successful_shell_command(tool_events: list[ActivityEvent], command: str) -> bool:
    expected = command.strip()
    for event in tool_events:
        if _tool_name(event) not in {"shell", "sh", "bash", "run_command"}:
            continue
        if event.data.get("is_error"):
            continue
        arguments = event.data.get("arguments")
        if not isinstance(arguments, dict):
            continue
        actual = str(arguments.get("command") or arguments.get("cmd") or "").strip()
        if actual == expected:
            return True
    return False


def _exact_path_result_required(*, node: WorkflowNode, run: WorkflowRun, result: str) -> bool:
    if not _EXACT_PATH_REQUEST_RE.search(f"{run.goal}\n{node.prompt}"):
        return False
    if node.kind in {"work", "verify", "merge"}:
        return True
    return bool(
        re.search(
            r"\bcurrent (?:working )?directory(?: path)?\b",
            result,
            re.IGNORECASE,
        )
    )


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _first_non_status_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.upper() in {"PASSED", "FAILED"}:
            continue
        return stripped
    return ""


def _first_line_quote_required(*, node: WorkflowNode, run: WorkflowRun) -> bool:
    if node.kind not in {"work", "verify", "merge"}:
        return False
    return bool(_FIRST_LINE_QUOTE_REQUEST_RE.search(f"{run.goal}\n{node.prompt}"))


def _append_unique(values: list[str], value: str) -> None:
    stripped = value.strip()
    if stripped and stripped not in values:
        values.append(stripped)


def _strip_inline_quote(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"`", '"', "'"}:
        return stripped[1:-1].strip()
    return stripped


def _exact_file_content_requests(*, node: WorkflowNode, run: WorkflowRun) -> list[tuple[str, str]]:
    requests: list[tuple[str, str]] = []
    prompt_text = f"{run.goal}\n{node.prompt}"
    for request_re in (
        _EXACT_FILE_CONTENT_REQUEST_RE,
        _EXACT_FILE_CONTENT_NAMED_REQUEST_RE,
        _EXACT_FILE_SHOULD_SAY_REQUEST_RE,
    ):
        for match in request_re.finditer(prompt_text):
            path = match.group("path").strip().rstrip(".,;:")
            content = _strip_inline_quote(match.group("content"))
            if path and content and (path, content) not in requests:
                requests.append((path, content))
    return requests


def _exact_stdout_requests(*, node: WorkflowNode, run: WorkflowRun) -> list[tuple[str, str]]:
    requests: list[tuple[str, str]] = []
    prompt_text = f"{run.goal}\n{node.prompt}"
    for request_re in (
        _EXACT_STDOUT_REQUEST_RE,
        _EXACT_STDOUT_NAMED_REQUEST_RE,
    ):
        for match in request_re.finditer(prompt_text):
            path = match.group("path").strip().rstrip(".,;:")
            content = _strip_inline_quote(match.group("content"))
            if path and content and (path, content) not in requests:
                requests.append((path, content))
    return requests


def _path_matches_request(actual: str, expected: str) -> bool:
    actual_path = actual.strip()
    expected_path = expected.strip()
    if actual_path.startswith("./"):
        actual_path = actual_path[2:]
    if expected_path.startswith("./"):
        expected_path = expected_path[2:]
    return actual_path == expected_path or actual_path.endswith(f"/{expected_path}")


def _command_has_shell_control_operator(command: str) -> bool:
    command = _strip_shell_comment_lines(command)
    in_single = False
    in_double = False
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and not in_single:
            escaped = True
            index += 1
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            index += 1
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            index += 1
            continue
        if not in_single and not in_double:
            if command.startswith("&&", index) or command.startswith("||", index):
                return True
            if char in {"&", ";", "\n"}:
                return True
        index += 1
    return False


def _tool_event_has_exact_file_content(
    event: ActivityEvent,
    *,
    path: str,
    content: str,
) -> bool:
    if event.data.get("is_error"):
        return False
    name = _tool_name(event)
    arguments = event.data.get("arguments")
    args = arguments if isinstance(arguments, dict) else {}
    arg_path = str(args.get("path") or args.get("file") or args.get("filename") or "")
    command = str(args.get("command") or args.get("cmd") or "")
    if name in {"write_file", "write"} and _path_matches_request(arg_path, path):
        return str(args.get("content") or "") == content
    if name == "edit_file" and _path_matches_request(arg_path, path):
        return str(args.get("new") or "") == content
    if name == "read_file" and _path_matches_request(arg_path, path):
        return _tool_content(event).strip() == content
    if name == "verify_work" and path in command:
        return _command_asserts_exact_content(command, path=path, content=content)
    if name in {"shell", "sh", "bash", "run_command"} and path in command:
        if not _command_reads_path(command, path=path):
            return False
        output = _tool_content(event)
        stripped_output = "\n".join(
            line
            for line in (item.strip() for item in output.splitlines())
            if line and line.upper() not in {"PASSED", "FAILED"}
        ).strip()
        return stripped_output == content or _first_non_status_line(output) == content
    return False


def _direct_file_content_write(
    event: ActivityEvent,
    *,
    path: str,
) -> str | None:
    if event.data.get("is_error"):
        return None
    name = _tool_name(event)
    arguments = event.data.get("arguments")
    args = arguments if isinstance(arguments, dict) else {}
    arg_path = str(args.get("path") or args.get("file") or args.get("filename") or "")
    if not _path_matches_request(arg_path, path):
        return None
    if name in {"write_file", "write"}:
        return str(args.get("content") or "")
    if name == "edit_file":
        return str(args.get("new") or "")
    return None


def _verify_work_event_asserts_exact_content(
    event: ActivityEvent,
    *,
    node: WorkflowNode,
    run: WorkflowRun,
    path: str,
    content: str,
) -> bool:
    if not _verify_event_passed(event):
        return False
    arguments = event.data.get("arguments")
    args = arguments if isinstance(arguments, dict) else {}
    command = str(args.get("command") or args.get("cmd") or "")
    if not _path_matches_request(command, path) and path not in command:
        return False
    if not _command_asserts_exact_content(command, path=path, content=content):
        return False
    return not _byte_size_check_requested(node=node, run=run) or _command_asserts_byte_size(
        command,
        path=path,
        size=len(content.encode("utf-8")),
    )


def _verify_work_events_assert_exact_content(
    events: list[ActivityEvent],
    *,
    node: WorkflowNode,
    run: WorkflowRun,
    path: str,
    content: str,
) -> bool:
    content_asserted = False
    byte_size_asserted = not _byte_size_check_requested(node=node, run=run)
    for event in events:
        if _tool_name(event) != "verify_work" or not _verify_event_passed(event):
            continue
        arguments = event.data.get("arguments")
        args = arguments if isinstance(arguments, dict) else {}
        command = str(args.get("command") or args.get("cmd") or "")
        if _command_asserts_exact_content(command, path=path, content=content):
            content_asserted = True
        if not byte_size_asserted and _command_asserts_byte_size(
            command,
            path=path,
            size=len(content.encode("utf-8")),
        ):
            byte_size_asserted = True
    return content_asserted and byte_size_asserted


def _verify_work_stdout_raw(event: ActivityEvent) -> str | None:
    metadata = event.data.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("stdout"), str):
        return str(metadata["stdout"])
    return None


def _verify_work_stdout(event: ActivityEvent) -> str:
    raw_stdout = _verify_work_stdout_raw(event)
    if raw_stdout is not None:
        return raw_stdout
    text = str(event.data.get("content_preview") or "")
    lines = text.splitlines()
    if lines and lines[0].strip().upper() == "PASSED":
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    return "\n".join(lines).strip()


def _verify_work_stdout_matches(
    event: ActivityEvent,
    *,
    content: str,
    no_trailing_newline_required: bool,
) -> bool:
    raw_stdout = _verify_work_stdout_raw(event)
    if raw_stdout is not None:
        if no_trailing_newline_required:
            return raw_stdout == content
        return raw_stdout.rstrip("\n") == content
    if no_trailing_newline_required:
        return False
    return _verify_work_stdout(event) == content


def _command_directly_runs_path(command: str, *, path: str) -> bool:
    command = _strip_shell_comment_lines(command)
    if re.search(r"[|<>`]", command) or _command_has_shell_control_operator(command):
        return False
    try:
        tokens = shlex.split(command)
    except ValueError:
        return False
    if not tokens:
        return False
    executable = Path(tokens[0]).name
    if executable in {"python", "python3", "node", "bash", "sh"}:
        return len(tokens) >= 2 and _path_matches_request(tokens[1], path)
    return _path_matches_request(tokens[0], path)


def _command_asserts_exact_stdout(command: str, *, path: str, content: str) -> bool:
    if path not in command or content not in command:
        return False
    if _command_can_mask_assertion_failure(command):
        return False
    return any(
        _segment_asserts_exact_stdout(segment, path=path, content=content)
        for segment in _reachable_shell_segments(command)
    )


def _segment_asserts_exact_stdout(segment: str, *, path: str, content: str) -> bool:
    if path not in segment or content not in segment:
        return False
    if _segment_asserts_exact_stdout_bytes(segment, path=path, content=content):
        return True
    escaped_path = re.escape(path)
    escaped_content = re.escape(content)
    expected_expr = rf"(?:[\"']{escaped_content}[\"']|{escaped_content}(?=\s|\]|\)|$))"
    run_expr = rf"(?:python3?|node|bash|sh)\s+{escaped_path}|\.\/{escaped_path}|{escaped_path}"
    command_substitution = rf"\$\(\s*(?:{run_expr})\s*\)"
    return bool(
        re.search(
            rf"[\"']?{command_substitution}[\"']?\s*(?:=|==)\s*{expected_expr}",
            segment,
            re.S,
        )
        or re.search(
            rf"\btest\s+[\"']?{command_substitution}[\"']?\s*=\s*{expected_expr}",
            segment,
            re.S,
        )
        or re.search(
            rf"\b(?:python3?|node|bash|sh)\b[^\n;&|]*{escaped_path}[^\n;&|]*\|\s*"
            rf"\bgrep\b[^\n;&|]*\s-[A-Za-z]*x[A-Za-z]*\b[^\n;&|]*"
            rf"{expected_expr}",
            segment,
            re.S,
        )
    )


def _python_command_code(command: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return ""
    if not tokens or Path(tokens[0]).name not in {"python", "python3"}:
        return ""
    try:
        index = tokens.index("-c")
    except ValueError:
        return ""
    if index + 1 >= len(tokens):
        return ""
    return tokens[index + 1]


def _command_asserts_exact_stdout_bytes(command: str, *, path: str, content: str) -> bool:
    if path not in command or content not in command:
        return False
    if _command_can_mask_assertion_failure(command):
        return False
    return any(
        _segment_asserts_exact_stdout_bytes(segment, path=path, content=content)
        for segment in _reachable_shell_segments(command)
    )


def _segment_asserts_exact_stdout_bytes(segment: str, *, path: str, content: str) -> bool:
    if path not in segment or content not in segment:
        return False
    code = _python_command_code(segment)
    if not code or path not in code:
        return False
    expected_bytes = repr(content.encode("utf-8"))
    if not re.search(rf"\bexpected\s*=\s*{re.escape(expected_bytes)}(?=\s|;|$)", code):
        return False
    if "subprocess.check_output" not in code:
        return False
    if not re.search(r"\b(out|stdout|actual)\s*=\s*subprocess\.check_output\s*\(", code):
        return False
    return bool(
        re.search(r"\bassert\s+(out|stdout|actual)\s*==\s*expected\b", code)
        or re.search(r"\bassert\s+expected\s*==\s*(out|stdout|actual)\b", code)
    )


def _stdout_no_trailing_newline_requested(*, node: WorkflowNode, run: WorkflowRun) -> bool:
    prompt_text = f"{run.goal}\n{node.prompt}"
    return bool(
        _exact_stdout_requests(node=node, run=run) and _NO_TRAILING_NEWLINE_RE.search(prompt_text)
    )


def _verify_work_event_asserts_exact_stdout(
    event: ActivityEvent,
    *,
    node: WorkflowNode,
    run: WorkflowRun,
    path: str,
    content: str,
) -> bool:
    if not _verify_event_passed(event) or _workflow_tool_event_changes_state(event):
        return False
    arguments = event.data.get("arguments")
    args = arguments if isinstance(arguments, dict) else {}
    command = str(args.get("command") or args.get("cmd") or "")
    no_trailing = _stdout_no_trailing_newline_requested(node=node, run=run)
    if no_trailing:
        return _command_asserts_exact_stdout_bytes(
            command,
            path=path,
            content=content,
        ) or (
            _command_directly_runs_path(command, path=path)
            and _verify_work_stdout_matches(
                event,
                content=content,
                no_trailing_newline_required=True,
            )
        )
    return _command_asserts_exact_stdout(command, path=path, content=content) or (
        _command_directly_runs_path(command, path=path)
        and _verify_work_stdout_matches(
            event,
            content=content,
            no_trailing_newline_required=False,
        )
    )


def _verify_work_events_assert_exact_stdout(
    events: list[ActivityEvent],
    *,
    node: WorkflowNode,
    run: WorkflowRun,
    path: str,
    content: str,
) -> bool:
    return any(
        _verify_work_event_asserts_exact_stdout(
            event,
            node=node,
            run=run,
            path=path,
            content=content,
        )
        for event in events
    )


def _command_asserts_exact_content(command: str, *, path: str, content: str) -> bool:
    if path not in command or content not in command:
        return False
    if _command_can_mask_assertion_failure(command):
        return False
    return any(
        _segment_asserts_exact_content(segment, path=path, content=content)
        for segment in _reachable_shell_segments(command)
    )


def _segment_asserts_exact_content(segment: str, *, path: str, content: str) -> bool:
    if path not in segment or content not in segment:
        return False
    escaped_path = re.escape(path)
    escaped_content = re.escape(content)
    file_read_expr = rf"\$\(\s*(?:cat\s+{escaped_path}|<\s*{escaped_path})\s*\)"
    expected_expr = rf"(?:[\"']{escaped_content}[\"']|{escaped_content}(?=\s|\]|\)|$))"
    has_cat_comparison = re.search(
        rf"(?:test\s+)?\[?\s*[\"']?{file_read_expr}" rf"[\"']?\s*(?:=|==)\s*{expected_expr}",
        segment,
        re.S,
    )
    if has_cat_comparison:
        return True
    for piece in _shell_command_segments(segment):
        if _grep_segment_asserts_exact_content(piece, path=path, content=content):
            return True
    process_substitution_cmp = re.search(
        rf"\b(?:cmp|diff)\b[^\n;&]*<\(\s*printf\b[^)]*{escaped_content}[^)]*\)"
        rf"[^\n;&]*{escaped_path}\b",
        segment,
        re.S,
    ) or re.search(
        rf"\b(?:cmp|diff)\b[^\n;&]*{escaped_path}\b[^\n;&]*"
        rf"<\(\s*printf\b[^)]*{escaped_content}[^)]*\)",
        segment,
        re.S,
    )
    if process_substitution_cmp:
        return True
    pipe_cmp = re.search(
        rf"\bprintf\b[^\n;&|]*{escaped_content}[^\n;&|]*\|\s*"
        rf"\b(?:cmp|diff)\b[^\n;&|]*-\s+{escaped_path}\b",
        segment,
        re.S,
    )
    if pipe_cmp:
        return True
    if re.search(r"\bpython3?\b", segment):
        read_expr = (
            rf"(?:open|Path)\s*\([^\n;&|]*{escaped_path}[^\n;&|]*" r"(?:read|read_text)\s*\(\s*\)"
        )
        if re.search(rf"{read_expr}\s*(?:=|==)\s*{expected_expr}", segment, re.S):
            return True
        if re.search(rf"[\"']{escaped_content}[\"']\s*(?:=|==)\s*{read_expr}", segment, re.S):
            return True
    return False


def _grep_segment_asserts_exact_content(segment: str, *, path: str, content: str) -> bool:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return False
    if not tokens:
        return False
    try:
        grep_index = next(
            index for index, token in enumerate(tokens) if Path(token).name.lower() == "grep"
        )
    except StopIteration:
        return False

    has_exact_match = False
    positional: list[str] = []
    index = grep_index + 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            positional.extend(tokens[index + 1 :])
            break
        if token.startswith("-") and token != "-":
            if "x" in token:
                has_exact_match = True
            if token in {"-e", "--regexp"} and index + 1 < len(tokens):
                positional.append(tokens[index + 1])
                index += 2
                continue
            index += 1
            continue
        positional.append(token)
        index += 1

    return (
        has_exact_match
        and len(positional) == 2
        and positional[0] == content
        and _path_matches_request(positional[1], path)
    )


def _strip_shell_comment_lines(command: str) -> str:
    lines: list[str] = []
    for line in command.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        in_single = False
        in_double = False
        escaped = False
        kept: list[str] = []
        for index, char in enumerate(line):
            if escaped:
                escaped = False
                kept.append(char)
                continue
            if char == "\\" and not in_single:
                escaped = True
                kept.append(char)
                continue
            if char == "'" and not in_double:
                in_single = not in_single
                kept.append(char)
                continue
            if char == '"' and not in_single:
                in_double = not in_double
                kept.append(char)
                continue
            if (
                char == "#"
                and not in_single
                and not in_double
                and (index == 0 or line[index - 1].isspace())
            ):
                break
            kept.append(char)
        lines.append("".join(kept))
    return "\n".join(lines).strip()


def _mask_shell_quoted_text(command: str) -> str:
    masked: list[str] = []
    quote = ""
    escaped = False
    for char in command:
        if quote:
            if escaped:
                escaped = False
            elif quote == '"' and char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
                masked.append(char)
                continue
            masked.append(" " if char != "\n" else "\n")
            continue
        if char in {"'", '"'}:
            quote = char
        masked.append(char)
    return "".join(masked)


def _shell_command_segments(command: str) -> list[str]:
    command = _strip_shell_comment_lines(command)
    return [
        segment.strip()
        for segment in re.split(r"\s*(?:&&|\|\||;|\n)\s*", command)
        if segment.strip()
    ]


def _shell_command_segments_with_operators(command: str) -> list[tuple[str, str]]:
    command = _strip_shell_comment_lines(command)
    pieces: list[tuple[str, str]] = []
    current: list[str] = []
    quote = ""
    escaped = False
    operator = ""
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            current.append(char)
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            current.append(char)
            index += 1
            continue
        if quote:
            if char == quote:
                quote = ""
            current.append(char)
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            index += 1
            continue
        if command.startswith("&&", index) or command.startswith("||", index):
            segment = "".join(current).strip()
            if segment:
                pieces.append((operator, segment))
            operator = command[index : index + 2]
            current = []
            index += 2
            continue
        if char in {";", "\n"}:
            segment = "".join(current).strip()
            if segment:
                pieces.append((operator, segment))
            operator = char
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    segment = "".join(current).strip()
    if segment:
        pieces.append((operator, segment))
    return pieces


def _reachable_shell_segments(command: str) -> list[str]:
    segments: list[str] = []
    previous_outcome = "success"
    for operator, segment in _shell_command_segments_with_operators(command):
        if operator == "&&" and previous_outcome in {"failure", "exit"}:
            continue
        if operator == "||" and previous_outcome in {"success", "exit"}:
            continue
        if operator in {";", "\n"} and previous_outcome == "exit":
            continue
        segments.append(segment)
        previous_outcome = _shell_segment_static_outcome(segment)
    return segments


def _shell_command_static_outcome(command: str) -> str:
    states: set[str] = {"success"}
    stripped = _strip_shell_comment_lines(command)
    for operator, segment in _shell_command_segments_with_operators(stripped):
        segment_states = _shell_outcome_states(_shell_segment_static_outcome(segment))
        next_states: set[str] = set()
        if operator == "&&":
            if "success" in states:
                next_states.update(segment_states)
            if "failure" in states:
                next_states.add("failure")
            if "exit" in states:
                next_states.add("exit")
        elif operator == "||":
            if "failure" in states:
                next_states.update(segment_states)
            if "success" in states:
                next_states.add("success")
            if "exit" in states:
                next_states.add("exit")
        elif operator in {";", "\n"}:
            if "exit" in states:
                next_states.add("exit")
            if states - {"exit"}:
                next_states.update(segment_states)
        else:
            next_states.update(segment_states)
        states = next_states or {"unknown"}
    if states == {"success"}:
        return "success"
    if states == {"failure"}:
        return "failure"
    if states == {"exit"}:
        return "exit"
    return "unknown"


def _shell_outcome_states(outcome: str) -> set[str]:
    if outcome in {"success", "failure", "exit"}:
        return {outcome}
    return {"success", "failure"}


def _shell_segment_static_outcome(segment: str) -> str:
    segment = _strip_shell_static_grouping(segment)
    if segment.startswith("!"):
        inverted_outcome = _shell_segment_static_outcome(segment[1:].lstrip())
        if inverted_outcome == "success":
            return "failure"
        if inverted_outcome in {"failure", "exit"}:
            return "success"
        return "unknown"
    pipeline_segments = _shell_pipeline_segments(segment)
    if len(pipeline_segments) > 1:
        return _shell_segment_static_outcome(pipeline_segments[-1])
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return "unknown"
    tokens = _strip_leading_env_assignments(tokens)
    if not tokens:
        return "success"
    executable = Path(tokens[0]).name.lower()
    if executable in {"command", "builtin"}:
        wrapped = tokens[1:]
        if wrapped and wrapped[0] == "--":
            wrapped = wrapped[1:]
        if not wrapped or wrapped[0].startswith("-"):
            return "unknown"
        return _shell_segment_static_outcome(shlex.join(wrapped))
    if executable in {"bash", "sh", "zsh"} and "-c" in tokens:
        command_index = tokens.index("-c") + 1
        if command_index >= len(tokens):
            return "failure"
        return _shell_segment_static_outcome(tokens[command_index])
    if executable in {"exit", "return"}:
        return "exit"
    if executable == "false":
        return "failure"
    if executable in {"true", ":"}:
        return "success"
    return "unknown"


def _strip_shell_static_grouping(segment: str) -> str:
    candidate = segment.strip()
    changed = True
    while changed:
        changed = False
        while candidate.startswith("("):
            candidate = candidate[1:].lstrip()
            changed = True
        while candidate.endswith(")"):
            candidate = candidate[:-1].rstrip()
            changed = True
    return candidate


def _shell_pipeline_segments(command: str) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    quote = ""
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            current.append(char)
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            current.append(char)
            index += 1
            continue
        if quote:
            if char == quote:
                quote = ""
            current.append(char)
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            index += 1
            continue
        if char == "|":
            previous_char = command[index - 1] if index > 0 else ""
            next_char = command[index + 1] if index + 1 < len(command) else ""
            if previous_char != "|" and next_char != "|":
                segment = "".join(current).strip()
                if segment:
                    pieces.append(segment)
                current = []
                index += 1
                continue
        current.append(char)
        index += 1
    segment = "".join(current).strip()
    if segment:
        pieces.append(segment)
    return pieces


def _strip_leading_env_assignments(tokens: list[str]) -> list[str]:
    remaining = list(tokens)
    if remaining and Path(remaining[0]).name.lower() == "env":
        remaining = remaining[1:]
    while remaining:
        token = remaining[0]
        if token.startswith("-") or "=" not in token:
            break
        key = token.split("=", 1)[0]
        if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            break
        remaining = remaining[1:]
    return remaining


def _command_can_mask_assertion_failure(command: str) -> bool:
    command = _strip_shell_comment_lines(command)
    in_single = False
    in_double = False
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and not in_single:
            escaped = True
            index += 1
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            index += 1
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            index += 1
            continue
        if not in_single and not in_double:
            if command.startswith("||", index):
                tail = command[index + 2 :].lstrip()
                if not re.match(r"(?:exit\s+[1-9]\d*|false)\b", tail):
                    return True
                index += 2
                continue
            if command.startswith("&&", index):
                tail = command[index + 2 :].lstrip()
                if _shell_tail_starts_with_static_failure(tail):
                    return True
                index += 2
                continue
            if char == "&":
                return True
            if char in {";", "\n"}:
                return True
        index += 1
    return False


def _shell_tail_starts_with_static_failure(tail: str) -> bool:
    return _shell_segment_static_outcome(tail) in {"failure", "exit"}


def _command_reads_path(command: str, *, path: str) -> bool:
    escaped_path = re.escape(path)
    return bool(
        re.search(rf"\b(?:cat|head|tail|grep|sed|awk)\b[^\n;&|]*{escaped_path}\b", command)
        or re.search(rf"<\s*{escaped_path}\b", command)
        or (
            re.search(r"\bpython3?\b", command)
            and re.search(
                rf"\b(?:open|Path)\s*\([^\n;&|]*{escaped_path}",
                command,
            )
        )
    )


def _command_asserts_byte_size(command: str, *, path: str, size: int) -> bool:
    size_text = str(size)
    if path not in command or size_text not in command:
        return False
    if _command_can_mask_assertion_failure(command):
        return False
    return any(
        _segment_asserts_byte_size(segment, path=path, size=size)
        for segment in _reachable_shell_segments(command)
    )


def _segment_asserts_byte_size(segment: str, *, path: str, size: int) -> bool:
    size_text = str(size)
    if path not in segment or size_text not in segment:
        return False
    escaped_path = re.escape(path)
    wc_expr = (
        rf"[\"']?\$\(\s*wc\s+-c\s*(?:<\s*{escaped_path}\b|{escaped_path}\b)"
        r"(?:\s*\|\s*(?:tr\s+-d\s+['\"]\s['\"]|xargs))?\s*\)[\"']?"
    )
    stat_expr = rf"[\"']?\$\(\s*stat\s+(?:-f\s*%z|-c\s*%s)\s+{escaped_path}\b\s*\)[\"']?"
    for expr in (wc_expr, stat_expr):
        if re.search(rf"{expr}\s*(?:-eq|=|==)\s*{size_text}\b", segment):
            return True
        if re.search(rf"\b{size_text}\s*(?:=|==)\s*{expr}", segment):
            return True
    if re.search(r"\bpython3?\b", segment):
        python_size_exprs = (
            rf"len\s*\([^\n;&|]*(?:open|Path)\s*\([^\n;&|]*{escaped_path}"
            rf"[^\n;&|]*(?:read|read_bytes)\s*\(\s*\)[^\n;&|]*\)",
            rf"(?:open|Path)\s*\([^\n;&|]*{escaped_path}[^\n;&|]*" r"stat\s*\(\s*\)\s*\.st_size",
            rf"\bgetsize\s*\([^\n;&|]*{escaped_path}[^\n;&|]*\)",
        )
        for expr in python_size_exprs:
            if re.search(rf"{expr}\s*(?:=|==)\s*{size_text}\b", segment, re.S):
                return True
            if re.search(rf"\b{size_text}\s*(?:=|==)\s*{expr}", segment, re.S):
                return True
    return False


def _byte_size_check_requested(*, node: WorkflowNode, run: WorkflowRun) -> bool:
    prompt_text = f"{run.goal}\n{node.prompt}"
    return bool(
        _BYTE_SIZE_REQUEST_RE.search(prompt_text)
        or (
            _exact_file_content_requests(node=node, run=run)
            and _NO_TRAILING_NEWLINE_RE.search(prompt_text)
        )
    )


def _completed_work_or_verify_has_exact_content(
    *,
    run: WorkflowRun,
    path: str,
    content: str,
) -> bool:
    for dep in run.nodes:
        if dep.kind not in {"work", "verify"} or dep.status != "completed":
            continue
        if path in dep.result and content in dep.result:
            return True
    return False


def _exact_file_content_evidenced(
    *,
    node: WorkflowNode,
    run: WorkflowRun,
    tool_events: list[ActivityEvent],
    path: str,
    content: str,
) -> bool:
    direct_writes = [
        written
        for event in _successful_tool_events(tool_events)
        if (written := _direct_file_content_write(event, path=path)) is not None
    ]
    if direct_writes and direct_writes[-1] != content:
        return False
    if any(
        _tool_event_has_exact_file_content(event, path=path, content=content)
        for event in _successful_tool_events(tool_events)
    ):
        return True
    if node.kind == "work" and _failure_history_exact_content_evidenced(
        node=node,
        path=path,
        content=content,
    ):
        return True
    if node.kind in {"verify", "review", "refute", "merge"}:
        return _completed_work_or_verify_has_exact_content(
            run=run,
            path=path,
            content=content,
        )
    return False


def _tool_event_has_exact_stdout(
    event: ActivityEvent,
    *,
    node: WorkflowNode,
    run: WorkflowRun,
    path: str,
    content: str,
) -> bool:
    if event.data.get("is_error"):
        return False
    name = _tool_name(event)
    arguments = event.data.get("arguments")
    args = arguments if isinstance(arguments, dict) else {}
    command = str(args.get("command") or args.get("cmd") or "")
    if name == "verify_work":
        return _verify_work_event_asserts_exact_stdout(
            event,
            node=node,
            run=run,
            path=path,
            content=content,
        )
    if name in {"shell", "sh", "bash", "run_command"}:
        no_trailing = _stdout_no_trailing_newline_requested(node=node, run=run)
        return _command_directly_runs_path(command, path=path) and _verify_work_stdout_matches(
            event,
            content=content,
            no_trailing_newline_required=no_trailing,
        )
    return False


def _exact_stdout_evidenced(
    *,
    node: WorkflowNode,
    run: WorkflowRun,
    tool_events: list[ActivityEvent],
    path: str,
    content: str,
) -> bool:
    if any(
        _tool_event_has_exact_stdout(
            event,
            node=node,
            run=run,
            path=path,
            content=content,
        )
        for event in _successful_tool_events(tool_events)
    ):
        return True
    if node.kind == "work" and _failure_history_exact_content_evidenced(
        node=node,
        path=path,
        content=content,
    ):
        return True
    if node.kind in {"verify", "review", "refute", "merge"}:
        return _completed_work_or_verify_has_exact_content(
            run=run,
            path=path,
            content=content,
        )
    return False


def _first_line_candidates_from_text(text: str) -> list[str]:
    candidates: list[str] = []
    for match in _FIRST_LINE_VALUE_RE.finditer(text):
        _append_unique(candidates, match.group(1))
    return candidates


def _first_line_candidates(
    *,
    node: WorkflowNode,
    run: WorkflowRun,
    tool_events: list[ActivityEvent],
) -> list[str]:
    candidates: list[str] = []
    for event in _successful_tool_events(tool_events):
        name = _tool_name(event)
        content = _tool_content(event)
        arguments = event.data.get("arguments")
        command = ""
        if isinstance(arguments, dict):
            command = str(arguments.get("command") or arguments.get("cmd") or "")
        if name == "read_file":
            _append_unique(candidates, _first_nonempty_line(content))
        elif re.search(r"\bhead\s+-n\s+1\b", command):
            _append_unique(candidates, _first_non_status_line(content))
        for value in _first_line_candidates_from_text(content):
            _append_unique(candidates, value)

    if node.kind in {"review", "refute", "merge"}:
        dependency_results = (dep.result for dep in run.nodes if dep.status == "completed")
    elif node.kind == "verify":
        by_id = node_by_id(run)
        dependency_results = (
            dep.result
            for dep_id in node.depends_on
            if (dep := by_id.get(dep_id)) is not None and dep.status == "completed"
        )
    else:
        dependency_results = ()

    for text in dependency_results:
        for value in _first_line_candidates_from_text(text):
            _append_unique(candidates, value)
    return candidates


def _claim_grounding_failure(
    *,
    node: WorkflowNode,
    result: str,
    run: WorkflowRun,
    tool_events: list[ActivityEvent],
) -> str:
    evidence_text = _claim_evidence_text(
        node=node,
        run=run,
        tool_events=tool_events,
        result=result,
    )
    if not evidence_text.strip():
        return "result has no successful tool or dependency evidence"
    if _PSEUDO_TOOL_RESULT_RE.search(result):
        return "result contains pseudo tool syntax instead of executed tool evidence"

    result_lower = result.lower()
    evidence_lower = evidence_text.lower()
    if node.kind in {"work", "verify", "review", "refute", "merge"} and (
        _INCOMPLETE_WORKFLOW_REPLY_RE.search(result_lower)
    ):
        return "result asks for a new task instead of completing the workflow node"

    if _exact_path_result_required(node=node, run=run, result=result):
        if not _path_claims(result):
            return (
                "result does not report the requested absolute path from `pwd`; "
                "include the tool stdout exactly"
            )
        if node.kind in {"work", "verify"} and not _successful_shell_command(tool_events, "pwd"):
            return (
                "requested current working directory path requires successful "
                "`shell` evidence with command `pwd`"
            )

    exact_content_requests = _exact_file_content_requests(node=node, run=run)
    decision = workflow_decision(result)
    for path, content in exact_content_requests:
        if node.kind in {"work", "verify", "merge"} and content not in result:
            return f"result does not include requested exact content for {path}: {content}"
        if (
            node.kind in {"work", "verify", "merge"} or decision == "pass"
        ) and not _exact_file_content_evidenced(
            node=node,
            run=run,
            tool_events=tool_events,
            path=path,
            content=content,
        ):
            return f"requested exact content for {path} is not grounded in evidence: {content}"

    exact_stdout_requests = _exact_stdout_requests(node=node, run=run)
    for path, content in exact_stdout_requests:
        if node.kind in {"work", "verify", "merge"} and content not in result:
            return f"result does not include requested exact output for {path}: {content}"
        if (
            node.kind in {"work", "verify", "merge"} or decision == "pass"
        ) and not _exact_stdout_evidenced(
            node=node,
            run=run,
            tool_events=tool_events,
            path=path,
            content=content,
        ):
            return f"requested exact output for {path} is not grounded in evidence: {content}"

    for path in _path_claims(result):
        if path.lower() not in evidence_lower:
            return f"unsupported path claim: {path}"

    if _first_line_quote_required(node=node, run=run):
        candidates = _first_line_candidates(node=node, run=run, tool_events=tool_events)
        if candidates and not any(candidate in result for candidate in candidates):
            return (
                "result does not include the requested exact first line from evidence: "
                f"{candidates[0]}"
            )

    url_failure = _web_result_url_failure(
        node=node,
        run=run,
        result=result,
        tool_events=tool_events,
    )
    if url_failure:
        return url_failure

    test_failure = _requested_test_evidence_failure(
        node=node,
        run=run,
        evidence_text=evidence_text,
    )
    if test_failure:
        return test_failure

    numeric_failure = _unsupported_numeric_claim_failure(
        node=node,
        result=result,
        evidence_text=evidence_text,
    )
    if numeric_failure:
        return numeric_failure

    if re.search(r"\b(?:md5|checksum)\b.{0,120}\bmatch(?:es|ed)?\b", result_lower, re.S):
        checksum_values = [value.lower() for value in _MD5_VALUE_RE.findall(evidence_text)]
        checksums = set(checksum_values)
        if len(checksums) > 1:
            return "result claims checksum match but evidence contains differing checksums"
        if len(checksum_values) < 2:
            return "result claims checksum match without comparable checksum evidence"

    if re.search(r"\b(?:empty|no files|nothing in (?:this|the) directory)\b", result_lower) and any(
        _list_dir_has_entries(event) for event in tool_events
    ):
        return "result claims the directory is empty but list_dir returned entries"

    for command in ("pwd", "ls", "pytest"):
        if re.search(rf"\b{re.escape(command)}\b", result_lower) and command not in evidence_lower:
            return f"unsupported command claim: {command}"

    return ""


def _verify_work_after_last_change(
    tool_events: list[ActivityEvent],
    *,
    node: WorkflowNode,
    run: WorkflowRun,
    exact_file_requests: list[tuple[str, str]] | None = None,
    exact_stdout_requests: list[tuple[str, str]] | None = None,
) -> bool:
    last_change_index = -1
    for index, event in enumerate(tool_events):
        if _workflow_tool_event_changes_state(event):
            last_change_index = index
    if last_change_index < 0:
        return True
    file_requests = exact_file_requests or []
    stdout_requests = exact_stdout_requests or []
    later_events = tool_events[last_change_index + 1 :]
    if not file_requests and not stdout_requests:
        return any(
            _tool_name(event) == "verify_work"
            and _verify_event_passed(event)
            and _verify_work_uses_independent_evidence(
                event,
                earlier_events=tool_events[:last_change_index],
                changed_events=tool_events[: last_change_index + 1],
                later_events=later_events,
            )
            for event in later_events
        )
    file_ok = not file_requests or all(
        _verify_work_events_assert_exact_content(
            later_events,
            node=node,
            run=run,
            path=path,
            content=content,
        )
        for path, content in file_requests
    )
    stdout_ok = not stdout_requests or all(
        _verify_work_events_assert_exact_stdout(
            later_events,
            node=node,
            run=run,
            path=path,
            content=content,
        )
        for path, content in stdout_requests
    )
    return file_ok and stdout_ok


def _verify_work_after_last_change_failure(
    tool_events: list[ActivityEvent],
    *,
    exact_file_requests: list[tuple[str, str]],
    exact_stdout_requests: list[tuple[str, str]],
) -> str:
    if exact_file_requests or exact_stdout_requests:
        return ""
    last_change_index = -1
    for index, event in enumerate(tool_events):
        if _workflow_tool_event_changes_state(event):
            last_change_index = index
    if last_change_index < 0:
        return ""
    later_events = tool_events[last_change_index + 1 :]
    passing_verifiers = [
        event
        for event in later_events
        if _tool_name(event) == "verify_work" and _verify_event_passed(event)
    ]
    if not passing_verifiers:
        return ""
    if any(
        _verify_work_uses_independent_evidence(
            event,
            earlier_events=tool_events[:last_change_index],
            changed_events=tool_events[: last_change_index + 1],
            later_events=later_events,
        )
        for event in passing_verifiers
    ):
        return ""
    return (
        "passing verify_work did not compare generated artifacts with source inputs; "
        "recompute from source inputs or run a broader test command"
    )


def _event_path_argument(event: ActivityEvent) -> str:
    arguments = event.data.get("arguments")
    args = arguments if isinstance(arguments, dict) else {}
    return str(args.get("path") or args.get("file") or args.get("filename") or "").strip()


def _event_command_argument(event: ActivityEvent) -> str:
    arguments = event.data.get("arguments")
    args = arguments if isinstance(arguments, dict) else {}
    return str(args.get("command") or args.get("cmd") or "").strip()


def _command_mentions_path(command: str, path: str) -> bool:
    normalized = path[2:] if path.startswith("./") else path
    if not normalized:
        return False
    return normalized in command or shlex.quote(normalized) in command


def _verify_command_only_checks_file_presence(command: str) -> bool:
    stripped = command.strip()
    if not stripped:
        return False
    remaining = re.sub(
        r"\btest\s+!?\s*-[efrs]\s+(?:--\s+)?(?:\"[^\"]+\"|'[^']+'|[^\s;&|()]+)",
        " ",
        stripped,
    )
    remaining = re.sub(
        r"\[\s+!?\s*-[efrs]\s+(?:\"[^\"]+\"|'[^']+'|[^\s;&|()]+)\s+\]",
        " ",
        remaining,
    )
    remaining = re.sub(r"(?:&&|\|\||;|\(|\)|\s)+", "", remaining)
    return not remaining


def _paths_read_before_change(events: list[ActivityEvent]) -> list[str]:
    paths: list[str] = []
    for event in _successful_tool_events(events):
        name = _tool_name(event)
        event_paths: list[str] = []
        if name == "read_file":
            event_paths = [_event_path_argument(event)]
        elif name in {"shell", "bash", "run_command", "execute", "verify_work"}:
            event_paths = _shell_read_paths(_event_command_argument(event))
        for path in event_paths:
            if path and path not in paths:
                paths.append(path)
    return paths


def _read_file_paths_from_events(events: list[ActivityEvent]) -> list[str]:
    paths: list[str] = []
    for event in _successful_tool_events(events):
        if _tool_name(event) != "read_file":
            continue
        path = _event_path_argument(event)
        if path and path not in paths:
            paths.append(path)
    return paths


def _safe_shell_read_path(raw: str, *, allow_bare: bool = False) -> str:
    path = raw.strip()
    if (
        not path
        or path in {"-", ".", "/dev/null"}
        or path.startswith("-")
        or path.startswith("$")
        or (path.startswith(".") and not path.startswith("./"))
        or path.startswith(("http://", "https://"))
    ):
        return ""
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return ""
    if path.startswith(("./", "tasks/")) or "/" in path:
        return path[2:] if path.startswith("./") else path
    if re.search(r"\.[A-Za-z0-9]{1,12}$", path):
        return path
    if allow_bare and re.match(r"^[A-Za-z0-9._@%+=:,~-]+$", path):
        return path
    return ""


def _shell_read_paths(command: str) -> list[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []
    paths: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        executable = Path(token).name.lower()
        if executable == "find" and index + 1 < len(tokens):
            path = _safe_shell_read_path(tokens[index + 1], allow_bare=True)
            if path and path not in paths:
                paths.append(path)
        if token in {"<", "<<<"} and index + 1 < len(tokens):
            path = _safe_shell_read_path(tokens[index + 1])
            if path and path not in paths:
                paths.append(path)
            index += 1
        elif token in {">", ">>"} and index + 1 < len(tokens):
            index += 1
        elif token.startswith((">", ">>")) or re.match(r"^\d+>{1,2}.+", token):
            pass
        else:
            path = _safe_shell_read_path(token)
            if path and path not in paths:
                paths.append(path)
        index += 1
    return paths


def _safe_shell_output_path(raw: str) -> str:
    path = raw.strip()
    if not path or path in {"-", "/dev/null"}:
        return ""
    if path.startswith("&") or path.isdigit():
        return ""
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return ""
    return str(candidate)


def _shell_output_paths(command: str) -> list[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return []
    paths: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        path = ""
        if token in {">", ">>"} and index + 1 < len(tokens):
            path = _safe_shell_output_path(tokens[index + 1])
            index += 1
        elif token.startswith((">", ">>")):
            path = _safe_shell_output_path(token.lstrip(">"))
        elif re.match(r"^\d+>{1,2}.+", token):
            path = _safe_shell_output_path(re.sub(r"^\d+>{1,2}", "", token))
        if path and path not in paths:
            paths.append(path)
        index += 1
    return paths


def _paths_changed_by_events(events: list[ActivityEvent]) -> list[str]:
    paths: list[str] = []
    for event in _successful_tool_events(events):
        name = _tool_name(event)
        if name in {"shell", "bash", "run_command", "execute"}:
            for path in _shell_output_paths(_event_command_argument(event)):
                if path and path not in paths:
                    paths.append(path)
            continue
        if name not in {"write_file", "edit_file", "write"}:
            continue
        path = _event_path_argument(event)
        if path and path not in paths:
            paths.append(path)
    return paths


def _verify_work_uses_independent_evidence(
    event: ActivityEvent,
    *,
    earlier_events: list[ActivityEvent],
    changed_events: list[ActivityEvent],
    later_events: list[ActivityEvent] | None = None,
) -> bool:
    command = _event_command_argument(event)
    if not command:
        return True
    changed_paths = _paths_changed_by_events(changed_events)
    referenced_artifact_paths = list(changed_paths)
    for path in _read_file_paths_from_events(later_events or []):
        if path and path not in referenced_artifact_paths:
            referenced_artifact_paths.append(path)
    if not referenced_artifact_paths:
        return True
    source_paths = [
        path
        for path in _paths_read_before_change(earlier_events)
        if not any(_path_matches_request(path, artifact) for artifact in referenced_artifact_paths)
    ]
    artifact_paths = [
        path
        for path in referenced_artifact_paths
        if not any(_path_matches_request(path, source) for source in source_paths)
    ]
    if not artifact_paths or not source_paths:
        return True
    mentions_artifact = any(_command_mentions_path(command, path) for path in artifact_paths)
    mentions_source = any(_command_mentions_path(command, path) for path in source_paths)
    if not mentions_artifact and not mentions_source:
        return True
    if _verify_command_only_checks_file_presence(command):
        return False
    if _verify_command_uses_weak_source_artifact_presence(
        command,
        source_paths=source_paths,
        artifact_paths=artifact_paths,
    ):
        return False
    return mentions_artifact and mentions_source


def _summary_paths(summary: dict[str, Any], key: str) -> list[str]:
    paths: list[str] = []
    for item in summary.get(key) or []:
        path = str(item or "").strip()
        if path and path not in paths:
            paths.append(path)
    return paths


def _dependency_source_artifact_paths(
    *,
    node: WorkflowNode,
    run: WorkflowRun,
) -> tuple[list[str], list[str]]:
    by_id = node_by_id(run)
    changed_paths: list[str] = []
    source_paths: list[str] = []
    for dep_id in node.depends_on:
        dep = by_id.get(dep_id)
        if dep is None or dep.status != "completed":
            continue
        summary = dep.metadata.get("activity_summary")
        if not isinstance(summary, dict):
            continue
        for path in _summary_paths(summary, "changed_paths"):
            if path not in changed_paths:
                changed_paths.append(path)
        for path in _summary_paths(summary, "read_paths"):
            if path not in source_paths:
                source_paths.append(path)
    source_paths = [
        path
        for path in source_paths
        if not any(_path_matches_request(path, changed) for changed in changed_paths)
    ]
    return source_paths, changed_paths


def _verify_work_uses_dependency_source_evidence(
    event: ActivityEvent,
    *,
    node: WorkflowNode,
    run: WorkflowRun,
) -> bool:
    command = _event_command_argument(event)
    if not command:
        return True
    source_paths, changed_paths = _dependency_source_artifact_paths(node=node, run=run)
    if not source_paths or not changed_paths:
        return True
    mentions_changed = any(_command_mentions_path(command, path) for path in changed_paths)
    mentions_source = any(_command_mentions_path(command, path) for path in source_paths)
    if not mentions_changed and not mentions_source:
        return True
    if _verify_command_only_checks_file_presence(command):
        return False
    if _verify_command_uses_weak_source_artifact_presence(
        command,
        source_paths=source_paths,
        artifact_paths=changed_paths,
    ):
        return False
    return mentions_changed and mentions_source


def _verify_work_dependency_source_failure(
    *,
    node: WorkflowNode,
    run: WorkflowRun,
    tool_events: list[ActivityEvent],
) -> str:
    passing_verifiers = [
        event
        for event in tool_events
        if _tool_name(event) == "verify_work" and _verify_event_passed(event)
    ]
    if not passing_verifiers:
        return ""
    if any(
        _verify_work_uses_dependency_source_evidence(event, node=node, run=run)
        for event in passing_verifiers
    ):
        return ""
    source_paths, changed_paths = _dependency_source_artifact_paths(node=node, run=run)
    if not source_paths or not changed_paths:
        return ""
    return (
        "passing verify_work did not compare dependency artifacts with source inputs; "
        "recompute from source inputs or run a broader test command"
    )


def _grep_segment_uses_weak_variable_presence(
    segment: str,
    *,
    artifact_paths: list[str],
) -> bool:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return False
    if not tokens:
        return False
    try:
        grep_index = next(
            index for index, token in enumerate(tokens) if Path(token).name.lower() == "grep"
        )
    except StopIteration:
        return False

    has_exact_line_match = False
    positional: list[str] = []
    index = grep_index + 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            positional.extend(tokens[index + 1 :])
            break
        if token.startswith("-") and token != "-":
            if "x" in token:
                has_exact_line_match = True
            if token in {"-e", "--regexp"} and index + 1 < len(tokens):
                positional.append(tokens[index + 1])
                index += 2
                continue
            index += 1
            continue
        positional.append(token)
        index += 1

    if has_exact_line_match or len(positional) < 2:
        return False
    pattern = positional[0].strip()
    targets = positional[1:]
    if not pattern.startswith("$"):
        return False
    return any(
        any(_path_matches_request(target, artifact) for artifact in artifact_paths)
        for target in targets
    )


def _command_uses_computed_pair_membership_assertion(command: str) -> bool:
    return bool(
        re.search(
            r"\bfor\s+\w+\s*,\s*\w+\s+in\s+\w+\.items\(\)",
            command,
            re.S,
        )
        and re.search(
            r"\bexpected\s*=\s*f?[\"'][^\"']*\{[^}]+\}[^\"']*\{[^}]+\}[^\"']*[\"']",
            command,
            re.S,
        )
        and re.search(
            r"\bassert\s+expected\s+in\s+(?:report|artifact|content|text)\b",
            command,
            re.IGNORECASE | re.S,
        )
    )


def _command_uses_weak_membership_presence(
    command: str,
    *,
    artifact_paths: list[str],
) -> bool:
    if not any(_command_mentions_path(command, path) for path in artifact_paths):
        return False
    if _command_uses_computed_pair_membership_assertion(command):
        return False
    artifact_text_names = r"(?:report|artifact|content|text)"
    return bool(
        re.search(
            rf"\ball\s*\([^;\n]*\bin\s+{artifact_text_names}\b[^;\n]*\bfor\b",
            command,
            re.IGNORECASE | re.S,
        )
        or re.search(
            rf"\bassert\s+[^;\n]*\bin\s+{artifact_text_names}\b",
            command,
            re.IGNORECASE | re.S,
        )
        or re.search(
            r"\bevery\s*\([^;\n]*(?:=>|function\b)[^;\n]*\.includes\s*\(",
            command,
            re.IGNORECASE | re.S,
        )
    )


def _verify_command_uses_weak_source_artifact_presence(
    command: str,
    *,
    source_paths: list[str],
    artifact_paths: list[str],
) -> bool:
    if not source_paths or not artifact_paths:
        return False
    if not any(_command_mentions_path(command, path) for path in source_paths):
        return False
    if not any(_command_mentions_path(command, path) for path in artifact_paths):
        return False
    if any(
        _grep_segment_uses_weak_variable_presence(segment, artifact_paths=artifact_paths)
        for segment in _shell_command_segments(command)
    ):
        return True
    return _command_uses_weak_membership_presence(
        command,
        artifact_paths=artifact_paths,
    )


def _verify_work_command_changes_state(command: str) -> bool:
    stripped = _strip_shell_comment_lines(command)
    if not stripped:
        return False
    return bool(_VERIFY_WORK_MUTATION_RE.search(_mask_shell_quoted_text(stripped)))


def _workflow_tool_event_changes_state(event: ActivityEvent) -> bool:
    name = _tool_name(event)
    metadata = event.data.get("metadata")
    if isinstance(metadata, dict) and metadata.get("workspace_changed") is True:
        return True
    if name == "verify_work":
        arguments = event.data.get("arguments")
        command = ""
        if isinstance(arguments, dict):
            command = str(arguments.get("command") or arguments.get("cmd") or "")
        return _verify_work_command_changes_state(command)
    if name in {"shell", "bash", "run_command", "execute"}:
        metadata = event.data.get("metadata")
        if isinstance(metadata, dict) and metadata.get("read_only_shell_refused"):
            return False
        arguments = event.data.get("arguments")
        command = ""
        if isinstance(arguments, dict):
            command = str(arguments.get("command") or arguments.get("cmd") or "")
        if (
            event.data.get("is_error")
            and _TOOL_UNAVAILABLE_TEXT_RE.search(_tool_content(event))
            and _tool_event_exit_code(event) is None
        ):
            return False
        if _shell_output_paths(command):
            return True
        if (
            event.data.get("is_error")
            and _tool_event_environment_failure_kind(event)
            and not _verify_work_command_changes_state(command)
            and not _ENVIRONMENT_SHELL_MUTATION_WORD_RE.search(command)
        ):
            return False
        if _shell_command_checks_environment(command):
            return False
        return tool_event_changes_state(event, _STATE_CHANGE_TOOL_NAMES)
    if event.data.get("is_error"):
        return False
    return tool_event_changes_state(event, _STATE_CHANGE_TOOL_NAMES)


def _activity_changed_state(tool_events: list[ActivityEvent]) -> bool:
    return any(_workflow_tool_event_changes_state(event) for event in tool_events)


def _shell_command_checks_environment(command: str) -> bool:
    stripped = _strip_shell_comment_lines(command)
    if not stripped:
        return False
    scan_text = _mask_shell_quoted_text(stripped)
    if _VERIFY_WORK_MUTATION_RE.search(scan_text):
        return False
    if _ENVIRONMENT_SHELL_MUTATION_WORD_RE.search(scan_text):
        return False
    try:
        parts = shlex.split(stripped)
    except ValueError:
        return False
    if not parts:
        return False
    executable = Path(parts[0]).name.lower()
    lowered = [part.lower() for part in parts]
    if executable in {"pwd", "ls", "find", "rg", "grep", "which", "env", "printenv", "uname"}:
        return True
    if executable == "command" and "-v" in lowered[1:]:
        return True
    if any(flag in lowered[1:] for flag in ("--version", "-v", "-V", "version", "--help", "-h")):
        return True
    return any(word in lowered[1:] for word in ("show", "list", "info", "help"))


def _tool_event_checks_environment(event: ActivityEvent) -> bool:
    if event.kind != "tool_call.completed":
        return False
    name = _tool_name(event)
    if name in _ENVIRONMENT_DISCOVERY_TOOL_NAMES:
        return not event.data.get("is_error")
    if name not in {"shell", "bash", "run_command", "execute", "verify_work"}:
        return False
    arguments = event.data.get("arguments")
    command = ""
    if isinstance(arguments, dict):
        command = str(arguments.get("command") or arguments.get("cmd") or "")
    if _shell_command_checks_environment(command):
        return True
    return bool(event.data.get("is_error") and _tool_event_environment_failure_kind(event))


def _shell_command_prepares_environment(command: str) -> bool:
    return any(
        _ENVIRONMENT_SETUP_WORD_RE.search(_mask_shell_quoted_text(segment))
        and not _shell_command_checks_environment(segment)
        for segment in _shell_command_segments(command)
    )


def _shell_command_contains_post_setup_environment_check(command: str) -> bool:
    setup_seen = False
    for segment in _shell_command_segments(command):
        if _ENVIRONMENT_SETUP_WORD_RE.search(
            _mask_shell_quoted_text(segment)
        ) and not _shell_command_checks_environment(segment):
            setup_seen = True
            continue
        if setup_seen and _shell_command_checks_environment(segment):
            return True
    return False


def _tool_event_prepares_environment(event: ActivityEvent) -> bool:
    if event.kind != "tool_call.completed" or event.data.get("is_error"):
        return False
    if _tool_name(event) not in {"shell", "bash", "run_command", "execute"}:
        return False
    arguments = event.data.get("arguments")
    command = ""
    if isinstance(arguments, dict):
        command = str(arguments.get("command") or arguments.get("cmd") or "")
    return _shell_command_prepares_environment(command)


def _tool_event_self_checks_environment_after_setup(event: ActivityEvent) -> bool:
    if event.kind != "tool_call.completed" or event.data.get("is_error"):
        return False
    if _tool_name(event) not in {"shell", "bash", "run_command", "execute"}:
        return False
    arguments = event.data.get("arguments")
    command = ""
    if isinstance(arguments, dict):
        command = str(arguments.get("command") or arguments.get("cmd") or "")
    return _shell_command_contains_post_setup_environment_check(command)


def _environment_probe_exit_indicates_missing(event: ActivityEvent) -> bool:
    if not event.data.get("is_error"):
        return False
    if _tool_name(event) not in {"shell", "bash", "run_command", "execute", "verify_work"}:
        return False
    arguments = event.data.get("arguments")
    command = ""
    if isinstance(arguments, dict):
        command = str(arguments.get("command") or arguments.get("cmd") or "")
    try:
        parts = shlex.split(command.strip())
    except ValueError:
        return False
    if not parts:
        return False
    executable = Path(parts[0]).name.lower()
    lowered = [part.lower() for part in parts]
    is_lookup = executable == "which" or (executable == "command" and "-v" in lowered[1:])
    if not is_lookup:
        return False
    metadata = event.data.get("metadata")
    if isinstance(metadata, dict):
        exit_code = metadata.get("exit_code")
        if isinstance(exit_code, int):
            return exit_code != 0
    return bool(re.search(r"\bexit_code:\s*[1-9]\d*\b", _tool_content(event)))


def _environment_lookup_reports_absent(event: ActivityEvent, content: str) -> bool:
    arguments = event.data.get("arguments")
    command = ""
    if isinstance(arguments, dict):
        command = str(arguments.get("command") or arguments.get("cmd") or "")
    try:
        parts = shlex.split(command.strip())
    except ValueError:
        return False
    if not parts:
        return False
    executable = Path(parts[0]).name.lower()
    lowered = [part.lower() for part in parts]
    is_lookup = executable == "which" or (executable == "command" and "-v" in lowered[1:])
    return is_lookup and bool(_ENVIRONMENT_LOOKUP_ABSENT_TEXT_RE.search(content))


def _tool_event_exit_code(event: ActivityEvent) -> int | None:
    metadata = event.data.get("metadata")
    if isinstance(metadata, dict):
        exit_code = metadata.get("exit_code")
        if isinstance(exit_code, int):
            return exit_code
    match = re.search(r"\bexit_code:\s*(-?\d+)\b", _tool_content(event))
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _tool_event_environment_failure_kind(event: ActivityEvent) -> str:
    if event.kind != "tool_call.completed":
        return ""
    if _tool_name(event) not in {"shell", "bash", "run_command", "execute", "verify_work"}:
        return ""
    content = _tool_content(event)
    exit_code = _tool_event_exit_code(event)
    if not event.data.get("is_error"):
        return "missing_tool" if _environment_lookup_reports_absent(event, content) else ""
    if (
        exit_code == 127
        or _ENVIRONMENT_MISSING_TEXT_RE.search(content)
        or _environment_probe_exit_indicates_missing(event)
    ):
        return "missing_tool"
    if _ENVIRONMENT_UNSUPPORTED_TEXT_RE.search(content):
        return "unsupported_tool_option"
    return ""


def _tool_event_command_executable(event: ActivityEvent) -> str:
    arguments = event.data.get("arguments")
    command = ""
    if isinstance(arguments, dict):
        command = str(arguments.get("command") or arguments.get("cmd") or "")
    try:
        parts = shlex.split(command.strip())
    except ValueError:
        return ""
    if not parts:
        return ""
    return Path(parts[0]).name.lower()


def _environment_failure_recovered_by_later_tool(
    event: ActivityEvent,
    later_events: list[ActivityEvent],
) -> bool:
    executable = _tool_event_command_executable(event)
    if not executable:
        return False
    for later in later_events:
        if later.kind != "tool_call.completed" or later.data.get("is_error"):
            continue
        if _tool_name(later) not in {"shell", "bash", "run_command", "execute", "verify_work"}:
            continue
        if _tool_event_command_executable(later) == executable:
            return True
    return False


def _failed_probe_recovered_by_later_verify_work(
    event: ActivityEvent,
    later_events: list[ActivityEvent],
) -> bool:
    if _tool_name(event) not in {"shell", "bash", "run_command", "execute"}:
        return False
    command = _event_command_argument(event)
    if not command:
        return False
    if not re.search(r"\b(?:grep|test|\[|diff|cmp|ls|find|stat|wc)\b", command):
        return False
    return _verify_work_passed(later_events)


def _failed_head_pipe_preview_is_informational(event: ActivityEvent) -> bool:
    if not event.data.get("is_error"):
        return False
    if _tool_name(event) not in {"shell", "bash", "run_command", "execute"}:
        return False
    if _tool_event_exit_code(event) not in {23, 56, 141}:
        return False
    command = _event_command_argument(event)
    if not re.search(r"\|\s*head\b", command):
        return False
    return "stdout:" in _tool_content(event)


def _environment_checked_before_first_change(tool_events: list[ActivityEvent]) -> bool:
    first_change_index = -1
    for index, event in enumerate(tool_events):
        if _workflow_tool_event_changes_state(event):
            first_change_index = index
            break
    if first_change_index < 0:
        return True
    return any(_tool_event_checks_environment(event) for event in tool_events[:first_change_index])


def _environment_setup_rechecked_after_change(tool_events: list[ActivityEvent]) -> bool:
    last_setup_index = -1
    last_setup_self_checked = False
    for index, event in enumerate(tool_events):
        if not _tool_event_prepares_environment(event):
            continue
        last_setup_index = index
        last_setup_self_checked = _tool_event_self_checks_environment_after_setup(event)
    if last_setup_index < 0:
        return True
    if last_setup_self_checked:
        return True
    return any(
        _tool_event_checks_environment(event) for event in tool_events[last_setup_index + 1 :]
    )


def _plan_claims_completed_work(result: str) -> bool:
    prospective_block = False
    for raw_line in result.splitlines() or [result]:
        line = raw_line.strip()
        if not line:
            prospective_block = False
            continue
        if _PLAN_FIRST_PERSON_EXECUTION_CLAIM_RE.search(line):
            return True
        if _PLAN_PROSPECTIVE_HEADER_RE.search(line):
            prospective_block = True
        if not _PLAN_EXECUTION_CLAIM_RE.search(line):
            continue
        if prospective_block or _PLAN_PROSPECTIVE_CONTEXT_RE.search(line):
            continue
        return True
    return False


def _planning_only_failure(*, result: str, tool_events: list[ActivityEvent]) -> str:
    if tool_events:
        return "plan node executed tools instead of only producing a plan"
    if _plan_claims_completed_work(result):
        return "plan node claims completed work instead of only producing a plan"
    return ""


def _successful_web_evidence(tool_events: list[ActivityEvent]) -> bool:
    return any(
        _tool_name(event) in {"web_search", "fetch_url"} and not event.data.get("is_error")
        for event in tool_events
    )


def _evidence_relevant_failed_tools(
    *,
    node: WorkflowNode,
    result: str,
    tool_events: list[ActivityEvent],
) -> list[ActivityEvent]:
    failures: list[ActivityEvent] = []
    for index, event in enumerate(tool_events):
        if not event.data.get("is_error"):
            continue
        metadata = event.data.get("metadata")
        if isinstance(metadata, dict) and metadata.get("read_only_shell_refused"):
            continue
        if _malformed_read_only_tool_call_was_recovered(
            index=index,
            event=event,
            tool_events=tool_events,
        ):
            continue
        if _result_acknowledges_missing_path_evidence(event, result=result):
            continue
        if _failed_environment_probe_is_informational(
            index=index,
            event=event,
            result=result,
            tool_events=tool_events,
        ):
            continue
        if _failed_probe_recovered_by_later_verify_work(event, tool_events[index + 1 :]):
            continue
        if _failed_head_pipe_preview_is_informational(event):
            continue
        if _tool_name(event) == "verify_work" and _verify_work_passed(tool_events[index + 1 :]):
            continue
        failures.append(event)
    successful_events = _successful_tool_events(tool_events)
    if node.kind != "research" or not failures or not successful_events:
        return failures

    result_lower = result.lower()
    relevant: list[ActivityEvent] = []
    for event in failures:
        name = _tool_name(event)
        content = _tool_content(event).lower()
        acknowledged_unavailable_tool = (
            name in result_lower
            and any(
                phrase in content
                for phrase in (
                    "unknown tool",
                    "not available",
                    "not permitted",
                    "not allowed",
                    "not accessible",
                )
            )
            and any(
                phrase in result_lower
                for phrase in (
                    "not permitted",
                    "not allowed",
                    "not available",
                    "could not run",
                    "unavailable",
                    "read-only",
                    "role",
                )
            )
        )
        if acknowledged_unavailable_tool:
            continue
        if name not in {"web_search", "fetch_url"}:
            relevant.append(event)
            continue
        if (name == "web_search" and not _successful_web_evidence(tool_events)) or (
            name == "fetch_url" and "fetch" in result_lower
        ):
            relevant.append(event)
    return relevant


def _malformed_read_only_tool_call_was_recovered(
    *,
    index: int,
    event: ActivityEvent,
    tool_events: list[ActivityEvent],
) -> bool:
    """Ignore schema slips when a later equivalent read-only call succeeds."""
    name = _tool_name(event)
    if name not in {"fetch_url", "glob", "list_dir", "read_file", "web_search"}:
        return False
    arguments = event.data.get("arguments")
    if isinstance(arguments, dict) and arguments:
        return False
    content = _tool_content(event).lower()
    if "argument" not in content:
        return False
    if not (
        "missing or invalid" in content
        or "missing required" in content
        or "invalid arguments" in content
    ):
        return False
    return any(
        _tool_name(later) == name and not later.data.get("is_error")
        for later in tool_events[index + 1 :]
    )


def _failed_environment_probe_is_informational(
    *,
    index: int,
    event: ActivityEvent,
    result: str,
    tool_events: list[ActivityEvent],
) -> bool:
    failure_kind = _tool_event_environment_failure_kind(event)
    if not _tool_event_checks_environment(event) and not failure_kind:
        return False
    if any(_tool_event_prepares_environment(later) for later in tool_events[index + 1 :]):
        return True
    if failure_kind == "unsupported_tool_option":
        return _environment_failure_recovered_by_later_tool(
            event,
            tool_events[index + 1 :],
        ) or bool(_ENVIRONMENT_UNSUPPORTED_TEXT_RE.search(result))
    content = _tool_content(event)
    if not (
        _ENVIRONMENT_MISSING_TEXT_RE.search(content)
        or _environment_probe_exit_indicates_missing(event)
    ):
        return False
    return bool(_ENVIRONMENT_MISSING_TEXT_RE.search(result))


def _failure_history_changed_state(node: WorkflowNode) -> bool:
    for item in node.metadata.get("failure_history") or []:
        if not isinstance(item, dict):
            continue
        summary = item.get("activity_summary")
        if isinstance(summary, dict) and summary.get("state_changed"):
            return True
    return False


def _failure_history_verify_work_passed(node: WorkflowNode) -> bool:
    for item in node.metadata.get("failure_history") or []:
        if not isinstance(item, dict):
            continue
        summary = item.get("activity_summary")
        if not (isinstance(summary, dict) and summary.get("verify_work_passed")):
            continue
        if _failure_history_item_evidence_passed(
            item,
            "verify_work_if_state_changed",
        ) or _failure_history_item_evidence_passed(item, "verify_work_passed"):
            return True
    return False


def _failure_history_item_evidence_passed(item: dict[str, Any], kind: str) -> bool:
    for raw_result in item.get("evidence_results") or []:
        if not isinstance(raw_result, dict):
            continue
        requirement = raw_result.get("requirement")
        if not isinstance(requirement, dict):
            continue
        if requirement.get("kind") == kind and raw_result.get("status") == "passed":
            return True
    return False


def _failure_history_evidence_passed(node: WorkflowNode, kind: str) -> bool:
    for item in node.metadata.get("failure_history") or []:
        if not isinstance(item, dict):
            continue
        if _failure_history_item_evidence_passed(item, kind):
            return True
    return False


def _failure_history_grounded_results(node: WorkflowNode) -> list[str]:
    results: list[str] = []
    for item in node.metadata.get("failure_history") or []:
        if not isinstance(item, dict):
            continue
        if not _failure_history_item_evidence_passed(item, "claim_grounded"):
            continue
        text = str(item.get("result") or "").strip()
        if text:
            results.append(text)
    return results


def _failure_history_exact_content_evidenced(
    *,
    node: WorkflowNode,
    path: str,
    content: str,
) -> bool:
    for item in node.metadata.get("failure_history") or []:
        if not isinstance(item, dict):
            continue
        if not _failure_history_item_evidence_passed(item, "claim_grounded"):
            continue
        if not _failure_history_item_evidence_passed(
            item,
            "verify_work_if_state_changed",
        ):
            continue
        text = str(item.get("result") or "")
        if path in text and content in text:
            return True
    return False


def _has_downstream_verify_gate(*, node: WorkflowNode, run: WorkflowRun) -> bool:
    for candidate in run.nodes:
        if candidate.kind != "verify" or node.id not in candidate.depends_on:
            continue
        if any(
            item.kind == "verify_work_passed" and item.required
            for item in candidate.expected_evidence
        ):
            return True
    return False


def evaluate_node_evidence(
    *,
    node: WorkflowNode,
    result: str,
    activity: list[ActivityEvent],
    run: WorkflowRun,
) -> tuple[bool, list[EvidenceResult]]:
    tool_events = _tool_events(activity)
    history_changed = _failure_history_changed_state(node)
    history_verify_passed = _failure_history_verify_work_passed(node)
    history_environment_checked = _failure_history_evidence_passed(
        node,
        "environment_checked",
    )
    completed_dependencies = {
        item.id for item in run.nodes if item.status == "completed" or item.id == node.id
    }
    results: list[EvidenceResult] = []
    for requirement in node.expected_evidence:
        passed = False
        skipped = False
        message = ""
        if requirement.kind == "result_nonempty":
            passed = bool(result.strip())
            message = "node returned a result" if passed else "node result is empty"
        elif requirement.kind == "tool_called":
            if requirement.name:
                passed = any(_tool_name(event) == requirement.name for event in tool_events)
                message = (
                    f"tool {requirement.name!r} was called"
                    if passed
                    else f"tool {requirement.name!r} was not called"
                )
            else:
                passed = any(not event.data.get("is_error") for event in tool_events)
                message = (
                    "at least one tool call succeeded"
                    if passed
                    else "no successful tool call observed"
                )
        elif requirement.kind == "no_failed_tools":
            failures = _evidence_relevant_failed_tools(
                node=node,
                result=result,
                tool_events=tool_events,
            )
            passed = not failures
            message = (
                "no failed tool calls"
                if passed
                else f"{len(failures)} tool call(s) returned errors"
            )
        elif requirement.kind == "no_state_change":
            passed = not _activity_changed_state(tool_events)
            message = (
                "node did not change workspace state"
                if passed
                else "read-only node changed workspace state"
            )
        elif requirement.kind == "planning_only":
            failure = _planning_only_failure(result=result, tool_events=tool_events)
            passed = not failure
            message = "plan node stayed in planning mode" if passed else failure
        elif requirement.kind == "claim_grounded":
            failure = _claim_grounding_failure(
                node=node,
                result=result,
                run=run,
                tool_events=tool_events,
            )
            passed = not failure
            message = "result claims are grounded in evidence" if passed else failure
        elif requirement.kind == "files_changed":
            current_changed = _activity_changed_state(tool_events)
            passed = current_changed or history_changed
            message = (
                "state-changing tool call observed"
                if current_changed
                else "state-changing tool call observed in previous attempt"
                if history_changed
                else "no state change observed"
            )
        elif requirement.kind == "environment_checked":
            changed = _activity_changed_state(tool_events)
            if not changed and not history_changed:
                passed = True
                skipped = True
                message = "no state-changing tool call observed"
            elif not changed:
                passed = history_environment_checked
                message = (
                    "environment checked around previous mutation"
                    if passed
                    else "previous mutation lacks environment inspection evidence"
                )
            else:
                checked_before = _environment_checked_before_first_change(tool_events)
                setup_rechecked = _environment_setup_rechecked_after_change(tool_events)
                passed = checked_before and setup_rechecked
                if passed:
                    message = "environment checked around mutation"
                elif not checked_before:
                    message = "no environment or workspace inspection before first mutation"
                else:
                    message = "environment setup was not rechecked after installation or update"
        elif requirement.kind in {"verify_work_passed", "tests_passed"}:
            passed = _verify_work_passed(tool_events)
            dependency_failure = (
                _verify_work_dependency_source_failure(
                    node=node,
                    run=run,
                    tool_events=tool_events,
                )
                if passed
                else ""
            )
            if dependency_failure:
                passed = False
                message = dependency_failure
            else:
                message = "verify_work passed" if passed else "verify_work did not pass"
        elif requirement.kind == "objective_status":
            decision = workflow_decision(result)
            passed = decision == "pass"
            message = (
                "objective status is pass"
                if passed
                else f"objective status is {decision or 'missing'}"
            )
        elif requirement.kind == "verify_work_if_state_changed":
            changed = _activity_changed_state(tool_events)
            if not changed and not history_changed:
                passed = True
                skipped = True
                message = "no state-changing tool call observed"
            elif not changed:
                passed = _verify_work_passed(tool_events) or history_verify_passed
                if (
                    not passed
                    and node.kind == "work"
                    and _has_downstream_verify_gate(
                        node=node,
                        run=run,
                    )
                ):
                    skipped = True
                    message = "verification deferred to dedicated verify node"
                else:
                    message = (
                        "verify_work passed after previous state change"
                        if passed
                        else "previous state change lacks a later passing verify_work call"
                    )
            else:
                exact_file_requests = _exact_file_content_requests(node=node, run=run)
                exact_stdout_requests = _exact_stdout_requests(node=node, run=run)
                passed = _verify_work_after_last_change(
                    tool_events,
                    node=node,
                    run=run,
                    exact_file_requests=exact_file_requests,
                    exact_stdout_requests=exact_stdout_requests,
                )
                exact_request_label = (
                    "content and output"
                    if exact_file_requests and exact_stdout_requests
                    else "content"
                    if exact_file_requests
                    else "output"
                )
                verify_failure = _verify_work_after_last_change_failure(
                    tool_events,
                    exact_file_requests=exact_file_requests,
                    exact_stdout_requests=exact_stdout_requests,
                )
                if (
                    not passed
                    and node.kind == "work"
                    and _has_downstream_verify_gate(
                        node=node,
                        run=run,
                    )
                ):
                    skipped = True
                    message = "verification deferred to dedicated verify node"
                else:
                    message = (
                        "verify_work ran after the last state change"
                        if passed
                        else (
                            "state changed without a later passing verify_work assertion "
                            f"for exact requested {exact_request_label}"
                            if exact_file_requests or exact_stdout_requests
                            else verify_failure
                            or "state changed without a later passing verify_work call"
                        )
                    )
        elif requirement.kind == "review_decision":
            decision = workflow_decision(result)
            passed = decision in {"pass", "retry"}
            message = (
                f"review decision is {decision}"
                if passed
                else "missing WORKFLOW_DECISION: pass|retry"
            )
        elif requirement.kind == "review_passed":
            decision = workflow_decision(result)
            passed = decision == "pass"
            message = f"review decision is {decision or 'missing'}"
        elif requirement.kind == "dependency_completed":
            dependency = requirement.name
            passed = bool(dependency and dependency in completed_dependencies)
            message = (
                f"dependency {dependency!r} completed"
                if passed
                else f"dependency {dependency!r} is not completed"
            )
        else:
            passed = not requirement.required
            skipped = True
            message = f"unknown optional evidence requirement {requirement.kind!r}"

        status: EvidenceStatus = "passed" if passed else "failed"
        if skipped:
            status = "skipped"
        results.append(EvidenceResult(requirement=requirement, status=status, message=message))

    required_passed = all(
        item.status != "failed" or not item.requirement.required for item in results
    )
    return required_passed, results


def _environment_observations(tool_events: list[ActivityEvent]) -> list[str]:
    observations: list[str] = []
    for event in tool_events:
        kind = _tool_event_environment_failure_kind(event)
        if not kind:
            continue
        arguments = event.data.get("arguments")
        command = ""
        if isinstance(arguments, dict):
            command = str(arguments.get("command") or arguments.get("cmd") or "").strip()
        content = " ".join(_tool_content(event).split())
        if len(content) > 260:
            content = content[:259].rstrip() + "..."
        label = kind.replace("_", " ")
        if command:
            observations.append(f"{label}: {command}: {content}")
        else:
            observations.append(f"{label}: {content}")
    return observations[-5:]


def summarize_activity(activity: list[ActivityEvent]) -> dict[str, Any]:
    tool_events = _tool_events(activity)
    usage_events = [event for event in activity if event.kind == "usage.recorded"]
    prompt_tokens = 0
    completion_tokens = 0
    for event in usage_events:
        prompt_tokens += int(event.data.get("prompt_tokens") or 0)
        completion_tokens += int(event.data.get("completion_tokens") or 0)
    return {
        "tool_calls": len(tool_events),
        "failed_tool_calls": sum(1 for event in tool_events if event.data.get("is_error")),
        "state_changed": _activity_changed_state(tool_events),
        "verify_work_passed": _verify_work_passed(tool_events),
        "environment_observations": _environment_observations(tool_events),
        "read_paths": _paths_read_before_change(tool_events),
        "changed_paths": _paths_changed_by_events(tool_events),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def workflow_usage(run: WorkflowRun) -> dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    for node in run.nodes:
        summary = node.metadata.get("activity_summary")
        if not isinstance(summary, dict):
            continue
        prompt_tokens += int(summary.get("prompt_tokens") or 0)
        completion_tokens += int(summary.get("completion_tokens") or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def render_workflow_mermaid(run: WorkflowRun) -> str:
    lines = ["flowchart TD"]
    for node in run.nodes:
        label = f'{node.id}["{node.id}: {node.title} ({node.status})"]'
        lines.append(f"  {label}")
    for node in run.nodes:
        for dep in node.depends_on:
            lines.append(f"  {dep} --> {node.id}")
    return "\n".join(lines) + "\n"


def render_workflow_report(run: WorkflowRun) -> str:
    usage = workflow_usage(run)
    lines = [
        f"# {run.title}",
        "",
        f"- Workflow: `{run.id}`",
        f"- Status: `{run.status}`",
        f"- Goal: {run.goal}",
        f"- Tokens: {usage['total_tokens']}",
        f"- Runtime: {run.metadata.get('runtime_seconds', 0)}s",
        "",
        "## Nodes",
        "",
    ]
    for node in run.nodes:
        summary = " ".join((node.error or node.result or "").split())
        if len(summary) > 500:
            summary = summary[:499].rstrip() + "…"
        lines.extend(
            [
                f"### {node.id}: {node.title}",
                "",
                f"- Kind: `{node.kind}`",
                f"- Role: `{node.role or _ROLE_BY_KIND.get(node.kind, node.kind)}`",
                f"- Status: `{node.status}`",
                f"- Attempts: {node.attempts}",
                f"- Runtime: {node.metadata.get('runtime_seconds', 0)}s",
                "",
                summary or "(no result)",
                "",
            ]
        )
    if run.final_report:
        lines.extend(["## Final Report", "", run.final_report, ""])
    lines.extend(["## Graph", "", "```mermaid", render_workflow_mermaid(run).rstrip(), "```", ""])
    return "\n".join(lines)


__all__ = [
    "EvidenceRequirement",
    "EvidenceResult",
    "WorkflowEvent",
    "WorkflowNode",
    "WorkflowRun",
    "WorkflowStore",
    "create_default_workflow",
    "create_workflow_from_plan_spec",
    "default_workflow_root",
    "dependent_node_ids",
    "evaluate_node_evidence",
    "extract_json_object",
    "node_by_id",
    "ready_pending_nodes",
    "render_workflow_mermaid",
    "render_workflow_report",
    "reset_nodes_for_retry",
    "summarize_activity",
    "terminal_node_ids",
    "update_node",
    "workflow_decision",
    "workflow_requests_retry",
    "workflow_usage",
]
