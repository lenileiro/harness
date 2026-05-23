"""RoleContract — typed contracts for sub-agent roles.

Pattern borrowed from the *Trace-Based Assurance Framework for Agentic AI
Orchestration* (arXiv 2603.18096) and *RL for LLM-based Multi-Agent
Systems through Orchestration Traces* (arXiv 2605.02801). Both papers
argue that sub-agent orchestration is exactly five decisions:

  1. when-to-spawn   — who decides a delegation is warranted
  2. whom-to-delegate — which role gets the work
  3. how-to-communicate — what messages cross between agents
  4. how-to-aggregate — how the spawner merges sub-results
  5. when-to-stop    — when a sub-agent's work is considered done

Today our `SpawnAgentsTool` answers #2 and #3 (role assignment + the
notify/check messaging primitives). The remaining three are implicit:
the spawner spawns, the workers run until their work item is "complete,"
the parent reads the reporter's text. A `RoleContract` makes the implicit
explicit by carrying:

  • ``inputs_schema``  — JSON Schema for the work item passed to the role
  • ``outputs_schema`` — JSON Schema the role's final output must satisfy
  • ``authority``      — which tools the role may invoke and which dirs
                         it may read/write under
  • ``stop_condition`` — a short natural-language description the role
                         can read to know when to stop, plus optional
                         deterministic checks the orchestrator runs

The orchestrator validates a returning sub-agent against the contract.
A contract violation is a soft signal (logged + surfaced via an activity
event) rather than a hard refusal — same advisory-mode philosophy we
landed on for phase gates after the cross-model A/B (advisory beats
refusal when the LLM is weak).

This module is the pure data + validation; the orchestrator integration
lives where roles are dispatched (see ``MultiAgentOrchestrator`` and
``SpawnAgentsTool``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from harness.core.telemetry import get_logger

logger = get_logger("harness.role_contract")


@dataclass(frozen=True)
class Authority:
    """What a role is permitted to do.

    Both lists are deny-list overrides: an empty list means *no
    restriction* (use the parent registry as-is). When non-empty, the
    orchestrator filters the sub-agent's tool registry to the
    intersection of `allowed_tools` and what the parent had registered.

    `cwd_subpaths` is similar for filesystem authority: when set, the
    sub-agent's filesystem tools must operate at or below these
    subpaths (relative to the parent's cwd). Enforcement is the
    individual tools' responsibility; we just carry the data here so
    the contract is auditable.
    """

    allowed_tools: tuple[str, ...] = ()
    cwd_subpaths: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowed_tools": list(self.allowed_tools),
            "cwd_subpaths": list(self.cwd_subpaths),
        }


@dataclass(frozen=True)
class RoleContract:
    """Typed contract attached to a sub-agent role.

    Args:
        role: short name (matches the orchestrator's role id).
        inputs_schema: JSON Schema (object) describing the work-item
            payload handed to the role. Empty dict = no constraint.
        outputs_schema: JSON Schema for the role's final output. Empty
            dict = unstructured / text only.
        authority: tool + filesystem authority bounds.
        stop_condition: human-readable description, included in the
            role's system prompt, of when work is considered complete.
        max_turns: cap on the sub-agent's ReAct loop. None = inherit
            the orchestrator default.

    Pydantic-style schemas are picked over Python types because the
    orchestrator already passes work items as JSON across the activity
    ledger; reusing JSON Schema keeps the contract serializable.
    """

    role: str
    inputs_schema: dict[str, Any] = field(default_factory=dict)
    outputs_schema: dict[str, Any] = field(default_factory=dict)
    authority: Authority = field(default_factory=Authority)
    stop_condition: str = ""
    max_turns: int | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "inputs_schema": dict(self.inputs_schema),
            "outputs_schema": dict(self.outputs_schema),
            "authority": self.authority.as_dict(),
            "stop_condition": self.stop_condition,
            "max_turns": self.max_turns,
        }


@dataclass(frozen=True)
class ContractValidation:
    """Outcome of validating a sub-agent's input or output against a contract.

    `ok=True` means no schema / authority / stop-condition violations
    were detected. Validation is structural-only: we don't try to
    semantically grade the agent's answer here (that's the verifier's
    job).
    """

    ok: bool
    issues: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.ok


def validate_inputs(contract: RoleContract, payload: dict[str, Any]) -> ContractValidation:
    """Check a work-item payload against the role's `inputs_schema`."""
    if not contract.inputs_schema:
        return ContractValidation(ok=True)
    issues = _check_schema(contract.inputs_schema, payload, where="inputs")
    return ContractValidation(ok=not issues, issues=tuple(issues))


def validate_outputs(contract: RoleContract, payload: Any) -> ContractValidation:
    """Check a role's final output against `outputs_schema`.

    The payload may be a dict (structured output) or a string (text-only
    result). Text-only output is acceptable when ``outputs_schema`` is
    empty or doesn't require an object.
    """
    if not contract.outputs_schema:
        return ContractValidation(ok=True)
    if isinstance(payload, str):
        # Try to parse JSON when the schema expects an object.
        import json

        try:
            payload = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            return ContractValidation(
                ok=False,
                issues=("outputs_schema requires JSON but output was unparseable text",),
            )
    if not isinstance(payload, dict):
        return ContractValidation(
            ok=False,
            issues=(f"outputs_schema expects object, got {type(payload).__name__}",),
        )
    issues = _check_schema(contract.outputs_schema, payload, where="outputs")
    return ContractValidation(ok=not issues, issues=tuple(issues))


def filter_tools_by_authority(tool_names: list[str], contract: RoleContract) -> list[str]:
    """Intersect `tool_names` with the contract's `allowed_tools`.

    Empty `allowed_tools` means "no restriction" — return everything.
    Otherwise return the intersection while preserving the parent's
    order. Tools the contract names but the parent doesn't register
    are silently dropped (logged at debug).
    """
    allowed = contract.authority.allowed_tools
    if not allowed:
        return list(tool_names)
    allowed_set = set(allowed)
    out: list[str] = []
    seen_unknown: list[str] = []
    for name in tool_names:
        if name in allowed_set:
            out.append(name)
    for needed in allowed:
        if needed not in tool_names:
            seen_unknown.append(needed)
    if seen_unknown:
        logger.debug(
            "role_contract.unknown_tools_in_authority",
            role=contract.role,
            missing=seen_unknown,
        )
    return out


# ---------------------------------------------------------------------------
# Minimal JSON-Schema-ish checker
# ---------------------------------------------------------------------------
#
# We deliberately don't pull in `jsonschema` here — the orchestrator only
# needs a tiny subset (object with `required` + `properties.<key>.type`).
# Skipping a full implementation keeps the dep footprint flat and makes
# the contract trivially serializable. Power users can swap in a richer
# validator later by replacing this function.

_PRIMITIVE_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _check_schema(schema: dict[str, Any], payload: Any, *, where: str) -> list[str]:
    issues: list[str] = []
    expected_type = schema.get("type")
    if expected_type == "object":
        if not isinstance(payload, dict):
            return [f"{where}: expected object, got {type(payload).__name__}"]
        required = schema.get("required") or []
        props = schema.get("properties") or {}
        for key in required:
            if key not in payload:
                issues.append(f"{where}: missing required key {key!r}")
        for key, value in payload.items():
            spec = props.get(key)
            if spec and "type" in spec:
                expected = _PRIMITIVE_TYPES.get(spec["type"])
                if expected is not None and not isinstance(value, expected):
                    issues.append(
                        f"{where}: key {key!r} expected {spec['type']!r}, "
                        f"got {type(value).__name__}"
                    )
    elif expected_type and expected_type in _PRIMITIVE_TYPES:
        expected = _PRIMITIVE_TYPES[expected_type]
        if not isinstance(payload, expected):
            issues.append(f"{where}: expected {expected_type!r}, got {type(payload).__name__}")
    return issues


__all__ = [
    "Authority",
    "ContractValidation",
    "RoleContract",
    "filter_tools_by_authority",
    "validate_inputs",
    "validate_outputs",
]
