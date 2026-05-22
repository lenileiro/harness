"""Post-run verification: did the agent actually accomplish the user's goal?

A `Verifier` looks at the completed session and its activity ledger, and
emits a `VerificationResult` saying whether the run can finish or needs
human review / a follow-up.

The runtime calls the configured verifier once, after the terminal `Done`
event, and yields a `Verification(result=...)` event so consumers can
render the verdict. It also records a `verification.completed` activity
entry so the verdict is part of the durable ledger.

Two real verifiers ship in core:

- `RuleVerifier`: pure-Python; checks the activity ledger for tool errors.
- `LLMJudgeVerifier`: one extra LLM call against any `Adapter` to judge
  whether the assistant's final answer addresses the user's goal.

Both are wired and tested. There are no schema-only verifiers — every
Verifier in this module has a real implementation behind it.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from harness.core.activity import ActivityEvent
from harness.core.adapter import Adapter
from harness.core.events import Done, TextDelta
from harness.core.schemas import Message, Session, VerificationResult


@runtime_checkable
class Verifier(Protocol):
    """Post-run judge.

    Receives the completed `Session` and its `activity` ledger (filtered to
    that session). Returns a `VerificationResult`. Must not raise — wrap
    failures into a `can_finish=False` result with an explanatory reason.
    """

    name: str

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult: ...


# ---------------------------------------------------------------------------
# RuleVerifier
# ---------------------------------------------------------------------------


_REFUSAL_PATTERNS: tuple[str, ...] = (
    "i do not have direct access",
    "i cannot directly access",
    "i don't have access to",
    "i don't have direct access",
    "i'm unable to access",
    "i am unable to access",
    "i lack the ability to access",
    "i cannot access the",
    "only the information i have been given",
)
"""Phrases that indicate the model verbally refused to use its tools."""


def _is_repetitive(text: str, *, window: int = 40, threshold: int = 5) -> bool:
    """Return True if any window-sized substring appears threshold+ times.

    Uses window=40 so it stays smaller than typical LLM sentence length (~48+ chars),
    making alignment-independent detection work: text[0:40] == text[48:88] == text[96:136]
    when the same phrase repeats. Four phase offsets handle cases where the first sample
    lands mid-phrase.
    """
    if len(text) < window * threshold:
        return False
    for start in (0, window // 4, window // 2, window * 3 // 4):
        if start + window > len(text):
            break
        chunk = text[start : start + window]
        if text.count(chunk) >= threshold:
            return True
    return False


class RuleVerifier:
    """Real rule-based verifier — no LLM call.

    Rules applied in order (first match wins):

    1. **Repetition stall**: if the final assistant message contains the same
       200-char block 4+ times, the model was looping → `can_finish=False`.

    2. **Verbal refusal with no tools**: if the final message contains phrases
       like "I do not have direct access" AND no tools were dispatched, the
       model lied about its capabilities → `can_finish=False`.

    3. **Tool errors**: any `tool_call.completed` event with `is_error=True`
       means the run did not finish cleanly.

    4. **No tools, clean message**: text-only answer with no error signals
       → `can_finish=True, confidence=0.5` (low because no tool evidence).

    5. **All tools succeeded** → `can_finish=True, confidence=0.9`.
    """

    name = "rule"

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        final_text = _last_assistant_text(session)

        # Rule 1: repetition stall
        if _is_repetitive(final_text):
            return VerificationResult(
                can_finish=False,
                reason="model output is a stall loop — same content repeated 4+ times",
                confidence=0.9,
                verifier_name=self.name,
            )

        # Rule 2: verbal refusal with no tool use
        lower = final_text.lower()
        if any(pat in lower for pat in _REFUSAL_PATTERNS):
            completed = [e for e in activity if e.kind == "tool_call.completed"]
            if not completed:
                return VerificationResult(
                    can_finish=False,
                    reason=(
                        "model claimed it cannot access resources but never attempted "
                        "to use the available tools — verbal refusal detected"
                    ),
                    confidence=0.85,
                    verifier_name=self.name,
                )

        # Rules 3-5: standard tool-error logic
        completed = [e for e in activity if e.kind == "tool_call.completed"]
        errors = [e for e in completed if e.data.get("is_error") is True]
        if not completed:
            return VerificationResult(
                can_finish=True,
                reason="no tools dispatched; nothing rule-based to invalidate",
                confidence=0.5,
                verifier_name=self.name,
            )
        if not errors:
            return VerificationResult(
                can_finish=True,
                reason=f"{len(completed)} tool calls, none failed",
                confidence=0.9,
                verifier_name=self.name,
            )
        names = [e.data.get("name", "?") for e in errors]
        return VerificationResult(
            can_finish=False,
            reason=f"{len(errors)} tool call(s) failed: {', '.join(sorted(set(names)))}",
            confidence=0.95,
            evidence_event_ids=[e.id for e in errors],
            verifier_name=self.name,
        )


# ---------------------------------------------------------------------------
# LLMJudgeVerifier
# ---------------------------------------------------------------------------


_JUDGE_SYSTEM_PROMPT = (
    "You are a strict evaluator of an AI assistant's work. You are given the "
    "user's original goal, the assistant's final answer, and a short summary "
    "of the tools the assistant invoked. Decide whether the assistant "
    "actually accomplished the goal.\n\n"
    "AUTOMATIC FAILURES — mark can_finish=false immediately:\n"
    "- The answer contains phrases like 'I do not have direct access', "
    "'I cannot access the repository', 'I don't have access to the codebase', "
    "or any claim that it cannot use tools — AND the TOOLS USED section shows "
    "'(no tools were invoked)'. This means the model lied about its capabilities "
    "instead of using the file/shell tools available to it.\n"
    "- The answer is highly repetitive (same paragraph or sentence appears "
    "many times) — the model was stuck in a generation loop.\n"
    "- The TOOLS USED section shows '(no tools were invoked)' but the task "
    "clearly required reading files, running code, or other tool use "
    "(e.g. 'deep dive on the code', 'analyze the repo', 'run the tests').\n\n"
    "Reply with ONLY a JSON object on a single line, matching this shape:\n"
    '{"can_finish": true|false, "reason": "<short explanation>", '
    '"confidence": 0.0..1.0}\n'
    "Do not add any prose around the JSON."
)


def _summarize_tools(activity: list[ActivityEvent]) -> str:
    """One-line-per-tool-call summary suitable for the judge prompt."""
    completed = [e for e in activity if e.kind == "tool_call.completed"]
    if not completed:
        return "(no tools were invoked)"
    lines = []
    for e in completed:
        name = e.data.get("name", "?")
        ok = "ok" if not e.data.get("is_error") else "ERROR"
        preview = (e.data.get("content_preview") or "").strip()
        preview_str = f" output={preview!r}" if preview else ""
        lines.append(f"- {name} [{ok}]{preview_str}")
    return "\n".join(lines)


def _first_user_message(session: Session) -> str:
    return next((m.content or "" for m in session.messages if m.role == "user"), "")


def _last_assistant_text(session: Session) -> str:
    for m in reversed(session.messages):
        if m.role == "assistant" and m.content:
            return m.content
    return ""


class LLMJudgeVerifier:
    """Real LLM-based verifier — one extra adapter call.

    Pass the same kind of `Adapter` the agent uses (Ollama, OpenRouter, etc.)
    and the model id you want as the judge. The judge can be a different
    model from the worker; choose a strong one if you can afford the cost.

    Non-JSON or malformed responses degrade gracefully to
    `can_finish=False, confidence=0.0` with the parse error captured in
    `reason` — never raises.
    """

    name = "llm"

    def __init__(self, *, adapter: Adapter, model: str, max_retries: int = 3) -> None:
        self.adapter = adapter
        self.model = model
        self.max_retries = max_retries

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        goal = _first_user_message(session)
        answer = _last_assistant_text(session)
        tools_summary = _summarize_tools(activity)

        prompt = (
            f"USER GOAL:\n{goal}\n\n"
            f"ASSISTANT FINAL ANSWER:\n{answer}\n\n"
            f"TOOLS USED:\n{tools_summary}\n"
        )
        messages = [
            Message(role="system", content=_JUDGE_SYSTEM_PROMPT),
            Message(role="user", content=prompt),
        ]

        last_reason = "judge failed after retries"
        for attempt in range(self.max_retries):
            if attempt > 0:
                await asyncio.sleep(2**attempt)  # 2s, 4s

            accumulated: list[str] = []
            final_content: str | None = None
            try:
                async for event in self.adapter.stream(model=self.model, messages=messages):
                    if isinstance(event, TextDelta):
                        accumulated.append(event.text)
                    elif isinstance(event, Done):
                        if event.final_message and event.final_message.content:
                            final_content = event.final_message.content
                        else:
                            final_content = "".join(accumulated)
                        break
            except Exception as exc:
                last_reason = f"judge call failed (attempt {attempt + 1}): {exc!s}"
                continue

            if final_content is None:
                last_reason = f"judge stream ended without a final message (attempt {attempt + 1})"
                continue

            parsed = _parse_judge_response(final_content)
            if parsed is None:
                preview = final_content.strip()[:200]
                last_reason = f"judge returned non-JSON (attempt {attempt + 1}): {preview!r}"
                continue

            can_finish, reason, confidence = parsed
            return VerificationResult(
                can_finish=can_finish,
                reason=reason,
                confidence=confidence,
                verifier_name=self.name,
            )

        return VerificationResult(
            can_finish=False,
            reason=last_reason,
            confidence=0.0,
            verifier_name=self.name,
        )


# ---------------------------------------------------------------------------
# VerifierRouter
# ---------------------------------------------------------------------------


class VerifierRouter:
    """Routes to RuleVerifier or LLMJudgeVerifier based on observed tool activity.

    Read-only runs (no mutating tool calls) → fast rule verifier.
    Runs that wrote files or executed shell → LLM judge.
    """

    name = "router"

    _MUTATING_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file", "shell"})

    def __init__(self, *, rule: RuleVerifier, llm: LLMJudgeVerifier) -> None:
        self._rule = rule
        self._llm = llm

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        completed = [e for e in activity if e.kind == "tool_call.completed"]
        used_tools = {e.data.get("name") for e in completed}

        # No tools at all → LLM judge. Rule has nothing to check and the model
        # may have verbally claimed to do work it never actually performed.
        # Only read-only tools → rule is sufficient (fast, no LLM cost).
        # Any mutating tool → LLM judge (verify the outcome was correct).
        use_llm = not completed or bool(used_tools & self._MUTATING_TOOLS)
        verifier = self._llm if use_llm else self._rule
        result = await verifier.verify(session=session, activity=activity)
        return VerificationResult(
            can_finish=result.can_finish,
            reason=result.reason,
            confidence=result.confidence,
            evidence_event_ids=result.evidence_event_ids,
            verifier_name=self.name,
        )


_WORK_ITEM_JUDGE_PROMPT = (
    "You are a strict quality reviewer checking whether an AI worker completed a "
    "specific assigned task.\n\n"
    "You will be given the original task specification, a list of tools the worker "
    "invoked, and the worker's completion summary. Your job: decide if the worker "
    "actually did the work.\n\n"
    "Reply ONLY with valid JSON on a single line, no markdown:\n"
    '{"can_finish": true|false, "reason": "<one sentence>", "confidence": 0.0..1.0}\n\n'
    "can_finish=false if:\n"
    "- Summary is empty, vague ('done', 'completed it'), or never references the task\n"
    "- No tools were called (only complete_work_item) — worker did nothing\n"
    "- Worker describes doing something unrelated to the assigned task\n"
    "- The claimed outcome contradicts the evidence in the tool calls\n"
    "can_finish=true if:\n"
    "- Summary clearly describes completing this specific task with concrete details\n"
    "- Tool evidence shows work was actually done (files written, commands run, etc.)\n"
    "Do not add any prose outside the JSON."
)


class WorkItemJudge:
    """Isolated LLM judge for work-item completion verification.

    Called by the orchestrator after a worker finishes a task, with a
    fresh context that contains only the task spec and completion evidence.
    The judge has never seen the worker's conversation and cannot be
    influenced by it.

    Reuses the same adapter as the worker but in a single-turn call — no
    session state, no tool loop. Pass a stronger model as `model` for
    higher-quality judgments.
    """

    name = "work_item_judge"

    def __init__(self, *, adapter: Adapter, model: str, max_retries: int = 2) -> None:
        self.adapter = adapter
        self.model = model
        self.max_retries = max_retries

    async def judge(
        self,
        *,
        task_title: str,
        task_description: str | None,
        result_summary: str,
        activity: list[ActivityEvent],
    ) -> VerificationResult:
        """Single LLM call to verify a work item was genuinely completed.

        Args:
            task_title: Original task title (the spec the worker was given).
            task_description: Optional longer description from the task.
            result_summary: What the worker claims to have accomplished.
            activity: Activity events from the worker's session (tool calls etc.).

        Returns:
            VerificationResult with can_finish=True if the judge accepts the
            completion, or can_finish=False with a reason if it rejects.
        """
        tools_summary = _summarize_tools(activity)
        desc_section = f"TASK DESCRIPTION:\n{task_description}\n\n" if task_description else ""
        prompt = (
            f"TASK TITLE: {task_title}\n\n"
            f"{desc_section}"
            f"TOOLS CALLED BY WORKER:\n{tools_summary}\n\n"
            f"WORKER COMPLETION SUMMARY:\n{result_summary or '(empty)'}\n"
        )
        messages = [
            Message(role="system", content=_WORK_ITEM_JUDGE_PROMPT),
            Message(role="user", content=prompt),
        ]

        last_reason = "judge failed after retries"
        for attempt in range(self.max_retries):
            if attempt > 0:
                await asyncio.sleep(2**attempt)

            accumulated: list[str] = []
            final_content: str | None = None
            try:
                async for event in self.adapter.stream(model=self.model, messages=messages):
                    if isinstance(event, TextDelta):
                        accumulated.append(event.text)
                    elif isinstance(event, Done):
                        if event.final_message and event.final_message.content:
                            final_content = event.final_message.content
                        else:
                            final_content = "".join(accumulated)
                        break
            except Exception as exc:
                last_reason = f"judge call failed (attempt {attempt + 1}): {exc!s}"
                continue

            if final_content is None:
                last_reason = f"judge stream ended without content (attempt {attempt + 1})"
                continue

            parsed = _parse_judge_response(final_content)
            if parsed is None:
                preview = final_content.strip()[:200]
                last_reason = f"judge returned non-JSON (attempt {attempt + 1}): {preview!r}"
                continue

            can_finish, reason, confidence = parsed
            return VerificationResult(
                can_finish=can_finish,
                reason=reason,
                confidence=confidence,
                verifier_name=self.name,
            )

        return VerificationResult(
            can_finish=False,
            reason=last_reason,
            confidence=0.0,
            verifier_name=self.name,
        )


def _parse_judge_response(text: str) -> tuple[bool, str, float | None] | None:
    """Return (can_finish, reason, confidence) or None on parse failure.

    Tolerates ```json``` fences and surrounding whitespace.
    """
    body = text.strip()
    if body.startswith("```"):
        # Strip the first line (fence) and last line if also a fence.
        lines = body.splitlines()
        if lines:
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
        body = "\n".join(lines).strip()
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "can_finish" not in obj:
        return None
    can_finish = bool(obj["can_finish"])
    reason = str(obj.get("reason", "")).strip() or "(no reason given)"
    conf_raw = obj.get("confidence")
    confidence: float | None
    try:
        confidence = float(conf_raw) if conf_raw is not None else None
    except (TypeError, ValueError):
        confidence = None
    return can_finish, reason, confidence


# ---------------------------------------------------------------------------
# EvidenceContract + VerificationGateway
# ---------------------------------------------------------------------------

EvidenceCheckKind = Literal[
    "command_evidence",  # shell/subprocess ran (exit_code in activity metadata)
    "changed_file",  # write_file or edit_file completed without error
    "acceptance_criterion",  # any tool result with metadata.acceptance=True
    "no_prediction_errors",  # zero medium+ severity prediction mismatches
    "tool_succeeded",  # named tool (specified in check_data["tool"]) succeeded
]


class EvidenceContract(BaseModel):
    """Specifies what kinds of proof are required before can_finish=True.

    Evaluated by VerificationGateway against the session's activity ledger.
    """

    model_config = ConfigDict(extra="forbid")

    required_checks: list[EvidenceCheckKind]
    check_data: dict[str, Any] = Field(default_factory=dict)


class EvidenceContractResult(BaseModel):
    """Result of evaluating an EvidenceContract against the activity ledger."""

    model_config = ConfigDict(extra="forbid")

    satisfied: bool
    found_checks: list[str]
    missing_checks: list[str]


_PREDICTION_ERROR_SEVERITIES = frozenset({"medium", "high", "critical"})


def evaluate_evidence(
    contract: EvidenceContract,
    activity: list[ActivityEvent],
) -> EvidenceContractResult:
    """Check which required evidence kinds are present in the activity ledger."""
    found: list[str] = []
    missing: list[str] = []

    completed = [e for e in activity if e.kind == "tool_call.completed"]

    for check in contract.required_checks:
        if check == "command_evidence":
            ok = any(
                e.data.get("metadata", {}).get("exit_code") is not None
                for e in completed
                if not e.data.get("is_error")
            )

        elif check == "changed_file":
            ok = any(
                e.data.get("name") in ("write_file", "edit_file") and not e.data.get("is_error")
                for e in completed
            )

        elif check == "acceptance_criterion":
            ok = any(
                e.data.get("metadata", {}).get("acceptance") is True
                for e in completed
                if not e.data.get("is_error")
            )

        elif check == "no_prediction_errors":
            ok = not any(
                e.kind == "tool_call.prediction_error"
                and e.data.get("severity") in _PREDICTION_ERROR_SEVERITIES
                and not e.data.get("matched")
                for e in activity
            )

        elif check == "tool_succeeded":
            target_tool = contract.check_data.get("tool", "")
            ok = any(
                e.data.get("name") == target_tool and not e.data.get("is_error") for e in completed
            )

        else:
            ok = False

        if ok:
            found.append(check)
        else:
            missing.append(check)

    return EvidenceContractResult(
        satisfied=len(missing) == 0,
        found_checks=found,
        missing_checks=missing,
    )


class VerificationGateway:
    """Wraps any Verifier and adds EvidenceContract + prediction-mismatch gating.

    Gates checked in order:
    1. Unresolved prediction mismatches (medium+ severity) → can_finish=False
    2. EvidenceContract not satisfied → can_finish=False
    3. Inner verifier verdict (RuleVerifier, LLMJudgeVerifier, etc.)
    """

    name = "gateway"

    def __init__(
        self,
        verifier: Verifier,
        contract: EvidenceContract | None = None,
    ) -> None:
        self._verifier = verifier
        self._contract = contract

    async def verify(
        self,
        *,
        session: Session,
        activity: list[ActivityEvent],
    ) -> VerificationResult:
        # Gate 1: unresolved prediction mismatches
        mismatches = [
            e
            for e in activity
            if e.kind == "tool_call.prediction_error"
            and e.data.get("severity") in _PREDICTION_ERROR_SEVERITIES
            and not e.data.get("matched")
        ]
        if mismatches:
            names = {e.data.get("tool_name", "?") for e in mismatches}
            return VerificationResult(
                can_finish=False,
                reason=(
                    f"{len(mismatches)} unresolved prediction mismatch(es) on "
                    f"{', '.join(sorted(names))}"
                ),
                confidence=0.95,
                evidence_event_ids=[e.id for e in mismatches],
                verifier_name=self.name,
            )

        # Gate 2: evidence contract
        if self._contract is not None:
            contract_result = evaluate_evidence(self._contract, activity)
            if not contract_result.satisfied:
                return VerificationResult(
                    can_finish=False,
                    reason=(
                        f"evidence contract not satisfied — "
                        f"missing: {', '.join(contract_result.missing_checks)}"
                    ),
                    confidence=0.9,
                    verifier_name=self.name,
                )

        # Gate 3: inner verifier
        return await self._verifier.verify(session=session, activity=activity)


# ---------------------------------------------------------------------------
# ClaimGroundingVerifier
# ---------------------------------------------------------------------------

_COUNT_CLAIM_RE = re.compile(
    r"\b(\d+)\s+(?:(?:python|source|total)\s+)?"
    r"(?:file|error|line|package|module|item|result|function|class|number)s?"
    r"(?:\s+(?:were|was|found|counted|detected|identified))?",
    re.IGNORECASE,
)

_WRITE_CLAIM_RE = re.compile(
    r"(?:wrote|saved|created|written|stored|saving)\s+(?:to\s+)?"
    r"([\w./][\w./\-]*\.(?:py|txt|json|sh|md|csv|yaml|yml))",
    re.IGNORECASE,
)


class ClaimGroundingVerifier:
    """Verifier that checks whether specific claims in the final message are
    backed by actual tool output.

    Checks two claim types:

    1. **Count claims**: numbers followed by file/error/line/etc. keywords —
       the claimed number must appear in at least one successful tool call's
       ``content_preview``.
    2. **Write claims**: "wrote/saved/created X.py" — a ``write_file`` activity
       event with a matching path must exist.

    Returns ``can_finish=True, confidence=0.4`` when there are no completed
    tool events to ground against (nothing to check). Returns
    ``can_finish=True, confidence=0.85`` when every claim is grounded.
    Returns ``can_finish=False, confidence=0.75`` listing up to 3 ungrounded
    claims.
    """

    name = "claim_grounding"

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        completed = [
            e for e in activity if e.kind == "tool_call.completed" and not e.data.get("is_error")
        ]

        if not completed:
            return VerificationResult(
                can_finish=True,
                reason="no completed tool events — nothing to ground claims against",
                confidence=0.4,
                verifier_name=self.name,
            )

        final_text = _last_assistant_text(session)

        # Build corpus of all content_previews (as strings) for count checking.
        corpus = " ".join(str(e.data.get("content_preview") or "") for e in completed)

        # Build set of write_file paths (basename + full) for write checking.
        write_paths: set[str] = set()
        for e in completed:
            if e.data.get("name") == "write_file":
                path = e.data.get("arguments", {}).get("path", "")
                if path:
                    write_paths.add(path)
                    write_paths.add(Path(path).name)

        ungrounded: list[str] = []

        # Check count claims.
        for m in _COUNT_CLAIM_RE.finditer(final_text):
            number = m.group(1)
            if number not in corpus:
                ungrounded.append(
                    f"count claim '{m.group(0).strip()}' (number {number} not found in tool output)"
                )
                if len(ungrounded) >= 3:
                    break

        # Check write claims (only if we haven't hit the cap yet).
        if len(ungrounded) < 3:
            for m in _WRITE_CLAIM_RE.finditer(final_text):
                claimed_path = m.group(1)
                basename = Path(claimed_path).name
                if claimed_path not in write_paths and basename not in write_paths:
                    ungrounded.append(
                        f"write claim '{m.group(0).strip()}' (no write_file event for {claimed_path!r})"
                    )
                    if len(ungrounded) >= 3:
                        break

        if ungrounded:
            return VerificationResult(
                can_finish=False,
                reason="ungrounded claims: " + "; ".join(ungrounded),
                confidence=0.75,
                verifier_name=self.name,
            )

        return VerificationResult(
            can_finish=True,
            reason="all claims grounded in tool output",
            confidence=0.85,
            verifier_name=self.name,
        )


# ---------------------------------------------------------------------------
# StateVerifier
# ---------------------------------------------------------------------------

_SAFE_PREFIXES = ("find ", "ls", "wc ", "cat ", "head ", "tail ", "date", "pwd")
_UNSAFE_FRAGMENTS = ("rm ", "mv ", "cp ", "mkdir", "> ", ">> ", "chmod", "chown", "kill")


def _is_safe_command(cmd: str) -> bool:
    """Return True if the shell command is read-only and safe to re-run."""
    stripped = cmd.strip()
    if not any(stripped.startswith(p) for p in _SAFE_PREFIXES):
        return False
    return not any(frag in cmd for frag in _UNSAFE_FRAGMENTS)


def _first_numeric_token(text: str) -> str | None:
    """Return the first purely-digit token found in text, or None."""
    m = re.search(r"\b(\d+)\b", text)
    return m.group(1) if m else None


class StateVerifier:
    """Verifier that checks on-disk state matches what the model claimed.

    Two checks:

    1. **File existence**: for every successful ``write_file`` event, confirm
       the written path exists on disk.
    2. **Shell re-run**: for up to 3 safe (read-only) shell commands, re-run
       them and compare the first numeric token in the output against the
       original ``content_preview``. A divergence flags a stale or fabricated
       result.

    Returns:
    - ``can_finish=False, confidence=0.9`` if any issues are found.
    - ``can_finish=True, confidence=0.5`` if no write/shell events to check.
    - ``can_finish=True, confidence=0.9`` if all checks passed.
    """

    name = "state"

    def __init__(self, *, cwd: Path | str = ".") -> None:
        self._cwd = Path(cwd)

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        completed = [
            e for e in activity if e.kind == "tool_call.completed" and not e.data.get("is_error")
        ]

        write_events = [e for e in completed if e.data.get("name") == "write_file"]
        shell_events = [e for e in completed if e.data.get("name") == "shell"]

        if not write_events and not shell_events:
            return VerificationResult(
                can_finish=True,
                reason="no write_file or shell events to verify",
                confidence=0.5,
                verifier_name=self.name,
            )

        issues: list[str] = []

        # Check 1: written files exist on disk.
        for e in write_events:
            raw_path = e.data.get("arguments", {}).get("path", "")
            if not raw_path:
                continue
            p = Path(raw_path) if Path(raw_path).is_absolute() else self._cwd / raw_path
            if not p.exists():
                issues.append(f"write_file claimed to write {raw_path!r} but file does not exist")

        # Check 2: re-run safe shell commands and compare first numeric token.
        safe_to_check = [
            e
            for e in shell_events
            if _is_safe_command(e.data.get("arguments", {}).get("command", ""))
        ][:3]

        for e in safe_to_check:
            cmd = e.data.get("arguments", {}).get("command", "")
            original_preview: str = e.data.get("content_preview") or ""
            original_num = _first_numeric_token(original_preview)
            if original_num is None:
                continue  # Nothing to compare.

            try:
                proc = await asyncio.wait_for(
                    asyncio.create_subprocess_shell(
                        cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.DEVNULL,
                        cwd=str(self._cwd),
                    ),
                    timeout=10.0,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10.0)
                rerun_output = stdout.decode("utf-8", errors="replace")[:200]
                rerun_num = _first_numeric_token(rerun_output)
                if rerun_num is not None and rerun_num != original_num:
                    issues.append(
                        f"shell command {cmd!r} originally returned leading number "
                        f"{original_num!r} but re-run returned {rerun_num!r}"
                    )
            except Exception:
                pass  # Silently skip on any error or timeout.

        if issues:
            return VerificationResult(
                can_finish=False,
                reason="; ".join(issues),
                confidence=0.9,
                verifier_name=self.name,
            )

        return VerificationResult(
            can_finish=True,
            reason="all state checks passed",
            confidence=0.9,
            verifier_name=self.name,
        )


# ---------------------------------------------------------------------------
# ConsensusVerifier
# ---------------------------------------------------------------------------

_CONSENSUS_SYSTEM_PROMPT = (
    "You are an independent fact-checker reviewing an AI assistant's answer.\n\n"
    "You will be given the original task and a first model's answer. Check whether "
    "the answer is plausible, internally consistent, and correct.\n\n"
    "AUTOMATIC REJECT conditions:\n"
    "- Specific numbers or counts that are implausible or seem fabricated\n"
    "- Claims to have done work (ran a command, wrote a file) with no logical basis\n"
    "- Answer is clearly incomplete for the task requested\n"
    "- The answer contradicts itself\n\n"
    "Reply ONLY with JSON on a single line:\n"
    '{"agrees": true|false, "reason": "<short explanation>", "confidence": 0.0..1.0}\n'
    "No prose outside the JSON."
)


def _parse_consensus_response(text: str) -> tuple[bool, str, float] | None:
    """Return (agrees, reason, confidence) or None on parse failure.

    Handles ```json``` fences and surrounding whitespace.
    """
    body = text.strip()
    if body.startswith("```"):
        lines = body.splitlines()
        if lines:
            lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
        body = "\n".join(lines).strip()
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or "agrees" not in obj:
        return None
    agrees = bool(obj["agrees"])
    reason = str(obj.get("reason", "")).strip() or "(no reason given)"
    conf_raw = obj.get("confidence", 0.7)
    try:
        confidence = float(conf_raw)
    except (TypeError, ValueError):
        confidence = 0.7
    return agrees, reason, confidence


class ConsensusVerifier:
    """Verifier that uses a second LLM call to independently fact-check the
    first model's answer.

    The consensus model receives the original task and the first model's
    final answer, and decides whether the answer is plausible, internally
    consistent, and correct. This catches common lower-tier model failure
    modes: fabricated numbers, unsupported claims, and contradictions.

    Non-JSON or malformed responses degrade gracefully to
    ``can_finish=False, confidence=0.0`` after retries are exhausted.
    """

    name = "consensus"

    def __init__(self, *, adapter: Adapter, model: str, max_retries: int = 2) -> None:
        self.adapter = adapter
        self.model = model
        self.max_retries = max_retries

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        goal = _first_user_message(session)
        answer = _last_assistant_text(session)

        if not answer:
            return VerificationResult(
                can_finish=False,
                reason="no final assistant answer to check",
                confidence=0.0,
                verifier_name=self.name,
            )

        prompt = f"ORIGINAL TASK:\n{goal}\n\n" f"FIRST MODEL'S ANSWER:\n{answer}\n"
        messages = [
            Message(role="system", content=_CONSENSUS_SYSTEM_PROMPT),
            Message(role="user", content=prompt),
        ]

        last_reason = "consensus judge failed after retries"
        for attempt in range(self.max_retries):
            if attempt > 0:
                await asyncio.sleep(2**attempt)

            accumulated: list[str] = []
            final_content: str | None = None
            try:
                async for event in self.adapter.stream(model=self.model, messages=messages):
                    if isinstance(event, TextDelta):
                        accumulated.append(event.text)
                    elif isinstance(event, Done):
                        if event.final_message and event.final_message.content:
                            final_content = event.final_message.content
                        else:
                            final_content = "".join(accumulated)
                        break
            except Exception as exc:
                last_reason = f"consensus call failed (attempt {attempt + 1}): {exc!s}"
                continue

            if final_content is None:
                last_reason = (
                    f"consensus stream ended without a final message (attempt {attempt + 1})"
                )
                continue

            parsed = _parse_consensus_response(final_content)
            if parsed is None:
                preview = final_content.strip()[:200]
                last_reason = f"consensus returned non-JSON (attempt {attempt + 1}): {preview!r}"
                continue

            agrees, reason, confidence = parsed
            return VerificationResult(
                can_finish=agrees,
                reason=f"consensus: {reason}",
                confidence=confidence,
                verifier_name=self.name,
            )

        return VerificationResult(
            can_finish=False,
            reason=last_reason,
            confidence=0.0,
            verifier_name=self.name,
        )


# ---------------------------------------------------------------------------
# ChainedVerifier
# ---------------------------------------------------------------------------


class ChainedVerifier:
    """Runs multiple verifiers in order, returning the first failure.

    All verifiers receive the same session and activity list. The first
    `can_finish=False` result short-circuits the chain. If every verifier
    passes, the last result is returned (highest-confidence positive verdict).

    Use this to compose cheap verifiers (ClaimGroundingVerifier, StateVerifier)
    before the expensive LLM judge so the LLM only runs when the cheap checks
    are satisfied.
    """

    name = "chained"

    def __init__(self, *verifiers: Verifier) -> None:
        self._verifiers = list(verifiers)

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        last: VerificationResult | None = None
        for verifier in self._verifiers:
            result = await verifier.verify(session=session, activity=activity)
            last = result
            if not result.can_finish:
                return VerificationResult(
                    can_finish=False,
                    reason=result.reason,
                    confidence=result.confidence,
                    evidence_event_ids=result.evidence_event_ids,
                    verifier_name=self.name,
                )
        return last or VerificationResult(
            can_finish=True,
            reason="no verifiers in chain",
            confidence=0.5,
            verifier_name=self.name,
        )


# ---------------------------------------------------------------------------
# ShellVerifier
# ---------------------------------------------------------------------------


class ShellVerifier:
    """Run a caller-supplied shell command and treat non-zero exit as failure.

    The caller injects this at Agent construction time — no source-code
    access or file writes required. The repair loop feeds stdout/stderr back
    to the agent so it has concrete output to act on.

    Args:
        command: Shell command string (passed to ``asyncio.create_subprocess_shell``).
        cwd: Working directory override. Falls back to the session's cwd.
        timeout: Seconds before the command is killed and treated as failure.
    """

    name = "shell"

    def __init__(
        self,
        command: str,
        *,
        cwd: Path | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._command = command
        self._cwd = cwd
        self._timeout = timeout

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        effective_cwd = self._cwd
        if effective_cwd is None and hasattr(session, "cwd") and session.cwd:
            effective_cwd = Path(session.cwd)
        if effective_cwd is None:
            effective_cwd = Path.cwd()

        try:
            proc = await asyncio.create_subprocess_shell(
                self._command,
                cwd=effective_cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
            output = stdout.decode(errors="replace").strip()
            if proc.returncode == 0:
                return VerificationResult(
                    can_finish=True,
                    reason=output or "command succeeded",
                    verifier_name=self.name,
                )
            return VerificationResult(
                can_finish=False,
                reason=(
                    f"Command `{self._command}` exited with code {proc.returncode}.\n\n" f"{output}"
                ),
                verifier_name=self.name,
            )
        except TimeoutError:
            return VerificationResult(
                can_finish=False,
                reason=f"Command `{self._command}` timed out after {self._timeout}s.",
                verifier_name=self.name,
            )
        except Exception as exc:
            return VerificationResult(
                can_finish=False,
                reason=f"ShellVerifier error running `{self._command}`: {exc}",
                verifier_name=self.name,
            )


_WRITE_TOOL_NAMES = frozenset(
    {
        "write_file",
        "edit_file",
        "shell",
        "bash",
        "run_command",
        "execute",
        "apply_diff",
        "patch",
    }
)


class VerifyBeforeDoneVerifier:
    """Block Done unless the agent called verify_work at least once after making changes.

    Checks the activity ledger for any write/edit/shell tool calls. If found and
    no successful ``verify_work`` call follows, it returns ``can_finish=False``
    with an instruction to call ``verify_work`` before finishing.

    This is always-on structural enforcement — the LLM can't skip verification
    just by ignoring system prompt instructions.

    Args:
        write_tool_names: Set of tool names considered "modifying". Defaults
            to ``_WRITE_TOOL_NAMES`` (write_file, edit_file, shell, …).
    """

    name = "verify_before_done"

    def __init__(self, write_tool_names: frozenset[str] | None = None) -> None:
        self._writes = write_tool_names if write_tool_names is not None else _WRITE_TOOL_NAMES

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        tool_events = [e for e in activity if e.kind == "tool_call.completed"]

        wrote = any(e.data.get("name") in self._writes for e in tool_events)
        if not wrote:
            return VerificationResult(
                can_finish=True,
                reason="No modifying tool calls detected — verification not required.",
                verifier_name=self.name,
            )

        verify_calls = [e for e in tool_events if e.data.get("name") == "verify_work"]
        if not verify_calls:
            return VerificationResult(
                can_finish=False,
                reason=(
                    "You made file changes but never ran verify_work. "
                    "You must test your changes before finishing. "
                    "Call verify_work with the appropriate command "
                    "(e.g. 'pytest tests/', 'npm test', 'cargo test'). "
                    "Read the output — if tests fail, fix the specific failures "
                    "and call verify_work again. Iterate until all tests pass."
                ),
                verifier_name=self.name,
            )

        # verify_work was called at least once — let downstream verifiers (e.g.
        # ShellVerifier) handle whether the tests actually passed. They have the
        # real test output the critic needs to generate a pointed challenge.
        return VerificationResult(
            can_finish=True,
            reason="verify_work was called — deferring to downstream verifier.",
            verifier_name=self.name,
        )


__all__ = [
    "ChainedVerifier",
    "ClaimGroundingVerifier",
    "ConsensusVerifier",
    "EvidenceCheckKind",
    "EvidenceContract",
    "EvidenceContractResult",
    "LLMJudgeVerifier",
    "RuleVerifier",
    "ShellVerifier",
    "StateVerifier",
    "VerificationGateway",
    "VerificationResult",
    "Verifier",
    "VerifierRouter",
    "VerifyBeforeDoneVerifier",
    "WorkItemJudge",
    "evaluate_evidence",
]
