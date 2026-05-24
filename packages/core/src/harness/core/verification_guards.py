"""Evidence, grounding, state, and consensus verification."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from harness.core.activity import ActivityEvent
from harness.core.adapter import Adapter
from harness.core.events import Done, TextDelta
from harness.core.schemas import Message, Session, VerificationResult
from harness.core.verification_judges import (
    Verifier,
    _first_user_message,
    _last_assistant_text,
)

EvidenceCheckKind = Literal[
    "command_evidence",
    "changed_file",
    "acceptance_criterion",
    "no_prediction_errors",
    "tool_succeeded",
]


class EvidenceContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_checks: list[EvidenceCheckKind]
    check_data: dict[str, Any] = Field(default_factory=dict)


class EvidenceContractResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    satisfied: bool
    found_checks: list[str]
    missing_checks: list[str]


_PREDICTION_ERROR_SEVERITIES = frozenset({"medium", "high", "critical"})


def evaluate_evidence(
    contract: EvidenceContract,
    activity: list[ActivityEvent],
) -> EvidenceContractResult:
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

        if self._contract is not None:
            contract_result = evaluate_evidence(self._contract, activity)
            if not contract_result.satisfied:
                return VerificationResult(
                    can_finish=False,
                    reason=(
                        "evidence contract not satisfied — "
                        f"missing: {', '.join(contract_result.missing_checks)}"
                    ),
                    confidence=0.9,
                    verifier_name=self.name,
                )

        return await self._verifier.verify(session=session, activity=activity)


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
        corpus = " ".join(str(e.data.get("content_preview") or "") for e in completed)

        write_paths: set[str] = set()
        for e in completed:
            if e.data.get("name") == "write_file":
                path = e.data.get("arguments", {}).get("path", "")
                if path:
                    write_paths.add(path)
                    write_paths.add(Path(path).name)

        ungrounded: list[str] = []

        for m in _COUNT_CLAIM_RE.finditer(final_text):
            number = m.group(1)
            if number not in corpus:
                ungrounded.append(
                    f"count claim '{m.group(0).strip()}' (number {number} not found in tool output)"
                )
                if len(ungrounded) >= 3:
                    break

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


_SAFE_PREFIXES = ("find ", "ls", "wc ", "cat ", "head ", "tail ", "date", "pwd")
_UNSAFE_FRAGMENTS = ("rm ", "mv ", "cp ", "mkdir", "> ", ">> ", "chmod", "chown", "kill")


def _is_safe_command(cmd: str) -> bool:
    stripped = cmd.strip()
    if not any(stripped.startswith(p) for p in _SAFE_PREFIXES):
        return False
    return not any(frag in cmd for frag in _UNSAFE_FRAGMENTS)


def _first_numeric_token(text: str) -> str | None:
    m = re.search(r"\b(\d+)\b", text)
    return m.group(1) if m else None


class StateVerifier:
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

        for e in write_events:
            raw_path = e.data.get("arguments", {}).get("path", "")
            if not raw_path:
                continue
            p = Path(raw_path) if Path(raw_path).is_absolute() else self._cwd / raw_path
            if not p.exists():
                issues.append(f"write_file claimed to write {raw_path!r} but file does not exist")

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
                continue

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
                pass

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

        prompt = f"ORIGINAL TASK:\n{goal}\n\nFIRST MODEL'S ANSWER:\n{answer}\n"
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


__all__ = [
    "ClaimGroundingVerifier",
    "ConsensusVerifier",
    "EvidenceCheckKind",
    "EvidenceContract",
    "EvidenceContractResult",
    "StateVerifier",
    "VerificationGateway",
    "evaluate_evidence",
]
