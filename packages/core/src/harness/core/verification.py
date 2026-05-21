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
from typing import Protocol, runtime_checkable

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


class RuleVerifier:
    """Real rule-based verifier — no LLM call.

    v1 rule: any `tool_call.completed` event with `is_error=True` means the
    run did not finish cleanly. The failing event ids are returned as
    `evidence_event_ids` so consumers can drill in via `harness evidence
    list`.

    Empty activity (no tools ran) is treated as `can_finish=True` with a
    "no tools dispatched" reason — the model produced a text-only answer
    and there's nothing rule-based to invalidate.
    """

    name = "rule"

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
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
    "accomplished the goal.\n\n"
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


__all__ = [
    "LLMJudgeVerifier",
    "RuleVerifier",
    "VerificationResult",
    "Verifier",
    "VerifierRouter",
]
