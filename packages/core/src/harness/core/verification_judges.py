"""Judge-oriented verification logic."""

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
    name: str

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult: ...


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


def _is_repetitive(text: str, *, window: int = 40, threshold: int = 5) -> bool:
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
    name = "rule"

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        final_text = _last_assistant_text(session)

        if _is_repetitive(final_text):
            return VerificationResult(
                can_finish=False,
                reason="model output is a stall loop — same content repeated 4+ times",
                confidence=0.9,
                verifier_name=self.name,
            )

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


class VerifierRouter:
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
    if not isinstance(obj, dict) or "can_finish" not in obj:
        return None
    can_finish = bool(obj["can_finish"])
    reason = str(obj.get("reason", "")).strip() or "(no reason given)"
    conf_raw = obj.get("confidence")
    try:
        confidence = float(conf_raw) if conf_raw is not None else None
    except (TypeError, ValueError):
        confidence = None
    return can_finish, reason, confidence


__all__ = [
    "LLMJudgeVerifier",
    "RuleVerifier",
    "Verifier",
    "VerifierRouter",
    "WorkItemJudge",
    "asyncio",
    "_first_user_message",
    "_is_repetitive",
    "_last_assistant_text",
]
