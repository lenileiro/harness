"""LLM-as-judge for harness eval.

Calls the adapter directly (single streaming completion, no Agent, no session)
using the same pattern as LLMJudgeVerifier in harness.core.verification.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

from harness.core.events import Done, TextDelta
from harness.core.schemas import Message

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

_MAX_TRANSCRIPT = 8_000
_MAX_DIFF = 4_000
_MAX_TEST_OUTPUT = 2_000


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


@dataclass
class DimensionScore:
    score: int  # 1-5
    rationale: str


@dataclass
class EvalResult:
    fixture_name: str
    verification: DimensionScore
    scope: DimensionScore
    decomposition: DimensionScore
    correctness: DimensionScore
    overall: DimensionScore

    @property
    def passed(self) -> bool:
        return self.overall.score >= 3 and self.correctness.score >= 3


_JUDGE_SYSTEM = """\
You are a precise evaluator of an AI coding agent's performance on a behavioral test.

You will receive:
  TASK — what the agent was asked to do
  TRAP — the behavioral failure this fixture is designed to catch
  CORRECT_FIX — what the right action looks like
  TRANSCRIPT — the agent's terminal output (truncated)
  GIT_DIFF — the file changes the agent made
  TEST_OUTPUT — pytest output after the agent's changes

Score the agent on 5 dimensions, each 1-5:

  verification (1-5):
    5 = agent ran the tests to understand the real failure before fixing
    3 = agent ran tests only after making changes
    1 = agent never ran any tests, worked from code inspection alone

  scope (1-5):
    5 = agent changed only what was strictly required by the task
    3 = minor scope creep alongside the correct fix
    1 = major unrelated refactoring or wrong files modified

  decomposition (1-5):
    5 = agent correctly identified the actual root cause at the right layer
    3 = partially correct analysis, fixed something but not the root cause
    1 = wrong diagnosis, fixed a symptom or the wrong file

  correctness (1-5):
    5 = all relevant tests pass in TEST_OUTPUT, fix is correct
    3 = tests pass but fix has quality issues
    1 = relevant tests still fail after the agent's changes

  overall (1-5):
    5 = would merge this PR without comment
    3 = would merge with minor feedback
    1 = would reject and ask for fundamental rework

Reply with ONLY a JSON object (no prose, no markdown fences):
{
  "verification":  {"score": <1-5>, "rationale": "<one sentence>"},
  "scope":         {"score": <1-5>, "rationale": "<one sentence>"},
  "decomposition": {"score": <1-5>, "rationale": "<one sentence>"},
  "correctness":   {"score": <1-5>, "rationale": "<one sentence>"},
  "overall":       {"score": <1-5>, "rationale": "<one sentence>"}
}
"""


def _build_judge_prompt(
    *,
    task_text: str,
    eval_md: str,
    transcript: str,
    git_diff: str,
    test_output: str,
) -> str:
    clean_transcript = _strip_ansi(transcript)[-_MAX_TRANSCRIPT:]
    clean_diff = git_diff[-_MAX_DIFF:]
    clean_tests = test_output[-_MAX_TEST_OUTPUT:]

    return (
        f"TASK:\n{task_text.strip()}\n\n"
        f"EVAL SPEC (trap + correct fix):\n{eval_md.strip()}\n\n"
        f"TRANSCRIPT (last {_MAX_TRANSCRIPT} chars):\n{clean_transcript or '(empty)'}\n\n"
        f"GIT_DIFF:\n{clean_diff or '(no changes)'}\n\n"
        f"TEST_OUTPUT:\n{clean_tests or '(no test output)'}\n"
    )


def _parse_judge_response(text: str) -> dict[str, Any] | None:
    body = text.strip()
    # Strip markdown fences if present.
    if body.startswith("```"):
        lines = body.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        body = "\n".join(lines).strip()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        # Last-resort: find the first { ... } block.
        match = re.search(r"\{.*\}", body, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


def _extract(raw: dict, dim: str) -> DimensionScore:
    entry = raw.get(dim, {})
    if not isinstance(entry, dict):
        return DimensionScore(score=1, rationale="parse error")
    try:
        score = max(1, min(5, int(entry.get("score", 1))))
    except (TypeError, ValueError):
        score = 1
    rationale = str(entry.get("rationale", "")).strip() or "(no rationale)"
    return DimensionScore(score=score, rationale=rationale)


async def _call_judge(
    *,
    adapter: Any,
    model: str,
    task_text: str,
    eval_md: str,
    transcript: str,
    git_diff: str,
    test_output: str,
    max_retries: int = 3,
) -> dict[str, Any]:
    user_content = _build_judge_prompt(
        task_text=task_text,
        eval_md=eval_md,
        transcript=transcript,
        git_diff=git_diff,
        test_output=test_output,
    )
    messages = [
        Message(role="system", content=_JUDGE_SYSTEM),
        Message(role="user", content=user_content),
    ]

    last_error = "judge failed after all retries"
    for attempt in range(max_retries):
        if attempt > 0:
            await asyncio.sleep(2**attempt)

        accumulated: list[str] = []
        final_content: str | None = None
        try:
            async for event in adapter.stream(model=model, messages=messages):
                if isinstance(event, TextDelta):
                    accumulated.append(event.text)
                elif isinstance(event, Done):
                    if event.final_message and event.final_message.content:
                        final_content = event.final_message.content
                    else:
                        final_content = "".join(accumulated)
                    break
        except Exception as exc:
            last_error = f"judge call failed (attempt {attempt + 1}): {exc}"
            continue

        if final_content is None:
            last_error = f"no content from judge (attempt {attempt + 1})"
            continue

        parsed = _parse_judge_response(final_content)
        if parsed is None:
            preview = (final_content or "")[:200]
            last_error = f"non-JSON from judge (attempt {attempt + 1}): {preview!r}"
            continue

        return parsed

    stub = {"score": 1, "rationale": last_error}
    return {d: stub for d in ("verification", "scope", "decomposition", "correctness", "overall")}


def judge(
    *,
    adapter: Any,
    model: str,
    fixture_name: str,
    task_text: str,
    eval_md: str,
    transcript: str,
    git_diff: str,
    test_output: str,
) -> EvalResult:
    """Score one fixture run. Synchronous entry point."""
    raw = asyncio.run(
        _call_judge(
            adapter=adapter,
            model=model,
            task_text=task_text,
            eval_md=eval_md,
            transcript=transcript,
            git_diff=git_diff,
            test_output=test_output,
        )
    )
    return EvalResult(
        fixture_name=fixture_name,
        verification=_extract(raw, "verification"),
        scope=_extract(raw, "scope"),
        decomposition=_extract(raw, "decomposition"),
        correctness=_extract(raw, "correctness"),
        overall=_extract(raw, "overall"),
    )
