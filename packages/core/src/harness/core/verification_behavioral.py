"""Prompt/diff-sensitive behavioral verifiers extracted from verification.py."""

from __future__ import annotations

import asyncio
import re
import subprocess
from pathlib import Path

from harness.core import verification_structural as _structural
from harness.core.activity import ActivityEvent
from harness.core.schemas import Session, VerificationResult

_PROMPT_TOKEN_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
_BACKTICK_IDENTIFIER_RE = re.compile(r"`([A-Za-z_][A-Za-z0-9_]*)`")
_BUGFIX_LEAD_RE = re.compile(r"^(?:#\s*)?(fix|debug|handle|correct)\b", re.IGNORECASE)
_first_user_prompt = _structural.first_user_prompt


class MisdirectedSuggestionVerifier:
    """When tests now pass, flag edits that share no vocabulary with anything that ever failed."""

    name = "misdirected_suggestion"

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        from harness.core.test_signals import (
            extract_failing_test_names,
            keywords_for_test_names,
            text_overlap,
        )

        verify_events = [
            e
            for e in activity
            if e.kind == "tool_call.completed" and e.data.get("name") == "verify_work"
        ]
        if not verify_events:
            return VerificationResult(
                can_finish=True,
                reason="no verify_work calls — nothing to align edits against",
                confidence=0.4,
                verifier_name=self.name,
            )

        if verify_events[-1].data.get("is_error"):
            return VerificationResult(
                can_finish=True,
                reason=("latest verify_work failed — deferring to diagnosis_alignment"),
                confidence=0.4,
                verifier_name=self.name,
            )

        all_failing: list[str] = []
        for ev in verify_events:
            if not ev.data.get("is_error"):
                continue
            preview = str(ev.data.get("content_preview") or "")
            for name in extract_failing_test_names(preview):
                if name not in all_failing:
                    all_failing.append(name)

        if not all_failing:
            return VerificationResult(
                can_finish=True,
                reason="no historical failing tests recorded — nothing to align against",
                confidence=0.4,
                verifier_name=self.name,
            )

        test_keywords = keywords_for_test_names(all_failing)
        if not test_keywords:
            return VerificationResult(
                can_finish=True,
                reason="historical test names yielded no significant keywords",
                confidence=0.3,
                verifier_name=self.name,
            )

        user_prompt = _first_user_prompt(session)
        prompt_keywords = text_overlap(user_prompt, test_keywords)
        novel_test_keywords = test_keywords - prompt_keywords
        if not novel_test_keywords:
            return VerificationResult(
                can_finish=True,
                reason=(
                    "prompt already covers all historical failing-test "
                    "vocabulary — every edit is implicitly aligned"
                ),
                confidence=0.5,
                verifier_name=self.name,
            )

        aligned_count = 0
        unaligned_snippets: list[str] = []
        for ev in activity:
            if ev.kind != "tool_call.completed":
                continue
            if ev.data.get("is_error"):
                continue
            name = ev.data.get("name")
            if name not in ("write_file", "edit_file", "apply_diff", "patch"):
                continue
            args = ev.data.get("arguments") or {}
            path = args.get("path")
            if not isinstance(path, str):
                continue
            parts: list[str] = []
            for key in ("content", "new", "new_text", "diff"):
                val = args.get(key)
                if isinstance(val, str) and val:
                    parts.append(val)
            preview = ev.data.get("content_preview")
            if isinstance(preview, str) and preview:
                parts.append(preview)
            combined = "\n".join(parts).strip()
            if not combined:
                continue
            if text_overlap(combined, novel_test_keywords):
                aligned_count += 1
            else:
                snippet = combined.splitlines()[0][:80]
                unaligned_snippets.append(f"{path}: {snippet!r}")

        if not unaligned_snippets:
            return VerificationResult(
                can_finish=True,
                reason="every edit aligns with at least one historical failing-test keyword",
                confidence=0.85,
                verifier_name=self.name,
            )

        if aligned_count == 0:
            return VerificationResult(
                can_finish=True,
                reason=(
                    "tests pass but no edit addresses historical failure "
                    "vocabulary — abstaining (unclear pattern)"
                ),
                confidence=0.3,
                verifier_name=self.name,
            )

        unaligned_block = "\n".join(f"  - {s}" for s in unaligned_snippets[:5])
        return VerificationResult(
            can_finish=False,
            reason=(
                f"STOP — the diff contains unnecessary changes that you MUST "
                f"revert before finishing.\n\n"
                f"PROOF this is wrong: the failing tests were "
                f"{', '.join(all_failing[:3])}. "
                f"Their names point at concepts "
                f"{sorted(novel_test_keywords)[:5]} which appear nowhere in "
                f"the user's prompt — that means the user's literal instruction "
                f"was a MISDIAGNOSIS. The real bug is what the test names "
                f"describe, and your {aligned_count} aligned edit(s) already "
                f"fix it. Tests pass right now because of those aligned edits, "
                f"NOT because of the unaligned ones below.\n\n"
                f"REQUIRED ACTION: use edit_file to revert these unaligned "
                f"changes to their ORIGINAL values:\n"
                f"{unaligned_block}\n\n"
                f"After reverting, call verify_work again. The tests will "
                f"still pass — that is the proof these edits were unnecessary "
                f"scope creep. Do NOT keep both fixes 'just in case'; the user "
                f"prompt was wrong about the fix, and leaving the wrong fix in "
                f"the diff is a real defect. Following the prompt literally is "
                f"the wrong call here; the failing test names are the ground "
                f"truth, not the prompt's suggested action."
            ),
            confidence=0.85,
            verifier_name=self.name,
        )


def _prompt_signal_tokens(prompt: str) -> set[str]:
    tokens: set[str] = set()
    for raw in _BACKTICK_IDENTIFIER_RE.findall(prompt or ""):
        candidate = raw.strip()
        if not candidate:
            continue
        if "(" in candidate:
            candidate = candidate.split("(", 1)[0].strip()
        if not candidate:
            continue
        if "/" in candidate or "." in candidate:
            tokens.add(candidate.lower())
            continue
        if candidate.isupper():
            tokens.add(candidate.lower())
            continue
        if "_" in candidate:
            tokens.add(candidate.lower())
    for raw in _PROMPT_TOKEN_RE.findall(prompt or ""):
        if raw.isupper() or "_" in raw:
            lower = raw.lower()
            if len(raw) >= 4:
                tokens.add(lower)
    return tokens


def _git_diff_unified_zero(cwd: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "diff", "--unified=0", "--no-ext-diff", "--"],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return ""
    if result.returncode not in (0, 1):
        return ""
    return result.stdout


def _diff_changed_lines(diff_text: str) -> list[str]:
    return [
        line
        for line in diff_text.splitlines()
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]


class PromptSurfaceRevertVerifier:
    """Revert prompt-driven symptom edits once tests prove a different root cause."""

    name = "prompt_surface_revert"

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        from harness.core.test_signals import text_overlap

        verify_events = [
            e
            for e in activity
            if e.kind == "tool_call.completed" and e.data.get("name") == "verify_work"
        ]
        if not verify_events:
            return VerificationResult(
                can_finish=True,
                reason="no verify_work calls — nothing to validate against prompt drift",
                confidence=0.4,
                verifier_name=self.name,
            )

        if verify_events[-1].data.get("is_error"):
            return VerificationResult(
                can_finish=True,
                reason="latest verify_work failed — prompt-surface revert not applicable yet",
                confidence=0.4,
                verifier_name=self.name,
            )

        historical_failures = [e for e in verify_events if e.data.get("is_error")]
        if not historical_failures:
            return VerificationResult(
                can_finish=True,
                reason="tests never failed during the run — no disproven prompt fix to revert",
                confidence=0.5,
                verifier_name=self.name,
            )

        user_prompt = _first_user_prompt(session)
        prompt_surface_keywords = _prompt_signal_tokens(user_prompt)
        if not prompt_surface_keywords:
            return VerificationResult(
                can_finish=True,
                reason="prompt names no concrete surface identifiers to protect",
                confidence=0.5,
                verifier_name=self.name,
            )

        first_failed_verify_idx = next(
            (
                idx
                for idx, ev in enumerate(activity)
                if ev.kind == "tool_call.completed"
                and ev.data.get("name") == "verify_work"
                and ev.data.get("is_error")
            ),
            None,
        )

        later_writes = 0
        if first_failed_verify_idx is not None:
            for ev in activity[first_failed_verify_idx + 1 :]:
                if ev.kind != "tool_call.completed" or ev.data.get("is_error"):
                    continue
                if ev.data.get("name") in ("write_file", "edit_file", "apply_diff", "patch"):
                    later_writes += 1
        if later_writes == 0:
            return VerificationResult(
                can_finish=True,
                reason="prompt-surface edit was not followed by any later repair writes",
                confidence=0.6,
                verifier_name=self.name,
            )

        diff_text = await asyncio.to_thread(_git_diff_unified_zero, session.cwd)
        if not diff_text.strip():
            return VerificationResult(
                can_finish=True,
                reason="git diff is empty — no lingering prompt-surface edits remain",
                confidence=0.8,
                verifier_name=self.name,
            )

        changed_lines = "\n".join(
            line[1:]
            for line in diff_text.splitlines()
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        )
        if not changed_lines.strip():
            return VerificationResult(
                can_finish=True,
                reason="git diff contained no changed lines after filtering headers",
                confidence=0.5,
                verifier_name=self.name,
            )

        lingering_prompt_keywords = text_overlap(changed_lines, prompt_surface_keywords)
        if not lingering_prompt_keywords:
            return VerificationResult(
                can_finish=True,
                reason="current diff no longer contains the prompt-surface edit",
                confidence=0.85,
                verifier_name=self.name,
            )

        return VerificationResult(
            can_finish=False,
            reason=(
                "STOP — tests now pass, but the current diff still contains the "
                "prompt's disproven symptom edit alongside later repair work. "
                f"There were {len(historical_failures)} failing verify_work run(s) before success, "
                f"and you made {later_writes} additional write(s) after the first failure. "
                "That means the literal prompt edit did not solve the problem on its own. "
                "The current diff still changes concrete prompt-surface identifiers "
                f"{sorted(lingering_prompt_keywords)[:5]}. "
                "That means you kept the user's suggested fix even after the test run "
                "proved it was the wrong layer. Revert the prompt-surface edit, keep "
                "the root-cause fix, and run verify_work again."
            ),
            confidence=0.9,
            verifier_name=self.name,
        )


class NegativeConstraintVerifier:
    """Enforce explicit "do not fix X" constraints from the prompt."""

    name = "negative_constraint"

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        verify_events = [
            e
            for e in activity
            if e.kind == "tool_call.completed" and e.data.get("name") == "verify_work"
        ]
        if not verify_events or verify_events[-1].data.get("is_error"):
            return VerificationResult(
                can_finish=True,
                reason="negative constraints deferred until verify_work passes",
                confidence=0.4,
                verifier_name=self.name,
            )

        prompt = _first_user_prompt(session).lower()
        guard_formatting = "inconsistent formatting" in prompt or "comment style" in prompt
        guard_imports = "unused import" in prompt or "unused imports" in prompt
        if not guard_formatting and not guard_imports:
            return VerificationResult(
                can_finish=True,
                reason="no explicit formatting/import negative constraints in prompt",
                confidence=0.5,
                verifier_name=self.name,
            )

        diff_text = await asyncio.to_thread(_git_diff_unified_zero, session.cwd)
        if not diff_text.strip():
            return VerificationResult(
                can_finish=True,
                reason="git diff is empty — no negative-constraint violations remain",
                confidence=0.8,
                verifier_name=self.name,
            )

        changed_lines = _diff_changed_lines(diff_text)
        violations: list[str] = []
        if guard_formatting:
            comment_changes = [
                line[1:].strip() for line in changed_lines if line[1:].lstrip().startswith("#")
            ]
            if comment_changes:
                preview = ", ".join(repr(line[:60]) for line in comment_changes[:3])
                violations.append(f"comment-style changes: {preview}")

        if guard_imports:
            import_changes = [
                line[1:].strip()
                for line in changed_lines
                if line[1:].lstrip().startswith(("import ", "from "))
            ]
            if import_changes:
                preview = ", ".join(repr(line[:60]) for line in import_changes[:3])
                violations.append(f"import cleanup: {preview}")

        if not violations:
            return VerificationResult(
                can_finish=True,
                reason="no explicit negative-constraint violations detected in diff",
                confidence=0.85,
                verifier_name=self.name,
            )

        return VerificationResult(
            can_finish=False,
            reason=(
                "STOP — the prompt explicitly said not to fix pre-existing cleanup "
                "issues, but the current diff still includes unrelated cleanup "
                f"changes ({'; '.join(violations)}). "
                "If the added lines are test comment banners, delete only those new `# ...` "
                "lines and keep the code/test body you added. Revert those comment/import "
                "edits, keep only the requested task changes, and run verify_work again."
            ),
            confidence=0.9,
            verifier_name=self.name,
        )


class BugfixCommentRewriteVerifier:
    """Block new source-comment additions on narrow bugfix prompts."""

    name = "bugfix_comment_rewrite"

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        first_line = next(
            (line.strip() for line in _first_user_prompt(session).splitlines() if line.strip()),
            "",
        )
        if not _BUGFIX_LEAD_RE.match(first_line):
            return VerificationResult(
                can_finish=True,
                reason="prompt is not a narrow bugfix request",
                confidence=0.4,
                verifier_name=self.name,
            )
        prompt_lower = _first_user_prompt(session).lower()
        if any(token in prompt_lower for token in ("readme", "docstring", "docs", "comment")):
            return VerificationResult(
                can_finish=True,
                reason="prompt explicitly mentions docs/comments; skipping comment rewrite guard",
                confidence=0.4,
                verifier_name=self.name,
            )
        diff_text = await asyncio.to_thread(_git_diff_unified_zero, session.cwd)
        if not diff_text.strip():
            return VerificationResult(
                can_finish=True,
                reason="git diff is empty — no comment rewrite detected",
                confidence=0.8,
                verifier_name=self.name,
            )
        offending: list[str] = []
        current_file: str | None = None
        for line in diff_text.splitlines():
            if line.startswith("diff --git "):
                parts = line.split()
                if len(parts) >= 4 and parts[3].startswith("b/"):
                    current_file = parts[3][2:]
                else:
                    current_file = None
                continue
            if line.startswith("+++ b/"):
                current_file = line[6:]
                continue
            if not current_file or not current_file.startswith("src/"):
                continue
            if line.startswith("+") and not line.startswith("+++"):
                content = line[1:].lstrip()
                if content.startswith("#"):
                    offending.append(f"{current_file}: {content[:80]}")
        if not offending:
            return VerificationResult(
                can_finish=True,
                reason="no new source comment lines added for this bugfix",
                confidence=0.8,
                verifier_name=self.name,
            )
        preview = ", ".join(offending[:3])
        return VerificationResult(
            can_finish=False,
            reason=(
                "STOP — this bugfix prompt did not ask for source comment updates, "
                f"but the current diff adds new source comment lines ({preview}). "
                "Remove the explanatory comment rewrite and keep only the code change."
            ),
            confidence=0.9,
            verifier_name=self.name,
        )


class DiagnosisAlignmentVerifier:
    """Block runs where the agent's edits don't share vocabulary with the failing tests."""

    name = "diagnosis_alignment"

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        from harness.core.test_signals import (
            extract_failing_test_names,
            keywords_for_test_names,
            text_overlap,
        )

        verify_events = [
            e
            for e in activity
            if e.kind == "tool_call.completed" and e.data.get("name") == "verify_work"
        ]
        if not verify_events:
            return VerificationResult(
                can_finish=True,
                reason="no verify_work calls — nothing to align edits against",
                confidence=0.4,
                verifier_name=self.name,
            )

        last_verify = verify_events[-1]
        if not last_verify.data.get("is_error"):
            return VerificationResult(
                can_finish=True,
                reason="latest verify_work succeeded — alignment trivially satisfied",
                confidence=0.8,
                verifier_name=self.name,
            )

        test_output = str(last_verify.data.get("content_preview") or "")
        failing_tests = extract_failing_test_names(test_output)
        if not failing_tests:
            return VerificationResult(
                can_finish=True,
                reason="could not parse failing test names — abstaining",
                confidence=0.3,
                verifier_name=self.name,
            )

        test_keywords = keywords_for_test_names(failing_tests)
        if not test_keywords:
            return VerificationResult(
                can_finish=True,
                reason="failing test names had no significant keywords",
                confidence=0.3,
                verifier_name=self.name,
            )

        user_prompt = _first_user_prompt(session)
        prompt_keywords = text_overlap(user_prompt, test_keywords)
        novel_test_keywords = test_keywords - prompt_keywords
        if not novel_test_keywords:
            return VerificationResult(
                can_finish=True,
                reason=(
                    "user prompt already mentions every failing-test keyword "
                    f"({sorted(test_keywords)[:4]}) — no novel-vocabulary "
                    "alignment to enforce"
                ),
                confidence=0.5,
                verifier_name=self.name,
            )

        edit_content_parts: list[str] = []
        edit_paths: set[str] = set()
        for ev in activity:
            if ev.kind != "tool_call.completed":
                continue
            if ev.data.get("is_error"):
                continue
            name = ev.data.get("name")
            if name not in ("write_file", "edit_file", "apply_diff", "patch"):
                continue
            args = ev.data.get("arguments") or {}
            path = args.get("path")
            if isinstance(path, str):
                edit_paths.add(path)
            for key in ("content", "new", "new_text", "diff"):
                val = args.get(key)
                if isinstance(val, str) and val:
                    edit_content_parts.append(val)
            preview = ev.data.get("content_preview")
            if isinstance(preview, str) and preview:
                edit_content_parts.append(preview)

        if not edit_content_parts:
            return VerificationResult(
                can_finish=True,
                reason="no edit content recorded — nothing to align",
                confidence=0.4,
                verifier_name=self.name,
            )

        combined = "\n".join(edit_content_parts)
        overlap = text_overlap(combined, novel_test_keywords)

        if overlap:
            return VerificationResult(
                can_finish=True,
                reason=(
                    f"edits address novel failing-test keywords (not in prompt): "
                    f"{sorted(overlap)[:4]}"
                ),
                confidence=0.85,
                verifier_name=self.name,
            )

        return VerificationResult(
            can_finish=False,
            reason=(
                f"Tests are still failing: {', '.join(failing_tests[:3])}. "
                f"Those test names point at concepts the user's prompt didn't "
                f"mention: {sorted(novel_test_keywords)[:5]}. Your edits to "
                f"{sorted(edit_paths)[:3]} don't contain any of those words. "
                f"The prompt likely pointed at a symptom — re-read the failing "
                f"test names, look at what they actually assert, and edit the "
                f"code that implements (or fails to implement) those concepts. "
                f"Don't just follow the prompt literally."
            ),
            confidence=0.85,
            verifier_name=self.name,
        )
