"""Deterministic structural verifiers extracted from verification.py."""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any

from harness.core.activity import ActivityEvent
from harness.core.schemas import Session, VerificationResult

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

_PROMOTION_ARTIFACT_PREFIX = ".harness/research/promotions/"
_PROMOTION_FLOW_COMMAND_HINTS: tuple[str, ...] = (
    "harness research create-candidate",
    "harness research candidate create",
    "harness research refine",
    "harness research promote",
    "harness research pr",
    "gh pr create",
    ".harness/research/promotions/",
)

_TEST_COMMAND_HINTS: tuple[str, ...] = (
    "pytest",
    "python -m pytest",
    "uv run pytest",
    "npm test",
    "pnpm test",
    "yarn test",
    "bun test",
    "cargo test",
    "go test",
    "jest",
    "vitest",
)

_TEST_OUTPUT_HINTS: tuple[str, ...] = (
    "test session starts",
    "collected ",
    " passed",
    " failed",
    "error:",
    "assertionerror",
)

_MINIMAL_HINTS_RE = re.compile(
    r"\b(minimal fix|don't (refactor|tackle|fix anything else)|only fix|just fix|"
    r"only modify|do not refactor|nothing else should change|no other changes|"
    r"minimal change|smallest fix)\b",
    re.IGNORECASE,
)

_FILE_PATH_RE = re.compile(
    r"`([^`\n]+?\.(?:py|ts|tsx|js|jsx|go|rs|java|kt|rb|php|c|cc|cpp|h|hpp|md|json|toml|yaml|yml|sql|sh|cfg|ini|html|css|tf|hcl))`"
    r"|"
    r"`([^`\n]*?/[^`\n]+?)`"
)

_FUNCTION_CALL_RE = re.compile(r"`[A-Za-z_][A-Za-z0-9_]*\([^`\n]*\)`")


def looks_like_feature_add(prompt: str) -> bool:
    if not prompt:
        return False
    header = ""
    for line in prompt.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            header = stripped.lower()
            break
    if not header:
        return False
    feature_verbs = ("add ", "implement ", "create ", "support ", "introduce ")
    bug_verbs = ("fix ", "debug ", "handle ", "resolve ", "repair ", "patch ")
    starts_with_feature = any(header.startswith(v) for v in feature_verbs)
    starts_with_bug = any(header.startswith(v) for v in bug_verbs)
    return starts_with_feature and not starts_with_bug


def looks_like_test_invocation(event: ActivityEvent) -> bool:
    if event.kind != "tool_call.completed":
        return False
    name = str(event.data.get("name") or "")
    if name == "verify_work":
        return True
    if name not in {"shell", "bash", "run_command"}:
        return False

    arguments = event.data.get("arguments")
    command = ""
    if isinstance(arguments, dict):
        command = str(
            arguments.get("command") or arguments.get("cmd") or arguments.get("text") or ""
        )
    command_lower = command.lower()
    if any(hint in command_lower for hint in _TEST_COMMAND_HINTS):
        return True

    preview = str(event.data.get("content_preview") or "").lower()
    return any(hint in preview for hint in _TEST_OUTPUT_HINTS)


def first_user_prompt(session: Session) -> str:
    for msg in session.messages:
        if getattr(msg, "role", None) == "user" and msg.content:
            return msg.content
    return ""


def _promotion_artifact_path(path: str) -> bool:
    normalized = path.strip().replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.startswith(_PROMOTION_ARTIFACT_PREFIX)


def _shell_command(event: ActivityEvent) -> str:
    arguments = event.data.get("arguments")
    if not isinstance(arguments, dict):
        return ""
    return str(arguments.get("command") or arguments.get("cmd") or arguments.get("text") or "")


def _normalize_shell_command(command: str) -> str:
    return " ".join(command.lower().split())


def _is_harness_research_pr_open_command(command: str) -> bool:
    normalized = _normalize_shell_command(command)
    return "harness research pr" in normalized and "--push" in normalized and "--open" in normalized


def _is_promotion_artifact_pr_flow(tool_events: list[ActivityEvent]) -> bool:
    file_write_events = _promotion_artifact_writes(tool_events)
    commands = _executed_shell_commands(tool_events)
    shell_driven_promotion = any(
        any(hint in command for hint in _PROMOTION_FLOW_COMMAND_HINTS) for command in commands
    )
    if not file_write_events and not shell_driven_promotion:
        return False
    return any(_is_harness_research_pr_open_command(command) for command in commands)


def _promotion_artifact_writes(tool_events: list[ActivityEvent]) -> list[ActivityEvent]:
    writes: list[ActivityEvent] = []
    for event in tool_events:
        if event.data.get("name") not in {"write_file", "edit_file", "apply_diff", "patch"}:
            continue
        if event.data.get("is_error"):
            continue
        arguments = event.data.get("arguments")
        if not isinstance(arguments, dict):
            continue
        path = str(arguments.get("path") or arguments.get("file") or "").strip()
        if path and _promotion_artifact_path(path):
            writes.append(event)
    return writes


def _executed_shell_commands(tool_events: list[ActivityEvent]) -> list[str]:
    return [
        _normalize_shell_command(_shell_command(event))
        for event in tool_events
        if event.data.get("name") in {"shell", "bash", "run_command", "execute"}
        and not event.data.get("is_error")
    ]


def minimal_hint(prompt: str) -> str | None:
    m = _MINIMAL_HINTS_RE.search(prompt or "")
    return m.group(0) if m else None


def extract_scope_paths(prompt: str) -> set[str]:
    found: set[str] = set()
    for m in _FILE_PATH_RE.finditer(prompt):
        path = m.group(1) or m.group(2)
        if not path:
            continue
        path = path.strip().lstrip("./")
        if not path:
            continue
        found.add(path)
        found.add(Path(path).name)
    full_paths = {path for path in found if "/" in path}
    first_line = next((line.strip().lower() for line in prompt.splitlines() if line.strip()), "")
    normalized_first_line = first_line.lstrip("#").strip()
    if (
        full_paths
        and all(path.startswith("tests/") for path in full_paths)
        and _FUNCTION_CALL_RE.search(prompt)
        and normalized_first_line.startswith(("fix ", "debug ", "handle ", "correct "))
    ):
        return set()
    return found


def touched_paths(activity: list[ActivityEvent]) -> set[str]:
    touched: set[str] = set()
    for e in activity:
        if e.kind != "tool_call.completed":
            continue
        if e.data.get("is_error"):
            continue
        name = e.data.get("name")
        if name not in ("write_file", "edit_file"):
            continue
        args = e.data.get("arguments") or {}
        path = args.get("path")
        if not isinstance(path, str) or not path:
            continue
        normalized = path.lstrip("./")
        touched.add(normalized)
        touched.add(Path(normalized).name)
    return touched


class ChainedVerifier:
    name = "chained"

    def __init__(self, *verifiers: Any) -> None:
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


class ShellVerifier:
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
                    f"Command `{self._command}` exited with code {proc.returncode}.\n\n{output}"
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


class VerifyBeforeDoneVerifier:
    name = "verify_before_done"

    def __init__(self, write_tool_names: frozenset[str] | None = None) -> None:
        self._writes = write_tool_names if write_tool_names is not None else _WRITE_TOOL_NAMES

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        tool_events = [e for e in activity if e.kind == "tool_call.completed"]

        if _is_promotion_artifact_pr_flow(tool_events):
            return VerificationResult(
                can_finish=True,
                reason=(
                    "Only generated research promotion artifacts were modified and the "
                    "PR flow was completed — generic verify_work is not required."
                ),
                verifier_name=self.name,
            )

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

        return VerificationResult(
            can_finish=True,
            reason="verify_work was called — deferring to downstream verifier.",
            verifier_name=self.name,
        )


class ResearchPromotionFlowVerifier:
    name = "research_promotion_flow"

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        tool_events = [e for e in activity if e.kind == "tool_call.completed"]
        promotion_writes = _promotion_artifact_writes(tool_events)
        commands = _executed_shell_commands(tool_events)
        shell_driven_promotion = any(
            any(hint in command for hint in _PROMOTION_FLOW_COMMAND_HINTS) for command in commands
        )
        if not promotion_writes and not shell_driven_promotion:
            return VerificationResult(
                can_finish=True,
                reason="no research promotion artifacts were edited",
                verifier_name=self.name,
            )

        created_candidate = any(
            needle in command
            for command in commands
            for needle in (
                "harness research refine",
                "harness research create-candidate",
                "harness research candidate create",
            )
        )
        promoted = any("harness research promote" in command for command in commands)
        opened_pr = any(_is_harness_research_pr_open_command(command) for command in commands)

        if created_candidate and promoted and opened_pr:
            return VerificationResult(
                can_finish=True,
                reason="research promotion artifacts were produced through the harness promotion flow",
                verifier_name=self.name,
            )

        missing: list[str] = []
        if not created_candidate:
            missing.append("candidate creation via refine/create-candidate")
        if not promoted:
            missing.append("promotion draft generation via `harness research promote`")
        if not opened_pr:
            missing.append("PR opening via `harness research pr --push --open [--draft]`")

        return VerificationResult(
            can_finish=False,
            reason=(
                "You edited `.harness/research/promotions/...` artifacts directly without "
                "using the full Harness promotion flow. Use "
                "`harness research create-candidate` (or `refine` / `candidate create`), "
                "`harness research promote`, and `harness research pr --push --open` instead. "
                f"Missing: {', '.join(missing)}."
            ),
            verifier_name=self.name,
        )


class MinimalFixVerifier:
    name = "minimal_fix"

    def __init__(self, *, max_lines: int = 8) -> None:
        self._max_lines = max_lines

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        prompt = first_user_prompt(session)
        hint = minimal_hint(prompt)
        if hint is None:
            return VerificationResult(
                can_finish=True,
                reason="no 'minimal fix' constraint in prompt",
                confidence=0.4,
                verifier_name=self.name,
            )

        written_lines = 0
        written_files: set[str] = set()
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
                written_files.add(path)
            content = args.get("content") or args.get("new_text") or args.get("diff")
            if not isinstance(content, str):
                content = str(ev.data.get("content_preview") or "")
            written_lines += content.count("\n") + (
                1 if content and not content.endswith("\n") else 0
            )

        if written_lines == 0:
            return VerificationResult(
                can_finish=True,
                reason=f"no writes recorded; minimal-fix hint {hint!r} satisfied vacuously",
                verifier_name=self.name,
            )

        if written_lines <= self._max_lines:
            return VerificationResult(
                can_finish=True,
                reason=(
                    f"diff is {written_lines} lines across {len(written_files)} "
                    f"file(s) — within the minimal-fix budget"
                ),
                confidence=0.85,
                verifier_name=self.name,
            )

        return VerificationResult(
            can_finish=False,
            reason=(
                f"Prompt requested a minimal fix ({hint!r}), but you wrote "
                f"~{written_lines} lines across {sorted(written_files)[:3]}. "
                f"Revert anything beyond the minimal change — leave cleanup, "
                f"refactors, and unrelated improvements for a follow-up."
            ),
            confidence=0.8,
            verifier_name=self.name,
        )


class PhaseGateVerifier:
    name = "phase_gate"

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        if not session.phases:
            return VerificationResult(
                can_finish=True,
                reason="no phases declared — nothing to enforce",
                confidence=0.4,
                verifier_name=self.name,
            )

        outstanding = [p.name for p in session.phases if not p.is_complete]
        declared_order = [p.name for p in session.phases]
        if outstanding:
            return VerificationResult(
                can_finish=False,
                reason=(
                    f"You declared phase(s) {declared_order} but these are "
                    f"still outstanding: {outstanding}. Finish each one and "
                    f"call phase(action='complete', name='<phase>') with "
                    f"evidence, or revisit the plan if the original phasing "
                    f"was wrong."
                ),
                confidence=0.9,
                verifier_name=self.name,
            )

        return VerificationResult(
            can_finish=True,
            reason=f"all {len(declared_order)} declared phase(s) completed",
            confidence=0.9,
            verifier_name=self.name,
        )


class TestsBeforeEditVerifier:
    name = "tests_before_edit"

    def __init__(self, write_tool_names: frozenset[str] | None = None) -> None:
        if write_tool_names is None:
            write_tool_names = frozenset({"write_file", "edit_file", "apply_diff", "patch"})
        self._writes = write_tool_names

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        tool_events = [e for e in activity if e.kind == "tool_call.completed"]

        first_edit_idx: int | None = None
        first_test_idx: int | None = None
        for idx, ev in enumerate(tool_events):
            name = ev.data.get("name")
            if first_edit_idx is None and name in self._writes:
                first_edit_idx = idx
            if first_test_idx is None and looks_like_test_invocation(ev):
                first_test_idx = idx
            if first_edit_idx is not None and first_test_idx is not None:
                break

        if first_edit_idx is None:
            return VerificationResult(
                can_finish=True,
                reason="no edits — nothing to gate on prior tests",
                verifier_name=self.name,
            )

        if first_test_idx is not None and first_test_idx < first_edit_idx:
            return VerificationResult(
                can_finish=True,
                reason="a test run happened before the first edit — tests informed the fix",
                verifier_name=self.name,
            )

        if looks_like_feature_add(first_user_prompt(session)):
            return VerificationResult(
                can_finish=True,
                reason="feature-add task — tests-before-edit bypass",
                verifier_name=self.name,
            )

        return VerificationResult(
            can_finish=False,
            reason=(
                "You edited files without running the test suite first. "
                "Before making changes, call verify_work to see which tests "
                "actually fail — the failing test names often reveal the real "
                "bug, which may differ from what the user's prompt suggests. "
                "Run verify_work, read the failing test names, THEN decide what "
                "to change."
            ),
            confidence=0.85,
            verifier_name=self.name,
        )


class FileScopeVerifier:
    name = "file_scope"

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        prompt = first_user_prompt(session)
        allowed = extract_scope_paths(prompt)
        if not allowed:
            return VerificationResult(
                can_finish=True,
                reason="no file-scope constraint detected in prompt",
                confidence=0.4,
                verifier_name=self.name,
            )

        touched = touched_paths(activity)
        if not touched:
            return VerificationResult(
                can_finish=True,
                reason="no file writes recorded — nothing to enforce scope against",
                confidence=0.4,
                verifier_name=self.name,
            )

        extra = sorted(p for p in touched if p not in allowed and "/" in p)
        extra = [p for p in extra if Path(p).name not in allowed]
        if not extra:
            return VerificationResult(
                can_finish=True,
                reason=f"all modified files were in scope: {sorted(allowed)[:3]}",
                confidence=0.85,
                verifier_name=self.name,
            )

        return VerificationResult(
            can_finish=False,
            reason=(
                f"Task named these files as in-scope: {sorted(allowed)[:5]}, "
                f"but you also modified: {extra}. Revert the out-of-scope "
                f"changes — the user explicitly asked for a minimal fix."
            ),
            confidence=0.9,
            verifier_name=self.name,
        )
