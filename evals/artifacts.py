"""Artifact persistence and trace/metric helpers for eval runs."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

from evals.types import BenchmarkIntegrityReport, HardMetrics, RunOutcome, TraceEvent

_TOOL_NAMES = (
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "glob",
    "shell",
    "fetch_url",
    "web_search",
    "google_search",
    "tavily_search",
    "verify_work",
    "complete_work_item",
)

_ACTION_PREFIXES = (*(f"{name}(" for name in _TOOL_NAMES), "→ ", "$ ")
_REFERENCE_SOLUTION_PATH_HINTS = (
    "/solution/",
    "/solution",
    "solution/",
    "solution.patch",
    "solve.sh",
)
_HIDDEN_TEST_PATH_HINTS = (
    "/tests/",
    "/tests",
    "test.patch",
    "hidden_tests/",
    "hidden-tests/",
)
_SECRET_PREFIXES = (
    "OPENROUTER_API_KEY=",
    "TAVILY_API_KEY=",
)


def build_trace_events(
    transcript: str,
    verify_command: str,
    *,
    agent_exit_code: int,
    verify_exit_code: int,
) -> list[TraceEvent]:
    events: list[TraceEvent] = [
        TraceEvent(kind="agent_exit", order=1, data={"exit_code": agent_exit_code}),
        TraceEvent(kind="verify_exit", order=2, data={"exit_code": verify_exit_code}),
    ]
    order = len(events) + 1
    for tool_name in extract_tool_sequence(transcript):
        events.append(TraceEvent(kind="tool_call", order=order, data={"tool": tool_name}))
        order += 1
    verify_name = verify_command.split()[0] if verify_command.strip() else "verify"
    if transcript_mentions_verification(transcript, verify_command):
        events.append(
            TraceEvent(
                kind="verification_observed",
                order=order,
                message=f"Detected verification marker for {verify_name}.",
                data={"command": verify_command},
            )
        )
    return events


def extract_tool_sequence(transcript: str) -> list[str]:
    sequence: list[str] = []
    for line in transcript.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if stripped.startswith("→ "):
            lowered = stripped[2:].lstrip().lower()
        elif lowered.startswith(("✓ verify_work:", "✗ verify_work:")):
            if not sequence or sequence[-1] != "verify_work":
                sequence.append("verify_work")
            continue
        elif not any(
            lowered == tool_name or lowered.startswith(f"{tool_name}(") for tool_name in _TOOL_NAMES
        ):
            continue
        for tool_name in _TOOL_NAMES:
            if lowered == tool_name or lowered.startswith(f"{tool_name}("):
                sequence.append(tool_name)
                break
    return sequence


def _redact_evidence(text: str) -> str:
    redacted = text
    for prefix in _SECRET_PREFIXES:
        if prefix in redacted:
            start = redacted.find(prefix) + len(prefix)
            end = start
            while end < len(redacted) and not redacted[end].isspace():
                end += 1
            redacted = redacted[:start] + "<redacted>" + redacted[end:]
    for marker in ("sk-or-v1" + "-", "tvly" + "-"):
        index = redacted.find(marker)
        while index != -1:
            end = index
            while end < len(redacted) and (redacted[end].isalnum() or redacted[end] in "-_"):
                end += 1
            redacted = redacted[:index] + marker + "<redacted>" + redacted[end:]
            index = redacted.find(marker, index + len(marker) + len("<redacted>"))
    return redacted[:240]


def _looks_like_action_line(line: str) -> bool:
    stripped = line.strip()
    lowered = stripped.lower()
    if lowered.startswith(_ACTION_PREFIXES):
        return True
    return any(lowered.startswith(f"{name}(") for name in _TOOL_NAMES)


def _repo_lookup_terms(forbidden_repo_urls: list[str] | tuple[str, ...]) -> set[str]:
    terms: set[str] = set()
    for raw_url in forbidden_repo_urls:
        cleaned = raw_url.strip().lower()
        if not cleaned:
            continue
        cleaned = cleaned.removeprefix("https://").removeprefix("http://")
        cleaned = cleaned.removeprefix("www.")
        cleaned = cleaned.rstrip("/")
        terms.add(cleaned)
        if "/" in cleaned:
            path = cleaned.split("/", 1)[1]
            if path:
                terms.add(path)
            repo_name = path.rsplit("/", 1)[-1]
            if repo_name:
                terms.add(repo_name)
    return terms


def _looks_like_external_lookup_action(lowered_action_line: str) -> bool:
    if lowered_action_line.startswith(
        ("web_search(", "fetch_url(", "google_search(", "tavily_search(")
    ):
        return True
    if lowered_action_line.startswith("→ "):
        return _looks_like_external_lookup_action(lowered_action_line[2:].lstrip())
    return any(
        command in lowered_action_line
        for command in (
            "curl ",
            "wget ",
            "git clone ",
            "gh repo clone ",
            "gh repo view ",
            "pip install git+http",
        )
    )


def check_benchmark_integrity(
    transcript: str,
    *,
    enabled: bool = True,
    agent_command: list[str] | tuple[str, ...] | None = None,
    forbidden_repo_urls: list[str] | tuple[str, ...] = (),
) -> BenchmarkIntegrityReport:
    """Detect benchmark leakage in an agent trajectory.

    This is intentionally about benchmark artifacts, not task-domain keywords:
    solution folders, hidden verifier tests, solution-bearing runners, and
    configured source-repo lookups.
    """
    if not enabled:
        return BenchmarkIntegrityReport(passed=True)

    violations: list[str] = []
    command = list(agent_command or [])
    command_text = " ".join(command).lower()
    if "--agent oracle" in command_text or any(
        prev == "--agent" and current.lower() == "oracle" for prev, current in pairwise(command)
    ):
        violations.append("solution-bearing oracle runner was used")

    repo_terms = _repo_lookup_terms(forbidden_repo_urls)
    seen: set[str] = set()
    for line in transcript.splitlines():
        if not _looks_like_action_line(line):
            continue
        lowered = line.lower()
        evidence = _redact_evidence(line.strip())
        if any(hint in lowered for hint in _REFERENCE_SOLUTION_PATH_HINTS):
            message = f"reference solution access detected: {evidence}"
            if message not in seen:
                violations.append(message)
                seen.add(message)
        if any(hint in lowered for hint in _HIDDEN_TEST_PATH_HINTS):
            message = f"hidden verifier test access detected: {evidence}"
            if message not in seen:
                violations.append(message)
                seen.add(message)
        if (
            repo_terms
            and _looks_like_external_lookup_action(lowered)
            and any(term in lowered for term in repo_terms)
        ):
            message = f"forbidden source repo lookup detected: {evidence}"
            if message not in seen:
                violations.append(message)
                seen.add(message)

    return BenchmarkIntegrityReport(passed=not violations, violations=violations)


def transcript_mentions_verification(transcript: str, verify_command: str) -> bool:
    if "verify_work" in extract_tool_sequence(transcript):
        return True
    verify_head = verify_command.strip().split()[0].lower() if verify_command.strip() else ""
    if not verify_head:
        return False
    for line in transcript.splitlines():
        lowered = line.strip().lower()
        if lowered.startswith("→ shell(") and verify_head in lowered:
            return True
        if lowered.startswith("$ ") and verify_head in lowered:
            return True
    return False


def diff_stats(git_diff: str) -> tuple[int, int, int]:
    files: set[str] = set()
    lines_added = 0
    lines_deleted = 0
    for line in git_diff.splitlines():
        if line.startswith("+++ b/"):
            files.add(line[6:])
            continue
        if line.startswith("--- ") or line.startswith("+++ "):
            continue
        if line.startswith("+"):
            lines_added += 1
        elif line.startswith("-"):
            lines_deleted += 1
    return len(files), lines_added, lines_deleted


def compute_hard_metrics(
    transcript: str,
    git_diff: str,
    verify_command: str,
    *,
    run_exit_code: int,
    verify_exit_code: int,
    agent_duration_seconds: float,
    verify_duration_seconds: float,
    benchmark_integrity_enabled: bool = False,
    agent_command: list[str] | tuple[str, ...] | None = None,
    forbidden_repo_urls: list[str] | tuple[str, ...] = (),
) -> HardMetrics:
    files_touched, lines_added, lines_deleted = diff_stats(git_diff)
    tool_sequence = extract_tool_sequence(transcript)
    did_run_verification = transcript_mentions_verification(transcript, verify_command)
    verify_positions = [idx for idx, name in enumerate(tool_sequence) if name == "verify_work"]
    first_verify_idx = verify_positions[0] if verify_positions else None
    mutating_tools = {"write_file", "edit_file", "shell"}
    edit_before_repro = False
    if first_verify_idx is not None:
        edit_before_repro = any(name in mutating_tools for name in tool_sequence[:first_verify_idx])
    redundant_tool_calls = 0
    retry_loops = 0
    streak = 1
    for prev, current in pairwise(tool_sequence):
        if prev == current:
            streak += 1
            redundant_tool_calls += 1
            if streak >= 3:
                retry_loops += 1
        else:
            streak = 1
    lowered = transcript.lower()
    success_claim = any(phrase in lowered for phrase in ("done", "fixed", "all set", "completed"))
    premature_completion = success_claim and verify_exit_code != 0
    verification_after_failure = tool_sequence.count("verify_work") >= 2 or (
        did_run_verification and verify_exit_code == 0 and run_exit_code != 0
    )
    shell_commands = tool_sequence.count("shell")
    integrity_report = check_benchmark_integrity(
        transcript,
        enabled=benchmark_integrity_enabled,
        agent_command=agent_command,
        forbidden_repo_urls=forbidden_repo_urls,
    )
    return HardMetrics(
        verify_passed=run_exit_code == 0 and verify_exit_code == 0 and integrity_report.passed,
        run_exit_code=run_exit_code,
        verify_exit_code=verify_exit_code,
        files_touched=files_touched,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        tool_calls=len(tool_sequence),
        shell_commands=shell_commands,
        did_run_verification=did_run_verification,
        agent_duration_seconds=agent_duration_seconds,
        verify_duration_seconds=verify_duration_seconds,
        total_duration_seconds=agent_duration_seconds + verify_duration_seconds,
        time_to_first_verification_seconds=agent_duration_seconds if did_run_verification else None,
        edit_before_repro=edit_before_repro,
        premature_completion=premature_completion,
        redundant_tool_calls=redundant_tool_calls,
        retry_loops=retry_loops,
        verification_after_failure=verification_after_failure,
        benchmark_integrity_passed=integrity_report.passed,
        benchmark_integrity_violations=integrity_report.violations,
    )


def persist_artifacts(artifact_dir: Path, outcome: RunOutcome) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "transcript.txt").write_text(outcome.transcript, encoding="utf-8")
    (artifact_dir / "git_diff.patch").write_text(outcome.git_diff, encoding="utf-8")
    (artifact_dir / "verify_output.txt").write_text(outcome.test_output, encoding="utf-8")
    (artifact_dir / "agent_command.json").write_text(
        json.dumps(outcome.agent_command, indent=2),
        encoding="utf-8",
    )
    (artifact_dir / "outcome.json").write_text(
        json.dumps(outcome.to_dict(), indent=2),
        encoding="utf-8",
    )
    with (artifact_dir / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for event in outcome.trace_events:
            handle.write(json.dumps(event.to_dict()) + "\n")
