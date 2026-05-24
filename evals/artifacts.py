"""Artifact persistence and trace/metric helpers for eval runs."""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

from evals.types import HardMetrics, RunOutcome, TraceEvent

_TOOL_NAMES = (
    "read_file",
    "write_file",
    "edit_file",
    "list_dir",
    "glob",
    "shell",
    "verify_work",
    "complete_work_item",
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
        lowered = line.lower()
        for tool_name in _TOOL_NAMES:
            if tool_name in lowered:
                sequence.append(tool_name)
                break
    return sequence


def transcript_mentions_verification(transcript: str, verify_command: str) -> bool:
    lowered = transcript.lower()
    if "verify_work" in lowered:
        return True
    verify_head = verify_command.strip().split()[0].lower() if verify_command.strip() else ""
    if verify_head and verify_head in lowered:
        return True
    return any(marker in lowered for marker in ("pytest", "cargo test", "go test", "npm test"))


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
    return HardMetrics(
        verify_passed=verify_exit_code == 0,
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
