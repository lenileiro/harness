"""Deterministic structural verifiers extracted from verification.py."""

from __future__ import annotations

import asyncio
import re
import shlex
from pathlib import Path
from typing import Any

from harness.core.activity import ActivityEvent
from harness.core.schemas import Session, VerificationResult
from harness.core.tools_verification import (
    _command_exits_before_trailing_command,
    _failure_branch_masks_exit_status,
    _output_reports_failure,
)

_DIRECT_WRITE_TOOL_NAMES = frozenset(
    {
        "apply_patch",
        "write_file",
        "edit_file",
        "apply_diff",
        "patch",
    }
)
_SHELL_TOOL_NAMES = frozenset({"shell", "bash", "run_command", "execute"})
_WRITE_TOOL_NAMES = _DIRECT_WRITE_TOOL_NAMES | _SHELL_TOOL_NAMES
_READ_ONLY_SHELL_TOOLS = frozenset(
    {
        "awk",
        "cat",
        "date",
        "echo",
        "find",
        "grep",
        "head",
        "jq",
        "ls",
        "pwd",
        "rg",
        "tail",
        "uname",
        "wc",
        "which",
    }
)
_READ_ONLY_GIT_COMMANDS = frozenset({"branch", "diff", "grep", "log", "ls-files", "show", "status"})
_SHELL_MUTATION_WORDS = frozenset(
    {
        "add",
        "create",
        "delete",
        "install",
        "remove",
        "uninstall",
        "update",
        "upgrade",
        "write",
    }
)
_WRITE_SHELL_RE = re.compile(
    r"(^|[;&|]\s*)(rm|mv|cp|mkdir|touch|chmod|chown|install|tee|patch|truncate)\b"
    r"|(^|[;&|]\s*)git\s+(apply|checkout|commit|merge|pull|push|rebase|reset|restore|switch)\b"
    r"|(^|[;&|]\s*)sed\s+-i\b"
    r"|(^|[;&|]\s*)perl\s+-p?i\b"
    r"|>{1,2}|<<",
    re.IGNORECASE,
)
_VERIFY_WORK_MUTATION_RE = re.compile(
    r"(^|[;&|]\s*)(rm|mv|cp|mkdir|touch|chmod|chown|install|tee|patch|truncate)\b"
    r"|(^|[;&|]\s*)git\s+(apply|checkout|commit|merge|pull|push|rebase|reset|restore|switch)\b"
    r"|(^|[;&|]\s*)sed\s+-i\b"
    r"|(^|[;&|]\s*)perl\s+-p?i\b"
    r"|>{1,2}"
    r"|\b(?:write_text|write_bytes|unlink|rename)\s*\("
    r"|\bopen\s*\([^)]*['\"][wax+]['\"]",
    re.IGNORECASE,
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
_PUBLIC_SOURCE_EVIDENCE_TOOLS = frozenset({"web_search", "fetch_url"})
_PUBLIC_SOURCE_SHELL_FETCH_TOOLS = frozenset({"curl", "wget", "fetch"})
_SHELL_WRAPPER_TOOLS = frozenset({"sh", "bash", "zsh", "ksh"})
_CURRENT_FACT_RE = re.compile(r"\b(latest|current|most\s+recent|newest|recent|stable)\b", re.I)
_PUBLIC_SOURCE_RE = re.compile(
    r"\b(public|web|online|official|external|internet|documentation|docs|changelog|"
    r"release\s+notes?)\b",
    re.I,
)
_PUBLIC_FACT_SUBJECT_RE = re.compile(
    r"\b(release|version|api|tool|library|package|dependency|url|source|status|docs?|"
    r"documentation|changelog|value|fact)\b",
    re.I,
)

_VERIFY_INTENT_WORDS = frozenset(
    {
        "build",
        "check",
        "checks",
        "ci",
        "lint",
        "spec",
        "specs",
        "test",
        "tests",
        "typecheck",
        "typechecks",
        "validate",
        "validation",
        "verify",
        "verification",
    }
)
_VERIFY_INTENT_SUFFIXES = ("test", "tests", "spec", "specs", "check", "checks")
_ASSERTION_TOOL_NAMES = frozenset({"test", "[", "grep", "cmp", "diff"})
_FILE_OBSERVATION_COMMANDS = frozenset(
    {
        "awk",
        "cat",
        "cmp",
        "diff",
        "echo",
        "find",
        "grep",
        "head",
        "ls",
        "printf",
        "sed",
        "stat",
        "tail",
        "test",
        "wc",
    }
)
_TRIVIAL_VERIFY_TOOL_NAMES = frozenset(
    {":", "true", "false", "echo", "printf", "date", "pwd", "sleep", "uname", "whoami"}
)
_NON_EXECUTING_VERIFY_FLAGS = frozenset(
    {
        "--version",
        "version",
        "--help",
        "-h",
        "--dry-run",
        "--list",
        "--list-tests",
        "--listtests",
        "-list",
        "--showconfig",
        "--show-config",
    }
)


def _token_has_non_executing_verify_flag(token: str) -> bool:
    lowered = token.lower()
    if lowered in _NON_EXECUTING_VERIFY_FLAGS:
        return True
    if any(lowered.startswith(f"{flag}=") for flag in _NON_EXECUTING_VERIFY_FLAGS):
        return True
    flag_words = set(re.split(r"[^a-z0-9]+", lowered))
    if "collect" in flag_words and "only" in flag_words:
        return True
    if "setup" in flag_words and any(word in flag_words for word in {"only", "plan"}):
        return True
    if "show" in flag_words and "config" in flag_words:
        return True
    if "=" not in lowered or lowered.startswith("-"):
        return False
    value = lowered.split("=", 1)[1].strip("'\"")
    candidates = [value, *re.split(r"\s+", value)]
    return any(
        candidate in _NON_EXECUTING_VERIFY_FLAGS
        or any(candidate.startswith(f"{flag}=") for flag in _NON_EXECUTING_VERIFY_FLAGS)
        for candidate in candidates
    )


def _strip_leading_env_assignments(tokens: list[str]) -> list[str]:
    remaining = list(tokens)
    if remaining and Path(remaining[0]).name.lower() == "env":
        remaining = remaining[1:]
    while remaining:
        token = remaining[0]
        if token.startswith("-") or "=" not in token:
            break
        key = token.split("=", 1)[0]
        if not key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", key):
            break
        remaining = remaining[1:]
    return remaining


def _intent_words_from_token(token: str) -> set[str]:
    lowered = Path(token).name.lower().strip()
    words = {word for word in re.split(r"[^a-z0-9]+", lowered) if word}
    for suffix in _VERIFY_INTENT_SUFFIXES:
        if lowered.endswith(suffix):
            words.add(suffix)
    return words


def _token_has_verify_intent(token: str) -> bool:
    return bool(_intent_words_from_token(token) & _VERIFY_INTENT_WORDS)


def _command_segment_has_verify_intent(tokens: list[str]) -> bool:
    if not tokens:
        return False
    executable = Path(tokens[0]).name.lower()
    if executable in _TRIVIAL_VERIFY_TOOL_NAMES:
        return False
    if executable in _ASSERTION_TOOL_NAMES:
        return False
    if executable in _READ_ONLY_SHELL_TOOLS:
        return False
    return any(_token_has_verify_intent(token) for token in tokens)


def _path_tokens(path: str) -> set[str]:
    normalized = path.strip().strip("/").lower()
    return {token for token in re.split(r"[^a-z0-9]+", normalized) if token}


_EXACT_FILE_CONTENT_REQUEST_RE = re.compile(
    r"\b(?:create|write|make)\s+(?P<path>[A-Za-z0-9._@%+=:,~/-]+)\s+"
    r"(?:containing|that\s+contains|with(?:\s+the)?(?:\s+exact)?(?:\s+contents?|\s+content|\s+text)?)"
    r"\s+exactly\s+"
    r"(?P<content>`[^`\n]+`|\"[^\"\n]+\"|'[^'\n]+'|[^\n.;]+?)"
    r"(?=\s+(?:and|then|without|do\s+not|by|using|via)\b"
    r"|\s+with\s+(?:no|zero)\s+(?:trailing\s+|final\s+)?(?:newline|line\s+break)s?\b"
    r"|[.;,\n]|$)",
    re.IGNORECASE,
)
_EXACT_FILE_CONTENT_NAMED_REQUEST_RE = re.compile(
    r"\b(?:create|write|make)\s+(?:(?:an?|the)\s+)?"
    r"(?:(?:[A-Za-z0-9_+.-]+)\s+)?"
    r"(?:file|script|program)\s+(?:(?:named|called|at|as)\s+)?"
    r"(?P<path>[A-Za-z0-9._@%+=:,~/-]+)\s+"
    r"(?:containing|that\s+contains|with(?:\s+the)?(?:\s+exact)?(?:\s+contents?|\s+content|\s+text)?)"
    r"\s+exactly\s+"
    r"(?P<content>`[^`\n]+`|\"[^\"\n]+\"|'[^'\n]+'|[^\n.;]+?)"
    r"(?=\s+(?:and|then|without|do\s+not|by|using|via)\b"
    r"|\s+with\s+(?:no|zero)\s+(?:trailing\s+|final\s+)?(?:newline|line\s+break)s?\b"
    r"|[.;,\n]|$)",
    re.IGNORECASE,
)
_EXACT_FILE_CONTENT_IS_REQUEST_RE = re.compile(
    r"\b(?:create|write|make)\s+(?:(?:an?|the)\s+)?"
    r"(?:(?:[A-Za-z0-9_+.-]+)\s+)?"
    r"(?:file|script|program)?\s*(?:(?:named|called|at|as)\s+)?"
    r"(?P<path>[A-Za-z0-9._@%+=:,~/-]+)\s+"
    r"(?:whose\s+)?(?:exact\s+)?(?:contents?|content|text)\s+"
    r"(?:is|are|should\s+be)\s+"
    r"(?P<content>`[^`\n]+`|\"[^\"\n]+\"|'[^'\n]+'|[^\n.;]+?)"
    r"(?=\s+(?:and|then|without|do\s+not|by|using|via)\b"
    r"|\s+with\s+(?:no|zero)\s+(?:trailing\s+|final\s+)?(?:newline|line\s+break)s?\b"
    r"|[.;,\n]|$)",
    re.IGNORECASE,
)
_EXACT_STDOUT_REQUEST_RE = re.compile(
    r"\b(?:create|write|make)\s+(?P<path>[A-Za-z0-9._@%+=:,~/-]+)\s+"
    r"(?:that\s+)?(?:prints?|outputs?|emits|writes\s+to\s+stdout)\s+exactly\s+"
    r"(?P<content>`[^`\n]+`|\"[^\"\n]+\"|'[^'\n]+'|[^\n.;]+?)"
    r"(?=\s+(?:and|then|without|do\s+not|by|using|via)\b"
    r"|\s+with\s+(?:no|zero)\s+(?:trailing\s+|final\s+)?(?:newline|line\s+break)s?\b"
    r"|[.;,\n]|$)",
    re.IGNORECASE,
)
_EXACT_STDOUT_NAMED_REQUEST_RE = re.compile(
    r"\b(?:create|write|make)\s+(?:(?:an?|the)\s+)?"
    r"(?:(?:[A-Za-z0-9_+.-]+)\s+)?(?:script|program|file)\s+"
    r"(?:(?:named|called|at|as)\s+)?(?P<path>[A-Za-z0-9._@%+=:,~/-]+)\s+"
    r"(?:that\s+)?(?:prints?|outputs?|emits|writes\s+to\s+stdout)\s+exactly\s+"
    r"(?P<content>`[^`\n]+`|\"[^\"\n]+\"|'[^'\n]+'|[^\n.;]+?)"
    r"(?=\s+(?:and|then|without|do\s+not|by|using|via)\b"
    r"|\s+with\s+(?:no|zero)\s+(?:trailing\s+|final\s+)?(?:newline|line\s+break)s?\b"
    r"|[.;,\n]|$)",
    re.IGNORECASE,
)
_BYTE_SIZE_REQUEST_RE = re.compile(r"\b(?:byte\s+size|file\s+size|checking\s+size|bytes?)\b", re.I)
_NO_TRAILING_NEWLINE_RE = re.compile(
    r"\b(?:no|zero)\s+(?:trailing\s+|final\s+)?(?:newline|line\s+break)s?\b",
    re.I,
)

_MINIMAL_HINTS_RE = re.compile(
    r"\b(minimal fix|don't (refactor|tackle|fix anything else)|only fix|just fix|"
    r"only modify|do not refactor|nothing else should change|no other changes|"
    r"minimal change|smallest fix)\b",
    re.IGNORECASE,
)

_FILE_PATH_RE = re.compile(r"`([^`\n]+?\.[A-Za-z0-9_+.-]{1,16})`|`([^`\n]*?/[^`\n]+?)`")

_FUNCTION_CALL_RE = re.compile(r"`[A-Za-z_][A-Za-z0-9_]*\([^`\n]*\)`")


def _strip_shell_comment_lines(command: str) -> str:
    lines: list[str] = []
    for line in command.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        in_single = False
        in_double = False
        escaped = False
        kept: list[str] = []
        for index, char in enumerate(line):
            if escaped:
                escaped = False
                kept.append(char)
                continue
            if char == "\\" and not in_single:
                escaped = True
                kept.append(char)
                continue
            if char == "'" and not in_double:
                in_single = not in_single
                kept.append(char)
                continue
            if char == '"' and not in_single:
                in_double = not in_double
                kept.append(char)
                continue
            if (
                char == "#"
                and not in_single
                and not in_double
                and (index == 0 or line[index - 1].isspace())
            ):
                break
            kept.append(char)
        lines.append("".join(kept))
    return "\n".join(lines).strip()


def _mask_shell_quoted_text(command: str) -> str:
    masked: list[str] = []
    quote = ""
    escaped = False
    for char in command:
        if quote:
            if escaped:
                escaped = False
            elif quote == '"' and char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
                masked.append(char)
                continue
            masked.append(" " if char != "\n" else "\n")
            continue
        if char in {"'", '"'}:
            quote = char
        masked.append(char)
    return "".join(masked)


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
    if _WRITE_SHELL_RE.search(command):
        return False
    for segment in _reachable_shell_segments(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            continue
        if _command_segment_has_verify_intent(_strip_leading_env_assignments(tokens)):
            return True

    preview = str(event.data.get("content_preview") or "").lower()
    return bool(
        re.search(r"\b[1-9]\d*\s+(?:tests?|specs?|checks?)\b", preview)
        or re.search(r"\b[1-9]\d*\s+(?:passed|failed)\b", preview)
        or re.search(r"\b(?:all\s+)?(?:tests?|specs?|checks?)\s+passed\b", preview)
    )


def _tool_event_path(event: ActivityEvent) -> str:
    arguments = event.data.get("arguments") or {}
    if not isinstance(arguments, dict):
        return ""
    path = arguments.get("path")
    if not isinstance(path, str):
        return ""
    return path.strip().lstrip("./")


def _is_test_path(path: str) -> bool:
    if not path:
        return False
    parts = Path(path).parts
    if any(_path_tokens(part) & {"test", "tests", "spec", "specs"} for part in parts[:-1]):
        return True
    name_tokens = _path_tokens(Path(path).stem)
    return bool(name_tokens & {"test", "tests", "spec", "specs"})


def _has_post_edit_regression_verification(
    tool_events: list[ActivityEvent],
    *,
    first_edit_idx: int,
    write_tool_names: frozenset[str],
) -> bool:
    saw_test_edit = False
    for event in tool_events[first_edit_idx + 1 :]:
        name = str(event.data.get("name") or "")
        if name in write_tool_names and not event.data.get("is_error"):
            path = _tool_event_path(event)
            if _is_test_path(path):
                saw_test_edit = True
                continue
        if saw_test_edit and looks_like_test_invocation(event) and not event.data.get("is_error"):
            return True
    return False


def shell_command_changes_state(command: str) -> bool:
    stripped = _strip_shell_comment_lines(command)
    if not stripped:
        return True
    scan_text = _mask_shell_quoted_text(stripped)
    if _WRITE_SHELL_RE.search(scan_text):
        return True
    if _command_has_shell_control_operator(stripped):
        segments = _reachable_shell_segments(stripped)
        if len(segments) != 1 or segments[0] != stripped:
            return any(shell_command_changes_state(segment) for segment in segments)
    try:
        parts = shlex.split(stripped)
    except ValueError:
        return True
    if not parts:
        return True

    parts = _drop_leading_env_assignments(parts)
    if not parts:
        return True

    executable = Path(parts[0]).name.lower()
    args = parts[1:]
    lowered_args = [part.lower() for part in args]
    if executable == "command" and not args:
        return False
    if executable == "command" and "-v" in lowered_args:
        return False
    has_version_or_help = any(
        arg in lowered_args for arg in ("--version", "version", "--help", "-h")
    ) or any(arg in {"-V", "-v"} for arg in args)
    if has_version_or_help and not any(arg in _SHELL_MUTATION_WORDS for arg in lowered_args):
        return False
    if executable in {"bash", "sh", "zsh"} and "-c" in args:
        command_index = args.index("-c") + 1
        if command_index >= len(args):
            return True
        return shell_command_changes_state(args[command_index])
    if executable in {"curl", "wget"}:
        return any(arg in {"-o", "-O", "--output", "--output-document"} for arg in args)
    if executable == "git":
        subcommand = next((part.lower() for part in args if not part.startswith("-")), "")
        return subcommand not in _READ_ONLY_GIT_COMMANDS
    if "--check" in lowered_args and not any(arg in _SHELL_MUTATION_WORDS for arg in lowered_args):
        return False
    return executable not in _READ_ONLY_SHELL_TOOLS


_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$", re.DOTALL)


def _drop_leading_env_assignments(parts: list[str]) -> list[str]:
    index = 0
    while index < len(parts) and _ENV_ASSIGNMENT_RE.match(parts[index]):
        index += 1
    return parts[index:]


def tool_event_changes_state(event: ActivityEvent, write_tool_names: frozenset[str]) -> bool:
    name = str(event.data.get("name") or "")
    metadata = event.data.get("metadata")
    if isinstance(metadata, dict) and metadata.get("workspace_changed") is True:
        return True
    if event.data.get("is_error") is True:
        return False
    if isinstance(metadata, dict) and (
        metadata.get("workspace_changed") is False
        or metadata.get("workspace_fingerprint_changed") is False
    ):
        return False
    if name not in write_tool_names:
        return False
    if name in _DIRECT_WRITE_TOOL_NAMES:
        return True
    if name in _SHELL_TOOL_NAMES:
        if looks_like_test_invocation(event):
            return False
        arguments = event.data.get("arguments")
        command = ""
        if isinstance(arguments, dict):
            command = str(
                arguments.get("command") or arguments.get("cmd") or arguments.get("text") or ""
            )
        return shell_command_changes_state(command)
    return True


def verify_work_command_changes_state(command: str) -> bool:
    stripped = _strip_shell_comment_lines(command)
    if not stripped:
        return False
    return bool(_VERIFY_WORK_MUTATION_RE.search(_mask_shell_quoted_text(stripped)))


def _verify_work_command(event: ActivityEvent) -> str:
    arguments = event.data.get("arguments")
    if isinstance(arguments, dict):
        command = str(arguments.get("command") or arguments.get("cmd") or "")
        if command:
            return command
    metadata = event.data.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("command") or "")
    return ""


def verify_work_event_changes_state(event: ActivityEvent) -> bool:
    if str(event.data.get("name") or "") != "verify_work":
        return False
    metadata = event.data.get("metadata")
    if isinstance(metadata, dict) and metadata.get("workspace_changed") is True:
        return True
    if isinstance(metadata, dict) and metadata.get("workspace_fingerprint_changed") is False:
        return False
    if isinstance(metadata, dict) and metadata.get("used_default_command") is True:
        return False
    command = _verify_work_command(event)
    return verify_work_command_changes_state(command)


def _verify_work_exit_code(event: ActivityEvent) -> int | None:
    metadata = event.data.get("metadata")
    if isinstance(metadata, dict):
        exit_code = metadata.get("exit_code")
        if isinstance(exit_code, int):
            return exit_code
    match = re.search(r"\bexit_code:\s*(-?\d+)\b", str(event.data.get("content_preview") or ""))
    if match:
        return int(match.group(1))
    return None


def _verify_work_exit_succeeded(event: ActivityEvent) -> bool:
    exit_code = _verify_work_exit_code(event)
    return exit_code is None or exit_code == 0


def _verify_work_command_has_generic_evidence(command: str) -> bool:
    stripped = _strip_shell_comment_lines(command)
    if not stripped:
        return False
    if _verify_work_command_is_broad_check(stripped):
        return True
    for segment in _reachable_shell_segments(stripped):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return False
        if not tokens:
            continue
        executable = Path(tokens[0]).name.lower()
        if executable in _TRIVIAL_VERIFY_TOOL_NAMES:
            continue
        if executable in _ASSERTION_TOOL_NAMES:
            return True
    return False


def _event_changed_paths(event: ActivityEvent) -> set[str]:
    paths: set[str] = set()
    name = str(event.data.get("name") or "")
    for source in (event.data.get("arguments"), event.data.get("metadata")):
        if not isinstance(source, dict):
            continue
        for key in ("path", "file"):
            value = source.get(key)
            if isinstance(value, str) and value.strip():
                normalized = value.strip().lstrip("./")
                paths.add(normalized)
                paths.add(Path(normalized).name)
        for key in ("paths", "files", "changed_paths"):
            value = source.get(key)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and item.strip():
                        normalized = item.strip().lstrip("./")
                        paths.add(normalized)
                        paths.add(Path(normalized).name)
        patch = source.get("patch") or source.get("diff")
        if isinstance(patch, str):
            for match in re.finditer(r"^diff --git a/(.+?) b/(.+?)$", patch, re.MULTILINE):
                for group in (1, 2):
                    normalized = match.group(group).strip().lstrip("./")
                    if normalized != "/dev/null":
                        paths.add(normalized)
                        paths.add(Path(normalized).name)
            for match in re.finditer(r"^\+\+\+\s+b/(.+)$", patch, re.MULTILINE):
                normalized = match.group(1).strip().lstrip("./")
                if normalized != "/dev/null":
                    paths.add(normalized)
                    paths.add(Path(normalized).name)
        command = source.get("command") or source.get("cmd") or source.get("text")
        if name in _SHELL_TOOL_NAMES | {"verify_work"} and isinstance(command, str):
            paths.update(_shell_mutation_target_paths(command))
    return paths


def _shell_mutation_target_paths(command: str) -> set[str]:
    paths: set[str] = set()
    stripped = _strip_shell_comment_lines(command)
    if not stripped:
        return paths
    try:
        tokens = shlex.split(stripped)
    except ValueError:
        tokens = []
    for index, token in enumerate(tokens):
        executable = Path(token).name.lower()
        if executable in {"touch", "mkdir", "rm", "rmdir", "truncate"}:
            for candidate in tokens[index + 1 :]:
                if candidate.startswith("-"):
                    continue
                normalized = candidate.strip().lstrip("./")
                if normalized:
                    paths.add(normalized)
                    paths.add(Path(normalized).name)
            break
        if executable in {"cp", "mv"} and len(tokens) > index + 2:
            candidate = tokens[-1].strip().lstrip("./")
            if candidate:
                paths.add(candidate)
                paths.add(Path(candidate).name)
            break
    for match in re.finditer(r"(?:^|[^\d])>{1,2}\s*([A-Za-z0-9._@%+=:,~/-]+)", stripped):
        normalized = match.group(1).strip().lstrip("./")
        if normalized and normalized not in {"/dev/null", "dev/null"}:
            paths.add(normalized)
            paths.add(Path(normalized).name)
    return paths


def _verify_work_command_is_broad_check(command: str) -> bool:
    stripped = _strip_shell_comment_lines(command)
    if not stripped:
        return False
    for segment in _reachable_shell_segments(stripped):
        if _verify_command_segment_is_broad_check(segment):
            return True
    return False


def _verify_command_segment_is_broad_check(segment: str) -> bool:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return False
    if not tokens:
        return False
    if any(_token_has_non_executing_verify_flag(token) for token in tokens):
        return False
    tokens = _strip_leading_env_assignments(tokens)
    if not tokens:
        return False
    return _command_segment_has_verify_intent(tokens)


def _verify_work_command_asserts_changed_path(command: str, paths: set[str]) -> bool:
    for segment in _reachable_shell_segments(command):
        if _assertion_segment_checks_changed_path(segment, paths):
            return True
    return False


def _assertion_segment_checks_changed_path(segment: str, paths: set[str]) -> bool:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return False
    if not tokens:
        return False
    executable = Path(tokens[0]).name.lower()
    if executable not in _ASSERTION_TOOL_NAMES:
        return False
    cleaned = [token for token in tokens[1:] if token != "]"]
    if executable in {"test", "["} and _tokens_are_presence_only_path_check(cleaned, paths):
        return False
    return any(_token_matches_any_path(token, paths) for token in cleaned)


def _token_matches_any_path(token: str, paths: set[str]) -> bool:
    cleaned = token.strip().strip("'\"")
    if not cleaned or cleaned.startswith("-"):
        return False
    candidates = {
        cleaned,
        cleaned.rstrip("):,;"),
        cleaned.lstrip("./").rstrip("):,;"),
    }
    return any(
        _path_matches_request(candidate, path)
        for candidate in candidates
        for path in paths
        if candidate and path
    )


def _tokens_are_presence_only_path_check(tokens: list[str], paths: set[str]) -> bool:
    presence_flags = {"-e", "-f", "-s", "-r", "-w", "-x", "-d", "-L", "-h"}
    logical_tokens = {"!", "-a", "-o", "(", ")"}
    saw_path = False
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in logical_tokens:
            index += 1
            continue
        if token not in presence_flags:
            return False
        if index + 1 >= len(tokens):
            return False
        candidate = tokens[index + 1]
        if candidate.startswith("-"):
            return False
        if not any(_path_matches_request(candidate, path) for path in paths if path):
            return False
        saw_path = True
        index += 2
    return saw_path


def verify_work_event_has_generic_evidence(
    event: ActivityEvent,
    *,
    changed_paths: set[str] | None = None,
) -> bool:
    if str(event.data.get("name") or "") != "verify_work":
        return False
    if event.data.get("is_error") or verify_work_event_changes_state(event):
        return False
    if not _verify_work_exit_succeeded(event):
        return False
    if _verify_work_output_reports_failure(event):
        return False
    metadata = event.data.get("metadata")
    if isinstance(metadata, dict) and metadata.get("used_default_command") is True:
        return True
    command = _verify_work_command(event)
    if _shell_command_static_outcome(command) in {"failure", "exit"}:
        return False
    if _failure_branch_masks_exit_status(command):
        return False
    if _command_exits_before_trailing_command(command):
        return False
    output_has_test_evidence = _verify_work_output_has_test_evidence(event)
    runs_test_path_with_pass_output = _verify_work_command_runs_test_path(
        command
    ) and _verify_work_output_has_test_pass_evidence(event)
    if (
        not _verify_work_command_has_generic_evidence(command)
        and not output_has_test_evidence
        and not runs_test_path_with_pass_output
    ):
        return False
    paths = changed_paths or set()
    broad_check = (
        _verify_work_command_is_broad_check(command)
        or output_has_test_evidence
        or runs_test_path_with_pass_output
    )
    if changed_paths is not None and not paths and not broad_check:
        return False
    if paths and not broad_check:
        return _verify_work_command_asserts_changed_path(command, paths)
    return True


def _verify_work_output_has_test_evidence(event: ActivityEvent) -> bool:
    metadata = event.data.get("metadata")
    extra = metadata if isinstance(metadata, dict) else {}
    output = "\n".join(
        str(value or "")
        for value in (
            event.data.get("content_preview"),
            extra.get("stdout"),
            extra.get("stderr"),
        )
    ).lower()
    if not output.strip():
        return False
    if re.search(r"(?m)^ok\s+\S+", output):
        return True
    if re.search(r"\bran\s+[1-9]\d*\s+tests?\b", output) and re.search(r"(?m)^ok\b", output):
        return True
    return bool(re.search(r"\b[1-9]\d*\s+passed\b", output))


def _verify_work_output_has_test_pass_evidence(event: ActivityEvent) -> bool:
    metadata = event.data.get("metadata")
    extra = metadata if isinstance(metadata, dict) else {}
    output = "\n".join(
        str(value or "")
        for value in (
            event.data.get("content_preview"),
            extra.get("stdout"),
            extra.get("stderr"),
        )
    ).lower()
    if _verify_work_output_has_test_evidence(event):
        return True
    return bool(
        re.search(r"\ball\s+tests?\s+passed\b", output)
        or re.search(r"\ball\s+test\s+cases?\s+passed\b", output)
        or re.search(r"\bverification\s+successful\b", output)
    )


def _verify_work_output_reports_failure(event: ActivityEvent) -> bool:
    metadata = event.data.get("metadata")
    extra = metadata if isinstance(metadata, dict) else {}
    output = "\n".join(
        str(value or "")
        for value in (
            event.data.get("content_preview"),
            extra.get("stdout"),
            extra.get("stderr"),
        )
    )
    return _output_reports_failure(output)


def missing_verify_work_after_last_state_change(
    activity: list[ActivityEvent],
    *,
    write_tool_names: frozenset[str] | None = None,
) -> bool:
    writes = write_tool_names if write_tool_names is not None else _WRITE_TOOL_NAMES
    tool_events = [event for event in activity if event.kind == "tool_call.completed"]
    state_change_indexes = [
        index
        for index, event in enumerate(tool_events)
        if tool_event_changes_state(event, writes) or verify_work_event_changes_state(event)
    ]
    if not state_change_indexes:
        return False
    last_state_change_index = state_change_indexes[-1]
    return not any(
        event.data.get("name") == "verify_work"
        for event in tool_events[last_state_change_index + 1 :]
    )


def _strip_inline_quote(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"`", '"', "'"}:
        return stripped[1:-1].strip()
    return stripped


def _exact_file_content_requests(text: str) -> list[tuple[str, str]]:
    requests: list[tuple[str, str]] = []
    for request_re in (
        _EXACT_FILE_CONTENT_REQUEST_RE,
        _EXACT_FILE_CONTENT_NAMED_REQUEST_RE,
        _EXACT_FILE_CONTENT_IS_REQUEST_RE,
    ):
        for match in request_re.finditer(text):
            path = match.group("path").strip().rstrip(".,;:")
            content = _strip_inline_quote(match.group("content"))
            if path and content and (path, content) not in requests:
                requests.append((path, content))
    return requests


def _exact_stdout_requests(text: str) -> list[tuple[str, str]]:
    requests: list[tuple[str, str]] = []
    for request_re in (
        _EXACT_STDOUT_REQUEST_RE,
        _EXACT_STDOUT_NAMED_REQUEST_RE,
    ):
        for match in request_re.finditer(text):
            path = match.group("path").strip().rstrip(".,;:")
            content = _strip_inline_quote(match.group("content"))
            if path and content and (path, content) not in requests:
                requests.append((path, content))
    return requests


def _command_asserts_exact_content(command: str, *, path: str, content: str) -> bool:
    if path not in command or content not in command:
        return False
    if _command_can_mask_assertion_failure(command):
        return False
    return any(
        _segment_asserts_exact_content(segment, path=path, content=content)
        for segment in _reachable_shell_segments(command)
    )


def _segment_asserts_exact_content(segment: str, *, path: str, content: str) -> bool:
    if path not in segment or content not in segment:
        return False
    escaped_path = re.escape(path)
    escaped_content = re.escape(content)
    file_read_expr = rf"\$\(\s*(?:cat\s+{escaped_path}|<\s*{escaped_path})\s*\)"
    expected_expr = rf"(?:[\"']{escaped_content}[\"']|{escaped_content}(?=\s|\]|\)|$))"
    return bool(
        re.search(
            rf"[\"']?{file_read_expr}[\"']?\s*" rf"(?:=|==)\s*{expected_expr}",
            segment,
            re.S,
        )
        or re.search(
            rf"\btest\s+[\"']?{file_read_expr}[\"']?\s*" rf"=\s*{expected_expr}",
            segment,
            re.S,
        )
        or any(
            _grep_segment_asserts_exact_content(piece, path=path, content=content)
            for piece in _shell_command_segments(segment)
        )
        or re.search(
            rf"\b(?:cmp|diff)\b[^\n;&]*<\(\s*printf\b[^)]*{escaped_content}[^)]*\)"
            rf"[^\n;&]*{escaped_path}\b",
            segment,
            re.S,
        )
        or re.search(
            rf"\bprintf\b[^\n;&|]*{escaped_content}[^\n;&|]*\|\s*"
            rf"\b(?:cmp|diff)\b[^\n;&|]*-\s+{escaped_path}\b",
            segment,
            re.S,
        )
    )


def _grep_segment_asserts_exact_content(segment: str, *, path: str, content: str) -> bool:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return False
    if not tokens:
        return False
    try:
        grep_index = next(
            index for index, token in enumerate(tokens) if Path(token).name.lower() == "grep"
        )
    except StopIteration:
        return False

    has_exact_match = False
    positional: list[str] = []
    index = grep_index + 1
    while index < len(tokens):
        token = tokens[index]
        if token == "--":
            positional.extend(tokens[index + 1 :])
            break
        if token.startswith("-") and token != "-":
            if "x" in token:
                has_exact_match = True
            if token in {"-e", "--regexp"} and index + 1 < len(tokens):
                positional.append(tokens[index + 1])
                index += 2
                continue
            index += 1
            continue
        positional.append(token)
        index += 1

    return (
        has_exact_match
        and len(positional) == 2
        and positional[0] == content
        and _path_matches_request(positional[1], path)
    )


def _path_matches_request(actual: str, expected: str) -> bool:
    actual_path = actual.strip()
    expected_path = expected.strip()
    if actual_path.startswith("./"):
        actual_path = actual_path[2:]
    if expected_path.startswith("./"):
        expected_path = expected_path[2:]
    return actual_path == expected_path or actual_path.endswith(f"/{expected_path}")


def _command_has_shell_control_operator(command: str) -> bool:
    command = _strip_shell_comment_lines(command)
    in_single = False
    in_double = False
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and not in_single:
            escaped = True
            index += 1
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            index += 1
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            index += 1
            continue
        if not in_single and not in_double:
            if command.startswith("&&", index) or command.startswith("||", index):
                return True
            if char in {"&", ";", "\n"}:
                return True
        index += 1
    return False


def _command_can_mask_assertion_failure(command: str) -> bool:
    command = _strip_shell_comment_lines(command)
    in_single = False
    in_double = False
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            index += 1
            continue
        if char == "\\" and not in_single:
            escaped = True
            index += 1
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            index += 1
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            index += 1
            continue
        if not in_single and not in_double:
            if command.startswith("||", index):
                tail = command[index + 2 :].lstrip()
                if not re.match(r"(?:exit\s+[1-9]\d*|false)\b", tail):
                    return True
                index += 2
                continue
            if command.startswith("&&", index):
                tail = command[index + 2 :].lstrip()
                if _shell_tail_starts_with_static_failure(tail):
                    return True
                index += 2
                continue
            if char == "&":
                return True
            if char in {";", "\n"}:
                return True
        index += 1
    return False


def _shell_tail_starts_with_static_failure(tail: str) -> bool:
    return _shell_segment_static_outcome(tail) in {"failure", "exit"}


def _shell_command_segments(command: str) -> list[str]:
    return [
        segment.strip()
        for segment in re.split(r"\s*(?:&&|\|\||;|\n)\s*", command)
        if segment.strip()
    ]


def _shell_command_segments_with_operators(command: str) -> list[tuple[str, str]]:
    pieces: list[tuple[str, str]] = []
    current: list[str] = []
    quote = ""
    escaped = False
    operator = ""
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            current.append(char)
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            current.append(char)
            index += 1
            continue
        if quote:
            if char == quote:
                quote = ""
            current.append(char)
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            index += 1
            continue
        if command.startswith("&&", index) or command.startswith("||", index):
            segment = "".join(current).strip()
            if segment:
                pieces.append((operator, segment))
            operator = command[index : index + 2]
            current = []
            index += 2
            continue
        if char in {";", "\n"}:
            segment = "".join(current).strip()
            if segment:
                pieces.append((operator, segment))
            operator = char
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    segment = "".join(current).strip()
    if segment:
        pieces.append((operator, segment))
    return pieces


def _reachable_shell_segments(command: str) -> list[str]:
    segments: list[str] = []
    previous_outcome = "success"
    for operator, segment in _shell_command_segments_with_operators(command):
        if operator == "&&" and previous_outcome in {"failure", "exit"}:
            continue
        if operator == "||" and previous_outcome in {"success", "exit"}:
            continue
        if operator in {";", "\n"} and previous_outcome == "exit":
            continue
        segments.append(segment)
        previous_outcome = _shell_segment_static_outcome(segment)
    return segments


def _shell_command_static_outcome(command: str) -> str:
    states: set[str] = {"success"}
    stripped = _strip_shell_comment_lines(command)
    for operator, segment in _shell_command_segments_with_operators(stripped):
        segment_states = _shell_outcome_states(_shell_segment_static_outcome(segment))
        next_states: set[str] = set()
        if operator == "&&":
            if "success" in states:
                next_states.update(segment_states)
            if "failure" in states:
                next_states.add("failure")
            if "exit" in states:
                next_states.add("exit")
        elif operator == "||":
            if "failure" in states:
                next_states.update(segment_states)
            if "success" in states:
                next_states.add("success")
            if "exit" in states:
                next_states.add("exit")
        elif operator in {";", "\n"}:
            if "exit" in states:
                next_states.add("exit")
            if states - {"exit"}:
                next_states.update(segment_states)
        else:
            next_states.update(segment_states)
        states = next_states or {"unknown"}
    if states == {"success"}:
        return "success"
    if states == {"failure"}:
        return "failure"
    if states == {"exit"}:
        return "exit"
    return "unknown"


def _shell_outcome_states(outcome: str) -> set[str]:
    if outcome in {"success", "failure", "exit"}:
        return {outcome}
    return {"success", "failure"}


def _shell_segment_static_outcome(segment: str) -> str:
    segment = _strip_shell_static_grouping(segment)
    if segment.startswith("!"):
        inverted_outcome = _shell_segment_static_outcome(segment[1:].lstrip())
        if inverted_outcome == "success":
            return "failure"
        if inverted_outcome in {"failure", "exit"}:
            return "success"
        return "unknown"
    pipeline_segments = _shell_pipeline_segments(segment)
    if len(pipeline_segments) > 1:
        return _shell_segment_static_outcome(pipeline_segments[-1])
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return "unknown"
    tokens = _strip_leading_env_assignments(tokens)
    if not tokens:
        return "success"
    executable = Path(tokens[0]).name.lower()
    if executable in {"command", "builtin"}:
        wrapped = tokens[1:]
        if wrapped and wrapped[0] == "--":
            wrapped = wrapped[1:]
        if not wrapped or wrapped[0].startswith("-"):
            return "unknown"
        return _shell_segment_static_outcome(shlex.join(wrapped))
    if executable in {"bash", "sh", "zsh"} and "-c" in tokens:
        command_index = tokens.index("-c") + 1
        if command_index >= len(tokens):
            return "failure"
        return _shell_segment_static_outcome(tokens[command_index])
    if executable in {"exit", "return"}:
        return "exit"
    if executable in {"false"}:
        return "failure"
    if executable in {"true", ":"}:
        return "success"
    return "unknown"


def _strip_shell_static_grouping(segment: str) -> str:
    candidate = segment.strip()
    changed = True
    while changed:
        changed = False
        while candidate.startswith("("):
            candidate = candidate[1:].lstrip()
            changed = True
        while candidate.endswith(")"):
            candidate = candidate[:-1].rstrip()
            changed = True
    return candidate


def _shell_pipeline_segments(command: str) -> list[str]:
    pieces: list[str] = []
    current: list[str] = []
    quote = ""
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
            current.append(char)
            index += 1
            continue
        if char == "\\" and quote != "'":
            escaped = True
            current.append(char)
            index += 1
            continue
        if quote:
            if char == quote:
                quote = ""
            current.append(char)
            index += 1
            continue
        if char in {"'", '"'}:
            quote = char
            current.append(char)
            index += 1
            continue
        if char == "|":
            previous_char = command[index - 1] if index > 0 else ""
            next_char = command[index + 1] if index + 1 < len(command) else ""
            if previous_char != "|" and next_char != "|":
                segment = "".join(current).strip()
                if segment:
                    pieces.append(segment)
                current = []
                index += 1
                continue
        current.append(char)
        index += 1
    segment = "".join(current).strip()
    if segment:
        pieces.append(segment)
    return pieces


def _command_asserts_byte_size(command: str, *, path: str, size: int) -> bool:
    if path not in command:
        return False
    if _command_can_mask_assertion_failure(command):
        return False
    return any(
        _segment_asserts_byte_size(segment, path=path, size=size)
        for segment in _reachable_shell_segments(command)
    )


def _segment_asserts_byte_size(segment: str, *, path: str, size: int) -> bool:
    if path not in segment:
        return False
    size_text = str(size)
    escaped_path = re.escape(path)
    wc_expr = (
        rf"[\"']?\$\(\s*wc\s+-c\s*(?:<\s*{escaped_path}\b|{escaped_path}\b)"
        r"(?:\s*\|\s*(?:tr\s+-d\s+['\"]\s['\"]|xargs))?\s*\)[\"']?"
    )
    return bool(
        re.search(rf"{wc_expr}\s*(?:-eq|=|==)\s*{size_text}\b", segment)
        or re.search(rf"\b{size_text}\s*(?:=|==)\s*{wc_expr}", segment)
        or re.search(
            rf"\btest\s+{wc_expr}\s+-eq\s+{size_text}\b",
            segment,
        )
    )


def _command_runs_path(command: str, *, path: str) -> bool:
    try:
        tokens = shlex.split(command)
    except ValueError:
        return path in command
    return any(_path_matches_request(token, path) for token in tokens)


def _command_directly_runs_path(command: str, *, path: str) -> bool:
    command = _strip_shell_comment_lines(command)
    if re.search(r"[|<>`]", command) or _command_has_shell_control_operator(command):
        return False
    return _shell_fragment_invokes_path(command, path=path)


def _shell_fragment_invokes_path(fragment: str, *, path: str) -> bool:
    try:
        tokens = shlex.split(fragment)
    except ValueError:
        return False
    if not tokens:
        return False
    if _path_matches_request(tokens[0], path):
        return True
    executable = Path(tokens[0]).name.lower()
    if executable in _FILE_OBSERVATION_COMMANDS:
        return False
    return any(
        _path_matches_request(token, path) for token in tokens[1:] if not token.startswith("-")
    )


def _directly_run_script_path(command: str) -> str:
    command = _strip_shell_comment_lines(command)
    if re.search(r"[|<>`]", command) or _command_has_shell_control_operator(command):
        return ""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return ""
    if not tokens:
        return ""
    executable = Path(tokens[0]).name.lower()
    if _is_test_path(tokens[0]):
        return tokens[0].strip().lstrip("./")
    if executable in _FILE_OBSERVATION_COMMANDS:
        return ""
    for token in tokens[1:]:
        if token.startswith("-"):
            continue
        normalized = token.strip().lstrip("./")
        if _is_test_path(normalized):
            return normalized
    return ""


def _verify_work_command_runs_test_path(command: str) -> bool:
    script_path = _directly_run_script_path(command)
    return bool(script_path and _is_test_path(script_path))


def _inline_script_from_command(command: str) -> str:
    command = _strip_shell_comment_lines(command)
    if _command_has_shell_control_operator(command):
        return ""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return ""
    if not tokens:
        return ""
    for option in ("-c", "-e", "--eval"):
        if option in tokens:
            index = tokens.index(option) + 1
            if index < len(tokens):
                return tokens[index]
    return ""


def _command_inline_script_asserts_exact_content(
    command: str,
    *,
    path: str,
    content: str,
    byte_size_required: bool,
) -> bool:
    script = _inline_script_from_command(command)
    if not script:
        return False
    return _script_asserts_exact_content(
        script,
        path=path,
        content=content,
        byte_size_required=byte_size_required,
    )


def _verify_work_stdout_raw(event: ActivityEvent) -> str | None:
    metadata = event.data.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("stdout"), str):
        return str(metadata["stdout"])
    return None


def _verify_work_stdout(event: ActivityEvent) -> str:
    raw_stdout = _verify_work_stdout_raw(event)
    if raw_stdout is not None:
        return raw_stdout
    text = str(event.data.get("content_preview") or "")
    lines = text.splitlines()
    if lines and lines[0].strip().upper() == "PASSED":
        lines = lines[1:]
    while lines and not lines[0].strip():
        lines = lines[1:]
    return "\n".join(lines).strip()


def _verify_work_stdout_matches(
    event: ActivityEvent,
    *,
    content: str,
    no_trailing_newline_required: bool,
) -> bool:
    raw_stdout = _verify_work_stdout_raw(event)
    if raw_stdout is not None:
        if no_trailing_newline_required:
            return raw_stdout == content
        return raw_stdout.rstrip("\n") == content
    if no_trailing_newline_required:
        return False
    return _verify_work_stdout(event) == content


def _command_asserts_exact_stdout(command: str, *, path: str, content: str) -> bool:
    if path not in command or content not in command:
        return False
    if _command_can_mask_assertion_failure(command):
        return False
    return any(
        _segment_asserts_exact_stdout(segment, path=path, content=content)
        for segment in _reachable_shell_segments(command)
    )


def _segment_asserts_exact_stdout(segment: str, *, path: str, content: str) -> bool:
    if path not in segment or content not in segment:
        return False
    escaped_content = re.escape(content)
    expected_expr = rf"(?:[\"']{escaped_content}[\"']|{escaped_content}(?=\s|\]|\)|$))"
    command_substitutions = _shell_command_substitutions(segment)
    for command_substitution, inner_command in command_substitutions:
        if not _shell_fragment_invokes_path(inner_command, path=path):
            continue
        if re.search(
            rf"[\"']?{re.escape(command_substitution)}[\"']?\s*(?:=|==)\s*{expected_expr}",
            segment,
            re.S,
        ) or re.search(
            rf"\btest\s+[\"']?{re.escape(command_substitution)}[\"']?\s*=\s*{expected_expr}",
            segment,
            re.S,
        ):
            return True
    return any(
        _shell_fragment_invokes_path(pipeline_segment, path=path)
        and re.search(
            rf"\bgrep\b[^\n;&|]*\s-[A-Za-z]*x[A-Za-z]*\b[^\n;&|]*{expected_expr}",
            pipeline_segments[index + 1],
            re.S,
        )
        for pipeline_segments in (_shell_pipeline_segments(segment),)
        for index, pipeline_segment in enumerate(pipeline_segments[:-1])
    )


def _shell_command_substitutions(command: str) -> list[tuple[str, str]]:
    substitutions: list[tuple[str, str]] = []
    index = 0
    while index < len(command):
        start = command.find("$(", index)
        if start < 0:
            break
        depth = 1
        cursor = start + 2
        in_single = False
        in_double = False
        escaped = False
        while cursor < len(command):
            char = command[cursor]
            if escaped:
                escaped = False
                cursor += 1
                continue
            if char == "\\" and not in_single:
                escaped = True
                cursor += 1
                continue
            if char == "'" and not in_double:
                in_single = not in_single
                cursor += 1
                continue
            if char == '"' and not in_single:
                in_double = not in_double
                cursor += 1
                continue
            if not in_single and not in_double:
                if command.startswith("$(", cursor):
                    depth += 1
                    cursor += 2
                    continue
                if char == ")":
                    depth -= 1
                    if depth == 0:
                        substitutions.append(
                            (command[start : cursor + 1], command[start + 2 : cursor])
                        )
                        cursor += 1
                        break
            cursor += 1
        index = max(cursor, start + 2)
    return substitutions


def _verify_work_event_asserts_exact_stdout_requests(
    event: ActivityEvent,
    *,
    requests: list[tuple[str, str]],
    no_trailing_newline_required: bool,
    source_events: list[ActivityEvent] | None = None,
) -> bool:
    if event.data.get("is_error") or verify_work_event_changes_state(event):
        return False
    if not _verify_work_exit_succeeded(event):
        return False
    if _verify_work_output_reports_failure(event):
        return False
    command = _verify_work_command(event)
    for path, content in requests:
        if no_trailing_newline_required:
            if _command_directly_runs_path(command, path=path) and _verify_work_stdout_matches(
                event,
                content=content,
                no_trailing_newline_required=True,
            ):
                continue
            if source_events is not None and _verify_work_runs_exact_stdout_assertion_script(
                event,
                source_events=source_events,
                path=path,
                content=content,
                no_trailing_newline_required=True,
            ):
                continue
            return False
        if _command_asserts_exact_stdout(command, path=path, content=content):
            continue
        if _command_directly_runs_path(command, path=path) and _verify_work_stdout_matches(
            event,
            content=content,
            no_trailing_newline_required=False,
        ):
            continue
        if source_events is not None and _verify_work_runs_exact_stdout_assertion_script(
            event,
            source_events=source_events,
            path=path,
            content=content,
            no_trailing_newline_required=no_trailing_newline_required,
        ):
            continue
        return False
    return True


def _verify_work_events_assert_exact_stdout_requests(
    events: list[ActivityEvent],
    *,
    requests: list[tuple[str, str]],
    no_trailing_newline_required: bool,
    source_events: list[ActivityEvent] | None = None,
) -> bool:
    sources = source_events if source_events is not None else events
    for path, content in requests:
        output_verified = False
        for event in events:
            if event.data.get("name") != "verify_work":
                continue
            if event.data.get("is_error") or verify_work_event_changes_state(event):
                continue
            if not _verify_work_exit_succeeded(event):
                continue
            if _verify_work_output_reports_failure(event):
                continue
            command = _verify_work_command(event)
            if not no_trailing_newline_required and _command_asserts_exact_stdout(
                command, path=path, content=content
            ):
                output_verified = True
            if _command_directly_runs_path(command, path=path) and _verify_work_stdout_matches(
                event,
                content=content,
                no_trailing_newline_required=no_trailing_newline_required,
            ):
                output_verified = True
            if _verify_work_runs_exact_stdout_assertion_script(
                event,
                source_events=sources,
                path=path,
                content=content,
                no_trailing_newline_required=no_trailing_newline_required,
            ):
                output_verified = True
        if not output_verified:
            return False
    return True


def _verify_work_event_asserts_exact_requests(
    event: ActivityEvent,
    *,
    requests: list[tuple[str, str]],
    byte_size_required: bool,
    source_events: list[ActivityEvent] | None = None,
) -> bool:
    if event.data.get("is_error") or verify_work_event_changes_state(event):
        return False
    if not _verify_work_exit_succeeded(event):
        return False
    if _verify_work_output_reports_failure(event):
        return False
    command = _verify_work_command(event)
    for path, content in requests:
        if _command_asserts_exact_content(command, path=path, content=content):
            continue
        if _command_inline_script_asserts_exact_content(
            command,
            path=path,
            content=content,
            byte_size_required=byte_size_required,
        ):
            continue
        if source_events is not None and _verify_work_runs_exact_assertion_script(
            event,
            source_events=source_events,
            path=path,
            content=content,
            byte_size_required=byte_size_required,
        ):
            continue
        return False
    if byte_size_required:
        for path, content in requests:
            if _command_asserts_byte_size(command, path=path, size=len(content.encode("utf-8"))):
                continue
            if _command_inline_script_asserts_exact_content(
                command,
                path=path,
                content=content,
                byte_size_required=True,
            ):
                continue
            if source_events is not None and _verify_work_runs_exact_assertion_script(
                event,
                source_events=source_events,
                path=path,
                content=content,
                byte_size_required=True,
            ):
                continue
            return False
    return True


def _verify_work_runs_exact_assertion_script(
    event: ActivityEvent,
    *,
    source_events: list[ActivityEvent],
    path: str,
    content: str,
    byte_size_required: bool,
) -> bool:
    command = _verify_work_command(event)
    script_path = _directly_run_script_path(command)
    if not script_path or not _is_test_path(script_path):
        return False
    script = _latest_written_content(source_events, script_path)
    if not script:
        return False
    return _script_asserts_exact_content(
        script,
        path=path,
        content=content,
        byte_size_required=byte_size_required,
    )


def _verify_work_runs_exact_stdout_assertion_script(
    event: ActivityEvent,
    *,
    source_events: list[ActivityEvent],
    path: str,
    content: str,
    no_trailing_newline_required: bool,
) -> bool:
    command = _verify_work_command(event)
    script_path = _directly_run_script_path(command)
    if not script_path or not _is_test_path(script_path):
        return False
    script = _latest_written_content(source_events, script_path)
    if not script:
        return False
    return _script_asserts_exact_stdout(
        script,
        path=path,
        content=content,
        no_trailing_newline_required=no_trailing_newline_required,
    )


def _latest_written_content(events: list[ActivityEvent], path: str) -> str:
    normalized_path = path.strip().lstrip("./")
    path_name = Path(normalized_path).name
    for event in reversed(events):
        if event.kind != "tool_call.completed" or event.data.get("is_error"):
            continue
        name = str(event.data.get("name") or "")
        if name not in _WRITE_TOOL_NAMES:
            continue
        changed_paths = _event_changed_paths(event)
        if normalized_path not in changed_paths and path_name not in changed_paths:
            continue
        metadata = event.data.get("metadata")
        if isinstance(metadata, dict) and isinstance(metadata.get("content_after"), str):
            return str(metadata["content_after"])
        arguments = event.data.get("arguments")
        if isinstance(arguments, dict):
            for key in ("content", "new", "replacement"):
                value = arguments.get(key)
                if isinstance(value, str):
                    return value
    return ""


def _script_asserts_exact_content(
    script: str,
    *,
    path: str,
    content: str,
    byte_size_required: bool,
) -> bool:
    normalized = path.strip().lstrip("./")
    path_name = Path(normalized).name
    if normalized not in script and path_name not in script:
        return False
    if content not in script:
        return False
    if ".read(" not in script:
        return False
    has_assertion = bool(
        re.search(r"\bassert\b[^=\n]*==", script)
        or "assertEqual" in script
        or re.search(r"!=[^\n]+raise\s+AssertionError", script, re.S)
    )
    if not has_assertion:
        return False
    if not byte_size_required:
        return True
    if re.search(r"\b(?:strip|rstrip)\s*\(", script):
        return bool(
            re.search(r"\b(?:len|byteLength)\s*\(", script)
            or re.search(r"\bendswith\s*\([^)]*\\n", script)
        )
    return bool(
        re.search(r"\b(?:len|byteLength)\s*\(", script)
        or re.search(r"\bendswith\s*\([^)]*\\n", script)
        or re.search(r"\bb['\"]", script)
        or "open(" in script
    )


def _script_asserts_exact_stdout(
    script: str,
    *,
    path: str,
    content: str,
    no_trailing_newline_required: bool,
) -> bool:
    normalized = path.strip().lstrip("./")
    path_name = Path(normalized).name
    if normalized not in script and path_name not in script:
        return False
    if content not in script:
        return False
    if "subprocess." not in script and "child_process" not in script:
        return False
    stdout_terms = ("stdout", "output")
    has_stdout_reference = any(term in script for term in stdout_terms)
    if not has_stdout_reference:
        return False
    if re.search(r"\b(?:strip|rstrip)\s*\(", script):
        return False
    has_exact_assertion = bool(
        re.search(r"\bassert\b[^\n]*(?:stdout|output)[^\n]*==[^\n]*" + re.escape(content), script)
        or re.search(
            r"assertEqual\s*\([^)]*(?:stdout|output)[^)]*" + re.escape(content),
            script,
        )
        or re.search(
            re.escape(content) + r"[^\n]*==[^\n]*(?:stdout|output)",
            script,
        )
    )
    if not has_exact_assertion:
        return False
    if not no_trailing_newline_required:
        return True
    return True


def _verify_work_events_assert_exact_requests(
    events: list[ActivityEvent],
    *,
    requests: list[tuple[str, str]],
    byte_size_required: bool,
    source_events: list[ActivityEvent] | None = None,
) -> bool:
    sources = source_events if source_events is not None else events
    for path, content in requests:
        content_asserted = False
        byte_size_asserted = not byte_size_required
        for event in events:
            if event.data.get("name") != "verify_work":
                continue
            if event.data.get("is_error") or verify_work_event_changes_state(event):
                continue
            if not _verify_work_exit_succeeded(event):
                continue
            if _verify_work_output_reports_failure(event):
                continue
            command = _verify_work_command(event)
            if _command_asserts_exact_content(command, path=path, content=content):
                content_asserted = True
            if _command_inline_script_asserts_exact_content(
                command,
                path=path,
                content=content,
                byte_size_required=byte_size_required,
            ):
                content_asserted = True
                byte_size_asserted = True
            if _verify_work_runs_exact_assertion_script(
                event,
                source_events=sources,
                path=path,
                content=content,
                byte_size_required=byte_size_required,
            ):
                content_asserted = True
                byte_size_asserted = True
            if byte_size_required and _command_asserts_byte_size(
                command,
                path=path,
                size=len(content.encode("utf-8")),
            ):
                byte_size_asserted = True
        if not content_asserted or not byte_size_asserted:
            return False
    return True


def deterministic_task_has_verified_evidence(
    *, session: Session, activity: list[ActivityEvent]
) -> bool:
    user_prompt = latest_user_prompt(session)
    exact_requests = _exact_file_content_requests(user_prompt)
    stdout_requests = _exact_stdout_requests(user_prompt)
    if not exact_requests and not stdout_requests:
        return False

    tool_events = [e for e in activity if e.kind == "tool_call.completed"]
    state_change_indexes = [
        index
        for index, event in enumerate(tool_events)
        if tool_event_changes_state(event, _WRITE_TOOL_NAMES)
        or verify_work_event_changes_state(event)
    ]
    if not state_change_indexes:
        return False

    byte_size_required = bool(
        _BYTE_SIZE_REQUEST_RE.search(user_prompt)
        or (exact_requests and _NO_TRAILING_NEWLINE_RE.search(user_prompt))
    )
    stdout_no_trailing_newline_required = bool(
        stdout_requests and _NO_TRAILING_NEWLINE_RE.search(user_prompt)
    )
    later_events = tool_events[state_change_indexes[-1] + 1 :]
    file_ok = not exact_requests or _verify_work_events_assert_exact_requests(
        later_events,
        requests=exact_requests,
        byte_size_required=byte_size_required,
        source_events=tool_events,
    )
    stdout_ok = not stdout_requests or _verify_work_events_assert_exact_stdout_requests(
        later_events,
        requests=stdout_requests,
        no_trailing_newline_required=stdout_no_trailing_newline_required,
        source_events=tool_events,
    )
    return file_ok and stdout_ok


def exact_file_task_has_verified_evidence(
    *, session: Session, activity: list[ActivityEvent]
) -> bool:
    return deterministic_task_has_verified_evidence(session=session, activity=activity)


def first_user_prompt(session: Session) -> str:
    for msg in session.messages:
        if getattr(msg, "role", None) == "user" and msg.content:
            return msg.content
    return ""


def latest_user_prompt(session: Session) -> str:
    for msg in reversed(session.messages):
        if getattr(msg, "role", None) == "user" and msg.content:
            return msg.content
    return ""


def latest_assistant_message(session: Session) -> str:
    for msg in reversed(session.messages):
        if getattr(msg, "role", None) == "assistant" and msg.content:
            return msg.content
    return ""


def _assistant_final_requests_user_decision(session: Session) -> bool:
    text = " ".join(latest_assistant_message(session).lower().split())
    if not text:
        return False
    direct_requests = (
        "please confirm",
        "reply with",
        "once you confirm",
        "wait for your confirmation",
        "need your confirmation",
        "need confirmation",
        "what i need from you",
        "need from you to proceed",
        "cannot proceed until",
        "blocked until",
    )
    if any(phrase in text for phrase in direct_requests):
        return True
    if "?" not in text:
        return False
    question_parts = (
        "should i ",
        "can i proceed",
        "do you want me to",
        "which path",
        "which option",
        "choose one",
        "choose either",
    )
    return any(part in text for part in question_parts)


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


def _prompt_requires_public_source_evidence(prompt: str) -> bool:
    if not prompt.strip():
        return False
    for segment in _public_source_prompt_segments(prompt):
        if (
            _CURRENT_FACT_RE.search(segment)
            and _PUBLIC_SOURCE_RE.search(segment)
            and _PUBLIC_FACT_SUBJECT_RE.search(segment)
        ):
            return True
    return False


def _public_source_prompt_segments(prompt: str) -> list[str]:
    segments: list[str] = []
    for line in prompt.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        segments.append(stripped)
        segments.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", stripped))
    return [segment for segment in segments if segment]


def _has_successful_public_source_evidence(activity: list[ActivityEvent]) -> bool:
    for event in activity:
        if event.kind != "tool_call.completed" or event.data.get("is_error"):
            continue
        name = str(event.data.get("name") or "")
        if name in _PUBLIC_SOURCE_EVIDENCE_TOOLS:
            return True
        if name in _SHELL_TOOL_NAMES and _shell_command_fetches_public_source(
            _shell_command(event)
        ):
            return True
    return False


def _shell_command_fetches_public_source(command: str, *, _depth: int = 0) -> bool:
    for segment in _reachable_shell_segments(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            continue
        tokens = _strip_leading_env_assignments(tokens)
        if not tokens:
            continue
        executable = Path(tokens[0]).name.lower()
        if executable in _SHELL_WRAPPER_TOOLS and _depth < 3:
            nested = _shell_wrapper_command_body(tokens)
            if nested and _shell_command_fetches_public_source(nested, _depth=_depth + 1):
                return True
        if executable not in _PUBLIC_SOURCE_SHELL_FETCH_TOOLS:
            continue
        if any(token.startswith(("http://", "https://")) for token in tokens[1:]):
            return True
    return False


def _shell_wrapper_command_body(tokens: list[str]) -> str:
    for index, token in enumerate(tokens[1:], start=1):
        if not token.startswith("-"):
            continue
        if "c" not in token:
            continue
        if index + 1 < len(tokens):
            return tokens[index + 1]
    return ""


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

    def __init__(self, *verifiers: Any, fail_fast: bool = True) -> None:
        self._verifiers = list(verifiers)
        self._fail_fast = fail_fast

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        last: VerificationResult | None = None
        failures: list[VerificationResult] = []
        for verifier in self._verifiers:
            result = await verifier.verify(session=session, activity=activity)
            last = result
            if not result.can_finish:
                if self._fail_fast:
                    return VerificationResult(
                        can_finish=False,
                        reason=result.reason,
                        confidence=result.confidence,
                        evidence_event_ids=result.evidence_event_ids,
                        verifier_name=self.name,
                    )
                failures.append(result)
        if failures:
            if len(failures) == 1:
                failure = failures[0]
                return VerificationResult(
                    can_finish=False,
                    reason=failure.reason,
                    confidence=failure.confidence,
                    evidence_event_ids=failure.evidence_event_ids,
                    verifier_name=self.name,
                )
            reason = "\n\n".join(
                f"{index}. {failure.reason}" for index, failure in enumerate(failures, start=1)
            )
            evidence_ids: list[str] = []
            for failure in failures:
                evidence_ids.extend(failure.evidence_event_ids)
            return VerificationResult(
                can_finish=False,
                reason=f"Multiple independent verification checks failed:\n\n{reason}",
                confidence=max(failure.confidence or 0.0 for failure in failures),
                evidence_event_ids=evidence_ids,
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

    def __init__(
        self,
        write_tool_names: frozenset[str] | None = None,
        *,
        default_verify_command_available: bool = False,
    ) -> None:
        self._writes = write_tool_names if write_tool_names is not None else _WRITE_TOOL_NAMES
        self._default_verify_command_available = default_verify_command_available

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        tool_events = [e for e in activity if e.kind == "tool_call.completed"]
        user_prompt = latest_user_prompt(session)
        exact_requests = _exact_file_content_requests(user_prompt)
        stdout_requests = _exact_stdout_requests(user_prompt)
        byte_size_required = bool(
            _BYTE_SIZE_REQUEST_RE.search(user_prompt)
            or (exact_requests and _NO_TRAILING_NEWLINE_RE.search(user_prompt))
        )
        stdout_no_trailing_newline_required = bool(
            stdout_requests and _NO_TRAILING_NEWLINE_RE.search(user_prompt)
        )

        if _is_promotion_artifact_pr_flow(tool_events):
            return VerificationResult(
                can_finish=True,
                reason=(
                    "Only generated research promotion artifacts were modified and the "
                    "PR flow was completed — generic verify_work is not required."
                ),
                verifier_name=self.name,
            )

        state_change_indexes = [
            index
            for index, event in enumerate(tool_events)
            if tool_event_changes_state(event, self._writes)
            or verify_work_event_changes_state(event)
        ]
        if not state_change_indexes:
            if exact_requests or stdout_requests:
                file_ok = not exact_requests or _verify_work_events_assert_exact_requests(
                    tool_events,
                    requests=exact_requests,
                    byte_size_required=byte_size_required,
                    source_events=tool_events,
                )
                stdout_ok = not stdout_requests or _verify_work_events_assert_exact_stdout_requests(
                    tool_events,
                    requests=stdout_requests,
                    no_trailing_newline_required=stdout_no_trailing_newline_required,
                    source_events=tool_events,
                )
                if file_ok and stdout_ok:
                    return VerificationResult(
                        can_finish=True,
                        reason="passing verify_work evidence confirmed the exact request.",
                        verifier_name=self.name,
                    )
                return VerificationResult(
                    can_finish=False,
                    reason=(
                        "The user requested an exact file or exact program output, but no "
                        "workspace-changing tool call or passing verify_work evidence was "
                        "observed. Create or inspect the requested artifact with tools, then "
                        "call verify_work with a command whose exit code proves the exact "
                        "content or output."
                    ),
                    verifier_name=self.name,
                )
            return VerificationResult(
                can_finish=True,
                reason="No modifying tool calls detected — verification not required.",
                verifier_name=self.name,
            )

        verify_calls = [e for e in tool_events if e.data.get("name") == "verify_work"]
        if not verify_calls:
            if _assistant_final_requests_user_decision(session):
                return VerificationResult(
                    can_finish=False,
                    reason=(
                        "The final response asks the user to choose, confirm, or approve "
                        "the next implementation step after workspace changes. Continue "
                        "autonomously with the available tools and verify the result instead "
                        "of handing the work back to the user."
                    ),
                    verifier_name=self.name,
                )
            default_hint = (
                " A configured default verifier is available."
                if self._default_verify_command_available
                else ""
            )
            return VerificationResult(
                can_finish=False,
                reason=(f"You made file changes but never ran verify_work.{default_hint}"),
                verifier_name=self.name,
            )

        last_state_change_index = state_change_indexes[-1]
        changed_paths_since_last_verify: set[str] = set()
        for event in tool_events[: last_state_change_index + 1]:
            if tool_event_changes_state(event, self._writes) or verify_work_event_changes_state(
                event
            ):
                changed_paths_since_last_verify.update(_event_changed_paths(event))
        for event in tool_events[last_state_change_index + 1 :]:
            if event.data.get("name") != "verify_work":
                continue
            if event.data.get("is_error"):
                continue
            if verify_work_event_changes_state(event):
                continue
            if (
                not exact_requests
                and not stdout_requests
                and not verify_work_event_has_generic_evidence(
                    event,
                    changed_paths=changed_paths_since_last_verify,
                )
            ):
                continue
            if exact_requests and not _verify_work_event_asserts_exact_requests(
                event,
                requests=exact_requests,
                byte_size_required=byte_size_required,
                source_events=tool_events[: last_state_change_index + 1],
            ):
                continue
            if stdout_requests and not _verify_work_event_asserts_exact_stdout_requests(
                event,
                requests=stdout_requests,
                no_trailing_newline_required=stdout_no_trailing_newline_required,
                source_events=tool_events[: last_state_change_index + 1],
            ):
                continue
            return VerificationResult(
                can_finish=True,
                reason="passing verify_work ran after the last state change.",
                verifier_name=self.name,
            )
        if _assistant_final_requests_user_decision(session):
            return VerificationResult(
                can_finish=False,
                reason=(
                    "The final response asks the user to choose, confirm, or approve "
                    "the next implementation step, but the changed workspace does not "
                    "have acceptable verification after the final change. Continue "
                    "autonomously with the available tools and verify the result instead "
                    "of handing the work back to the user."
                ),
                verifier_name=self.name,
            )
        later_events = tool_events[last_state_change_index + 1 :]
        latest_later_verify = next(
            (event for event in reversed(later_events) if event.data.get("name") == "verify_work"),
            None,
        )
        if latest_later_verify is not None and latest_later_verify.data.get("is_error"):
            return VerificationResult(
                can_finish=False,
                reason="The latest verify_work after the final state change failed.",
                verifier_name=self.name,
            )
        if (
            latest_later_verify is not None
            and not exact_requests
            and not stdout_requests
            and not verify_work_event_has_generic_evidence(
                latest_later_verify,
                changed_paths=changed_paths_since_last_verify,
            )
        ):
            return VerificationResult(
                can_finish=False,
                reason=(
                    "The latest passing verify_work after the final state change did not "
                    "run a meaningful test/check command tied to the changed work."
                ),
                verifier_name=self.name,
            )
        file_ok = not exact_requests or _verify_work_events_assert_exact_requests(
            later_events,
            requests=exact_requests,
            byte_size_required=byte_size_required,
            source_events=tool_events[: last_state_change_index + 1],
        )
        stdout_ok = not stdout_requests or _verify_work_events_assert_exact_stdout_requests(
            later_events,
            requests=stdout_requests,
            no_trailing_newline_required=stdout_no_trailing_newline_required,
            source_events=tool_events[: last_state_change_index + 1],
        )
        if (exact_requests or stdout_requests) and file_ok and stdout_ok:
            return VerificationResult(
                can_finish=True,
                reason="passing verify_work evidence ran after the last state change.",
                verifier_name=self.name,
            )

        exact_hint = ""
        if exact_requests:
            example_path, example_content = exact_requests[0]
            quoted_content = "'" + example_content.replace("'", "'\"'\"'") + "'"
            content_check = f'test "$(cat {shlex.quote(example_path)})" = {quoted_content}'
            if byte_size_required:
                byte_check = (
                    f"test $(wc -c < {shlex.quote(example_path)}) -eq "
                    f"{len(example_content.encode('utf-8'))}"
                )
                example = f"{content_check} && {byte_check}"
            else:
                example = content_check
            exact_hint = (
                " For exact-content tasks, verify_work must assert the requested file content"
                + (" and byte size." if byte_size_required else ".")
                + f" Use a command whose exit code depends on the assertion, e.g. `{example}`. "
                "Do not append `|| true` or `|| echo ...`, because that can hide failures."
            )
        elif stdout_requests:
            example_path, example_content = stdout_requests[0]
            quoted_content = "'" + example_content.replace("'", "'\"'\"'") + "'"
            example = f'test "$(./{shlex.quote(example_path)})" = {quoted_content}'
            if stdout_no_trailing_newline_required:
                exact_hint = (
                    " For exact-output tasks with no trailing newline, verify_work must run "
                    "the program directly so Harness can inspect raw stdout; shell command "
                    "substitution can hide trailing newlines."
                )
            else:
                exact_hint = (
                    " For exact-output tasks, verify_work must run the program and prove the "
                    "requested stdout. Use a command whose exit code depends on the output, "
                    f"e.g. `{example}`, or run the program and produce exactly the requested stdout."
                )
        return VerificationResult(
            can_finish=False,
            reason=(
                "You changed files or state after the last passing read-only verify_work. "
                "Call verify_work with a non-mutating test/check command after the final "
                "change whose exit code depends on the changed work, read the output, "
                "and fix any failures before finishing. The assertion must inspect the "
                "changed file, command output, or behavior."
                f"{exact_hint}"
            ),
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


class PublicSourceEvidenceVerifier:
    name = "public_source_evidence"

    async def verify(
        self, *, session: Session, activity: list[ActivityEvent]
    ) -> VerificationResult:
        prompt = first_user_prompt(session)
        if not _prompt_requires_public_source_evidence(prompt):
            return VerificationResult(
                can_finish=True,
                reason="task does not require latest/current public-source evidence",
                confidence=0.6,
                verifier_name=self.name,
            )
        if _has_successful_public_source_evidence(activity):
            return VerificationResult(
                can_finish=True,
                reason="latest/current public-source task has successful web evidence",
                confidence=0.95,
                verifier_name=self.name,
            )
        return VerificationResult(
            can_finish=False,
            reason=(
                "This task asks for a latest/current fact grounded in public sources. "
                "Use web_search or fetch_url to retrieve an official or public source. "
                "If this provider only exposes shell tools, run a read-only HTTP(S) "
                "fetch such as curl or wget against the official source. Then base the "
                "update and verification on that evidence before finishing."
            ),
            confidence=0.95,
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

        if _has_post_edit_regression_verification(
            tool_events,
            first_edit_idx=first_edit_idx,
            write_tool_names=self._writes,
        ):
            return VerificationResult(
                can_finish=True,
                reason=(
                    "post-edit regression-test workflow — a test file changed and "
                    "a successful test run happened before finish"
                ),
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
