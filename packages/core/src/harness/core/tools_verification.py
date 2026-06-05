"""Agent-callable verification and critique tools.

These tools are always registered in the agent's tool registry — no flags
required. They give the LLM the ability to self-verify and self-critique
proactively during its work, not just receive feedback at the end.

``verify_work``
    Run a verification command chosen by the agent (language-agnostic). The
    agent supplies the command; the tool runs it and returns stdout+stderr plus
    a pass/fail verdict. Designed to be called BEFORE the agent declares done.

``request_critique``
    Ask a second LLM reviewer to challenge the agent's proposed approach. The
    agent describes what it plans to do and why; the critic returns a pointed
    question or objection the agent must address before proceeding. Designed
    for moments of uncertainty: "am I about to fix the right thing?"
"""

from __future__ import annotations

import asyncio
import re
import shlex
from pathlib import Path
from typing import Any

from harness.core.adapter import Adapter
from harness.core.command_env import clean_command_env
from harness.core.critic import SearchFn
from harness.core.events import Done, TextDelta
from harness.core.schemas import ApprovalDecision, Message, ToolCall, ToolResult

_CRITIQUE_SYSTEM = """\
You are a code review critic. An AI agent is about to make a change and wants \
a second opinion.

Your job: identify whether the agent's proposed approach actually addresses the \
problem as described, then ask a pointed question if something looks wrong.

Rules:
- Be concise: 3-5 sentences maximum
- If the approach looks correct, say so briefly and confirm the agent should proceed
- If the approach has a flaw: name it and ask one specific question the agent must \
answer before proceeding
- Do NOT provide the correct solution unprompted
- Tone: direct and collegial — "have you considered..." not "you are wrong"\
"""

_CRITIQUE_USER = """\
## Proposed approach

{approach}

## Problem context / failure output

{context}

Does this approach address what the problem actually requires? Critique in 3-5 \
sentences. If you spot a flaw, ask one specific question.\
"""


def _verify_schema(*, has_default_command: bool) -> dict[str, Any]:
    command_description = (
        "Optional verification command override. Omit this when a configured "
        "default verifier is available and you want to run that authoritative "
        "check. Otherwise provide a repository command whose exit code verifies "
        "the current workspace state; this may be a containerized command when "
        "that is the available project runtime."
        if has_default_command
        else (
            "The repository command to run. Its exit code must verify the current "
            "workspace state; this may be a containerized command when that is "
            "the available project runtime."
        )
    )
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": command_description,
            },
        },
    }
    if not has_default_command:
        schema["required"] = ["command"]
    return schema


_VERIFY_SCHEMA: dict[str, Any] = _verify_schema(has_default_command=False)

_VERIFY_DEFAULT_DESCRIPTION = (
    " A default verifier command is configured for this run; omitting command runs that verifier."
)

_VERIFY_BASE_DESCRIPTION = (
    "Run a repository verification command against the current workspace. "
    "Commands run in a clean command environment rather than Harness' ambient "
    "virtualenv or provider process environment. "
    "If a containerized command is the available project runtime, use that "
    "same read-only command here for final verification. "
    "The command must report failure through its exit code."
)


_GIT_INSPECTION_SUBCOMMANDS = {
    "branch",
    "diff",
    "log",
    "ls-files",
    "rev-parse",
    "show",
    "status",
}
_NONZERO_EXIT_RE = re.compile(r"\b(?:exit|return)\s+[1-9]\d*\b")
_GIT_STATUS_COMMAND = ("git", "status", "--porcelain", "--untracked-files=all")


def _failure_branch_masks_exit_status(command: str) -> bool:
    command_without_comments = _strip_shell_comments(command)
    if _command_echoes_shell_status_after_sequence(command_without_comments):
        return True
    masked_command = _mask_shell_quoted_text(command_without_comments)
    if _conditional_failure_branch_masks_exit_status(masked_command):
        return True
    if _loop_condition_masks_exit_status(masked_command):
        return True
    if _for_loop_masks_exit_status(command_without_comments):
        return True
    if _case_statement_masks_exit_status(masked_command):
        return True
    for nested_command in _nested_shell_c_commands(command_without_comments):
        if _failure_branch_masks_exit_status(nested_command):
            return True
    for branch in _split_unquoted_double_pipe(command_without_comments)[1:]:
        stripped = branch.lstrip()
        if not stripped:
            continue
        if not _branch_exits_nonzero(_mask_shell_quoted_text(stripped)):
            return True
    return False


def _command_echoes_shell_status_after_sequence(command: str) -> bool:
    masked_command = _mask_shell_quoted_text(command)
    return bool(
        re.search(
            r"(?:;|\n)\s*(?:echo|printf)\b[^\n;&|]*\$\?",
            masked_command,
        )
    )


def _nested_shell_c_commands(command: str) -> list[str]:
    nested: list[str] = []
    for segment in _shell_command_segments(command):
        try:
            tokens = shlex.split(segment)
        except ValueError:
            continue
        tokens = _tokens_after_env(tokens)
        for index, token in enumerate(tokens[:-2]):
            executable = Path(token).name.lower()
            if executable not in {"bash", "sh", "zsh"}:
                continue
            option = tokens[index + 1]
            if option == "-c" or (option.startswith("-") and "c" in option):
                nested.append(tokens[index + 2])
    return nested


def _strip_shell_comments(command: str) -> str:
    lines: list[str] = []
    for line in command.splitlines():
        current: list[str] = []
        quote = ""
        escaped = False
        index = 0
        while index < len(line):
            char = line[index]
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
            if char == "#" and (not current or current[-1].isspace()):
                break
            current.append(char)
            index += 1
        lines.append("".join(current).rstrip())
    return "\n".join(lines)


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
            masked.append("\n" if char == "\n" else " ")
            continue
        if char in {"'", '"'}:
            quote = char
        masked.append(char)
    return "".join(masked)


def _split_unquoted_double_pipe(command: str) -> list[str]:
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
        if char == "#" and (not current or current[-1].isspace()):
            while index < len(command) and command[index] != "\n":
                current.append(command[index])
                index += 1
            continue
        if command.startswith("||", index):
            pieces.append("".join(current).strip())
            current = []
            index += 2
            continue
        current.append(char)
        index += 1
    pieces.append("".join(current).strip())
    return pieces


def _command_exits_before_trailing_command(command: str) -> bool:
    saw_exiting_segment = False
    for segment in _shell_command_segments(command):
        if saw_exiting_segment:
            return True
        if _segment_exits_shell(segment):
            saw_exiting_segment = True
    return False


def _verification_command_is_noop(command: str) -> bool:
    segments = _shell_command_segments(command)
    if not segments:
        return True
    return all(_segment_is_noop_verification(segment) for segment in segments)


def _shell_command_segments(command: str) -> list[str]:
    segments: list[str] = []
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
        if char == "#" and (not current or (current[-1].isspace())):
            while index < len(command) and command[index] != "\n":
                index += 1
            continue
        if command.startswith("&&", index) or command.startswith("||", index):
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            index += 2
            continue
        if char in {";", "\n"}:
            segment = "".join(current).strip()
            if segment:
                segments.append(segment)
            current = []
            index += 1
            continue
        current.append(char)
        index += 1
    segment = "".join(current).strip()
    if segment:
        segments.append(segment)
    return segments


def _segment_is_noop_verification(segment: str) -> bool:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return False
    tokens = _tokens_after_env(tokens)
    if not tokens:
        return True
    executable = Path(tokens[0]).name.lower()
    if executable in {"cd", "pwd", "true", ":"}:
        return True
    if executable in {"exit", "return"}:
        if len(tokens) == 1:
            return True
        try:
            return int(tokens[1]) == 0
        except ValueError:
            return False
    if executable in {"bash", "sh", "zsh"}:
        for index, token in enumerate(tokens[:-1]):
            if token in {"-c", "-lc"}:
                return _verification_command_is_noop(tokens[index + 1])
    if executable in {"docker", "podman", "nerdctl"}:
        return _container_command_is_noop_verification(tokens)
    if executable == "git" and len(tokens) > 1:
        return tokens[1].lower() in _GIT_INSPECTION_SUBCOMMANDS
    return False


def _container_command_is_noop_verification(tokens: list[str]) -> bool:
    if not any(token == "run" for token in tokens[1:]):
        return False
    for index, token in enumerate(tokens[1:], start=1):
        executable = Path(token).name.lower()
        if executable not in {"bash", "sh", "zsh"}:
            continue
        shell_command = " ".join(shlex.quote(part) for part in tokens[index:])
        if _verification_command_is_noop(shell_command):
            return True
    return False


def _tokens_after_env(tokens: list[str]) -> list[str]:
    index = 0
    if tokens and Path(tokens[0]).name == "env":
        index = 1
    while index < len(tokens):
        token = tokens[index]
        if token.startswith("-"):
            index += 1
            continue
        if "=" in token and not token.startswith("="):
            name = token.split("=", 1)[0]
            if name.replace("_", "").isalnum():
                index += 1
                continue
        break
    return tokens[index:]


def _segment_exits_shell(segment: str) -> bool:
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return False
    if not tokens:
        return False
    executable = Path(tokens[0]).name.lower()
    return executable in {"exit", "return"}


def _branch_exits_nonzero(branch: str) -> bool:
    return bool(_NONZERO_EXIT_RE.search(branch) or re.search(r"(?:^|[;&|({]\s*)false\b", branch))


def _conditional_failure_branch_masks_exit_status(command: str) -> bool:
    for match in re.finditer(
        r"\bif\s+!\s+.+?(?:;|\n)\s*then\s+(?P<branch>.+?)(?:\belse\b|\bfi\b)",
        command,
        re.DOTALL,
    ):
        if not _branch_exits_nonzero(match.group("branch")):
            return True
    for match in re.finditer(
        r"\bif\s+.+?(?:;|\n)\s*then\s+.+?\belse\b\s+(?P<branch>.+?)\bfi\b",
        command,
        re.DOTALL,
    ):
        if not _branch_exits_nonzero(match.group("branch")):
            return True
    for match in re.finditer(
        r"\bif\s+(?!\!)\s*.+?(?:;|\n)\s*then\s+.+?\bfi\b",
        command,
        re.DOTALL,
    ):
        if " else " not in re.sub(r"\s+", " ", match.group(0)):
            return True
    return False


def _loop_condition_masks_exit_status(command: str) -> bool:
    return bool(
        re.search(
            r"\b(?:while|until)\s+.+?(?:;|\n)\s*do\s+.+?\bdone\b",
            command,
            re.DOTALL,
        )
    )


def _for_loop_masks_exit_status(command: str) -> bool:
    masked_command = _mask_shell_quoted_text(command)
    for _match in re.finditer(
        r"\bfor\s+[A-Za-z_][A-Za-z0-9_]*\s*(?:;|\n)\s*do\s+.+?\bdone\b",
        masked_command,
        re.DOTALL,
    ):
        return True
    for match in re.finditer(
        r"\bfor\s+[A-Za-z_][A-Za-z0-9_]*\s+in(?P<items>.*?)(?:;|\n)\s*do\s+.+?\bdone\b",
        masked_command,
        re.DOTALL,
    ):
        raw_items = command[match.start("items") : match.end("items")]
        if not raw_items.strip():
            return True
        if _shell_fragment_has_runtime_expansion(raw_items):
            return True
    return False


def _shell_fragment_has_runtime_expansion(fragment: str) -> bool:
    quote = ""
    escaped = False
    for char in fragment:
        if escaped:
            escaped = False
            continue
        if char == "\\" and quote != "'":
            escaped = True
            continue
        if quote:
            if char == quote:
                quote = ""
            elif quote == '"' and char in {"$", "`"}:
                return True
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char in {"$", "`"}:
            return True
    return False


def _case_statement_masks_exit_status(command: str) -> bool:
    for match in re.finditer(
        r"\bcase\s+.+?\bin\s+(?P<body>.+?)\besac\b",
        command,
        re.DOTALL,
    ):
        if not _case_default_branch_exits_nonzero(match.group("body")):
            return True
    return False


def _case_default_branch_exits_nonzero(body: str) -> bool:
    for match in re.finditer(r"(?:^|;;|;|\n)\s*\*\)\s*(?P<branch>.*?)(?:;;|$)", body, re.DOTALL):
        if _branch_exits_nonzero(match.group("branch")):
            return True
    return False


def _normalize_successful_package_sweep_output(output: str) -> str:
    """Remove no-test package rows when a package sweep has real passes too."""
    real_passing_package = any(
        re.match(r"^\s*ok\s+\S+", line)
        and not re.search(r"\[no\s+tests?\s+to\s+run\]", line, re.IGNORECASE)
        for line in output.splitlines()
    )
    if not real_passing_package:
        return output
    return "\n".join(
        line
        for line in output.splitlines()
        if not re.match(r"^\s*\?\s+\S+\s+\[no\s+test\s+files?\]\s*$", line, re.IGNORECASE)
        and not re.match(
            r"^\s*ok\s+\S+\s+.*\s+\[no\s+tests?\s+to\s+run\]\s*$",
            line,
            re.IGNORECASE,
        )
    )


def _output_reports_failure(output: str) -> bool:
    normalized = _normalize_successful_package_sweep_output(output).lower()
    if not normalized.strip():
        return False
    if re.search(
        r"(?im)^\s*(?:[a-z][\w-]*[_ -])?(?:exit(?:[_ -]?code)?|status|rc)\s*[:=]\s*[1-9]\d*\s*$",
        normalized,
    ):
        return True
    skip_words = r"(?:skipped|skip|ignored|pending|todo|deselected|excluded)"
    no_test_patterns = (
        r"\bcollected\s+0\s+items\b",
        r"\bno\s+tests?\s+(?:ran|run|found|collected|executed|discovered)\b",
        r"\bno\s+tests?\s+to\s+run\b",
        r"\bno\s+(?:matching\s+)?tests?\b",
        r"\bfound\s+0\s+tests?\b",
        r"\b0\s+tests?\s+(?:ran|run|found|executed|discovered)\b",
        r"\brunning\s+0\s+tests?\b",
        r"\b0\s+(?:passed|passing)\b",
        r"\b(?:tests?|test\s+suites?|specs?|suites?|checks?)\s*[:=]\s*0\s+total\b",
        r"\b0\s+total\s+(?:tests?|test\s+suites?|specs?|suites?|checks?)\b",
        r"\b0\s+examples?,\s+0\s+failures?\b",
        r"\bno\s+test\s+files?\s+found\b",
        r"\[no\s+test\s+files?\]",
        rf"\ball\s+(?:tests?|checks?|specs?|suites?)\s+(?:were\s+)?{skip_words}\b",
    )
    if any(re.search(pattern, normalized) for pattern in no_test_patterns):
        return True
    line_failure_patterns = (
        r"(?m)^\s*---\s+fail:",
        r"(?m)^\s*fail(?:\s|$)",
    )
    if any(re.search(pattern, normalized) for pattern in line_failure_patterns):
        return True
    runtime_error_patterns = (
        r"\btraceback\s+\(most recent call last\):",
        r"\bassertionerror\b",
        r"\b(?:uncaught|unhandled)\s+exception\b",
        r"\bhere-document\b.*\bdelimited\s+by\s+end-of-file\b",
        r"(?m)^\s*(?:error|fatal|[a-z][a-z]+error):\s+\S",
    )
    if any(re.search(pattern, normalized) for pattern in runtime_error_patterns):
        return True
    if re.search(r"\b[1-9]\d*\s+errors?\b", normalized):
        return True
    if re.search(r"\berrors?\s*[:=]\s*[1-9]\d*\b", normalized):
        return True
    zero_pass_with_skips = (
        rf"\b0\s+passed\b.*\b[1-9]\d*\s+{skip_words}\b"
        rf"|\b[1-9]\d*\s+{skip_words}\b.*\b0\s+passed\b"
        rf"|\bpassed\s*[:=]\s*0\b.*\b{skip_words}\s*[:=]\s*[1-9]\d*\b"
        rf"|\b{skip_words}\s*[:=]\s*[1-9]\d*\b.*\bpassed\s*[:=]\s*0\b"
    )
    if re.search(zero_pass_with_skips, normalized):
        return True
    positive_pass_count = bool(
        re.search(r"\b[1-9]\d*\s+(?:passed|passing)\b", normalized)
        or re.search(r"\bpassed\s*[:=]\s*[1-9]\d*\b", normalized)
    )
    skip_count = bool(
        re.search(rf"\b[1-9]\d*\s+{skip_words}\b", normalized)
        or re.search(rf"\b{skip_words}\s*[:=]\s*[1-9]\d*\b", normalized)
    )
    if skip_count and not positive_pass_count:
        return True
    if re.search(r"\b[1-9]\d*\s+(?:failed|failures?)\b", normalized):
        return True
    if re.search(r"\b(?:failed|failures?)\s*[:=]\s*[1-9]\d*\b", normalized):
        return True
    line_failure_patterns = (
        r"(?m)^\s*verification\s+failed\b",
        r"(?m)^\s*(?:failed|failure)\s*$",
        r"(?m)^\s*failed(?:\s|$)",
        r"(?m)^\s*failed\s+(?:tests?|specs?|checks?|examples?)\b",
        r"(?m)^\s*=+\s*(?:failures?|failed)\s*=+\s*$",
        r"(?m)^\s*(?:failures?|failed)\s*[:=]\s*[1-9]\d*\b",
        r"(?m)^\s*(?:tests?|test\s+suites?|specs?|checks?)\s*[:=].*\b[1-9]\d*\s+(?:failed|failures?)\b",
    )
    return any(re.search(pattern, normalized) for pattern in line_failure_patterns)


def _successful_shell_stderr_reports_failure(
    *,
    exit_code: int,
    stderr: str,
) -> bool:
    if exit_code != 0:
        return False
    normalized = stderr.lower()
    if not normalized.strip():
        return False
    shell_failure_patterns = (
        r"\btraceback\s+\(most recent call last\):",
        r"\bassertionerror\b",
        r"\bsyntax error\b",
        r"\bhere-document\b.*\bdelimited\s+by\s+end-of-file\b",
        r"\bcommand not found\b",
        r"\bno such file or directory\b",
        r"\bpermission denied\b",
    )
    return any(re.search(pattern, normalized) for pattern in shell_failure_patterns)


def _porcelain_paths(status: str) -> list[tuple[str, str]]:
    paths: list[tuple[str, str]] = []
    for raw_line in status.splitlines():
        if not raw_line.strip():
            continue
        code = raw_line[:2]
        path = raw_line[3:].strip()
        if " -> " in path:
            path = path.rsplit(" -> ", 1)[-1].strip()
        if path:
            paths.append((code, path))
    return paths


def _path_is_generated_artifact(path: str) -> bool:
    parts = [part.lower() for part in path.strip("/").split("/") if part]
    return any(_path_part_looks_generated(part) for part in parts)


def _path_part_looks_generated(part: str) -> bool:
    normalized = part.strip("._-")
    return normalized == "cache" or normalized.endswith("cache")


def _relevant_status_entries(status: str) -> tuple[tuple[str, str], ...]:
    return tuple(
        (code, path)
        for code, path in _porcelain_paths(status)
        if not _path_is_generated_artifact(path)
    )


async def _git_status_entries(cwd: Path) -> tuple[tuple[str, str], ...] | None:
    proc = await asyncio.create_subprocess_exec(
        *_GIT_STATUS_COMMAND,
        cwd=cwd,
        env=clean_command_env(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode != 0:
        return None
    return _relevant_status_entries(stdout.decode(errors="replace"))


_CRITIQUE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "approach": {
            "type": "string",
            "description": "Describe what you plan to change and why you think it fixes the problem.",
        },
        "context": {
            "type": "string",
            "description": "Paste the relevant error, test failure, or problem description you're working with.",
        },
    },
    "required": ["approach", "context"],
}


class VerifyWorkTool:
    """Run a verification command and return pass/fail + output.

    The agent chooses the appropriate repository command for the project.
    """

    name = "verify_work"
    description = _VERIFY_BASE_DESCRIPTION
    # `verify_work` runs an arbitrary shell command — same blast radius as
    # `shell`. Keep the approval level consistent so an agent can't bypass the
    # shell approval gate by passing its command through here instead.
    approval: ApprovalDecision = "prompt"
    # NOTE: declared read_only because the typical use-case is "run the test
    # suite," which is read-only. But the underlying mechanism is subprocess
    # shell, so the structural denylist (check_dangerous_command) is applied
    # below as a backstop regardless of the declared scope.
    effect_scope = "read_only"
    prediction_expected_status = "ok_or_error"

    def __init__(
        self,
        cwd: Path,
        *,
        timeout: float = 120.0,
        default_command: str | None = None,
    ) -> None:
        self._cwd = cwd
        self._timeout = timeout
        self._default_command = (default_command or "").strip()
        self.parameters_schema = _verify_schema(has_default_command=bool(self._default_command))
        self.description = _VERIFY_BASE_DESCRIPTION + (
            _VERIFY_DEFAULT_DESCRIPTION if self._default_command else ""
        )

    @property
    def has_default_command(self) -> bool:
        return bool(self._default_command)

    async def __call__(self, call: ToolCall) -> ToolResult:
        from harness.core.shell_safety import check_dangerous_command

        command = str(call.arguments.get("command") or "").strip()
        used_default_command = False
        if not command and self._default_command:
            command = self._default_command
            used_default_command = True
        if not command or not command.strip():
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="command argument is required",
                is_error=True,
            )
        if not used_default_command and _verification_command_is_noop(command):
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=(
                    "refused invalid verification command: the command does not "
                    "run a meaningful check. verify_work evidence must come from "
                    "a repository command that can fail when the work is wrong."
                ),
                is_error=True,
                metadata={
                    "invalid_verification_command": True,
                    "reason": "noop_verification_command",
                    "used_default_command": used_default_command,
                },
            )
        if _failure_branch_masks_exit_status(command):
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=(
                    "refused invalid verification command: the failure branch can "
                    "exit successfully. verify_work checks exit status, so failed "
                    "assertions must return a non-zero exit code."
                ),
                is_error=True,
                metadata={
                    "invalid_verification_command": True,
                    "reason": "masked_failure_exit_status",
                    "used_default_command": used_default_command,
                },
            )
        if _command_exits_before_trailing_command(command):
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=(
                    "refused invalid verification command: the command exits before "
                    "a later check can run. verify_work evidence must come from "
                    "commands that actually execute."
                ),
                is_error=True,
                metadata={
                    "invalid_verification_command": True,
                    "reason": "unreachable_verification_command",
                    "used_default_command": used_default_command,
                },
            )

        denial = check_dangerous_command(command)
        if denial is not None:
            tier, reason = denial
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=(
                    f"refused [{tier} deny]: {reason}. verify_work only accepts "
                    "verification commands, not arbitrary shell."
                ),
                is_error=True,
                metadata={
                    "refused_reason": reason,
                    "deny_tier": tier,
                    "denied": True,
                    "used_default_command": used_default_command,
                },
            )

        try:
            shell_command = f"set -e -o pipefail; {command}"
            before_status = await _git_status_entries(self._cwd)
            proc = await asyncio.create_subprocess_shell(
                shell_command,
                cwd=self._cwd,
                env=clean_command_env(self._cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                executable="/bin/bash",
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
            output_raw = stdout.decode(errors="replace")
            output = output_raw.strip()
            after_status = await _git_status_entries(self._cwd)
            workspace_changed = (
                before_status is not None
                and after_status is not None
                and after_status != before_status
            )
            passed = proc.returncode == 0
            output_failure = False
            if passed and _output_reports_failure(output):
                output_failure = True
                passed = False
            if passed and workspace_changed:
                passed = False
            verdict = (
                "PASSED"
                if passed
                else "FAILED (verification changed workspace)"
                if workspace_changed
                else "FAILED (output reports failure)"
                if output_failure
                else f"FAILED (exit {proc.returncode})"
            )
            content = f"{verdict}\n\n{output}" if output else verdict
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=content,
                is_error=not passed,
                metadata={
                    "command": command,
                    "exit_code": proc.returncode,
                    "stdout": output_raw,
                    "output_reports_failure": output_failure,
                    "workspace_changed": workspace_changed,
                    "used_default_command": used_default_command,
                    "clean_env": True,
                    "pipefail": True,
                    "errexit": True,
                },
            )
        except TimeoutError:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"command timed out after {self._timeout}s",
                is_error=True,
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"error running command: {exc}",
                is_error=True,
            )


class RequestCritiqueTool:
    """Ask a second LLM reviewer to challenge your proposed approach.

    Describe what you plan to change and why; the critic returns a pointed
    challenge or confirms you're on the right track. Call this when you're
    uncertain about your diagnosis before making changes.
    """

    name = "request_critique"
    description = (
        "Get a second opinion on your proposed approach before making changes. "
        "Describe what you plan to do and why — the critic will identify any flaws "
        "in your reasoning and ask you a specific question if something looks wrong. "
        "Call this when you're uncertain about your diagnosis."
    )
    approval: ApprovalDecision = "auto"
    effect_scope = "read_only"

    def __init__(
        self,
        adapter: Adapter,
        model: str,
        *,
        max_tokens: int = 400,
        temperature: float = 0.3,
        search_fn: SearchFn | None = None,
        max_searches: int = 2,
    ) -> None:
        self._adapter = adapter
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._search_fn = search_fn
        self._max_searches = max_searches
        self.parameters_schema = _CRITIQUE_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        approach = str(call.arguments.get("approach", "")).strip()
        context = str(call.arguments.get("context", "")).strip()

        if not approach:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="approach argument is required",
                is_error=True,
            )

        # Pre-fetch web context without LLM tool-calling (works with local models).
        research_lines: list[str] = []
        if self._search_fn is not None:
            queries = [approach[:120], context[:120] if context else ""]
            for q in queries[: self._max_searches]:
                if not q.strip():
                    continue
                try:
                    result = await self._search_fn(q.strip())  # type: ignore[misc]
                    if result:
                        research_lines.append(f"Search: {q.strip()[:80]}\n{result[:800]}")
                except Exception:
                    pass

        research_block = (
            "\n## Web research\n\n" + "\n\n".join(research_lines) + "\n\n" if research_lines else ""
        )

        messages: list[Message] = [
            Message(role="system", content=_CRITIQUE_SYSTEM),
            Message(
                role="user",
                content=research_block
                + _CRITIQUE_USER.format(
                    approach=approach[:2000],
                    context=context[:2000] if context else "(no context provided)",
                ),
            ),
        ]

        try:
            text_parts: list[str] = []
            async for event in self._adapter.stream(
                model=self._model,
                messages=messages,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            ):
                if isinstance(event, TextDelta):
                    text_parts.append(event.text)
                elif isinstance(event, Done):
                    if event.final_message and event.final_message.content:
                        return ToolResult(
                            tool_call_id=call.id,
                            name=self.name,
                            content=event.final_message.content.strip()
                            or "(critic produced no output)",
                        )
                    break
            critique = "".join(text_parts).strip()
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=critique or "(critic produced no output)",
            )
        except Exception as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"critic unavailable: {exc}",
                is_error=True,
            )


__all__ = ["RequestCritiqueTool", "VerifyWorkTool"]
