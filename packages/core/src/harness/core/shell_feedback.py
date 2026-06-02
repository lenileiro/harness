from __future__ import annotations

import os
import re
import shlex

_AVAILABILITY_CHECKS = {"which", "type"}
_SEARCH_COMMANDS = {"ack", "ag", "grep", "rg"}


def _tokenize_shell_command(command: str) -> list[str]:
    try:
        return shlex.split(command)
    except ValueError:
        return []


def _basename(token: str) -> str:
    return os.path.basename(token.rstrip()) or token


def _availability_target(tokens: list[str]) -> str:
    if not tokens:
        return ""
    command = _basename(tokens[0])
    if command in _AVAILABILITY_CHECKS:
        for token in tokens[1:]:
            if not token.startswith("-"):
                return token
        return ""
    if command == "command" and len(tokens) >= 3 and tokens[1] == "-v":
        return tokens[2]
    return ""


def shell_empty_failure_hint(
    command: str,
    *,
    exit_code: int | None,
    stdout: str,
    stderr: str,
) -> str:
    if exit_code == 0 or stdout.strip() or stderr.strip():
        return ""
    tokens = _tokenize_shell_command(command)
    target = _availability_target(tokens)
    if target:
        return (
            f"[empty failure] `{target}` was not found on PATH. Install the missing "
            "tool, use an available equivalent, or inspect the project setup before "
            "retrying this command."
        )
    if any(_basename(token) in _SEARCH_COMMANDS for token in tokens):
        return (
            "[empty failure] This search returned no matches. Broaden the query, "
            "search related terms, or inspect likely files directly before retrying."
        )
    return (
        "[empty failure] The command exited non-zero with no stdout or stderr. "
        "Treat this as failed evidence; choose a different check or inspect the "
        "repository setup to find the missing precondition."
    )


def shell_failure_hint(
    command: str,
    *,
    exit_code: int | None,
    stdout: str,
    stderr: str,
) -> str:
    output = f"{stdout}\n{stderr}"
    lowered = output.lower()
    if "unable to read tree" in lowered or "reference is not a tree" in lowered:
        return (
            "[setup hint] This looks like a shallow or incomplete git checkout. "
            "Fetch the required commit explicitly, for example "
            "`git fetch --depth 1 origin <commit>` followed by `git checkout <commit>`, "
            "or reclone with enough history before editing."
        )
    if exit_code == 127:
        match = re.search(
            r"(?:^|\n)(?:[^:\n]+:\s+line\s+\d+:\s+)?([A-Za-z0-9_.+-]+): command not found",
            output,
        )
        if match:
            tool = match.group(1)
            return (
                f"[setup hint] `{tool}` is not installed on PATH. Inspect project setup "
                "files such as environment/Dockerfile, package manifests, or toolchain "
                "files; if those files declare Docker or another project runtime, run "
                "`command -v docker` / `docker info` or the equivalent availability "
                "check before concluding the toolchain is unavailable, then use an "
                "available install, container, or equivalent command before retrying. "
                "For final verification, verify_work can run the same read-only "
                "containerized command. If you discover the tool inside a container "
                "but it is not on that container's PATH, use the discovered executable "
                "path in the container command instead of retrying the host command."
            )
    if (
        "editable mode currently requires" in lowered
        or 'file "setup.py" or "setup.cfg" not found' in lowered
        or "successfully installed unknown" in lowered
        or "no module named build" in lowered
    ):
        return (
            "[setup hint] The package/build setup path did not produce a usable "
            "project install. Inspect the project metadata and declared runtime files "
            "such as environment/Dockerfile or package manifests, then use the "
            "declared environment, a supported non-editable install, or a source-path "
            "test command. Confirm the setup by importing the project package or "
            "running a focused test before continuing implementation work."
        )
    if exit_code and exit_code != 0 and _looks_like_terse_test_failure(output):
        return (
            "[diagnostic hint] The command failed but only reported a terse test "
            "case label. Before editing more implementation code, rerun the failing "
            "check with execution tracing or temporary diagnostics so the exact "
            "failing command, inputs, and variable values are visible."
        )
    return shell_empty_failure_hint(command, exit_code=exit_code, stdout=stdout, stderr=stderr)


def _looks_like_terse_test_failure(output: str) -> bool:
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if not lines or len(lines) > 4:
        return False
    lowered = [line.lower() for line in lines]
    return any(line == "not ok" or line.startswith("not ok:") for line in lowered)


__all__ = ["shell_empty_failure_hint", "shell_failure_hint"]
