"""Shell execution tool for Harness agents.

Runs commands via the user's `/bin/sh` (or default shell) inside the session's
cwd. Default approval is `prompt` because shell access is the highest-risk
capability we expose.

Output is captured in full and surfaced to the agent as the ToolResult
content. Each stream (stdout + stderr) is truncated to `max_output_bytes` to
keep transcripts manageable.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from pathlib import Path
from typing import Any

from harness.core import ApprovalDecision, ToolCall, ToolResult

__version__ = "0.0.0"


_SHELL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "command": {
            "type": "string",
            "description": "Shell command to execute. Runs through /bin/sh (-c).",
        },
        "timeout": {
            "type": "integer",
            "description": "Seconds before killing the command. Capped by the tool's own max.",
        },
    },
    "required": ["command"],
}


def _truncate(data: bytes, limit: int) -> tuple[str, bool]:
    if len(data) <= limit:
        return data.decode("utf-8", errors="replace"), False
    head = data[:limit].decode("utf-8", errors="replace")
    return head, True


class ShellTool:
    """Execute a shell command. Returns combined exit code + stdout + stderr.

    Args:
        cwd: Working directory the command runs in. Required.
        default_timeout: Timeout (seconds) when the call doesn't specify one.
        max_timeout: Hard cap on the per-call timeout.
        max_output_bytes: Per-stream truncation cap.
    """

    name = "shell"
    description = (
        "Execute a shell command in the session's working directory. Returns "
        "the exit code, stdout, and stderr. Has a strict timeout — long-running "
        "commands should be avoided."
    )
    approval: ApprovalDecision = "prompt"
    effect_scope = "workspace_durable"
    # Mutating side effects — restricted to the `act` phase.
    phases: tuple[str, ...] = ("act",)

    def __init__(
        self,
        *,
        cwd: Path | str,
        default_timeout: float = 30.0,
        max_timeout: float = 300.0,
        max_output_bytes: int = 64 * 1024,
    ) -> None:
        self.cwd = Path(cwd).resolve()
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout
        self.max_output_bytes = max_output_bytes
        self.parameters_schema: dict[str, Any] = _SHELL_SCHEMA

    async def __call__(self, call: ToolCall) -> ToolResult:
        command = call.arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="missing or empty `command` argument",
                is_error=True,
            )

        timeout_arg = call.arguments.get("timeout", self.default_timeout)
        try:
            timeout = float(timeout_arg)
        except (TypeError, ValueError):
            timeout = self.default_timeout
        timeout = max(0.1, min(timeout, self.max_timeout))

        proc: asyncio.subprocess.Process | None = None
        started = time.perf_counter()
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(self.cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except TimeoutError:
                with contextlib.suppress(ProcessLookupError):
                    proc.kill()
                # Drain whatever buffered I/O is available.
                with contextlib.suppress(Exception):
                    await proc.communicate()
                return ToolResult(
                    tool_call_id=call.id,
                    name=self.name,
                    content=f"command timed out after {timeout}s and was killed",
                    is_error=True,
                    metadata={
                        "duration_ms": int((time.perf_counter() - started) * 1000),
                        "timed_out": True,
                        "timeout_s": timeout,
                    },
                )
        except FileNotFoundError as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"shell not available: {exc}",
                is_error=True,
            )
        except OSError as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"could not start command: {exc}",
                is_error=True,
            )

        duration_ms = int((time.perf_counter() - started) * 1000)
        stdout_raw = stdout_b or b""
        stderr_raw = stderr_b or b""
        stdout_text, stdout_trunc = _truncate(stdout_raw, self.max_output_bytes)
        stderr_text, stderr_trunc = _truncate(stderr_raw, self.max_output_bytes)

        parts: list[str] = [f"exit_code: {proc.returncode}"]
        if stdout_text:
            suffix = (
                f"\n…[stdout truncated at {self.max_output_bytes} bytes]" if stdout_trunc else ""
            )
            parts.append(f"stdout:\n{stdout_text}{suffix}")
        if stderr_text:
            suffix = (
                f"\n…[stderr truncated at {self.max_output_bytes} bytes]" if stderr_trunc else ""
            )
            parts.append(f"stderr:\n{stderr_text}{suffix}")
        if not stdout_text and not stderr_text:
            parts.append("(no output)")

        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content="\n\n".join(parts),
            is_error=proc.returncode != 0,
            metadata={
                "exit_code": proc.returncode,
                "stdout_bytes": len(stdout_raw),
                "stderr_bytes": len(stderr_raw),
                "stdout_truncated": stdout_trunc,
                "stderr_truncated": stderr_trunc,
                "duration_ms": duration_ms,
                "timed_out": False,
            },
        )


__all__ = ["ShellTool", "__version__"]
