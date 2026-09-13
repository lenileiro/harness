"""Verification evidence produced by the selected execution backend."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from harness.core.schemas import ApprovalDecision, ToolCall, ToolResult
from harness.core.tools_verification import (
    _command_exits_before_trailing_command,
    _failure_branch_masks_exit_status,
    _output_reports_failure,
    _verification_command_is_noop,
    _verify_schema,
)

if TYPE_CHECKING:
    from harness.tools.execution.toolset import ExecutionToolset


class BackendVerifyTool:
    name = "verify_work"
    description = "Run a meaningful verification command on the selected execution backend. Returns pass/fail evidence. Uses Bash errexit/pipefail; failures never fall back to the host."
    approval: ApprovalDecision = "prompt"
    effect_scope = "workspace_durable"
    phases = ("*",)
    prediction_expected_status = "ok_or_error"

    def __init__(self, owner: ExecutionToolset, *, default_command: str | None) -> None:
        self.owner = owner
        self.default_command = (default_command or "").strip()
        self.parameters_schema = _verify_schema(has_default_command=self.has_default_command)

    @property
    def has_default_command(self) -> bool:
        return bool(self.default_command)

    async def __call__(self, call: ToolCall) -> ToolResult:
        if self.owner._closing:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content="execution context is closing",
                is_error=True,
            )
        explicit = str(call.arguments.get("command") or "").strip()
        command = explicit or self.default_command
        used_default = not explicit and bool(self.default_command)
        reason = None
        if not command:
            reason = "command argument is required"
        elif not used_default and _verification_command_is_noop(command):
            reason = "noop_verification_command"
        elif _failure_branch_masks_exit_status(command):
            reason = "masked_failure_exit_status"
        elif _command_exits_before_trailing_command(command):
            reason = "unreachable_verification_command"
        if reason:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"refused invalid verification command: {reason}",
                is_error=True,
                metadata={
                    "invalid_verification_command": True,
                    "reason": reason,
                    "used_default_command": used_default,
                },
            )
        try:
            before = await self.owner.backend.files("git_status", {})
            process = await self.owner.start(
                command=f"set -e -o pipefail; {command}", shell="/bin/bash"
            )
            try:
                await process.done.wait()
                result = await process.poll()
            finally:
                await asyncio.shield(process.terminate())
                self.owner.processes.pop(process.id, None)
            after = await self.owner.backend.files("git_status", {})
            changed = bool(
                before["metadata"]["available"]
                and after["metadata"]["available"]
                and before["content"] != after["content"]
            )
            output_failure = _output_reports_failure(result["output"])
            passed = (
                result["exit_code"] == 0
                and result["status"] == "exited"
                and not (changed or output_failure or result["truncated"])
            )
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=("PASSED" if passed else "FAILED") + "\n\n" + result["output"],
                is_error=not passed,
                metadata={
                    **result,
                    "command": command,
                    "stdout": result["output"],
                    "output_reports_failure": output_failure,
                    "workspace_changed": changed,
                    "used_default_command": used_default,
                    "clean_env": False,
                    "pipefail": True,
                    "errexit": True,
                },
            )
        except (ValueError, OSError, RuntimeError, TimeoutError) as exc:
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=f"error running backend verification: {exc}",
                is_error=True,
            )


__all__ = ["BackendVerifyTool"]
