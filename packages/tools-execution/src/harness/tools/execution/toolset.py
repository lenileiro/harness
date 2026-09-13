"""Backend-consistent tool schemas and one owned execution context."""

from __future__ import annotations

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from harness.core.schemas import ApprovalDecision, ToolCall, ToolResult
from harness.core.shell_safety import check_dangerous_command
from harness.core.tools import Tool
from harness.tools.execution.backend import ExecutionBackend
from harness.tools.execution.cloud_process import CloudManagedProcess
from harness.tools.execution.config import ExecutionConfig
from harness.tools.execution.process import ManagedProcess
from harness.tools.execution.transfers import WorkspaceTransfers
from harness.tools.execution.verification import BackendVerifyTool


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReadArguments(Arguments):
    path: str = Field(min_length=1)


class WriteArguments(ReadArguments):
    content: str


class EditArguments(ReadArguments):
    old: str = Field(min_length=1)
    new: str


class ListArguments(Arguments):
    path: str = "."


class GlobArguments(Arguments):
    pattern: str = Field(min_length=1)


class ShellArguments(Arguments):
    command: str = Field(min_length=1, max_length=65536)
    cwd: str = "."
    timeout: float | None = Field(default=None, gt=0)


class ProcessArguments(Arguments):
    action: Literal["start", "poll", "input", "terminate"]
    process_id: str | None = None
    command: str | None = Field(default=None, min_length=1, max_length=65536)
    cwd: str = "."
    timeout: float | None = Field(default=None, gt=0)
    pty: bool = False
    input: str = Field(default="", max_length=65536)
    wait_seconds: float = Field(default=0, ge=0, le=30)


_FILE_ARGUMENTS: dict[str, type[Arguments]] = {
    "read_file": ReadArguments,
    "write_file": WriteArguments,
    "edit_file": EditArguments,
    "list_dir": ListArguments,
    "glob": GlobArguments,
}


class ExecutionTool:
    def __init__(self, name: str, owner: ExecutionToolset) -> None:
        self.name = name
        self.owner = owner
        self.description = (
            f"{name} on the explicitly configured {owner.config.backend} backend. "
            "File paths and command cwd must stay within the backend workspace. "
            "Commands execute with that backend's OS permissions. "
        )
        self.approval: ApprovalDecision = (
            "auto" if name in {"read_file", "list_dir", "glob"} else "prompt"
        )
        self.effect_scope = "read_only" if self.approval == "auto" else "workspace_durable"
        self.phases = ("*",) if self.approval == "auto" else ("act",)
        model = _FILE_ARGUMENTS.get(name)
        if name == "shell":
            model = ShellArguments
            self.description += (
                "Returns exit_code and bounded combined output; timeout is capped by configuration."
            )
        if name == "process":
            model = ProcessArguments
            self.description += (
                "Start returns a process_id retained within this execution context. "
                "Poll returns cumulative bounded output, input writes literal text, "
                "terminate stops the process group. PTY is available for local commands. "
                "Handles do not survive closing the execution context or host restart."
            )
        assert model is not None
        self.arguments_type = model
        self.parameters_schema = model.model_json_schema()

    async def __call__(self, call: ToolCall) -> ToolResult:
        try:
            if not self.owner.backend.active or self.owner._closing:
                raise RuntimeError("execution context is closed")
            args = self.arguments_type.model_validate(call.arguments)
            if isinstance(args, ShellArguments):
                result = await self.owner.shell(args)
                return ToolResult(
                    tool_call_id=call.id,
                    name=self.name,
                    content=f"exit_code: {result['exit_code']}\n{result['output']}",
                    is_error=result["exit_code"] != 0 or result["status"] != "exited",
                    metadata=result,
                )
            if isinstance(args, ProcessArguments):
                result = await self.owner.process(args)
                return ToolResult(
                    tool_call_id=call.id,
                    name=self.name,
                    content=json.dumps(result, ensure_ascii=False),
                    is_error=result["status"] in {"transport_error", "timed_out", "disconnected"}
                    or (
                        not result["running"]
                        and result["exit_code"] not in (0, None)
                        and result["status"] != "terminated"
                    ),
                    metadata=result,
                )
            result = await self.owner.backend.files(self.name, args.model_dump())
            return ToolResult(
                tool_call_id=call.id,
                name=self.name,
                content=result["content"],
                metadata=result.get("metadata"),
            )
        except (ValueError, OSError, RuntimeError, TimeoutError) as exc:
            return ToolResult(tool_call_id=call.id, name=self.name, content=str(exc), is_error=True)


class ExecutionToolset:
    """Owned tools used as ``async with ExecutionToolset(config, cwd=...)``.

    ``tools`` replace built-in filesystem/shell registrations; adding them beside
    host tools with the same names is an integration error. Keep this context
    open across model calls/turns that need shared process handles.
    """

    def __init__(
        self, config: ExecutionConfig, *, cwd: Path, verify_command: str | None = None
    ) -> None:
        self.config = config
        self.backend = ExecutionBackend(config, cwd)
        self.processes: dict[str, ManagedProcess] = {}
        self.tools: tuple[Tool, ...] = (
            *tuple(
                ExecutionTool(name, self)
                for name in (
                    "read_file",
                    "write_file",
                    "edit_file",
                    "list_dir",
                    "glob",
                    "shell",
                    "process",
                )
            ),
            BackendVerifyTool(self, default_command=verify_command),
        )
        self._entered = False
        self._closing = False
        self._closed = False
        self._close_lock = asyncio.Lock()
        self.transfers = WorkspaceTransfers(self.backend)
        self.context_id = uuid.uuid4().hex
        self._start_lock = asyncio.Lock()

    async def __aenter__(self) -> ExecutionToolset:
        if self._entered:
            raise RuntimeError("ExecutionToolset contexts cannot be reopened")
        self._entered = True
        await self.transfers.preflight()
        await self.backend.open()
        try:
            await self.transfers.import_files()
        except BaseException:
            await asyncio.shield(self.backend.close())
            raise
        return self

    async def __aexit__(self, *exc: object) -> None:
        await asyncio.shield(self.close())

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            async with self._start_lock:
                self._closing = True
            failures: list[Exception] = []
            try:
                results = await asyncio.gather(
                    *(process.terminate() for process in self.processes.values()),
                    return_exceptions=True,
                )
                failures.extend(result for result in results if isinstance(result, Exception))
                if self.backend.active:
                    await self.transfers.export_files(context_id=self.context_id)
            except Exception as exc:
                failures.append(exc)
            finally:
                try:
                    await self.backend.close()
                finally:
                    self._closed = True
            if failures:
                raise ExceptionGroup("execution cleanup or artifact export failed", failures)

    @property
    def exported_files(self) -> list[str]:
        return list(self.transfers.exported_files)

    async def start(
        self,
        *,
        command: str,
        cwd: str = ".",
        timeout_seconds: float | None = None,
        pty: bool = False,
        shell: str | None = None,
    ) -> ManagedProcess:
        denial = check_dangerous_command(command)
        if denial is not None:
            raise ValueError(f"command refused: {denial[1]}")
        timeout = min(timeout_seconds or self.config.timeout_seconds, self.config.timeout_seconds)
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        async with self._start_lock:
            if self._closing or not self.backend.active:
                raise RuntimeError("execution context is closed")
            if (
                len([process for process in self.processes.values() if not process.done.is_set()])
                >= self.config.max_processes
            ):
                raise ValueError("maximum active processes reached; terminate an existing process")
            if len(self.processes) >= self.config.max_processes * 4:
                oldest = next(
                    (key for key, process in self.processes.items() if process.done.is_set()), None
                )
                if oldest is not None:
                    del self.processes[oldest]
            process = (
                CloudManagedProcess(self.backend)
                if self.backend.cloud is not None
                else ManagedProcess(self.backend)
            )
            self.processes[process.id] = process
            try:
                await process.start(
                    command=command, cwd=cwd, timeout_seconds=timeout, pty=pty, shell=shell
                )
            except BaseException:
                self.processes.pop(process.id, None)
                raise
            return process

    async def shell(self, args: ShellArguments) -> dict[str, Any]:
        process = await self.start(command=args.command, cwd=args.cwd, timeout_seconds=args.timeout)
        try:
            await process.done.wait()
            return await process.poll()
        finally:
            await asyncio.shield(process.terminate())
            self.processes.pop(process.id, None)

    async def process(self, args: ProcessArguments) -> dict[str, Any]:
        if args.action == "start":
            if args.command is None:
                raise ValueError("start requires command")
            process = await self.start(
                command=args.command, cwd=args.cwd, timeout_seconds=args.timeout, pty=args.pty
            )
        else:
            if not args.process_id or args.process_id not in self.processes:
                raise ValueError("unknown process_id in this execution context")
            process = self.processes[args.process_id]
            if args.action == "input":
                await process.input(args.input)
            elif args.action == "terminate":
                await process.terminate()
        return await process.poll(args.wait_seconds)


__all__ = ["ExecutionToolset"]
