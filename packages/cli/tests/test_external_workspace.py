from __future__ import annotations

# pyright: reportOptionalSubscript=false, reportOptionalMemberAccess=false, reportOperatorIssue=false, reportAttributeAccessIssue=false, reportIndexIssue=false, reportCallIssue=false
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import typer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from harness.cli.external_workspace import (
    _GIT_WORKSPACE_FINGERPRINT_COMMAND,
    ExternalWorkspaceCoverageVerifier,
    ExternalWorkspacePolicy,
    ExternalWorkspaceVerifier,
    PolicyFetchUrlTool,
    PolicyWebSearchTool,
    RemoteApplyPatchTool,
    RemoteEditFileTool,
    RemoteListDirTool,
    RemoteReadFileRangeTool,
    RemoteReadFileTool,
    RemoteShellTool,
    RemoteVerifyWorkTool,
    RemoteWorkspaceState,
    RemoteWriteFileTool,
    _external_workspace_model_candidates,
    _fingerprint_has_untracked_test_path,
    _merge_workspace_status,
    _normalize_unified_diff_hunk_headers,
    _tool_result_counts_as_source_change,
    _verification_command_covers_test_changes,
    _workspace_deleted_source_paths,
    _workspace_source_change_status,
    _workspace_test_change_paths,
    build_remote_tool_registry,
    external_workspace_repair_attempts,
    external_workspace_total_attempts,
    run_harness_on_external_environment,
)
from harness.core import (
    Capabilities,
    Done,
    Event,
    Message,
    ModelSelectedEvent,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolResult,
    Verification,
)
from harness.core.activity import ActivityEvent
from harness.core.events import ErrorEvent


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    return_code: int


@pytest.fixture(autouse=True)
def _disable_live_coverage_review_for_unit_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_EXTERNAL_WORKSPACE_COVERAGE_REVIEW", "0")


class LocalEnvironment:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.calls: list[dict[str, object]] = []

    async def exec(self, command: str, *, cwd: str | None = None, timeout_sec: int | None = None):
        self.calls.append({"command": command, "cwd": cwd, "timeout_sec": timeout_sec})
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd or str(self.root),
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_sec or 30)
        return ExecResult(
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
            return_code=proc.returncode or 0,
        )


class StaticStatusEnvironment:
    def __init__(
        self,
        root: Path,
        statuses: list[str],
        baseline_status: str = "",
        tracked_paths: set[str] | None = None,
    ) -> None:
        self.root = root
        self.statuses = statuses
        self.baseline_status = baseline_status
        self.tracked_paths = tracked_paths or set()
        self._baseline_returned = False
        self.calls: list[dict[str, object]] = []

    async def exec(self, command: str, *, cwd: str | None = None, timeout_sec: int | None = None):
        self.calls.append({"command": command, "cwd": cwd, "timeout_sec": timeout_sec})
        if command == "pwd":
            return ExecResult(stdout=str(self.root) + "\n", stderr="", return_code=0)
        if command in {
            "git status --porcelain",
            "git status --porcelain --untracked-files=all",
        }:
            if not self._baseline_returned:
                self._baseline_returned = True
                return ExecResult(stdout=self.baseline_status, stderr="", return_code=0)
            status = self.statuses.pop(0) if self.statuses else ""
            return ExecResult(stdout=status, stderr="", return_code=0)
        if command == "git ls-files":
            return ExecResult(
                stdout="\n".join(sorted(self.tracked_paths)), stderr="", return_code=0
            )
        return ExecResult(stdout="", stderr="", return_code=0)


class StderrDroppingEnvironment:
    async def exec(self, command: str, *, cwd: str | None = None, timeout_sec: int | None = None):
        return ExecResult(stdout="", stderr="", return_code=4)


class FixedExitEnvironment:
    def __init__(self, return_code: int) -> None:
        self.return_code = return_code

    async def exec(self, command: str, *, cwd: str | None = None, timeout_sec: int | None = None):
        return ExecResult(stdout="", stderr="", return_code=self.return_code)


def test_external_workspace_gates_do_not_reintroduce_language_specific_policy() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "src" / "harness" / "cli" / "external_workspace.py"
    ).read_text(encoding="utf-8")

    clone_gate = source[
        source.index("def _shell_command_is_allowed_git_clone") : source.index(
            "def _git_fetch_segment_uses_named_remote"
        )
    ]
    setup_detector = source[
        source.index("def _shell_command_requests_setup") : source.index("def _porcelain_paths")
    ]
    coverage_gate = source[
        source.index("def _is_broad_test_command") : source.index(
            "def _diff_references_untracked_test_paths"
        )
    ]
    semantic_coverage_gate = source[
        source.index("def _release_value_format_reason") : source.index(
            "class ExternalWorkspaceCoverageVerifier"
        )
    ]
    generated_artifact_filter = (
        source[
            source.index("_GIT_WORKSPACE_FINGERPRINT_COMMAND") : source.index(
                "_ROOT_PROBE_COMMANDS"
            )
        ]
        + source[
            source.index("def _path_part_is_generated_cache") : source.index(
                "def _workspace_test_change_paths"
            )
        ]
    )
    language_specific_gate_terms = frozenset(
        {
            "python",
            "python3",
            "pytest",
            "pip",
            "pip3",
            "node",
            "npm",
            "npx",
            "pnpm",
            "yarn",
            "cargo",
            "go",
            "rust",
            "javascript",
            "js",
            "typescript",
            "ts",
        }
    )
    for section in (
        clone_gate,
        setup_detector,
        coverage_gate,
        semantic_coverage_gate,
        generated_artifact_filter,
    ):
        words = set(re.findall(r"[a-z0-9_+.-]+", section.lower()))
        assert words.isdisjoint(language_specific_gate_terms)


class FlakyVerifyEnvironment:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.verify_runs = 0

    async def exec(self, command: str, *, cwd: str | None = None, timeout_sec: int | None = None):
        self.calls.append({"command": command, "cwd": cwd, "timeout_sec": timeout_sec})
        if command == _GIT_WORKSPACE_FINGERPRINT_COMMAND:
            return ExecResult(
                stdout="untracked tests/generated_regression.sh\nmode 755 tests/generated_regression.sh\nabc123\n",
                stderr="",
                return_code=0,
            )
        if command.startswith("bash -lc ") and "make test" in command:
            self.verify_runs += 1
            if self.verify_runs == 1:
                return ExecResult(stdout="ok\n", stderr="", return_code=0)
            return ExecResult(stdout="not ok\n", stderr="make: *** [test] Error 1\n", return_code=2)
        return ExecResult(stdout="", stderr="", return_code=0)


class FakeWebTool:
    def __init__(
        self,
        *,
        name: str,
        content: str,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.name = name
        self.content = content
        self.metadata = metadata or {}
        self.parameters_schema = {"type": "object", "properties": {}}
        self.calls: list[ToolCall] = []

    async def __call__(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        return ToolResult(
            tool_call_id=call.id,
            name=self.name,
            content=self.content,
            metadata=self.metadata,
        )


class CoverageReviewAdapter:
    def __init__(self, payload: dict[str, object] | str) -> None:
        self.payload = payload
        self.calls: list[dict[str, object]] = []

    async def stream(self, *, model: str, messages: list[Message], **_kwargs: object):
        self.calls.append({"model": model, "messages": messages})
        body = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        yield TextDelta(text=body)
        yield Done(final_message=Message(role="assistant", content=body))


def test_workspace_test_change_paths_detects_common_test_layouts() -> None:
    status = (
        " M src/lib.rs\n"
        " M t/unit/transport/virtual/test_base.py\n"
        "?? tests/test_feature.py\n"
        " M packages/foo/src/foo.test.ts\n"
        " M core/testdata/func.ank\n"
        " M docs/notes.md\n"
    )

    assert _workspace_test_change_paths(status) == [
        "t/unit/transport/virtual/test_base.py",
        "tests/test_feature.py",
        "packages/foo/src/foo.test.ts",
        "core/testdata/func.ank",
    ]


def test_verification_command_must_cover_changed_tests() -> None:
    changed = ["t/unit/transport/virtual/test_sac_priority.case"]

    assert _verification_command_covers_test_changes(
        "project-test t/unit/transport/virtual/test_sac_priority.case",
        changed,
    )
    assert _verification_command_covers_test_changes(
        "project-test t/unit/transport/virtual",
        changed,
    )
    assert not _verification_command_covers_test_changes("project-test", changed)
    assert _verification_command_covers_test_changes(
        "project-check tests/test_feature.case",
        ["tests/test_feature.case"],
    )
    assert _verification_command_covers_test_changes(
        "cd repo && project-test tests/test_helper.py",
        ["repo/tests/test_helper.py"],
    )
    assert _verification_command_covers_test_changes(
        "cd repo && project-test ./...",
        ["repo/ansi/truncate_test.go"],
    )
    assert not _verification_command_covers_test_changes(
        "cd repo && project-test ./...",
        ["other/tests/test_helper.py"],
    )
    assert _verification_command_covers_test_changes(
        "cd repo && PROJECT_ENV=. project-test -q tests/test_helper.py",
        ["repo/tests/test_helper.py"],
    )
    assert _verification_command_covers_test_changes(
        "cd .termenv-src && docker run --rm -t -v $(pwd):/app "
        "public.ecr.aws/x8v8d7g8/mars-base:latest bash -lc "
        "'cd /app && project-test ./... -count=1'",
        [".termenv-src/ansi/truncate_test.go"],
    )
    assert _verification_command_covers_test_changes(
        "cd .termenv-src && docker run --rm -t -v $(pwd):/app "
        "public.ecr.aws/x8v8d7g8/mars-base:latest bash -lc "
        "'cd /app && project-test ./ansi -count=1'",
        [".termenv-src/ansi/truncate_test.go"],
    )
    assert not _verification_command_covers_test_changes(
        "cd .termenv-src && docker run --rm -t -v $(pwd):/app "
        "public.ecr.aws/x8v8d7g8/mars-base:latest bash -lc "
        "'cd /app && project-test ./... -run Other'",
        [".termenv-src/ansi/truncate_test.go"],
    )
    assert not _verification_command_covers_test_changes("CI=true project-test", changed)
    assert not _verification_command_covers_test_changes(
        "make test",
        ["tests/regression.sh"],
        untracked_test_paths=["tests/regression.sh"],
    )
    assert _verification_command_covers_test_changes(
        "make test",
        ["tests/run.sh", "tests/regression.sh"],
        untracked_test_paths=["tests/regression.sh"],
    )
    assert _verification_command_covers_test_changes(
        "make test",
        ["tests/regression.sh"],
        untracked_test_paths=["tests/regression.sh"],
        runner_wires_untracked_tests=True,
    )
    assert _verification_command_covers_test_changes(
        "bash tests/regression.sh",
        ["tests/regression.sh"],
    )
    assert _verification_command_covers_test_changes(
        "./tests/regression.sh",
        ["tests/regression.sh"],
    )
    assert _verification_command_covers_test_changes(
        "sh quick_check.sh",
        ["tests/test_slugify.py"],
        runner_wires_changed_tests=True,
    )
    assert _verification_command_covers_test_changes(
        "project-test run tests.unit.cli.test_incremental_cache_cli",
        ["repo/bandit/tests/unit/cli/test_incremental_cache_cli.py"],
    )
    assert _verification_command_covers_test_changes(
        "cd repo/bandit && docker run --network none --rm -v $(pwd):/app/bandit "
        "-w /app/bandit public.example/task:latest "
        "project-test run tests.unit.cli.test_incremental_cache_cli",
        ["repo/bandit/tests/unit/cli/test_incremental_cache_cli.py"],
    )
    assert _verification_command_covers_test_changes(
        'cd repo && docker run --rm --network none -v "$PWD:/work" -w /work '
        'public.example/task:latest bash -lc "project-test -q '
        'core/engine/tests/evaluation_cancel.rs"',
        ["repo/core/engine/tests/evaluation_cancel.rs"],
    )
    assert not _verification_command_covers_test_changes("project-test --filter unrelated", changed)
    assert not _verification_command_covers_test_changes("project-test --only slow", changed)
    assert not _verification_command_covers_test_changes("project-test -k unrelated", changed)
    assert not _verification_command_covers_test_changes("project-test -m slow", changed)
    assert not _verification_command_covers_test_changes("project-test ./... -run Other", changed)
    assert _verification_command_covers_test_changes("project-test -m " + changed[0], changed)
    assert not _verification_command_covers_test_changes(
        "PROJECT_TEST_OPTS=--collect-only project-test "
        "t/unit/transport/virtual/test_sac_priority.case",
        changed,
    )
    assert not _verification_command_covers_test_changes(
        "PROJECT_TEST_FLAGS='-run Other' project-test ./...",
        changed,
    )
    assert not _verification_command_covers_test_changes(
        "project-test ./... --filter Other",
        changed,
    )
    assert not _verification_command_covers_test_changes(
        "project-test t/unit/transport/virtual/test_sac_priority.case --only slow",
        changed,
    )
    assert not _verification_command_covers_test_changes(
        "project-test t/unit/transport/virtual --ignore "
        "t/unit/transport/virtual/test_sac_priority.case",
        changed,
    )
    assert not _verification_command_covers_test_changes(
        "project-test t/unit/transport/virtual "
        "--ignore=t/unit/transport/virtual/test_sac_priority.case",
        changed,
    )
    assert not _verification_command_covers_test_changes(
        "project-test t/unit/transport/virtual --ignore-glob=t/unit/transport/virtual/test_sac_*",
        changed,
    )

    assert not _verification_command_covers_test_changes(
        "project-test --collect-only t/unit/transport/virtual/test_sac_priority.case",
        changed,
    )
    assert not _verification_command_covers_test_changes(
        "project-test --collect t/unit/transport/virtual/test_sac_priority.case",
        changed,
    )
    assert not _verification_command_covers_test_changes(
        "project-test --fixtures t/unit/transport/virtual/test_sac_priority.case",
        changed,
    )
    assert not _verification_command_covers_test_changes("project-test -list . ./...", changed)
    assert not _verification_command_covers_test_changes(
        "project-test -- --testNamePattern=Other",
        ["tests/test_feature.case"],
    )
    assert not _verification_command_covers_test_changes(
        "project-test t/unit/transport/virtual/test_sac_priority.case::test_specific",
        changed,
    )
    assert not _verification_command_covers_test_changes(
        "project-test t/unit/transport/virtual/test_base.case",
        changed,
    )
    assert not _verification_command_covers_test_changes("test -f " + changed[0], changed)
    assert not _verification_command_covers_test_changes("echo " + changed[0], changed)
    assert not _verification_command_covers_test_changes("rg case " + changed[0], changed)
    assert not _verification_command_covers_test_changes("cat " + changed[0], changed)
    assert not _verification_command_covers_test_changes(
        "runner -c \"open('t/unit/transport/virtual/test_sac_priority.case').read()\"",
        changed,
    )
    assert not _verification_command_covers_test_changes(
        "cat tests/regression.sh",
        ["tests/regression.sh"],
    )
    assert not _verification_command_covers_test_changes("true", changed)


def test_merge_workspace_status_expands_nested_git_checkout_changes() -> None:
    merged = _merge_workspace_status(
        "?? repo/\n",
        " M repo/mashumaro/helper.py\n M repo/tests/test_helper.py\n",
    )

    assert "?? repo/" not in merged
    assert _workspace_source_change_status(merged)[1] == ["repo/mashumaro/helper.py"]
    assert _workspace_test_change_paths(merged) == ["repo/tests/test_helper.py"]


def test_merge_workspace_status_preserves_hidden_nested_checkout_name() -> None:
    merged = _merge_workspace_status(
        "?? .anko-src/\n",
        " M .anko-src/anko_test.go\n",
    )

    assert "?? .anko-src/" not in merged
    assert _workspace_source_change_status(merged)[1] == []
    assert _workspace_test_change_paths(merged) == [".anko-src/anko_test.go"]


def test_merge_workspace_status_removes_clean_nested_checkout_root() -> None:
    merged = _merge_workspace_status(
        "?? repo/\n",
        "",
        nested_roots={"repo"},
    )

    assert "?? repo/" not in merged
    assert _workspace_source_change_status(merged)[1] == []
    assert _workspace_test_change_paths(merged) == []


def test_merge_workspace_status_removes_deep_nested_checkout_root() -> None:
    merged = _merge_workspace_status(
        "?? nested/anko/\n",
        " M nested/anko/ast/expr.go\n M nested/anko/core/testdata/func.ank\n",
        nested_roots={"nested/anko"},
    )

    assert "?? nested/anko/" not in merged
    assert _workspace_source_change_status(merged)[1] == ["nested/anko/ast/expr.go"]
    assert _workspace_test_change_paths(merged) == ["nested/anko/core/testdata/func.ank"]


def _call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id=f"call_{name}", name=name, arguments=arguments)


def _restricted_benchmark_policy() -> ExternalWorkspacePolicy:
    return ExternalWorkspacePolicy(
        forbidden_path_prefixes=("solution",),
        forbidden_path_parts=("hidden_tests", "hidden-tests"),
        forbidden_path_names=("solution.patch", "solve.sh", "test.patch"),
        forbidden_absolute_paths=("/solution", "/tests"),
        forbidden_text_fragments=(
            "solution.patch",
            "solve.sh",
            "test.patch",
            "github.com/datacurve-ai/deep-swe",
            "datacurve-ai/deep-swe",
            "datacurve-ai",
            "deep-swe",
        ),
        refusal_message=(
            "refused: external workspace policy blocks access to benchmark-private "
            "artifacts or forbidden source repositories"
        ),
    )


def test_absolute_policy_paths_do_not_block_public_repo_test_directories() -> None:
    policy = ExternalWorkspacePolicy(
        forbidden_absolute_paths=("/solution", "/tests"),
    )

    assert policy.references_forbidden_material("cat /tests/test.sh")
    assert policy.references_forbidden_material("ls /solution")
    assert not policy.references_forbidden_material("ls repo/tests")
    assert not policy.references_forbidden_material("pytest repo/tests/test_public.py")
    assert not policy.references_forbidden_material("find ./repo/tests -type f")


def test_external_workspace_attempt_budget_allows_multi_turn_verifier_repair() -> None:
    assert (
        external_workspace_repair_attempts(
            source_change_retries=0,
            verification_retries=0,
        )
        == 0
    )
    assert (
        external_workspace_repair_attempts(
            source_change_retries=1,
            verification_retries=0,
        )
        == 2
    )
    assert (
        external_workspace_repair_attempts(
            source_change_retries=0,
            verification_retries=1,
        )
        == 8
    )
    assert (
        external_workspace_repair_attempts(
            source_change_retries=1,
            verification_retries=1,
        )
        == 10
    )
    assert (
        external_workspace_total_attempts(
            source_change_retries=1,
            verification_retries=1,
        )
        == 11
    )


@pytest.mark.asyncio
async def test_remote_read_write_and_edit_file_tools_operate_in_workdir(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    writer = RemoteWriteFileTool(env, workdir=str(tmp_path))
    reader = RemoteReadFileTool(env, workdir=str(tmp_path))
    editor = RemoteEditFileTool(env, workdir=str(tmp_path))

    write_result = await writer(_call("write_file", path="src/demo.txt", content="hello world"))
    assert write_result.is_error is False
    assert (tmp_path / "src" / "demo.txt").read_text(encoding="utf-8") == "hello world"

    overwrite_result = await writer(_call("write_file", path="src/demo.txt", content="bad"))
    assert overwrite_result.is_error is True
    assert "file exists" in overwrite_result.content
    assert (tmp_path / "src" / "demo.txt").read_text(encoding="utf-8") == "hello world"

    edit_result = await editor(_call("edit_file", path="src/demo.txt", old="hello", new="goodbye"))
    assert edit_result.is_error is False

    read_result = await reader(_call("read_file", path="src/demo.txt"))
    assert read_result.is_error is False
    assert read_result.content == "goodbye world"

    noop_result = await writer(
        _call("write_file", path="src/demo.txt", content="goodbye world", overwrite=True)
    )
    assert noop_result.is_error is True
    assert "no-op" in noop_result.content


@pytest.mark.asyncio
async def test_remote_write_file_tool_rejects_repeated_failed_arguments(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    writer = RemoteWriteFileTool(env, workdir=str(tmp_path))

    first = await writer(_call("write_file", content="missing path"))
    second = await writer(_call("write_file", content="missing path"))

    assert first.is_error is True
    assert "path is required" in first.content
    assert second.is_error is True
    assert "already failed" in second.content
    assert "requires both path and content" in second.content
    assert not env.calls


@pytest.mark.asyncio
async def test_remote_write_file_tool_rejects_repeated_invalid_path_with_new_content(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    writer = RemoteWriteFileTool(env, workdir=str(tmp_path))

    first = await writer(_call("write_file", content="missing path one"))
    second = await writer(_call("write_file", content="missing path two"))

    assert first.is_error is True
    assert "path is required" in first.content
    assert second.is_error is True
    assert "path was already rejected" in second.content
    assert "relative workspace path" in second.content
    assert "Do not omit the path argument" not in second.content
    assert not env.calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("return_code", "expected"),
    [
        (5, "file exists and overwrite was not enabled"),
        (6, "no-op: file already has requested content"),
    ],
)
async def test_remote_write_file_error_guidance_survives_missing_stderr(
    tmp_path: Path,
    return_code: int,
    expected: str,
) -> None:
    writer = RemoteWriteFileTool(FixedExitEnvironment(return_code), workdir=str(tmp_path))

    result = await writer(_call("write_file", path="demo.txt", content="hello"))

    assert result.is_error is True
    assert expected in result.content


@pytest.mark.asyncio
async def test_remote_file_tools_reject_absolute_or_parent_paths(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    reader = RemoteReadFileTool(env, workdir=str(tmp_path))

    absolute = await reader(_call("read_file", path="/etc/passwd"))
    parent = await reader(_call("read_file", path="../secret"))

    assert absolute.is_error is True
    assert parent.is_error is True
    assert not env.calls


@pytest.mark.asyncio
async def test_remote_read_file_range_reads_large_file_sections(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    path = tmp_path / "large.txt"
    path.write_text("".join(f"line {idx}\n" for idx in range(1, 101)), encoding="utf-8")
    reader = RemoteReadFileTool(env, workdir=str(tmp_path), max_output_bytes=80)
    range_reader = RemoteReadFileRangeTool(env, workdir=str(tmp_path))

    full_result = await reader(_call("read_file", path="large.txt"))
    range_result = await range_reader(
        _call("read_file_range", path="large.txt", start_line=10, line_count=3)
    )

    assert full_result.is_error is True
    assert "use read_file_range" in full_result.content
    assert range_result.is_error is False
    assert range_result.content == "10:line 10\n11:line 11\n12:line 12\n"


@pytest.mark.asyncio
async def test_remote_read_file_large_file_error_survives_missing_stderr(tmp_path: Path) -> None:
    reader = RemoteReadFileTool(
        StderrDroppingEnvironment(),
        workdir=str(tmp_path),
        max_output_bytes=80,
    )

    result = await reader(_call("read_file", path="large.txt"))

    assert result.is_error is True
    assert "file too large" in result.content
    assert "use read_file_range" in result.content


@pytest.mark.asyncio
async def test_remote_read_only_inspection_is_tracked_without_blocking(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b\n", encoding="utf-8")
    env = LocalEnvironment(tmp_path)
    state = RemoteWorkspaceState()
    reader = RemoteReadFileTool(env, workdir=str(tmp_path), state=state)
    lister = RemoteListDirTool(env, workdir=str(tmp_path), state=state)
    writer = RemoteWriteFileTool(env, workdir=str(tmp_path), state=state)

    first = await lister(_call("list_dir", path="."))
    second = await reader(_call("read_file", path="a.txt"))
    third = await reader(_call("read_file", path="b.txt"))

    assert first.is_error is False
    assert second.is_error is False
    assert third.is_error is False
    assert third.content == "b\n"
    assert state.read_only_calls_since_change == 3

    changed = await writer(_call("write_file", path="src/change.go", content="package src\n"))
    after_change = await reader(_call("read_file", path="b.txt"))

    assert changed.is_error is False
    assert after_change.is_error is False
    assert after_change.content == "b\n"
    assert state.read_only_calls_since_change == 1


def test_remote_workspace_state_tracks_read_only_calls() -> None:
    assert RemoteWorkspaceState().read_only_calls_since_change == 0


@pytest.mark.asyncio
async def test_remote_apply_patch_tool_checks_then_applies_patch(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "demo.txt").write_text("hello\n", encoding="utf-8")
    patch_tool = RemoteApplyPatchTool(env, workdir=str(tmp_path))
    patch = """diff --git a/demo.txt b/demo.txt
--- a/demo.txt
+++ b/demo.txt
@@ -1 +1 @@
-hello
+goodbye
"""

    result = await patch_tool(_call("apply_patch", patch=patch))

    assert result.is_error is False
    assert (tmp_path / "demo.txt").read_text(encoding="utf-8") == "goodbye\n"


@pytest.mark.asyncio
async def test_remote_apply_patch_tool_recovers_unified_diff_with_codex_end_marker(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "demo.txt").write_text("hello\n", encoding="utf-8")
    patch_tool = RemoteApplyPatchTool(env, workdir=str(tmp_path))
    patch = """diff --git a/demo.txt b/demo.txt
--- a/demo.txt
+++ b/demo.txt
@@ -1 +1 @@
-hello
+goodbye
*** End Patch
"""

    result = await patch_tool(_call("apply_patch", patch=patch))

    assert result.is_error is False
    assert result.metadata["stripped_patch_envelope"] is True
    assert (tmp_path / "demo.txt").read_text(encoding="utf-8") == "goodbye\n"


@pytest.mark.asyncio
async def test_remote_apply_patch_tool_recovers_wrapped_unified_diff(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "demo.txt").write_text("hello\n", encoding="utf-8")
    patch_tool = RemoteApplyPatchTool(env, workdir=str(tmp_path))
    patch = """*** Begin Patch
diff --git a/demo.txt b/demo.txt
--- a/demo.txt
+++ b/demo.txt
@@ -1 +1 @@
-hello
+goodbye
*** End Patch
"""

    result = await patch_tool(_call("apply_patch", patch=patch))

    assert result.is_error is False
    assert result.metadata["stripped_patch_envelope"] is True
    assert (tmp_path / "demo.txt").read_text(encoding="utf-8") == "goodbye\n"


@pytest.mark.asyncio
async def test_remote_apply_patch_tool_applies_codex_update_patch(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    (tmp_path / "ast").mkdir()
    (tmp_path / "ast" / "expr.go").write_text(
        "type FuncExpr struct {\n"
        "\tExprImpl\n"
        "\tName   string\n"
        "\tStmt   Stmt\n"
        "\tParams []string\n"
        "\tVarArg bool\n"
        "}\n",
        encoding="utf-8",
    )
    patch_tool = RemoteApplyPatchTool(env, workdir=str(tmp_path))
    patch = """*** Begin Patch
*** Update File: ast/expr.go
@@
 type FuncExpr struct {
 \tExprImpl
 \tName   string
 \tStmt   Stmt
 \tParams []string
-\tVarArg bool
+\tDefaults []Expr
+\tVarArg   bool
 }
*** End Patch
"""

    result = await patch_tool(_call("apply_patch", patch=patch))

    assert result.is_error is False
    assert result.metadata["patch_format"] == "codex"
    assert (tmp_path / "ast" / "expr.go").read_text(encoding="utf-8") == (
        "type FuncExpr struct {\n"
        "\tExprImpl\n"
        "\tName   string\n"
        "\tStmt   Stmt\n"
        "\tParams []string\n"
        "\tDefaults []Expr\n"
        "\tVarArg   bool\n"
        "}\n"
    )


@pytest.mark.asyncio
async def test_remote_apply_patch_tool_applies_codex_add_file_patch(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    patch_tool = RemoteApplyPatchTool(env, workdir=str(tmp_path))
    patch = """*** Begin Patch
*** Add File: ast/param.go
+package ast
+
+type Param struct {
+\tName string
+}
*** End Patch
"""

    result = await patch_tool(_call("apply_patch", patch=patch))

    assert result.is_error is False
    assert (tmp_path / "ast" / "param.go").read_text(encoding="utf-8") == (
        "package ast\n\ntype Param struct {\n\tName string\n}\n"
    )


@pytest.mark.asyncio
async def test_remote_apply_patch_tool_rejects_codex_workspace_escape(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    patch_tool = RemoteApplyPatchTool(env, workdir=str(tmp_path))
    patch = """*** Begin Patch
*** Add File: ../escape.txt
+nope
*** End Patch
"""

    result = await patch_tool(_call("apply_patch", patch=patch))

    assert result.is_error is True
    assert "escapes the workspace" in result.content
    assert not (tmp_path.parent / "escape.txt").exists()
    assert not env.calls


@pytest.mark.asyncio
async def test_repository_url_policy_does_not_block_local_source_imports(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "loader.go").write_text("package evaluator\n\n", encoding="utf-8")
    policy = ExternalWorkspacePolicy(
        forbidden_web_fragments=("github.com/abs-lang/abs", "abs-lang/abs"),
        refusal_message="refused: forbidden source repository lookup",
    )
    patch_tool = RemoteApplyPatchTool(env, workdir=str(tmp_path), policy=policy)
    patch = (
        "diff --git a/loader.go b/loader.go\n"
        "--- a/loader.go\n"
        "+++ b/loader.go\n"
        "@@ -1,2 +1,6 @@\n"
        " package evaluator\n"
        " \n"
        "+import (\n"
        '+\t"github.com/abs-lang/abs/token"\n'
        "+)\n"
    )

    result = await patch_tool(_call("apply_patch", patch=patch))

    assert result.is_error is False
    assert "github.com/abs-lang/abs/token" in (tmp_path / "loader.go").read_text(encoding="utf-8")


def test_normalize_unified_diff_hunk_headers_recounts_body_lines() -> None:
    patch = """--- a/demo.txt
+++ b/demo.txt
@@ -1,99 +1,99 @@
 keep
-old
+new
"""

    normalized, changed = _normalize_unified_diff_hunk_headers(patch)

    assert changed is True
    assert "@@ -1,2 +1,2 @@" in normalized
    assert "-old\n+new\n" in normalized


def test_normalize_unified_diff_hunk_headers_repairs_bare_blank_context() -> None:
    patch = """--- a/demo.txt
+++ b/demo.txt
@@ -1,99 +1,99 @@
 one

+inserted
 two
"""

    normalized, changed = _normalize_unified_diff_hunk_headers(patch)

    assert changed is True
    assert "@@ -1,3 +1,4 @@" in normalized
    assert " one\n \n+inserted\n two\n" in normalized


def test_normalize_unified_diff_hunk_headers_repairs_missing_context_prefixes() -> None:
    patch = """--- a/demo.txt
+++ b/demo.txt
@@ -1,99 +1,99 @@
one
-old
+new
\ttwo
"""

    normalized, changed = _normalize_unified_diff_hunk_headers(patch)

    assert changed is True
    assert "@@ -1,3 +1,3 @@" in normalized
    assert " one\n-old\n+new\n \ttwo\n" in normalized


@pytest.mark.asyncio
async def test_remote_apply_patch_tool_recovers_bad_hunk_counts(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "demo.txt").write_text("hello\nworld\n", encoding="utf-8")
    patch_tool = RemoteApplyPatchTool(env, workdir=str(tmp_path))
    patch = """--- a/demo.txt
+++ b/demo.txt
@@ -1,99 +1,99 @@
-hello
+goodbye
 world
"""

    result = await patch_tool(_call("apply_patch", patch=patch))

    assert result.is_error is False
    assert "normalized unified diff hunk headers" in result.content
    assert result.metadata == {"exit_code": 0, "normalized_hunk_headers": True}
    assert (tmp_path / "demo.txt").read_text(encoding="utf-8") == "goodbye\nworld\n"


@pytest.mark.asyncio
async def test_remote_apply_patch_tool_recovers_bare_blank_context_lines(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "demo.txt").write_text("one\n\ntwo\n", encoding="utf-8")
    patch_tool = RemoteApplyPatchTool(env, workdir=str(tmp_path))
    patch = """--- a/demo.txt
+++ b/demo.txt
@@ -1,99 +1,99 @@
 one

+inserted
 two
"""

    result = await patch_tool(_call("apply_patch", patch=patch))

    assert result.is_error is False
    assert "normalized unified diff hunk headers" in result.content
    assert result.metadata == {"exit_code": 0, "normalized_hunk_headers": True}
    assert (tmp_path / "demo.txt").read_text(encoding="utf-8") == "one\n\ninserted\ntwo\n"


@pytest.mark.asyncio
async def test_remote_apply_patch_tool_recovers_missing_context_prefixes(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "demo.txt").write_text("one\nold\n\ttwo\n", encoding="utf-8")
    patch_tool = RemoteApplyPatchTool(env, workdir=str(tmp_path))
    patch = """--- a/demo.txt
+++ b/demo.txt
@@ -1,99 +1,99 @@
one
-old
+new
\ttwo
"""

    result = await patch_tool(_call("apply_patch", patch=patch))

    assert result.is_error is False
    assert "normalized unified diff hunk headers" in result.content
    assert result.metadata == {"exit_code": 0, "normalized_hunk_headers": True}
    assert (tmp_path / "demo.txt").read_text(encoding="utf-8") == "one\nnew\n\ttwo\n"


@pytest.mark.asyncio
async def test_remote_apply_patch_tool_recovers_multi_hunk_missing_context_prefixes(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "demo.txt").write_text(
        "alpha\nold one\nbeta\nmiddle\nold two\nomega\n",
        encoding="utf-8",
    )
    patch_tool = RemoteApplyPatchTool(env, workdir=str(tmp_path))
    patch = """--- a/demo.txt
+++ b/demo.txt
@@ -1,99 +1,99 @@
alpha
-old one
+new one
beta
@@ -4,99 +4,99 @@
middle
-old two
+new two
omega
"""

    result = await patch_tool(_call("apply_patch", patch=patch))

    assert result.is_error is False
    assert "normalized unified diff hunk headers" in result.content
    assert (tmp_path / "demo.txt").read_text(encoding="utf-8") == (
        "alpha\nnew one\nbeta\nmiddle\nnew two\nomega\n"
    )


@pytest.mark.asyncio
async def test_remote_apply_patch_tool_reports_failed_normalized_attempt(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "demo.txt").write_text("one\nold\n", encoding="utf-8")
    patch_tool = RemoteApplyPatchTool(env, workdir=str(tmp_path))
    patch = """--- a/demo.txt
+++ b/demo.txt
@@ -1,99 +1,99 @@
one
-missing
+new
"""

    result = await patch_tool(_call("apply_patch", patch=patch))

    assert result.is_error is True
    assert "Normalized patch attempt also failed" in result.content
    assert "The patch was not applied." in result.content
    assert "edit_file with an exact old/new block" in result.content
    assert "write_file with overwrite=true" in result.content
    assert "Do not retry this exact patch" not in result.content


@pytest.mark.asyncio
async def test_remote_apply_patch_tool_guides_ambiguous_hunk(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "demo.txt").write_text(
        "class A:\n    def copy(self):\n        return res\n\n"
        "class B:\n    def copy(self):\n        return res\n",
        encoding="utf-8",
    )
    patch_tool = RemoteApplyPatchTool(env, workdir=str(tmp_path))
    patch = """*** Begin Patch
*** Update File: demo.txt
@@
     def copy(self):
         return res
+    def partial(self):
+        return res
*** End Patch
"""

    result = await patch_tool(_call("apply_patch", patch=patch))

    assert result.is_error is True
    assert "matched demo.txt 2 times" in result.content
    assert "include more unique surrounding context" in result.content
    assert "read_file_range" in result.content
    assert "The patch was not applied." in result.content
    assert "edit_file with an exact old/new block" in result.content


@pytest.mark.asyncio
async def test_remote_apply_patch_tool_guides_missing_patch_argument(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    patch_tool = RemoteApplyPatchTool(env, workdir=str(tmp_path))

    result = await patch_tool(_call("apply_patch"))

    assert result.is_error is True
    assert "patch must be a non-empty string" in result.content
    assert "The patch was not applied." in result.content
    assert "Do not retry this exact patch" not in result.content
    assert "edit_file with an exact old/new block" in result.content
    assert "write_file with overwrite=true" in result.content
    assert not env.calls


@pytest.mark.asyncio
async def test_remote_apply_patch_tool_rejects_exact_failed_retry_until_workspace_changes(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    patch_tool = RemoteApplyPatchTool(env, workdir=str(tmp_path))
    bad_patch = "--- a/demo.txt\n+++ b/demo.txt\n@@\n"

    first = await patch_tool(_call("apply_patch", patch=bad_patch))
    call_count_after_first = len(env.calls)
    repeated = await patch_tool(_call("apply_patch", patch=bad_patch))
    patch_tool._mark_workspace_changed()
    after_change = await patch_tool(_call("apply_patch", patch=bad_patch))

    assert first.is_error is True
    assert "The patch was not applied." in first.content
    assert "Do not retry this exact patch" not in first.content
    assert "edit_file with an exact old/new block" in first.content
    assert repeated.is_error is True
    assert "already failed" in repeated.content
    assert "The patch was not applied." in repeated.content
    assert "Do not call apply_patch with this payload again" not in repeated.content
    assert "write_file with overwrite=true" in repeated.content
    assert len(env.calls) == call_count_after_first + 1
    assert after_change.is_error is True


@pytest.mark.asyncio
async def test_remote_apply_patch_tool_rejects_benchmark_artifact_paths(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    patch_tool = RemoteApplyPatchTool(
        env,
        workdir=str(tmp_path),
        policy=_restricted_benchmark_policy(),
    )

    result = await patch_tool(
        _call("apply_patch", patch="diff --git a/solution.patch b/solution.patch\n")
    )

    assert result.is_error is True
    assert "benchmark-private artifacts" in result.content
    assert not env.calls


@pytest.mark.asyncio
async def test_remote_edit_file_tool_rejects_exact_failed_retry_until_workspace_changes(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    (tmp_path / "demo.txt").write_text("hello\n", encoding="utf-8")
    editor = RemoteEditFileTool(env, workdir=str(tmp_path))
    call = _call("edit_file", path="demo.txt", old="missing", new="replacement")

    first = await editor(call)
    call_count_after_first = len(env.calls)
    repeated = await editor(call)
    editor._mark_workspace_changed()
    after_change = await editor(call)

    assert first.is_error is True
    assert "old text must appear exactly once" in first.content
    assert "The edit was not applied." in first.content
    assert "Read the current file slice" not in first.content
    assert "apply_patch" not in first.content
    assert "write_file overwrite=true" not in first.content
    assert repeated.is_error is True
    assert "already failed" in repeated.content
    assert "The edit was not applied." in repeated.content
    assert "Read the current file slice" not in repeated.content
    assert len(env.calls) == call_count_after_first + 1
    assert after_change.is_error is True


@pytest.mark.asyncio
async def test_remote_path_tools_reject_benchmark_artifact_paths(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    policy = _restricted_benchmark_policy()
    reader = RemoteReadFileTool(env, workdir=str(tmp_path), policy=policy)
    range_reader = RemoteReadFileRangeTool(env, workdir=str(tmp_path), policy=policy)
    lister = RemoteListDirTool(env, workdir=str(tmp_path), policy=policy)
    writer = RemoteWriteFileTool(env, workdir=str(tmp_path), policy=policy)

    read_result = await reader(_call("read_file", path="solution/solve.sh"))
    range_result = await range_reader(
        _call("read_file_range", path="solution/solve.sh", start_line=1)
    )
    list_result = await lister(_call("list_dir", path="solution"))
    write_result = await writer(
        _call("write_file", path="solution.patch", content="patch", overwrite=True)
    )

    assert read_result.is_error is True
    assert range_result.is_error is True
    assert list_result.is_error is True
    assert write_result.is_error is True
    assert "benchmark-private artifacts" in read_result.content
    assert "benchmark-private artifacts" in range_result.content
    assert "benchmark-private artifacts" in list_result.content
    assert "benchmark-private artifacts" in write_result.content
    assert not env.calls


@pytest.mark.asyncio
async def test_remote_tools_hide_harness_owned_log_artifacts(tmp_path: Path) -> None:
    logs_dir = tmp_path / "harness-logs"
    logs_dir.mkdir()
    (logs_dir / "harness-events.jsonl").write_text("private verifier command\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    env = LocalEnvironment(tmp_path)
    policy = ExternalWorkspacePolicy(
        forbidden_path_prefixes=("harness-logs",),
        forbidden_text_fragments=("harness-logs",),
    )
    reader = RemoteReadFileTool(env, workdir=str(tmp_path), policy=policy)
    lister = RemoteListDirTool(env, workdir=str(tmp_path), policy=policy)
    shell = RemoteShellTool(env, workdir=str(tmp_path), policy=policy)

    read_result = await reader(_call("read_file", path="harness-logs/harness-events.jsonl"))
    list_result = await lister(_call("list_dir", path="."))
    shell_result = await shell(_call("shell", command="find . -maxdepth 2 -type f | sort"))

    assert read_result.is_error is True
    assert list_result.is_error is False
    assert shell_result.is_error is False
    assert "harness-logs" not in list_result.content
    assert "harness-logs" not in shell_result.content
    assert "harness-events.jsonl" not in shell_result.content
    assert "src" in list_result.content
    assert "src/app.py" in shell_result.content


@pytest.mark.asyncio
async def test_verify_classification_uses_raw_output_before_redaction(tmp_path: Path) -> None:
    class HiddenFailureOutputEnvironment:
        async def exec(
            self,
            command: str,
            *,
            cwd: str | None = None,
            timeout_sec: int | None = None,
        ) -> ExecResult:
            if command == "git status --porcelain --untracked-files=all":
                return ExecResult(stdout="", stderr="", return_code=0)
            return ExecResult(stdout="FAILED harness-logs\n", stderr="", return_code=0)

    env = HiddenFailureOutputEnvironment()
    policy = ExternalWorkspacePolicy(
        forbidden_path_prefixes=("harness-logs",),
        forbidden_text_fragments=("harness-logs",),
    )
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path), policy=policy)

    result = await verify(_call("verify_work", command="printf ok"))

    assert result.is_error is True
    assert result.content.startswith("FAILED (output reports failure)")
    assert "harness-logs" not in result.content


@pytest.mark.asyncio
async def test_remote_path_policy_is_configurable_not_hardcoded(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    (tmp_path / "solution").mkdir()
    (tmp_path / "solution" / "solve.sh").write_text("echo ok\n", encoding="utf-8")

    unrestricted_reader = RemoteReadFileTool(env, workdir=str(tmp_path))
    restricted_reader = RemoteReadFileTool(
        env,
        workdir=str(tmp_path),
        policy=_restricted_benchmark_policy(),
    )

    unrestricted = await unrestricted_reader(_call("read_file", path="solution/solve.sh"))
    restricted = await restricted_reader(_call("read_file", path="solution/solve.sh"))

    assert unrestricted.is_error is False
    assert "echo ok" in unrestricted.content
    assert restricted.is_error is True
    assert "benchmark-private artifacts" in restricted.content


@pytest.mark.asyncio
async def test_remote_shell_tool_blocks_benchmark_leakage_paths(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path), policy=_restricted_benchmark_policy())

    result = await shell(_call("shell", command="cat /solution/solve.sh"))
    hidden_tests = await shell(_call("shell", command="cat /tests/hidden_case.py"))
    repo_result = await shell(
        _call("shell", command="curl https://github.com/datacurve-ai/deep-swe")
    )
    encoded_repo_result = await shell(
        _call(
            "shell",
            command=("curl https://github.com/datacurve-ai/%64%65%65%70%2d%73%77%65"),
        )
    )
    search_result = await shell(
        _call("shell", command="curl 'https://search.example/?q=deep-swe+anko'")
    )

    assert result.is_error is True
    assert "benchmark-private artifacts" in result.content
    assert hidden_tests.is_error is True
    assert "benchmark-private artifacts" in hidden_tests.content
    assert repo_result.is_error is True
    assert "forbidden source repositories" in repo_result.content
    assert encoded_repo_result.is_error is True
    assert "forbidden source repositories" in encoded_repo_result.content
    assert search_result.is_error is True
    assert "forbidden source repositories" in search_result.content
    assert not env.calls


@pytest.mark.asyncio
async def test_remote_web_tools_block_forbidden_benchmark_lookups(tmp_path: Path) -> None:
    policy = _restricted_benchmark_policy()
    web_search = PolicyWebSearchTool(policy=policy)
    fetch_url = PolicyFetchUrlTool(policy=policy)

    search_result = await web_search(
        _call("web_search", query="deep-swe anko default function arguments")
    )
    fetch_result = await fetch_url(
        _call("fetch_url", url="https://github.com/datacurve-ai/deep-swe")
    )

    assert search_result.is_error is True
    assert fetch_result.is_error is True
    assert "forbidden source repositories" in search_result.content
    assert "forbidden source repositories" in fetch_result.content


@pytest.mark.asyncio
async def test_remote_web_tools_can_be_disabled_by_external_policy(tmp_path: Path) -> None:
    policy = ExternalWorkspacePolicy(
        allow_web_access=False,
        refusal_message="refused: web access disabled",
    )
    web_search = PolicyWebSearchTool(policy=policy)
    fetch_url = PolicyFetchUrlTool(policy=policy)

    search_result = await web_search(_call("web_search", query="python cattrs docs"))
    fetch_result = await fetch_url(_call("fetch_url", url="https://docs.python.org/3/"))

    assert search_result.is_error is True
    assert fetch_result.is_error is True
    assert "web access disabled" in search_result.content
    assert "web access disabled" in fetch_result.content


@pytest.mark.asyncio
async def test_remote_web_tools_allow_public_research_by_default() -> None:
    policy = ExternalWorkspacePolicy(
        forbidden_web_fragments=("github.com/datacurve-ai/deep-swe", "datacurve-ai/deep-swe"),
        refusal_message="refused: forbidden source repository lookup",
    )
    search_backend = FakeWebTool(
        name="web_search",
        content="Results for: docker command not found\n\n1. Docker documentation\n   URL: https://docs.docker.com/",
        metadata={
            "results": [
                {
                    "title": "Docker documentation",
                    "url": "https://docs.docker.com/",
                    "content": "Install and run Docker.",
                }
            ]
        },
    )
    fetch_backend = FakeWebTool(
        name="fetch_url",
        content="status: 200\ncontent-type: text/html\n\nDocker documentation",
    )
    web_search = PolicyWebSearchTool(policy=policy, tool=search_backend)  # type: ignore[arg-type]
    fetch_url = PolicyFetchUrlTool(policy=policy, tool=fetch_backend)  # type: ignore[arg-type]

    search_result = await web_search(
        _call("web_search", query="docker command not found go test container setup")
    )
    fetch_result = await fetch_url(_call("fetch_url", url="https://docs.docker.com/engine/"))

    assert search_result.is_error is False
    assert fetch_result.is_error is False
    assert "Docker documentation" in search_result.content
    assert "Docker documentation" in fetch_result.content
    assert len(search_backend.calls) == 1
    assert len(fetch_backend.calls) == 1


@pytest.mark.asyncio
async def test_remote_web_search_redacts_entire_restricted_result_block() -> None:
    policy = ExternalWorkspacePolicy(
        forbidden_web_fragments=("Add flattened dataclass fields to Mashumaro field options",),
        refusal_message="refused: forbidden benchmark lookup",
    )
    search_backend = FakeWebTool(
        name="web_search",
        content=(
            "Results for: field options flattening\n\n"
            "1. Add flattened dataclass fields to Mashumaro field options\n"
            "   Validate at class creation and account for flattened keys.\n"
            "   URL: https://example.test/restricted\n\n"
            "2. Python dataclasses documentation\n"
            "   Public dataclass reference material.\n"
            "   URL: https://docs.python.org/3/library/dataclasses.html"
        ),
        metadata={
            "results": [
                {
                    "title": "Add flattened dataclass fields to Mashumaro field options",
                    "url": "https://example.test/restricted",
                    "content": "Validate at class creation and account for flattened keys.",
                },
                {
                    "title": "Python dataclasses documentation",
                    "url": "https://docs.python.org/3/library/dataclasses.html",
                    "content": "Public dataclass reference material.",
                },
            ]
        },
    )
    web_search = PolicyWebSearchTool(policy=policy, tool=search_backend)  # type: ignore[arg-type]

    result = await web_search(_call("web_search", query="field options flattening public docs"))

    assert result.is_error is False
    assert "Add flattened dataclass fields" not in result.content
    assert "Validate at class creation" not in result.content
    assert "https://example.test/restricted" not in result.content
    assert "Python dataclasses documentation" in result.content
    assert "restricted artifact result(s) omitted" in result.content
    assert result.metadata["restricted_results_omitted"] == 1
    assert len(result.metadata["results"]) == 1
    assert result.metadata["results"][0]["title"] == "Python dataclasses documentation"


@pytest.mark.asyncio
async def test_web_disabled_policy_still_allows_local_shell_discovery(tmp_path: Path) -> None:
    policy = ExternalWorkspacePolicy(
        allow_web_access=False,
        refusal_message="refused: web access disabled",
    )
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path), policy=policy)

    local_result = await shell(_call("shell", command="python -c \"print('ok')\""))
    network_result = await shell(_call("shell", command="curl https://docs.python.org/3/"))

    assert local_result.is_error is False
    assert "ok" in local_result.content
    assert network_result.is_error is True
    assert "web access disabled" in network_result.content


@pytest.mark.asyncio
async def test_web_only_repository_policy_blocks_network_lookups_not_local_grep(
    tmp_path: Path,
) -> None:
    policy = ExternalWorkspacePolicy(
        forbidden_web_fragments=("github.com/abs-lang/abs", "abs-lang/abs"),
        refusal_message="refused: forbidden source repository lookup",
    )
    web_search = PolicyWebSearchTool(policy=policy)
    fetch_url = PolicyFetchUrlTool(policy=policy)

    search_result = await web_search(
        _call("web_search", query="github.com/abs-lang/abs require cache")
    )
    fetch_result = await fetch_url(_call("fetch_url", url="https://github.com/abs-lang/abs"))

    assert search_result.is_error is True
    assert fetch_result.is_error is True

    env = LocalEnvironment(tmp_path)
    (tmp_path / "go.mod").write_text("module github.com/abs-lang/abs\n", encoding="utf-8")
    shell = RemoteShellTool(env, workdir=str(tmp_path), policy=policy)
    grep_result = await shell(_call("shell", command="grep github.com/abs-lang/abs go.mod"))
    curl_result = await shell(_call("shell", command="curl https://github.com/abs-lang/abs"))

    assert grep_result.is_error is False
    assert "module github.com/abs-lang/abs" in grep_result.content
    assert curl_result.is_error is True
    assert "forbidden source repository" in curl_result.content


@pytest.mark.asyncio
async def test_repository_policy_allows_task_repo_clone_but_blocks_repo_searches(
    tmp_path: Path,
) -> None:
    fragments = ("https://github.com/python-attrs/cattrs", "python-attrs/cattrs")
    policy = ExternalWorkspacePolicy(
        forbidden_web_fragments=fragments,
        allowed_git_clone_fragments=fragments,
        refusal_message="refused: forbidden source repository lookup",
    )
    env = StaticStatusEnvironment(tmp_path, statuses=[])
    shell = RemoteShellTool(env, workdir=str(tmp_path), policy=policy)
    web_search = PolicyWebSearchTool(policy=policy)
    fetch_url = PolicyFetchUrlTool(policy=policy)

    clone_result = await shell(
        _call(
            "shell",
            command=(
                "git clone https://github.com/python-attrs/cattrs repo && "
                "cd repo && git checkout 6bc4708fb9b2ac52d9a18997e923da6a58916102 && "
                "python -m pip install -e ."
            ),
        )
    )
    clone_inspect_result = await shell(
        _call(
            "shell",
            command=(
                "git clone https://github.com/python-attrs/cattrs repo2 && "
                "cd repo2 && git checkout 6bc4708fb9b2ac52d9a18997e923da6a58916102 && "
                "ls"
            ),
        )
    )
    ls_remote_result = await shell(
        _call("shell", command="git ls-remote https://github.com/python-attrs/cattrs.git")
    )
    fetch_commit_result = await shell(
        _call(
            "shell",
            command=(
                "git fetch --depth 1 https://github.com/python-attrs/cattrs "
                "6bc4708fb9b2ac52d9a18997e923da6a58916102"
            ),
        )
    )
    clone_fetch_origin_result = await shell(
        _call(
            "shell",
            command=(
                "git clone --depth 1 https://github.com/python-attrs/cattrs repo3 && "
                "cd repo3 && "
                "git fetch --depth 1 origin 6bc4708fb9b2ac52d9a18997e923da6a58916102 && "
                "git checkout 6bc4708fb9b2ac52d9a18997e923da6a58916102"
            ),
        )
    )
    cleanup_clone_result = await shell(
        _call(
            "shell",
            command=(
                "rm -rf repo && mkdir -p repo && "
                "git clone https://github.com/python-attrs/cattrs repo/cattrs"
            ),
        )
    )
    cleanup_contents_clone_result = await shell(
        _call(
            "shell",
            command=(
                "mkdir -p cattrs_src && rm -rf cattrs_src/* && "
                "git clone https://github.com/python-attrs/cattrs cattrs_src && "
                "cd cattrs_src && git checkout 6bc4708fb9b2ac52d9a18997e923da6a58916102"
            ),
        )
    )
    docker_clone_result = await shell(
        _call(
            "shell",
            command=(
                'docker run --rm -v "$PWD":/app -w /app example.test/toolchain:latest '
                "bash -lc 'git clone https://github.com/python-attrs/cattrs . && "
                "git checkout 6bc4708fb9b2ac52d9a18997e923da6a58916102'"
            ),
        )
    )
    docker_clone_inspect_result = await shell(
        _call(
            "shell",
            command=(
                'docker run --rm -t -v "$PWD":/app -w /app example.test/toolchain:latest '
                "bash -lc 'git clone https://github.com/python-attrs/cattrs repo && "
                "cd repo && git checkout 6bc4708fb9b2ac52d9a18997e923da6a58916102 && ls'"
            ),
        )
    )
    docker_clone_project_command_result = await shell(
        _call(
            "shell",
            command=(
                "docker run --rm -t example.test/toolchain:latest "
                "bash -lc 'git clone https://github.com/python-attrs/cattrs repo4 && "
                "cd repo4 && git checkout 6bc4708fb9b2ac52d9a18997e923da6a58916102 && "
                "./node_modules/.bin/local-check --version && make test'"
            ),
        )
    )
    unsafe_cleanup_clone_result = await shell(
        _call(
            "shell",
            command=(
                "rm -rf /tmp/repo && mkdir -p repo && "
                "git clone https://github.com/python-attrs/cattrs repo/cattrs"
            ),
        )
    )
    remote_context_result = await shell(
        _call(
            "shell",
            command="git remote -v && git ls-remote https://github.com/python-attrs/cattrs | head",
        )
    )
    curl_result = await shell(
        _call("shell", command="curl https://github.com/python-attrs/cattrs/issues")
    )
    docker_curl_result = await shell(
        _call(
            "shell",
            command=(
                "docker run --rm example.test/toolchain:latest "
                "curl https://github.com/python-attrs/cattrs/issues"
            ),
        )
    )
    search_result = await web_search(
        _call("web_search", query="python-attrs/cattrs partial_structure solution")
    )
    fetch_result = await fetch_url(
        _call("fetch_url", url="https://github.com/python-attrs/cattrs/issues")
    )

    assert clone_result.is_error is False
    assert clone_inspect_result.is_error is False
    assert ls_remote_result.is_error is False
    assert fetch_commit_result.is_error is False
    assert clone_fetch_origin_result.is_error is False
    assert cleanup_clone_result.is_error is False
    assert cleanup_contents_clone_result.is_error is False
    assert docker_clone_result.is_error is False
    assert docker_clone_inspect_result.is_error is False
    assert docker_clone_project_command_result.is_error is False
    assert unsafe_cleanup_clone_result.is_error is True
    assert remote_context_result.is_error is False
    assert env.calls
    assert any(
        "git clone https://github.com/python-attrs/cattrs" in str(call["command"])
        for call in env.calls
    )
    assert curl_result.is_error is True
    assert docker_curl_result.is_error is True
    assert search_result.is_error is True
    assert fetch_result.is_error is True
    assert "forbidden source repository" in curl_result.content
    assert "forbidden source repository" in docker_curl_result.content
    assert "forbidden source repository" in search_result.content
    assert "forbidden source repository" in fetch_result.content
    assert "use shell git clone or git ls-remote" in search_result.content
    assert "use shell git clone or git ls-remote" in fetch_result.content


@pytest.mark.asyncio
async def test_repository_policy_blocks_allowed_clone_cleanup_of_restricted_paths(
    tmp_path: Path,
) -> None:
    fragments = ("https://github.com/python-attrs/cattrs", "python-attrs/cattrs")
    policy = ExternalWorkspacePolicy(
        forbidden_path_parts=("solution",),
        forbidden_web_fragments=fragments,
        allowed_git_clone_fragments=fragments,
        refusal_message="refused: forbidden source repository lookup",
    )
    env = StaticStatusEnvironment(tmp_path, statuses=[])
    shell = RemoteShellTool(env, workdir=str(tmp_path), policy=policy)

    result = await shell(
        _call(
            "shell",
            command=("rm -rf solution/* && git clone https://github.com/python-attrs/cattrs repo"),
        )
    )

    assert result.is_error is True


@pytest.mark.asyncio
async def test_repository_policy_allows_task_repo_clone_when_url_is_forbidden_text(
    tmp_path: Path,
) -> None:
    fragments = ("https://github.com/reagento/adaptix", "reagento/adaptix")
    policy = ExternalWorkspacePolicy(
        forbidden_text_fragments=fragments,
        forbidden_web_fragments=fragments,
        allowed_git_clone_fragments=fragments,
        refusal_message="refused: forbidden source repository lookup",
    )
    env = StaticStatusEnvironment(tmp_path, statuses=[])
    shell = RemoteShellTool(env, workdir=str(tmp_path), policy=policy)

    allowed_clone = await shell(
        _call(
            "shell",
            command=(
                "git clone https://github.com/reagento/adaptix repo && "
                "cd repo && git checkout a691069fcadf9131e5f7a5a130a022dc678f3e1d"
            ),
        )
    )
    blocked_non_clone = await shell(
        _call(
            "shell",
            command="printf 'https://github.com/reagento/adaptix\\n'",
        )
    )

    assert allowed_clone.is_error is False
    assert "forbidden source repository" not in allowed_clone.content
    assert blocked_non_clone.is_error is True
    assert "forbidden source repository" in blocked_non_clone.content


@pytest.mark.asyncio
async def test_remote_shell_tool_blocks_obfuscated_benchmark_leakage_paths(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path), policy=_restricted_benchmark_policy())

    solution_result = await shell(
        _call("shell", command="python -c \"open('/sol'+'ution/solve.sh').read()\"")
    )
    hidden_tests = await shell(
        _call("shell", command="python -c \"open('/te'+'sts/hidden_case.py').read()\"")
    )
    repo_result = await shell(
        _call("shell", command="curl https://github.com/datacurve-ai/deep'-'swe")
    )

    assert solution_result.is_error is True
    assert "benchmark-private artifacts" in solution_result.content
    assert hidden_tests.is_error is True
    assert "benchmark-private artifacts" in hidden_tests.content
    assert repo_result.is_error is True
    assert "forbidden source repositories" in repo_result.content
    assert not env.calls


@pytest.mark.asyncio
async def test_remote_verify_work_tool_blocks_obfuscated_benchmark_leakage_paths(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path), policy=_restricted_benchmark_policy())

    result = await verify(
        _call("verify_work", command="python -c \"open('/sol'+'ution/solve.sh').read()\"")
    )

    assert result.is_error is True
    assert "benchmark-private artifacts" in result.content
    assert not env.calls


@pytest.mark.asyncio
async def test_remote_verify_work_tool_allows_policy_restricted_default_command(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(
        env,
        workdir=str(tmp_path),
        policy=_restricted_benchmark_policy(),
        default_command="printf ok # /tests/test.patch",
    )

    default_result = await verify(_call("verify_work"))
    supplied_result = await verify(_call("verify_work", command="printf ok # /solution/solve.sh"))

    assert default_result.is_error is False
    assert default_result.metadata["used_default_command"] is True
    assert supplied_result.is_error is True
    assert "benchmark-private artifacts" in supplied_result.content


@pytest.mark.asyncio
async def test_remote_shell_tool_allows_repository_tests_directory(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))
    (tmp_path / "tests").mkdir()

    result = await shell(_call("shell", command="ls ./tests"))

    assert result.is_error is False
    assert env.calls


@pytest.mark.asyncio
async def test_remote_shell_tool_blocks_root_filesystem_probe(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(_call("shell", command="find / -maxdepth 2 -type d"))

    assert result.is_error is True
    assert "container root filesystem" in result.content
    assert not env.calls


@pytest.mark.asyncio
async def test_remote_shell_tool_blocks_parent_directory_escape(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(
        _call("shell", command='grep -R "invalid default argument declaration" -n ..')
    )

    assert result.is_error is True
    assert "parent directories outside the repository" in result.content
    assert not env.calls


@pytest.mark.asyncio
async def test_remote_shell_tool_blocks_nested_parent_directory_escape(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(_call("shell", command="python -c \"open('../secret').read()\""))

    assert result.is_error is True
    assert "parent directories outside the repository" in result.content
    assert not env.calls


@pytest.mark.asyncio
async def test_remote_shell_tool_blocks_absolute_host_path_escape(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(_call("shell", command="ls -la /Users/leiro/workspace/harness"))

    assert result.is_error is True
    assert "absolute host filesystem paths" in result.content
    assert not env.calls


@pytest.mark.asyncio
async def test_remote_shell_tool_blocks_nested_absolute_host_path_escape(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(
        _call("shell", command="python -c \"open('/Users/leiro/workspace/harness').read()\"")
    )

    assert result.is_error is True
    assert "absolute host filesystem paths" in result.content
    assert not env.calls


@pytest.mark.asyncio
async def test_remote_shell_tool_allows_backslash_escaped_search_regex(tmp_path: Path) -> None:
    (tmp_path / "main.go").write_text("package main\n", encoding="utf-8")
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(
        _call(
            "shell",
            command="find . -maxdepth 2 -type f | grep -E '\\.(go|mod|sum)$' | head",
        )
    )

    assert result.is_error is False
    assert "stdout:\n./main.go" in result.content
    assert "absolute host filesystem paths" not in result.content
    assert env.calls


@pytest.mark.asyncio
async def test_remote_shell_tool_allows_heredoc_body_with_code_comments(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(
        _call(
            "shell",
            command=("cat <<'EOF'\n// harmless code comment in a heredoc body\nEOF"),
        )
    )

    assert result.is_error is False
    assert "harmless code comment" in result.content
    assert "absolute host filesystem paths" not in result.content
    assert env.calls


@pytest.mark.asyncio
async def test_remote_verify_work_tool_blocks_parent_directory_escape(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    result = await verify(_call("verify_work", command="find .. -name '*.go'"))

    assert result.is_error is True
    assert "parent directories outside the repository" in result.content
    assert not env.calls


@pytest.mark.asyncio
async def test_remote_verify_work_tool_blocks_absolute_host_path_escape(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    result = await verify(_call("verify_work", command="ls -la /Users/leiro/workspace/harness"))

    assert result.is_error is True
    assert "absolute host filesystem paths" in result.content
    assert not env.calls


@pytest.mark.asyncio
async def test_remote_shell_tool_runs_allowed_command(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(_call("shell", command="printf ok"))

    assert result.is_error is False
    assert "stdout:\nok" in result.content
    assert env.calls[0]["cwd"] == str(tmp_path)


@pytest.mark.asyncio
async def test_remote_shell_tool_allows_docker_container_internal_paths(tmp_path: Path) -> None:
    env = StaticStatusEnvironment(tmp_path, statuses=[])
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(
        _call(
            "shell",
            command="docker run --rm -w /app example.test/toolchain:latest go test ./...",
        )
    )

    assert result.is_error is False
    assert "absolute host filesystem paths" not in result.content


@pytest.mark.asyncio
async def test_remote_shell_tool_allows_chained_docker_container_internal_paths(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(tmp_path, statuses=[])
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(
        _call(
            "shell",
            command=(
                'cd repo && docker run --rm -v "$(pwd)":/app -w /app '
                "example.test/toolchain:latest go test ./..."
            ),
        )
    )

    assert result.is_error is False
    assert "absolute host filesystem paths" not in result.content


@pytest.mark.asyncio
async def test_remote_verify_work_allows_chained_docker_container_internal_paths(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(tmp_path, statuses=[])
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    result = await verify(
        _call(
            "verify_work",
            command=(
                'cd repo && docker run --rm -v "$(pwd)":/app -w /app '
                "example.test/toolchain:latest go test ./..."
            ),
        )
    )

    assert result.is_error is False
    assert "absolute host filesystem paths" not in result.content


@pytest.mark.asyncio
async def test_remote_shell_tool_blocks_docker_host_mount_absolute_paths(tmp_path: Path) -> None:
    env = StaticStatusEnvironment(tmp_path, statuses=[])
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(
        _call(
            "shell",
            command="docker run --rm -v /Users/leiro/workspace/repo:/app image go test ./...",
        )
    )

    assert result.is_error is True
    assert "absolute host filesystem paths" in result.content


@pytest.mark.asyncio
async def test_remote_shell_tool_treats_head_pipe_preview_as_evidence(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(_call("shell", command="yes | head -n 1"))

    assert result.is_error is False
    assert result.metadata["head_pipe_preview_exit_status"] is True
    assert "preview evidence" in result.content
    assert "stdout:\ny" in result.content


@pytest.mark.asyncio
async def test_remote_shell_tool_guides_clone_into_populated_workspace(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "task.toml").write_text("metadata\n", encoding="utf-8")
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(_call("shell", command="git clone . . || true"))

    assert result.is_error is True
    assert "destination path '.' already exists" in result.content
    assert "cloning the project into a new subdirectory" in result.content


@pytest.mark.asyncio
async def test_remote_shell_tool_guides_shallow_checkout_base_commit_failure(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(
        _call(
            "shell",
            command=(
                "printf 'fatal: unable to read tree "
                "(9d2d84bb1564e9513287998c56ccf16c01c19008)\\n' >&2; exit 128"
            ),
        )
    )

    assert result.is_error is True
    assert "shallow or incomplete git checkout" in result.content
    assert "git fetch --depth 1 origin <commit>" in result.content


@pytest.mark.asyncio
async def test_remote_shell_tool_guides_missing_toolchain_setup(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(_call("shell", command="go test ./..."))

    assert result.is_error is True
    assert "`go` is not installed on PATH" in result.content
    assert "environment/Dockerfile" in result.content
    assert "Docker" in result.content
    assert "command -v docker" in result.content


@pytest.mark.asyncio
async def test_remote_shell_read_only_inspection_is_tracked_without_blocking(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("a\n", encoding="utf-8")
    env = LocalEnvironment(tmp_path)
    state = RemoteWorkspaceState()
    shell = RemoteShellTool(env, workdir=str(tmp_path), state=state)

    first = await shell(_call("shell", command="cat a.txt"))
    second = await shell(_call("shell", command="sed -n '1p' a.txt"))

    assert first.is_error is False
    assert second.is_error is False
    assert "too many read-only inspection commands" not in second.content
    assert any(
        "sed -n" in str(call["command"]) and "a.txt" in str(call["command"]) for call in env.calls
    )
    assert state.read_only_calls_since_change == 2


@pytest.mark.asyncio
async def test_remote_shell_project_verification_commands_are_not_inspection_limited(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    state = RemoteWorkspaceState()
    shell = RemoteShellTool(env, workdir=str(tmp_path), state=state)

    result = await shell(_call("shell", command="go test ./..."))

    assert "too many read-only inspection commands" not in result.content
    assert any("go test ./..." in str(call["command"]) for call in env.calls)


@pytest.mark.asyncio
async def test_remote_shell_tool_marks_git_workspace_mutations(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(_call("shell", command="touch created.txt"))

    assert result.is_error is False
    assert result.metadata["workspace_changed"] is True


@pytest.mark.asyncio
async def test_remote_shell_tool_rejects_repeated_failed_command(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    first = await shell(_call("shell", command="false"))
    second = await shell(_call("shell", command="false"))

    assert first.is_error is True
    assert second.is_error is True
    assert "already failed" in second.content
    assert sum("false" in str(call["command"]) for call in env.calls) == 1


@pytest.mark.asyncio
async def test_remote_shell_setup_unblocks_failed_verification_retry(
    tmp_path: Path,
) -> None:
    class ToolSetupEnvironment:
        def __init__(self) -> None:
            self.ready = False
            self.calls: list[str] = []

        async def exec(
            self,
            command: str,
            *,
            cwd: str | None = None,
            timeout_sec: int | None = None,
        ) -> ExecResult:
            self.calls.append(command)
            if command.startswith("git diff --binary --no-ext-diff"):
                return ExecResult(stdout="constant\n", stderr="", return_code=0)
            if "setup-tooling" in command:
                self.ready = True
                return ExecResult(stdout="setup complete\n", stderr="", return_code=0)
            if "check-tool" in command:
                if self.ready:
                    return ExecResult(stdout="ready\n", stderr="", return_code=0)
                return ExecResult(stdout="", stderr="not ready\n", return_code=1)
            return ExecResult(stdout="", stderr="", return_code=0)

    env = ToolSetupEnvironment()
    state = RemoteWorkspaceState()
    shell = RemoteShellTool(env, workdir=str(tmp_path), state=state)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path), state=state)

    failed = await verify(_call("verify_work", command="check-tool"))
    repeated = await verify(_call("verify_work", command="check-tool"))
    setup = await shell(_call("shell", command="setup-tooling"))
    retried = await verify(_call("verify_work", command="check-tool"))

    assert failed.is_error is True
    assert repeated.is_error is True
    assert "already failed" in repeated.content
    assert setup.is_error is False
    assert retried.is_error is False
    assert retried.content.startswith("PASSED")
    assert sum("check-tool" in command for command in env.calls) == 2


@pytest.mark.asyncio
async def test_shell_decodes_bytes_output_from_environment(tmp_path: Path) -> None:
    class BytesOutputEnvironment:
        async def exec(
            self,
            command: str,
            *,
            cwd: str | None = None,
            timeout_sec: int | None = None,
        ) -> object:
            if command == _GIT_WORKSPACE_FINGERPRINT_COMMAND:
                return ExecResult(stdout="constant\n", stderr="", return_code=0)
            return SimpleNamespace(
                stdout=b"binary stdout\n",
                stderr=b"binary stderr\n",
                return_code=124,
            )

    shell = RemoteShellTool(BytesOutputEnvironment(), workdir=str(tmp_path))

    result = await shell(_call("shell", command="python -m pytest -q"))

    assert result.is_error is True
    assert "binary stdout" in result.content
    assert "binary stderr" in result.content
    assert "b'binary" not in result.content


@pytest.mark.asyncio
async def test_verify_work_decodes_bytes_output_from_environment(tmp_path: Path) -> None:
    class BytesOutputEnvironment:
        async def exec(
            self,
            command: str,
            *,
            cwd: str | None = None,
            timeout_sec: int | None = None,
        ) -> object:
            if command == _GIT_WORKSPACE_FINGERPRINT_COMMAND:
                return ExecResult(stdout="constant\n", stderr="", return_code=0)
            return SimpleNamespace(
                stdout=b"verify stdout\n",
                stderr=b"verify stderr\n",
                return_code=1,
            )

    verify = RemoteVerifyWorkTool(BytesOutputEnvironment(), workdir=str(tmp_path))

    result = await verify(_call("verify_work", command="python -m pytest -q"))

    assert result.is_error is True
    assert "verify stdout" in result.content
    assert "verify stderr" in result.content
    assert "b'verify" not in result.content


@pytest.mark.asyncio
async def test_verify_work_guides_missing_toolchain_setup(
    tmp_path: Path,
) -> None:
    class MissingToolchainEnvironment:
        async def exec(
            self,
            command: str,
            *,
            cwd: str | None = None,
            timeout_sec: int | None = None,
        ) -> ExecResult:
            if command.startswith("git diff --binary --no-ext-diff"):
                return ExecResult(stdout="constant\n", stderr="", return_code=0)
            if "go test ./..." in command:
                return ExecResult(
                    stdout="",
                    stderr="bash: line 1: go: command not found\n",
                    return_code=127,
                )
            return ExecResult(stdout="", stderr="", return_code=0)

    verify = RemoteVerifyWorkTool(
        MissingToolchainEnvironment(),
        workdir=str(tmp_path),
    )

    result = await verify(_call("verify_work", command="go test ./..."))

    assert result.is_error is True
    assert "containerized command" in verify.description
    assert "`go` is not installed on PATH" in result.content
    assert "environment/Dockerfile" in result.content
    assert "Docker" in result.content
    assert "command -v docker" in result.content
    assert "verify_work can run the same read-only containerized command" in result.content
    assert "discovered executable path" in result.content
    assert "instead of retrying the host command" in result.content


@pytest.mark.asyncio
async def test_remote_shell_tool_empty_missing_tool_check_is_actionable(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(_call("shell", command="command -v harness-definitely-missing-tool"))

    assert result.is_error is True
    assert "not found on PATH" in result.content


@pytest.mark.asyncio
async def test_remote_shell_tool_empty_search_failure_is_actionable(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(_call("shell", command="printf abc | grep zzz"))

    assert result.is_error is True
    assert "search returned no matches" in result.content


@pytest.mark.asyncio
async def test_remote_shell_tool_marks_masked_failure_as_error(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(_call("shell", command="false || echo masked"))

    assert result.is_error is True
    assert "exit_code: 0" in result.content
    assert "unreliable result" in result.content
    assert result.metadata["masked_failure_exit_status"] is True


@pytest.mark.asyncio
async def test_remote_shell_tool_allows_masked_environment_probe_with_output(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        "if [ \"$1\" = info ]; then echo 'Server Version: test'; exit 0; fi\n"
        "echo 'Docker version test'\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(
        _call(
            "shell",
            command=(
                'PATH="$PWD/bin:$PATH" command -v docker || true && '
                'PATH="$PWD/bin:$PATH" docker info || true'
            ),
        )
    )

    assert result.is_error is False
    assert "Server Version: test" in result.content
    assert "environment probe" in result.content
    assert result.metadata["masked_failure_exit_status"] is False
    assert result.metadata["informational_environment_probe"] is True


@pytest.mark.asyncio
async def test_remote_shell_tool_does_not_downgrade_masked_missing_probe_without_output(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(
        _call("shell", command="command -v harness-definitely-missing-tool || true")
    )

    assert result.is_error is True
    assert "unreliable result" in result.content
    assert result.metadata["masked_failure_exit_status"] is True
    assert result.metadata["informational_environment_probe"] is False


@pytest.mark.asyncio
async def test_remote_shell_tool_allows_masked_read_only_docker_image_probe(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = image ] && [ "$2" = inspect ]; then '
        "echo 'image metadata'; exit 0; fi\n"
        "echo 'unexpected docker command' >&2\n"
        "exit 2\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(
        _call(
            "shell",
            command='PATH="$PWD/bin:$PATH" docker image inspect example:latest || true',
        )
    )

    assert result.is_error is False
    assert "image metadata" in result.content
    assert result.metadata["masked_failure_exit_status"] is False
    assert result.metadata["informational_environment_probe"] is True


@pytest.mark.asyncio
async def test_remote_shell_tool_does_not_downgrade_masked_docker_prune(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = image ] && [ "$2" = prune ]; then '
        "echo 'deleted image cache'; exit 1; fi\n"
        "echo 'unexpected docker command' >&2\n"
        "exit 2\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(
        _call(
            "shell",
            command='PATH="$PWD/bin:$PATH" docker image prune || true',
        )
    )

    assert result.is_error is True
    assert "deleted image cache" in result.content
    assert "unreliable result" in result.content
    assert result.metadata["masked_failure_exit_status"] is True
    assert result.metadata["informational_environment_probe"] is False


@pytest.mark.asyncio
async def test_remote_shell_tool_marks_zero_exit_stderr_failure_as_error(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(
        _call(
            "shell",
            command="sh -c 'printf \"syntax error near token\\n\" >&2; exit 0'",
        )
    )

    assert result.is_error is True
    assert "exit_code: 0" in result.content
    assert "syntax error near token" in result.content
    assert "unreliable result" in result.content
    assert result.metadata["stderr_failure_exit_status"] is True


@pytest.mark.asyncio
async def test_remote_shell_tool_marks_zero_exit_heredoc_warning_as_error(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(
        _call(
            "shell",
            command=(
                "sh -c 'printf \"script.sh: line 9: warning: here-document at "
                "line 1 delimited by end-of-file (wanted `EOF`)\\n\" >&2; exit 0'"
            ),
        )
    )

    assert result.is_error is True
    assert "here-document" in result.content
    assert "unreliable result" in result.content
    assert result.metadata["stderr_failure_exit_status"] is True


@pytest.mark.asyncio
async def test_remote_shell_tool_marks_zero_exit_test_stdout_failure_as_error(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))
    command = (
        "cat <<'EOF'\n"
        "--- FAIL: TestRunInteractive (0.00s)\n"
        "FAIL\n"
        "FAIL\tgithub.com/mattn/anko\t0.008s\n"
        "ok  \tgithub.com/mattn/anko/vm\t0.152s\n"
        "EOF"
    )

    result = await shell(_call("shell", command="echo go test ./...; " + command))

    assert result.is_error is True
    assert "exit_code: 0" in result.content
    assert "stdout contains test failure output" in result.content
    assert result.metadata["stdout_failure_exit_status"] is True


@pytest.mark.asyncio
async def test_remote_shell_tool_allows_zero_exit_go_output_with_some_no_test_packages(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))
    command = (
        "cat <<'EOF'\n"
        "ok  \tgithub.com/mattn/anko\t0.123s\n"
        "?   \tgithub.com/mattn/anko/parser\t[no test files]\n"
        "ok  \tgithub.com/mattn/anko/vm\t0.152s\n"
        "EOF"
    )

    result = await shell(_call("shell", command="echo go test ./...; " + command))

    assert result.is_error is False
    assert result.metadata["stdout_failure_exit_status"] is False


@pytest.mark.asyncio
async def test_remote_shell_tool_allows_zero_exit_go_output_with_some_no_matching_tests(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))
    command = (
        "cat <<'EOF'\n"
        "ok  \tgithub.com/go-git/go-git/v6\t0.012s\n"
        "ok  \tgithub.com/go-git/go-git/v6/backend/http\t0.003s [no tests to run]\n"
        "?   \tgithub.com/go-git/go-git/v6/internal/pathutil\t[no test files]\n"
        "EOF"
    )

    result = await shell(_call("shell", command="echo go test ./...; " + command))

    assert result.is_error is False
    assert result.metadata["stdout_failure_exit_status"] is False


@pytest.mark.asyncio
async def test_remote_shell_tool_uses_pipefail_for_masked_pipeline(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(_call("shell", command="false | tee output.log"))

    assert result.is_error is True
    assert "exit_code: 1" in result.content
    assert result.metadata["pipefail"] is True


@pytest.mark.asyncio
async def test_remote_shell_tool_detects_content_change_when_status_text_is_same(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "app.txt").write_text("base\n", encoding="utf-8")
    await env.exec("git add app.txt")
    (tmp_path / "app.txt").write_text("base\nmodified before shell\n", encoding="utf-8")
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    result = await shell(_call("shell", command="printf 'modified by shell\\n' >> app.txt"))

    assert result.is_error is False
    assert result.metadata["workspace_changed"] is True
    assert result.metadata["workspace_fingerprint_changed"] is True


@pytest.mark.asyncio
async def test_remote_shell_tool_rejects_repeated_masked_failure(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    shell = RemoteShellTool(env, workdir=str(tmp_path))

    first = await shell(_call("shell", command="false || echo masked"))
    second = await shell(_call("shell", command="false || echo masked"))

    assert first.is_error is True
    assert second.is_error is True
    assert "already failed" in second.content


@pytest.mark.asyncio
async def test_remote_shell_tool_rejects_duplicate_command_until_workspace_changes(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    registry = build_remote_tool_registry(env, workdir=str(tmp_path))
    shell = registry.get("shell")
    writer = registry.get("write_file")

    first = await shell(_call("shell", command="printf ok"))
    duplicate = await shell(_call("shell", command="printf ok"))
    write = await writer(_call("write_file", path="src/change.txt", content="changed"))
    after_change = await shell(_call("shell", command="printf ok"))

    assert first.is_error is False
    assert duplicate.is_error is True
    assert "already ran at the current workspace state" in duplicate.content
    assert write.is_error is False
    assert after_change.is_error is False


@pytest.mark.asyncio
async def test_remote_verify_work_tool_reports_pass_and_failure(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    passed = await verify(_call("verify_work", command="printf ok"))
    failed = await verify(_call("verify_work", command="printf failed"))

    assert passed.is_error is False
    assert passed.content.startswith("PASSED")
    assert failed.is_error is True
    assert failed.content.startswith("FAILED (output reports failure)")


@pytest.mark.asyncio
async def test_remote_verify_work_tool_allows_passing_test_names_with_failure_words(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))
    output = """

  arktypeFastCheck
    number
      ✔ Invalid Bound
      ✔ error handling does not cause failure
    composition
      ✔ if/then/else semantics

  42 passing (120ms)
"""

    result = await verify(_call("verify_work", command="cat <<'EOF'\n" + output + "\nEOF"))

    assert result.is_error is False
    assert result.content.startswith("PASSED")
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is False


@pytest.mark.asyncio
async def test_remote_verify_work_tool_rejects_noop_commands(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    direct = await verify(_call("verify_work", command="true"))
    chained = await verify(_call("verify_work", command="cd . && true"))
    git_diff = await verify(_call("verify_work", command="git diff --name-only"))
    git_status = await verify(_call("verify_work", command="cd . && git status --porcelain"))
    container_wrapped = await verify(
        _call(
            "verify_work",
            command=("docker run --rm -v \"$PWD\":/app -w /app image bash -lc 'cd /app && true'"),
        )
    )

    assert direct.is_error is True
    assert chained.is_error is True
    assert container_wrapped.is_error is True
    assert direct.metadata["reason"] == "noop_verification_command"
    assert chained.metadata["reason"] == "noop_verification_command"
    assert git_diff.metadata["reason"] == "noop_verification_command"
    assert git_status.metadata["reason"] == "noop_verification_command"
    assert container_wrapped.metadata["reason"] == "noop_verification_command"
    assert "meaningful check" in direct.content
    assert "meaningful check" in git_diff.content


@pytest.mark.asyncio
async def test_remote_verify_work_tool_rejects_setup_commands(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    install = await verify(
        _call("verify_work", command="cd repo && python3 -m pip -q install hypothesis")
    )
    compile_check = await verify(_call("verify_work", command="python3 -m compileall -q ."))

    assert install.is_error is True
    assert install.metadata["reason"] == "setup_command_not_verification"
    assert "must use the shell tool first" in install.content
    assert compile_check.is_error is False


@pytest.mark.asyncio
async def test_remote_verify_work_tool_reuses_identical_success_at_same_state(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    first = await verify(_call("verify_work", command="printf ok"))
    call_count_after_first = len(env.calls)
    second = await verify(_call("verify_work", command="printf ok"))

    assert first.is_error is False
    assert second.is_error is False
    assert second.content.startswith("PASSED (cached previous verification)")
    assert second.metadata["command"] == "printf ok"
    assert second.metadata["cached_previous_success"] is True
    assert len(env.calls) == call_count_after_first


@pytest.mark.asyncio
async def test_remote_verify_work_tool_runs_default_after_explicit_same_command(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(
        env,
        workdir=str(tmp_path),
        default_command="printf ok",
    )

    explicit = await verify(_call("verify_work", command="printf ok"))
    call_count_after_explicit = len(env.calls)
    default = await verify(_call("verify_work"))

    assert explicit.is_error is False
    assert default.is_error is False
    assert default.metadata["used_default_command"] is True
    assert default.metadata.get("cached_previous_success") is not True
    assert len(env.calls) > call_count_after_explicit


@pytest.mark.asyncio
async def test_remote_verify_work_tool_uses_configured_default_command(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(
        env,
        workdir=str(tmp_path),
        default_command="printf ok",
    )

    result = await verify(_call("verify_work"))

    assert result.is_error is False
    assert result.content.startswith("PASSED")
    assert result.metadata["command"] == "printf ok"
    assert result.metadata["used_default_command"] is True
    assert verify.parameters_schema["required"] == []


def test_fingerprint_detects_untracked_test_paths() -> None:
    assert _fingerprint_has_untracked_test_path(
        "untracked tests/generated_regression.sh\nmode 755 tests/generated_regression.sh\nabc\n"
    )
    assert not _fingerprint_has_untracked_test_path(
        "untracked README.md\nmode 644 README.md\nabc\n"
    )


@pytest.mark.asyncio
async def test_remote_verify_work_reruns_success_with_untracked_regression_tests(
    tmp_path: Path,
) -> None:
    env = FlakyVerifyEnvironment()
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    result = await verify(_call("verify_work", command="make test"))

    assert result.is_error is True
    assert result.content.startswith("FAILED (immediate rerun exit 2)")
    assert "[immediate rerun]" in result.content
    assert result.metadata["rerun_required"] is True
    assert result.metadata["rerun_exit_code"] == 2
    assert env.verify_runs == 2


@pytest.mark.asyncio
async def test_remote_verify_work_hints_to_trace_empty_shell_script_failure(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    (tmp_path / "tests").mkdir()
    script = tmp_path / "tests" / "run.sh"
    script.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    script.chmod(0o755)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    result = await verify(_call("verify_work", command="./tests/run.sh"))

    assert result.is_error is True
    assert "bash -x ./tests/run.sh" in result.content
    assert "Do not append `; echo $?`" in result.content


@pytest.mark.asyncio
async def test_remote_verify_work_tool_uses_configured_default_timeout(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    registry = build_remote_tool_registry(
        env,
        workdir=str(tmp_path),
        default_verify_command="printf ok",
        default_verify_timeout_seconds=240,
    )
    verify = registry.get("verify_work")

    result = await verify(_call("verify_work"))

    assert result.is_error is False
    assert result.metadata["timeout"] == 240
    assert any(
        call["command"] == "bash -lc 'set -e -o pipefail; printf ok'" and call["timeout_sec"] == 240
        for call in env.calls
    )


@pytest.mark.asyncio
async def test_remote_verify_work_tool_hides_default_verifier_failure_details(
    tmp_path: Path,
) -> None:
    hidden_dir = tmp_path / "hidden-verifier"
    hidden_dir.mkdir()
    hidden_test = hidden_dir / "test_secret_behavior.py"
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(
        env,
        workdir=str(tmp_path),
        default_command=(
            f"printf 'AssertionError at {hidden_test}: expected secret value 42\\n'; exit 1"
        ),
    )

    result = await verify(_call("verify_work"))

    assert result.is_error is True
    assert result.content.startswith("FAILED (exit 1)")
    assert "Configured default verification failed" in result.content
    assert "Configured default verifier evidence is required" in result.content
    assert "does not replace a later passing default verify_work" not in result.content
    assert str(hidden_test) not in result.content
    assert "test_secret_behavior.py" not in result.content
    assert "expected secret value 42" not in result.content
    assert result.metadata["command"]
    assert result.metadata["used_default_command"] is True


@pytest.mark.asyncio
async def test_remote_verify_work_tool_repeated_default_failure_stays_model_safe(
    tmp_path: Path,
) -> None:
    hidden_test = tmp_path / "hidden-verifier" / "test_secret_behavior.py"
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(
        env,
        workdir=str(tmp_path),
        default_command=f"printf 'AssertionError at {hidden_test}\\n'; exit 1",
    )

    first = await verify(_call("verify_work"))
    second = await verify(_call("verify_work"))

    assert first.is_error is True
    assert second.is_error is True
    assert "already failed at this workspace state" in second.content
    assert "Detailed verifier output is withheld" in second.content
    assert "Configured default verifier evidence is required" in second.content
    assert "easier agent-selected command" not in second.content
    assert str(hidden_test) not in second.content
    assert "test_secret_behavior.py" not in second.content
    assert second.metadata["reason"] == "repeated_failed_default_command"
    assert "used_default_command" not in second.metadata


@pytest.mark.asyncio
async def test_remote_verify_work_reruns_after_untracked_executable_bit_changes(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    script = tests_dir / "regression.sh"
    script.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho ok\n", encoding="utf-8")
    os.chmod(script, 0o644)
    registry = build_remote_tool_registry(
        env,
        workdir=str(tmp_path),
        default_verify_command="./tests/regression.sh",
    )
    verify = registry.get("verify_work")
    shell = registry.get("shell")

    first = await verify(_call("verify_work"))
    chmod = await shell(_call("shell", command="chmod +x tests/regression.sh"))
    second = await verify(_call("verify_work"))

    assert first.is_error is True
    assert chmod.is_error is False
    assert chmod.metadata["workspace_fingerprint_changed"] is True
    assert second.is_error is False
    assert second.content.startswith("PASSED")


@pytest.mark.asyncio
async def test_remote_verify_work_tool_rejects_no_tests_output(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    result = await verify(
        _call(
            "verify_work",
            command=("printf 'test session starts\\ncollected 0 items\\nno tests ran in 0.01s\\n'"),
        )
    )

    assert result.is_error is True
    assert result.content.startswith("FAILED (output reports failure)")
    assert result.metadata["output_reports_failure"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "output",
    [
        "testing: warning: no tests to run\\nPASS\\nok  example.com/pkg 0.003s\\n",
        "?   \\texample.com/pkg\\t[no test files]\\n",
        "running 0 tests\\n\\ntest result: ok. 0 passed; 0 failed; 0 ignored\\n",
    ],
)
async def test_remote_verify_work_tool_rejects_common_no_test_outputs(
    tmp_path: Path,
    output: str,
) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    result = await verify(_call("verify_work", command="cat <<'EOF'\n" + output + "\nEOF"))

    assert result.is_error is True
    assert result.content.startswith("FAILED (output reports failure)")
    assert result.metadata["output_reports_failure"] is True


@pytest.mark.asyncio
async def test_remote_verify_work_tool_rejects_successful_heredoc_warning(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))
    command = (
        "printf 'ok\\n'; "
        "printf 'tests/broken.sh: line 65: warning: here-document at line 7 "
        "delimited by end-of-file (wanted `INI`)\\n' >&2"
    )

    result = await verify(_call("verify_work", command=command))

    assert result.is_error is True
    assert result.content.startswith("FAILED (output reports failure)")
    assert "here-document" in result.content
    assert result.metadata["output_reports_failure"] is True


@pytest.mark.asyncio
async def test_remote_verify_work_tool_allows_go_package_sweep_with_some_no_test_packages(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))
    output = (
        "ok  github.com/mattn/anko 0.123s\n"
        "?   \tgithub.com/mattn/anko/ast\t[no test files]\n"
        "ok  github.com/mattn/anko/core 0.456s\n"
        "?   \tgithub.com/mattn/anko/parser\t[no test files]\n"
    )

    result = await verify(_call("verify_work", command=f"printf {output!r}"))

    assert result.is_error is False
    assert result.content.startswith("PASSED")
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is False


@pytest.mark.asyncio
async def test_remote_verify_work_tool_allows_go_sweep_with_some_no_matching_tests(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))
    output = (
        "ok  github.com/go-git/go-git/v6 0.012s\n"
        "ok  github.com/go-git/go-git/v6/backend/http 0.003s [no tests to run]\n"
        "?   \tgithub.com/go-git/go-git/v6/internal/pathutil\t[no test files]\n"
        "ok  github.com/go-git/go-git/v6/config 0.004s [no tests to run]\n"
    )

    result = await verify(_call("verify_work", command=f"printf {output!r}"))

    assert result.is_error is False
    assert result.content.startswith("PASSED")
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is False


@pytest.mark.asyncio
async def test_remote_verify_work_tool_rejects_go_sweep_with_only_no_matching_tests(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))
    output = (
        "ok  github.com/go-git/go-git/v6 0.012s [no tests to run]\n"
        "ok  github.com/go-git/go-git/v6/config 0.004s [no tests to run]\n"
        "?   \tgithub.com/go-git/go-git/v6/internal/pathutil\t[no test files]\n"
    )

    result = await verify(_call("verify_work", command=f"printf {output!r}"))

    assert result.is_error is True
    assert result.content.startswith("FAILED (output reports failure)")
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is True


@pytest.mark.asyncio
async def test_remote_verify_work_tool_rejects_successful_go_output_with_failures(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))
    output = (
        "--- FAIL: TestRunInteractive (0.00s)\n"
        "    anko_test.go:109: OpenFile error\n"
        "FAIL\n"
        "FAIL\tgithub.com/mattn/anko\t0.008s\n"
        "ok  \tgithub.com/mattn/anko/vm\t0.152s\n"
    )

    result = await verify(_call("verify_work", command="cat <<'EOF'\n" + output + "\nEOF"))

    assert result.is_error is True
    assert result.content.startswith("FAILED (output reports failure)")
    assert result.metadata["exit_code"] == 0
    assert result.metadata["output_reports_failure"] is True


@pytest.mark.asyncio
async def test_remote_verify_work_tool_fails_multiline_masked_failure(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    result = await verify(_call("verify_work", command="false\necho masked"))

    assert result.is_error is True
    assert result.metadata["errexit"] is True
    assert result.metadata["exit_code"] != 0


@pytest.mark.asyncio
async def test_remote_verify_work_tool_preserves_rejected_command_metadata(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    result = await verify(_call("verify_work", command="pytest tests/test_app.py || true"))

    assert result.is_error is True
    assert "failed assertions must return a non-zero exit code" in result.content
    assert result.metadata["command"] == "pytest tests/test_app.py || true"
    assert result.metadata["reason"] == "masked_failure_exit_status"


@pytest.mark.asyncio
async def test_remote_verify_work_tool_rejects_exit_before_later_test_command(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    result = await verify(_call("verify_work", command="exit 0; pytest tests/test_app.py"))

    assert result.is_error is True
    assert "exits before a later check can run" in result.content
    assert result.metadata["command"] == "exit 0; pytest tests/test_app.py"
    assert result.metadata["reason"] == "unreachable_verification_command"


@pytest.mark.asyncio
async def test_remote_verify_work_tool_rejects_suppressed_failure_exit_zero(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    result = await verify(
        _call(
            "verify_work",
            command="pytest tests/test_app.py >/dev/null 2>&1 || exit 0",
        )
    )

    assert result.is_error is True
    assert "failed assertions must return" in result.content
    assert result.metadata["command"] == "pytest tests/test_app.py >/dev/null 2>&1 || exit 0"
    assert result.metadata["reason"] == "masked_failure_exit_status"


@pytest.mark.asyncio
async def test_remote_verify_work_tool_rejects_workspace_mutation(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text(
        "def test_ok(): pass\n",
        encoding="utf-8",
    )
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    result = await verify(
        _call(
            "verify_work",
            command="pytest tests/test_ok.py && mkdir -p src && touch src/after_verify.py",
        )
    )

    assert result.is_error is True
    assert result.content.startswith("FAILED (verification changed workspace)")
    assert result.metadata["workspace_changed"] is True
    assert (tmp_path / "src" / "after_verify.py").is_file()


@pytest.mark.asyncio
async def test_remote_verify_work_tool_detects_mutation_when_status_text_is_same(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "app.txt").write_text("base\n", encoding="utf-8")
    await env.exec("git add app.txt")
    (tmp_path / "app.txt").write_text("base\nmodified before verify\n", encoding="utf-8")
    verify = RemoteVerifyWorkTool(env, workdir=str(tmp_path))

    result = await verify(_call("verify_work", command="printf 'modified by verify\\n' >> app.txt"))

    assert result.is_error is True
    assert result.content.startswith("FAILED (verification changed workspace)")
    assert result.metadata["workspace_changed"] is True
    assert result.metadata["workspace_fingerprint_changed"] is True


def test_build_remote_tool_registry_contains_expected_tools(tmp_path: Path) -> None:
    registry = build_remote_tool_registry(LocalEnvironment(tmp_path), workdir=str(tmp_path))

    assert registry.names() == [
        "apply_patch",
        "edit_file",
        "fetch_url",
        "list_dir",
        "read_file",
        "read_file_range",
        "shell",
        "verify_work",
        "web_search",
        "write_file",
    ]
    assert registry.get("web_search").approval == "auto"
    assert registry.get("web_search").effect_scope == "read_only"
    assert registry.get("fetch_url").approval == "auto"
    assert registry.get("fetch_url").effect_scope == "read_only"


@pytest.mark.asyncio
async def test_build_remote_tool_registry_applies_external_policy(tmp_path: Path) -> None:
    env = LocalEnvironment(tmp_path)
    registry = build_remote_tool_registry(
        env,
        workdir=str(tmp_path),
        policy=_restricted_benchmark_policy(),
    )
    shell = registry.get("shell")
    assert registry.has("web_search")
    assert registry.has("fetch_url")

    result = await shell(_call("shell", command="cat /solution/solve.sh"))

    assert result.is_error is True
    assert "benchmark-private artifacts" in result.content
    assert not env.calls


def test_workspace_source_change_status_separates_scratch_from_source() -> None:
    passed, source_paths, scratch_paths = _workspace_source_change_status("?? test_defaults.ank\n")
    assert passed is False
    assert source_paths == []
    assert scratch_paths == ["test_defaults.ank"]

    passed, source_paths, scratch_paths = _workspace_source_change_status("?? repo/.gitkeep\n")
    assert passed is False
    assert source_paths == []
    assert scratch_paths == ["repo/.gitkeep"]

    passed, source_paths, scratch_paths = _workspace_source_change_status(" M config.go\n")
    assert passed is True
    assert source_paths == ["config.go"]
    assert scratch_paths == []

    passed, source_paths, scratch_paths = _workspace_source_change_status("?? config.go\n")
    assert passed is False
    assert source_paths == []
    assert scratch_paths == ["config.go"]

    passed, source_paths, scratch_paths = _workspace_source_change_status(
        "?? cattrs_repo/\n?? scratch/\n"
    )
    assert passed is True
    assert source_paths == ["cattrs_repo/"]
    assert scratch_paths == ["scratch/"]

    passed, source_paths, scratch_paths = _workspace_source_change_status(
        "?? .harness-home/\n?? src/new_feature.py\n"
    )
    assert passed is True
    assert source_paths == ["src/new_feature.py"]
    assert scratch_paths == []

    passed, source_paths, scratch_paths = _workspace_source_change_status(
        "?? =1.0\n?? src/new_feature.py\n",
        baseline_untracked_paths={"=1.0"},
    )
    assert passed is True
    assert source_paths == ["src/new_feature.py"]
    assert scratch_paths == []

    passed, source_paths, scratch_paths = _workspace_source_change_status(
        "?? rule_action_pinning.go\n?? scratch.py\n",
        tracked_paths={"config.go", "project.yaml"},
    )
    assert passed is True
    assert source_paths == ["rule_action_pinning.go"]
    assert scratch_paths == ["scratch.py"]

    passed, source_paths, scratch_paths = _workspace_source_change_status(
        "?? IMPLEMENTATION_PLAN.md\n?? parser_changes.patch\n?? docs/notes.rst\n",
        tracked_paths={"README.md", "src/app.py"},
    )
    assert passed is False
    assert source_paths == []
    assert scratch_paths == [
        "IMPLEMENTATION_PLAN.md",
        "parser_changes.patch",
        "docs/notes.rst",
    ]

    passed, source_paths, scratch_paths = _workspace_source_change_status(
        " M ast/expr.go\n?? parser/validation.go\n"
    )
    assert passed is True
    assert source_paths == ["ast/expr.go", "parser/validation.go"]
    assert scratch_paths == []

    passed, source_paths, scratch_paths = _workspace_source_change_status(
        " M ast/stmt.go\n?? ast/stmt.go_backup\n?? ast/stmt.go.bak\n"
    )
    assert passed is True
    assert source_paths == ["ast/stmt.go"]
    assert scratch_paths == ["ast/stmt.go_backup", "ast/stmt.go.bak"]

    passed, source_paths, scratch_paths = _workspace_source_change_status(
        " M tests/models/test_cookie_store.py\n?? tests/models/new_case.py\n"
    )
    assert passed is False
    assert source_paths == []
    assert scratch_paths == []

    assert _workspace_test_change_paths(
        "?? tests/existing_test.py\n?? tests/new_test.py\n",
        baseline_untracked_paths={"tests/existing_test.py"},
    ) == ["tests/new_test.py"]
    assert _workspace_test_change_paths("?? ark/json-schema/__tests__/helper.ts\n") == [
        "ark/json-schema/__tests__/helper.ts"
    ]
    assert _workspace_test_change_paths(
        "?? tests/.runner_cache/generated.bin\n?? tests/test_app.js\n"
    ) == ["tests/test_app.js"]
    assert _workspace_test_change_paths(" D tests/test_removed.py\n") == []
    assert _workspace_test_change_paths("R  tests/test_old.py -> tests/test_new.py\n") == []
    assert _workspace_deleted_source_paths(
        " D src/app.py\n D tests/test_removed.py\n M src/other.py\n?? scratch.py\n",
        tracked_paths={"src/app.py", "tests/test_removed.py", "src/other.py"},
    ) == ["src/app.py"]
    passed, source_paths, scratch_paths = _workspace_source_change_status(
        " M src/app.py\n D tests/test_removed.py\n"
    )
    assert passed is True
    assert source_paths == ["src/app.py"]
    assert scratch_paths == ["tests/test_removed.py"]
    passed, source_paths, scratch_paths = _workspace_source_change_status(
        " M src/app.py\nR  tests/test_old.py -> tests/test_new.py\n"
    )
    assert passed is True
    assert source_paths == ["src/app.py"]
    assert scratch_paths == ["tests/test_new.py"]
    passed, source_paths, scratch_paths = _workspace_source_change_status(
        "?? src/.build_cache/generated.bin\n?? src/app.ts\n"
    )
    assert passed is True
    assert source_paths == ["src/app.ts"]
    assert scratch_paths == []

    passed, source_paths, scratch_paths = _workspace_source_change_status(
        "?? repro_sac.py\n?? test_sac_logic.py\n?? t/unit/test_sac_logic.py\n",
        tracked_paths={
            "setup.py",
            "kombu/transport/virtual/base.py",
            "t/unit/transport/virtual/test_base.py",
        },
    )
    assert passed is False
    assert source_paths == []
    assert scratch_paths == ["repro_sac.py", "test_sac_logic.py"]
    assert _workspace_test_change_paths(
        "?? repro_sac.py\n?? test_sac_logic.py\n?? t/unit/test_sac_logic.py\n",
        tracked_paths={
            "setup.py",
            "kombu/transport/virtual/base.py",
            "t/unit/transport/virtual/test_base.py",
        },
    ) == ["t/unit/test_sac_logic.py"]

    assert _workspace_test_change_paths(
        "?? test_root_feature.case\n",
        tracked_paths={"test_existing.case", "setup.cfg"},
    ) == ["test_root_feature.case"]

    language_named_status = (
        " M src/app.py\n"
        "?? pkg/test_default_args.py\n"
        "?? vm/default_args_test.go\n"
        "?? web/urlJoin.regression.test.ts\n"
    )
    language_named_tracked_paths = {
        "src/app.py",
        "tests/test_app.py",
        "vm/vm.go",
        "web/urlJoin.ts",
    }
    passed, source_paths, scratch_paths = _workspace_source_change_status(
        language_named_status,
        tracked_paths=language_named_tracked_paths,
    )
    assert passed is True
    assert source_paths == ["src/app.py"]
    assert scratch_paths == [
        "pkg/test_default_args.py",
        "vm/default_args_test.go",
        "web/urlJoin.regression.test.ts",
    ]
    assert (
        _workspace_test_change_paths(
            language_named_status,
            tracked_paths=language_named_tracked_paths,
        )
        == []
    )

    package_local_test_status = " M anko/ast/expr.go\n?? vm/default_args_test.go\n"
    package_local_tracked_paths = {
        "anko/ast/expr.go",
        "anko_test.go",
        "vm/vm.go",
    }
    passed, source_paths, scratch_paths = _workspace_source_change_status(
        package_local_test_status,
        tracked_paths=package_local_tracked_paths,
    )
    assert passed is True
    assert source_paths == ["anko/ast/expr.go"]
    assert scratch_paths == ["vm/default_args_test.go"]
    assert (
        _workspace_test_change_paths(
            package_local_test_status,
            tracked_paths=package_local_tracked_paths,
        )
        == []
    )


def test_tool_result_counts_root_overwrite_as_possible_source_change() -> None:
    assert _tool_result_counts_as_source_change(
        ToolResult(
            tool_call_id="edit",
            name="edit_file",
            content="edited config.go",
            metadata={"path": "config.go", "overwrite": True},
        )
    )
    assert not _tool_result_counts_as_source_change(
        ToolResult(
            tool_call_id="write",
            name="write_file",
            content="wrote scratch.go",
            metadata={"path": "scratch.go", "overwrite": False},
        )
    )
    assert not _tool_result_counts_as_source_change(
        ToolResult(
            tool_call_id="write",
            name="write_file",
            content="wrote ast/stmt.go_backup",
            metadata={"path": "ast/stmt.go_backup", "overwrite": False},
        )
    )


def _completed(
    name: str,
    *,
    is_error: bool = False,
    metadata: dict[str, object] | None = None,
    arguments: dict[str, object] | None = None,
    content_preview: str = "",
) -> ActivityEvent:
    return ActivityEvent(
        kind="tool_call.completed",
        data={
            "tool_call_id": f"call_{name}",
            "name": name,
            "is_error": is_error,
            "content_preview": content_preview,
            "content_size": len(content_preview),
            "arguments": arguments or {},
            "metadata": metadata or {},
        },
    )


def _scripted_tool_turn(call_id: str, name: str, arguments: dict[str, object]) -> list[Event]:
    call = ToolCall(id=call_id, name=name, arguments=arguments)
    return [
        ToolCallEvent(call=call),
        Done(final_message=Message(role="assistant", content=None, tool_calls=[call])),
    ]


def _planner_turn() -> list[Event]:
    return [
        Done(
            final_message=Message(
                role="assistant",
                content='{"steps":[{"description":"inspect, edit, and verify"}]}',
            )
        )
    ]


class ScriptedAdapter:
    name = "openrouter"

    def __init__(self, scripts: list[list[Event]]) -> None:
        self.scripts = scripts

    def stream(self, **_kwargs: object):
        async def _events():
            if not self.scripts:
                raise AssertionError("adapter called too many times")
            for event in self.scripts.pop(0):
                yield event

        return _events()

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=True)

    async def cancel(self, _session_id: str) -> None:
        return None


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_no_source_change(tmp_path: Path) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status="?? test_defaults.ank\n",
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(session=SimpleNamespace(), activity=[])

    assert result.can_finish is False
    assert verifier.latest.source_change_passed is False
    assert verifier.latest.scratch_paths == ["test_defaults.ank"]
    assert "scratch-only" in result.reason
    assert "inspect task metadata, environment files, project setup files" in result.reason
    assert "clone or create the patchable project in a subdirectory" in result.reason
    assert "metadata wrapper `.git` checkout" in result.reason
    assert "source repository URL, a base commit, a container image" in result.reason
    assert "Do not ask the user to provide repository files" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_source_tree_handoff(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(tmp_path, statuses=[], baseline_status="")
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))
    session = SimpleNamespace(
        messages=[
            Message(role="user", content="Implement the requested behavior."),
            Message(
                role="assistant",
                content=(
                    "There is no source code checked out here. Please upload the "
                    "repository contents into repo/ or tell me where the source tree "
                    "is located, then I can patch it."
                ),
            ),
        ]
    )

    result = await verifier.verify(session=session, activity=[])

    assert result.can_finish is False
    assert "defers repository, source, environment, or dependency setup" in result.reason
    assert "Run available setup, clone, install, build" in result.reason
    assert "check Docker availability" in result.reason
    assert "instead of asking the user to provide source files or tooling" in result.reason
    assert verifier.latest.source_change_passed is False


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_placeholder_repo_source_handoff(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(tmp_path, statuses=[], baseline_status="?? repo/\n")
    verifier = ExternalWorkspaceVerifier(
        env,
        workdir=str(tmp_path),
        require_regression_test_change=False,
    )
    session = SimpleNamespace(
        messages=[
            Message(role="user", content="Implement the requested behavior."),
            Message(
                role="assistant",
                content=(
                    "I created a repo/ placeholder, but the project files are missing. "
                    "Please provide or checkout the repository source code so I can continue."
                ),
            ),
        ]
    )

    result = await verifier.verify(
        session=session,
        activity=[_completed("write_file", metadata={"path": "repo/.gitkeep"})],
    )

    assert result.can_finish is False
    assert "defers repository, source, environment, or dependency setup" in result.reason
    assert "instead of asking the user to provide source files or tooling" in result.reason
    assert verifier.latest.source_change_passed is True
    assert verifier.latest.verification_passed_after_source_change is False


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_wrong_base_nested_checkout(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "task.toml").write_text("task = 'sample'\n", encoding="utf-8")
    await env.exec("git add task.toml")
    await env.exec("git -c user.name=test -c user.email=test@example.com commit -m wrapper")

    repo = tmp_path / "repo"
    repo.mkdir()
    await env.exec("git init", cwd=str(repo))
    (repo / "app.py").write_text("VALUE = 'base'\n", encoding="utf-8")
    await env.exec("git add app.py", cwd=str(repo))
    await env.exec(
        "git -c user.name=test -c user.email=test@example.com commit -m base", cwd=str(repo)
    )
    base = (await env.exec("git rev-parse HEAD", cwd=str(repo))).stdout.strip()
    (repo / "app.py").write_text("VALUE = 'upstream'\n", encoding="utf-8")
    await env.exec("git add app.py", cwd=str(repo))
    await env.exec(
        "git -c user.name=test -c user.email=test@example.com commit -m upstream",
        cwd=str(repo),
    )
    wrong_head = (await env.exec("git rev-parse HEAD", cwd=str(repo))).stdout.strip()
    (repo / "app.py").write_text("VALUE = 'agent-fix'\n", encoding="utf-8")

    verifier = ExternalWorkspaceVerifier(
        env,
        workdir=str(tmp_path),
        require_regression_test_change=False,
        policy=ExternalWorkspacePolicy(required_git_base_commit=base),
    )

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "repo/app.py"}),
            _completed("verify_work", metadata={"command": "pytest", "exit_code": 0}),
        ],
    )

    assert result.can_finish is False
    assert "base commit" in result.reason
    assert base in result.reason
    assert wrong_head[:12] in result.reason
    assert "repo" in result.reason
    assert verifier.latest.source_change_passed is True
    assert verifier.latest.verification_passed_after_source_change is False


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_parent_staged_clean_nested_checkout(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "task.toml").write_text("task = 'sample'\n", encoding="utf-8")
    await env.exec("git add task.toml")
    await env.exec("git -c user.name=test -c user.email=test@example.com commit -m wrapper")

    repo = tmp_path / "repo"
    repo.mkdir()
    await env.exec("git init", cwd=str(repo))
    (repo / "app.py").write_text("VALUE = 'base'\n", encoding="utf-8")
    await env.exec("git add app.py", cwd=str(repo))
    await env.exec(
        "git -c user.name=test -c user.email=test@example.com commit -m base",
        cwd=str(repo),
    )
    base = (await env.exec("git rev-parse HEAD", cwd=str(repo))).stdout.strip()
    await env.exec("git add repo")

    verifier = ExternalWorkspaceVerifier(
        env,
        workdir=str(tmp_path),
        require_regression_test_change=False,
        policy=ExternalWorkspacePolicy(required_git_base_commit=base),
    )

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("shell", metadata={"command": "git add repo", "exit_code": 0}),
            _completed(
                "verify_work",
                metadata={"command": "cd repo && python -m py_compile app.py", "exit_code": 0},
            ),
        ],
    )

    assert result.can_finish is False
    assert "did not make an implementation/source change" in result.reason
    assert "nested target git checkout already exists" in result.reason
    assert "repo" in result.reason
    assert verifier.latest.source_change_passed is False
    assert verifier.latest.verification_passed_after_source_change is False


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_permission_to_continue_handoff(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(tmp_path, statuses=[], baseline_status="")
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))
    session = SimpleNamespace(
        messages=[
            Message(role="user", content="Implement the requested behavior."),
            Message(
                role="assistant",
                content=(
                    "I verified Docker is available. Unless you object, I'll now "
                    "start implementing the code changes and verify them."
                ),
            ),
        ]
    )

    result = await verifier.verify(session=session, activity=[])

    assert result.can_finish is False
    assert "asks the user for permission or confirmation to continue" in result.reason
    assert "Continue autonomously from the current evidence" in result.reason
    assert "Do not ask the user to approve the next" in result.reason
    assert "defers repository, source, environment, or dependency setup" not in result.reason


@pytest.mark.asyncio
async def test_external_workspace_verifier_requires_regression_test_change(tmp_path: Path) -> None:
    env = StaticStatusEnvironment(tmp_path, statuses=[], baseline_status=" M ast/expr.go\n")
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[_completed("write_file", metadata={"path": "ast/expr.go"})],
    )

    assert result.can_finish is False
    assert verifier.latest.source_change_passed is True
    assert verifier.latest.test_change_paths == []
    assert "no in-repository regression test changes" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_test_only_change(tmp_path: Path) -> None:
    env = StaticStatusEnvironment(tmp_path, statuses=[], baseline_status=" M tests/test_expr.py\n")
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "tests/test_expr.py"}),
            _completed("verify_work", metadata={"command": "pytest tests/test_expr.py"}),
        ],
    )

    assert result.can_finish is False
    assert verifier.latest.source_change_passed is False
    assert verifier.latest.source_change_paths == []
    assert verifier.latest.test_change_paths == ["tests/test_expr.py"]
    assert "only in-repository test or fixture changes were detected" in result.reason
    assert "implement the source behavior they exercise" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_leftover_scratch_after_source_change(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M ast/expr.go\n M tests/test_expr.py\n?? test.ini\n",
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "ast/expr.go"}),
            _completed("write_file", metadata={"path": "tests/test_expr.py"}),
            _completed("verify_work", metadata={"command": "pytest tests/test_expr.py"}),
        ],
    )

    assert result.can_finish is False
    assert verifier.latest.source_change_passed is True
    assert verifier.latest.scratch_paths == ["test.ini"]
    assert "scratch/non-source artifacts" in result.reason
    assert "test.ini" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_deleted_test_as_regression_change(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M ast/expr.go\n D tests/test_expr.py\n",
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "ast/expr.go"}),
            _completed("shell", metadata={"workspace_changed": True, "exit_code": 0}),
            _completed("verify_work", metadata={"command": "pytest tests"}),
        ],
    )

    assert result.can_finish is False
    assert verifier.latest.source_change_paths == ["ast/expr.go"]
    assert verifier.latest.test_change_paths == []
    assert "no in-repository regression test changes" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_deleted_source_file(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" D parser/parser.go.y\n M parser/parser.go\n M parser/parser_test.go\n",
        tracked_paths={
            "parser/parser.go.y",
            "parser/parser.go",
            "parser/parser_test.go",
        },
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("edit_file", metadata={"path": "parser/parser.go"}),
            _completed("write_file", metadata={"path": "parser/parser_test.go"}),
            _completed("verify_work", metadata={"command": "go test ./..."}),
        ],
    )

    assert result.can_finish is False
    assert verifier.latest.source_change_paths == ["parser/parser.go.y", "parser/parser.go"]
    assert verifier.latest.deleted_source_paths == ["parser/parser.go.y"]
    assert "Tracked source files were deleted" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_verifier_allows_deleted_source_when_policy_allows(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" D src/legacy.py\n M src/app.py\n M tests/test_app.py\n",
        tracked_paths={"src/legacy.py", "src/app.py", "tests/test_app.py"},
    )
    verifier = ExternalWorkspaceVerifier(
        env,
        workdir=str(tmp_path),
        policy=ExternalWorkspacePolicy(allow_tracked_source_deletions=True),
    )

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("shell", metadata={"workspace_changed": True, "exit_code": 0}),
            _completed("write_file", metadata={"path": "tests/test_app.py"}),
            _completed("verify_work", metadata={"command": "pytest tests/test_app.py"}),
        ],
    )

    assert result.can_finish is True
    assert verifier.latest.deleted_source_paths == ["src/legacy.py"]


@pytest.mark.asyncio
async def test_external_workspace_verifier_accepts_source_test_and_later_verify(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M ast/expr.go\n M tests/test_expr.py\n",
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "ast/expr.go"}),
            _completed("write_file", metadata={"path": "tests/test_expr.py"}),
            _completed("verify_work", metadata={"command": "pytest tests/test_expr.py"}),
        ],
    )

    assert result.can_finish is True
    assert verifier.latest.source_change_paths == ["ast/expr.go"]
    assert verifier.latest.test_change_paths == ["tests/test_expr.py"]
    assert verifier.latest.verification_passed_after_source_change is True


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_language_specific_test_parent_inference(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=(
            " M src/app.py\n?? pkg/test_default_args.py\n?? vm/default_args_test.go\n"
        ),
        tracked_paths={
            "src/app.py",
            "tests/test_app.py",
            "vm/vm.go",
        },
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "src/app.py"}),
            _completed("write_file", metadata={"path": "pkg/test_default_args.py"}),
            _completed("write_file", metadata={"path": "vm/default_args_test.go"}),
            _completed("verify_work", metadata={"command": "pytest tests && go test ./..."}),
        ],
    )

    assert result.can_finish is False
    assert verifier.latest.source_change_paths == ["src/app.py"]
    assert verifier.latest.test_change_paths == []
    assert verifier.latest.scratch_paths == [
        "pkg/test_default_args.py",
        "vm/default_args_test.go",
    ]
    assert "no in-repository regression test changes" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_verifier_accepts_nested_docker_project_verify(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=(
            " M vendor/aiomonitor/aiomonitor/monitor.py\n"
            " M vendor/aiomonitor/tests/test_monitor.py\n"
        ),
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "vendor/aiomonitor/aiomonitor/monitor.py"}),
            _completed("write_file", metadata={"path": "vendor/aiomonitor/tests/test_monitor.py"}),
            _completed(
                "verify_work",
                metadata={
                    "command": (
                        'cd vendor/aiomonitor && docker run --rm -v "$PWD":/app '
                        "-w /app example.test/toolchain:latest bash -lc "
                        '"pip install -e . && pytest -q"'
                    )
                },
            ),
        ],
    )

    assert result.can_finish is True
    assert verifier.latest.source_change_paths == ["vendor/aiomonitor/aiomonitor/monitor.py"]
    assert verifier.latest.test_change_paths == ["vendor/aiomonitor/tests/test_monitor.py"]
    assert verifier.latest.verification_passed_after_source_change is True


@pytest.mark.asyncio
async def test_external_workspace_verifier_accepts_shell_test_after_failed_verify(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M bin/ini_get\n M tests/run.sh\n?? tests/test_ini_get.sh\n",
        tracked_paths={"Makefile", "bin/ini_get", "tests/run.sh"},
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "bin/ini_get"}),
            _completed("write_file", metadata={"path": "tests/run.sh"}),
            _completed("write_file", metadata={"path": "tests/test_ini_get.sh"}),
            _completed(
                "verify_work",
                is_error=True,
                metadata={"command": "make test", "exit_code": 2},
                content_preview="FAILED (exit 2)",
            ),
            _completed(
                "shell",
                arguments={"command": "make test"},
                metadata={"exit_code": 0, "workspace_changed": False},
                content_preview="exit_code: 0\n\nstdout:\nok\n",
            ),
        ],
    )

    assert result.can_finish is True
    assert verifier.latest.latest_verification_command == "make test"
    assert verifier.latest.verification_passed_after_source_change is True


@pytest.mark.asyncio
async def test_external_workspace_verifier_accepts_shell_test_without_regression_requirement(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M src/app.go\n",
    )
    verifier = ExternalWorkspaceVerifier(
        env,
        workdir=str(tmp_path),
        require_regression_test_change=False,
    )

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "src/app.go"}),
            _completed(
                "shell",
                arguments={"command": "go test ./..."},
                metadata={"exit_code": 0, "workspace_changed": False},
                content_preview="exit_code: 0\n\nstdout:\nok example.test/app\n",
            ),
        ],
    )

    assert result.can_finish is True
    assert verifier.latest.latest_verification_command == "go test ./..."
    assert verifier.latest.verification_passed_after_source_change is True


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_later_failed_shell_test_after_pass(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M ast/expr.go\n M tests/test_expr.py\n",
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "ast/expr.go"}),
            _completed("write_file", metadata={"path": "tests/test_expr.py"}),
            _completed("verify_work", metadata={"command": "pytest tests/test_expr.py"}),
            _completed(
                "shell",
                is_error=True,
                arguments={"command": "pytest tests/test_expr.py"},
                metadata={"exit_code": 1, "workspace_changed": False},
                content_preview="exit_code: 1\n\nstdout:\n1 failed\n",
            ),
        ],
    )

    assert result.can_finish is False
    assert "latest verify_work after the final workspace mutation did not pass" in result.reason
    assert verifier.latest.latest_verification_error == "exit_code: 1\n\nstdout:\n1 failed\n"
    assert verifier.latest.verification_passed_after_source_change is False


@pytest.mark.asyncio
async def test_external_workspace_verifier_reports_latest_failed_coverage_after_recovered_verify(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M ast/expr.go\n",
    )
    verifier = ExternalWorkspaceVerifier(
        env,
        workdir=str(tmp_path),
        require_regression_test_change=False,
    )

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "ast/expr.go"}),
            _completed(
                "verify_work",
                is_error=True,
                metadata={"command": "go test ./...", "exit_code": 127},
                content_preview="FAILED (exit 127)\n\nstderr:\nbash: go: command not found\n",
            ),
            _completed(
                "verify_work",
                metadata={"command": "docker run image sh -lc 'go test ./ast ./parser'"},
                content_preview="PASSED\n\nstdout:\nok ast\nok parser\n",
            ),
            _completed(
                "shell",
                is_error=True,
                arguments={"command": "go test ./..."},
                metadata={"exit_code": 1, "workspace_changed": False},
                content_preview="exit_code: 1\n\nstdout:\nFAIL evaluator\n",
            ),
        ],
    )

    assert result.can_finish is False
    assert "latest verify_work after the final workspace mutation did not pass" in result.reason
    assert verifier.latest.latest_verification_command == "go test ./..."
    assert verifier.latest.latest_verification_error == "exit_code: 1\n\nstdout:\nFAIL evaluator\n"
    assert "go: command not found" not in result.reason
    assert verifier.latest.verification_passed_after_source_change is False


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_shell_read_as_verification(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M bin/ini_get\n M tests/test_ini_get.sh\n",
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "bin/ini_get"}),
            _completed("write_file", metadata={"path": "tests/test_ini_get.sh"}),
            _completed(
                "shell",
                arguments={"command": "cat tests/test_ini_get.sh"},
                metadata={"exit_code": 0, "workspace_changed": False},
                content_preview="exit_code: 0\n\nstdout:\n#!/usr/bin/env bash\n",
            ),
        ],
    )

    assert result.can_finish is False
    assert "did not produce a later passing" in result.reason
    assert verifier.latest.verification_passed_after_source_change is False


@pytest.mark.asyncio
async def test_external_workspace_verifier_reads_nested_checkout_changes(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    repo = tmp_path / "repo"
    (repo / "mashumaro").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "mashumaro" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "tests" / "test_helper.py").write_text(
        "def test_existing():\n    assert True\n",
        encoding="utf-8",
    )
    await env.exec("git init")
    await env.exec("git init", cwd=str(repo))
    await env.exec("git add mashumaro/helper.py tests/test_helper.py", cwd=str(repo))
    await env.exec(
        "git -c user.name=test -c user.email=test@example.com commit -m baseline",
        cwd=str(repo),
    )
    (repo / "mashumaro" / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    (repo / "tests" / "test_helper.py").write_text(
        "def test_existing():\n    assert True\n\ndef test_regression():\n    assert True\n",
        encoding="utf-8",
    )

    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))
    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "repo/mashumaro/helper.py"}),
            _completed("write_file", metadata={"path": "repo/tests/test_helper.py"}),
            _completed(
                "verify_work",
                metadata={"command": "cd repo && python -m pytest tests/test_helper.py"},
            ),
        ],
    )

    assert result.can_finish is True
    assert verifier.latest.source_change_paths == ["repo/mashumaro/helper.py"]
    assert verifier.latest.test_change_paths == ["repo/tests/test_helper.py"]
    assert verifier.latest.verification_passed_after_source_change is True


@pytest.mark.asyncio
async def test_external_workspace_verifier_reads_committed_nested_checkout_changes(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "task.toml").write_text("task = 'sample'\n", encoding="utf-8")
    await env.exec("git add task.toml")
    await env.exec("git -c user.name=test -c user.email=test@example.com commit -m wrapper")

    repo = tmp_path / "repo"
    (repo / "mashumaro").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "mashumaro" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "tests" / "test_helper.py").write_text(
        "def test_existing():\n    assert True\n",
        encoding="utf-8",
    )
    await env.exec("git init", cwd=str(repo))
    await env.exec("git add mashumaro/helper.py tests/test_helper.py", cwd=str(repo))
    await env.exec(
        "git -c user.name=test -c user.email=test@example.com commit -m baseline",
        cwd=str(repo),
    )
    base = (await env.exec("git rev-parse HEAD", cwd=str(repo))).stdout.strip()

    (repo / "mashumaro" / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
    (repo / "tests" / "test_helper.py").write_text(
        "def test_existing():\n    assert True\n\ndef test_regression():\n    assert True\n",
        encoding="utf-8",
    )
    await env.exec("git add mashumaro/helper.py tests/test_helper.py", cwd=str(repo))
    await env.exec(
        "git -c user.name=test -c user.email=test@example.com commit -m agent-fix",
        cwd=str(repo),
    )

    verifier = ExternalWorkspaceVerifier(
        env,
        workdir=str(tmp_path),
        policy=ExternalWorkspacePolicy(required_git_base_commit=base),
    )
    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("shell", metadata={"command": "cd repo && git commit -m agent-fix"}),
            _completed(
                "verify_work",
                metadata={"command": "cd repo && python -m pytest tests/test_helper.py"},
            ),
        ],
    )

    assert result.can_finish is True
    assert verifier.latest.source_change_paths == ["repo/mashumaro/helper.py"]
    assert verifier.latest.test_change_paths == ["repo/tests/test_helper.py"]
    assert verifier.latest.verification_passed_after_source_change is True


async def _accepted_slug_workspace(
    tmp_path: Path,
    *,
    test_body: str | None = None,
) -> tuple[LocalEnvironment, ExternalWorkspaceVerifier]:
    env = LocalEnvironment(tmp_path)
    (tmp_path / "tests").mkdir()
    (tmp_path / "slugify.py").write_text(
        "def slugify(value):\n    return str(value).lower().replace(' ', '-')\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_slugify.py").write_text(
        "from slugify import slugify\n\n"
        "def test_basic_slugify():\n    assert slugify('Hello World') == 'hello-world'\n",
        encoding="utf-8",
    )
    await env.exec("git init")
    await env.exec("git add slugify.py tests/test_slugify.py")
    await env.exec("git -c user.name=test -c user.email=test@example.com commit -m baseline")
    (tmp_path / "slugify.py").write_text(
        "import re\nimport unicodedata\n\n"
        "_SEPARATOR_RE = re.compile(r'[-_\\s/\\\\]+')\n\n"
        "def slugify(value):\n"
        "    text = unicodedata.normalize('NFKD', str(value)).encode('ascii', 'ignore').decode('ascii')\n"
        "    text = _SEPARATOR_RE.sub('-', text.lower()).strip('-')\n"
        "    return text or 'untitled'\n",
        encoding="utf-8",
    )
    default_test_body = (
        "from slugify import slugify\n\n"
        "def test_basic_slugify():\n    assert slugify('Hello World') == 'hello-world'\n\n"
        "def test_regression():\n"
        "    assert slugify('Crème Brûlée') == 'creme-brulee'\n"
        "    assert slugify('---') == 'untitled'\n"
    )
    (tmp_path / "tests" / "test_slugify.py").write_text(
        test_body if test_body is not None else default_test_body,
        encoding="utf-8",
    )
    structural = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))
    result = await structural.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "slugify.py"}),
            _completed("write_file", metadata={"path": "tests/test_slugify.py"}),
            _completed("verify_work", metadata={"command": "project-test tests/test_slugify.py"}),
        ],
    )
    assert result.can_finish is True
    return env, structural


async def _accepted_release_metadata_workspace(
    tmp_path: Path,
    *,
    release_value: str,
    normalized_version: str | None = None,
    source_url: str = "https://vendor.example/releases/widget-3.14.5",
    test_body: str | None = None,
) -> tuple[LocalEnvironment, ExternalWorkspaceVerifier]:
    env = LocalEnvironment(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src" / "release.json").write_text(
        '{\n  "release": "unknown",\n  "source_url": ""\n}\n',
        encoding="utf-8",
    )
    (tmp_path / "tests" / "run_release_checks.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "grep -F '\"release\":' src/release.json >/dev/null\n"
        "grep -F '\"source_url\":' src/release.json >/dev/null\n",
        encoding="utf-8",
    )
    await env.exec("git init")
    await env.exec("git add src/release.json tests/run_release_checks.sh")
    await env.exec("git -c user.name=test -c user.email=test@example.com commit -m baseline")
    normalized_version_line = (
        f',\n  "normalized_version": "{normalized_version}"'
        if normalized_version is not None
        else ""
    )
    (tmp_path / "src" / "release.json").write_text(
        "{\n"
        f'  "release": "{release_value}",\n'
        f'  "source_url": "{source_url}"'
        f"{normalized_version_line}\n"
        "}\n",
        encoding="utf-8",
    )
    default_test_body = (
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        f'expected_release="{release_value}"\n'
        f'expected_source_url="{source_url}"\n'
        'grep -F "\\"release\\": \\"$expected_release\\"" src/release.json >/dev/null\n'
        'grep -F "\\"source_url\\": \\"$expected_source_url\\"" src/release.json >/dev/null\n'
    )
    (tmp_path / "tests" / "run_release_checks.sh").write_text(
        test_body if test_body is not None else default_test_body,
        encoding="utf-8",
    )
    structural = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))
    result = await structural.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "src/release.json"}),
            _completed("write_file", metadata={"path": "tests/run_release_checks.sh"}),
            _completed("verify_work", metadata={"command": "bash tests/run_release_checks.sh"}),
        ],
    )
    assert result.can_finish is True
    return env, structural


async def _accepted_url_join_workspace(
    tmp_path: Path,
    *,
    test_body: str,
) -> tuple[LocalEnvironment, ExternalWorkspaceVerifier]:
    env = LocalEnvironment(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "test").mkdir()
    (tmp_path / "src" / "urlJoin.js").write_text(
        'function joinUrl(...segments) { return segments.filter(Boolean).join("/"); }\n'
        "module.exports = { joinUrl };\n",
        encoding="utf-8",
    )
    (tmp_path / "test" / "urlJoin.test.js").write_text(
        'const assert = require("node:assert/strict");\n'
        'const { joinUrl } = require("../src/urlJoin");\n'
        'assert.equal(joinUrl("api", "v1"), "api/v1");\n',
        encoding="utf-8",
    )
    await env.exec("git init")
    await env.exec("git add src/urlJoin.js test/urlJoin.test.js")
    await env.exec("git -c user.name=test -c user.email=test@example.com commit -m baseline")
    (tmp_path / "src" / "urlJoin.js").write_text(
        "function joinUrl(...segments) {\n"
        "  return segments.filter((part) => part != null && part !== '').join('/');\n"
        "}\n"
        "module.exports = { joinUrl };\n",
        encoding="utf-8",
    )
    (tmp_path / "test" / "urlJoin.test.js").write_text(test_body, encoding="utf-8")
    structural = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))
    result = await structural.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "src/urlJoin.js"}),
            _completed("write_file", metadata={"path": "test/urlJoin.test.js"}),
            _completed("verify_work", metadata={"command": "project-test test/urlJoin.test.js"}),
        ],
    )
    assert result.can_finish is True
    return env, structural


async def _accepted_ini_lookup_workspace(
    tmp_path: Path,
    *,
    test_body: str,
) -> tuple[LocalEnvironment, ExternalWorkspaceVerifier]:
    env = LocalEnvironment(tmp_path)
    (tmp_path / "bin").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "Makefile").write_text("test:\n\t./tests/run.sh\n", encoding="utf-8")
    (tmp_path / "bin" / "ini_get").write_text(
        "#!/usr/bin/env bash\n"
        'section="$1"\n'
        'key="$2"\n'
        'file="$3"\n'
        'current=""\n'
        "while IFS= read -r line; do\n"
        '  case "$line" in\n'
        '    "["*"]") current="${line#\\[}"; current="${current%\\]}" ;;\n'
        '    "$key="*) if [ "$current" = "$section" ]; then printf \'%s\\n\' "${line#*=}"; exit 0; fi ;;\n'
        "  esac\n"
        'done < "$file"\n'
        "exit 1\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "run.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'tmp="$(mktemp)"\n'
        "trap 'rm -f \"$tmp\"' EXIT\n"
        "cat > \"$tmp\" <<'INI'\n"
        "[server]\n"
        "port=8080\n"
        "INI\n"
        '[ "$(./bin/ini_get server port "$tmp")" = "8080" ]\n',
        encoding="utf-8",
    )
    await env.exec("chmod +x bin/ini_get tests/run.sh")
    await env.exec("git init")
    await env.exec("git add Makefile bin/ini_get tests/run.sh")
    await env.exec("git -c user.name=test -c user.email=test@example.com commit -m baseline")
    (tmp_path / "bin" / "ini_get").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'section="${1:-}"\n'
        'key="${2:-}"\n'
        'file="${3:-}"\n'
        'current=""\n'
        "found=0\n"
        'found_value=""\n'
        "trim() { printf '%s' \"$1\" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//'; }\n"
        'section="$(trim "$section")"\n'
        'key="$(trim "$key")"\n'
        'while IFS= read -r line || [ -n "$line" ]; do\n'
        '  stripped="$(trim "$line")"\n'
        '  [ -z "$stripped" ] && continue\n'
        '  case "$stripped" in \\#*|\\;*) continue ;; esac\n'
        '  if [[ "$stripped" =~ ^\\[.*\\]$ ]]; then inner="${stripped#\\[}"; '
        'inner="${inner%\\]}"; current="$(trim "$inner")"; continue; fi\n'
        '  if [[ "$stripped" == *"="* ]]; then k="${stripped%%=*}"; '
        'v="${stripped#*=}"; k="$(trim "$k")"; v="$(trim "$v")"; '
        'if [ "$current" = "$section" ] && [ "$k" = "$key" ]; then '
        'found=1; found_value="$v"; fi; fi\n'
        'done < "$file"\n'
        'if [ "$found" -eq 1 ]; then printf \'%s\\n\' "$found_value"; exit 0; fi\n'
        "exit 1\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "run.sh").write_text(test_body, encoding="utf-8")
    await env.exec("chmod +x bin/ini_get tests/run.sh")
    structural = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))
    result = await structural.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "bin/ini_get"}),
            _completed("write_file", metadata={"path": "tests/run.sh"}),
            _completed("verify_work", metadata={"command": "./tests/run.sh"}),
        ],
    )
    assert result.can_finish is True
    return env, structural


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_weak_regression_tests(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_slug_workspace(tmp_path)
    adapter = CoverageReviewAdapter(
        {
            "can_finish": False,
            "reason": "tests miss punctuation/non slug-safe character handling",
            "confidence": 0.82,
        }
    )
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "slugify() should produce lowercase ASCII-ish slugs and return untitled "
            "when nothing slug-safe remains"
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(
        session=SimpleNamespace(
            messages=[
                Message(role="user", content="fix slugify"),
                Message(role="assistant", content="punctuation is handled"),
            ]
        ),
        activity=[],
    )

    assert result.can_finish is False
    assert "Regression coverage is too weak" in result.reason
    prompt = adapter.calls[0]["messages"][1].content
    assert "ORIGINAL TASK" in prompt
    assert "slugify.py" in prompt
    assert "tests/test_slugify.py" in prompt
    system_prompt = adapter.calls[0]["messages"][0].content
    assert "counterexample" in system_prompt
    assert "sanitize" in system_prompt
    assert "safe" in system_prompt
    assert "equivalence class" in system_prompt
    assert "near-duplicate" in system_prompt
    assert "contradict the task" in system_prompt
    assert "merely arguable" in system_prompt
    assert "Stay inside the user's stated behavior" in system_prompt
    assert "adjacent protocol" in system_prompt
    assert "query-string or fragment" in system_prompt
    assert "locale/Unicode collation" in system_prompt
    assert "should not break when present" in system_prompt
    assert "boundary words" in system_prompt
    assert "interior separator" in system_prompt
    assert "transient setup or verification command failure" in system_prompt
    assert "Harness/tooling environment" in system_prompt
    assert "command the agent tried" in system_prompt


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_weak_slugify_tests(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_slug_workspace(
        tmp_path,
        test_body=(
            "from slugify import slugify\n\n"
            "def test_basic_slugify():\n"
            "    assert slugify('Hello World') == 'hello-world'\n\n"
            "def test_unicode_and_fallback():\n"
            "    assert slugify('Crème Brûlée') == 'creme-brulee'\n"
            "    assert slugify('---') == 'untitled'\n"
        ),
    )
    adapter = CoverageReviewAdapter({"can_finish": True, "reason": "looks good", "confidence": 0.9})
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix slugify. It should produce lowercase ASCII-ish slugs, collapse "
            "repeated separators into one hyphen, trim leading/trailing separators, "
            "and return 'untitled' when nothing slug-safe remains."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "repeated or leading/trailing separator input" in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_unasserted_slugify_separator_sample(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_slug_workspace(
        tmp_path,
        test_body=(
            "from slugify import slugify\n\n"
            "SEPARATOR_SAMPLE = '---Hello__World---'\n\n"
            "def test_unicode_and_fallback():\n"
            "    assert slugify('Crème Brûlée') == 'creme-brulee'\n"
            "    assert slugify('---') == 'untitled'\n"
        ),
    )
    adapter = CoverageReviewAdapter({"can_finish": True, "reason": "looks good", "confidence": 0.9})
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix slugify. It should produce lowercase ASCII-ish slugs, collapse "
            "repeated separators into one hyphen, trim leading/trailing separators, "
            "and return 'untitled' when nothing slug-safe remains."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "repeated or leading/trailing separator input" in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_accepts_representative_slugify_tests(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_slug_workspace(
        tmp_path,
        test_body=(
            "from slugify import slugify\n\n"
            "def test_basic_slugify():\n"
            "    assert slugify('Hello World') == 'hello-world'\n\n"
            "def test_slugify_contract():\n"
            "    assert slugify('Crème Brûlée') == 'creme-brulee'\n"
            "    assert slugify('  Hello__World--- ') == 'hello-world'\n"
            "    assert slugify('!!!---   $$$') == 'untitled'\n"
        ),
    )
    adapter = CoverageReviewAdapter(
        {"can_finish": True, "reason": "tests cover slug behavior", "confidence": 0.9}
    )
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix slugify. It should produce lowercase ASCII-ish slugs, collapse "
            "repeated separators into one hyphen, trim leading/trailing separators, "
            "and return 'untitled' when nothing slug-safe remains."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is True
    assert "coverage review passed" in result.reason
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_weak_ini_lookup_tests(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_ini_lookup_workspace(
        tmp_path,
        test_body=(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'tmp="$(mktemp)"\n'
            "trap 'rm -f \"$tmp\"' EXIT\n"
            "cat > \"$tmp\" <<'INI'\n"
            "[server]\n"
            "port=8080\n"
            "# comment\n"
            "INI\n"
            '[ "$(./bin/ini_get server port "$tmp")" = "8080" ]\n'
        ),
    )
    adapter = CoverageReviewAdapter({"can_finish": True, "reason": "looks good", "confidence": 0.9})
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix the INI lookup CLI. It should ignore blank lines and full-line "
            "comments beginning with # or ;, trim surrounding whitespace around "
            "section names, keys, and values, preserve equals signs inside values, "
            "keep section scoping correct, and exit non-zero with no output when "
            "the key is missing."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "# and ; full-line comment cases" in result.reason
    assert "indented full-line comment" in result.reason
    assert "section header with surrounding whitespace" in result.reason
    assert "leading whitespace before the key" in result.reason
    assert "value containing an additional `=`" in result.reason
    assert "value with surrounding whitespace" in result.reason
    assert "repeated key in the same section" in result.reason
    assert "missing-key behavior" in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_ini_lookup_missing_indented_cases(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_ini_lookup_workspace(
        tmp_path,
        test_body=(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'tmp="$(mktemp)"\n'
            "trap 'rm -f \"$tmp\"' EXIT\n"
            "assert_failure_no_output() {\n"
            "  local out rc\n"
            '  out="$($* 2>/dev/null)"; rc=$?\n'
            '  [ "$rc" -ne 0 ] && [ -z "$out" ]\n'
            "}\n"
            "cat > \"$tmp\" <<'INI'\n"
            "# hash comment\n"
            "; semicolon comment\n"
            "[ server ]\n"
            "port = 8080\n"
            "token = abc=def=ghi\n"
            "[client]\n"
            "port = 9090\n"
            "INI\n"
            '[ "$(./bin/ini_get server port "$tmp")" = "8080" ]\n'
            '[ "$(./bin/ini_get server token "$tmp")" = "abc=def=ghi" ]\n'
            '[ "$(./bin/ini_get client port "$tmp")" = "9090" ]\n'
            'assert_failure_no_output ./bin/ini_get server missing "$tmp"\n'
        ),
    )
    adapter = CoverageReviewAdapter({"can_finish": True, "reason": "looks good", "confidence": 0.9})
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix the INI lookup CLI. It should ignore blank lines and full-line "
            "comments beginning with # or ;, trim surrounding whitespace around "
            "section names, keys, and values, preserve equals signs inside values, "
            "keep section scoping correct, and exit non-zero with no output when "
            "the key is missing."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "indented full-line comment" in result.reason
    assert "leading whitespace before the key" in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_ini_lookup_missing_value_trim_assertion(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_ini_lookup_workspace(
        tmp_path,
        test_body=(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'tmp="$(mktemp)"\n'
            "trap 'rm -f \"$tmp\"' EXIT\n"
            "assert_failure_no_output() {\n"
            "  local out rc\n"
            '  out="$($* 2>/dev/null)"; rc=$?\n'
            '  [ "$rc" -ne 0 ] && [ -z "$out" ]\n'
            "}\n"
            "cat > \"$tmp\" <<'INI'\n"
            "# hash comment\n"
            "  ; semicolon comment\n"
            "[ server ]\n"
            "  port = 8080\n"
            "token = abc=def=ghi\n"
            "retry = first\n"
            "retry = second\n"
            "[client]\n"
            "port = 9090\n"
            "INI\n"
            '[ "$(./bin/ini_get server port "$tmp")" = "8080" ]\n'
            '[ "$(./bin/ini_get server token "$tmp")" = "abc=def=ghi" ]\n'
            '[ "$(./bin/ini_get server retry "$tmp")" = "second" ]\n'
            '[ "$(./bin/ini_get client port "$tmp")" = "9090" ]\n'
            'assert_failure_no_output ./bin/ini_get server missing "$tmp"\n'
        ),
    )
    adapter = CoverageReviewAdapter({"can_finish": True, "reason": "looks good", "confidence": 0.9})
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix the INI lookup CLI. It should ignore blank lines and full-line "
            "comments beginning with # or ;, trim surrounding whitespace around "
            "section names, keys, and values, preserve equals signs inside values, "
            "keep section scoping correct, and exit non-zero with no output when "
            "the key is missing."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "value with surrounding whitespace" in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_ini_lookup_missing_repeated_key_case(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_ini_lookup_workspace(
        tmp_path,
        test_body=(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'tmp="$(mktemp)"\n'
            "trap 'rm -f \"$tmp\"' EXIT\n"
            "assert_failure_no_output() {\n"
            "  local out rc\n"
            '  out="$($* 2>/dev/null)"; rc=$?\n'
            '  [ "$rc" -ne 0 ] && [ -z "$out" ]\n'
            "}\n"
            "cat > \"$tmp\" <<'INI'\n"
            "# hash comment\n"
            "  ; semicolon comment\n"
            "[ server ]\n"
            "  port = 8080\n"
            "token =   abc=def=ghi   \n"
            "[client]\n"
            "port = 9090\n"
            "INI\n"
            '[ "$(./bin/ini_get server port "$tmp")" = "8080" ]\n'
            '[ "$(./bin/ini_get server token "$tmp")" = "abc=def=ghi" ]\n'
            '[ "$(./bin/ini_get client port "$tmp")" = "9090" ]\n'
            'assert_failure_no_output ./bin/ini_get server missing "$tmp"\n'
        ),
    )
    adapter = CoverageReviewAdapter({"can_finish": True, "reason": "looks good", "confidence": 0.9})
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix the INI lookup CLI. It should ignore blank lines and full-line "
            "comments beginning with # or ;, trim surrounding whitespace around "
            "section names, keys, and values, preserve equals signs inside values, "
            "keep section scoping correct, and exit non-zero with no output when "
            "the key is missing."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "repeated key in the same section" in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_accepts_representative_ini_lookup_tests(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_ini_lookup_workspace(
        tmp_path,
        test_body=(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'tmp="$(mktemp)"\n'
            "trap 'rm -f \"$tmp\"' EXIT\n"
            "assert_failure_no_output() {\n"
            "  local out rc\n"
            '  out="$($* 2>/dev/null)"; rc=$?\n'
            '  [ "$rc" -ne 0 ] && [ -z "$out" ]\n'
            "}\n"
            "cat > \"$tmp\" <<'INI'\n"
            "# hash comment\n"
            "  ; semicolon comment\n"
            "[ server ]\n"
            "  port = 8080\n"
            "token =   abc=def=ghi   \n"
            "retry = first\n"
            "retry = second\n"
            "[client]\n"
            "port = 9090\n"
            "INI\n"
            '[ "$(./bin/ini_get server port "$tmp")" = "8080" ]\n'
            '[ "$(./bin/ini_get server token "$tmp")" = "abc=def=ghi" ]\n'
            '[ "$(./bin/ini_get server retry "$tmp")" = "second" ]\n'
            '[ "$(./bin/ini_get client port "$tmp")" = "9090" ]\n'
            'assert_failure_no_output ./bin/ini_get server missing "$tmp"\n'
        ),
    )
    adapter = CoverageReviewAdapter(
        {"can_finish": True, "reason": "tests cover INI behavior", "confidence": 0.9}
    )
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix the INI lookup CLI. It should ignore blank lines and full-line "
            "comments beginning with # or ;, trim surrounding whitespace around "
            "section names, keys, and values, preserve equals signs inside values, "
            "keep section scoping correct, and exit non-zero with no output when "
            "the key is missing."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is True
    assert "coverage review passed" in result.reason
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_url_join_missing_clean_boundary(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_url_join_workspace(
        tmp_path,
        test_body=(
            'const assert = require("node:assert/strict");\n'
            'const { joinUrl } = require("../src/urlJoin");\n'
            'assert.equal(joinUrl("https://example.com/", "/api/"), "https://example.com/api");\n'
            'assert.equal(joinUrl("https://example.com", "///"), "https://example.com");\n'
        ),
    )
    adapter = CoverageReviewAdapter({"can_finish": True, "reason": "looks good", "confidence": 0.9})
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix URL joining. joinUrl should preserve a URL scheme and host, collapse "
            "duplicate slashes between path segments, and trim trailing slashes."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "clean scheme+host segment is followed by a clean path segment" in result.reason
    assert "https://example.com/api" in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_allows_url_join_clean_boundary(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_url_join_workspace(
        tmp_path,
        test_body=(
            'const assert = require("node:assert/strict");\n'
            'const { joinUrl } = require("../src/urlJoin");\n'
            'assert.equal(joinUrl("https://example.com", "api"), "https://example.com/api");\n'
            'assert.equal(joinUrl("https://example.com/", "/api/"), "https://example.com/api");\n'
            'assert.equal(joinUrl("https://example.com", "", null, "api"), "https://example.com/api");\n'
            'assert.equal(joinUrl("/api", "v1"), "/api/v1");\n'
            'assert.equal(joinUrl("api", "/v1"), "api/v1");\n'
            'assert.equal(joinUrl("api", "//v1//users"), "api/v1/users");\n'
            'assert.equal(joinUrl("https://example.com", "/"), "https://example.com");\n'
        ),
    )
    adapter = CoverageReviewAdapter(
        {"can_finish": True, "reason": "tests cover URL boundaries", "confidence": 0.9}
    )
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix URL joining. joinUrl should preserve a URL scheme and host, ignore "
            "nullish or empty segments, collapse duplicate slashes between path "
            "segments, keep a leading slash for absolute paths, and trim trailing "
            "slashes."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is True
    assert "coverage review passed" in result.reason
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_url_join_missing_empty_segment_case(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_url_join_workspace(
        tmp_path,
        test_body=(
            'const assert = require("node:assert/strict");\n'
            'const { joinUrl } = require("../src/urlJoin");\n'
            'assert.equal(joinUrl("https://example.com", "api"), "https://example.com/api");\n'
            'assert.equal(joinUrl("https://example.com/", "/api/"), "https://example.com/api");\n'
            'assert.equal(joinUrl("/api", "v1"), "/api/v1");\n'
        ),
    )
    adapter = CoverageReviewAdapter({"can_finish": True, "reason": "looks good", "confidence": 0.9})
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix URL joining. joinUrl should preserve a URL scheme and host, ignore "
            "nullish or empty segments, collapse duplicate slashes between path "
            "segments, keep a leading slash for absolute paths, and trim trailing "
            "slashes."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "ignored nullish/empty segments" in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_url_join_relative_only_empty_segment_case(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_url_join_workspace(
        tmp_path,
        test_body=(
            'const assert = require("node:assert/strict");\n'
            'const { joinUrl } = require("../src/urlJoin");\n'
            'assert.equal(joinUrl("https://example.com", "api"), "https://example.com/api");\n'
            'assert.equal(joinUrl("https://example.com/", "/api/"), "https://example.com/api");\n'
            'assert.equal(joinUrl("api", null, "", undefined, "users"), "api/users");\n'
            'assert.equal(joinUrl("/api", "v1"), "/api/v1");\n'
        ),
    )
    adapter = CoverageReviewAdapter({"can_finish": True, "reason": "looks good", "confidence": 0.9})
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix URL joining. joinUrl should preserve a URL scheme and host, ignore "
            "nullish or empty segments, collapse duplicate slashes between path "
            "segments, keep a leading slash for absolute paths, and trim trailing "
            "slashes."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "ignored nullish/empty segments" in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_url_join_missing_absolute_path_case(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_url_join_workspace(
        tmp_path,
        test_body=(
            'const assert = require("node:assert/strict");\n'
            'const { joinUrl } = require("../src/urlJoin");\n'
            'assert.equal(joinUrl("https://example.com", "api"), "https://example.com/api");\n'
            'assert.equal(joinUrl("https://example.com/", "/api/"), "https://example.com/api");\n'
            'assert.equal(joinUrl("https://example.com", "", null, "api"), "https://example.com/api");\n'
        ),
    )
    adapter = CoverageReviewAdapter({"can_finish": True, "reason": "looks good", "confidence": 0.9})
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix URL joining. joinUrl should preserve a URL scheme and host, ignore "
            "nullish or empty segments, collapse duplicate slashes between path "
            "segments, keep a leading slash for absolute paths, and trim trailing "
            "slashes."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "keeps the leading slash" in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_url_join_missing_origin_root_trim_case(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_url_join_workspace(
        tmp_path,
        test_body=(
            'const assert = require("node:assert/strict");\n'
            'const { joinUrl } = require("../src/urlJoin");\n'
            'assert.equal(joinUrl("https://example.com", "api"), "https://example.com/api");\n'
            'assert.equal(joinUrl("https://example.com/", "/api/"), "https://example.com/api");\n'
            'assert.equal(joinUrl("https://example.com", "", null, "api"), "https://example.com/api");\n'
            'assert.equal(joinUrl("/api", "v1"), "/api/v1");\n'
            'assert.equal(joinUrl("api", "/v1"), "api/v1");\n'
            'assert.equal(joinUrl("api", "//v1//users"), "api/v1/users");\n'
        ),
    )
    adapter = CoverageReviewAdapter({"can_finish": True, "reason": "looks good", "confidence": 0.9})
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix URL joining. joinUrl should preserve a URL scheme and host, ignore "
            "nullish or empty segments, collapse duplicate slashes between path "
            "segments, keep a leading slash for absolute paths, and trim trailing "
            "slashes except for the root URL."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "slash-only root path" in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_url_join_later_absolute_segment_reset(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_url_join_workspace(
        tmp_path,
        test_body=(
            'const assert = require("node:assert/strict");\n'
            'const { joinUrl } = require("../src/urlJoin");\n'
            'assert.equal(joinUrl("https://example.com", "api"), "https://example.com/api");\n'
            'assert.equal(joinUrl("https://example.com/", "/api/"), "https://example.com/api");\n'
            'assert.equal(joinUrl("https://example.com", "", null, "api"), "https://example.com/api");\n'
            'assert.equal(joinUrl("/api", "v1"), "/api/v1");\n'
            'assert.equal(joinUrl("api", "/v1"), "/api/v1");\n'
            'assert.equal(joinUrl("https://example.com", "/"), "https://example.com");\n'
        ),
    )
    adapter = CoverageReviewAdapter({"can_finish": True, "reason": "looks good", "confidence": 0.9})
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix URL joining. joinUrl should preserve a URL scheme and host, ignore "
            "nullish or empty segments, collapse duplicate slashes between path "
            "segments, keep a leading slash for absolute paths, and trim trailing "
            "slashes except for the root URL."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "stays relative" in result.reason
    assert 'joinUrl("api", "/v1")' in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_url_join_relative_duplicate_slash_reset(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_url_join_workspace(
        tmp_path,
        test_body=(
            'const assert = require("node:assert/strict");\n'
            'const { joinUrl } = require("../src/urlJoin");\n'
            'assert.equal(joinUrl("https://example.com", "api"), "https://example.com/api");\n'
            'assert.equal(joinUrl("https://example.com/", "/api/"), "https://example.com/api");\n'
            'assert.equal(joinUrl("https://example.com", "", null, "api"), "https://example.com/api");\n'
            'assert.equal(joinUrl("/api", "v1"), "/api/v1");\n'
            'assert.equal(joinUrl("api", "/v1"), "api/v1");\n'
            'assert.equal(joinUrl("api", "//v1//users"), "/api/v1/users");\n'
            'assert.equal(joinUrl("https://example.com", "/"), "https://example.com");\n'
        ),
    )
    adapter = CoverageReviewAdapter({"can_finish": True, "reason": "looks good", "confidence": 0.9})
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix URL joining. joinUrl should preserve a URL scheme and host, ignore "
            "nullish or empty segments, collapse duplicate slashes between path "
            "segments, keep a leading slash for absolute paths, and trim trailing "
            "slashes except for the root URL."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "duplicate slashes in relative paths" in result.reason
    assert 'joinUrl("api", "//v1//users")' in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_url_join_missing_origin_slash_boundary(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_url_join_workspace(
        tmp_path,
        test_body=(
            'const assert = require("node:assert/strict");\n'
            'const { joinUrl } = require("../src/urlJoin");\n'
            'assert.equal(joinUrl("https://example.com", "api"), "https://example.com/api");\n'
            'assert.equal(joinUrl("api//", "/v1/"), "api/v1");\n'
        ),
    )
    adapter = CoverageReviewAdapter({"can_finish": True, "reason": "looks good", "confidence": 0.9})
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Fix URL joining. joinUrl should preserve a URL scheme and host, collapse "
            "duplicate slashes between path segments, and trim trailing slashes."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "duplicate-slash collapse across the scheme+host/path boundary" in result.reason
    assert 'joinUrl("https://example.com/", "/api/")' in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_release_without_exact_source_url(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_release_metadata_workspace(
        tmp_path,
        release_value="3.14.5",
        test_body=(
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            'grep -F \'"release": "3.14.5"\' src/release.json >/dev/null\n'
            'grep -F \'"source_url": "https://vendor.example/\' src/release.json >/dev/null\n'
        ),
    )
    adapter = CoverageReviewAdapter(
        {"can_finish": True, "reason": "tests check release metadata", "confidence": 0.9}
    )
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Update the product metadata with the latest stable release version "
            "from official public sources. Add a focused project test for the "
            "release value and source URL."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "exact source URL value" in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_release_without_exact_value(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_release_metadata_workspace(
        tmp_path,
        release_value="3.14.5",
        source_url="https://vendor.example/releases/current",
        test_body=(
            "#!/usr/bin/env bash\n"
            "set -eu\n"
            "grep -F '\"release\":' src/release.json >/dev/null\n"
            'grep -F \'"source_url": "https://vendor.example/releases/current"\' '
            "src/release.json >/dev/null\n"
        ),
    )
    adapter = CoverageReviewAdapter(
        {"can_finish": True, "reason": "tests check release metadata", "confidence": 0.9}
    )
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Update the product metadata with the latest stable release version "
            "from official public sources. Add a focused project test for the "
            "release value and source URL."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "concrete current/latest release value" in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_rejects_labelled_release_value(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_release_metadata_workspace(
        tmp_path,
        release_value="Widget 3.14.5",
        normalized_version="3.14.5",
        source_url="https://vendor.example/releases/widget-3145",
    )
    adapter = CoverageReviewAdapter(
        {"can_finish": True, "reason": "tests assert the display label", "confidence": 0.9}
    )
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Update the product metadata with the latest stable release version "
            "from official public sources. Add a focused project test for the "
            "release value and source URL."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "label prefix" in result.reason
    assert "3.14.5" in result.reason
    assert adapter.calls == []


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_allows_labelled_release_when_requested(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_release_metadata_workspace(
        tmp_path,
        release_value="Widget 3.14.5",
        normalized_version="3.14.5",
        source_url="https://vendor.example/releases/widget-3145",
    )
    adapter = CoverageReviewAdapter(
        {"can_finish": True, "reason": "tests assert the display label", "confidence": 0.9}
    )
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction=(
            "Update the product metadata with the latest stable release display "
            "title including the product name from official public sources."
        ),
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is True
    assert result.reason == "coverage review passed: tests assert the display label"
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_treats_low_confidence_rejection_as_advisory(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_slug_workspace(tmp_path)
    adapter = CoverageReviewAdapter(
        {
            "can_finish": False,
            "reason": "maybe add another nearby punctuation variant",
            "confidence": 0.5,
        }
    )
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction="fix slugify",
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
        block_confidence=0.75,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is True
    assert "advisory below blocking confidence" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_blocks_borderline_weak_tests_by_default(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_slug_workspace(tmp_path)
    adapter = CoverageReviewAdapter(
        {
            "can_finish": False,
            "reason": "tests miss a distinct behavior claimed by the task",
            "confidence": 0.74,
        }
    )
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction="fix slugify",
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "Regression coverage is too weak" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_blocks_counterexample_advisory_by_default(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_slug_workspace(tmp_path)
    adapter = CoverageReviewAdapter(
        {
            "can_finish": False,
            "reason": (
                "tests are not strong enough to catch obvious spec violations; "
                "counterexample: preserve-resets could be entirely broken while "
                "current tests still pass"
            ),
            "confidence": 0.66,
        }
    )
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction="fix slugify",
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is False
    assert "Regression coverage is too weak" in result.reason
    assert "preserve-resets could be entirely broken" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_coverage_verifier_accepts_reviewed_tests(
    tmp_path: Path,
) -> None:
    env, structural = await _accepted_slug_workspace(tmp_path)
    adapter = CoverageReviewAdapter(
        {"can_finish": True, "reason": "tests cover the main behavior", "confidence": 0.9}
    )
    verifier = ExternalWorkspaceCoverageVerifier(
        environment=env,
        workdir=str(tmp_path),
        instruction="fix slugify",
        adapter=adapter,
        model="judge",
        structural_verifier=structural,
    )

    result = await verifier.verify(session=SimpleNamespace(messages=[]), activity=[])

    assert result.can_finish is True
    assert "coverage review passed" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_verifier_accepts_default_runner_wired_untracked_test(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "bin").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "bin" / "tool").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    (tmp_path / "tests" / "run.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\necho ok\n",
        encoding="utf-8",
    )
    (tmp_path / "Makefile").write_text("test:\n\t./tests/run.sh\n", encoding="utf-8")
    await env.exec("git add Makefile bin/tool tests/run.sh")
    await env.exec("git -c user.name=test -c user.email=test@example.com commit -m baseline")
    (tmp_path / "bin" / "tool").write_text(
        "#!/usr/bin/env bash\necho fixed\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "regression.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\necho regression\n",
        encoding="utf-8",
    )
    (tmp_path / "Makefile").write_text(
        "test:\n\tbash tests/run.sh && bash tests/regression.sh\n",
        encoding="utf-8",
    )
    verifier = ExternalWorkspaceVerifier(
        env,
        workdir=str(tmp_path),
        require_default_verify_command=True,
    )

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "bin/tool"}),
            _completed("write_file", metadata={"path": "tests/regression.sh"}),
            _completed("edit_file", metadata={"path": "Makefile"}),
            _completed(
                "verify_work",
                metadata={"command": "make test", "used_default_command": True},
            ),
        ],
    )

    assert result.can_finish is True
    assert verifier.latest.source_change_paths == ["Makefile", "bin/tool"]
    assert verifier.latest.test_change_paths == ["tests/regression.sh"]
    assert verifier.latest.verification_passed_after_source_change is True


@pytest.mark.asyncio
async def test_external_workspace_verifier_accepts_changed_runner_script_for_changed_tests(
    tmp_path: Path,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "tests").mkdir()
    (tmp_path / "slugify.py").write_text(
        "def slugify(value):\n    return value\n", encoding="utf-8"
    )
    (tmp_path / "quick_check.sh").write_text(
        '#!/bin/sh\necho "All tests passed"\n',
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_slugify.py").write_text(
        "from slugify import slugify\n\n\ndef test_basic():\n    assert slugify('ok') == 'ok'\n",
        encoding="utf-8",
    )
    await env.exec("git add slugify.py quick_check.sh tests/test_slugify.py")
    await env.exec("git -c user.name=test -c user.email=test@example.com commit -m baseline")
    (tmp_path / "slugify.py").write_text(
        "def slugify(value):\n    return str(value).lower().replace(' ', '-')\n",
        encoding="utf-8",
    )
    (tmp_path / "quick_check.sh").write_text(
        "#!/bin/sh\nset -eu\npython -m unittest -v tests.test_slugify\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_slugify.py").write_text(
        "import unittest\nfrom slugify import slugify\n\n\n"
        "class TestSlugify(unittest.TestCase):\n"
        "    def test_basic(self):\n"
        "        self.assertEqual(slugify('Hello World'), 'hello-world')\n",
        encoding="utf-8",
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("edit_file", metadata={"path": "slugify.py"}),
            _completed("edit_file", metadata={"path": "quick_check.sh"}),
            _completed("edit_file", metadata={"path": "tests/test_slugify.py"}),
            _completed("verify_work", metadata={"command": "sh quick_check.sh"}),
        ],
    )

    assert result.can_finish is True
    assert verifier.latest.source_change_paths == ["quick_check.sh", "slugify.py"]
    assert verifier.latest.test_change_paths == ["tests/test_slugify.py"]
    assert verifier.latest.verification_passed_after_source_change is True


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_verify_that_misses_changed_tests(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M ast/expr.go\n M tests/test_expr.py\n",
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "ast/expr.go"}),
            _completed("write_file", metadata={"path": "tests/test_expr.py"}),
            _completed("verify_work", metadata={"command": "pytest tests/test_other.py"}),
        ],
    )

    assert result.can_finish is False
    assert "did not cover the changed regression tests" in result.reason
    assert "configured default verify_work" not in result.reason
    assert verifier.latest.verification_passed_after_source_change is False


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_broad_verify_for_changed_fixture(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M ast/expr.go\n M core/testdata/func.ank\n",
    )
    verifier = ExternalWorkspaceVerifier(
        env,
        workdir=str(tmp_path),
        require_regression_test_change=False,
    )

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "ast/expr.go"}),
            _completed("write_file", metadata={"path": "core/testdata/func.ank"}),
            _completed("verify_work", metadata={"command": "go test ./..."}),
        ],
    )

    assert result.can_finish is False
    assert "did not cover the changed regression tests" in result.reason
    assert verifier.latest.source_change_paths == ["ast/expr.go"]
    assert verifier.latest.test_change_paths == ["core/testdata/func.ank"]
    assert verifier.latest.verification_passed_after_source_change is False


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_later_failed_verify_after_pass(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M ast/expr.go\n M tests/test_expr.py\n",
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "ast/expr.go"}),
            _completed("write_file", metadata={"path": "tests/test_expr.py"}),
            _completed("verify_work", metadata={"command": "pytest tests/test_expr.py"}),
            _completed(
                "verify_work",
                is_error=True,
                metadata={"command": "pytest tests/test_expr.py"},
                content_preview="1 failed",
            ),
        ],
    )

    assert result.can_finish is False
    assert "latest verify_work after the final workspace mutation did not pass" in result.reason
    assert "at least one current assumption is false" in result.reason
    assert "switch implementation path" in result.reason
    assert verifier.latest.latest_verification_error == "1 failed"
    assert verifier.latest.verification_passed_after_source_change is False


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_dependency_setup_handoff(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M parser/parser.go\n M vm/vmExprFunction.go\n",
    )
    verifier = ExternalWorkspaceVerifier(
        env,
        workdir=str(tmp_path),
        require_regression_test_change=False,
    )
    session = SimpleNamespace(
        messages=[
            Message(role="user", content="Implement the feature."),
            Message(
                role="assistant",
                content=(
                    "I cannot complete this change because the parser generator "
                    "binary is not available in the environment. If you can provide "
                    "the missing tool or tell me how the project normally regenerates "
                    "the parser, I can finish it."
                ),
            ),
        ]
    )

    result = await verifier.verify(
        session=session,
        activity=[
            _completed("write_file", metadata={"path": "parser/parser.go"}),
            _completed(
                "verify_work",
                is_error=True,
                metadata={"command": "go test ./...", "used_default_command": True},
                content_preview="FAILED (exit 1)",
            ),
        ],
    )

    assert result.can_finish is False
    assert "defers repository, source, environment, or dependency setup" in result.reason
    assert "Continue autonomously" in result.reason
    assert "check Docker availability" in result.reason
    assert verifier.latest.verification_passed_after_source_change is False


async def _prepare_repo_with_declared_docker_runtime(tmp_path: Path) -> LocalEnvironment:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "environment").mkdir()
    (tmp_path / "environment" / "Dockerfile").write_text(
        "FROM example.test/toolchain:latest\nWORKDIR /app\n",
        encoding="utf-8",
    )
    (tmp_path / "parser").mkdir()
    (tmp_path / "parser" / "parser.go").write_text("package parser\n", encoding="utf-8")
    await env.exec("git add environment/Dockerfile parser/parser.go")
    await env.exec("git -c user.name=test -c user.email=test@example.com commit -m baseline")
    (tmp_path / "parser" / "parser.go").write_text(
        "package parser\n\nconst fixed = true\n",
        encoding="utf-8",
    )
    return env


@pytest.mark.asyncio
async def test_external_workspace_verifier_requires_declared_runtime_check_after_missing_tool(
    tmp_path: Path,
) -> None:
    env = await _prepare_repo_with_declared_docker_runtime(tmp_path)
    verifier = ExternalWorkspaceVerifier(
        env,
        workdir=str(tmp_path),
        require_regression_test_change=False,
    )

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("edit_file", metadata={"path": "parser/parser.go"}),
            _completed(
                "verify_work",
                is_error=True,
                metadata={"command": "go test ./...", "exit_code": 127},
                content_preview=(
                    "FAILED (exit 127)\n\nstderr:\nbash: line 1: go: command not found\n"
                ),
            ),
        ],
    )

    assert result.can_finish is False
    assert "declares a Dockerfile or container image" in result.reason
    assert "command -v docker" in result.reason
    assert "docker info" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_verifier_accepts_missing_tool_exploration_after_docker_check(
    tmp_path: Path,
) -> None:
    env = await _prepare_repo_with_declared_docker_runtime(tmp_path)
    verifier = ExternalWorkspaceVerifier(
        env,
        workdir=str(tmp_path),
        require_regression_test_change=False,
    )

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("edit_file", metadata={"path": "parser/parser.go"}),
            _completed(
                "shell",
                is_error=True,
                metadata={"command": "command -v docker", "exit_code": 1},
                content_preview="exit_code: 1\n\nstdout:\n\nstderr:\n",
            ),
            _completed(
                "verify_work",
                is_error=True,
                metadata={"command": "go test ./...", "exit_code": 127},
                content_preview=(
                    "FAILED (exit 127)\n\nstderr:\nbash: line 1: go: command not found\n"
                ),
            ),
        ],
    )

    assert result.can_finish is False
    assert "declares a Dockerfile or container image" not in result.reason
    assert "latest verify_work after the final workspace mutation did not pass" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_verifier_accepts_commandless_docker_output(
    tmp_path: Path,
) -> None:
    env = await _prepare_repo_with_declared_docker_runtime(tmp_path)
    verifier = ExternalWorkspaceVerifier(
        env,
        workdir=str(tmp_path),
        require_regression_test_change=False,
    )

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("edit_file", metadata={"path": "parser/parser.go"}),
            _completed(
                "shell",
                metadata={"exit_code": 0},
                content_preview="exit_code: 0\n\nstdout:\nDocker version 29.4.0, build 9d7ad9f\n",
            ),
            _completed(
                "shell",
                metadata={"exit_code": 0},
                content_preview=(
                    "exit_code: 0\n\nstdout:\nSending build context to Docker daemon  68.13MB\n"
                ),
            ),
            _completed(
                "verify_work",
                is_error=True,
                metadata={"exit_code": 127},
                content_preview=(
                    "FAILED (exit 127)\n\n"
                    "stderr:\n./scripts/check: line 13: project-check: command not found\n"
                ),
            ),
        ],
    )

    assert result.can_finish is False
    assert "declares a Dockerfile or container image" not in result.reason
    assert "latest verify_work after the final workspace mutation did not pass" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_permission_handoff_after_failed_verify(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M src/app.py\n M tests/test_app.py\n",
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))
    session = SimpleNamespace(
        messages=[
            Message(role="user", content="Implement the requested behavior."),
            Message(
                role="assistant",
                content=(
                    "Verification is still failing. If you want me to proceed, "
                    "I'll inspect the implementation and continue fixing it."
                ),
            ),
        ]
    )

    result = await verifier.verify(
        session=session,
        activity=[
            _completed("write_file", metadata={"path": "src/app.py"}),
            _completed("write_file", metadata={"path": "tests/test_app.py"}),
            _completed(
                "verify_work",
                is_error=True,
                metadata={"command": "pytest tests/test_app.py", "exit_code": 1},
                content_preview="FAILED (exit 1)\n\nstdout:\n1 failed\n",
            ),
        ],
    )

    assert result.can_finish is False
    assert "asks the user for permission or confirmation to continue" in result.reason
    assert "Latest verify_work error: FAILED (exit 1)" in result.reason
    assert "defers repository, source, environment, or dependency setup" not in result.reason
    assert verifier.latest.verification_passed_after_source_change is False


@pytest.mark.asyncio
async def test_external_workspace_verifier_does_not_treat_verification_setup_as_user_handoff(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M src/app.py\n M tests/test_app.py\n",
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))
    session = SimpleNamespace(
        messages=[
            Message(role="user", content="Implement the requested behavior."),
            Message(
                role="assistant",
                content=(
                    "I am blocked on verification because of an environment "
                    "inconsistency. I will inspect the harness verification setup "
                    "and run verify_work again."
                ),
            ),
        ]
    )

    result = await verifier.verify(
        session=session,
        activity=[
            _completed("write_file", metadata={"path": "src/app.py"}),
            _completed("write_file", metadata={"path": "tests/test_app.py"}),
            _completed(
                "verify_work",
                is_error=True,
                metadata={"command": "pytest tests/test_app.py", "exit_code": 1},
                content_preview="FAILED (exit 1)\n\nstdout:\n1 failed\n",
            ),
        ],
    )

    assert result.can_finish is False
    assert "latest verify_work after the final workspace mutation did not pass" in result.reason
    assert "defers repository, source, environment, or dependency setup" not in result.reason
    assert "asks the user for permission or confirmation to continue" not in result.reason
    assert verifier.latest.verification_passed_after_source_change is False


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_later_mutating_verify_after_pass(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M ast/expr.go\n M tests/test_expr.py\n",
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "ast/expr.go"}),
            _completed("write_file", metadata={"path": "tests/test_expr.py"}),
            _completed("verify_work", metadata={"command": "pytest tests/test_expr.py"}),
            _completed(
                "verify_work",
                metadata={
                    "command": "pytest tests/test_expr.py",
                    "workspace_changed": True,
                },
                content_preview="workspace changed",
            ),
        ],
    )

    assert result.can_finish is False
    assert "did not produce a later passing" in result.reason
    assert verifier.latest.latest_verification_error == "workspace changed"
    assert verifier.latest.verification_passed_after_source_change is False


@pytest.mark.asyncio
async def test_external_workspace_verifier_requires_configured_default_verifier(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M ast/expr.go\n M tests/test_expr.py\n",
    )
    verifier = ExternalWorkspaceVerifier(
        env,
        workdir=str(tmp_path),
        require_default_verify_command=True,
    )

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "ast/expr.go"}),
            _completed("write_file", metadata={"path": "tests/test_expr.py"}),
            _completed(
                "verify_work",
                metadata={"command": "pytest", "used_default_command": False},
            ),
        ],
    )

    assert result.can_finish is False
    assert "configured default verifier" in result.reason
    assert verifier.latest.latest_verification_command == "pytest"
    assert verifier.latest.latest_verification_used_default_command is False


@pytest.mark.asyncio
async def test_external_workspace_verifier_requires_public_no_network_image_check(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M ast/expr.go\n M tests/test_expr.py\n",
    )
    verifier = ExternalWorkspaceVerifier(
        env,
        workdir=str(tmp_path),
        policy=ExternalWorkspacePolicy(
            required_no_network_verify_image="public.example/task:latest"
        ),
    )

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "ast/expr.go"}),
            _completed("write_file", metadata={"path": "tests/test_expr.py"}),
            _completed("verify_work", metadata={"command": "go test ./..."}),
        ],
    )

    assert result.can_finish is False
    assert "declared no-network task image" in result.reason
    assert "public.example/task:latest" in result.reason
    assert "verify_work with the Docker command itself" in result.reason
    assert "local-only verify_work" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_verifier_accepts_public_no_network_image_check(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M ast/expr.go\n M tests/test_expr.py\n",
    )
    verifier = ExternalWorkspaceVerifier(
        env,
        workdir=str(tmp_path),
        policy=ExternalWorkspacePolicy(
            required_no_network_verify_image="public.example/task:latest"
        ),
    )

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "ast/expr.go"}),
            _completed("write_file", metadata={"path": "tests/test_expr.py"}),
            _completed(
                "verify_work",
                metadata={
                    "command": (
                        "docker run --rm --network none -v $PWD:/app -w /app "
                        "public.example/task:latest pytest tests/test_expr.py"
                    )
                },
            ),
        ],
    )

    assert result.can_finish is True
    assert verifier.latest.verification_passed_after_source_change is True


@pytest.mark.asyncio
async def test_external_workspace_verifier_invalidates_verify_after_later_mutation(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M ast/expr.go\n M tests/test_expr.py\n",
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "ast/expr.go"}),
            _completed("write_file", metadata={"path": "tests/test_expr.py"}),
            _completed("verify_work", metadata={"command": "pytest tests/test_expr.py"}),
            _completed("shell", metadata={"workspace_changed": True, "exit_code": 0}),
        ],
    )

    assert result.can_finish is False
    assert verifier.latest.verification_passed_after_source_change is False
    assert "did not produce a later passing" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_verifier_treats_failed_shell_mutation_as_final_change(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M ast/expr.go\n M tests/test_expr.py\n",
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "ast/expr.go"}),
            _completed("write_file", metadata={"path": "tests/test_expr.py"}),
            _completed("verify_work", metadata={"command": "pytest tests/test_expr.py"}),
            _completed(
                "shell",
                is_error=True,
                metadata={"workspace_changed": True, "exit_code": 1},
            ),
        ],
    )

    assert result.can_finish is False
    assert verifier.latest.verification_passed_after_source_change is False
    assert "did not produce a later passing" in result.reason


@pytest.mark.asyncio
async def test_external_workspace_verifier_rejects_verify_work_that_changed_workspace(
    tmp_path: Path,
) -> None:
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[],
        baseline_status=" M ast/expr.go\n M tests/test_expr.py\n",
    )
    verifier = ExternalWorkspaceVerifier(env, workdir=str(tmp_path))

    result = await verifier.verify(
        session=SimpleNamespace(),
        activity=[
            _completed("write_file", metadata={"path": "ast/expr.go"}),
            _completed("write_file", metadata={"path": "tests/test_expr.py"}),
            _completed(
                "verify_work",
                metadata={"command": "pytest tests/test_expr.py", "workspace_changed": True},
            ),
        ],
    )

    assert result.can_finish is False
    assert verifier.latest.verification_passed_after_source_change is False
    assert "did not produce a later passing" in result.reason


@pytest.mark.asyncio
async def test_external_environment_runner_delegates_to_normal_harness_run_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_once(**kwargs: object) -> str:
        prompt = kwargs["prompt"]
        assert isinstance(prompt, str)
        assert prompt == "fix the task"
        assert "add or update focused" not in prompt
        assert "place tests" not in prompt
        assert "minimal changes" not in prompt
        assert "non-test files" not in prompt
        assert "do not modify" not in prompt.lower()
        assert "follow these steps" not in prompt.lower()
        render = kwargs["render"]
        assert callable(render)
        render(Done(final_message=Message(role="assistant", content="done")))
        build_verifier = kwargs["build_verifier"]
        verifier = build_verifier("external-workspace")
        result = await verifier.verify(
            session=SimpleNamespace(),
            activity=[
                _completed("write_file", metadata={"path": "ast/expr.go"}),
                _completed("write_file", metadata={"path": "tests/test_expr.py"}),
                _completed("verify_work", metadata={"command": "pytest tests/test_expr.py"}),
            ],
        )
        assert result.can_finish is True
        return "done"

    monkeypatch.setattr("harness.cli.external_workspace._harness_run_once", fake_run_once)
    logs_dir = tmp_path / "logs"
    env = StaticStatusEnvironment(
        tmp_path,
        statuses=[" M ast/expr.go\n M tests/test_expr.py\n"],
    )
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    await run_harness_on_external_environment(
        instruction="fix the task",
        environment=env,
        context=context,
        logs_dir=str(logs_dir),
        source_change_retries=0,
        verification_retries=0,
    )

    assert context.metadata["runtime"] == "harness.run_once"
    assert context.metadata["source_change_passed"] is True
    assert context.metadata["verification_passed_after_source_change"] is True
    assert (logs_dir / "harness-events.jsonl").is_file()


@pytest.mark.asyncio
async def test_external_environment_runner_disables_host_project_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_agent_kwargs: dict[str, object] = {}

    def fake_runtime_build_agent(**kwargs: object) -> object:
        observed_agent_kwargs.update(kwargs)
        return object()

    async def fake_run_once(**kwargs: object) -> str:
        build_agent = kwargs["build_agent"]
        assert callable(build_agent)
        build_agent(
            chain=["openrouter"],
            base_url=None,
            model="test-model",
            storage=object(),
            cwd=tmp_path,
            config=object(),
            yes=True,
            build_tools=kwargs["build_tools"],
        )
        render = kwargs["render"]
        assert callable(render)
        render(Done(final_message=Message(role="assistant", content="done")))
        return "done"

    monkeypatch.setattr("harness.cli.external_workspace._harness_run_once", fake_run_once)
    monkeypatch.setattr(
        "harness.cli.external_workspace._runtime_build_agent", fake_runtime_build_agent
    )
    env = StaticStatusEnvironment(tmp_path, statuses=[""])
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    await run_harness_on_external_environment(
        instruction="fix the task",
        environment=env,
        context=context,
        logs_dir=str(tmp_path / "logs"),
        source_change_retries=0,
        verification_retries=0,
        fail_without_source_change=False,
        fail_without_verification=False,
        require_regression_test_change=False,
    )

    assert observed_agent_kwargs["project_context_enabled"] is False
    assert observed_agent_kwargs["auxiliary_tools_enabled"] is False
    assert observed_agent_kwargs["memory_tools_enabled"] is False


@pytest.mark.asyncio
async def test_external_environment_runner_blocks_in_workspace_harness_logs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observations: dict[str, ToolResult] = {}

    async def fake_run_once(**kwargs: object) -> str:
        registry = kwargs["build_tools"](tmp_path)  # type: ignore[index,operator]
        observations["read"] = await registry.get("read_file")(
            _call("read_file", path="harness-logs/harness-events.jsonl")
        )
        observations["list"] = await registry.get("list_dir")(_call("list_dir", path="."))
        observations["shell"] = await registry.get("shell")(
            _call("shell", command="find . -maxdepth 2 -type f | sort")
        )
        render = kwargs["render"]  # type: ignore[index]
        assert callable(render)
        render(Done(final_message=Message(role="assistant", content="done")))
        return "done"

    monkeypatch.setattr("harness.cli.external_workspace._harness_run_once", fake_run_once)
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    logs_dir = tmp_path / "harness-logs"
    logs_dir.mkdir()
    (logs_dir / "harness-events.jsonl").write_text("private verifier command\n", encoding="utf-8")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    await run_harness_on_external_environment(
        instruction="inspect the task",
        environment=env,
        context=context,
        logs_dir=str(logs_dir),
        source_change_retries=0,
        verification_retries=0,
        fail_without_source_change=False,
        fail_without_verification=False,
        require_regression_test_change=False,
    )

    assert observations["read"].is_error is True
    assert observations["list"].is_error is False
    assert observations["shell"].is_error is False
    assert "harness-logs" not in observations["list"].content
    assert "harness-logs" not in observations["shell"].content
    assert "harness-events.jsonl" not in observations["shell"].content
    assert "src/app.py" in observations["shell"].content


@pytest.mark.asyncio
async def test_external_environment_runner_uses_normal_harness_runtime_with_remote_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src",
                "write_file",
                {"path": "src/app.py", "content": "ok\n"},
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_app.py", "content": "def test_app(): pass\n"},
            ),
            _scripted_tool_turn("verify", "verify_work", {}),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    await LocalEnvironment(tmp_path).exec("git init")
    logs_dir = tmp_path / "logs"
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    await run_harness_on_external_environment(
        instruction="fix the task",
        environment=LocalEnvironment(tmp_path),
        context=context,
        logs_dir=str(logs_dir),
        source_change_retries=0,
        verification_retries=0,
        pass_timeout_seconds=20,
        default_verify_command="pytest tests/test_app.py",
    )

    assert (tmp_path / "src/app.py").read_text(encoding="utf-8") == "ok\n"
    assert (tmp_path / "tests/test_app.py").is_file()
    assert context.metadata["runtime"] == "harness.run_once"
    assert context.metadata["source_change_paths"] == ["src/app.py"]
    assert context.metadata["test_change_paths"] == ["tests/test_app.py"]
    assert context.metadata["latest_verification_command"] == "pytest tests/test_app.py"
    assert context.metadata["verification_passed_after_source_change"] is True


@pytest.mark.asyncio
async def test_external_environment_runner_can_widen_model_timeouts_for_long_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, str | None] = {}

    async def fake_run_once(**kwargs: object) -> str:
        observed["idle"] = os.environ.get("HARNESS_MODEL_STREAM_IDLE_TIMEOUT")
        observed["turn"] = os.environ.get("HARNESS_MODEL_TURN_TIMEOUT")
        render = kwargs["render"]  # type: ignore[index]
        assert callable(render)
        render(Done(final_message=Message(role="assistant", content="done")))
        return "done"

    monkeypatch.setenv("HARNESS_MODEL_STREAM_IDLE_TIMEOUT", "45")
    monkeypatch.delenv("HARNESS_MODEL_TURN_TIMEOUT", raising=False)
    monkeypatch.setattr("harness.cli.external_workspace._harness_run_once", fake_run_once)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    await run_harness_on_external_environment(
        instruction="fix the task",
        environment=LocalEnvironment(tmp_path),
        context=context,
        logs_dir=str(tmp_path / "logs"),
        source_change_retries=0,
        verification_retries=0,
        fail_without_source_change=False,
        fail_without_verification=False,
        require_regression_test_change=False,
        model_stream_idle_timeout_seconds=180,
        model_turn_timeout_seconds=240,
    )

    assert observed == {"idle": "180", "turn": "240"}
    assert os.environ["HARNESS_MODEL_STREAM_IDLE_TIMEOUT"] == "45"
    assert "HARNESS_MODEL_TURN_TIMEOUT" not in os.environ
    assert context.metadata["model_stream_idle_timeout_seconds"] == 180
    assert context.metadata["model_turn_timeout_seconds"] == 240


@pytest.mark.asyncio
async def test_external_environment_runner_auto_runs_default_verify_after_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src",
                "write_file",
                {"path": "src/app.py", "content": "ok\n"},
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_app.py", "content": "def test_app(): pass\n"},
            ),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    await run_harness_on_external_environment(
        instruction="fix the task",
        environment=LocalEnvironment(tmp_path),
        context=context,
        logs_dir=str(tmp_path / "logs"),
        source_change_retries=0,
        verification_retries=0,
        pass_timeout_seconds=20,
        default_verify_command="pytest tests/test_app.py",
    )

    assert context.metadata["latest_verification_command"] == "pytest tests/test_app.py"
    assert context.metadata["verification_passed_after_source_change"] is True
    assert "✓ verify_work: PASSED" in (tmp_path / "logs" / "harness.txt").read_text(
        encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_external_environment_runner_rejects_bypassing_default_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src",
                "write_file",
                {"path": "src/app.py", "content": "ok\n"},
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_app.py", "content": "def test_app(): pass\n"},
            ),
            _scripted_tool_turn("verify", "verify_work", {"command": "pytest"}),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="configured default verifier"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
            require_regression_test_change=False,
            default_verify_command="pytest",
        )

    assert context.metadata["latest_verification_command"] == "pytest"
    assert context.metadata["latest_verification_used_default_command"] is False
    assert context.metadata["verification_passed_after_source_change"] is False


@pytest.mark.asyncio
async def test_external_environment_runner_preserves_verification_snapshot_after_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_once(**kwargs: object) -> str:
        render = kwargs["render"]
        build_verifier = kwargs["build_verifier"]
        assert callable(render)
        assert callable(build_verifier)
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "src" / "app.py").write_text("VALUE = 'fixed'\n", encoding="utf-8")
        (tmp_path / "tests" / "test_app.py").write_text(
            "def test_app():\n    assert True\n",
            encoding="utf-8",
        )
        verifier = build_verifier()
        result = await verifier.verify(
            session=SimpleNamespace(messages=[]),
            activity=[
                _completed("write_file", metadata={"path": "src/app.py"}),
                _completed("write_file", metadata={"path": "tests/test_app.py"}),
                _completed(
                    "verify_work",
                    metadata={
                        "command": "python -m pytest tests/test_app.py",
                        "exit_code": 0,
                    },
                    content_preview="PASSED\n\nstdout:\n1 passed\n",
                ),
            ],
        )
        assert result.can_finish is True
        render(
            ErrorEvent(
                kind="internal",
                error="model died after verified work",
                recoverable=False,
            )
        )
        return ""

    monkeypatch.setattr("harness.cli.external_workspace._harness_run_once", fake_run_once)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="model died after verified work"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["source_change_paths"] == ["src/app.py"]
    assert context.metadata["test_change_paths"] == ["tests/test_app.py"]
    assert context.metadata["latest_verification_command"] == ("python -m pytest tests/test_app.py")
    assert context.metadata["verification_passed_after_source_change"] is True
    assert context.metadata["verification_error"] is None


@pytest.mark.asyncio
async def test_external_environment_runner_preserves_completion_verifier_rejection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = CoverageReviewAdapter(
        {
            "can_finish": False,
            "reason": "changed tests do not exercise the requested syntax",
            "confidence": 0.91,
        }
    )
    monkeypatch.setenv("HARNESS_EXTERNAL_WORKSPACE_COVERAGE_REVIEW", "1")
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)

    async def fake_run_once(**kwargs: object) -> str:
        render = kwargs["render"]
        build_verifier = kwargs["build_verifier"]
        assert callable(render)
        assert callable(build_verifier)
        (tmp_path / "src").mkdir()
        (tmp_path / "tests").mkdir()
        (tmp_path / "src" / "app.py").write_text("VALUE = 'fixed'\n", encoding="utf-8")
        (tmp_path / "tests" / "test_app.py").write_text(
            "def test_app():\n    assert True\n",
            encoding="utf-8",
        )
        verifier = build_verifier()
        result = await verifier.verify(
            session=SimpleNamespace(
                messages=[Message(role="assistant", content="implemented the syntax")]
            ),
            activity=[
                _completed("write_file", metadata={"path": "src/app.py"}),
                _completed("write_file", metadata={"path": "tests/test_app.py"}),
                _completed(
                    "verify_work",
                    metadata={
                        "command": "python -m pytest tests/test_app.py",
                        "exit_code": 0,
                    },
                    content_preview="PASSED\n\nstdout:\n1 passed\n",
                ),
            ],
        )
        assert result.can_finish is False
        render(Verification(result=result))
        raise typer.Exit(2)

    monkeypatch.setattr("harness.cli.external_workspace._harness_run_once", fake_run_once)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="Regression coverage is too weak"):
        await run_harness_on_external_environment(
            instruction="add support for default argument syntax",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["source_change_paths"] == ["src/app.py"]
    assert context.metadata["test_change_paths"] == ["tests/test_app.py"]
    assert context.metadata["latest_verification_command"] == ("python -m pytest tests/test_app.py")
    assert context.metadata["verification_passed_after_source_change"] is True
    assert context.metadata["verification_error"] is None
    assert context.metadata["run_error"] == "harness run exited with 2"
    assert context.metadata["completion_verification_can_finish"] is False
    assert context.metadata["completion_verifier_name"] == "chained"
    assert "Regression coverage is too weak" in str(
        context.metadata["completion_verification_reason"]
    )
    assert (
        context.metadata["completion_verification_error"]
        == context.metadata["completion_verification_reason"]
    )


@pytest.mark.asyncio
async def test_external_environment_runner_repairs_after_failed_real_verify_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_content = (
        "import sys\n"
        "from pathlib import Path\n\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n\n"
        "from mathlib import normalize_name\n\n\n"
        "def test_normalize_name_trims_and_lowercases():\n"
        "    assert normalize_name('  Ada Lovelace  ') == 'ada lovelace'\n"
    )
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src_bad",
                "write_file",
                {
                    "path": "src/mathlib.py",
                    "content": "def normalize_name(value):\n    return value\n",
                },
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_mathlib.py", "content": test_content},
            ),
            _scripted_tool_turn(
                "verify_fails",
                "verify_work",
                {"command": "python -m pytest tests/test_mathlib.py"},
            ),
            _scripted_tool_turn(
                "repair_src",
                "edit_file",
                {
                    "path": "src/mathlib.py",
                    "old": "    return value\n",
                    "new": "    return value.strip().lower()\n",
                },
            ),
            _scripted_tool_turn(
                "verify_passes",
                "verify_work",
                {"command": "python -m pytest tests/test_mathlib.py"},
            ),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    await LocalEnvironment(tmp_path).exec("git init")
    logs_dir = tmp_path / "logs"
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    await run_harness_on_external_environment(
        instruction=(
            "normalize_name should trim surrounding whitespace and compare names case-insensitively"
        ),
        environment=LocalEnvironment(tmp_path),
        context=context,
        logs_dir=str(logs_dir),
        source_change_retries=1,
        verification_retries=1,
        pass_timeout_seconds=20,
    )

    assert (tmp_path / "src/mathlib.py").read_text(encoding="utf-8") == (
        "def normalize_name(value):\n    return value.strip().lower()\n"
    )
    log_text = (logs_dir / "harness.txt").read_text(encoding="utf-8")
    assert "FAILED" in log_text
    assert "assert '  Ada Lovelace  ' == 'ada lovelace'" in log_text
    assert "1 passed" in log_text
    assert context.metadata["source_change_paths"] == ["src/mathlib.py"]
    assert context.metadata["test_change_paths"] == ["tests/test_mathlib.py"]
    assert context.metadata["latest_verification_command"] == (
        "python -m pytest tests/test_mathlib.py"
    )
    assert context.metadata["verification_passed_after_source_change"] is True


@pytest.mark.asyncio
async def test_external_environment_runner_repairs_after_dependency_setup_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_content = (
        "import sys\n"
        "from pathlib import Path\n\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n\n"
        "from mathlib import normalize_name\n\n\n"
        "def test_normalize_name_trims():\n"
        "    assert normalize_name('  Ada  ') == 'Ada'\n"
    )
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src_bad",
                "write_file",
                {
                    "path": "src/mathlib.py",
                    "content": "def normalize_name(value):\n    return value\n",
                },
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_mathlib.py", "content": test_content},
            ),
            _scripted_tool_turn(
                "verify_fails",
                "verify_work",
                {"command": "python -m pytest tests/test_mathlib.py"},
            ),
            [
                Done(
                    final_message=Message(
                        role="assistant",
                        content=(
                            "I cannot complete this because a missing dependency tool "
                            "is not available. If you can provide or install it, I can "
                            "continue."
                        ),
                    )
                )
            ],
            _scripted_tool_turn(
                "still_bad_once",
                "edit_file",
                {
                    "path": "src/mathlib.py",
                    "old": "    return value\n",
                    "new": '    return value + ""\n',
                },
            ),
            _scripted_tool_turn(
                "verify_fails_again",
                "verify_work",
                {"command": "python -m pytest tests/test_mathlib.py"},
            ),
            [
                Done(
                    final_message=Message(
                        role="assistant",
                        content=(
                            "I am blocked because the dependency generator is unavailable. "
                            "Please install or provide that tool so I can continue."
                        ),
                    )
                )
            ],
            _scripted_tool_turn(
                "still_bad_twice",
                "edit_file",
                {
                    "path": "src/mathlib.py",
                    "old": '    return value + ""\n',
                    "new": "    return str(value)\n",
                },
            ),
            _scripted_tool_turn(
                "verify_fails_third",
                "verify_work",
                {"command": "python -m pytest tests/test_mathlib.py"},
            ),
            [
                Done(
                    final_message=Message(
                        role="assistant",
                        content=(
                            "I still cannot complete it because the generator binary is missing. "
                            "If you can enable or install it, I can proceed."
                        ),
                    )
                )
            ],
            _scripted_tool_turn(
                "still_bad_thrice",
                "edit_file",
                {
                    "path": "src/mathlib.py",
                    "old": "    return str(value)\n",
                    "new": '    return value.replace("Ada", "Ada")\n',
                },
            ),
            _scripted_tool_turn(
                "verify_fails_fourth",
                "verify_work",
                {"command": "python -m pytest tests/test_mathlib.py"},
            ),
            [
                Done(
                    final_message=Message(
                        role="assistant",
                        content=(
                            "I cannot proceed because the local compiler tool is missing. "
                            "Tell me how to install or configure it."
                        ),
                    )
                )
            ],
            _scripted_tool_turn(
                "repair_src",
                "edit_file",
                {
                    "path": "src/mathlib.py",
                    "old": '    return value.replace("Ada", "Ada")\n',
                    "new": "    return value.strip()\n",
                },
            ),
            _scripted_tool_turn(
                "verify_passes",
                "verify_work",
                {"command": "python -m pytest tests/test_mathlib.py"},
            ),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    await LocalEnvironment(tmp_path).exec("git init")
    logs_dir = tmp_path / "logs"
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    await run_harness_on_external_environment(
        instruction="normalize_name should trim surrounding whitespace",
        environment=LocalEnvironment(tmp_path),
        context=context,
        logs_dir=str(logs_dir),
        source_change_retries=1,
        verification_retries=1,
        pass_timeout_seconds=20,
    )

    assert (tmp_path / "src/mathlib.py").read_text(encoding="utf-8") == (
        "def normalize_name(value):\n    return value.strip()\n"
    )
    event_log = (logs_dir / "harness-events.jsonl").read_text(encoding="utf-8")
    assert "defers repository, source, environment, or dependency setup" in event_log
    assert context.metadata["verification_passed_after_source_change"] is True
    assert context.metadata["model_fallback_used"] is False
    assert adapter.scripts == []


@pytest.mark.asyncio
async def test_external_environment_runner_reserves_budget_for_long_dependency_repair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    test_content = (
        "import sys\n"
        "from pathlib import Path\n\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n\n"
        "from mathlib import normalize_name\n\n\n"
        "def test_normalize_name_trims():\n"
        "    assert normalize_name('  Ada  ') == 'Ada'\n"
    )
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            [Done(final_message=Message(role="assistant", content="I am still inspecting."))],
            [Done(final_message=Message(role="assistant", content="I need another probe."))],
            _scripted_tool_turn(
                "write_src_bad",
                "write_file",
                {
                    "path": "src/mathlib.py",
                    "content": "def normalize_name(value):\n    return value\n",
                },
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_mathlib.py", "content": test_content},
            ),
            _scripted_tool_turn(
                "verify_fails",
                "verify_work",
                {"command": "python -m pytest tests/test_mathlib.py"},
            ),
            [
                Done(
                    final_message=Message(
                        role="assistant",
                        content=(
                            "I am blocked because the required generator tool is missing. "
                            "If you can install or provide it, I can continue."
                        ),
                    )
                )
            ],
            _scripted_tool_turn(
                "repair_src",
                "edit_file",
                {
                    "path": "src/mathlib.py",
                    "old": "    return value\n",
                    "new": "    return value.strip()\n",
                },
            ),
            _scripted_tool_turn(
                "verify_passes",
                "verify_work",
                {"command": "python -m pytest tests/test_mathlib.py"},
            ),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    await LocalEnvironment(tmp_path).exec("git init")
    logs_dir = tmp_path / "logs"
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    await run_harness_on_external_environment(
        instruction="normalize_name should trim surrounding whitespace",
        environment=LocalEnvironment(tmp_path),
        context=context,
        logs_dir=str(logs_dir),
        source_change_retries=1,
        verification_retries=1,
        pass_timeout_seconds=20,
    )

    event_log = (logs_dir / "harness-events.jsonl").read_text(encoding="utf-8")
    assert event_log.count("defers repository, source, environment, or dependency setup") >= 4
    assert context.metadata["repair_attempt_budget"] == 10
    assert context.metadata["verification_passed_after_source_change"] is True
    assert adapter.scripts == []


@pytest.mark.asyncio
async def test_external_environment_runner_rejects_claimed_completion_without_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            [Done(final_message=Message(role="assistant", content="I fixed it."))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="implementation/source change"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["source_change_passed"] is False
    assert context.metadata["verification_passed_after_source_change"] is False


@pytest.mark.asyncio
async def test_external_environment_runner_preserves_runtime_error_when_source_gate_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_EXTERNAL_WORKSPACE_MODEL_FALLBACKS", raising=False)
    monkeypatch.delenv("HARNESS_OPENROUTER_MODEL_FALLBACKS", raising=False)

    async def fake_run_once(**kwargs: object) -> str:
        render = kwargs["render"]
        assert callable(render)
        render(
            ErrorEvent(
                kind="rate_limit",
                error="OpenRouter rate-limited (429)",
                recoverable=True,
            )
        )
        return ""

    monkeypatch.setattr("harness.cli.external_workspace._harness_run_once", fake_run_once)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError) as excinfo:
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    message = str(excinfo.value)
    assert "rate_limit: OpenRouter rate-limited (429)" in message
    assert "implementation/source change" in message
    assert context.metadata["latest_runtime_error_kind"] == "rate_limit"
    assert context.metadata["latest_runtime_error"] == "OpenRouter rate-limited (429)"
    assert context.metadata["source_change_passed"] is False


def test_external_workspace_model_candidates_skip_excluded_qwen_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "HARNESS_EXTERNAL_WORKSPACE_MODEL_FALLBACKS",
        "qwen/qwen3-coder, openai/gpt-4.1-mini",
    )
    monkeypatch.setenv("HARNESS_EXTERNAL_WORKSPACE_MODEL_FALLBACK_LIMIT", "2")

    assert _external_workspace_model_candidates("openai/gpt-5.4-nano") == [
        "openai/gpt-5.4-nano",
        "openai/gpt-4.1-mini",
    ]


@pytest.mark.asyncio
async def test_external_environment_runner_retries_runtime_failure_with_fallback_before_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_EXTERNAL_WORKSPACE_MODEL_FALLBACKS", "openai/gpt-4.1-mini")
    monkeypatch.setenv("HARNESS_EXTERNAL_WORKSPACE_MODEL_FALLBACK_LIMIT", "1")
    calls: list[str] = []

    async def fake_run_once(**kwargs: object) -> str:
        calls.append(str(kwargs["model"]))
        render = kwargs["render"]
        assert callable(render)
        if len(calls) == 1:
            render(
                ErrorEvent(
                    kind="internal",
                    error="adapter returned empty final response after tool evidence",
                    recoverable=True,
                )
            )
            return ""

        (tmp_path / "src").mkdir(exist_ok=True)
        (tmp_path / "src" / "app.py").write_text("print('fixed')\n", encoding="utf-8")
        verifier = kwargs["build_verifier"]()
        await verifier.verify(session=SimpleNamespace(), activity=[])
        render(Done(final_message=Message(role="assistant", content="done")))
        return "done"

    monkeypatch.setattr("harness.cli.external_workspace._harness_run_once", fake_run_once)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    await run_harness_on_external_environment(
        instruction="fix the task",
        environment=LocalEnvironment(tmp_path),
        context=context,
        logs_dir=str(tmp_path / "logs"),
        model_name="google/gemma-4-31b-it",
        source_change_retries=0,
        verification_retries=0,
        pass_timeout_seconds=20,
        fail_without_verification=False,
        require_regression_test_change=False,
    )

    assert calls == ["google/gemma-4-31b-it", "openai/gpt-4.1-mini"]
    assert context.metadata["requested_model"] == "google/gemma-4-31b-it"
    assert context.metadata["attempt_model"] == "openai/gpt-4.1-mini"
    assert context.metadata["model_fallback_used"] is True
    assert context.metadata["source_change_passed"] is True
    assert context.metadata["source_change_paths"] == ["src/app.py"]
    assert context.metadata["model_attempts"][0]["run_error"] == (
        "internal: adapter returned empty final response after tool evidence"
    )
    assert context.metadata["model_attempts"][1]["model"] == "openai/gpt-4.1-mini"
    log_text = (tmp_path / "logs" / "harness.txt").read_text(encoding="utf-8")
    assert "retrying external workspace run with fallback model openai/gpt-4.1-mini" in log_text


@pytest.mark.asyncio
async def test_external_environment_runner_does_not_retry_nonrecoverable_internal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_EXTERNAL_WORKSPACE_MODEL_FALLBACKS", "openai/gpt-4.1-mini")
    monkeypatch.setenv("HARNESS_EXTERNAL_WORKSPACE_MODEL_FALLBACK_LIMIT", "1")
    calls: list[str] = []

    async def fake_run_once(**kwargs: object) -> str:
        calls.append(str(kwargs["model"]))
        render = kwargs["render"]
        assert callable(render)
        render(
            ErrorEvent(
                kind="internal",
                error="exceeded max_steps=5 without final answer",
                recoverable=False,
            )
        )
        return ""

    monkeypatch.setattr("harness.cli.external_workspace._harness_run_once", fake_run_once)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="exceeded max_steps=5 without final answer"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            model_name="google/gemma-4-31b-it",
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
            fail_without_verification=False,
            require_regression_test_change=False,
        )

    assert calls == ["google/gemma-4-31b-it"]
    assert context.metadata["model_fallback_used"] is False
    assert context.metadata["latest_runtime_error_kind"] == "internal"
    assert context.metadata["latest_runtime_error_recoverable"] is False
    log_text = (tmp_path / "logs" / "harness.txt").read_text(encoding="utf-8")
    assert "retrying external workspace run with fallback model" not in log_text


@pytest.mark.asyncio
async def test_external_environment_runner_retries_runtime_failure_after_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_EXTERNAL_WORKSPACE_MODEL_FALLBACKS", "openai/gpt-4.1-mini")
    monkeypatch.setenv("HARNESS_EXTERNAL_WORKSPACE_MODEL_FALLBACK_LIMIT", "1")
    calls: list[str] = []

    async def fake_run_once(**kwargs: object) -> str:
        calls.append(str(kwargs["model"]))
        render = kwargs["render"]
        assert callable(render)
        (tmp_path / "src").mkdir(exist_ok=True)
        if len(calls) == 1:
            (tmp_path / "src" / "app.py").write_text("print('partial')\n", encoding="utf-8")
            render(
                ErrorEvent(
                    kind="timeout",
                    error="model stream produced no events for 120.0s",
                    recoverable=True,
                )
            )
            return ""

        (tmp_path / "src" / "app.py").write_text("print('fixed')\n", encoding="utf-8")
        verifier = kwargs["build_verifier"]()
        await verifier.verify(session=SimpleNamespace(), activity=[])
        render(Done(final_message=Message(role="assistant", content="done")))
        return "done"

    monkeypatch.setattr("harness.cli.external_workspace._harness_run_once", fake_run_once)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    await run_harness_on_external_environment(
        instruction="fix the task",
        environment=LocalEnvironment(tmp_path),
        context=context,
        logs_dir=str(tmp_path / "logs"),
        model_name="google/gemma-4-31b-it",
        source_change_retries=0,
        verification_retries=0,
        pass_timeout_seconds=20,
        fail_without_verification=False,
        require_regression_test_change=False,
    )

    assert calls == ["google/gemma-4-31b-it", "openai/gpt-4.1-mini"]
    assert context.metadata["attempt_model"] == "openai/gpt-4.1-mini"
    assert context.metadata["model_fallback_used"] is True
    assert context.metadata["source_change_passed"] is True
    assert context.metadata["source_change_paths"] == ["src/app.py"]
    assert context.metadata["model_attempts"][0]["latest_runtime_error_kind"] == "timeout"
    assert context.metadata["model_attempts"][0]["source_change_paths"] == ["src/app.py"]
    log_text = (tmp_path / "logs" / "harness.txt").read_text(encoding="utf-8")
    assert "continuing from current workspace" in log_text


@pytest.mark.asyncio
async def test_external_environment_runner_refreshes_stale_snapshot_after_runtime_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_EXTERNAL_WORKSPACE_MODEL_FALLBACKS", raising=False)
    monkeypatch.delenv("HARNESS_OPENROUTER_MODEL_FALLBACKS", raising=False)

    async def fake_run_once(**kwargs: object) -> str:
        verifier = kwargs["build_verifier"]()
        await verifier.verify(
            session=SimpleNamespace(
                messages=[
                    Message(
                        role="assistant",
                        content=(
                            "I can't continue because the source tree is missing. "
                            "Please provide the repository files."
                        ),
                    )
                ]
            ),
            activity=[],
        )
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "app.py").write_text("print('partial')\n", encoding="utf-8")
        render = kwargs["render"]
        assert callable(render)
        render(
            ErrorEvent(
                kind="timeout",
                error="model stream produced no events for 120.0s",
                recoverable=False,
            )
        )
        return ""

    monkeypatch.setattr("harness.cli.external_workspace._harness_run_once", fake_run_once)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="did not produce a later passing"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
            require_regression_test_change=False,
        )

    assert context.metadata["latest_runtime_error_kind"] == "timeout"
    assert context.metadata["source_change_paths"] == ["src/app.py"]
    assert context.metadata["model_attempts"][0]["source_change_paths"] == ["src/app.py"]
    assert "did not produce a later passing" in context.metadata["verification_error"]


@pytest.mark.asyncio
async def test_external_environment_runner_does_not_retry_after_scratch_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_EXTERNAL_WORKSPACE_MODEL_FALLBACKS", "openai/gpt-4.1-mini")
    calls: list[str] = []

    async def fake_run_once(**kwargs: object) -> str:
        calls.append(str(kwargs["model"]))
        (tmp_path / "notes.md").write_text("scratch\n", encoding="utf-8")
        render = kwargs["render"]
        assert callable(render)
        render(
            ErrorEvent(
                kind="internal",
                error="model stopped before completion",
                recoverable=True,
            )
        )
        return ""

    monkeypatch.setattr("harness.cli.external_workspace._harness_run_once", fake_run_once)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="model stopped before completion"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            model_name="google/gemma-4-31b-it",
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert calls == ["google/gemma-4-31b-it"]
    assert context.metadata["model_fallback_used"] is False
    assert context.metadata["scratch_paths"] == ["notes.md"]


@pytest.mark.asyncio
async def test_external_environment_runner_records_effective_fallback_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_run_once(**kwargs: object) -> str:
        render = kwargs["render"]
        assert callable(render)
        render(
            ModelSelectedEvent(
                provider="openrouter",
                requested_model="harness/definitely-unavailable-model",
                model="google/gemma-4-31b-it",
                fallback=True,
                attempt=1,
            )
        )
        render(Done(final_message=Message(role="assistant", content="done")))
        return "done"

    monkeypatch.setattr("harness.cli.external_workspace._harness_run_once", fake_run_once)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    await run_harness_on_external_environment(
        instruction="fix the task",
        environment=LocalEnvironment(tmp_path),
        context=context,
        logs_dir=str(tmp_path / "logs"),
        model_name="harness/definitely-unavailable-model",
        fail_without_source_change=False,
        fail_without_verification=False,
        require_regression_test_change=False,
    )

    assert context.metadata["requested_model"] == "harness/definitely-unavailable-model"
    assert context.metadata["effective_model"] == "google/gemma-4-31b-it"
    assert context.metadata["model"] == "google/gemma-4-31b-it"
    assert context.metadata["model_fallback_used"] is True
    assert context.metadata["model_selection"] == {
        "provider": "openrouter",
        "requested_model": "harness/definitely-unavailable-model",
        "model": "google/gemma-4-31b-it",
        "fallback": True,
        "attempt": 1,
    }


@pytest.mark.asyncio
async def test_external_environment_runner_rejects_source_and_test_without_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src",
                "write_file",
                {"path": "src/app.py", "content": "ok\n"},
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_app.py", "content": "def test_app(): pass\n"},
            ),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="did not produce a later passing"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["source_change_paths"] == ["src/app.py"]
    assert context.metadata["test_change_paths"] == ["tests/test_app.py"]
    assert context.metadata["verification_passed_after_source_change"] is False


@pytest.mark.asyncio
async def test_external_environment_runner_rejects_test_only_change_with_passing_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_app.py", "content": "def test_app(): pass\n"},
            ),
            _scripted_tool_turn(
                "verify",
                "verify_work",
                {"command": "pytest tests/test_app.py"},
            ),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="implementation/source change"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["source_change_passed"] is False
    assert context.metadata["test_change_paths"] == ["tests/test_app.py"]
    assert context.metadata["verification_passed_after_source_change"] is False


@pytest.mark.asyncio
async def test_external_environment_runner_rejects_deleted_test_as_regression_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_existing.py").write_text(
        "def test_existing(): pass\n",
        encoding="utf-8",
    )
    await env.exec("git add tests/test_existing.py")
    await env.exec("git -c user.name=test -c user.email=test@example.com commit -m baseline")

    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src",
                "write_file",
                {"path": "src/app.py", "content": "ok\n"},
            ),
            _scripted_tool_turn(
                "delete_test",
                "shell",
                {"command": "rm tests/test_existing.py"},
            ),
            _scripted_tool_turn("verify", "verify_work", {"command": "pytest tests"}),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="no in-repository regression test changes"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=env,
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["source_change_paths"] == ["src/app.py"]
    assert context.metadata["test_change_paths"] == []
    assert context.metadata["verification_passed_after_source_change"] is False


@pytest.mark.asyncio
async def test_external_environment_runner_rejects_deleted_source_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("VALUE = 1\n", encoding="utf-8")
    await env.exec("git add src/app.py")
    await env.exec("git -c user.name=test -c user.email=test@example.com commit -m baseline")

    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "delete_source",
                "shell",
                {"command": "rm src/app.py && mkdir -p tests"},
            ),
            _scripted_tool_turn(
                "write_src",
                "write_file",
                {"path": "src/replacement.py", "content": "VALUE = 2\n"},
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_app.py", "content": "def test_app(): pass\n"},
            ),
            _scripted_tool_turn(
                "verify",
                "verify_work",
                {"command": "python -m py_compile src/replacement.py tests/test_app.py"},
            ),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="Tracked source files were deleted"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=env,
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["source_change_paths"] == [
        "src/app.py",
        "src/replacement.py",
    ]
    assert context.metadata["deleted_source_paths"] == ["src/app.py"]
    assert context.metadata["test_change_paths"] == ["tests/test_app.py"]
    assert context.metadata["verification_passed_after_source_change"] is False


@pytest.mark.asyncio
async def test_external_environment_runner_rejects_renamed_test_as_regression_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_existing.py").write_text(
        "def test_existing(): pass\n",
        encoding="utf-8",
    )
    await env.exec("git add tests/test_existing.py")
    await env.exec("git -c user.name=test -c user.email=test@example.com commit -m baseline")

    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src",
                "write_file",
                {"path": "src/app.py", "content": "ok\n"},
            ),
            _scripted_tool_turn(
                "rename_test",
                "shell",
                {"command": "git mv tests/test_existing.py tests/test_renamed.py"},
            ),
            _scripted_tool_turn("verify", "verify_work", {"command": "pytest tests"}),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="no in-repository regression test changes"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=env,
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["source_change_paths"] == ["src/app.py"]
    assert context.metadata["test_change_paths"] == []
    assert context.metadata["verification_passed_after_source_change"] is False


@pytest.mark.asyncio
async def test_external_environment_runner_rejects_failed_verification_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src",
                "write_file",
                {"path": "src/app.py", "content": "ok\n"},
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_app.py", "content": "def test_app(): assert False\n"},
            ),
            _scripted_tool_turn(
                "verify",
                "verify_work",
                {"command": "pytest tests/test_app.py"},
            ),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="did not produce a later passing"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["source_change_paths"] == ["src/app.py"]
    assert context.metadata["test_change_paths"] == ["tests/test_app.py"]
    assert context.metadata["latest_verification_command"] == "pytest tests/test_app.py"
    assert "FAILED" in (context.metadata["latest_verification_error"] or "")
    assert context.metadata["verification_passed_after_source_change"] is False


@pytest.mark.asyncio
async def test_external_environment_runner_rejects_masked_verification_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src",
                "write_file",
                {"path": "src/app.py", "content": "ok\n"},
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_app.py", "content": "def test_app(): assert False\n"},
            ),
            _scripted_tool_turn(
                "verify",
                "verify_work",
                {"command": "pytest tests/test_app.py || true"},
            ),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="did not produce a later passing"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["latest_verification_command"] == "pytest tests/test_app.py || true"
    assert "failed assertions must return" in (context.metadata["latest_verification_error"] or "")
    assert context.metadata["verification_passed_after_source_change"] is False


@pytest.mark.asyncio
async def test_external_environment_runner_rejects_suppressed_exit_zero_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src",
                "write_file",
                {"path": "src/app.py", "content": "ok\n"},
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_app.py", "content": "def test_app(): assert False\n"},
            ),
            _scripted_tool_turn(
                "verify",
                "verify_work",
                {"command": "pytest tests/test_app.py >/dev/null 2>&1 || exit 0"},
            ),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="did not produce a later passing"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["latest_verification_command"] == (
        "pytest tests/test_app.py >/dev/null 2>&1 || exit 0"
    )
    assert "failed assertions must return" in (context.metadata["latest_verification_error"] or "")
    assert context.metadata["verification_passed_after_source_change"] is False


@pytest.mark.asyncio
async def test_external_environment_runner_rejects_verify_that_does_not_cover_changed_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src",
                "write_file",
                {"path": "src/app.py", "content": "ok\n"},
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_app.py", "content": "def test_app(): pass\n"},
            ),
            _scripted_tool_turn("verify", "verify_work", {"command": "test -f src/app.py"}),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="did not cover the changed regression tests"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["latest_verification_command"] == "test -f src/app.py"
    assert context.metadata["verification_passed_after_source_change"] is False


@pytest.mark.asyncio
async def test_external_environment_runner_rejects_file_existence_check_for_changed_tests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src",
                "write_file",
                {"path": "src/app.py", "content": "ok\n"},
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_app.py", "content": "def test_app(): pass\n"},
            ),
            _scripted_tool_turn(
                "verify",
                "verify_work",
                {"command": "test -f tests/test_app.py"},
            ),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="did not cover the changed regression tests"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["latest_verification_command"] == "test -f tests/test_app.py"
    assert context.metadata["verification_passed_after_source_change"] is False


@pytest.mark.asyncio
async def test_external_environment_runner_rejects_verification_that_ignores_changed_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_existing.py").write_text(
        "def test_existing(): pass\n",
        encoding="utf-8",
    )
    (tmp_path / "project-test").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (tmp_path / "project-test").chmod(0o755)
    await env.exec("git add project-test tests/test_existing.py")
    await env.exec("git -c user.name=test -c user.email=test@example.com commit -m baseline")

    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src",
                "write_file",
                {"path": "src/app.py", "content": "ok\n"},
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_app.py", "content": "def test_app(): assert False\n"},
            ),
            _scripted_tool_turn(
                "verify",
                "verify_work",
                {"command": "./project-test tests --ignore tests/test_app.py"},
            ),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="did not cover the changed regression tests"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=env,
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["latest_verification_command"] == (
        "./project-test tests --ignore tests/test_app.py"
    )
    assert context.metadata["verification_passed_after_source_change"] is False


@pytest.mark.asyncio
async def test_external_environment_runner_rejects_collect_only_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = LocalEnvironment(tmp_path)
    await env.exec("git init")
    (tmp_path / "project-test").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (tmp_path / "project-test").chmod(0o755)
    await env.exec("git add project-test")
    await env.exec("git -c user.name=test -c user.email=test@example.com commit -m baseline")
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src",
                "write_file",
                {"path": "src/app.py", "content": "ok\n"},
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_app.py", "content": "def test_app(): assert False\n"},
            ),
            _scripted_tool_turn(
                "verify",
                "verify_work",
                {"command": "./project-test --collect-only tests/test_app.py"},
            ),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="did not cover the changed regression tests"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=env,
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["latest_verification_command"] == (
        "./project-test --collect-only tests/test_app.py"
    )
    assert context.metadata["verification_passed_after_source_change"] is False


@pytest.mark.asyncio
async def test_external_environment_runner_rejects_verify_that_mutates_workspace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src",
                "write_file",
                {"path": "src/app.py", "content": "ok\n"},
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_app.py", "content": "def test_app(): pass\n"},
            ),
            _scripted_tool_turn(
                "verify",
                "verify_work",
                {"command": "pytest tests/test_app.py && touch src/after_verify.py"},
            ),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="did not produce a later passing"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["source_change_paths"] == ["src/after_verify.py", "src/app.py"]
    assert "verification changed workspace" in (context.metadata["latest_verification_error"] or "")
    assert context.metadata["verification_passed_after_source_change"] is False


@pytest.mark.asyncio
async def test_external_environment_runner_rejects_workspace_mutation_after_verify(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = ScriptedAdapter(
        [
            _planner_turn(),
            _scripted_tool_turn(
                "write_src",
                "write_file",
                {"path": "src/app.py", "content": "ok\n"},
            ),
            _scripted_tool_turn(
                "write_test",
                "write_file",
                {"path": "tests/test_app.py", "content": "def test_app(): pass\n"},
            ),
            _scripted_tool_turn(
                "verify",
                "verify_work",
                {"command": "pytest tests/test_app.py"},
            ),
            _scripted_tool_turn("late_change", "shell", {"command": "touch src/late.py"}),
            [Done(final_message=Message(role="assistant", content="done"))],
        ]
    )
    monkeypatch.setattr("harness.cli.external_workspace._build_adapter", lambda *_a, **_kw: adapter)
    await LocalEnvironment(tmp_path).exec("git init")
    context = SimpleNamespace(n_agent_steps=0, metadata={})

    with pytest.raises(RuntimeError, match="did not produce a later passing"):
        await run_harness_on_external_environment(
            instruction="fix the task",
            environment=LocalEnvironment(tmp_path),
            context=context,
            logs_dir=str(tmp_path / "logs"),
            source_change_retries=0,
            verification_retries=0,
            pass_timeout_seconds=20,
        )

    assert context.metadata["source_change_paths"] == ["src/app.py", "src/late.py"]
    assert context.metadata["verification_passed_after_source_change"] is False
