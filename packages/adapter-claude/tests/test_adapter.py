"""Unit tests for the Claude Code CLI adapter.

No real `claude` process is launched except in the process-group teardown test,
which uses a local stand-in script.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from harness.adapters import claude as claude_adapter
from harness.adapters.claude import ClaudeAdapter, _Decision
from harness.core import (
    ConfigurationError,
    Done,
    InternalError,
    Message,
    TextDelta,
    ToolCallEvent,
)
from harness.core.errors import TimeoutError as HarnessTimeoutError

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List a directory",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    },
]

USAGE = {
    "input_tokens": 11,
    "output_tokens": 7,
    "cache_read_input_tokens": 3289,
    "cache_creation_input_tokens": 1477,
}


def _result_line(**overrides: Any) -> bytes:
    payload: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "usage": USAGE,
        "total_cost_usd": 0.0075,
        "result": "",
    }
    payload.update(overrides)
    return json.dumps(payload).encode() + b"\n"


def _decision_line(action: str, tool_calls: list[dict[str, Any]], answer: str = "") -> bytes:
    decision = _Decision(
        action=action,  # type: ignore[arg-type]
        tool_calls_json=json.dumps(tool_calls),
        answer=answer,
    )
    return _result_line(structured_output=decision.model_dump())


def _delta_line(text: str) -> bytes:
    return (
        json.dumps(
            {
                "type": "stream_event",
                "event": {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": text},
                },
            }
        ).encode()
        + b"\n"
    )


class _FakeStdin:
    def __init__(self) -> None:
        self.data = b""
        self.closed = False

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class _FakeStdout:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    async def readline(self) -> bytes:
        if self._lines:
            return self._lines.pop(0)
        return b""


class _HangingStdout:
    async def readline(self) -> bytes:
        await asyncio.sleep(60)
        return b""


class _NoiseStdout:
    """Emits lines the adapter must not treat as visible progress."""

    async def readline(self) -> bytes:
        await asyncio.sleep(0.01)
        return json.dumps({"type": "system", "subtype": "status"}).encode() + b"\n"


class _FakeStderr:
    def __init__(self, payload: bytes = b"") -> None:
        self._payload = payload

    async def read(self) -> bytes:
        return self._payload


class _FakeProcess:
    pid = None  # Forces the non-process-group teardown path in tests.

    def __init__(self, stdout: Any, *, returncode: int = 0, stderr: bytes = b"") -> None:
        self.stdin = _FakeStdin()
        self.stdout = stdout
        self.stderr = _FakeStderr(stderr)
        self.returncode: int | None = None
        self.terminated = False
        self.killed = False
        self._final = returncode

    async def wait(self) -> int:
        if self.returncode is None:
            self.returncode = self._final
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr(
        claude_adapter,
        "inspect_claude_cli_auth",
        lambda binary=None: {"logged_in": True, "auth_method": "claude.ai"},
    )


def _patch_exec(monkeypatch: pytest.MonkeyPatch, proc: Any) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    async def _exec(*argv: str, **kwargs: Any) -> Any:
        captured["argv"] = list(argv)
        captured["kwargs"] = kwargs
        return proc

    monkeypatch.setattr("asyncio.create_subprocess_exec", _exec)
    return captured


async def _collect(iterator: Any) -> list[Any]:
    return [event async for event in iterator]


def _user(text: str = "what is here?") -> list[Message]:
    return [Message(role="system", content="policy"), Message(role="user", content=text)]


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_missing_binary(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(ConfigurationError, match="not found on PATH"):
        ClaudeAdapter()


def test_missing_login(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr(claude_adapter, "inspect_claude_cli_auth", lambda binary=None: None)
    with pytest.raises(ConfigurationError, match="claude auth login"):
        ClaudeAdapter()


async def test_capabilities(installed: None):
    capabilities = await ClaudeAdapter().capabilities()
    assert capabilities.tool_use is True
    # Harness dispatches the tools; the CLI only proposes them.
    assert capabilities.external_tools is True
    assert capabilities.structured_output is True


# --------------------------------------------------------------------------
# Decision path
# --------------------------------------------------------------------------


async def test_single_tool_call(installed: None, monkeypatch: pytest.MonkeyPatch):
    proc = _FakeProcess(
        _FakeStdout([_decision_line("tool", [{"name": "list_dir", "arguments": {"path": "."}}])])
    )
    captured = _patch_exec(monkeypatch, proc)
    events = await _collect(ClaudeAdapter().stream(model="sonnet", messages=_user(), tools=TOOLS))

    calls = [event for event in events if isinstance(event, ToolCallEvent)]
    assert [call.call.name for call in calls] == ["list_dir"]
    assert calls[0].call.arguments == {"path": "."}
    done = events[-1]
    assert isinstance(done, Done)
    assert done.final_message is not None
    assert [call.name for call in done.final_message.tool_calls or []] == ["list_dir"]
    # The transcript goes on stdin, never argv.
    assert b"what is here?" in proc.stdin.data
    assert not any("what is here?" in argument for argument in captured["argv"])


async def test_parallel_tool_calls(installed: None, monkeypatch: pytest.MonkeyPatch):
    proc = _FakeProcess(
        _FakeStdout(
            [
                _decision_line(
                    "tool",
                    [
                        {"name": "list_dir", "arguments": {"path": "."}},
                        {"name": "read_file", "arguments": {"path": "a.txt"}},
                    ],
                )
            ]
        )
    )
    _patch_exec(monkeypatch, proc)
    events = await _collect(ClaudeAdapter().stream(model="sonnet", messages=_user(), tools=TOOLS))

    done = events[-1]
    assert isinstance(done, Done)
    assert done.final_message is not None
    assert [call.name for call in done.final_message.tool_calls or []] == ["list_dir", "read_file"]
    assert len({call.id for call in done.final_message.tool_calls or []}) == 2


async def test_final_answer_is_emitted_as_a_text_delta(
    installed: None, monkeypatch: pytest.MonkeyPatch
):
    """Renderers surface assistant prose from TextDelta, never from Done.

    The decision path withholds the streamed deltas because they spell out the
    decision JSON, so without an explicit emit the answer never reaches the screen.
    """

    proc = _FakeProcess(_FakeStdout([_decision_line("final", [], answer="all done")]))
    _patch_exec(monkeypatch, proc)
    events = await _collect(ClaudeAdapter().stream(model="sonnet", messages=_user(), tools=TOOLS))

    assert [event.text for event in events if isinstance(event, TextDelta)] == ["all done"]
    done = events[-1]
    assert isinstance(done, Done)
    assert done.final_message is not None
    assert done.final_message.content == "all done"


async def test_tool_turn_preamble_is_emitted_as_a_text_delta(
    installed: None, monkeypatch: pytest.MonkeyPatch
):
    """Prose alongside proposed calls is shown too, like the codex adapter does."""

    proc = _FakeProcess(
        _FakeStdout(
            [
                _decision_line(
                    "tool",
                    [{"name": "list_dir", "arguments": {"path": "."}}],
                    answer="Let me look at the directory.",
                )
            ]
        )
    )
    _patch_exec(monkeypatch, proc)
    events = await _collect(ClaudeAdapter().stream(model="sonnet", messages=_user(), tools=TOOLS))

    assert [event.text for event in events if isinstance(event, TextDelta)] == [
        "Let me look at the directory."
    ]
    # The prose precedes the calls it introduces.
    assert isinstance(events[0], TextDelta)
    assert [event.call.name for event in events if isinstance(event, ToolCallEvent)] == ["list_dir"]


async def test_tool_call_ids_unique_across_turns(installed: None, monkeypatch: pytest.MonkeyPatch):
    """The runtime matches results against the whole transcript, not one turn."""

    adapter = ClaudeAdapter()
    seen: set[str] = set()
    for _ in range(3):
        proc = _FakeProcess(
            _FakeStdout(
                [_decision_line("tool", [{"name": "list_dir", "arguments": {"path": "."}}])]
            )
        )
        _patch_exec(monkeypatch, proc)
        events = await _collect(adapter.stream(model="sonnet", messages=_user(), tools=TOOLS))
        done = events[-1]
        assert isinstance(done, Done)
        assert done.final_message is not None
        for call in done.final_message.tool_calls or []:
            assert call.id not in seen
            seen.add(call.id)
    assert len(seen) == 3


async def test_final_answer_and_usage(installed: None, monkeypatch: pytest.MonkeyPatch):
    proc = _FakeProcess(_FakeStdout([_decision_line("final", [], answer="all done")]))
    _patch_exec(monkeypatch, proc)
    events = await _collect(ClaudeAdapter().stream(model="sonnet", messages=_user(), tools=TOOLS))

    done = events[-1]
    assert isinstance(done, Done)
    assert done.final_message is not None
    assert done.final_message.content == "all done"
    assert done.usage is not None
    assert done.usage.prompt_tokens == 11
    assert done.usage.completion_tokens == 7
    assert done.usage.total_tokens == 18
    assert done.usage.cache_read_input_tokens == 3289
    assert done.usage.cache_creation_input_tokens == 1477


async def test_result_fallback_when_structured_output_absent(
    installed: None, monkeypatch: pytest.MonkeyPatch
):
    decision = {"action": "final", "tool_calls_json": "[]", "answer": "from result"}
    proc = _FakeProcess(_FakeStdout([_result_line(result=json.dumps(decision))]))
    _patch_exec(monkeypatch, proc)
    events = await _collect(ClaudeAdapter().stream(model="sonnet", messages=_user(), tools=TOOLS))
    done = events[-1]
    assert isinstance(done, Done)
    assert done.final_message is not None
    assert done.final_message.content == "from result"


async def test_tool_outside_registry_rejected(installed: None, monkeypatch: pytest.MonkeyPatch):
    proc = _FakeProcess(_FakeStdout([_decision_line("tool", [{"name": "rm_rf", "arguments": {}}])]))
    _patch_exec(monkeypatch, proc)
    with pytest.raises(ConfigurationError, match="outside the Harness registry"):
        await _collect(ClaudeAdapter().stream(model="sonnet", messages=_user(), tools=TOOLS))


async def test_empty_tool_proposal_rejected(installed: None, monkeypatch: pytest.MonkeyPatch):
    proc = _FakeProcess(_FakeStdout([_decision_line("tool", [])]))
    _patch_exec(monkeypatch, proc)
    with pytest.raises(ConfigurationError, match="without any tool call"):
        await _collect(ClaudeAdapter().stream(model="sonnet", messages=_user(), tools=TOOLS))


async def test_isolation_flags(installed: None, monkeypatch: pytest.MonkeyPatch):
    proc = _FakeProcess(_FakeStdout([_decision_line("final", [], answer="ok")]))
    captured = _patch_exec(monkeypatch, proc)
    await _collect(ClaudeAdapter().stream(model="anthropic/sonnet", messages=_user(), tools=TOOLS))

    argv = captured["argv"]
    for flag in (
        "--safe-mode",
        "--no-chrome",
        "--disable-slash-commands",
        "--strict-mcp-config",
        "--no-session-persistence",
    ):
        assert flag in argv
    # Native tools disabled, MCP emptied, prompts never block.
    assert argv[argv.index("--tools") + 1] == ""
    assert argv[argv.index("--mcp-config") + 1] == '{"mcpServers":{}}'
    assert argv[argv.index("--permission-prompts") + 1] == "none"
    # The provider prefix is stripped for the CLI.
    assert argv[argv.index("--model") + 1] == "sonnet"
    assert captured["kwargs"]["start_new_session"] is True


# --------------------------------------------------------------------------
# Chat path
# --------------------------------------------------------------------------


async def test_chat_streams_text_deltas(installed: None, monkeypatch: pytest.MonkeyPatch):
    proc = _FakeProcess(
        _FakeStdout(
            [
                json.dumps({"type": "system", "subtype": "init"}).encode() + b"\n",
                _delta_line("he"),
                _delta_line("llo"),
                _result_line(result="hello"),
            ]
        )
    )
    captured = _patch_exec(monkeypatch, proc)
    events = await _collect(ClaudeAdapter().stream(model="sonnet", messages=_user(), tools=None))

    assert [event.text for event in events if isinstance(event, TextDelta)] == ["he", "llo"]
    done = events[-1]
    assert isinstance(done, Done)
    assert done.final_message is not None
    assert done.final_message.content == "hello"
    assert "--include-partial-messages" in captured["argv"]
    # No tools in play, so no decision schema is requested.
    assert "--json-schema" not in captured["argv"]


# --------------------------------------------------------------------------
# Failure mapping
# --------------------------------------------------------------------------


async def test_decision_deltas_never_reach_the_transcript(
    installed: None, monkeypatch: pytest.MonkeyPatch
):
    """In decision mode the streamed deltas are JSON, not prose."""

    proc = _FakeProcess(
        _FakeStdout(
            [
                _delta_line('{"action":"too'),
                _delta_line('l","tool_calls_json"'),
                _decision_line("tool", [{"name": "list_dir", "arguments": {"path": "."}}]),
            ]
        )
    )
    captured = _patch_exec(monkeypatch, proc)
    events = await _collect(ClaudeAdapter().stream(model="sonnet", messages=_user(), tools=TOOLS))

    assert not [event for event in events if isinstance(event, TextDelta)]
    assert [event.call.name for event in events if isinstance(event, ToolCallEvent)] == ["list_dir"]
    assert "--json-schema" in captured["argv"]


async def test_decision_deltas_hold_off_the_idle_deadline(
    installed: None, monkeypatch: pytest.MonkeyPatch
):
    """A long decision turn streams JSON deltas; that must count as progress."""

    class _SlowDecisionStdout:
        def __init__(self) -> None:
            self._sent = 0

        async def readline(self) -> bytes:
            await asyncio.sleep(0.06)
            self._sent += 1
            if self._sent <= 5:
                return _delta_line("x")
            if self._sent == 6:
                return _decision_line("final", [], answer="done")
            return b""

    proc = _FakeProcess(_SlowDecisionStdout())
    _patch_exec(monkeypatch, proc)
    adapter = ClaudeAdapter(timeout=30.0, idle_timeout=0.2)

    events = await _collect(adapter.stream(model="sonnet", messages=_user(), tools=TOOLS))

    done = events[-1]
    assert isinstance(done, Done)
    assert done.final_message is not None
    assert done.final_message.content == "done"


async def test_nonzero_exit(installed: None, monkeypatch: pytest.MonkeyPatch):
    proc = _FakeProcess(_FakeStdout([]), returncode=1, stderr=b"something broke")
    _patch_exec(monkeypatch, proc)
    with pytest.raises(InternalError, match="something broke"):
        await _collect(ClaudeAdapter().stream(model="sonnet", messages=_user(), tools=TOOLS))


async def test_auth_failure_is_configuration_error(
    installed: None, monkeypatch: pytest.MonkeyPatch
):
    proc = _FakeProcess(_FakeStdout([]), returncode=1, stderr=b"Please run claude auth login")
    _patch_exec(monkeypatch, proc)
    with pytest.raises(ConfigurationError, match="auth failed"):
        await _collect(ClaudeAdapter().stream(model="sonnet", messages=_user(), tools=TOOLS))


async def test_error_result_payload(installed: None, monkeypatch: pytest.MonkeyPatch):
    proc = _FakeProcess(
        _FakeStdout([_result_line(is_error=True, subtype="error_max_budget", result="over budget")])
    )
    _patch_exec(monkeypatch, proc)
    with pytest.raises(ConfigurationError, match="error_max_budget"):
        await _collect(ClaudeAdapter().stream(model="sonnet", messages=_user(), tools=TOOLS))


async def test_missing_result_payload(installed: None, monkeypatch: pytest.MonkeyPatch):
    proc = _FakeProcess(_FakeStdout([]))
    _patch_exec(monkeypatch, proc)
    with pytest.raises(InternalError, match="no result payload"):
        await _collect(ClaudeAdapter().stream(model="sonnet", messages=_user(), tools=TOOLS))


async def test_hard_timeout(installed: None, monkeypatch: pytest.MonkeyPatch):
    proc = _FakeProcess(_HangingStdout())
    _patch_exec(monkeypatch, proc)
    adapter = ClaudeAdapter(timeout=0.05, idle_timeout=30.0)
    with pytest.raises(HarnessTimeoutError, match="timed out"):
        await _collect(adapter.stream(model="sonnet", messages=_user(), tools=TOOLS))
    assert proc.terminated


async def test_idle_timeout_ignores_non_progress_lines(
    installed: None, monkeypatch: pytest.MonkeyPatch
):
    proc = _FakeProcess(_NoiseStdout())
    _patch_exec(monkeypatch, proc)
    adapter = ClaudeAdapter(timeout=30.0, idle_timeout=0.1)
    with pytest.raises(HarnessTimeoutError, match="no visible output"):
        await _collect(adapter.stream(model="sonnet", messages=_user(), tools=None))
    assert proc.terminated


def test_temperature_rejected_synchronously(installed: None):
    with pytest.raises(ConfigurationError, match="temperature"):
        ClaudeAdapter().stream(model="sonnet", messages=_user(), temperature=0.5)


def test_media_rejected_synchronously(installed: None):
    from harness.core.schemas import MediaAttachment

    message = Message(
        role="user",
        content="look",
        attachments=[
            MediaAttachment(kind="image", mime_type="image/png", data="AAAA", model_visible=True)
        ],
    )
    with pytest.raises(ConfigurationError, match="does not support media"):
        ClaudeAdapter().stream(model="sonnet", messages=[message])


# --------------------------------------------------------------------------
# Real process-group teardown
# --------------------------------------------------------------------------


@pytest.mark.skipif(os.name != "posix", reason="process groups are POSIX-only")
async def test_terminate_kills_grandchildren(
    installed: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """A stalled CLI must not leave Claude Code's helper processes behind."""

    sentinel = tmp_path / "grandchild.pid"
    script = tmp_path / "claude-stub.py"
    script.write_text(
        textwrap.dedent(
            f"""
            import subprocess, sys, time
            child = subprocess.Popen([
                sys.executable, "-c",
                "import os,sys,time;"
                "open({str(sentinel)!r}, 'w').write(str(os.getpid()));"
                "sys.stdout.flush();"
                "time.sleep(60)",
            ])
            time.sleep(60)
            """
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        claude_adapter, "_claude_bin", lambda explicit=None: explicit or sys.executable
    )
    adapter = ClaudeAdapter(claude_bin=sys.executable, timeout=30.0, idle_timeout=0.5)
    # Run the stub instead of the real CLI: argv[1:] are ignored by the stub.
    adapter.claude_bin = sys.executable
    original_argv = adapter._argv

    def _argv(**kwargs: Any) -> list[str]:
        del kwargs
        return [sys.executable, str(script)]

    adapter._argv = _argv  # type: ignore[method-assign]
    assert callable(original_argv)

    with pytest.raises(HarnessTimeoutError):
        await _collect(adapter.stream(model="sonnet", messages=_user(), tools=None))

    assert sentinel.exists(), "stub never started its grandchild"
    grandchild = int(sentinel.read_text())
    for _ in range(50):
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.1)
    else:  # pragma: no cover - only reached when teardown regresses
        os.kill(grandchild, 9)
        pytest.fail("grandchild survived adapter teardown")
