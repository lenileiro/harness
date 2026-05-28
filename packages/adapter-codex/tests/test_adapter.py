from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from harness.adapters.codex import CodexAdapter, inspect_codex_cli_auth
from harness.core import (
    ConfigurationError,
    Done,
    Message,
    TextDelta,
    ToolCallEvent,
    ToolResultEvent,
)
from harness.core.errors import TimeoutError as HarnessTimeoutError


class _FakeStdout:
    def __init__(self, lines: list[bytes]) -> None:
        self._lines = list(lines)

    def __aiter__(self) -> _FakeStdout:
        return self

    async def __anext__(self) -> bytes:
        if not self._lines:
            raise StopAsyncIteration
        return self._lines.pop(0)

    async def readline(self) -> bytes:
        if not self._lines:
            return b""
        return self._lines.pop(0)


class _HangingStdout:
    def __aiter__(self) -> _HangingStdout:
        return self

    async def __anext__(self) -> bytes:
        await asyncio.sleep(60)
        raise StopAsyncIteration

    async def readline(self) -> bytes:
        await asyncio.sleep(60)
        return b""


class _IgnoredEventStdout:
    async def readline(self) -> bytes:
        await asyncio.sleep(0.002)
        return (
            json.dumps(
                {
                    "type": "item.updated",
                    "item": {
                        "id": "cmd1",
                        "type": "command_execution",
                        "aggregated_output": "",
                        "status": "running",
                    },
                }
            ).encode()
            + b"\n"
        )


class _FakeStderr:
    def __init__(self, payload: bytes = b"") -> None:
        self._payload = payload

    async def read(self) -> bytes:
        return self._payload


class _FakeProcess:
    def __init__(self, *, lines: list[bytes], returncode: int = 0, stderr: bytes = b"") -> None:
        self.stdout = _FakeStdout(lines)
        self.stderr = _FakeStderr(stderr)
        self.returncode = returncode
        self.terminated = False

    async def wait(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True


class _HangingProcess(_FakeProcess):
    def __init__(self) -> None:
        super().__init__(lines=[])
        self.stdout = _HangingStdout()


class _IgnoredEventProcess(_FakeProcess):
    def __init__(self) -> None:
        super().__init__(lines=[])
        self.stdout = _IgnoredEventStdout()


async def _collect(it) -> list:
    out = []
    async for event in it:
        out.append(event)
    return out


class TestConstruction:
    def test_missing_codex_binary_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("shutil.which", lambda _name: None)
        with pytest.raises(ConfigurationError, match="Codex CLI not found"):
            CodexAdapter()

    def test_missing_codex_auth_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/codex")
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        with pytest.raises(ConfigurationError, match="run `codex login` first"):
            CodexAdapter()


class TestAuthHelper:
    def test_reports_chatgpt_oauth_presence(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(
            json.dumps(
                {
                    "auth_mode": "chatgpt",
                    "OPENAI_API_KEY": None,
                    "tokens": {"access_token": "oauth-token"},
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        assert inspect_codex_cli_auth() == {
            "auth_mode": "chatgpt",
            "has_openai_api_key": False,
            "has_access_token": True,
        }


@pytest.mark.asyncio
class TestStream:
    async def test_stream_parses_agent_messages_and_shell_activity(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(
            json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "oauth-token"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/codex")

        lines = [
            b"Reading additional input from stdin...\n",
            json.dumps({"type": "thread.started", "thread_id": "t"}).encode() + b"\n",
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "m1", "type": "agent_message", "text": "Looking around."},
                }
            ).encode()
            + b"\n",
            json.dumps(
                {
                    "type": "item.started",
                    "item": {
                        "id": "cmd1",
                        "type": "command_execution",
                        "command": "/bin/zsh -lc pwd",
                    },
                }
            ).encode()
            + b"\n",
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "cmd1",
                        "type": "command_execution",
                        "command": "/bin/zsh -lc pwd",
                        "aggregated_output": "/tmp/demo\n",
                        "exit_code": 0,
                        "status": "completed",
                    },
                }
            ).encode()
            + b"\n",
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"id": "m2", "type": "agent_message", "text": "Done."},
                }
            ).encode()
            + b"\n",
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {
                        "input_tokens": 10,
                        "cached_input_tokens": 4,
                        "output_tokens": 2,
                    },
                }
            ).encode()
            + b"\n",
        ]

        async def fake_exec(*_args, **_kwargs):
            return _FakeProcess(lines=lines)

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

        adapter = CodexAdapter(cwd=tmp_path)
        events = await _collect(
            adapter.stream(model="gpt-5.5", messages=[Message(role="user", content="Say hi")])
        )

        text_events = [event for event in events if isinstance(event, TextDelta)]
        assert [event.text for event in text_events] == ["Looking around.", "Done."]
        tool_calls = [event for event in events if isinstance(event, ToolCallEvent)]
        assert len(tool_calls) == 1
        assert tool_calls[0].call.arguments["command"] == "/bin/zsh -lc pwd"
        tool_results = [event for event in events if isinstance(event, ToolResultEvent)]
        assert len(tool_results) == 1
        assert tool_results[0].result.content == "/tmp/demo\n"
        done = events[-1]
        assert isinstance(done, Done)
        assert done.final_message is not None
        assert done.final_message.content == "Done."
        assert done.usage is not None
        assert done.usage.prompt_tokens == 10
        assert done.usage.cache_read_input_tokens == 4

    async def test_stream_surfaces_partial_command_updates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(
            json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "oauth-token"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/codex")

        lines = [
            json.dumps(
                {
                    "type": "item.started",
                    "item": {
                        "id": "cmd1",
                        "type": "command_execution",
                        "command": "pytest -q",
                    },
                }
            ).encode()
            + b"\n",
            json.dumps(
                {
                    "type": "item.updated",
                    "item": {
                        "id": "cmd1",
                        "type": "command_execution",
                        "command": "pytest -q",
                        "aggregated_output": "collecting tests...\n",
                        "status": "running",
                    },
                }
            ).encode()
            + b"\n",
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {
                        "id": "cmd1",
                        "type": "command_execution",
                        "command": "pytest -q",
                        "aggregated_output": "collecting tests...\n1 passed\n",
                        "exit_code": 0,
                    },
                }
            ).encode()
            + b"\n",
        ]

        async def fake_exec(*_args, **_kwargs):
            return _FakeProcess(lines=lines)

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)

        adapter = CodexAdapter(cwd=tmp_path)
        events = await _collect(
            adapter.stream(model="gpt-5.5", messages=[Message(role="user", content="test")])
        )

        tool_results = [event for event in events if isinstance(event, ToolResultEvent)]
        assert len(tool_results) == 2
        assert tool_results[0].result.content == "collecting tests...\n"
        assert tool_results[0].result.metadata is not None
        assert tool_results[0].result.metadata["partial"] is True
        assert tool_results[1].result.content == "collecting tests...\n1 passed\n"

    async def test_nonzero_exit_maps_auth_failures_to_configuration_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(
            json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "oauth-token"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/codex")

        async def fake_exec(*_args, **_kwargs):
            return _FakeProcess(lines=[], returncode=1, stderr=b"Auth failed, please login")

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        adapter = CodexAdapter(cwd=tmp_path)
        with pytest.raises(ConfigurationError, match="auth failed"):
            await _collect(
                adapter.stream(model="gpt-5.5", messages=[Message(role="user", content="hi")])
            )

    async def test_idle_timeout_terminates_codex_process_and_raises_harness_timeout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(
            json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "oauth-token"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/codex")

        process = _HangingProcess()

        async def fake_exec(*_args, **_kwargs):
            return process

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        adapter = CodexAdapter(cwd=tmp_path, timeout=60.0, idle_timeout=0.01)

        with pytest.raises(HarnessTimeoutError, match="produced no visible output"):
            await _collect(
                adapter.stream(model="gpt-5.5", messages=[Message(role="user", content="hi")])
            )
        assert process.terminated is True

    async def test_ignored_codex_events_do_not_mask_visible_idle_timeout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(
            json.dumps({"auth_mode": "chatgpt", "tokens": {"access_token": "oauth-token"}}),
            encoding="utf-8",
        )
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/codex")

        process = _IgnoredEventProcess()

        async def fake_exec(*_args, **_kwargs):
            return process

        monkeypatch.setattr("asyncio.create_subprocess_exec", fake_exec)
        adapter = CodexAdapter(cwd=tmp_path, timeout=60.0, idle_timeout=0.01)

        with pytest.raises(HarnessTimeoutError, match="produced no visible output"):
            await _collect(
                adapter.stream(model="gpt-5.5", messages=[Message(role="user", content="hi")])
            )
        assert process.terminated is True
