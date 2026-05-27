"""Tests for `harness chat` REPL.

CliRunner's `input=` feeds lines to stdin so we can drive the REPL.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, ClassVar

import pytest
from click.testing import Result
from rich.console import Console
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.cli import chat_commands
from harness.core import (
    Capabilities,
    Done,
    Event,
    HandoffEvent,
    Message,
    TextDelta,
    ToolCall,
    ToolResult,
)
from harness.core.events import ToolCallEvent, ToolResultEvent


def text_turn(text: str) -> list[Event]:
    return [TextDelta(text=text), Done(final_message=Message(role="assistant", content=text))]


def sleep_marker(seconds: float) -> tuple[str, float]:
    return ("sleep", seconds)


def tool_call_turn(*, call_id: str, name: str, arguments: dict[str, Any]) -> list[Event]:
    call = ToolCall(id=call_id, name=name, arguments=arguments)
    return [
        ToolCallEvent(call=call),
        Done(final_message=Message(role="assistant", content=None, tool_calls=[call])),
    ]


class FakeAdapter:
    name = "ollama"
    next_script: ClassVar[list[list[Event]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def stream(self, **_kwargs: Any) -> AsyncIterator[Event]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Event]:
        if not FakeAdapter.next_script:
            raise RuntimeError("FakeAdapter has no scripts left")
        for ev in FakeAdapter.next_script.pop(0):
            if isinstance(ev, tuple) and len(ev) == 2 and ev[0] == "sleep":
                await asyncio.sleep(ev[1])
                continue
            yield ev

    async def capabilities(self) -> Capabilities:
        return Capabilities(streaming=True, tool_use=True)

    async def cancel(self, session_id: str) -> None:
        pass


class ClassifierAdapter:
    def __init__(self, text: str) -> None:
        self._text = text

    def stream(self, **_kwargs: Any) -> AsyncIterator[Event]:
        return self._stream()

    async def _stream(self) -> AsyncIterator[Event]:
        yield Done(final_message=Message(role="assistant", content=self._text))


@pytest.fixture
def patch_adapter(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(cli_main, "OllamaAdapter", FakeAdapter)
    monkeypatch.setattr(
        chat_commands,
        "_classify_chat_turn_policy",
        lambda **_: asyncio.sleep(0, result=chat_commands._GENERAL_TURN_POLICY),
    )

    def configure(scripts: list[list[Event]]) -> None:
        FakeAdapter.next_script = scripts

    yield configure
    FakeAdapter.next_script = []


def _run(args: list[str], stdin: str) -> Result:
    return CliRunner().invoke(cli_main.app, args, input=stdin)


class TestSlashCommands:
    @pytest.mark.asyncio
    async def test_classifier_parses_non_exact_review_token(self) -> None:
        result = await chat_commands._classify_chat_turn_policy(
            adapter=ClassifierAdapter("This should be code-review."),
            model="m",
            prompt="Review the current codebase. Give top findings only.",
        )
        assert result == chat_commands._REVIEW_TURN_POLICY

    @pytest.mark.asyncio
    async def test_classifier_routes_durable_work_prompt_to_workflow_policy(self) -> None:
        result = await chat_commands._classify_chat_turn_policy(
            adapter=ClassifierAdapter("workflow"),
            model="m",
            prompt="Set up a long-running workflow for a checkout migration using Harness primitives.",
        )
        assert result == chat_commands._WORKFLOW_TURN_POLICY

    @pytest.mark.asyncio
    async def test_classifier_routes_read_only_repo_question_to_research_policy(self) -> None:
        result = await chat_commands._classify_chat_turn_policy(
            adapter=ClassifierAdapter("research"),
            model="m",
            prompt="Where does SQLite session storage live? Keep it brief.",
        )
        assert result == chat_commands._RESEARCH_TURN_POLICY

    def test_quit_exits_cleanly(self, patch_adapter, tmp_path: Path) -> None:
        result = _run(["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"], stdin="/quit\n")
        assert result.exit_code == 0
        assert "bye" in result.stdout

    def test_help_lists_commands(self, patch_adapter, tmp_path: Path) -> None:
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="/help\n/quit\n",
        )
        assert result.exit_code == 0
        assert "/help" in result.stdout
        assert "/tools" in result.stdout

    def test_tools_lists_built_in_tools(self, patch_adapter, tmp_path: Path) -> None:
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="/tools\n/quit\n",
        )
        assert result.exit_code == 0
        assert "read_file" in result.stdout
        assert "shell" in result.stdout
        assert "handoff_to_research_specialist" in result.stdout
        assert "handoff_to_review_specialist" in result.stdout
        assert "handoff_to_workflow_specialist" in result.stdout

    def test_review_prompt_scopes_tools_without_domain_flag(
        self, patch_adapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_adapter([text_turn("reviewed")])
        monkeypatch.setattr(
            chat_commands,
            "_classify_chat_turn_policy",
            lambda **_: asyncio.sleep(0, result=chat_commands._REVIEW_TURN_POLICY),
        )
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="Review the current codebase. Give top findings only.\n/tools\n/quit\n",
        )
        assert result.exit_code == 0
        assert "reviewed" in result.stdout
        assert "read_file" in result.stdout
        assert "shell" not in result.stdout
        assert "spawn_agents" not in result.stdout

    def test_policy_reclassifies_on_later_turns(
        self, patch_adapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_adapter([text_turn("reviewed"), text_turn("researched")])
        seen_prompts: list[str] = []
        policies = iter(
            [
                chat_commands._REVIEW_TURN_POLICY,
                chat_commands._RESEARCH_TURN_POLICY,
            ]
        )

        async def classify(**kwargs: Any) -> chat_commands._ChatTurnPolicy:
            seen_prompts.append(str(kwargs["prompt"]))
            return next(policies)

        monkeypatch.setattr(chat_commands, "_classify_chat_turn_policy", classify)
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin=(
                "Review the current codebase. Give top findings only.\n"
                "/tools\n"
                "Where does SQLite session storage live? Keep it brief.\n"
                "/tools\n"
                "/quit\n"
            ),
        )
        assert result.exit_code == 0, result.stdout
        assert seen_prompts == [
            "Review the current codebase. Give top findings only.",
            "Where does SQLite session storage live? Keep it brief.",
        ]
        assert "reviewed" in result.stdout
        assert "researched" in result.stdout
        assert result.stdout.count("shell") == 1

    def test_review_prompt_hides_tool_trace_and_formats_findings(
        self, patch_adapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_adapter(
            [
                tool_call_turn(call_id="c1", name="list_dir", arguments={"path": "."}),
                text_turn(
                    '{"summary":"Two issues matter.","findings":['
                    '{"severity":"high","file":"a.py","line":12,"issue":"bug","rationale":"breaks users"},'
                    '{"severity":"low","file":"b.py","line":3,"issue":"nit","rationale":"minor"},'
                    '{"severity":"medium","file":"c.py","line":9,"issue":"race","rationale":"can misorder"},'
                    '{"severity":"low","file":"d.py","line":1,"issue":"extra","rationale":"should be truncated"}'
                    "]}"
                ),
            ]
        )
        monkeypatch.setattr(
            chat_commands,
            "_classify_chat_turn_policy",
            lambda **_: asyncio.sleep(0, result=chat_commands._REVIEW_TURN_POLICY),
        )
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="Review the current codebase. Give top findings only.\n/quit\n",
        )
        assert result.exit_code == 0, result.stdout
        assert "→ list_dir" not in result.stdout
        assert "✓ list_dir" not in result.stdout
        assert "Two issues matter." in result.stdout
        assert "[high]" in result.stdout
        assert "a.py:12" in result.stdout
        assert "d.py:1" not in result.stdout

    def test_review_formatter_accepts_fenced_json(self) -> None:
        rendered = chat_commands._format_code_review_output(
            """```json
            {
              "summary": "Two issues matter.",
              "findings": [
                {
                  "severity": "high",
                  "file": "a.py",
                  "line": 12,
                  "issue": "bug",
                  "rationale": "breaks users"
                }
              ]
            }
            ```"""
        )
        assert rendered is not None

    def test_workflow_renderer_falls_back_to_tool_result_artifact_summary(self) -> None:
        capture = Console(record=True, width=100)
        adapter = chat_commands._ChatRenderAdapter(
            console=capture, default_render=lambda _event: None
        )
        adapter.set_policy(chat_commands._WORKFLOW_TURN_POLICY)
        adapter.render(
            ToolResultEvent(
                result=ToolResult(
                    tool_call_id="t1",
                    name="shell",
                    content=(
                        "Launched long-running workflow\n"
                        "task_ref=T-123\n"
                        "mission_id=mission-checkout-1234\n"
                        "resume_feature=checkout-migration\n"
                        "scheduler_job_id=sched-1234\n"
                    ),
                )
            )
        )
        adapter.render(Done(final_message=Message(role="assistant", content=None)))
        output = capture.export_text()
        assert "Harness bootstrapped the durable workflow." in output
        assert "mission-checkout-1234" in output
        assert "T-123" in output
        assert "checkout-migration" in output
        assert "sched-1234" in output

    def test_renderer_prints_handoff_event(self) -> None:
        capture = Console(record=True, width=100)
        adapter = chat_commands._ChatRenderAdapter(
            console=capture, default_render=lambda _event: None
        )
        adapter.render(
            HandoffEvent(target_name="research-specialist", reason="needs repo research")
        )
        output = capture.export_text()
        assert "handoff" in output
        assert "research-specialist" in output
        assert "needs repo research" in output

    def test_unknown_command_warns(self, patch_adapter, tmp_path: Path) -> None:
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="/nope\n/quit\n",
        )
        assert result.exit_code == 0
        assert "Unknown command" in result.stdout

    def test_new_switch_and_sessions_manage_multiple_conversations(
        self, patch_adapter, tmp_path: Path
    ) -> None:
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="/new alpha\n/new beta\n/sessions\n/switch alpha\n/session\n/quit\n",
        )
        assert result.exit_code == 0, result.stdout
        assert "alpha" in result.stdout
        assert "beta" in result.stdout
        assert "Switched to: alpha" in result.stdout

    def test_send_runs_background_conversation_while_other_conversation_is_active(
        self, patch_adapter, tmp_path: Path
    ) -> None:
        patch_adapter(
            [
                [sleep_marker(0.05), *text_turn("alpha done")],
                [sleep_marker(0.10), *text_turn("beta done")],
            ]
        )
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="/new alpha\n/send . work on alpha\n/new beta\nwork on beta\n/quit\n",
        )
        assert result.exit_code == 0, result.stdout
        assert "Started background turn: alpha" in result.stdout
        assert "alpha done" in result.stdout
        assert "beta done" in result.stdout

    def test_general_turn_can_handoff_to_specialist(
        self, patch_adapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_adapter(
            [
                tool_call_turn(
                    call_id="h1",
                    name="handoff_to_research_specialist",
                    arguments={"reason": "needs repo research"},
                ),
                text_turn("specialist answer"),
            ]
        )
        monkeypatch.setattr(
            chat_commands,
            "_classify_chat_turn_policy",
            lambda **_: asyncio.sleep(0, result=chat_commands._GENERAL_TURN_POLICY),
        )
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="Use a specialist handoff to research this repo.\n/quit\n",
        )
        assert result.exit_code == 0, result.stdout
        assert "handoff" in result.stdout
        assert "needs repo research" in result.stdout
        assert "specialist answer" in result.stdout

    def test_research_turn_routes_to_specialist_automatically(
        self, patch_adapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_adapter([text_turn("researched answer")])
        monkeypatch.setattr(
            chat_commands,
            "_classify_chat_turn_policy",
            lambda **_: asyncio.sleep(0, result=chat_commands._RESEARCH_TURN_POLICY),
        )
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="Research how handoffs work in this repo and cite the relevant files.\n/quit\n",
        )
        assert result.exit_code == 0, result.stdout
        assert "routed" in result.stdout
        assert "research specialist" in result.stdout
        assert "researched answer" in result.stdout

    def test_workflow_turn_routes_to_specialist_automatically(
        self, patch_adapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_adapter([text_turn("workflow answer")])
        monkeypatch.setattr(
            chat_commands,
            "_classify_chat_turn_policy",
            lambda **_: asyncio.sleep(0, result=chat_commands._WORKFLOW_TURN_POLICY),
        )
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="Please set up a long-running checkout migration workflow for me.\n/quit\n",
        )
        assert result.exit_code == 0, result.stdout
        assert "routed" in result.stdout
        assert "workflow specialist" in result.stdout
        assert "workflow answer" in result.stdout

    def test_routed_specialist_retries_after_silent_attempt(
        self, patch_adapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        patch_adapter(
            [
                [Done(final_message=Message(role="assistant", content=None))],
                text_turn("workflow answer"),
            ]
        )
        monkeypatch.setattr(
            chat_commands,
            "_classify_chat_turn_policy",
            lambda **_: asyncio.sleep(0, result=chat_commands._WORKFLOW_TURN_POLICY),
        )
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="Please set up a long-running checkout migration workflow for me.\n/quit\n",
        )
        assert result.exit_code == 0, result.stdout
        assert "retrying" in result.stdout
        assert "workflow specialist" in result.stdout
        assert "workflow answer" in result.stdout


class TestRegularTurns:
    def test_default_system_prompt_prefers_harness_primitives_for_long_running_work(self) -> None:
        prompt = cli_main._DEFAULT_SYSTEM_PROMPT
        assert "general-purpose AI work agent" in prompt
        assert "Harness is general, not coding-only." in prompt
        assert "Code-change workflow" in prompt
        assert "Use Harness primitives instead." in prompt
        assert "harness mission launch" in prompt
        assert "harness resume show" in prompt
        assert "harness scheduler list" in prompt
        assert "harness approvals list" in prompt
        assert "harness evidence list" in prompt
        assert "bootstrap it in this order" in prompt
        assert "harness mission show <mission_id>" in prompt
        assert "harness mission show-contract --mission <mission_id>" in prompt
        assert "harness mission summarize --mission <mission_id>" in prompt
        assert "Do not call `resume init` after `mission launch`" in prompt

    def test_workflow_turn_policy_prompt_enforces_exact_harness_bootstrap_order(self) -> None:
        prompt = chat_commands._WORKFLOW_BOOTSTRAP_SYSTEM_PROMPT
        assert chat_commands._WORKFLOW_TURN_POLICY.disable_verify is True
        assert chat_commands._WORKFLOW_TURN_POLICY.profile == "bare"
        assert "harness mission launch --title <title> --goal <goal>" in prompt
        assert "harness resume show" in prompt
        assert "harness mission show <mission_id>" in prompt
        assert "harness mission show-contract --mission <mission_id>" in prompt
        assert "harness mission summarize --mission <mission_id>" in prompt
        assert "harness scheduler list-runs" in prompt
        assert "harness approvals list" in prompt
        assert "harness evidence list" in prompt

    def test_research_turn_policy_is_read_only_and_unverified(self) -> None:
        assert chat_commands._RESEARCH_TURN_POLICY.disable_verify is True
        assert chat_commands._RESEARCH_TURN_POLICY.profile == "bare"
        assert chat_commands._RESEARCH_TURN_POLICY.allowed_tools is not None
        assert "shell" in chat_commands._RESEARCH_TURN_POLICY.allowed_tools
        assert "read-only" in chat_commands._RESEARCH_SYSTEM_PROMPT.lower()

    def test_single_turn_streams_response(self, patch_adapter, tmp_path: Path) -> None:
        patch_adapter([text_turn("hello there")])
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="hi\n/quit\n",
        )
        assert result.exit_code == 0, result.stdout
        assert "hello there" in result.stdout

    def test_multi_turn_uses_resume(self, patch_adapter, tmp_path: Path) -> None:
        # Two turns: first via agent.run(), second via agent.resume().
        patch_adapter([text_turn("first"), text_turn("second")])
        result = _run(
            ["chat", "--cwd", str(tmp_path), "--in-memory", "--yes"],
            stdin="hello\nfollow up\n/quit\n",
        )
        assert result.exit_code == 0, result.stdout
        assert "first" in result.stdout
        assert "second" in result.stdout


class TestSession:
    def test_session_command_reports_state(self, patch_adapter, tmp_path: Path) -> None:
        patch_adapter([text_turn("ok")])
        result = _run(
            [
                "chat",
                "--cwd",
                str(tmp_path),
                "--in-memory",
                "--yes",
                "--session",
                "sess_explicit",
            ],
            stdin="hello\n/session\n/quit\n",
        )
        assert result.exit_code == 0
        assert "sess_explicit" in result.stdout
        # After one turn: user + assistant = 2 messages
        assert "2 messages" in result.stdout
