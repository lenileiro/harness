from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

import pytest
import typer
from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.cli import gateway_commands, gateway_runtime
from harness.core import (
    ActivityEvent,
    GatewaySessionStore,
    Message,
    PendingApproval,
    WhatsAppBridgeConfig,
    default_gateway_root,
    save_whatsapp_bridge_config,
)
from harness.core.gateway_evidence import successful_tool_evidence_reply
from harness.storage.sqlite import SQLiteStorage


def test_gateway_turn_timeout_follows_whatsapp_child_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_GATEWAY_WORKFLOW_TIMEOUT", raising=False)
    monkeypatch.setenv("HARNESS_WHATSAPP_CHILD_TIMEOUT_MS", "300000")

    assert gateway_runtime._gateway_turn_timeout_seconds() == 290.0


def test_gateway_turn_timeout_default_allows_repair_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_GATEWAY_WORKFLOW_TIMEOUT", raising=False)
    monkeypatch.delenv("HARNESS_WHATSAPP_CHILD_TIMEOUT_MS", raising=False)

    assert gateway_runtime._gateway_turn_timeout_seconds() == 590.0


def test_gateway_turn_timeout_override_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_GATEWAY_WORKFLOW_TIMEOUT", "42")
    monkeypatch.setenv("HARNESS_WHATSAPP_CHILD_TIMEOUT_MS", "300000")

    assert gateway_runtime._gateway_turn_timeout_seconds() == 42.0


def test_gateway_max_repair_defaults_to_multi_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_GATEWAY_MAX_REPAIR", raising=False)

    assert gateway_runtime._gateway_max_repair_attempts() == 3


def test_gateway_max_repair_override_wins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HARNESS_GATEWAY_MAX_REPAIR", "5")

    assert gateway_runtime._gateway_max_repair_attempts() == 5


def test_tool_failure_reply_from_messages_reports_live_tool_error() -> None:
    result = gateway_runtime._tool_failure_reply_from_messages(
        [
            Message(role="user", content="What is the weather in Tokyo?"),
            Message(
                role="tool",
                name="web_search",
                tool_call_id="call_1",
                content="search failed: This request exceeds your plan's set usage limit.",
            ),
        ]
    )

    assert result is not None
    assert "`web_search`" in result
    assert "exceeds your plan" in result


def test_tool_failure_reply_from_messages_ignores_stale_failure_after_verify_work() -> None:
    result = gateway_runtime._tool_failure_reply_from_messages(
        [
            Message(role="user", content="Fix the failing tests."),
            Message(
                role="tool",
                name="shell",
                tool_call_id="call_1",
                content="exit_code: 1\n\nstdout:\nFAILED tests/test_calculator.py",
            ),
            Message(
                role="tool",
                name="edit_file",
                tool_call_id="call_2",
                content="replaced 1 occurrence in calculator.py",
            ),
            Message(
                role="tool",
                name="verify_work",
                tool_call_id="call_3",
                content="PASSED\n\n1 passed in 0.01s",
            ),
        ]
    )

    assert result is None


def test_successful_tool_evidence_reply_uses_web_search_source() -> None:
    event = ActivityEvent(
        session_id="sess-weather",
        kind="tool_call.completed",
        data={
            "name": "web_search",
            "is_error": False,
            "metadata": {
                "results": [
                    {
                        "title": "Weather in Tokyo",
                        "url": "https://www.weatherapi.com/",
                        "content": "Tokyo is sunny, 21 C, humidity 64%.",
                    }
                ]
            },
        },
    )
    result = successful_tool_evidence_reply([event])

    assert result is not None
    assert "Weather in Tokyo: https://www.weatherapi.com/" in result
    assert "Tokyo is sunny" in result


def test_successful_tool_evidence_reply_uses_local_shell_source() -> None:
    event = ActivityEvent(
        session_id="sess-time",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {"command": "TZ='America/New_York' date"},
            "content_preview": "exit_code: 0\n\nstdout:\nSun May 31 08:21:39 EDT 2026\n",
        },
    )
    result = successful_tool_evidence_reply([event])

    assert result is not None
    assert "local shell evidence" in result
    assert "TZ='America/New_York' date" in result
    assert "Sun May 31 08:21:39 EDT 2026" in result


def test_successful_tool_evidence_reply_uses_verify_work_source() -> None:
    event = ActivityEvent(
        session_id="sess-verify",
        kind="tool_call.completed",
        data={
            "name": "verify_work",
            "is_error": False,
            "arguments": {"command": "python3 test_slugify.py"},
            "content_preview": "PASSED\n\n......\nRan 6 tests in 0.001s\n\nOK\n",
        },
    )
    result = successful_tool_evidence_reply([event])

    assert result is not None
    assert "verify_work" in result
    assert "python3 test_slugify.py" in result
    assert "Ran 6 tests" in result


def test_successful_tool_evidence_reply_redacts_shell_secrets() -> None:
    fake_key = "sk-or-v1" + "-secret"
    event = ActivityEvent(
        session_id="sess-secret",
        kind="tool_call.completed",
        data={
            "name": "shell",
            "is_error": False,
            "arguments": {"command": f"OPENROUTER_API_KEY={fake_key} printenv"},
            "content_preview": f"exit_code: 0\n\nstdout:\n{fake_key}\n",
        },
    )
    result = successful_tool_evidence_reply([event])

    assert result is not None
    assert "secret" not in result
    assert "OPENROUTER_API_KEY=<redacted>" in result


def test_default_gateway_provider_prefers_configured_api_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert gateway_runtime._default_gateway_provider() == "ollama"

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    assert gateway_runtime._default_gateway_provider() == "openrouter"


def test_gateway_dispatch_can_start_mission_and_report_status(tmp_path: Path) -> None:
    runner = CliRunner()

    created = runner.invoke(
        cli_main.app,
        [
            "mission",
            "create",
            "--title",
            "Gateway mission demo",
            "--goal",
            "Start a mission through the gateway.",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    mission_id = next((tmp_path / ".harness" / "missions" / "missions").iterdir()).name

    planned = runner.invoke(
        cli_main.app,
        [
            "mission",
            "plan",
            "--mission",
            mission_id,
            "--contract-summary",
            "Assertions define correctness before implementation.",
            "--milestone",
            "m1|Milestone 1|Ship a single validated slice.",
            "--assertion",
            "a1|Gateway runs|The mission can be started remotely.|behavior|Run the mission loop.",
            "--feature",
            "f1|m1|Implement slice|Add the first mission slice.|worker|app/demo.py||a1",
            "--cwd",
            str(tmp_path),
        ],
    )
    assert planned.exit_code == 0, planned.stdout

    approved = runner.invoke(
        cli_main.app,
        ["mission", "approve", "--mission", mission_id, "--cwd", str(tmp_path)],
    )
    assert approved.exit_code == 0, approved.stdout

    started = runner.invoke(
        cli_main.app,
        [
            "gateway",
            "dispatch",
            "--transport",
            "local",
            "--user",
            "tester",
            "--thread",
            "demo",
            "--message",
            f"mission start {mission_id}",
            "--cwd",
            str(tmp_path),
            "--in-memory",
            "--json",
        ],
    )
    assert started.exit_code == 0, started.stdout
    started_payload = json.loads(started.stdout)
    assert started_payload["reply"]["command"] == "mission.start"
    assert started_payload["reply"]["status"] == "ok"
    assert started_payload["session"]["current_mission_id"] == mission_id

    status = runner.invoke(
        cli_main.app,
        [
            "gateway",
            "dispatch",
            "--transport",
            "local",
            "--user",
            "tester",
            "--thread",
            "demo",
            "--message",
            "status",
            "--cwd",
            str(tmp_path),
            "--in-memory",
            "--json",
        ],
    )
    assert status.exit_code == 0, status.stdout
    status_payload = json.loads(status.stdout)
    assert status_payload["reply"]["command"] == "status"
    assert status_payload["reply"]["data"]["jobs_total"] >= 1
    assert status_payload["reply"]["data"]["shared_queue_total"] >= 1
    assert status_payload["reply"]["data"]["shared_queue_ready"] >= 0

    runs = runner.invoke(
        cli_main.app,
        [
            "gateway",
            "dispatch",
            "--transport",
            "local",
            "--user",
            "tester",
            "--thread",
            "demo",
            "--message",
            "runs",
            "--cwd",
            str(tmp_path),
            "--in-memory",
            "--json",
        ],
    )
    assert runs.exit_code == 0, runs.stdout
    runs_payload = json.loads(runs.stdout)
    assert runs_payload["reply"]["command"] == "runs"
    assert runs_payload["reply"]["data"]["runs"]

    report = runner.invoke(
        cli_main.app,
        [
            "gateway",
            "dispatch",
            "--transport",
            "local",
            "--user",
            "tester",
            "--thread",
            "demo",
            "--message",
            f"report {mission_id}",
            "--cwd",
            str(tmp_path),
            "--in-memory",
            "--json",
        ],
    )
    assert report.exit_code == 0, report.stdout
    report_payload = json.loads(report.stdout)
    assert report_payload["reply"]["command"] == "report"
    assert report_payload["reply"]["data"]["mission_id"] == mission_id

    sessions = runner.invoke(
        cli_main.app,
        ["gateway", "list-sessions", "--cwd", str(tmp_path), "--json"],
    )
    assert sessions.exit_code == 0, sessions.stdout
    sessions_payload = json.loads(sessions.stdout)
    assert sessions_payload[0]["user_id"] == "tester"
    assert sessions_payload[0]["thread_id"] == "demo"


def test_gateway_dispatch_can_grant_approval(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "gateway.db"

    async def _seed() -> str:
        storage = SQLiteStorage(path=db_path)
        try:
            saved = await storage.create_approval(
                PendingApproval(
                    session_id="sess_gateway",
                    tool_call_id="tool_call_1",
                    tool_name="shell",
                    arguments={"cmd": "echo hi"},
                )
            )
            return saved.id
        finally:
            await storage.close()

    approval_id = asyncio.run(_seed())

    granted = runner.invoke(
        cli_main.app,
        [
            "gateway",
            "dispatch",
            "--transport",
            "local",
            "--user",
            "approver",
            "--thread",
            "approvals",
            "--message",
            f"approve {approval_id}",
            "--cwd",
            str(tmp_path),
            "--db",
            str(db_path),
            "--json",
        ],
    )
    assert granted.exit_code == 0, granted.stdout
    payload = json.loads(granted.stdout)
    assert payload["reply"]["command"] == "approve"
    assert payload["reply"]["status"] == "ok"

    async def _load_status() -> str:
        storage = SQLiteStorage(path=db_path)
        try:
            approval = await storage.get_approval(approval_id)
            assert approval is not None
            return approval.status
        finally:
            await storage.close()

    assert asyncio.run(_load_status()) == "granted"


def test_gateway_whatsapp_setup_can_configure_self_chat_noninteractively(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli_main.app,
        [
            "gateway",
            "whatsapp",
            "setup",
            "--cwd",
            str(tmp_path),
            "--mode",
            "self-chat",
            "--allowed-user",
            "15551234567",
            "--bridge-port",
            "9918",
            "--max-gateway-concurrency",
            "1",
            "--max-gateway-queue",
            "2",
            "--gateway-child-timeout-seconds",
            "90",
            "--no-install",
            "--no-pair",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["provider"] == "ollama"
    assert payload["model"] == "gemma4:latest"
    assert payload["mode"] == "self-chat"
    assert payload["allowed_users"] == ["15551234567"]
    assert payload["bridge_port"] == 9918
    assert payload["max_gateway_concurrency"] == 1
    assert payload["max_gateway_queue"] == 2
    assert payload["gateway_child_timeout_seconds"] == 90
    assert payload["paired"] is False
    assert payload["enabled"] is False


def test_gateway_whatsapp_setup_defaults_openrouter_model(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli_main.app,
        [
            "gateway",
            "whatsapp",
            "setup",
            "--cwd",
            str(tmp_path),
            "--provider",
            "openrouter",
            "--mode",
            "self-chat",
            "--allowed-user",
            "15551234567",
            "--no-install",
            "--no-pair",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["provider"] == "openrouter"
    assert payload["model"] == "openai/gpt-5.4-nano"


def test_gateway_whatsapp_status_reports_bridge_state(tmp_path: Path) -> None:
    runner = CliRunner()
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            mode="bot",
            allowed_users=["15550001111"],
            bridge_port=9919,
        ),
    )

    result = runner.invoke(
        cli_main.app,
        [
            "gateway",
            "whatsapp",
            "status",
            "--cwd",
            str(tmp_path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["enabled"] is True
    assert payload["mode"] == "bot"
    assert payload["allowed_users"] == ["15550001111"]
    assert payload["bridge_port"] == 9919


def test_gateway_whatsapp_send_uses_local_bridge(tmp_path: Path) -> None:
    runner = CliRunner()
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(enabled=True, mode="self-chat", allowed_users=[], bridge_port=9907),
    )

    def _fake_send(*, cwd: Path | None = None, to: str, text: str, reply_to: str | None = None):
        assert cwd == tmp_path
        assert to == "15551234567"
        assert text == "hello"
        assert reply_to is None
        return {"ok": True, "messageId": "wamid.local"}

    original = gateway_commands.send_whatsapp_text_message
    gateway_commands.send_whatsapp_text_message = _fake_send
    try:
        result = runner.invoke(
            cli_main.app,
            [
                "gateway",
                "whatsapp",
                "send",
                "--cwd",
                str(tmp_path),
                "--to",
                "15551234567",
                "--text",
                "hello",
                "--json",
            ],
        )
    finally:
        gateway_commands.send_whatsapp_text_message = original
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True


def test_gateway_whatsapp_pair_marks_config_enabled(tmp_path: Path) -> None:
    runner = CliRunner()
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(enabled=False, mode="self-chat", allowed_users=["15551234567"]),
    )

    original_pair = gateway_commands.run_whatsapp_pairing
    original_is_paired = gateway_commands.is_whatsapp_paired

    def _fake_pair(cwd: Path, **_: object) -> None:
        session_dir = tmp_path / ".harness" / "gateway" / "whatsapp" / "session"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "creds.json").write_text("{}", encoding="utf-8")

    def _fake_is_paired(cwd: Path) -> bool:
        return (tmp_path / ".harness" / "gateway" / "whatsapp" / "session" / "creds.json").exists()

    gateway_commands.run_whatsapp_pairing = _fake_pair
    gateway_commands.is_whatsapp_paired = _fake_is_paired
    try:
        result = runner.invoke(
            cli_main.app,
            [
                "gateway",
                "whatsapp",
                "pair",
                "--cwd",
                str(tmp_path),
                "--no-install",
                "--json",
            ],
        )
    finally:
        gateway_commands.run_whatsapp_pairing = original_pair
        gateway_commands.is_whatsapp_paired = original_is_paired

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["paired"] is True
    assert payload["enabled"] is True


def test_gateway_converse_returns_chat_reply(tmp_path: Path) -> None:
    runner = CliRunner()

    async def _fake_converse(**kwargs: object) -> dict[str, object]:
        assert kwargs["transport"] == "whatsapp"
        assert kwargs["user_id"] == "15551234567"
        assert kwargs["thread_id"] == "15551234567@s.whatsapp.net"
        assert kwargs["message"] == "hello there"
        return {
            "reply": {
                "session_id": "gw-test",
                "command": "chat",
                "status": "ok",
                "text": "Hi from Harness",
                "data": {"harness_session_id": "sess_test"},
            },
            "session": {
                "id": "gw-test",
                "transport": "whatsapp",
                "user_id": "15551234567",
                "thread_id": "15551234567@s.whatsapp.net",
                "current_mission_id": "",
                "last_job_id": "",
                "last_run_id": "",
                "last_command": "chat",
                "updated_at": "2026-05-27T00:00:00+00:00",
                "metadata": {"harness_session_id": "sess_test"},
            },
        }

    original = gateway_commands._run_gateway_converse_payload
    gateway_commands._run_gateway_converse_payload = _fake_converse
    try:
        result = runner.invoke(
            cli_main.app,
            [
                "gateway",
                "converse",
                "--cwd",
                str(tmp_path),
                "--transport",
                "whatsapp",
                "--user",
                "15551234567",
                "--thread",
                "15551234567@s.whatsapp.net",
                "--message",
                "hello there",
                "--json",
            ],
        )
    finally:
        gateway_commands._run_gateway_converse_payload = original

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["reply"]["command"] == "chat"
    assert payload["reply"]["text"] == "Hi from Harness"


def test_gateway_receive_routes_general_message_directly_to_converse(tmp_path: Path) -> None:
    runner = CliRunner()

    async def _fake_converse(**kwargs: object) -> dict[str, object]:
        assert kwargs["message"] == "hello there"
        return {
            "reply": {
                "session_id": "gw-test",
                "command": "chat",
                "status": "ok",
                "text": "Hi from receive",
                "data": {"harness_session_id": "sess_test"},
            },
            "session": {
                "id": "gw-test",
                "transport": "whatsapp",
                "user_id": "15551234567",
                "thread_id": "15551234567@s.whatsapp.net",
                "current_mission_id": "",
                "last_job_id": "",
                "last_run_id": "",
                "last_command": "chat",
                "updated_at": "2026-05-27T00:00:00+00:00",
                "metadata": {"harness_session_id": "sess_test"},
            },
        }

    original = gateway_commands._run_gateway_receive_payload
    gateway_commands._run_gateway_receive_payload = _fake_converse
    try:
        result = runner.invoke(
            cli_main.app,
            [
                "gateway",
                "receive",
                "--cwd",
                str(tmp_path),
                "--transport",
                "whatsapp",
                "--user",
                "15551234567",
                "--thread",
                "15551234567@s.whatsapp.net",
                "--message",
                "hello there",
                "--json",
            ],
        )
    finally:
        gateway_commands._run_gateway_receive_payload = original

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["reply"]["command"] == "chat"
    assert payload["reply"]["text"] == "Hi from receive"


def test_gateway_receive_routes_control_message_to_dispatch(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(
        cli_main.app,
        [
            "gateway",
            "receive",
            "--cwd",
            str(tmp_path),
            "--transport",
            "whatsapp",
            "--user",
            "15551234567",
            "--thread",
            "15551234567@s.whatsapp.net",
            "--message",
            "status",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["reply"]["command"] == "status"


def test_run_gateway_conversation_uses_core_runtime_for_all_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="openrouter",
            model="google/gemma-4-31b-it",
            mode="self-chat",
            allowed_users=["15551234567"],
        ),
    )
    session_store = GatewaySessionStore(root=default_gateway_root(tmp_path))
    captured: dict[str, object] = {}

    async def _fake_chat_turn(**kwargs: object) -> str:
        captured.update(kwargs)
        return "Core runtime answer"

    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", _fake_chat_turn)

    payload = cast(
        dict[str, Any],
        asyncio.run(
            gateway_runtime._run_gateway_conversation(
                cwd=tmp_path,
                session_store=session_store,
                transport="whatsapp",
                user_id="15551234567",
                thread_id="15551234567@s.whatsapp.net",
                message="What is the weather in Tokyo?",
            )
        ),
    )

    assert payload["reply"]["text"] == "Core runtime answer"
    assert captured["prompt"] == "What is the weather in Tokyo?"
    assert captured["chain"] == ["openrouter"]
    assert captured["model"] == "google/gemma-4-31b-it"
    assert "general-purpose AI work agent" in cast(str, captured["system_prompt"])
    session_payload = cast(dict[str, Any], payload["session"])
    assert "_general_openrouter-" in session_payload["metadata"]["harness_session_id"]
    assert "workflow_id" not in payload["reply"]["data"]


def test_run_gateway_conversation_reuses_stable_core_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="openrouter",
            model="google/gemma-4-31b-it",
            mode="self-chat",
            allowed_users=["15551234567"],
        ),
    )
    session_store = GatewaySessionStore(root=default_gateway_root(tmp_path))
    session_ids: list[str] = []

    async def _fake_chat_turn(**kwargs: object) -> str:
        session_ids.append(cast(str, kwargs["session_id"]))
        return "ok"

    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", _fake_chat_turn)

    for message in ["Hello", "What is the time in Tokyo?"]:
        asyncio.run(
            gateway_runtime._run_gateway_conversation(
                cwd=tmp_path,
                session_store=session_store,
                transport="whatsapp",
                user_id="15551234567",
                thread_id="15551234567@s.whatsapp.net",
                message=message,
            )
        )

    assert len(session_ids) == 2
    assert session_ids[0] == session_ids[1]


def test_run_gateway_conversation_includes_shared_user_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_store = GatewaySessionStore(root=default_gateway_root(tmp_path))
    profile = session_store.get_or_create_profile(transport="whatsapp", user_id="15551234567")
    session_store.save_profile(
        profile.__class__.from_dict(
            {
                **profile.to_dict(),
                "active_work": [
                    {
                        "ref": "job:weather",
                        "kind": "reminder",
                        "title": "continue the Tokyo weather follow-up",
                        "summary": "Okay. I'll remind you in 5 minute(s): continue the Tokyo weather follow-up",
                        "source_thread_id": "thread-a",
                    }
                ],
                "recent_threads": ["thread-a"],
            }
        )
    )
    thread_a = session_store.get_or_create_session(
        transport="whatsapp",
        user_id="15551234567",
        thread_id="thread-a",
    )
    session_store.save_session(
        thread_a.__class__(
            **{
                **thread_a.to_dict(),
                "metadata": {
                    "thread_summary": (
                        "Latest user ask: I started the Tokyo weather work. "
                        "Last reply: The current weather in Tokyo is clear and 22°C."
                    )
                },
            }
        )
    )
    captured: dict[str, object] = {}

    async def _fake_chat_turn(**kwargs: object) -> str:
        captured.update(kwargs)
        return "shared context ok"

    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", _fake_chat_turn)

    payload = cast(
        dict[str, Any],
        asyncio.run(
            gateway_runtime._run_gateway_conversation(
                cwd=tmp_path,
                session_store=session_store,
                transport="whatsapp",
                user_id="15551234567",
                thread_id="thread-b",
                message="continue that work and also check New York",
            )
        ),
    )

    prompt = cast(str, captured["prompt"])
    assert "Continuity context only. It is not verified evidence" in prompt
    assert "Shared active work for this user:" in prompt
    assert "continue the Tokyo weather follow-up [reminder]" in prompt
    assert "Other recent chats for this user:" in prompt
    assert "thread-a: Latest user ask: I started the Tokyo weather work." in prompt
    assert "The current weather in Tokyo is clear and 22°C" not in prompt
    assert "User message:\ncontinue that work and also check New York" in prompt
    assert payload["reply"]["text"] == "shared context ok"


def test_run_gateway_conversation_compacts_thread_context_and_runtime_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="openrouter",
            model="google/gemma-4-31b-it",
            mode="self-chat",
            allowed_users=["15551234567"],
        ),
    )
    session_store = GatewaySessionStore(root=default_gateway_root(tmp_path))
    session = session_store.get_or_create_session(
        transport="whatsapp",
        user_id="15551234567",
        thread_id="15551234567@s.whatsapp.net",
    )
    session_store.save_session(
        session.__class__(
            **{
                **session.to_dict(),
                "metadata": {
                    **session.metadata,
                    "thread_context": ["assistant: " + ("long response " * 200)],
                    "harness_session_id_openrouter-4bd07e53_chat-deadbeef": "old",
                },
            }
        )
    )

    async def _fake_chat_turn(**_kwargs: object) -> str:
        return "reply " + ("body " * 300)

    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", _fake_chat_turn)

    asyncio.run(
        gateway_runtime._run_gateway_conversation(
            cwd=tmp_path,
            session_store=session_store,
            transport="whatsapp",
            user_id="15551234567",
            thread_id="15551234567@s.whatsapp.net",
            message="Hello",
        )
    )

    updated = session_store.get_or_create_session(
        transport="whatsapp",
        user_id="15551234567",
        thread_id="15551234567@s.whatsapp.net",
    )
    assert "harness_session_id_openrouter-4bd07e53_chat-deadbeef" not in (updated.metadata)
    context = updated.metadata["thread_context"]
    assert all(len(line) < 650 for line in context)
    assert context[-1].endswith("...")


def test_gateway_core_turn_uses_core_runner_with_tools_prediction_and_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harness.cli import run_commands

    captured: dict[str, object] = {}

    async def _fake_run_once(**kwargs: object) -> str:
        captured.update(kwargs)
        return "Plain sourced answer."

    monkeypatch.setattr(run_commands, "run_once", _fake_run_once)

    result = asyncio.run(
        gateway_runtime._run_gateway_chat_turn(
            cwd=tmp_path,
            prompt="What is happening with the service?",
            chain=["openrouter"],
            model="google/gemma-4-31b-it",
            session_id="sess-test",
            max_steps=8,
            config=object(),
            system_prompt="system prompt",
        )
    )

    assert result == "Plain sourced answer."
    assert captured["chain"] == ["openrouter"]
    assert captured["domain"] == "coding"
    assert captured["require_tools"] is False
    assert captured["verify"] == "auto"
    assert captured["predict"] is True
    assert captured["max_repair"] == 3
    assert captured["profile"] == "adaptive"
    assert captured["include_workspace_context"] is False
    assert captured["prompt"] == "What is happening with the service?"


def test_gateway_run_failure_reply_reports_rate_limit_without_configuration_blame() -> None:
    result = gateway_runtime._run_failure_reply_from_activity(
        [
            ActivityEvent(
                session_id="sess",
                kind="agent_run.failed",
                data={
                    "kind": "rate_limit",
                    "error": "OpenRouter rate-limited (429). Body: upstream limit",
                },
            )
        ]
    )

    assert result is not None
    assert "provider rate limit" in result
    assert "core runtime retried" in result
    assert "Configure a working model/provider" not in result


def test_gateway_verification_failure_reply_reports_core_blocker() -> None:
    result = gateway_runtime._verification_failure_reply_from_activity(
        [
            ActivityEvent(
                session_id="sess",
                kind="verification.completed",
                data={
                    "can_finish": False,
                    "reason": (
                        "The latest passing verify_work after the final state change "
                        "did not run a meaningful test/check command tied to the changed work."
                    ),
                },
            )
        ]
    )

    assert result is not None
    assert "core verification blocked" in result
    assert "meaningful test/check command" in result
    assert "Configure a working model/provider" not in result


def test_gateway_verification_failure_reply_ignores_stale_blocker_after_success() -> None:
    result = gateway_runtime._verification_failure_reply_from_activity(
        [
            ActivityEvent(
                session_id="sess",
                kind="verification.completed",
                data={
                    "can_finish": False,
                    "reason": "verify_work was missing after the edit.",
                },
            ),
            ActivityEvent(
                session_id="sess",
                kind="verification.completed",
                data={
                    "can_finish": True,
                    "reason": "latest verify_work passed after the edit.",
                },
            ),
        ]
    )

    assert result is None


def test_gateway_successful_verified_reply_uses_latest_verify_work_evidence() -> None:
    result = gateway_runtime._successful_verified_reply_from_activity(
        [
            ActivityEvent(
                session_id="sess",
                kind="tool_call.completed",
                data={
                    "name": "write_file",
                    "is_error": False,
                    "arguments": {"path": "slug_cli.py"},
                    "content_preview": "wrote 700 bytes to slug_cli.py",
                },
            ),
            ActivityEvent(
                session_id="sess",
                kind="tool_call.completed",
                data={
                    "name": "verify_work",
                    "is_error": False,
                    "arguments": {"command": "python3 test_slug.py"},
                    "content_preview": "PASSED\n\nAll tests passed!",
                },
            ),
            ActivityEvent(
                session_id="sess",
                kind="verification.completed",
                data={
                    "can_finish": True,
                    "reason": "verified with verify_work",
                },
            ),
        ]
    )

    assert result is not None
    assert "verify_work" in result
    assert "python3 test_slug.py" in result
    assert "All tests passed" in result


def test_gateway_unverified_workspace_change_reply_reports_change_after_verifier() -> None:
    result = gateway_runtime._unverified_workspace_change_reply_from_activity(
        [
            ActivityEvent(
                session_id="sess",
                kind="tool_call.completed",
                data={
                    "name": "write_file",
                    "is_error": False,
                    "arguments": {"path": "weather.py"},
                },
            ),
            ActivityEvent(
                session_id="sess",
                kind="verification.completed",
                data={
                    "can_finish": False,
                    "reason": "verify_work was missing after the edit.",
                },
            ),
            ActivityEvent(
                session_id="sess",
                kind="tool_call.completed",
                data={
                    "name": "write_file",
                    "is_error": False,
                    "arguments": {"path": "test_weather.py"},
                },
            ),
        ]
    )

    assert result is not None
    assert "after the final workspace change" in result
    assert "`write_file` on `test_weather.py`" in result


def test_gateway_unverified_workspace_change_reply_allows_current_verifier() -> None:
    result = gateway_runtime._unverified_workspace_change_reply_from_activity(
        [
            ActivityEvent(
                session_id="sess",
                kind="tool_call.completed",
                data={
                    "name": "write_file",
                    "is_error": False,
                    "arguments": {"path": "weather.py"},
                },
            ),
            ActivityEvent(
                session_id="sess",
                kind="verification.completed",
                data={
                    "can_finish": False,
                    "reason": "verify_work was missing after the edit.",
                },
            ),
        ]
    )

    assert result is None


def test_run_gateway_conversation_rate_limit_uses_core_failure_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="openrouter",
            model="google/gemma-4-31b-it",
            mode="self-chat",
            allowed_users=["15551234567"],
        ),
    )
    session_store = GatewaySessionStore(root=default_gateway_root(tmp_path))

    async def _rate_limited_chat(**_kwargs: object) -> str:
        raise typer.Exit(1)

    async def _rate_limit_reply(**_kwargs: object) -> str:
        return (
            "Harness hit a provider rate limit while generating this reply. "
            "The core runtime retried the request; please try again shortly."
        )

    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", _rate_limited_chat)
    monkeypatch.setattr(gateway_runtime, "_latest_run_failure_reply", _rate_limit_reply)

    payload = cast(
        dict[str, Any],
        asyncio.run(
            gateway_runtime._run_gateway_conversation(
                cwd=tmp_path,
                session_store=session_store,
                transport="whatsapp",
                user_id="15551234567",
                thread_id="15551234567@s.whatsapp.net",
                message="Hello there",
            )
        ),
    )

    assert "provider rate limit" in payload["reply"]["text"]
    assert "Configure a working model/provider" not in payload["reply"]["text"]


def test_run_gateway_conversation_verification_failure_uses_core_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="openrouter",
            model="google/gemma-4-31b-it",
            mode="self-chat",
            allowed_users=["15551234567"],
        ),
    )
    session_store = GatewaySessionStore(root=default_gateway_root(tmp_path))

    async def _blocked_chat(**_kwargs: object) -> str:
        raise typer.Exit(1)

    async def _no_run_failure(**_kwargs: object) -> None:
        return None

    async def _verification_failure(**_kwargs: object) -> str:
        return (
            "Harness core verification blocked the final reply: "
            "the latest verify_work did not validate the changed work."
        )

    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", _blocked_chat)
    monkeypatch.setattr(gateway_runtime, "_latest_run_failure_reply", _no_run_failure)
    monkeypatch.setattr(
        gateway_runtime,
        "_latest_verification_failure_reply",
        _verification_failure,
    )

    payload = cast(
        dict[str, Any],
        asyncio.run(
            gateway_runtime._run_gateway_conversation(
                cwd=tmp_path,
                session_store=session_store,
                transport="whatsapp",
                user_id="15551234567",
                thread_id="15551234567@s.whatsapp.net",
                message="Create the script and verify it.",
            )
        ),
    )

    assert "core verification blocked" in payload["reply"]["text"]
    assert "Configure a working model/provider" not in payload["reply"]["text"]


def test_run_gateway_conversation_timeout_prefers_verification_blocker_over_tool_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="openrouter",
            model="google/gemma-4-31b-it",
            mode="self-chat",
            allowed_users=["15551234567"],
        ),
    )
    session_store = GatewaySessionStore(root=default_gateway_root(tmp_path))

    async def _slow_chat(**_kwargs: object) -> str:
        await asyncio.Event().wait()
        return "never"

    async def _no_reply(**_kwargs: object) -> None:
        return None

    async def _verification_failure(**_kwargs: object) -> str:
        return (
            "Harness core verification blocked the final reply: "
            "You made file changes but never ran verify_work."
        )

    async def _misleading_tool_evidence(**_kwargs: object) -> str:
        return "I verified this with local shell evidence from `python3 script.py`."

    monkeypatch.setattr(gateway_runtime, "_gateway_turn_timeout_seconds", lambda: 0.01)
    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", _slow_chat)
    monkeypatch.setattr(gateway_runtime, "_latest_run_failure_reply", _no_reply)
    monkeypatch.setattr(
        gateway_runtime,
        "_latest_verification_failure_reply",
        _verification_failure,
    )
    monkeypatch.setattr(gateway_runtime, "_latest_tool_failure_reply", _no_reply)
    monkeypatch.setattr(
        gateway_runtime,
        "_latest_successful_tool_evidence_reply",
        _misleading_tool_evidence,
    )

    payload = cast(
        dict[str, Any],
        asyncio.run(
            gateway_runtime._run_gateway_conversation(
                cwd=tmp_path,
                session_store=session_store,
                transport="whatsapp",
                user_id="15551234567",
                thread_id="15551234567@s.whatsapp.net",
                message="Create the script and verify it.",
            )
        ),
    )

    assert "core verification blocked" in payload["reply"]["text"]
    assert "never ran verify_work" in payload["reply"]["text"]
    assert "local shell evidence" not in payload["reply"]["text"]


def test_run_gateway_conversation_timeout_prefers_unverified_change_over_stale_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="openrouter",
            model="google/gemma-4-31b-it",
            mode="self-chat",
            allowed_users=["15551234567"],
        ),
    )
    session_store = GatewaySessionStore(root=default_gateway_root(tmp_path))

    async def _slow_chat(**_kwargs: object) -> str:
        await asyncio.Event().wait()
        return "never"

    async def _no_reply(**_kwargs: object) -> None:
        return None

    async def _unverified_change(**_kwargs: object) -> str:
        return (
            "Harness stopped before completing verification after the final workspace "
            "change. Latest unverified change: `write_file` on `test_weather.py`."
        )

    async def _stale_verification(**_kwargs: object) -> str:
        return "Harness core verification blocked the final reply: stale verifier text."

    monkeypatch.setattr(gateway_runtime, "_gateway_turn_timeout_seconds", lambda: 0.01)
    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", _slow_chat)
    monkeypatch.setattr(gateway_runtime, "_latest_run_failure_reply", _no_reply)
    monkeypatch.setattr(
        gateway_runtime,
        "_latest_unverified_workspace_change_reply",
        _unverified_change,
    )
    monkeypatch.setattr(
        gateway_runtime,
        "_latest_verification_failure_reply",
        _stale_verification,
    )
    monkeypatch.setattr(gateway_runtime, "_latest_tool_failure_reply", _no_reply)

    payload = cast(
        dict[str, Any],
        asyncio.run(
            gateway_runtime._run_gateway_conversation(
                cwd=tmp_path,
                session_store=session_store,
                transport="whatsapp",
                user_id="15551234567",
                thread_id="15551234567@s.whatsapp.net",
                message="Create the script and verify it.",
            )
        ),
    )

    assert "after the final workspace change" in payload["reply"]["text"]
    assert "test_weather.py" in payload["reply"]["text"]
    assert "stale verifier text" not in payload["reply"]["text"]


def test_run_gateway_conversation_handoff_uses_successful_verified_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="openrouter",
            model="google/gemma-4-31b-it",
            mode="self-chat",
            allowed_users=["15551234567"],
        ),
    )
    session_store = GatewaySessionStore(root=default_gateway_root(tmp_path))

    async def _handoff_chat(**_kwargs: object) -> str:
        return gateway_runtime._RUNTIME_VERIFICATION_HANDOFF_TEXT

    async def _no_reply(**_kwargs: object) -> None:
        return None

    async def _verified_reply(**_kwargs: object) -> str:
        return "I verified this with `verify_work` using `python3 test_slug.py`: PASSED"

    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", _handoff_chat)
    monkeypatch.setattr(
        gateway_runtime,
        "_latest_unverified_workspace_change_reply",
        _no_reply,
    )
    monkeypatch.setattr(gateway_runtime, "_latest_verification_failure_reply", _no_reply)
    monkeypatch.setattr(
        gateway_runtime,
        "_latest_successful_verified_reply",
        _verified_reply,
    )

    payload = cast(
        dict[str, Any],
        asyncio.run(
            gateway_runtime._run_gateway_conversation(
                cwd=tmp_path,
                session_store=session_store,
                transport="whatsapp",
                user_id="15551234567",
                thread_id="15551234567@s.whatsapp.net",
                message="Create the script and verify it.",
            )
        ),
    )

    assert "verify_work" in payload["reply"]["text"]
    assert "test_slug.py" in payload["reply"]["text"]
    assert "Handing the current state" not in payload["reply"]["text"]


def test_run_gateway_conversation_without_whatsapp_config_uses_openrouter_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    session_store = GatewaySessionStore(root=default_gateway_root(tmp_path))
    captured: dict[str, object] = {}

    async def _fake_chat_turn(**kwargs: object) -> str:
        captured.update(kwargs)
        return "Tokyo weather answer"

    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", _fake_chat_turn)

    asyncio.run(
        gateway_runtime._run_gateway_conversation(
            cwd=tmp_path,
            session_store=session_store,
            transport="whatsapp",
            user_id="15551234567",
            thread_id="15551234567@s.whatsapp.net",
            message="What is the weather in Tokyo?",
        )
    )

    assert captured["chain"] == ["openrouter"]
    assert captured["model"] == "openai/gpt-5.4-nano"


def test_run_gateway_conversation_honors_provider_model_env_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("HARNESS_GATEWAY_PROVIDER", "openrouter")
    monkeypatch.setenv("HARNESS_GATEWAY_MODEL", "anthropic/claude-3.5-sonnet")
    session_store = GatewaySessionStore(root=default_gateway_root(tmp_path))
    captured: dict[str, object] = {}

    async def _fake_chat_turn(**kwargs: object) -> str:
        captured.update(kwargs)
        return "Done"

    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", _fake_chat_turn)

    asyncio.run(
        gateway_runtime._run_gateway_conversation(
            cwd=tmp_path,
            session_store=session_store,
            transport="whatsapp",
            user_id="15551234567",
            thread_id="15551234567@s.whatsapp.net",
            message="Do the task.",
        )
    )

    assert captured["chain"] == ["openrouter"]
    assert captured["model"] == "anthropic/claude-3.5-sonnet"


def test_run_gateway_conversation_times_out_before_bridge_child_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gateway_runtime, "_GATEWAY_TURN_TIMEOUT_SECONDS", 0.01)
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="openrouter",
            model="google/gemma-4-31b-it",
            mode="self-chat",
            allowed_users=["15551234567"],
        ),
    )
    session_store = GatewaySessionStore(root=default_gateway_root(tmp_path))

    async def _slow_chat_turn(**_kwargs: object) -> str:
        await asyncio.Event().wait()
        return "never"

    monkeypatch.setattr(gateway_runtime, "_run_gateway_chat_turn", _slow_chat_turn)

    payload = cast(
        dict[str, Any],
        asyncio.run(
            asyncio.wait_for(
                gateway_runtime._run_gateway_conversation(
                    cwd=tmp_path,
                    session_store=session_store,
                    transport="whatsapp",
                    user_id="15551234567",
                    thread_id="15551234567@s.whatsapp.net",
                    message="Yo my guy",
                ),
                timeout=1.0,
            )
        ),
    )

    assert "timed out while waiting for the model" in payload["reply"]["text"]
