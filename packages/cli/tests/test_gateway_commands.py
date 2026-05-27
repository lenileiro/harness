from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, cast

from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.cli import gateway_commands
from harness.core import (
    GatewaySessionStore,
    PendingApproval,
    ToolCall,
    ToolCallEvent,
    WhatsAppBridgeConfig,
    default_gateway_root,
    save_whatsapp_bridge_config,
)
from harness.storage.sqlite import SQLiteStorage


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
    assert payload["model"] == "google/gemma-4-31b-it"


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

    original = gateway_commands._run_gateway_conversation
    gateway_commands._run_gateway_conversation = _fake_converse
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
        gateway_commands._run_gateway_conversation = original

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["reply"]["command"] == "chat"
    assert payload["reply"]["text"] == "Hi from Harness"


def test_run_gateway_conversation_auto_approves_tool_work(tmp_path: Path) -> None:
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

    async def _fake_run_once(**kwargs: object) -> str:
        captured.update(kwargs)
        return "Done from chat"

    original = gateway_commands.__dict__["_run_once_impl"]
    gateway_commands.__dict__["_run_once_impl"] = _fake_run_once
    try:
        payload = cast(
            dict[str, Any],
            asyncio.run(
                gateway_commands._run_gateway_conversation(
                    cwd=tmp_path,
                    session_store=session_store,
                    transport="whatsapp",
                    user_id="15551234567",
                    thread_id="15551234567@s.whatsapp.net",
                    message="write a file",
                )
            ),
        )
    finally:
        gateway_commands.__dict__["_run_once_impl"] = original

    assert payload["reply"]["text"] == "Done from chat"
    assert captured["yes"] is True
    assert captured["inbox"] is False


def test_run_gateway_conversation_sends_event_driven_progress_updates(tmp_path: Path) -> None:
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
    seeded = session_store.get_or_create_session(
        transport="whatsapp",
        user_id="15551234567",
        thread_id="15551234567@s.whatsapp.net",
    )
    session_store.save_session(
        seeded.__class__(
            **{
                **seeded.to_dict(),
                "metadata": {
                    **seeded.metadata,
                    "thread_context": [
                        "user: can you make the script?",
                        "assistant: I am starting with the file layout.",
                    ],
                },
            }
        )
    )
    sent: list[str] = []

    async def _fake_run_once(**kwargs: object) -> str:
        render = cast(Any, kwargs)["render"]
        assert callable(render)
        render(
            ToolCallEvent(
                call=ToolCall(
                    id="call_write",
                    name="write_file",
                    arguments={"path": "weather_tokyo.py", "content": "print('hi')"},
                )
            )
        )
        render(
            ToolCallEvent(
                call=ToolCall(
                    id="call_verify",
                    name="verify_work",
                    arguments={"command": "uv run python weather_tokyo.py"},
                )
            )
        )
        await asyncio.sleep(0)
        return "Completed."

    def _fake_send(*, cwd: Path | None = None, to: str, text: str, reply_to: str | None = None):
        assert cwd == tmp_path
        assert to == "15551234567"
        assert reply_to is None
        sent.append(text)
        return {"ok": True, "messageId": "wamid.local"}

    async def _fake_progress_note(
        *,
        provider: str,
        model: str,
        config: Any,
        thread_context: list[str],
        user_prompt: str,
        event_summary: str,
        work_context: list[str],
    ) -> str:
        assert provider == "openrouter"
        assert model == "google/gemma-4-31b-it"
        assert thread_context
        assert "assistant: I am starting with the file layout." in thread_context
        assert user_prompt == "make the script"
        assert work_context
        assert "Writing weather_tokyo.py." in work_context
        return f"LLM progress: {event_summary}"

    original_run_once = gateway_commands.__dict__["_run_once_impl"]
    original_send = gateway_commands.send_whatsapp_text_message
    original_progress_note = gateway_commands._generate_progress_note
    gateway_commands.__dict__["_run_once_impl"] = _fake_run_once
    gateway_commands.send_whatsapp_text_message = _fake_send
    gateway_commands._generate_progress_note = _fake_progress_note
    try:
        payload = cast(
            dict[str, Any],
            asyncio.run(
                gateway_commands._run_gateway_conversation(
                    cwd=tmp_path,
                    session_store=session_store,
                    transport="whatsapp",
                    user_id="15551234567",
                    thread_id="15551234567@s.whatsapp.net",
                    message="make the script",
                )
            ),
        )
    finally:
        gateway_commands.__dict__["_run_once_impl"] = original_run_once
        gateway_commands.send_whatsapp_text_message = original_send
        gateway_commands._generate_progress_note = original_progress_note

    assert payload["reply"]["text"] == "Completed."
    assert "LLM progress: Writing weather_tokyo.py." in sent
