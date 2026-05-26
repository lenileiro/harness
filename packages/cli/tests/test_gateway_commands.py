from __future__ import annotations

import asyncio
import json
from pathlib import Path

from typer.testing import CliRunner

from harness.cli import __main__ as cli_main
from harness.core import PendingApproval
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


def test_gateway_whatsapp_dispatch_routes_incoming_text_payload(tmp_path: Path) -> None:
    runner = CliRunner()
    payload_path = tmp_path / "whatsapp.json"
    payload_path.write_text(
        json.dumps(
            {
                "entry": [
                    {
                        "changes": [
                            {
                                "field": "messages",
                                "value": {
                                    "metadata": {"phone_number_id": "phone-123"},
                                    "contacts": [
                                        {
                                            "wa_id": "15551234567",
                                            "profile": {"name": "Tester"},
                                        }
                                    ],
                                    "messages": [
                                        {
                                            "from": "15551234567",
                                            "id": "wamid.1",
                                            "timestamp": "1710000000",
                                            "type": "text",
                                            "text": {"body": "status"},
                                        }
                                    ],
                                },
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli_main.app,
        [
            "gateway",
            "whatsapp-dispatch",
            "--payload",
            str(payload_path),
            "--cwd",
            str(tmp_path),
            "--in-memory",
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload[0]["message"]["transport"] == "whatsapp"
    assert payload[0]["reply"]["command"] == "status"
    assert payload[0]["session"]["user_id"] == "15551234567"


def test_gateway_whatsapp_dispatch_can_grant_approval(tmp_path: Path) -> None:
    runner = CliRunner()
    db_path = tmp_path / "whatsapp-gateway.db"

    async def _seed() -> str:
        storage = SQLiteStorage(path=db_path)
        try:
            saved = await storage.create_approval(
                PendingApproval(
                    session_id="sess_gateway",
                    tool_call_id="tool_call_2",
                    tool_name="shell",
                    arguments={"cmd": "echo from whatsapp"},
                )
            )
            return saved.id
        finally:
            await storage.close()

    approval_id = asyncio.run(_seed())
    payload_path = tmp_path / "whatsapp-approve.json"
    payload_path.write_text(
        json.dumps(
            {
                "entry": [
                    {
                        "changes": [
                            {
                                "field": "messages",
                                "value": {
                                    "metadata": {"phone_number_id": "phone-123"},
                                    "contacts": [
                                        {
                                            "wa_id": "15551234567",
                                            "profile": {"name": "Approver"},
                                        }
                                    ],
                                    "messages": [
                                        {
                                            "from": "15551234567",
                                            "id": "wamid.approve",
                                            "timestamp": "1710000001",
                                            "type": "text",
                                            "text": {"body": f"approve {approval_id}"},
                                        }
                                    ],
                                },
                            }
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    result = runner.invoke(
        cli_main.app,
        [
            "gateway",
            "whatsapp-dispatch",
            "--payload",
            str(payload_path),
            "--cwd",
            str(tmp_path),
            "--db",
            str(db_path),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload[0]["reply"]["command"] == "approve"
    assert payload[0]["reply"]["status"] == "ok"

    async def _load_status() -> str:
        storage = SQLiteStorage(path=db_path)
        try:
            approval = await storage.get_approval(approval_id)
            assert approval is not None
            return approval.status
        finally:
            await storage.close()

    assert asyncio.run(_load_status()) == "granted"
