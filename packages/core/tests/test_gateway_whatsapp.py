from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from harness.core.gateway_whatsapp import (
    WhatsAppBridgeConfig,
    build_whatsapp_bridge_env,
    ensure_whatsapp_bridge_project,
    load_whatsapp_bridge_config,
    read_whatsapp_bridge_status,
    save_whatsapp_bridge_config,
    send_whatsapp_text_message,
)


def test_whatsapp_bridge_config_roundtrip(tmp_path: Path) -> None:
    config = WhatsAppBridgeConfig(
        enabled=True,
        provider="ollama",
        model="gemma4:latest",
        mode="self-chat",
        allowed_users=["15551234567"],
        bridge_port=9901,
        reply_prefix="Harness\n",
    )
    save_whatsapp_bridge_config(tmp_path, config)

    loaded = load_whatsapp_bridge_config(tmp_path)
    assert loaded == config


def test_ensure_whatsapp_bridge_project_writes_assets(tmp_path: Path) -> None:
    project_dir = ensure_whatsapp_bridge_project(tmp_path)
    assert (project_dir / "package.json").exists()
    assert (project_dir / "bridge.js").exists()
    package_json = (project_dir / "package.json").read_text(encoding="utf-8")
    bridge_js = (project_dir / "bridge.js").read_text(encoding="utf-8")
    assert "link-preview-js" in package_json
    assert "messages.upsert" in bridge_js
    assert "'dispatch'" in bridge_js
    assert "'converse'" in bridge_js
    assert "ownIdentityCandidates" in bridge_js
    assert "chatId.endsWith('@g.us')" in bridge_js
    assert "BRIDGE_STARTED_AT_MS" in bridge_js
    assert "processedMessageIds" in bridge_js
    assert "MAX_GATEWAY_CONCURRENCY" in bridge_js
    assert "pendingGatewayTasks" in bridge_js
    assert "enqueueGatewayTask" in bridge_js
    assert "GATEWAY_CHILD_TIMEOUT_MS" in bridge_js
    assert "GATEWAY_OUTPUT_LIMIT_BYTES" in bridge_js
    assert "workspace_cwd" in bridge_js
    assert "active_gateway_tasks" in bridge_js


def test_build_whatsapp_bridge_env_includes_workspace_and_uv(tmp_path: Path) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="ollama",
            model="gemma4:latest",
            mode="self-chat",
            allowed_users=["15551234567"],
            bridge_port=8741,
            max_gateway_concurrency=1,
            max_gateway_queue=2,
            gateway_child_timeout_seconds=90,
        ),
    )
    env = build_whatsapp_bridge_env(tmp_path)
    assert env["HARNESS_WHATSAPP_MODE"] == "self-chat"
    assert env["HARNESS_WHATSAPP_ALLOWED_USERS"] == "15551234567"
    assert env["HARNESS_WHATSAPP_WORKSPACE_CWD"] == str(tmp_path.resolve())
    assert env["HARNESS_WHATSAPP_UV_BIN"]
    assert env["HARNESS_WHATSAPP_ENV_FILE"] == ""
    assert env["HARNESS_WHATSAPP_MAX_CONCURRENCY"] == "1"
    assert env["HARNESS_WHATSAPP_MAX_QUEUE"] == "2"
    assert env["HARNESS_WHATSAPP_CHILD_TIMEOUT_MS"] == "90000"


def test_build_whatsapp_bridge_env_exposes_dotenv_path_when_present(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("OPENROUTER_API_KEY=test-key\n", encoding="utf-8")
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="openrouter",
            model="google/gemma-4-31b-it",
            mode="self-chat",
            allowed_users=["15551234567"],
            bridge_port=8741,
        ),
    )
    env = build_whatsapp_bridge_env(tmp_path)
    assert env["HARNESS_WHATSAPP_ENV_FILE"] == str((tmp_path / ".env").resolve())


def test_read_whatsapp_bridge_status_reports_defaults(tmp_path: Path) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=False,
            provider="ollama",
            model="gemma4:latest",
            mode="self-chat",
            allowed_users=[],
            bridge_port=19841,
        ),
    )
    status = read_whatsapp_bridge_status(tmp_path)
    assert status.config.mode == "self-chat"
    assert status.paired is False
    assert status.bridge_running is False
    assert status.bridge_connected is False


def test_probe_whatsapp_bridge_rejects_other_workspace_on_same_port(tmp_path: Path) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="ollama",
            model="gemma4:latest",
            mode="self-chat",
            allowed_users=[],
            bridge_port=9913,
        ),
    )

    class _Response:
        def __enter__(self):  # type: ignore[override]
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[override]
            return False

        def read(self) -> bytes:
            return json.dumps(
                {
                    "status": "connected",
                    "workspace_cwd": str((tmp_path / "other").resolve()),
                }
            ).encode("utf-8")

    with patch("harness.core.gateway_whatsapp.request.urlopen", lambda req, timeout=0: _Response()):
        status = read_whatsapp_bridge_status(tmp_path)

    assert status.bridge_running is False
    assert status.bridge_connected is False


def test_send_whatsapp_text_message_posts_to_local_bridge(tmp_path: Path) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(
            enabled=True,
            provider="ollama",
            model="gemma4:latest",
            mode="self-chat",
            allowed_users=[],
            bridge_port=9912,
        ),
    )
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):  # type: ignore[override]
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[override]
            return False

        def read(self) -> bytes:
            if captured.get("health_checked"):
                return json.dumps({"ok": True, "messageId": "wamid.local"}).encode("utf-8")
            captured["health_checked"] = True
            return json.dumps(
                {
                    "status": "connected",
                    "workspace_cwd": str(tmp_path.resolve()),
                }
            ).encode("utf-8")

    def _fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        if req.data is not None:
            captured["body"] = json.loads(req.data.decode("utf-8"))
        return _Response()

    with patch("harness.core.gateway_whatsapp.request.urlopen", _fake_urlopen):
        payload = send_whatsapp_text_message(cwd=tmp_path, to="15551234567", text="hello")

    assert payload["ok"] is True
    assert captured["url"] == "http://127.0.0.1:9912/send"
    assert captured["body"] == {
        "chatId": "15551234567",
        "message": "hello",
    }
