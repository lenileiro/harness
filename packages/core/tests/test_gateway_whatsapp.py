from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from harness.core.gateway_whatsapp import (
    WhatsAppBridgeConfig,
    ensure_whatsapp_bridge_project,
    load_whatsapp_bridge_config,
    read_whatsapp_bridge_status,
    save_whatsapp_bridge_config,
    send_whatsapp_text_message,
)


def test_whatsapp_bridge_config_roundtrip(tmp_path: Path) -> None:
    config = WhatsAppBridgeConfig(
        enabled=True,
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


def test_read_whatsapp_bridge_status_reports_defaults(tmp_path: Path) -> None:
    status = read_whatsapp_bridge_status(tmp_path)
    assert status.config.mode == "self-chat"
    assert status.paired is False
    assert status.bridge_running is False
    assert status.bridge_connected is False


def test_send_whatsapp_text_message_posts_to_local_bridge(tmp_path: Path) -> None:
    save_whatsapp_bridge_config(
        tmp_path,
        WhatsAppBridgeConfig(enabled=True, mode="self-chat", allowed_users=[], bridge_port=9912),
    )
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):  # type: ignore[override]
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[override]
            return False

        def read(self) -> bytes:
            return json.dumps({"ok": True, "messageId": "wamid.local"}).encode("utf-8")

    def _fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["timeout"] = timeout
        return _Response()

    with patch("harness.core.gateway_whatsapp.request.urlopen", _fake_urlopen):
        payload = send_whatsapp_text_message(cwd=tmp_path, to="15551234567", text="hello")

    assert payload["ok"] is True
    assert captured["url"] == "http://127.0.0.1:9912/send"
    assert captured["body"] == {
        "chatId": "15551234567",
        "message": "hello",
    }
