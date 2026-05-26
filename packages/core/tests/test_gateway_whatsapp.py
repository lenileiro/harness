from __future__ import annotations

import json
from unittest.mock import patch

from harness.core.gateway_whatsapp import extract_whatsapp_messages, send_whatsapp_text_message


def test_extract_whatsapp_messages_reads_text_payloads() -> None:
    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "field": "messages",
                        "value": {
                            "metadata": {"phone_number_id": "phone-123"},
                            "contacts": [
                                {"wa_id": "15551234567", "profile": {"name": "Tester"}},
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

    messages = extract_whatsapp_messages(payload)
    assert len(messages) == 1
    assert messages[0].transport == "whatsapp"
    assert messages[0].user_id == "15551234567"
    assert messages[0].thread_id == "phone-123"
    assert messages[0].text == "status"
    assert messages[0].metadata["profile_name"] == "Tester"


def test_send_whatsapp_text_message_posts_cloud_api_request() -> None:
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self):  # type: ignore[override]
            return self

        def __exit__(self, exc_type, exc, tb):  # type: ignore[override]
            return False

        def read(self) -> bytes:
            return json.dumps({"messages": [{"id": "wamid.outbound"}]}).encode("utf-8")

    def _fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode("utf-8"))
        captured["auth"] = req.headers.get("Authorization")
        captured["timeout"] = timeout
        return _Response()

    with patch("harness.core.gateway_whatsapp.request.urlopen", _fake_urlopen):
        payload = send_whatsapp_text_message(
            to="15551234567",
            text="hello",
            phone_number_id="phone-123",
            access_token="token-abc",
            api_version="v22.0",
        )

    assert payload["messages"][0]["id"] == "wamid.outbound"
    assert captured["url"] == "https://graph.facebook.com/v22.0/phone-123/messages"
    assert captured["body"] == {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": "15551234567",
        "type": "text",
        "text": {"preview_url": False, "body": "hello"},
    }
    assert captured["auth"] == "Bearer token-abc"
