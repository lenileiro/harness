from __future__ import annotations

import json
import os
from typing import Any
from urllib import request

from harness.core.gateway_models import GatewayMessage


def extract_whatsapp_messages(payload: dict[str, Any]) -> list[GatewayMessage]:
    messages: list[GatewayMessage] = []
    entries = payload.get("entry", [])
    if not isinstance(entries, list):
        return messages
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        changes = entry.get("changes", [])
        if not isinstance(changes, list):
            continue
        for change in changes:
            if not isinstance(change, dict):
                continue
            value = change.get("value", {})
            if not isinstance(value, dict):
                continue
            metadata = value.get("metadata", {})
            phone_number_id = ""
            if isinstance(metadata, dict):
                phone_number_id = str(metadata.get("phone_number_id", "")).strip()
            contacts_by_wa_id: dict[str, dict[str, Any]] = {}
            contacts = value.get("contacts", [])
            if isinstance(contacts, list):
                for contact in contacts:
                    if not isinstance(contact, dict):
                        continue
                    wa_id = str(contact.get("wa_id", "")).strip()
                    if wa_id:
                        contacts_by_wa_id[wa_id] = contact
            raw_messages = value.get("messages", [])
            if not isinstance(raw_messages, list):
                continue
            for item in raw_messages:
                if not isinstance(item, dict):
                    continue
                if str(item.get("type", "")).strip() != "text":
                    continue
                text_block = item.get("text", {})
                if not isinstance(text_block, dict):
                    continue
                body = str(text_block.get("body", "")).strip()
                if not body:
                    continue
                from_number = str(item.get("from", "")).strip()
                message_id = str(item.get("id", "")).strip()
                contact = contacts_by_wa_id.get(from_number, {})
                profile = contact.get("profile", {}) if isinstance(contact, dict) else {}
                profile_name = (
                    str(profile.get("name", "")).strip() if isinstance(profile, dict) else ""
                )
                messages.append(
                    GatewayMessage(
                        id=message_id or f"whatsapp-{from_number}",
                        transport="whatsapp",
                        user_id=from_number,
                        thread_id=phone_number_id or from_number,
                        text=body,
                        metadata={
                            "profile_name": profile_name,
                            "phone_number_id": phone_number_id,
                            "timestamp": str(item.get("timestamp", "")).strip(),
                        },
                    )
                )
    return messages


def send_whatsapp_text_message(
    *,
    to: str,
    text: str,
    phone_number_id: str | None = None,
    access_token: str | None = None,
    api_version: str | None = None,
    base_url: str = "https://graph.facebook.com",
) -> dict[str, Any]:
    resolved_phone_number_id = (
        phone_number_id or os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "")
    ).strip()
    resolved_access_token = (access_token or os.environ.get("WHATSAPP_ACCESS_TOKEN", "")).strip()
    resolved_api_version = (api_version or os.environ.get("WHATSAPP_API_VERSION", "v22.0")).strip()
    if not resolved_phone_number_id:
        raise ValueError("WhatsApp phone number id is required.")
    if not resolved_access_token:
        raise ValueError("WhatsApp access token is required.")
    body = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": text,
        },
    }
    url = f"{base_url.rstrip('/')}/{resolved_api_version}/{resolved_phone_number_id}/messages"
    req = request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {resolved_access_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=30) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


__all__ = ["extract_whatsapp_messages", "send_whatsapp_text_message"]
