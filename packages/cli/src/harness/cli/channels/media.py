"""Bounded platform media transfer; remote descriptors never become local paths."""

from __future__ import annotations

import base64
import json
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import httpx

from harness.core.gateway_channels import ChannelMessage
from harness.core.schemas import MediaAttachment


def media_descriptors(name: str, event: dict[str, Any]) -> list[dict[str, Any]]:
    if name == "telegram":
        items = []
        if event.get("photo"):
            items.append(
                {
                    "file_id": event["photo"][-1]["file_id"],
                    "mime_type": "image/jpeg",
                    "name": "photo.jpg",
                }
            )
        for kind in ("audio", "voice", "document", "video"):
            if event.get(kind):
                item = event[kind]
                items.append(
                    {
                        "file_id": item["file_id"],
                        "mime_type": item.get("mime_type")
                        or ("audio/ogg" if kind == "voice" else "application/octet-stream"),
                        "name": item.get("file_name") or kind,
                    }
                )
        return items
    if name == "discord":
        return [
            {
                "url": item["url"],
                "mime_type": item.get("content_type") or "application/octet-stream",
                "name": item.get("filename") or "attachment",
            }
            for item in event.get("attachments", [])
            if item.get("url")
        ][:16]
    if name == "slack":
        return [
            {
                "url": item["url_private_download"],
                "mime_type": item.get("mimetype") or "application/octet-stream",
                "name": item.get("name") or "attachment",
            }
            for item in event.get("files", [])
            if item.get("url_private_download")
        ][:16]
    return []


def validate_media_url(url: str, domains: tuple[str, ...]) -> None:
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or not parsed.hostname
        or not any(
            parsed.hostname == domain or parsed.hostname.endswith("." + domain)
            for domain in domains
        )
    ):
        raise ValueError("Media address is outside the authenticated platform")


async def prepare_media(transport: Any, message: ChannelMessage) -> list[MediaAttachment]:
    result = []
    total = 0
    for item in message.attachments[:16]:
        auth = ""
        if transport.name == "telegram":
            file = await transport.call("getFile", {"file_id": item["file_id"]})
            path = str(file["file_path"])
            if ".." in PurePosixPath(path).parts or path.startswith("/"):
                raise ValueError("Invalid Telegram media path")
            url = f"https://api.telegram.org/file/bot{transport.token}/{path}"
            domains = ("api.telegram.org",)
        elif transport.name == "discord":
            url, domains = str(item["url"]), ("cdn.discordapp.com", "media.discordapp.net")
        elif transport.name == "slack":
            url, domains = str(item["url"]), ("files.slack.com",)
            auth = f"Bearer {transport.token}"
        else:
            raise ValueError("Unsupported media transport")
        validate_media_url(url, domains)
        try:
            async with transport.client.stream(
                "GET", url, headers={"Authorization": auth} if auth else {}
            ) as response:
                if not response.is_success:
                    raise ValueError("Channel media download failed")
                raw = bytearray()
                async for chunk in response.aiter_bytes():
                    raw.extend(chunk)
                    total += len(chunk)
                    if total > 20 * 1024 * 1024:
                        raise ValueError("Channel attachments exceed 20 MiB per message")
        except httpx.HTTPError:
            raise ValueError("Channel media download interrupted") from None
        mime = str(item.get("mime_type") or "application/octet-stream")
        kind = (
            "image"
            if mime.startswith("image/")
            else "audio"
            if mime.startswith("audio/")
            else "file"
        )
        result.append(
            MediaAttachment(
                kind=kind,
                mime_type=mime,
                data=base64.b64encode(raw).decode(),
                name=PurePosixPath(str(item.get("name") or "attachment")).name[:255],
            )
        )
    return result


async def send_media(
    transport: Any, thread_id: str, attachment: MediaAttachment, delivery_id: str
) -> None:
    if attachment.data is None:
        raise ValueError("Outbound channel media requires inline data, not a remote URL")
    filename = PurePosixPath(attachment.name or "attachment").name
    raw = base64.b64decode(attachment.data, validate=True)
    upload = (filename, raw, attachment.mime_type)
    if transport.name == "telegram":
        chat, _, topic = thread_id.partition(":")
        kind = (
            "photo"
            if attachment.kind == "image"
            else "audio"
            if attachment.kind == "audio"
            else "document"
        )
        data = {"chat_id": chat}
        if topic:
            data["message_thread_id"] = topic
        result = await transport.api(
            "POST",
            f"https://api.telegram.org/bot{transport.token}/send{kind.title()}",
            data=data,
            files={kind: upload},
        )
        if not result.get("ok"):
            raise ValueError("Telegram media upload failed")
    elif transport.name == "discord":
        if not thread_id.isdigit():
            raise ValueError("Invalid Discord conversation ID")
        payload = {
            "attachments": [{"id": 0, "filename": filename}],
            "allowed_mentions": {"parse": []},
            "nonce": delivery_id[:24],
            "enforce_nonce": True,
        }
        await transport.api(
            "POST",
            f"https://discord.com/api/v10/channels/{thread_id}/messages",
            data={"payload_json": json.dumps(payload)},
            auth=f"Bot {transport.token}",
            files={"files[0]": upload},
        )
    elif transport.name == "slack":
        parts = thread_id.split(":", 2)
        if len(parts) != 3 or parts[0] != transport.team_id:
            raise ValueError("Slack conversation does not belong to the authenticated workspace")
        slot = await transport.call(
            "files.getUploadURLExternal", {"filename": filename, "length": len(raw)}
        )
        url = str(slot["upload_url"])
        validate_media_url(url, ("files.slack.com",))
        transport.secrets.append(url)
        try:
            response = await transport.client.post(
                url, content=raw, headers={"Content-Type": "application/octet-stream"}
            )
        except httpx.HTTPError:
            raise ValueError("Slack media upload interrupted") from None
        if not response.is_success:
            raise ValueError("Slack media upload failed")
        payload = {"files": [{"id": slot["file_id"], "title": filename}], "channel_id": parts[1]}
        if parts[2]:
            payload["thread_ts"] = parts[2]
        await transport.call("files.completeUploadExternal", payload)
    else:
        raise ValueError("Unsupported media transport")
