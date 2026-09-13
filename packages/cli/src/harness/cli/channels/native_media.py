"""Bounded byte transfers for native, platform-validated attachment resources.

Callers own URL/identity validation. This module never discovers or fetches URLs
from arbitrary text, follows redirects, or opens local attachment paths.
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

import httpx

from harness.cli.channels.transports import ChannelError, RateLimited
from harness.core.schemas import MediaAttachment

if TYPE_CHECKING:
    from harness.cli.channels.transports import Transport

MEDIA_LIMIT = 20 * 1024 * 1024
ATTACHMENT_LIMIT = 16


def resource_segment(value: object) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9_.:@+-]{1,512}", value)
        or value in {".", ".."}
    ):
        raise ChannelError("Invalid native attachment resource identity")
    return value


def media_name(value: object) -> str:
    name = PurePosixPath(str(value or "attachment").replace("\\", "/")).name
    if name in {".", ".."}:
        return "attachment"
    return "".join(c for c in name if ord(c) >= 32 and ord(c) != 127)[:255] or "attachment"


def media_type(value: object, name: str) -> str:
    mime = str(value or "").split(";", 1)[0].strip().lower()
    if not re.fullmatch(r"[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+", mime):
        mime = mimetypes.guess_type(name)[0] or "application/octet-stream"
    return mime


def attachment_bytes(attachment: MediaAttachment) -> bytes:
    if attachment.data is None or len(attachment.data) > ((MEDIA_LIMIT + 2) // 3) * 4:
        raise ChannelError("Native attachment delivery requires inline bytes up to 20 MiB")
    data = base64.b64decode(attachment.data, validate=True)
    if len(data) > MEDIA_LIMIT:
        raise ChannelError("Native attachment exceeds 20 MiB")
    return data


def media_attachment(data: bytes, name: object, mime: object) -> MediaAttachment:
    filename = media_name(name)
    content_type = media_type(mime, filename)
    return MediaAttachment(
        kind="image"
        if content_type.startswith("image/")
        else "audio"
        if content_type.startswith("audio/")
        else "file",
        mime_type=content_type,
        data=base64.b64encode(data).decode(),
        name=filename,
    )


async def download_bytes(
    transport: Transport, url: str, *, remaining: int, auth: str = ""
) -> tuple[bytes, str]:
    if remaining <= 0:
        raise ChannelError("Native attachments exceed the aggregate 20 MiB limit")
    try:
        async with (
            asyncio.timeout(60),
            transport.client.stream(
                "GET",
                url,
                headers={
                    "Accept-Encoding": "identity",
                    **({"Authorization": auth} if auth else {}),
                },
                follow_redirects=False,
            ) as response,
        ):
            if response.status_code == 429:
                raise RateLimited(float(response.headers.get("Retry-After", "1")))
            if not response.is_success:
                raise ChannelError(
                    f"Native attachment download failed (HTTP {response.status_code})"
                )
            length = response.headers.get("Content-Length")
            if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                raise ChannelError("Native attachment transfer requires uncompressed bytes")
            if length and int(length) > remaining:
                raise ChannelError("Native attachments exceed the aggregate 20 MiB limit")
            result = bytearray()
            async for chunk in response.aiter_bytes(min(65536, remaining + 1)):
                result.extend(chunk)
                if len(result) > remaining:
                    raise ChannelError("Native attachments exceed the aggregate 20 MiB limit")
            return bytes(result), response.headers.get("Content-Type", "")
    except (httpx.HTTPError, TimeoutError):
        raise ChannelError("Native attachment download was interrupted") from None
