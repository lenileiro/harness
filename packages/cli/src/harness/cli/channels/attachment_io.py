"""Bounded native attachment transfers and private expiring media publication."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import re
import secrets
import time
from pathlib import PurePosixPath
from urllib.parse import quote, urljoin, urlsplit

import httpx
from aiohttp import web

from harness.cli.channels.media import validate_media_url
from harness.cli.channels.transports import ChannelError, RateLimited
from harness.core.gateway_channels import ChannelStore
from harness.core.schemas import MediaAttachment

MAX_BYTES = 20 * 1024 * 1024


def attachment_bytes(attachment: MediaAttachment, *, max_bytes: int = MAX_BYTES) -> bytes:
    if attachment.data is None or len(attachment.data) > (max_bytes + 2) // 3 * 4:
        raise ChannelError("Native media requires bounded inline attachment data")
    value = base64.b64decode(attachment.data, validate=True)
    if not value or len(value) > max_bytes:
        raise ChannelError("Attachment exceeds the native media size limit")
    return value


def from_bytes(data: bytes, mime: str, name: str) -> MediaAttachment:
    if not re.fullmatch(r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+", mime):
        mime = "application/octet-stream"
    return MediaAttachment(
        kind="image"
        if mime.startswith("image/")
        else "audio"
        if mime.startswith("audio/")
        else "file",
        mime_type=mime,
        data=base64.b64encode(data).decode(),
        name=PurePosixPath(name.replace("\\", "/")).name[:255] or "attachment",
    )


async def download(
    client: httpx.AsyncClient,
    url: str,
    *,
    domains: tuple[str, ...],
    headers: dict[str, str] | None = None,
    max_bytes: int = MAX_BYTES,
    mime: str = "application/octet-stream",
    name: str = "attachment",
) -> MediaAttachment:
    initial_host = urlsplit(url).netloc
    async with asyncio.timeout(60):
        for _ in range(4):
            validate_media_url(url, domains)
            request_headers = dict(headers or {}) if urlsplit(url).netloc == initial_host else {}
            request_headers["Accept-Encoding"] = "identity"
            async with client.stream(
                "GET", url, headers=request_headers, follow_redirects=False
            ) as response:
                if response.status_code in {301, 302, 303, 307, 308}:
                    target = response.headers.get("Location")
                    if not target:
                        raise ChannelError("Platform media redirect has no destination")
                    url = urljoin(url, target)
                    continue
                if response.status_code == 429:
                    raise RateLimited(float(response.headers.get("Retry-After", "30")))
                if not response.is_success:
                    raise ChannelError("Platform media download failed")
                if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise ChannelError("Compressed platform media transfers are not accepted")
                content_length = response.headers.get("Content-Length", "")
                if content_length.isdigit() and int(content_length) > max_bytes:
                    raise ChannelError("Channel attachments exceed their byte limit")
                raw = bytearray()
                async for chunk in response.aiter_bytes(chunk_size=min(65536, max_bytes + 1)):
                    raw.extend(chunk)
                    if len(raw) > max_bytes:
                        raise ChannelError("Channel attachments exceed their byte limit")
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0]
                return from_bytes(bytes(raw), content_type or mime, name)
    raise ChannelError("Platform media exceeded its redirect limit")


class MediaPublisher:
    """Serve only outbox byte snapshots using expiring unguessable capabilities.

    Required by LINE's URL-only media protocol. Never accepts filesystem paths.
    The configured public callback origin is an explicit operator trust choice.
    """

    def __init__(self, store: ChannelStore, *, public_url: str, callback_path: str):
        parsed = urlsplit(public_url)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.path != callback_path
        ):
            raise ChannelError(
                "Native media hosting requires an exact HTTPS webhook_url matching webhook_path"
            )
        self.store = store
        self.base = public_url.rstrip("/") + "/media/"
        self.path = callback_path.rstrip("/") + "/media/{capability}"
        with store.db:
            store.db.execute("""CREATE TABLE IF NOT EXISTS published_media (
                delivery TEXT PRIMARY KEY, owner TEXT NOT NULL, capability TEXT UNIQUE NOT NULL,
                digest TEXT NOT NULL, body BLOB NOT NULL, mime TEXT NOT NULL,
                name TEXT NOT NULL, expires REAL NOT NULL
            )""")

    def publish(self, delivery: str, owner: str, attachment: MediaAttachment) -> str:
        raw = attachment_bytes(attachment)
        if not re.fullmatch(r"[A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+", attachment.mime_type):
            raise ChannelError("Published attachment requires a valid MIME type")
        digest = hashlib.sha256(raw).hexdigest()
        with self.store.db:
            self.store.db.execute("DELETE FROM published_media WHERE expires<=?", (time.time(),))
            row = self.store.db.execute(
                "SELECT * FROM published_media WHERE delivery=?", (delivery,)
            ).fetchone()
            if row:
                if (
                    row["owner"] != owner
                    or row["digest"] != digest
                    or row["mime"] != attachment.mime_type
                ):
                    raise ChannelError(
                        "Media delivery snapshot differs from its existing owner/content"
                    )
                return self.base + row["capability"]
            total = self.store.db.execute(
                "SELECT COALESCE(SUM(LENGTH(body)),0),COUNT(*) FROM published_media"
            ).fetchone()
            if total[0] + len(raw) > 100 * 1024 * 1024 or total[1] >= 100:
                raise ChannelError(
                    "Public media cache is full; wait for existing capabilities to expire"
                )
            capability = secrets.token_urlsafe(32)
            self.store.db.execute(
                "INSERT INTO published_media VALUES (?,?,?,?,?,?,?,?)",
                (
                    delivery,
                    owner,
                    capability,
                    digest,
                    raw,
                    attachment.mime_type,
                    PurePosixPath(attachment.name or "attachment").name,
                    time.time() + 3600,
                ),
            )
        return self.base + capability

    def install(self, app: web.Application) -> None:
        async def serve(request: web.Request) -> web.Response:
            capability = request.match_info["capability"]
            if not re.fullmatch(r"[A-Za-z0-9_-]{43}", capability):
                raise web.HTTPNotFound()
            row = self.store.db.execute(
                "SELECT body,mime,name FROM published_media WHERE capability=? AND expires>?",
                (capability, time.time()),
            ).fetchone()
            if not row:
                raise web.HTTPNotFound()
            body = bytes(row["body"])
            headers = {
                "Content-Type": row["mime"],
                "Cache-Control": "private, max-age=300",
                "X-Content-Type-Options": "nosniff",
                "Accept-Ranges": "bytes",
                "Content-Disposition": "attachment; filename*=UTF-8''"
                + quote(row["name"], safe=""),
            }
            status = 200
            if request.headers.get("Range"):
                match = re.fullmatch(r"bytes=([0-9]+)-([0-9]*)", request.headers["Range"])
                if not match:
                    raise web.HTTPRequestRangeNotSatisfiable()
                start, end = int(match[1]), int(match[2]) if match[2] else len(body) - 1
                if not 0 <= start <= end < len(body):
                    raise web.HTTPRequestRangeNotSatisfiable()
                headers["Content-Range"] = f"bytes {start}-{end}/{len(body)}"
                body, status = body[start : end + 1], 206
            return web.Response(body=body, headers=headers, status=status)

        app.router.add_get(self.path, serve)
