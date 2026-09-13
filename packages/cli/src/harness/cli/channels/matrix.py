"""Matrix client-server transport with persisted sync tokens and transaction IDs."""

from __future__ import annotations

import base64
import json
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode, urlsplit

from harness.cli.channels.transports import ChannelError, RateLimited, Transport
from harness.core.gateway_channels import ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment

if TYPE_CHECKING:
    from harness.cli.channels.matrix_crypto import MatrixCrypto


class MatrixTransport(Transport):
    name = "matrix"
    limit = 4000
    crypto: MatrixCrypto | None = None

    @property
    def base(self) -> str:
        value = self.config.homeserver.rstrip("/")
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ChannelError("Matrix homeserver must be an operator-configured HTTPS base URL")
        return value

    async def call(self, method: str, path: str, data: dict[str, Any] | None = None) -> Any:
        return await self.api(method, self.base + path, data=data, auth=f"Bearer {self.token}")

    async def authenticate(self) -> None:
        data = await self.call("GET", "/_matrix/client/v3/account/whoami")
        self.bot_id = str(data["user_id"])
        self.identity = json.dumps([self.base, self.bot_id, data.get("device_id", "")])
        if self.config.matrix_e2ee:
            try:
                from harness.cli.channels.matrix_crypto import MatrixCrypto
            except (ImportError, ImportWarning):
                raise ChannelError(
                    "Matrix E2EE requires the CLI matrix extra: matrix-nio[e2e]==0.26.0 with vodozemac"
                ) from None
            try:
                self.crypto = MatrixCrypto(
                    self, self.base, self.bot_id, str(data.get("device_id", ""))
                )
                await self.crypto.initialize_keys()
            except BaseException as exc:
                if self.crypto is not None:
                    await self.crypto.close()
                    self.crypto = None
                if isinstance(exc, Exception) and not isinstance(exc, ChannelError):
                    raise ChannelError(
                        "Matrix crypto initialization failed; check the original device store and its referenced passphrase"
                    ) from None
                raise

    async def close(self) -> None:
        try:
            if self.crypto is not None:
                await self.crypto.close()
                self.crypto = None
        finally:
            await super().close()

    def channel_id(self, thread_id: str) -> str:
        room, _ = json.loads(thread_id)
        return room

    async def receive(self, store: ChannelStore) -> None:
        if self.config.matrix_e2ee:
            if self.crypto is None:
                raise ChannelError("Matrix E2EE has not authenticated")
            data = await self.crypto.sync(store)
            self._ingest_sync(data, store)
            self.crypto.commit_sync(store)
            return
        since = store.get("since")
        params = {"timeout": "30000"}
        if since:
            params["since"] = since
        data = await self.call("GET", "/_matrix/client/v3/sync?" + urlencode(params))
        self._ingest_sync(data, store)

    def _ingest_sync(self, data: dict[str, Any], store: ChannelStore) -> None:
        since = store.get("since")
        direct = set(store.get("direct_rooms", []))
        for event in data.get("account_data", {}).get("events", []):
            if event.get("type") == "m.direct":
                direct = {room for rooms in event.get("content", {}).values() for room in rooms}
                store.set("direct_rooms", sorted(direct))
        for room_id, room in data.get("rooms", {}).get("join", {}).items():
            members = room.get("summary", {}).get("m.joined_member_count")
            if isinstance(members, int):
                store.set(f"members:{room_id}", members)
            group = room_id not in direct or store.get(f"members:{room_id}", 3) > 2
            for event in room.get("timeline", {}).get("events", []):
                if not since:  # Establish a baseline; never execute historic initial-sync text.
                    continue
                if event.get("type") != "m.room.message" or event.get("sender") == self.bot_id:
                    continue
                content = event.get("content", {})
                relation = content.get("m.relates_to", {})
                if relation.get("rel_type") == "m.replace":
                    continue
                root = (
                    relation.get("event_id", "") if relation.get("rel_type") == "m.thread" else ""
                )
                mentioned = self.bot_id in content.get("m.mentions", {}).get("user_ids", [])
                attachments = []
                if content.get("msgtype") in {
                    "m.image",
                    "m.audio",
                    "m.file",
                    "m.video",
                } and (content.get("url") or content.get("file")):
                    encrypted_file = content.get("file")
                    if self.config.matrix_e2ee and not isinstance(encrypted_file, dict):
                        continue  # E2EE media must also be encrypted at the media repository.
                    attachments = [
                        {
                            "url": encrypted_file["url"]
                            if isinstance(encrypted_file, dict)
                            else content["url"],
                            **(
                                {"encrypted_file": encrypted_file}
                                if isinstance(encrypted_file, dict)
                                else {}
                            ),
                            "mime_type": content.get("info", {}).get(
                                "mimetype", "application/octet-stream"
                            ),
                            "name": content.get("body", "attachment"),
                        }
                    ]
                if content.get("msgtype") not in {
                    "m.text",
                    "m.notice",
                    "m.image",
                    "m.audio",
                    "m.file",
                    "m.video",
                }:
                    continue
                text = str(content.get("body", ""))
                if mentioned:
                    text = text.replace(self.bot_id, "").strip()
                self.accept(
                    ChannelMessage(
                        id=str(event["event_id"]),
                        user_id=str(event["sender"]),
                        channel_id=room_id,
                        thread_id=json.dumps([room_id, root], separators=(",", ":")),
                        text=text,
                        group=group,
                        mentioned=mentioned,
                        attachments=attachments,
                    ),
                    store,
                )
        if not isinstance(data.get("next_batch"), str):
            raise ChannelError("Matrix sync omitted its restart cursor")
        store.set("since", data["next_batch"])

    async def _plaintext_room(self, room: str) -> None:
        response = await self.client.get(
            self.base + f"/_matrix/client/v3/rooms/{quote(room, safe='')}/state/m.room.encryption",
            headers={"Authorization": f"Bearer {self.token}"},
        )
        if response.status_code == 404:
            return
        if response.status_code == 429:
            raise RateLimited(float(response.headers.get("Retry-After", "1")))
        raise ChannelError(
            "Cannot send plaintext to an encrypted or unverifiable Matrix room; configure Matrix E2EE"
        )

    async def _send_content(
        self, thread_id: str, content: dict[str, Any], delivery_id: str
    ) -> None:
        room, root = json.loads(thread_id)
        if root:
            content["m.relates_to"] = {
                "rel_type": "m.thread",
                "event_id": root,
                "is_falling_back": True,
                "m.in_reply_to": {"event_id": root},
            }
        if self.config.matrix_e2ee:
            if self.crypto is None:
                raise ChannelError("Matrix E2EE has not authenticated")
            await self.crypto.send(room, content, delivery_id)
            return
        await self._plaintext_room(room)
        await self.call(
            "PUT",
            f"/_matrix/client/v3/rooms/{quote(room, safe='')}/send/m.room.message/{quote(delivery_id, safe='')}",
            content,
        )

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        await self._send_content(
            thread_id, {"msgtype": "m.text", "body": text, "m.mentions": {}}, delivery_id
        )

    async def prepare_media(self, message: ChannelMessage) -> list[MediaAttachment]:
        result = []
        total = 0
        for item in message.attachments[:16]:
            parsed = urlsplit(str(item["url"]))
            media = parsed.path.lstrip("/")
            if (
                parsed.scheme != "mxc"
                or not parsed.netloc
                or not media
                or "/" in media
                or media in {".", ".."}
                or parsed.query
                or parsed.fragment
                or parsed.username
                or parsed.password
            ):
                raise ChannelError("Matrix media must be an mxc resource, never an arbitrary URL")
            url = (
                self.base
                + f"/_matrix/client/v1/media/download/{quote(parsed.netloc, safe='')}/{quote(media, safe='')}"
            )
            raw = bytearray()
            async with self.client.stream(
                "GET",
                url,
                headers={"Authorization": f"Bearer {self.token}"},
                follow_redirects=False,
            ) as response:
                if not response.is_success:
                    raise ChannelError("Matrix media download rejected")
                async for chunk in response.aiter_bytes():
                    total += len(chunk)
                    if total > 20 * 1024 * 1024:
                        raise ChannelError("Matrix media exceeds 20 MiB")
                    raw.extend(chunk)
            mime = str(item.get("mime_type", "application/octet-stream"))
            if self.config.matrix_e2ee:
                if self.crypto is None or not isinstance(item.get("encrypted_file"), dict):
                    raise ChannelError("Matrix E2EE media lacks verified encryption metadata")
                raw = bytearray(self.crypto.decrypt_media(bytes(raw), item["encrypted_file"]))
            result.append(
                MediaAttachment(
                    kind="image"
                    if mime.startswith("image/")
                    else "audio"
                    if mime.startswith("audio/")
                    else "file",
                    mime_type=mime,
                    data=base64.b64encode(raw).decode(),
                    name=PurePosixPath(str(item.get("name", "attachment"))).name[:255],
                )
            )
        return result

    async def send_media(
        self, thread_id: str, attachment: MediaAttachment, delivery_id: str
    ) -> None:
        room, _ = json.loads(thread_id)
        if self.config.matrix_e2ee:
            if self.crypto is None:
                raise ChannelError("Matrix E2EE has not authenticated")
            await self.crypto.check_room(room)
        else:
            await self._plaintext_room(room)
        if attachment.data is None:
            raise ChannelError("Matrix uploads require inline attachment bytes")
        raw = base64.b64decode(attachment.data, validate=True)
        original_size = len(raw)
        encrypted_file = None
        if self.crypto is not None:
            raw, encrypted_file = self.crypto.encrypt_media(raw)
        response = await self.client.post(
            self.base
            + "/_matrix/media/v3/upload?"
            + urlencode(
                {"filename": "encrypted" if encrypted_file else attachment.name or "attachment"}
            ),
            content=raw,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/octet-stream"
                if encrypted_file
                else attachment.mime_type,
            },
            follow_redirects=False,
        )
        if response.status_code == 429:
            raise RateLimited(float(response.headers.get("Retry-After", "1")))
        if not response.is_success:
            raise ChannelError("Matrix upload rejected")
        uri = response.json()["content_uri"]
        media_location = (
            {"file": {**encrypted_file, "url": uri}} if encrypted_file else {"url": uri}
        )
        await self._send_content(
            thread_id,
            {
                "msgtype": "m.image"
                if attachment.kind == "image"
                else "m.audio"
                if attachment.kind == "audio"
                else "m.file",
                "body": attachment.name or "attachment",
                **media_location,
                "info": {"mimetype": attachment.mime_type, "size": original_size},
            },
            delivery_id,
        )
