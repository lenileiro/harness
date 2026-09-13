"""Mattermost bot WebSocket events with REST catch-up after reconnect."""

from __future__ import annotations

import base64
import json
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

from harness.cli.channels.attachment_io import MAX_BYTES, attachment_bytes, download
from harness.cli.channels.transports import ChannelError, Transport
from harness.core.gateway_channels import ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment


class MattermostTransport(Transport):
    name = "mattermost"
    limit = 4000

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
            raise ChannelError("Mattermost requires an operator-configured HTTPS homeserver")
        return value

    async def call(self, method: str, path: str, data: dict[str, Any] | None = None) -> Any:
        return await self.api(
            method, self.base + "/api/v4/" + path, data=data, auth=f"Bearer {self.token}"
        )

    async def authenticate(self) -> None:
        if not self.config.allowed_channels:
            raise ChannelError(
                "Mattermost requires explicit channel IDs for bounded history recovery"
            )
        user = await self.call("GET", "users/me")
        self.bot_id, self.username = str(user["id"]), str(user["username"])
        self.identity = json.dumps([self.base, self.bot_id])
        self.channels = {}
        for channel in self.config.allowed_channels:
            data = await self.call("GET", "channels/" + quote(channel, safe=""))
            if data.get("id") != channel:
                raise ChannelError("Mattermost channel identity mismatch")
            self.channels[channel] = data

    def channel_id(self, thread_id: str) -> str:
        channel, _ = json.loads(thread_id)
        return channel

    def ingest(self, post: dict[str, Any], store: ChannelStore) -> None:
        channel = str(post.get("channel_id", ""))
        sender = str(post.get("user_id", ""))
        text = str(post.get("message", ""))
        if (
            channel not in self.channels
            or not sender
            or sender == self.bot_id
            or post.get("type")
            or post.get("delete_at")
            or (not text and not post.get("file_ids"))
        ):
            return
        # Bot accounts and webhook integrations must not impersonate a human command.
        if post.get("props", {}).get("from_webhook") or post.get("props", {}).get("from_bot"):
            return
        group = self.channels[channel].get("type") != "D"
        mention = "@" + self.username
        import re

        pattern = r"(?<![\w.-])" + re.escape(mention) + r"(?![\w.-])"
        mentioned = re.search(pattern, text) is not None
        text = re.sub(pattern, "", text).strip()
        root = str(post.get("root_id") or post["id"]) if group else ""
        self.accept(
            ChannelMessage(
                id=str(post["id"]),
                user_id=sender,
                channel_id=channel,
                thread_id=json.dumps([channel, root], separators=(",", ":")),
                text=text,
                group=group,
                mentioned=mentioned,
                attachments=[
                    {"file_id": str(identifier), "post_id": str(post["id"])}
                    for identifier in post.get("file_ids", [])[:16]
                ],
            ),
            store,
        )

    async def catch_up(self, store: ChannelStore) -> None:
        for channel in self.channels:
            key = "since:" + channel
            since = store.get(key)
            if since is None:
                # Use server-created timestamps, avoiding local clock skew. The
                # socket is already open; new buffered events are consumed below.
                page = await self.call(
                    "GET", "channels/" + quote(channel, safe="") + "/posts?per_page=1"
                )
                posts = page.get("posts", {})
                store.set(key, max((int(post["create_at"]) for post in posts.values()), default=0))
                continue
            posts = []
            before = ""
            for _ in range(100):
                params = {"per_page": "100"}
                if before:
                    params["before"] = before
                page = await self.call(
                    "GET", "channels/" + quote(channel, safe="") + "/posts?" + urlencode(params)
                )
                ordered = [page["posts"][identifier] for identifier in page.get("order", [])]
                posts.extend(item for item in ordered if int(item["create_at"]) >= since)
                if len(ordered) < 100 or min(int(item["create_at"]) for item in ordered) < since:
                    break
                next_before = str(ordered[-1]["id"])
                if next_before == before:
                    raise ChannelError("Mattermost history pagination did not advance")
                before = next_before
            else:
                store.set("connection", "history_gap")
                raise ChannelError("Mattermost history exceeds the 10,000-post recovery bound")
            for post in sorted(posts, key=lambda item: (item["create_at"], item["id"])):
                self.ingest(post, store)
            if posts:
                store.set(key, max(since, *(int(post["create_at"]) for post in posts)))

    async def receive(self, store: ChannelStore) -> None:
        parsed = urlsplit(self.base)
        url = "wss://" + parsed.netloc + parsed.path + "/api/v4/websocket"
        async with self.socket(url, domains=(str(parsed.hostname),)) as socket:
            await socket.send(
                json.dumps(
                    {"seq": 1, "action": "authentication_challenge", "data": {"token": self.token}}
                )
            )
            authenticated = False
            # hello may precede the authentication response; no message dispatch before OK.
            for _ in range(10):
                frame = json.loads(await socket.recv())
                if frame.get("seq_reply") == 1:
                    if frame.get("status") != "OK":
                        raise ChannelError("Mattermost WebSocket authentication failed")
                    authenticated = True
                    break
            if not authenticated:
                raise ChannelError("Mattermost WebSocket omitted authentication acknowledgment")
            await self.catch_up(store)
            store.set("connection", "connected")
            async for raw in socket:
                frame = json.loads(raw)
                if frame.get("event") != "posted":
                    continue
                post = json.loads(frame.get("data", {}).get("post", "{}"))
                self.ingest(post, store)
                channel = str(post.get("channel_id", ""))
                if channel in self.channels:
                    store.set(
                        "since:" + channel,
                        max(store.get("since:" + channel, 0), int(post.get("create_at", 0))),
                    )

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        channel, root = json.loads(thread_id)
        if channel not in self.channels:
            raise ChannelError("Mattermost destination is outside configured channels")
        await self.call(
            "POST",
            "posts",
            {
                "channel_id": channel,
                "root_id": root,
                "message": text,
                "pending_post_id": delivery_id[:26],
            },
        )

    async def prepare_media(self, message: ChannelMessage) -> list[MediaAttachment]:
        results, remaining = [], MAX_BYTES
        for item in message.attachments[:16]:
            identifier = str(item["file_id"])
            if not identifier.isalnum() or len(identifier) > 100:
                raise ChannelError("Mattermost attachment ID is invalid")
            info = await self.call("GET", "files/" + identifier + "/info")
            if info.get("id") != identifier or info.get("post_id") != item["post_id"]:
                raise ChannelError("Mattermost attachment belongs to a different post")
            attachment = await download(
                self.client,
                self.base + "/api/v4/files/" + identifier,
                domains=(str(urlsplit(self.base).hostname),),
                headers={"Authorization": "Bearer " + self.token},
                max_bytes=remaining,
                mime=str(info.get("mime_type") or "application/octet-stream"),
                name=str(info.get("name") or "attachment"),
            )
            remaining -= len(base64.b64decode(attachment.data or ""))
            results.append(attachment)
        return results

    async def send_media(
        self, thread_id: str, attachment: MediaAttachment, delivery_id: str
    ) -> None:
        channel, root = json.loads(thread_id)
        if channel not in self.channels:
            raise ChannelError("Mattermost destination is outside configured channels")
        from pathlib import PurePosixPath

        raw = attachment_bytes(attachment)
        result = await self.api(
            "POST",
            self.base + "/api/v4/files",
            data={"channel_id": channel},
            auth="Bearer " + self.token,
            files={
                "files": (
                    PurePosixPath(attachment.name or "attachment").name,
                    raw,
                    attachment.mime_type,
                )
            },
        )
        infos = result.get("file_infos", [])
        if len(infos) != 1 or not infos[0].get("id"):
            raise ChannelError("Mattermost did not acknowledge the uploaded file")
        await self.call(
            "POST",
            "posts",
            {
                "channel_id": channel,
                "root_id": root,
                "message": "",
                "file_ids": [infos[0]["id"]],
                "pending_post_id": delivery_id[:26],
            },
        )
