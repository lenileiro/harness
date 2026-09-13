"""DingTalk authenticated Stream callbacks; reply capabilities remain private."""

from __future__ import annotations

import base64
import json
import mimetypes
import time
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

from harness.cli.channels.attachment_io import MAX_BYTES, download
from harness.cli.channels.transports import ChannelError, RateLimited, Transport
from harness.core.gateway_channels import ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment

TOPIC = "/v1.0/im/bot/messages/get"
OPEN_CONNECTION = "https://api.dingtalk.com/v1.0/gateway/connections/open"


class DingTalkTransport(Transport):
    name = "dingtalk"
    limit = 4000

    async def authenticate(self) -> None:
        if not self.config.app_id:
            raise ChannelError("DingTalk requires app_id and app secret token_env")
        self.connection = await self._connection()
        self.identity = "dingtalk:" + self.config.app_id
        self.store: ChannelStore | None = None
        self.access_token = ""
        self.access_expires = 0.0

    async def _connection(self) -> dict[str, Any]:
        payload = await self.api(
            "POST",
            OPEN_CONNECTION,
            data={
                "clientId": self.config.app_id,
                "clientSecret": self.token,
                "subscriptions": [{"type": "CALLBACK", "topic": TOPIC}],
                "ua": "Harness/1.0",
                "localIp": "127.0.0.1",
            },
        )
        if not payload.get("endpoint") or not payload.get("ticket"):
            raise ChannelError("DingTalk did not authorize a Stream connection")
        return payload

    def channel_id(self, thread_id: str) -> str:
        corp, conversation = json.loads(thread_id)
        return f"{corp}:{conversation}"

    def can_send(self) -> bool:
        return getattr(self, "store", None) is not None

    @staticmethod
    def _reply_url(url: str) -> None:
        parsed = urlsplit(url)
        if (
            parsed.scheme != "https"
            or parsed.netloc != "oapi.dingtalk.com"
            or parsed.path != "/robot/sendBySession"
            or parsed.fragment
        ):
            raise ChannelError("DingTalk reply capability is outside the official session endpoint")

    def _ingest(self, payload: dict[str, Any], store: ChannelStore) -> None:
        if payload.get("msgtype") not in {"text", "picture", "image", "file", "audio", "richText"}:
            return
        if payload.get("robotCode") and payload["robotCode"] != self.config.app_id:
            return
        corp = str(payload["senderCorpId"])
        user = corp + ":" + str(payload.get("senderStaffId") or payload["senderId"])
        conversation = str(payload["conversationId"])
        group = str(payload.get("conversationType")) != "1"
        mentioned = payload.get("isInAtList") is True
        thread = json.dumps([corp, conversation], separators=(",", ":"))
        channel = f"{corp}:{conversation}"
        if not self.config.permits(
            user_id=user, channel_id=channel, group=group, mentioned=mentioned
        ):
            return
        url = str(payload["sessionWebhook"])
        self._reply_url(url)
        self.secrets.extend([url, *(value for _, value in parse_qsl(urlsplit(url).query) if value)])
        store.set(
            "dingtalk-route:" + thread,
            {"url": url, "expires": int(payload["sessionWebhookExpiredTime"]) / 1000},
        )
        text = str(payload.get("text", {}).get("content", ""))
        content = payload.get("content", {})
        if not isinstance(content, dict):
            content = {}
        items = content.get("richText", []) if payload.get("msgtype") == "richText" else [content]
        attachments = []
        for item in items[:16]:
            if not isinstance(item, dict):
                continue
            if item.get("text"):
                text += str(item["text"])
            code = item.get("downloadCode") or item.get("pictureDownloadCode")
            if code:
                kind = str(item.get("type") or payload.get("msgtype") or "")
                name = str(item.get("fileName") or "attachment")
                mime = mimetypes.guess_type(name)[0] or {
                    "picture": "image/jpeg",
                    "image": "image/jpeg",
                    "audio": "audio/ogg",
                    "voice": "audio/ogg",
                }.get(kind, "application/octet-stream")
                attachments.append({"download_code": str(code), "mime_type": mime, "name": name})
        if not text:
            text = str(content.get("recognition") or "")
        store.ingest(
            ChannelMessage(
                id=json.dumps([corp, conversation, payload["msgId"]], separators=(",", ":")),
                user_id=user,
                channel_id=channel,
                thread_id=thread,
                text=text,
                group=group,
                mentioned=mentioned,
                attachments=attachments,
            )
        )

    async def receive(self, store: ChannelStore) -> None:
        self.store = store
        connection = self.connection or await self._connection()
        self.connection = None
        endpoint = str(connection["endpoint"])
        if urlsplit(endpoint).query or urlsplit(endpoint).fragment:
            raise ChannelError("DingTalk Stream endpoint must not already contain a ticket")
        self.secrets.append(str(connection["ticket"]))
        async with self.socket(
            endpoint + "?" + urlencode({"ticket": connection["ticket"]}), domains=("dingtalk.com",)
        ) as socket:
            async for raw in socket:
                frame = json.loads(raw)
                headers = frame.get("headers", {})
                kind = frame.get("type")
                data = json.loads(frame.get("data") or "{}")
                if kind == "CALLBACK" and headers.get("topic") == TOPIC:
                    self._ingest(data, store)
                    response = {"response": "accepted"}
                elif kind == "SYSTEM":
                    response = data
                else:
                    continue
                await socket.send(
                    json.dumps(
                        {
                            "code": 200,
                            "headers": {
                                "messageId": headers["messageId"],
                                "contentType": "application/json",
                            },
                            "message": "OK",
                            "data": json.dumps(response),
                        }
                    )
                )
                if kind == "SYSTEM" and headers.get("topic") == "disconnect":
                    return

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        await self.send_native(
            thread_id, {"msgtype": "text", "text": {"content": text}, "at": {"isAtAll": False}}
        )

    async def send_native(self, thread_id: str, data: dict[str, Any]) -> None:
        if self.store is None:
            raise ChannelError("DingTalk session routes are unavailable")
        route = self.store.get("dingtalk-route:" + thread_id)
        if not route or route["expires"] <= time.time():
            raise ChannelError(
                "DingTalk reply capability expired; a new message in that conversation is required"
            )
        self._reply_url(route["url"])
        self.secrets.extend(
            [
                route["url"],
                *(value for _, value in parse_qsl(urlsplit(route["url"]).query) if value),
            ]
        )
        payload = await self.api(
            "POST",
            route["url"],
            data=data,
        )
        if payload.get("errcode", 0) != 0:
            raise ChannelError("DingTalk rejected the reply")

    async def prepare_media(self, message: ChannelMessage) -> list[MediaAttachment]:
        if not self.access_token or time.time() >= self.access_expires:
            auth = await self.api(
                "POST",
                "https://api.dingtalk.com/v1.0/oauth2/accessToken",
                data={"appKey": self.config.app_id, "appSecret": self.token},
            )
            self.access_token = str(auth.get("accessToken") or "")
            if not self.access_token:
                raise ChannelError("DingTalk did not authorize media access")
            self.access_expires = time.time() + max(
                1, min(7200, int(auth.get("expireIn", 7200))) - 60
            )
            self.secrets.append(self.access_token)
        results, remaining = [], MAX_BYTES
        for item in message.attachments[:16]:
            code = str(item["download_code"])
            if not code or len(code) > 4096:
                raise ChannelError("DingTalk media download code is invalid")
            response = await self.client.post(
                "https://api.dingtalk.com/v1.0/robot/messageFiles/download",
                headers={"x-acs-dingtalk-access-token": self.access_token},
                json={"robotCode": self.config.app_id, "downloadCode": code},
            )
            if response.status_code == 429:
                raise RateLimited(float(response.headers.get("Retry-After", "30")))
            if not response.is_success or len(response.content) > 16384:
                raise ChannelError("DingTalk media capability exchange failed")
            url = str(response.json().get("downloadUrl") or "")
            self.secrets.append(url)
            attachment = await download(
                self.client,
                url,
                domains=("dingtalk.com", "alicdn.com", "aliyuncs.com"),
                max_bytes=remaining,
                mime=item["mime_type"],
                name=item["name"],
            )
            remaining -= len(base64.b64decode(attachment.data or ""))
            results.append(attachment)
        return results

    async def send_media(
        self, thread_id: str, attachment: MediaAttachment, delivery_id: str
    ) -> None:
        # The pinned session-webhook protocol has no local-file upload endpoint.
        if attachment.kind != "image" or not attachment.url or attachment.data is not None:
            raise ChannelError(
                "DingTalk session replies support hosted image URLs, not inline file uploads"
            )
        url = urlsplit(attachment.url)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or any(ord(char) < 32 for char in attachment.url)
        ):
            raise ChannelError("DingTalk image requires a valid hosted HTTPS URL")
        safe_url = attachment.url.replace("(", "%28").replace(")", "%29")
        await self.send_native(
            thread_id,
            {
                "msgtype": "markdown",
                "markdown": {"title": "Image", "text": "![](" + safe_url + ")"},
                "at": {"isAtAll": False},
            },
        )
