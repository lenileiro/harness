"""Teams Bot Connector callbacks with signed destination and tenant binding."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections.abc import Mapping
from typing import Any
from urllib.parse import parse_qsl, quote, urlsplit

import jwt

from harness.cli.channels.native_media import (
    ATTACHMENT_LIMIT,
    MEDIA_LIMIT,
    attachment_bytes,
    download_bytes,
    media_attachment,
    media_name,
    media_type,
)
from harness.cli.channels.transports import ChannelError, DeliveryDeferred, DeliveryRejected
from harness.cli.channels.webhooks import AuthenticationError, WebhookTransport
from harness.core.gateway_channels import ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment

OPENID = "https://login.botframework.com/v1/.well-known/openidconfiguration"
FILE_DOWNLOAD = "application/vnd.microsoft.teams.file.download.info"
FILE_CONSENT = "application/vnd.microsoft.teams.card.file.consent"
FILE_INFO = "application/vnd.microsoft.teams.card.file.info"


class TeamsTransport(WebhookTransport):
    name = "teams"
    limit = 4000

    async def authenticate(self) -> None:
        if (
            not self.config.app_id
            or not self.config.tenant_id
            or any(
                char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-"
                for char in self.config.tenant_id
            )
        ):
            raise ChannelError("Teams requires an app_id and explicit tenant_id")
        self.identity = f"{self.config.tenant_id}:{self.config.app_id}"
        self.bot_id = self.config.app_id
        self.access_token = ""
        self.token_expires = 0.0
        self.keys: dict[str, dict[str, Any]] = {}
        self.keys_fetched = 0.0
        self.store: ChannelStore | None = None
        self.media_lock = asyncio.Lock()
        await self._access()

    async def _access(self) -> str:
        if time.time() < self.token_expires - 60:
            return self.access_token
        response = await self.client.post(
            f"https://login.microsoftonline.com/{self.config.tenant_id}/oauth2/v2.0/token",
            data={
                "grant_type": "client_credentials",
                "client_id": self.config.app_id,
                "client_secret": self.token,
                "scope": "https://api.botframework.com/.default",
            },
        )
        if not response.is_success or not response.json().get("access_token"):
            raise ChannelError("Teams app authentication failed")
        payload = response.json()
        self.access_token = str(payload["access_token"])
        self.secrets.append(self.access_token)
        self.token_expires = time.time() + min(3600, float(payload.get("expires_in", 3600)))
        return self.access_token

    @staticmethod
    def _service_url(value: str) -> str:
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if (
            parsed.scheme != "https"
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
            or parsed.port not in {None, 443}
            or not any(
                host == domain or host.endswith("." + domain)
                for domain in (
                    "trafficmanager.net",
                    "botframework.com",
                    "smba.infra.teams.microsoft.com",
                )
            )
        ):
            raise AuthenticationError("Untrusted Teams connector destination")
        return value.rstrip("/")

    async def _verify(self, headers: Mapping[str, str], activity: dict[str, Any]) -> None:
        bearer = headers.get("Authorization", "")
        if not bearer.startswith("Bearer ") or len(bearer) > 16384:
            raise AuthenticationError("Teams requires a Bot Connector token")
        try:
            header = jwt.get_unverified_header(bearer[7:])
            if header.get("alg") != "RS256":
                raise ValueError
            now = time.time()
            if now - self.keys_fetched > 3600 or (
                header.get("kid") not in self.keys and now - self.keys_fetched > 60
            ):
                metadata = await self.api("GET", OPENID)
                uri = metadata["jwks_uri"]
                if (
                    urlsplit(uri).scheme != "https"
                    or urlsplit(uri).netloc != "login.botframework.com"
                    or "RS256" not in metadata.get("id_token_signing_alg_values_supported", [])
                ):
                    raise ValueError
                self.keys = {
                    key["kid"]: key
                    for key in (await self.api("GET", uri))["keys"]
                    if key.get("kty") == "RSA"
                }
                self.keys_fetched = now
            key = self.keys[header["kid"]]
            if "msteams" not in key.get("endorsements", []):
                raise ValueError
            claims = jwt.decode(
                bearer[7:],
                jwt.PyJWK.from_dict(key).key,
                algorithms=["RS256"],
                audience=self.config.app_id,
                issuer="https://api.botframework.com",
                leeway=300,
                options={"require": ["exp", "nbf", "aud", "iss", "serviceurl"]},
            )
            if (
                claims["serviceurl"] != activity.get("serviceUrl")
                or activity.get("channelId") != "msteams"
            ):
                raise ValueError
            self._service_url(activity["serviceUrl"])
            if activity.get("channelData", {}).get("tenant", {}).get("id") != self.config.tenant_id:
                raise ValueError
        except (jwt.PyJWTError, KeyError, TypeError, ValueError):
            raise AuthenticationError("Teams callback authentication failed") from None

    async def receive(self, store: ChannelStore) -> None:
        self.store = store
        await super().receive(store)

    def can_send(self) -> bool:
        return getattr(self, "store", None) is not None

    def channel_id(self, thread_id: str) -> str:
        tenant, conversation, _ = json.loads(thread_id)
        return f"{tenant}:{conversation}"

    async def handle_event(
        self, headers: Mapping[str, str], body: bytes, store: ChannelStore
    ) -> dict[str, Any]:
        activity = json.loads(body)
        await self._verify(headers, activity)
        if activity.get("type") == "invoke" and activity.get("name") == "fileConsent/invoke":
            self.store = store
            return self._file_consent(activity, store)
        if activity.get("type") == "invoke" and activity.get("name") == "adaptiveCard/action":
            action = activity.get("value", {}).get("action", {})
            if action.get("verb") != "harness_approval" or not isinstance(action.get("data"), dict):
                return {
                    "statusCode": 400,
                    "type": "application/vnd.microsoft.error",
                    "value": {"code": "BadRequest", "message": "Unsupported action"},
                }
            self.store = store
            result = await self._approval_callback({**activity, "value": action["data"]}, store)
            return {
                "statusCode": 200,
                "type": "application/vnd.microsoft.activity.message",
                "value": result["text"],
            }
        if (
            activity.get("type") == "message"
            and isinstance(activity.get("value"), dict)
            and "harness_approval" in activity["value"]
        ):
            self.store = store
            return await self._approval_callback(activity, store)
        if activity.get("type") != "message":
            return {}
        sender, conversation = activity["from"], activity["conversation"]
        tenant = self.config.tenant_id
        user_id = tenant + ":" + str(sender.get("aadObjectId") or sender["id"])
        group = conversation.get("conversationType") != "personal"
        # Group chats have one conversation; only channel posts have reply roots.
        root = (
            str(activity.get("replyToId") or activity["id"])
            if conversation.get("conversationType") == "channel"
            else ""
        )
        thread = json.dumps([tenant, conversation["id"], root], separators=(",", ":"))
        text = str(activity.get("text", ""))
        recipient = activity["recipient"]
        if sender.get("id") == recipient.get("id"):
            return {}
        mentioned = False
        for entity in activity.get("entities", []):
            if entity.get("type") == "mention" and entity.get("mentioned", {}).get(
                "id"
            ) == recipient.get("id"):
                mentioned = True
                text = text.replace(str(entity.get("text", "")), "").strip()
        message = ChannelMessage(
            id=json.dumps([tenant, conversation["id"], activity["id"]], separators=(",", ":")),
            user_id=user_id,
            thread_id=thread,
            channel_id=f"{tenant}:{conversation['id']}",
            text=text,
            group=group,
            mentioned=mentioned,
            attachments=[
                {**item, "service_url": activity["serviceUrl"], "personal": not group}
                for item in activity.get("attachments", [])
                if item.get("contentType") != "text/html"
            ],
        )
        if self.config.permits(
            user_id=user_id, channel_id=message.channel_id, group=group, mentioned=mentioned
        ):
            store.set(
                "teams-route:" + thread,
                {
                    "service_url": activity["serviceUrl"],
                    "recipient": recipient,
                    "sender": sender,
                    "user_id": user_id,
                    "personal": not group,
                },
            )
            store.ingest(message)
        self.store = store
        return {}

    def _route(self, thread_id: str) -> tuple[str, dict[str, Any]]:
        if self.store is None:
            raise ChannelError("Teams callback routes are unavailable")
        tenant, conversation, root = json.loads(thread_id)
        if tenant != self.config.tenant_id:
            raise ChannelError("Teams destination belongs to a different tenant")
        route = self.store.get("teams-route:" + thread_id)
        if not route:
            raise ChannelError("Teams destination requires a previously authenticated conversation")
        service = self._service_url(route["service_url"])
        path = f"/v3/conversations/{quote(conversation, safe='')}/activities"
        if root:
            path += "/" + quote(root, safe="")
        return service + path, route

    async def _send_activity(self, thread_id: str, content: dict[str, Any]) -> str:
        url, route = self._route(thread_id)
        response = await self.api(
            "POST",
            url,
            data={
                "type": "message",
                "from": route["recipient"],
                "recipient": route["sender"],
                **content,
            },
            auth=f"Bearer {await self._access()}",
        )
        return str(response.get("id") or "")

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        await self._send_activity(thread_id, {"text": text, "textFormat": "plain"})

    async def send_approval(
        self, thread_id: str, text: str, delivery_id: str, buttons: dict[str, str]
    ) -> str:
        return await self._send_activity(
            thread_id,
            {
                "attachments": [
                    {
                        "contentType": "application/vnd.microsoft.card.adaptive",
                        "content": {
                            "type": "AdaptiveCard",
                            "version": "1.4",
                            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                            "body": [{"type": "TextBlock", "text": text, "wrap": True}],
                            "actions": [
                                {
                                    "type": "Action.Execute",
                                    "verb": "harness_approval",
                                    "title": "Approve",
                                    "associatedInputs": "none",
                                    "data": {"harness_approval": buttons["approve"]},
                                },
                                {
                                    "type": "Action.Execute",
                                    "verb": "harness_approval",
                                    "title": "Deny",
                                    "associatedInputs": "none",
                                    "data": {"harness_approval": buttons["deny"]},
                                },
                            ],
                        },
                    }
                ],
            },
        )

    async def _approval_callback(
        self, activity: dict[str, Any], store: ChannelStore
    ) -> dict[str, Any]:
        from harness.cli.channels.approval_interactions import accept_choice
        from harness.core.channel_interactions import lookup

        token = activity["value"].get("harness_approval", "")
        row = lookup(store, token)
        if not row:
            return {"text": "Unavailable, expired, or already used."}
        _, route = self._route(row["thread_id"])
        if (
            activity.get("recipient", {}).get("id") != route["recipient"].get("id")
            or activity.get("serviceUrl") != route["service_url"]
        ):
            raise AuthenticationError("Teams approval callback route does not match its card")
        sender = activity.get("from", {})
        accepted = await accept_choice(
            self,
            store,
            token=token,
            user_id=self.config.tenant_id
            + ":"
            + str(sender.get("aadObjectId") or sender.get("id") or ""),
            channel_id=self.config.tenant_id
            + ":"
            + str(activity.get("conversation", {}).get("id") or ""),
            native_message_id=str(activity.get("replyToId") or ""),
            event_id=str(activity.get("id") or ""),
        )
        return {
            "text": "Decision queued." if accepted else "Unavailable, expired, or already used."
        }

    @staticmethod
    def _file_url(value: object) -> str:
        if not isinstance(value, str) or len(value) > 16384:
            raise ChannelError("Invalid Teams file resource URL")
        parsed = urlsplit(value)
        host = parsed.hostname or ""
        if (
            parsed.scheme != "https"
            or parsed.username
            or parsed.password
            or parsed.fragment
            or parsed.port not in {None, 443}
            or not any(host.endswith("." + domain) for domain in ("sharepoint.com", "1drv.com"))
        ):
            raise ChannelError("Teams file transfer requires a native SharePoint/OneDrive URL")
        return value

    def _secret_url(self, value: str) -> str:
        self.secrets.append(value)
        self.secrets.extend(v for _, v in parse_qsl(urlsplit(value).query) if v)
        return value

    async def prepare_media(self, message: ChannelMessage) -> list[MediaAttachment]:
        if len(message.attachments) > ATTACHMENT_LIMIT:
            raise ChannelError("Too many native attachments (maximum 16)")
        resources = []
        for item in message.attachments:
            kind = str(item.get("contentType", ""))
            if kind == FILE_DOWNLOAD:
                if message.group or not item.get("personal"):
                    raise ChannelError(
                        "Teams channel/group files require separate Microsoft Graph authorization"
                    )
                content = item.get("content", {})
                self._file_url(item.get("contentUrl"))
                url = self._file_url(content.get("downloadUrl"))
                resources.append((item, url, False, ""))
            elif kind.startswith("image/"):
                url = str(item.get("contentUrl", ""))
                parsed = urlsplit(url)
                service = urlsplit(self._service_url(str(item.get("service_url", ""))))
                host = parsed.hostname or ""
                connector = parsed.netloc == service.netloc and parsed.path.startswith(
                    service.path.rstrip("/") + "/v3/attachments/"
                )
                skype = host.endswith(".asm.skype.com") and parsed.path.startswith("/v1/objects/")
                if (
                    parsed.scheme != "https"
                    or parsed.username
                    or parsed.password
                    or parsed.fragment
                    or parsed.port not in {None, 443}
                    or not (connector or skype)
                ):
                    raise ChannelError("Teams inline image is not a native Connector resource")
                resources.append((item, url, True, kind))
            else:
                raise ChannelError(
                    "Unsupported Teams attachment; only native images and personal-chat files can be downloaded"
                )
        remaining = MEDIA_LIMIT
        result = []
        for item, url, authenticated, mime in resources:
            data, received_mime = await download_bytes(
                self,
                self._secret_url(url),
                remaining=remaining,
                auth=f"Bearer {await self._access()}" if authenticated else "",
            )
            remaining -= len(data)
            result.append(media_attachment(data, item.get("name"), mime or received_mime))
        return result

    def _file_consent(self, activity: dict[str, Any], store: ChannelStore) -> dict[str, Any]:
        value = activity.get("value", {})
        context = value.get("context", {})
        consent_id = context.get("harness_consent")
        if not isinstance(consent_id, str) or len(consent_id) != 32:
            raise AuthenticationError("Unknown Teams file consent")
        record = store.get("teams-consent:" + consent_id)
        sender, conversation = activity.get("from", {}), activity.get("conversation", {})
        owner = self.config.tenant_id + ":" + str(sender.get("aadObjectId") or sender.get("id", ""))
        thread = json.dumps(
            [self.config.tenant_id, conversation.get("id"), ""], separators=(",", ":")
        )
        if (
            not record
            or value.get("type") != "fileUpload"
            or conversation.get("conversationType") != "personal"
            or record["thread"] != thread
            or record["owner"] != owner
            or record["sender_id"] != sender.get("id")
            or record["recipient_id"] != activity.get("recipient", {}).get("id")
            or record["service_url"] != activity.get("serviceUrl")
            or not self.config.permits(
                user_id=owner, channel_id=self.channel_id(thread), group=False, mentioned=False
            )
        ):
            raise AuthenticationError(
                "Teams file consent does not belong to this recipient and conversation"
            )
        if record["status"] in {"accepted", "uploaded", "delivered", "declined", "expired"}:
            return {"status": 200}
        if time.time() > record["expires"]:
            record["status"] = "expired"
        elif value.get("action") == "decline":
            record["status"] = "declined"
        elif value.get("action") == "accept":
            info = value.get("uploadInfo", {})
            self._file_url(info.get("uploadUrl"))
            self._file_url(info.get("contentUrl"))
            if not info.get("uniqueId") or media_name(info.get("name")) != record["name"]:
                raise AuthenticationError(
                    "Teams consent upload metadata does not match the offered file"
                )
            record["upload"] = {
                k: info.get(k, "")
                for k in ("uploadUrl", "contentUrl", "uniqueId", "fileType", "name")
            }
            record["status"] = "accepted"
        else:
            raise ChannelError("Unsupported Teams file consent action")
        store.set("teams-consent:" + consent_id, record)
        return {"status": 200}

    async def send_media(
        self, thread_id: str, attachment: MediaAttachment, delivery_id: str
    ) -> None:
        raw = attachment_bytes(attachment)
        name = media_name(attachment.name)
        mime = media_type(attachment.mime_type, name)
        _, route = self._route(thread_id)
        if attachment.kind == "image" and len(raw) <= 20 * 1024:
            # Teams supports base64 contentUrl for inline images; enforce the
            # connector's small message payload budget before issuing a request.
            await self._send_activity(
                thread_id,
                {
                    "attachments": [
                        {
                            "contentType": mime,
                            "name": name,
                            "contentUrl": f"data:{mime};base64,{attachment.data}",
                        }
                    ]
                },
            )
            return
        if not route.get("personal"):
            raise DeliveryRejected(
                "Teams file uploads require a personal chat with supportsFiles enabled; channel/group uploads require separate Graph authorization"
            )
        if not raw:
            raise DeliveryRejected("Teams file consent uploads require a nonempty file")
        assert self.store is not None
        async with self.media_lock:
            state_key = "teams-delivery-consent:" + delivery_id
            consent_id = self.store.get(state_key)
            record = self.store.get("teams-consent:" + consent_id) if consent_id else None
            if record is None:
                consent_id = uuid.uuid4().hex
                record = {
                    "thread": thread_id,
                    "owner": route["user_id"],
                    "sender_id": route["sender"]["id"],
                    "recipient_id": route["recipient"]["id"],
                    "service_url": route["service_url"],
                    "name": name,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "expires": time.time() + 900,
                    "status": "requesting",
                }
                self.store.set("teams-consent:" + consent_id, record)
                self.store.set(state_key, consent_id)
                context = {"harness_consent": consent_id}
                await self._send_activity(
                    thread_id,
                    {
                        "attachments": [
                            {
                                "contentType": FILE_CONSENT,
                                "name": name,
                                "content": {
                                    "description": "Harness generated file",
                                    "sizeInBytes": len(raw),
                                    "acceptContext": context,
                                    "declineContext": context,
                                },
                            }
                        ]
                    },
                )
                # An immediate invoke can race the send response; preserve it.
                record = self.store.get("teams-consent:" + consent_id)
                if record["status"] == "requesting":
                    record["status"] = "awaiting_consent"
                    self.store.set("teams-consent:" + consent_id, record)
            if record["thread"] != thread_id or record["sha256"] != hashlib.sha256(raw).hexdigest():
                raise DeliveryRejected("Teams pending file does not match this delivery")
            if record["status"] == "delivered":
                return
            if record["status"] in {"declined", "expired"}:
                raise DeliveryRejected("Teams file consent was declined or expired")
            if record["status"] != "uploaded" and time.time() > record["expires"]:
                record["status"] = "expired"
                self.store.set("teams-consent:" + consent_id, record)
                raise DeliveryRejected("Teams file consent expired")
            if record["status"] in {"requesting", "awaiting_consent"}:
                raise DeliveryDeferred("Waiting for this recipient's Teams file consent", delay=5)
            if record["status"] == "accepted":
                info = record["upload"]
                async with asyncio.timeout(60):
                    response = await self.client.put(
                        self._secret_url(self._file_url(info["uploadUrl"])),
                        content=raw,
                        headers={
                            "Content-Type": "application/octet-stream",
                            "Content-Range": f"bytes 0-{len(raw) - 1}/{len(raw)}",
                        },
                        follow_redirects=False,
                    )
                if response.status_code not in {200, 201}:
                    raise ChannelError("Teams OneDrive upload did not complete")
                record["status"] = "uploaded"
                self.store.set("teams-consent:" + consent_id, record)
            info = record["upload"]
            await self._send_activity(
                thread_id,
                {
                    "attachments": [
                        {
                            "contentType": FILE_INFO,
                            "contentUrl": info["contentUrl"],
                            "name": name,
                            "content": {"uniqueId": info["uniqueId"], "fileType": info["fileType"]},
                        }
                    ]
                },
            )
            record["status"] = "delivered"
            record.pop("upload", None)
            self.store.set("teams-consent:" + consent_id, record)
