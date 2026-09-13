"""Google Chat HTTPS interaction events and service-account REST replies."""

from __future__ import annotations

import hashlib
import html
import json
import time
import uuid
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urlencode, urlsplit

import jwt

from harness.cli.channels.native_media import (
    ATTACHMENT_LIMIT,
    MEDIA_LIMIT,
    attachment_bytes,
    download_bytes,
    media_attachment,
    media_name,
    media_type,
    resource_segment,
)
from harness.cli.channels.transports import ChannelError
from harness.cli.channels.webhooks import AuthenticationError, WebhookTransport
from harness.core.gateway_channels import ChannelMessage, ChannelStore
from harness.core.schemas import MediaAttachment

GOOGLE_KEYS = "https://www.googleapis.com/oauth2/v3/certs"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
QUESTION_ACTION = "harness_question"


class GoogleChatTransport(WebhookTransport):
    name = "google_chat"
    limit = 4000

    async def authenticate(self) -> None:
        audience = urlsplit(self.config.audience)
        if (
            audience.scheme != "https"
            or not audience.hostname
            or audience.username
            or audience.password
        ):
            raise ChannelError(
                "Google Chat audience must be its configured public HTTPS endpoint URL"
            )
        try:
            account = json.loads(self.token)
            if account["type"] != "service_account" or not account["client_email"].endswith(
                ".gserviceaccount.com"
            ):
                raise ValueError
            self.account = account
            self.secrets.append(account["private_key"])
        except (ValueError, KeyError, TypeError):
            raise ChannelError("Google Chat token_env must contain service-account JSON") from None
        self.identity = account["client_email"]
        self.bot_id = self.config.app_id
        self.access_token = ""
        self.token_expires = 0.0
        self.keys: dict[str, Any] = {}
        self.keys_fetched = 0.0
        self.user_account: dict[str, Any] | None = None
        self.user_access_token = ""
        self.user_token_expires = 0.0
        if self.app_token:
            try:
                account = json.loads(self.app_token)
                scopes = account.get("scopes", [])
                if (
                    account.get("type") != "authorized_user"
                    or not all(
                        isinstance(account.get(k), str) and account[k]
                        for k in ("client_id", "client_secret", "refresh_token")
                    )
                    or not isinstance(scopes, list)
                    or not any(
                        scope in scopes
                        for scope in (
                            "https://www.googleapis.com/auth/chat.messages.create",
                            "https://www.googleapis.com/auth/chat.messages",
                        )
                    )
                ):
                    raise ValueError
                self.user_account = account
                self.secrets.extend([account["client_secret"], account["refresh_token"]])
            except (ValueError, TypeError, AttributeError):
                raise ChannelError(
                    "Google Chat app_token_env requires authorized_user OAuth JSON with an explicit Chat message creation scope"
                ) from None
        await self._access()

    async def _user_access(self) -> str:
        if self.user_account is None:
            raise ChannelError(
                "Google Chat attachment upload requires authorized-user OAuth in app_token_env; chat.bot cannot upload files"
            )
        if time.time() < self.user_token_expires - 60:
            return self.user_access_token
        response = await self.client.post(
            TOKEN_ENDPOINT,
            data={
                "grant_type": "refresh_token",
                **{
                    key: self.user_account[key]
                    for key in ("client_id", "client_secret", "refresh_token")
                },
            },
            follow_redirects=False,
        )
        if not response.is_success or not response.json().get("access_token"):
            raise ChannelError("Google Chat authorized-user token refresh failed")
        payload = response.json()
        granted = str(payload.get("scope", "")).split()
        if granted and not any(
            scope in granted
            for scope in (
                "https://www.googleapis.com/auth/chat.messages.create",
                "https://www.googleapis.com/auth/chat.messages",
            )
        ):
            raise ChannelError("Google Chat user token lacks attachment upload scope")
        self.user_access_token = str(payload["access_token"])
        self.secrets.append(self.user_access_token)
        self.user_token_expires = time.time() + min(3600, float(payload.get("expires_in", 3600)))
        return self.user_access_token

    async def _access(self) -> str:
        now = time.time()
        if now < self.token_expires - 60:
            return self.access_token
        assertion = jwt.encode(
            {
                "iss": self.account["client_email"],
                "scope": "https://www.googleapis.com/auth/chat.bot",
                "aud": TOKEN_ENDPOINT,
                "iat": int(now),
                "exp": int(now) + 3600,
            },
            self.account["private_key"],
            algorithm="RS256",
        )
        self.secrets.append(assertion)
        response = await self.client.post(
            TOKEN_ENDPOINT,
            data={
                "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
                "assertion": assertion,
            },
        )
        if not response.is_success:
            raise ChannelError("Google service-account token exchange failed")
        payload = response.json()
        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise ChannelError("Google token exchange omitted its token")
        self.access_token = token
        self.secrets.append(token)
        self.token_expires = now + min(3600, float(payload.get("expires_in", 3600)))
        return token

    async def _verify(self, headers: Mapping[str, str]) -> None:
        authorization = headers.get("Authorization", "")
        if not authorization.startswith("Bearer ") or len(authorization) > 16384:
            raise AuthenticationError("Google Chat authentication required")
        token = authorization[7:]
        try:
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "RS256" or not isinstance(header.get("kid"), str):
                raise ValueError
            now = time.time()
            if now - self.keys_fetched > 3600 or (
                header["kid"] not in self.keys and now - self.keys_fetched > 60
            ):
                payload = await self.api("GET", GOOGLE_KEYS)
                self.keys = {
                    key["kid"]: jwt.PyJWK.from_dict(key).key
                    for key in payload["keys"]
                    if key.get("kty") == "RSA" and key.get("alg", "RS256") == "RS256"
                }
                self.keys_fetched = now
            key = self.keys.get(header["kid"])
            if key is None:
                raise ValueError
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                audience=self.config.audience,
                issuer=["https://accounts.google.com", "accounts.google.com"],
                leeway=30,
                options={"require": ["exp", "iat", "iss", "aud", "email", "email_verified"]},
            )
            if (
                claims["email"] != "chat@system.gserviceaccount.com"
                or claims["email_verified"] is not True
            ):
                raise ValueError
        except (jwt.PyJWTError, KeyError, TypeError, ValueError):
            raise AuthenticationError("Google Chat identity token is invalid") from None

    def channel_id(self, thread_id: str) -> str:
        space, _ = json.loads(thread_id)
        return space

    async def handle_event(
        self, headers: Mapping[str, str], body: bytes, store: ChannelStore
    ) -> dict[str, Any]:
        await self._verify(headers)
        event = json.loads(body)
        if event.get("type") == "CARD_CLICKED":
            return await self._question_clicked(event, store)
        if event.get("type") != "MESSAGE":
            return {}
        message, space, user = event["message"], event["space"], event["user"]
        if user.get("type") != "HUMAN":
            return {}
        group = space.get("type") != "DM"
        root = message.get("thread", {}).get("name", "") if group else ""
        mentioned = any(
            annotation.get("type") == "USER_MENTION"
            and (
                annotation.get("userMention", {}).get("user", {}).get("name") == self.bot_id
                if self.bot_id
                else annotation.get("userMention", {}).get("user", {}).get("type") == "BOT"
            )
            for annotation in message.get("annotations", [])
        )
        self.accept(
            ChannelMessage(
                id=str(message["name"]),
                user_id=str(user["name"]),
                channel_id=str(space["name"]),
                thread_id=json.dumps([space["name"], root], separators=(",", ":")),
                text=str(message.get("argumentText", message.get("text", ""))),
                group=group,
                mentioned=mentioned,
                attachments=message.get("attachment", []),
            ),
            store,
        )
        return {}

    async def _question_clicked(self, event: dict[str, Any], store: ChannelStore) -> dict[str, Any]:
        from harness.cli.channels.question_interactions import accept_question_choice

        action, common = event.get("action", {}), event.get("common", {})
        message, space, user = (
            event.get("message", {}),
            event.get("space", {}),
            event.get("user", {}),
        )
        if not all(isinstance(item, dict) for item in (action, common, message, space, user)):
            return {}
        methods = [
            value
            for value in (action.get("actionMethodName"), common.get("invokedFunction"))
            if value
        ]
        if (
            not methods
            or any(value != QUESTION_ACTION for value in methods)
            or user.get("type") != "HUMAN"
        ):
            return {}
        parameters = action.get("parameters", [])
        common_parameters = common.get("parameters", {})
        if not isinstance(parameters, list) or not isinstance(common_parameters, dict):
            return {}
        tokens = [
            item.get("value")
            for item in parameters
            if isinstance(item, dict) and item.get("key") == "token"
        ]
        if "token" in common_parameters:
            tokens.append(common_parameters["token"])
        if (
            not tokens
            or not isinstance(tokens[0], str)
            or any(value != tokens[0] for value in tokens)
        ):
            return {}
        user_id, space_id, native_id, event_time = (
            user.get("name"),
            space.get("name"),
            message.get("name"),
            event.get("eventTime"),
        )
        if (
            not all(
                isinstance(value, str) and value
                for value in (user_id, space_id, native_id, event_time)
            )
            or not isinstance(event_time, str)
            or len(event_time) > 100
        ):
            return {}
        root = message.get("thread", {}).get("name", "") if space.get("type") != "DM" else ""
        thread_id = json.dumps([space_id, root], separators=(",", ":"))
        self._destination(thread_id)
        self._message_name(native_id, space_id)
        # Google interaction events have a timestamp rather than an event ID.
        # Include the card, actor and token so distinct toggles remain distinct.
        event_id = hashlib.sha256(
            json.dumps([native_id, user_id, tokens[0], event_time]).encode()
        ).hexdigest()
        outcome = await accept_question_choice(
            self,
            store,
            token=tokens[0],
            user_id=user_id,
            channel_id=space_id,
            native_message_id=native_id,
            event_id=event_id,
            thread_id=thread_id,
        )
        return {"text": outcome, "privateMessageViewer": {"name": user_id}}

    @staticmethod
    def _message_name(name: Any, space: str) -> str:
        if (
            not isinstance(name, str)
            or not name.startswith(space + "/messages/")
            or len(name.split("/")) != 4
        ):
            raise ChannelError("Google Chat response omitted its owned message identity")
        resource_segment(name.split("/")[-1])
        return name

    @staticmethod
    def _destination(thread_id: str) -> tuple[str, str]:
        space, root = json.loads(thread_id)
        if (
            not isinstance(space, str)
            or not space.startswith("spaces/")
            or len(space.split("/")) != 2
            or any(char in space for char in "?#%")
        ):
            raise ChannelError("Invalid Google Chat space identity")
        if root and (not root.startswith(space + "/threads/") or len(root.split("/")) != 4):
            raise ChannelError("Google Chat thread belongs to a different space")
        resource_segment(space.split("/")[-1])
        if root:
            resource_segment(root.split("/")[-1])
        return space, root

    async def prepare_media(self, message: ChannelMessage) -> list[MediaAttachment]:
        space, _ = self._destination(message.thread_id)
        if not message.id.startswith(space + "/messages/") or len(message.id.split("/")) != 4:
            raise ChannelError("Google Chat attachment message belongs to a different space")
        if len(message.attachments) > ATTACHMENT_LIMIT:
            raise ChannelError("Too many native attachments (maximum 16)")
        resources = []
        for item in message.attachments:
            if item.get("driveDataRef") or item.get("source") == "DRIVE_FILE":
                raise ChannelError(
                    "Google Drive attachments require separate Drive authorization and are not supported by this Chat transport"
                )
            name = str(item.get("name", ""))
            if not name.startswith(message.id + "/attachments/") or len(name.split("/")) != 6:
                raise ChannelError("Google Chat attachment belongs to a different message")
            resource_segment(name.split("/")[-1])
            resource = item.get("attachmentDataRef", {}).get("resourceName")
            if not isinstance(resource, str) or not resource or len(resource) > 8192:
                raise ChannelError("Google Chat attachment omitted its media resource")
            # Resource names are opaque. Quote every path component, rejecting
            # traversal before httpx normalizes dot segments. Never use downloadUri.
            for segment in resource.split("/"):
                if segment in {"", ".", ".."}:
                    raise ChannelError("Invalid Google Chat media resource")
            resources.append((item, quote(resource, safe="/")))
        result = []
        remaining = MEDIA_LIMIT
        for item, resource in resources:
            data, mime = await download_bytes(
                self,
                f"https://chat.googleapis.com/v1/media/{resource}?alt=media",
                remaining=remaining,
                auth=f"Bearer {await self._access()}",
            )
            remaining -= len(data)
            result.append(
                media_attachment(data, item.get("contentName"), item.get("contentType") or mime)
            )
        return result

    async def _send_payload(
        self, thread_id: str, data: dict[str, Any], delivery_id: str, token: str
    ) -> Any:
        space, root = self._destination(thread_id)
        query = {"requestId": str(uuid.uuid5(uuid.NAMESPACE_URL, delivery_id))}
        if root:
            data["thread"] = {"name": root}
            query["messageReplyOption"] = "REPLY_MESSAGE_OR_FAIL"
        return await self.api(
            "POST",
            f"https://chat.googleapis.com/v1/{space}/messages?{urlencode(query)}",
            data=data,
            auth=f"Bearer {token}",
        )

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        self._destination(thread_id)
        await self._send_payload(thread_id, {"text": text}, delivery_id, await self._access())

    async def send_question(
        self, thread_id: str, text: str, delivery_id: str, buttons: list[dict[str, str]]
    ) -> str:
        from harness.cli.channels.question_interactions import display_text

        space, _ = self._destination(thread_id)
        if not 1 <= len(buttons) <= 6 or any(
            not button.get("token", "").startswith("hq_") or len(button["token"]) != 35
            for button in buttons
        ):
            raise ChannelError("Invalid Google Chat clarification controls")
        text = display_text(text, self.limit)
        payload = await self._send_payload(
            thread_id,
            {
                "text": text,
                "cardsV2": [
                    {
                        "cardId": "question_"
                        + hashlib.sha256(delivery_id.encode()).hexdigest()[:32],
                        "card": {
                            "header": {"title": "Clarification"},
                            "sections": [
                                {
                                    "widgets": [
                                        {
                                            "textParagraph": {
                                                "text": html.escape(text).replace("\n", "<br>")
                                            }
                                        },
                                        {
                                            "buttonList": {
                                                "buttons": [
                                                    {
                                                        "text": html.escape(
                                                            display_text(button["label"], 80)
                                                        ),
                                                        "onClick": {
                                                            "action": {
                                                                "function": QUESTION_ACTION,
                                                                "parameters": [
                                                                    {
                                                                        "key": "token",
                                                                        "value": button["token"],
                                                                    }
                                                                ],
                                                            }
                                                        },
                                                    }
                                                    for button in buttons
                                                ]
                                            }
                                        },
                                    ]
                                }
                            ],
                        },
                    }
                ],
            },
            delivery_id,
            await self._access(),
        )
        return self._message_name(payload.get("name"), space)

    async def send_media(
        self, thread_id: str, attachment: MediaAttachment, delivery_id: str
    ) -> None:
        space, _ = self._destination(thread_id)
        data = attachment_bytes(attachment)
        token = await self._user_access()
        filename = media_name(attachment.name)
        # Google media upload uses multipart/related (JSON metadata + raw bytes),
        # not multipart/form-data. An unpredictable boundary avoids collisions.
        boundary = "harness_" + uuid.uuid4().hex
        metadata = json.dumps({"filename": filename}).encode()
        mime = media_type(attachment.mime_type, filename)
        body = (
            f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode()
            + metadata
            + f"\r\n--{boundary}\r\nContent-Type: {mime}\r\n\r\n".encode()
            + data
            + f"\r\n--{boundary}--\r\n".encode()
        )
        response = await self.client.post(
            f"https://chat.googleapis.com/upload/v1/{space}/attachments:upload?uploadType=multipart",
            content=body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": f"multipart/related; boundary={boundary}",
            },
            follow_redirects=False,
        )
        if not response.is_success or not response.json().get("attachmentDataRef"):
            raise ChannelError("Google Chat attachment upload failed")
        await self._send_payload(
            thread_id,
            {"attachment": [{"attachmentDataRef": response.json()["attachmentDataRef"]}]},
            delivery_id,
            token,
        )
