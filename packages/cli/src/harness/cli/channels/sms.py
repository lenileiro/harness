"""Twilio SMS webhooks bound to exact account, destination and sender."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from urllib.parse import parse_qsl, quote, urlsplit

from aiohttp import web

from harness.cli.channels.transports import ChannelError, RateLimited
from harness.cli.channels.webhooks import AuthenticationError, WebhookTransport
from harness.core.gateway_channels import ChannelMessage, ChannelStore


def twilio_signature(url: str, params: dict[str, str], token: str) -> str:
    message = url + "".join(key + params[key] for key in sorted(params))
    return base64.b64encode(
        hmac.new(token.encode(), message.encode(), hashlib.sha1).digest()
    ).decode()


class SMSTransport(WebhookTransport):
    name = "sms"
    limit = 1000
    content_types = frozenset({"application/x-www-form-urlencoded"})

    async def authenticate(self) -> None:
        url = urlsplit(self.config.webhook_url)
        if (
            url.scheme != "https"
            or not url.hostname
            or url.username
            or url.password
            or url.fragment
            or url.path != self.config.webhook_path
        ):
            raise ChannelError(
                "SMS requires the exact public HTTPS webhook_url matching webhook_path"
            )
        if (
            not self.config.app_id.startswith("AC")
            or not self.config.app_id.isalnum()
            or not self.config.username.startswith("+")
            or not self.config.username[1:].isdigit()
        ):
            raise ChannelError(
                "SMS requires Twilio account SID as app_id and E.164 sender number as username"
            )
        response = await self.client.get(
            f"https://api.twilio.com/2010-04-01/Accounts/{self.config.app_id}.json",
            auth=(self.config.app_id, self.token),
        )
        if not response.is_success or response.json().get("sid") != self.config.app_id:
            raise ChannelError("Twilio account authentication failed")
        self.bot_id = self.config.username
        self.identity = json.dumps([self.config.app_id, self.bot_id])

    def channel_id(self, thread_id: str) -> str:
        sender, recipient = json.loads(thread_id)
        if sender != self.bot_id:
            raise ChannelError("SMS reply belongs to a different sending number")
        return recipient

    async def handle_request(
        self, request: web.Request, body: bytes, store: ChannelStore
    ) -> web.Response:
        # Match the configured external URL rather than untrusted proxy headers.
        expected_url = urlsplit(self.config.webhook_url)
        if request.query_string != expected_url.query:
            raise AuthenticationError("SMS callback query does not match configured URL")
        pairs = parse_qsl(body.decode("utf-8"), keep_blank_values=True, max_num_fields=200)
        params = dict(pairs)
        if len(params) != len(pairs):
            raise AuthenticationError("Duplicate SMS parameters are ambiguous")
        expected = twilio_signature(self.config.webhook_url, params, self.token)
        if (
            not hmac.compare_digest(expected, request.headers.get("X-Twilio-Signature", ""))
            or params.get("AccountSid") != self.config.app_id
            or params.get("To") != self.bot_id
        ):
            raise AuthenticationError(
                "Twilio callback signature/account/destination verification failed"
            )
        sender = params.get("From", "")
        if (
            sender.startswith("+")
            and sender[1:].isdigit()
            and params.get("MessageSid")
            and params.get("Body")
        ):
            self.accept(
                ChannelMessage(
                    id=params["MessageSid"],
                    user_id=sender,
                    channel_id=sender,
                    thread_id=json.dumps([self.bot_id, sender], separators=(",", ":")),
                    text=params["Body"],
                ),
                store,
            )
        return web.Response(
            text='<?xml version="1.0" encoding="UTF-8"?><Response/>', content_type="text/xml"
        )

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        recipient = self.channel_id(thread_id)
        if (
            recipient not in self.config.allowed_users
            or not recipient.startswith("+")
            or not recipient[1:].isdigit()
        ):
            raise ChannelError("SMS destination is outside the allowed sender identities")
        response = await self.client.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{quote(self.config.app_id, safe='')}/Messages.json",
            data={"From": self.bot_id, "To": recipient, "Body": text},
            auth=(self.config.app_id, self.token),
        )
        if response.status_code == 429:
            raise RateLimited(float(response.headers.get("Retry-After", "60")))
        if not response.is_success:
            raise ChannelError(
                "Twilio rejected the SMS; inspect account permissions and destination"
            )
