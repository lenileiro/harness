"""WeCom internal application callbacks with encrypted corporation binding."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import struct
import time
import xml.etree.ElementTree as ET
from urllib.parse import quote, urlencode

from aiohttp import web
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from harness.cli.channels.transports import ChannelError, RateLimited
from harness.cli.channels.webhooks import AuthenticationError, WebhookTransport
from harness.core.gateway_channels import ChannelMessage, ChannelStore


def xml_fields(body: bytes) -> dict[str, str]:
    if b"<!DOCTYPE" in body.upper() or b"<!ENTITY" in body.upper():
        raise ValueError("XML declarations are unsupported")
    root = ET.fromstring(body)
    if root.tag != "xml" or any(list(child) for child in root):
        raise ValueError("Invalid callback XML structure")
    fields = {child.tag: child.text or "" for child in root}
    if len(fields) != len(root):
        raise ValueError("Duplicate callback XML fields")
    return fields


class WeComTransport(WebhookTransport):
    name = "wecom"
    limit = 500
    content_types = frozenset({"text/xml", "application/xml"})
    http_methods = ("GET", "POST")

    async def authenticate(self) -> None:
        if not self.config.tenant_id or not self.config.app_id.isdigit() or not self.app_token:
            raise ChannelError(
                "WeCom requires tenant_id corporation, numeric app_id, app secret and callback token"
            )
        encoded_key = os.environ.get(self.config.signing_secret_env, "")
        try:
            if len(encoded_key) != 43:
                raise ValueError
            self.aes_key = base64.b64decode(encoded_key + "=", validate=True)
            if len(self.aes_key) != 32:
                raise ValueError
        except ValueError:
            raise ChannelError(
                "WeCom signing_secret_env must contain its 43-character EncodingAESKey"
            ) from None
        self.secrets.extend([encoded_key, quote(self.token, safe="")])
        self.access_token = ""
        self.token_expires = 0.0
        self.bot_id = self.config.app_id
        self.identity = json.dumps([self.config.tenant_id, self.bot_id])
        token = await self._access()
        account = await self.api(
            "GET",
            "https://qyapi.weixin.qq.com/cgi-bin/agent/get?"
            + urlencode({"access_token": token, "agentid": self.bot_id}),
        )
        if account.get("errcode", 0) or str(account.get("agentid", "")) != self.bot_id:
            raise ChannelError("WeCom application identity verification failed")

    async def _access(self) -> str:
        if self.access_token and time.time() < self.token_expires - 60:
            return self.access_token
        payload = await self.api(
            "GET",
            "https://qyapi.weixin.qq.com/cgi-bin/gettoken?"
            + urlencode({"corpid": self.config.tenant_id, "corpsecret": self.token}),
        )
        if payload.get("errcode", 0) or not payload.get("access_token"):
            raise ChannelError("WeCom application authentication failed")
        self.access_token = str(payload["access_token"])
        self.secrets.extend([self.access_token, quote(self.access_token, safe="")])
        self.token_expires = time.time() + min(7200, float(payload.get("expires_in", 7200)))
        return self.access_token

    def _decrypt(self, encrypted: str, signature: str, timestamp: str, nonce: str) -> bytes:
        expected = hashlib.sha1(
            "".join(sorted([self.app_token, timestamp, nonce, encrypted])).encode()
        ).hexdigest()
        if not hmac.compare_digest(expected, signature) or abs(time.time() - int(timestamp)) > 300:
            raise AuthenticationError("WeCom callback signature or timestamp verification failed")
        try:
            ciphertext = base64.b64decode(encrypted, validate=True)
            decryptor = Cipher(
                algorithms.AES(self.aes_key), modes.CBC(self.aes_key[:16])
            ).decryptor()
            plain = decryptor.update(ciphertext) + decryptor.finalize()
            padding = plain[-1]
            if not 1 <= padding <= 32 or plain[-padding:] != bytes([padding]) * padding:
                raise ValueError
            plain = plain[:-padding]
            if len(plain) < 20:
                raise ValueError
            size = struct.unpack("!I", plain[16:20])[0]
            if plain[20 + size :] != self.config.tenant_id.encode() or size > len(plain) - 20:
                raise ValueError
            return plain[20 : 20 + size]
        except (ValueError, IndexError):
            raise AuthenticationError("WeCom encrypted callback verification failed") from None

    def channel_id(self, thread_id: str) -> str:
        corp, agent, user = json.loads(thread_id)
        if corp != self.config.tenant_id or agent != self.bot_id:
            raise ChannelError("WeCom destination belongs to a different corporation/application")
        return corp + ":" + user

    async def handle_request(
        self, request: web.Request, body: bytes, store: ChannelStore
    ) -> web.Response:
        if len(list(request.query.items())) != len(set(request.query)):
            raise AuthenticationError("Duplicate WeCom callback parameters")
        encrypted = (
            request.query["echostr"] if request.method == "GET" else xml_fields(body)["Encrypt"]
        )
        plain = self._decrypt(
            encrypted,
            request.query["msg_signature"],
            request.query["timestamp"],
            request.query["nonce"],
        )
        if request.method == "GET":
            return web.Response(body=plain, content_type="text/plain")
        fields = xml_fields(plain)
        if (
            fields.get("ToUserName") != self.config.tenant_id
            or fields.get("AgentID") != self.bot_id
        ):
            raise AuthenticationError("WeCom callback corporation or application mismatch")
        if fields.get("MsgType") == "text" and fields.get("MsgId"):
            user = fields.get("FromUserName", "")
            if user and not any(char in "|@\r\n " for char in user):
                self.accept(
                    ChannelMessage(
                        id=fields["MsgId"],
                        user_id=self.config.tenant_id + ":" + user,
                        channel_id=self.config.tenant_id + ":" + user,
                        thread_id=json.dumps(
                            [self.config.tenant_id, self.bot_id, user], separators=(",", ":")
                        ),
                        text=fields.get("Content", ""),
                    ),
                    store,
                )
        return web.Response(text="success", content_type="text/plain")

    async def send(self, thread_id: str, text: str, delivery_id: str) -> None:
        owner = self.channel_id(thread_id)
        _, _, user = json.loads(thread_id)
        if (
            owner not in self.config.allowed_users
            or not user
            or any(char in "|@\r\n " for char in user)
        ):
            raise ChannelError("WeCom reply requires exactly one allowed user")
        for attempt in range(2):
            token = await self._access()
            result = await self.api(
                "POST",
                "https://qyapi.weixin.qq.com/cgi-bin/message/send?"
                + urlencode({"access_token": token}),
                data={
                    "touser": user,
                    "agentid": int(self.bot_id),
                    "msgtype": "text",
                    "text": {"content": text},
                    "safe": 0,
                },
            )
            code = result.get("errcode", 0)
            if code in {40014, 42001} and attempt == 0:
                self.token_expires = 0
                continue
            if code in {45009, 45011}:
                raise RateLimited(60)
            if code or result.get("invaliduser") or result.get("unlicenseduser"):
                raise ChannelError("WeCom rejected the application message")
            return
