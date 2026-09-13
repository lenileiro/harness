"""Explicit, cancellable iLink QR pairing; importing this module performs no I/O."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlencode

import httpx

from harness.cli.channels.transports import ChannelError, SecretFilter
from harness.cli.channels.weixin import ILINK, ilink_base


class WeixinPairing:
    def __init__(self, *, client: httpx.AsyncClient | None = None):
        self.client = client or httpx.AsyncClient(
            timeout=40, follow_redirects=False, trust_env=False
        )
        self.owns_client = client is None
        self.base = ILINK
        self.qrcode = ""
        self.secrets: list[str] = []
        self.log_filter = SecretFilter(self.secrets)

    async def __aenter__(self):
        import logging

        logging.getLogger("httpx").addFilter(self.log_filter)
        return self

    async def __aexit__(self, *_):
        import logging

        try:
            if self.owns_client:
                await self.client.aclose()
        finally:
            logging.getLogger("httpx").removeFilter(self.log_filter)

    async def request(self, method: str, endpoint: str, *, data=None) -> dict[str, Any]:
        url = self.base + "/ilink/bot/" + endpoint
        if "?" in endpoint:
            self.secrets.append(url)
        try:
            response = await self.client.request(
                method,
                url,
                json=data,
                follow_redirects=False,
                headers={
                    "iLink-App-Id": "bot",
                    "iLink-App-ClientVersion": str((2 << 16) | (4 << 8) | 8),
                },
            )
        except httpx.TimeoutException:
            if endpoint.startswith("get_qrcode_status?"):
                return {"status": "wait"}
            raise ChannelError("Weixin QR creation timed out") from None
        except httpx.HTTPError:
            raise ChannelError("Weixin pairing connection interrupted") from None
        if not response.is_success:
            raise ChannelError("Weixin pairing request rejected")
        result = response.json()
        if not isinstance(result, dict) or result.get("ret", 0) not in {0, None}:
            raise ChannelError("Weixin pairing API reported an error")
        return result

    async def start(self) -> str:
        result = await self.request(
            "POST", "get_bot_qrcode?bot_type=3", data={"local_token_list": []}
        )
        qrcode, display = result.get("qrcode"), result.get("qrcode_img_content")
        if (
            not isinstance(qrcode, str)
            or not qrcode
            or len(qrcode) > 8192
            or not isinstance(display, str)
            or not display
            or len(display) > 8192
        ):
            raise ChannelError("Weixin pairing response omitted a bounded QR code")
        self.qrcode = qrcode
        self.secrets.extend([qrcode, display])
        return display

    async def poll(self, *, verify_code: str | None = None) -> dict[str, Any]:
        if not self.qrcode:
            raise ChannelError("Start Weixin pairing before polling")
        params = {"qrcode": self.qrcode}
        if verify_code:
            if len(verify_code) > 128:
                raise ValueError("Verification code is too long")
            params["verify_code"] = verify_code
            self.secrets.append(verify_code)
        result = await self.request("GET", "get_qrcode_status?" + urlencode(params))
        if result.get("status") == "scaned_but_redirect":
            self.base = ilink_base("https://" + str(result.get("redirect_host", "")))
        return result

    async def pair(
        self,
        *,
        display_qr: Callable[[str], Awaitable[None]],
        verification_code: Callable[[], Awaitable[str]] | None = None,
        timeout_seconds: float = 300,
    ) -> dict[str, str]:
        if not 1 <= timeout_seconds <= 600:
            raise ValueError("Pairing timeout must be between 1 and 600 seconds")
        async with asyncio.timeout(timeout_seconds):
            await display_qr(await self.start())
            code = None
            while True:
                result = await self.poll(verify_code=code)
                status = result.get("status")
                if status == "confirmed":
                    fields = {
                        key: result.get(key)
                        for key in ("bot_token", "ilink_bot_id", "ilink_user_id")
                    }
                    if not all(isinstance(value, str) and value for value in fields.values()):
                        raise ChannelError("Weixin pairing omitted account credentials or owner ID")
                    self.secrets.append(str(fields["bot_token"]))
                    return {
                        **{key: str(value) for key, value in fields.items()},
                        "base_url": ilink_base(result.get("baseurl") or self.base),
                    }
                if status in {"expired", "verify_code_blocked", "binded_redirect"}:
                    raise ChannelError(
                        "Weixin pairing expired, was blocked, or is already bound; restart pairing explicitly"
                    )
                if status == "need_verifycode":
                    if verification_code is None:
                        raise ChannelError("Weixin pairing requires an operator verification code")
                    code = await verification_code()
                    if not code:
                        raise ChannelError("Weixin verification code cannot be empty")
                elif status in {"wait", "scaned", "scaned_but_redirect"}:
                    if status == "scaned":
                        code = None
                else:
                    raise ChannelError("Weixin pairing returned an unsupported state")
                await asyncio.sleep(1)
