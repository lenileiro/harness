"""Native text receipts and edits; callers enforce persisted owner/thread scope."""

from __future__ import annotations

import json
import re
from urllib.parse import quote

from harness.cli.channels.feishu import FeishuTransport
from harness.cli.channels.google_chat import GoogleChatTransport
from harness.cli.channels.native_media import resource_segment
from harness.cli.channels.transports import (
    ChannelError,
    DiscordTransport,
    RateLimited,
    SlackTransport,
    TelegramTransport,
    Transport,
)

SUPPORTED = {"telegram", "discord", "slack", "feishu", "lark", "google_chat"}


def _slack(transport: SlackTransport, thread: str) -> tuple[str, str]:
    team, channel, root = thread.split(":", 2)
    if team != transport.team_id or not re.fullmatch(r"[A-Z0-9]+", channel):
        raise ChannelError("Progress conversation belongs to another Slack workspace")
    return channel, root


async def send_preview(transport: Transport, thread: str, text: str, key: str) -> str:
    if isinstance(transport, TelegramTransport):
        chat, _, topic = thread.partition(":")
        result = await transport.call(
            "sendMessage",
            {
                "chat_id": chat,
                "text": text,
                "disable_notification": True,
                "link_preview_options": {"is_disabled": True},
                **({"message_thread_id": int(topic)} if topic else {}),
            },
        )
        if str(result.get("chat", {}).get("id")) != chat:
            raise ChannelError("Telegram progress response has a different conversation")
        receipt = str(result.get("message_id", ""))
    elif isinstance(transport, DiscordTransport):
        if not thread.isdigit():
            raise ChannelError("Invalid Discord progress conversation")
        result = await transport.call(
            "POST",
            f"channels/{thread}/messages",
            {
                "content": text,
                "allowed_mentions": {"parse": []},
                "flags": 4096,
                "nonce": key[:24],
                "enforce_nonce": True,
            },
        )
        if str(result.get("channel_id")) != thread:
            raise ChannelError("Discord progress response has a different conversation")
        receipt = str(result.get("id", ""))
    elif isinstance(transport, SlackTransport):
        channel, root = _slack(transport, thread)
        result = await transport.call(
            "chat.postMessage",
            {
                "channel": channel,
                "text": text,
                "mrkdwn": False,
                "parse": "none",
                "unfurl_links": False,
                "unfurl_media": False,
                **({"thread_ts": root} if root else {}),
            },
        )
        if result.get("channel") != channel:
            raise ChannelError("Slack progress response has a different conversation")
        receipt = str(result.get("ts", ""))
    elif isinstance(transport, GoogleChatTransport):
        space, _ = transport._destination(thread)
        result = await transport._send_payload(
            thread, {"text": text}, key, await transport._access()
        )
        return transport._message_name(result.get("name"), space)
    elif isinstance(transport, FeishuTransport):
        receipt = await transport._send_content(thread, "text", {"text": text}, key)
    else:
        raise ChannelError("This transport has no native progress support")
    validate_receipt(transport, thread, receipt)
    return receipt


def validate_receipt(transport: Transport, thread: str, receipt: str) -> None:
    if not receipt or len(receipt) > 1024:
        raise ChannelError("Progress response omitted its message identity")
    if isinstance(transport, GoogleChatTransport):
        space, _ = transport._destination(thread)
        transport._message_name(receipt, space)
    elif isinstance(transport, SlackTransport):
        _slack(transport, thread)
        if not re.fullmatch(r"[0-9]+\.[0-9]+", receipt):
            raise ChannelError("Invalid Slack progress message identity")
    elif isinstance(transport, (TelegramTransport, DiscordTransport)):
        if not receipt.isascii() or not receipt.isdigit():
            raise ChannelError("Invalid progress message identity")
    else:
        resource_segment(receipt)


async def edit_preview(transport: Transport, thread: str, receipt: str, text: str) -> None:
    validate_receipt(transport, thread, receipt)
    if isinstance(transport, TelegramTransport):
        response = await transport.client.post(
            f"https://api.telegram.org/bot{transport.token}/editMessageText",
            json={
                "chat_id": thread.partition(":")[0],
                "message_id": int(receipt),
                "text": text,
                "link_preview_options": {"is_disabled": True},
            },
        )
        result = response.json()
        if result.get("ok") or (
            result.get("error_code") == 400
            and "message is not modified" in str(result.get("description", "")).lower()
        ):
            return
        if response.status_code == 429 or result.get("error_code") == 429:
            raise RateLimited(result.get("parameters", {}).get("retry_after", 1))
        raise ChannelError("Telegram progress edit rejected")
    if isinstance(transport, DiscordTransport):
        if not thread.isdigit():
            raise ChannelError("Invalid Discord progress conversation")
        await transport.call(
            "PATCH",
            f"channels/{thread}/messages/{receipt}",
            {
                "content": text,
                "allowed_mentions": {"parse": []},
            },
        )
    elif isinstance(transport, SlackTransport):
        channel, _ = _slack(transport, thread)
        await transport.call(
            "chat.update",
            {
                "channel": channel,
                "ts": receipt,
                "text": text,
                "mrkdwn": False,
                "parse": "none",
                "link_names": False,
            },
        )
    elif isinstance(transport, GoogleChatTransport):
        await transport.api(
            "PATCH",
            f"https://chat.googleapis.com/v1/{receipt}?updateMask=text",
            data={"text": text},
            auth=f"Bearer {await transport._access()}",
        )
    elif isinstance(transport, FeishuTransport):
        result = await transport.api(
            "PUT",
            f"{transport.base}/im/v1/messages/{quote(receipt, safe='')}",
            data={"msg_type": "text", "content": json.dumps({"text": text})},
            auth=f"Bearer {await transport._access()}",
        )
        if result.get("code", 0) != 0:
            raise ChannelError("Feishu/Lark progress edit rejected")
    else:
        raise ChannelError("This transport has no native progress support")


async def send_typing(transport: Transport, thread: str) -> None:
    if isinstance(transport, TelegramTransport):
        chat, _, topic = thread.partition(":")
        await transport.call(
            "sendChatAction",
            {
                "chat_id": chat,
                "action": "typing",
                **({"message_thread_id": int(topic)} if topic else {}),
            },
        )
    elif isinstance(transport, DiscordTransport):
        if not thread.isdigit():
            raise ChannelError("Invalid Discord progress conversation")
        response = await transport.client.post(
            f"https://discord.com/api/v10/channels/{thread}/typing",
            headers={"Authorization": f"Bot {transport.token}"},
        )
        if response.status_code == 429:
            raise RateLimited(float(response.headers.get("Retry-After", 1)))
        if response.status_code != 204:
            raise ChannelError("Discord typing indicator rejected")
