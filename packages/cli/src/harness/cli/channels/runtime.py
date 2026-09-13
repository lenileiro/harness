from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout
from websockets.exceptions import ConnectionClosed

from harness.cli.channels.approval_interactions import pending_cards
from harness.cli.channels.bluebubbles import BlueBubblesTransport
from harness.cli.channels.buzz import BuzzTransport
from harness.cli.channels.dingtalk import DingTalkTransport
from harness.cli.channels.extra import EmailTransport, SignalTransport
from harness.cli.channels.feishu import FeishuTransport, LarkTransport
from harness.cli.channels.google_chat import GoogleChatTransport
from harness.cli.channels.irc import IRCTransport
from harness.cli.channels.line import LINETransport
from harness.cli.channels.matrix import MatrixTransport
from harness.cli.channels.mattermost import MattermostTransport
from harness.cli.channels.msgraph_webhook import MSGraphWebhookTransport
from harness.cli.channels.ntfy import NtfyTransport
from harness.cli.channels.photon import PhotonTransport
from harness.cli.channels.progress import NativeProgress, deliver_final_preview, recover_progress
from harness.cli.channels.qqbot import QQBotTransport
from harness.cli.channels.question_interactions import capture_answer, pending_card
from harness.cli.channels.raft import RaftTransport
from harness.cli.channels.simplex import SimpleXTransport
from harness.cli.channels.sms import SMSTransport
from harness.cli.channels.teams import TeamsTransport
from harness.cli.channels.transports import (
    TRANSPORTS,
    ChannelError,
    DeliveryDeferred,
    DeliveryRejected,
    RateLimited,
    Transport,
)
from harness.cli.channels.wecom import WeComTransport
from harness.cli.channels.wecom_bot import WeComBotTransport
from harness.cli.channels.weixin import WeixinTransport
from harness.cli.channels.whatsapp_cloud import WhatsAppCloudTransport
from harness.cli.channels.yuanbao import YuanbaoTransport
from harness.core.channel_interactions import bind_message, lookup
from harness.core.gateway_channels import ChannelConfig, ChannelStore
from harness.core.gateway_router import is_gateway_control_message
from harness.core.schemas import MediaAttachment

GatewayReceiver = Callable[..., Awaitable[dict[str, Any]]]


async def process_message(
    *, cwd: Path, transport: Transport, store: ChannelStore, receiver: GatewayReceiver | None = None
) -> bool:
    message = store.claim_message()
    if message is None:
        return False
    try:
        output_attachments: list[dict[str, Any]] = []
        approval_cards: list[dict[str, Any]] = []
        question_card: dict[str, Any] | None = None
        if not transport.config.permits(
            user_id=message.user_id,
            channel_id=message.channel_id,
            group=message.group,
            mentioned=message.mentioned,
        ):
            store.complete(
                message,
                "This identity or conversation is no longer allowed.",
                limit=transport.limit,
            )
            return True
        text = await capture_answer(transport, store, message, message.text.strip())
        from harness.cli.gateway_conversation_commands import is_conversation_command

        if text.startswith("/") and not is_conversation_command(text):
            text = text[1:]
        command = text.split(maxsplit=1)[0].lower() if text else ""
        if (
            is_gateway_control_message(text)
            and not (message.attachments and not text)
            and command not in {"approve", "deny", "approvals", "status", "runs"}
            and message.user_id not in transport.config.operator_users
        ):
            reply = "This legacy workspace control requires a configured trusted operator. Ordinary chat and scoped approvals remain available."
        else:
            if receiver is None:
                from harness.cli.gateway_runtime import _run_gateway_receive_payload

                receiver = _run_gateway_receive_payload
            async with NativeProgress(transport, store, message):
                payload = await receiver(
                    working_dir=cwd,
                    message=text,
                    transport=transport.name,
                    user_id=message.user_id,
                    thread_id=message.thread_id,
                    max_steps=20,
                    **(
                        {"attachments": await transport.prepare_media(message)}
                        if message.attachments
                        else {}
                    ),
                )
            raw_reply = payload.get("reply")
            reply = (
                str(raw_reply.get("text") or "(empty response)")
                if isinstance(raw_reply, dict)
                else "(empty response)"
            )
            if isinstance(raw_reply, dict) and isinstance(raw_reply.get("data"), dict):
                output_attachments = [
                    MediaAttachment.model_validate(item).model_dump(mode="json")
                    for item in raw_reply["data"].get("attachments", [])
                ][:16]
                approval_cards = await pending_cards(
                    store, transport, message, raw_reply["data"].get("approval_ids", [])
                )
                question_card = await pending_card(
                    store, transport, message, raw_reply["data"].get("question_id", "")
                )
        store.complete(
            message,
            reply,
            limit=transport.limit,
            attachments=output_attachments,
            approval_cards=approval_cards,
            question_card=question_card,
        )
    except BaseException:
        store.fail_message(message)
        raise
    return True


async def deliver_message(*, transport: Transport, store: ChannelStore) -> bool:
    if not transport.can_send():
        return False
    if store.get("rate_limit_until", 0) > time.time():
        return False
    delivery = store.claim_delivery()
    if delivery is None:
        return False
    key = delivery["id"]
    if delivery["user_id"] not in transport.config.allowed_users or (
        transport.config.allowed_channels
        and transport.channel_id(delivery["thread_id"]) not in transport.config.allowed_channels
    ):
        store.delivery_result(key, status="failed", error="Destination owner is not allowed")
        return True
    try:
        if delivery.get("interaction"):
            buttons = json.loads(delivery["interaction"])
            if buttons.get("kind") == "question":
                from harness.core.question_interactions import (
                    bind_message as bind_question,
                )
                from harness.core.question_interactions import (
                    lookup as lookup_question,
                )

                question_buttons = buttons["buttons"]
                question = lookup_question(store, question_buttons[0]["token"])
                if (
                    not question
                    or question["mode"] != "choices"
                    or question["expires"] <= time.time()
                ):
                    raise DeliveryRejected("Question card expired or was already used")
                native_id = await transport.send_question(
                    delivery["thread_id"], delivery["text"], key, question_buttons
                )
                bind_question(store, key, native_id)
                store.delivery_result(key, status="sent")
                return True
            interaction = lookup(store, buttons.get("approve", ""))
            if not interaction or interaction["consumed"] or interaction["expires"] <= time.time():
                raise DeliveryRejected("Approval card expired or was already used")
            native_id = await transport.send_approval(
                delivery["thread_id"],
                delivery["text"],
                key,
                buttons,
            )
            bind_message(store, key, native_id)
        elif delivery.get("attachment"):
            await transport.send_media(
                delivery["thread_id"],
                MediaAttachment.model_validate(json.loads(delivery["attachment"])),
                key,
            )
        else:
            if not await deliver_final_preview(transport, store, delivery):
                await transport.send(delivery["thread_id"], delivery["text"], key)
    except DeliveryDeferred as exc:
        store.delivery_result(key, status="pending", retry_after=exc.delay, error=str(exc))
    except DeliveryRejected as exc:
        store.delivery_result(key, status="failed", error=str(exc))
    except RateLimited as exc:
        store.set("rate_limit_until", time.time() + exc.delay)
        store.delivery_result(key, status="pending", retry_after=exc.delay, error="Rate limited")
    except Exception:
        # A timed-out request might have posted successfully. Require operator
        # inspection instead of blindly duplicating a private message.
        store.delivery_result(
            key,
            status="uncertain",
            error="Send failed or was interrupted; inspect destination before retry",
        )
    else:
        store.delivery_result(key, status="sent")
    return True


async def run_transport(
    *, cwd: Path, transport: Transport, receiver: GatewayReceiver | None = None
) -> None:
    store = ChannelStore(cwd=cwd, transport=transport.name)
    lock = FileLock(store.path.with_suffix(".lock"), timeout=0, mode=0o600)
    try:
        try:
            lock.acquire()
        except Timeout:
            raise ChannelError("This channel is already running in this workspace") from None
        await transport.authenticate()
        store.bind_identity(transport.identity)
        store.recover()
        await recover_progress(transport, store)

        async def receive() -> None:
            failures = 0
            while True:
                try:
                    await transport.receive(store)
                    failures = 0
                    store.set("connection_error", "")
                    await asyncio.sleep(1)
                except RateLimited as exc:
                    store.set("connection", "rate_limited")
                    await asyncio.sleep(exc.delay)
                except ConnectionClosed as exc:
                    code = exc.rcvd.code if exc.rcvd else None
                    if transport.name == "discord" and code in {4007, 4009}:
                        store.set("resume", {})
                    if transport.name == "discord" and code in {
                        4004,
                        4010,
                        4011,
                        4012,
                        4013,
                        4014,
                    }:
                        store.set("connection", "authentication_or_intent_error")
                        raise ChannelError(
                            "Discord credentials or Gateway intents require operator correction"
                        ) from None
                    store.set("connection", "disconnected")
                    await asyncio.sleep(5)
                except ChannelError as exc:
                    failures += 1
                    detail = str(exc)
                    for secret in transport.secrets:
                        if secret:
                            detail = detail.replace(secret, "[REDACTED]")
                    store.set("connection_error", detail[:500])
                    store.set("connection", "disconnected")
                    await asyncio.sleep(min(60, 2 ** min(failures, 6)))
                except Exception:
                    failures += 1
                    store.set("connection", "disconnected")
                    await asyncio.sleep(min(60, 2 ** min(failures, 6)))

        async def dispatch() -> None:
            while True:
                try:
                    handled = await process_message(
                        cwd=cwd, transport=transport, store=store, receiver=receiver
                    )
                except Exception:
                    handled = False
                if not handled:
                    await asyncio.sleep(0.2)

        async def deliver() -> None:
            while True:
                handled = await deliver_message(transport=transport, store=store)
                # Conservative per-bot pacing also respects Slack per-channel
                # posting guidance; platform 429 values remain authoritative.
                await asyncio.sleep(1 if handled else 0.2)

        store.set("connection", "connected")
        store.set("connection_error", "")
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(receive())
            tasks.create_task(dispatch())
            tasks.create_task(deliver())
    finally:
        lock.release()
        store.close()
        await transport.close()


def build_transport(name: str, config: ChannelConfig) -> Transport:
    transports = {
        **TRANSPORTS,
        "signal": SignalTransport,
        "email": EmailTransport,
        "matrix": MatrixTransport,
        "google_chat": GoogleChatTransport,
        "feishu": FeishuTransport,
        "lark": LarkTransport,
        "teams": TeamsTransport,
        "dingtalk": DingTalkTransport,
        "mattermost": MattermostTransport,
        "ntfy": NtfyTransport,
        "irc": IRCTransport,
        "line": LINETransport,
        "sms": SMSTransport,
        "wecom": WeComBotTransport if config.wecom_mode == "bot" else WeComTransport,
        "simplex": SimpleXTransport,
        "buzz": BuzzTransport,
        "raft": RaftTransport,
        "photon": PhotonTransport,
        "yuanbao": YuanbaoTransport,
        "weixin": WeixinTransport,
        "qqbot": QQBotTransport,
        "whatsapp_cloud": WhatsAppCloudTransport,
        "bluebubbles": BlueBubblesTransport,
        "msgraph_webhook": MSGraphWebhookTransport,
    }
    if name not in transports:
        raise ChannelError("Supported channels: " + ", ".join(sorted(transports)))
    if not config.allowed_users:
        raise ChannelError("Configure allowed_users before starting a channel")
    token = os.environ.get(config.token_env, "").strip()
    app_token = os.environ.get(config.app_token_env, "").strip() if config.app_token_env else ""
    if name in {"weixin", "photon"} and not token and config.account_file:
        from harness.core.paths import read_regular_file

        token = read_regular_file(
            Path(config.account_file).expanduser(), max_bytes=16 * 1024
        ).decode("utf-8")
        if name == "photon":
            account = json.loads(token)
            if not isinstance(account, dict) or account.get("project_id") != config.app_id:
                raise ChannelError("Photon credential file belongs to a different project")
            token = account.get("project_secret", "")
            if not isinstance(token, str):
                raise ChannelError("Photon credential file has an invalid project secret")
    if not token and name != "simplex":
        raise ChannelError("The channel token environment variable is missing or empty")
    return transports[name](config=config, token=token, app_token=app_token)
