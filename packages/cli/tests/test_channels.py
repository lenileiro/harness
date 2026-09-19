from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner
from websockets.asyncio.client import connect
from websockets.asyncio.server import serve

from harness.cli.channel_commands import channel_app
from harness.cli.channels.runtime import build_transport, deliver_message, process_message
from harness.cli.channels.transports import (
    ChannelError,
    DiscordTransport,
    SlackTransport,
    TelegramTransport,
)
from harness.cli.gateway_hooks import ChannelNotificationHook, WhatsAppNotificationHook
from harness.core.gateway_channels import (
    ChannelConfig,
    ChannelMessage,
    ChannelStore,
    split_channel_text,
)
from harness.core.scheduler.models import SchedulerJob, SchedulerRunRecord, ScheduleSpec


def telegram_update(update_id=1, *, user=123, chat=123, group=False, text="hello", topic=None):
    message = {
        "message_id": update_id,
        "from": {"id": user},
        "chat": {"id": chat, "type": "supergroup" if group else "private"},
        "text": text,
    }
    if topic:
        message["message_thread_id"] = topic
    return {"update_id": update_id, "message": message}


def discord_message(
    *, id="message-1", user="123", channel="456", guild="", text="hello", mentions=None
):
    return {
        "id": id,
        "author": {"id": user},
        "channel_id": channel,
        "guild_id": guild,
        "content": text,
        "mentions": mentions or [],
    }


def slack_event(*, user="U1", channel="D1", text="hello", ts="100.001", thread=None):
    event = {"type": "message", "user": user, "channel": channel, "text": text, "ts": ts}
    if thread:
        event["thread_ts"] = thread
    return {"team_id": "T1", "event_id": "Ev1", "event": event}


@pytest.mark.parametrize("limit", [2000, 4000, 4096])
def test_split_preserves_unicode_and_platform_limits(limit):
    text = "ab🦜\n" * 2500
    chunks = split_channel_text(text, limit)
    assert "".join(chunks) == text
    assert all(len(chunk.encode("utf-16-le")) // 2 <= limit for chunk in chunks)


@pytest.mark.parametrize(
    "payload",
    [
        {"token": "secret"},
        {"allowed_users": "123"},
        {"allow_groups": "false"},
        {"operator_users": [1]},
    ],
)
def test_channel_config_rejects_ambiguous_security_settings(payload):
    with pytest.raises(ValueError):
        ChannelConfig.from_dict(payload)


def test_telegram_authenticated_poll_cursor_and_delivery(tmp_path, caplog):
    requests = []
    updates = [telegram_update(), telegram_update()]
    config = ChannelConfig(token_env="TOKEN", allowed_users=["123"])

    def request(request):
        body = json.loads(request.content)
        requests.append((request.url.path.split("/")[-1], body))
        method = requests[-1][0]
        result: Any = (
            {"id": 777, "username": "HarnessBot"}
            if method == "getMe"
            else {"url": ""}
            if method == "getWebhookInfo"
            else updates
            if method == "getUpdates"
            else {"message_id": 99}
        )
        return httpx.Response(200, json={"ok": True, "result": result})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
            transport = TelegramTransport(
                config=config, token="private-telegram-token", client=client
            )
            store = ChannelStore(cwd=tmp_path, transport="telegram")
            try:
                await transport.authenticate()
                store.bind_identity(transport.identity)
                await transport.poll_once(store)
                assert len(store.status()["inbox"]) == 1
                assert store.get("offset") == 2
                store.close()
                store = ChannelStore(cwd=tmp_path, transport="telegram")
                await transport.poll_once(store)
                assert requests[-1][1]["offset"] == 2
                assert len(store.status()["inbox"]) == 1
                await transport.send("-100:42", "plain <text>", "outbound")
                assert requests[-1][1] == {
                    "chat_id": "-100",
                    "message_thread_id": 42,
                    "text": "plain <text>",
                    "link_preview_options": {"is_disabled": True},
                }
            finally:
                store.close()
                await transport.close()

    with caplog.at_level(logging.INFO, logger="httpx"):
        asyncio.run(run())
    assert "private-telegram-token" not in caplog.text


def test_telegram_group_rules_and_bot_identity(tmp_path):
    transport = TelegramTransport(
        config=ChannelConfig(allowed_users=["123"], allowed_channels=["-100"], allow_groups=True),
        token="fake",
    )
    transport.username, transport.bot_id = "HarnessBot", "777"
    store = ChannelStore(cwd=tmp_path, transport="telegram")
    try:
        for update in [
            telegram_update(1, chat=-100, group=True),
            telegram_update(2, user=999, chat=-100, group=True, text="@HarnessBot hello"),
            telegram_update(3, chat=-200, group=True, text="@HarnessBot hello"),
            telegram_update(
                4, chat=-100, group=True, text="@HarnessBot /approve appr_123", topic=42
            ),
        ]:
            transport.accept(transport.parse(update), store)
        message = store.claim_message()
        assert message and message.id == "4" and message.thread_id == "-100:42"
        assert message.text == "/approve appr_123"
        assert store.claim_message() is None
    finally:
        store.close()
        asyncio.run(transport.close())


def test_discord_real_local_gateway_heartbeat_and_resume(tmp_path):
    frames = []
    config = ChannelConfig(allowed_users=["123"])

    async def run():
        store = ChannelStore(cwd=tmp_path, transport="discord")

        async def server(socket):
            await socket.send(json.dumps({"op": 10, "d": {"heartbeat_interval": 15}}))
            frames.append(json.loads(await socket.recv()))
            await socket.send(
                json.dumps(
                    {
                        "op": 0,
                        "t": "READY",
                        "s": 1,
                        "d": {
                            "session_id": "session-1",
                            "resume_gateway_url": "wss://gateway.discord.gg",
                        },
                    }
                )
            )
            for seq in (2, 3):
                await socket.send(
                    json.dumps({"op": 0, "t": "MESSAGE_CREATE", "s": seq, "d": discord_message()})
                )
            frame = json.loads(await socket.recv())
            frames.append(frame)
            await socket.send(json.dumps({"op": 11}))
            await socket.close()

        def request(request):
            return httpx.Response(
                200,
                json={"id": "bot"}
                if request.url.path.endswith("@me")
                else {"url": "wss://gateway.discord.gg"},
            )

        async with serve(server, "127.0.0.1", 0) as local:
            address = f"ws://127.0.0.1:{local.sockets[0].getsockname()[1]}"

            def socket_connect(url, **kwargs):
                assert url.startswith("wss://gateway.discord.gg")
                return connect(address, **kwargs)

            async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
                transport = DiscordTransport(
                    config=config, token="fake-token", client=client, socket_connect=socket_connect
                )
                await transport.authenticate()
                await transport.receive(store)
                assert len(store.status()["inbox"]) == 1
                assert store.get("resume")["sequence"] == 3
                assert frames[0]["op"] == 2 and frames[1]["op"] == 1
                assert frames[1]["d"] == 3
                await transport.receive(store)
                assert frames[2]["op"] == 6
                assert frames[2]["d"]["session_id"] == "session-1"
                assert len(store.status()["inbox"]) == 1
                await transport.close()
        store.close()

    asyncio.run(run())


def test_discord_mentions_and_no_unintended_pings():
    sent = []

    def request(request):
        sent.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bot fake"
        return httpx.Response(200, json={"id": "outbound"})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
            transport = DiscordTransport(config=ChannelConfig(), token="fake", client=client)
            transport.bot_id = "777"
            message = transport.parse(
                discord_message(guild="g", text="<@777> approve x", mentions=[{"id": "777"}])
            )
            assert message and message.mentioned and message.group and message.text == "approve x"
            await transport.send("456", "@everyone private", "x" * 64)
            assert sent[0]["allowed_mentions"] == {"parse": []}
            assert sent[0]["enforce_nonce"] and len(sent[0]["nonce"]) <= 25
            await transport.close()

    asyncio.run(run())


def test_slack_real_socket_ack_after_persistence_and_dedup(tmp_path):
    acknowledgments = []
    calls = []

    async def run():
        store = ChannelStore(cwd=tmp_path, transport="slack")

        async def server(socket):
            await socket.send(json.dumps({"type": "hello"}))
            for envelope in ("one", "two"):
                await socket.send(
                    json.dumps(
                        {"type": "events_api", "envelope_id": envelope, "payload": slack_event()}
                    )
                )
                ack = json.loads(await socket.recv())
                assert len(store.status()["inbox"]) == 1
                acknowledgments.append(ack)
            await socket.send(json.dumps({"type": "disconnect", "reason": "refresh_requested"}))

        def request(request):
            calls.append(
                (request.url.path, request.headers["authorization"], json.loads(request.content))
            )
            if request.url.path.endswith("auth.test"):
                return httpx.Response(200, json={"ok": True, "team_id": "T1", "user_id": "UBOT"})
            if request.url.path.endswith("apps.connections.open"):
                return httpx.Response(
                    200, json={"ok": True, "url": "wss://wss.slack.com/link?ticket=private"}
                )
            return httpx.Response(200, json={"ok": True, "ts": "123"})

        async with serve(server, "127.0.0.1", 0) as local:

            def socket_connect(url, **kwargs):
                return connect(f"ws://127.0.0.1:{local.sockets[0].getsockname()[1]}", **kwargs)

            async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
                transport = SlackTransport(
                    config=ChannelConfig(allowed_users=["T1:U1"]),
                    token="bot-token",
                    app_token="app-token",
                    client=client,
                    socket_connect=socket_connect,
                )
                await transport.authenticate()
                await transport.receive(store)
                assert acknowledgments == [{"envelope_id": "one"}, {"envelope_id": "two"}]
                message = store.claim_message()
                assert message and message.user_id == "T1:U1" and message.thread_id == "T1:D1:"
                await transport.send("T1:C1:123.456", "literal <!channel>", "id")
                assert calls[-1][2]["thread_ts"] == "123.456" and calls[-1][2]["mrkdwn"] is False
                assert calls[1][1] == "Bearer app-token"
                with pytest.raises(ChannelError):
                    await transport.send("T2:C1:", "private", "id")
                await transport.close()
        store.close()

    asyncio.run(run())


def test_slack_group_thread_scope_and_foreign_workspace():
    transport = SlackTransport(config=ChannelConfig(), token="fake", app_token="fake-app")
    transport.team_id, transport.bot_id = "T1", "UBOT"
    message = transport.parse(slack_event(channel="C1", text="<@UBOT> hello", thread="12.34"))
    assert message and message.group and message.mentioned
    assert message.thread_id == "T1:C1:12.34" and message.channel_id == "T1:C1"
    assert transport.parse({**slack_event(), "team_id": "T2"}) is None
    asyncio.run(transport.close())


def test_durable_dispatch_delivery_restart_and_rate_limit(tmp_path):
    turns, sends = [], []

    async def receiver(**kwargs):
        turns.append(kwargs)
        return {"reply": {"text": "x" * 5000}}

    def request(request):
        sends.append(json.loads(request.content))
        if len(sends) == 1:
            return httpx.Response(429, headers={"Retry-After": "3"}, json={"ok": False})
        return httpx.Response(200, json={"ok": True, "result": {"message_id": len(sends)}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(request)) as client:
            transport = TelegramTransport(
                config=ChannelConfig(allowed_users=["123"]), token="fake", client=client
            )
            store = ChannelStore(cwd=tmp_path, transport="telegram")
            message = ChannelMessage("incoming", "123", "123", "/approve appr_1", "123")
            assert store.ingest(message)
            assert await process_message(
                cwd=tmp_path, transport=transport, store=store, receiver=receiver
            )
            assert turns[0]["message"] == "approve appr_1" and turns[0]["user_id"] == "123"
            assert await deliver_message(transport=transport, store=store)
            assert store.status()["outbox"][-1]["status"] == "pending"
            assert not await deliver_message(transport=transport, store=store)
            with store.db:
                store.db.execute("UPDATE outbox SET next_attempt=0")
            store.set("rate_limit_until", 0)
            store.close()
            store = ChannelStore(cwd=tmp_path, transport="telegram")
            store.recover()
            assert not store.ingest(message)
            assert not await process_message(
                cwd=tmp_path, transport=transport, store=store, receiver=receiver
            )
            assert await deliver_message(transport=transport, store=store)
            assert await deliver_message(transport=transport, store=store)
            assert len(turns) == 1
            assert "".join(item["text"] for item in sends[1:]) == "x" * 5000
            store.close()
            await transport.close()

    asyncio.run(run())


def test_interrupted_claims_require_explicit_retry(tmp_path):
    store = ChannelStore(cwd=tmp_path, transport="telegram")
    original = ChannelMessage("incoming", "123", "123", "private", "123")
    store.ingest(original)
    assert store.claim_message() == original
    store.queue(source="test", user_id="123", thread_id="123", text="private", limit=4096)
    delivery = store.claim_delivery()
    assert delivery
    store.close()
    store = ChannelStore(cwd=tmp_path, transport="telegram")
    store.recover()
    assert store.claim_message() is None and store.claim_delivery() is None
    assert store.retry("incoming", inbound=True) and store.retry(delivery["id"])
    assert store.claim_message() == original and store.claim_delivery()
    store.close()


def test_channel_private_control_permissions_and_allowlist_recheck(tmp_path):
    turns = []

    async def receiver(**kwargs):
        turns.append(kwargs)
        return {"reply": {"text": "response"}}

    async def run():
        transport = TelegramTransport(config=ChannelConfig(allowed_users=["123"]), token="fake")
        store = ChannelStore(cwd=tmp_path, transport="telegram")
        store.ingest(ChannelMessage("control", "123", "123", "workflow resume existing", "123"))
        await process_message(cwd=tmp_path, transport=transport, store=store, receiver=receiver)
        assert not turns
        delivery = store.claim_delivery()
        assert delivery and "trusted operator" in delivery["text"]
        store.ingest(ChannelMessage("revoked", "999", "999", "hello", "999"))
        await process_message(cwd=tmp_path, transport=transport, store=store, receiver=receiver)
        assert not turns
        transport.config = ChannelConfig(allowed_users=["123"], operator_users=["123"])
        store.ingest(ChannelMessage("operator", "123", "123", "workflow resume existing", "123"))
        await process_message(cwd=tmp_path, transport=transport, store=store, receiver=receiver)
        assert len(turns) == 1
        store.close()
        await transport.close()

    asyncio.run(run())


def test_channel_store_rejects_different_bot_identity(tmp_path):
    store = ChannelStore(cwd=tmp_path, transport="slack")
    store.bind_identity("T1:B1")
    with pytest.raises(ValueError):
        store.bind_identity("T1:B2")
    store.close()


def test_cli_requires_credentials_and_explicit_allowed_users(monkeypatch):
    runner = CliRunner()
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    result = runner.invoke(channel_app, ["run", "telegram", "--allow-user", "123"])
    assert result.exit_code == 1 and "environment variable" in result.output
    with pytest.raises(ChannelError, match="allowed_users"):
        build_transport("telegram", ChannelConfig(token_env="TOKEN"))


def test_scheduler_hook_hands_off_original_target_once_and_skips_whatsapp(tmp_path, monkeypatch):
    def unexpected(**kwargs):
        raise AssertionError("Must not send Telegram reminders via WhatsApp")

    monkeypatch.setattr("harness.cli.gateway_hooks.send_whatsapp_text_message", unexpected)
    job = SchedulerJob(
        id="job",
        kind="reminder.once",
        cwd=str(tmp_path),
        status="completed",
        schedule=ScheduleSpec("at", "now"),
        next_run_at="",
        payload={
            "notify_transport": "telegram",
            "notify_to": "123",
            "notify_chat_id": "-100:42",
            "text": "private reminder",
        },
    )
    record = SchedulerRunRecord(
        id="run",
        job_id=job.id,
        kind=job.kind,
        cwd=str(tmp_path),
        trigger="test",
        status="completed",
        result_status="completed",
        result_stop_reason="done",
        started_at="",
        finished_at="",
        record_dir="",
        summary="",
    )
    for _ in range(2):
        ChannelNotificationHook().on_job_completed(
            cwd=tmp_path, job=job, trigger="test", record=record
        )
        WhatsAppNotificationHook().on_job_completed(
            cwd=tmp_path, job=job, trigger="test", record=record
        )
    store = ChannelStore(cwd=tmp_path, transport="telegram")
    delivery = store.claim_delivery()
    assert delivery and delivery["thread_id"] == "-100:42" and delivery["user_id"] == "123"
    assert delivery["text"] == "Reminder: private reminder"
    assert store.claim_delivery() is None
    store.close()
