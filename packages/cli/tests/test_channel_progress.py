"""Real Agent streaming and native receipt contracts, without external sends."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from harness.cli import __main__ as cli_main
from harness.cli import gateway_runtime, runtime_helpers
from harness.cli.channels import progress
from harness.cli.channels.feishu import FeishuTransport, LarkTransport
from harness.cli.channels.google_chat import GoogleChatTransport
from harness.cli.channels.progress_transport import edit_preview, send_preview, send_typing
from harness.cli.channels.runtime import deliver_message, process_message
from harness.cli.channels.transports import DiscordTransport, SlackTransport, TelegramTransport
from harness.cli.config import HarnessConfig
from harness.core import (
    Agent,
    Capabilities,
    Done,
    Event,
    FailoverPolicy,
    Message,
    StepStarted,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolRegistry,
)
from harness.core import channel_progress as ledger
from harness.core.gateway_channels import ChannelConfig, ChannelMessage, ChannelStore


@pytest.fixture
async def telegram(tmp_path, monkeypatch):
    requests: list[httpx.Request] = []
    controls = {"fail_create": False, "fail_edit": False}

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("sendMessage"):
            if controls["fail_create"]:
                controls["fail_create"] = False
                raise httpx.ReadTimeout("fixture")
            return httpx.Response(
                200, json={"ok": True, "result": {"chat": {"id": 7}, "message_id": 99}}
            )
        if request.url.path.endswith("editMessageText") and controls["fail_edit"]:
            controls["fail_edit"] = False
            raise httpx.ReadTimeout("fixture")
        return httpx.Response(200, json={"ok": True, "result": True})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport = TelegramTransport(
        config=ChannelConfig(allowed_users=["u"], allow_groups=True),
        token="fixture-token",
        client=client,
    )
    store = ChannelStore(cwd=tmp_path, transport="telegram")
    message = ChannelMessage(
        id="input",
        user_id="u",
        thread_id="7:11",
        channel_id="7",
        text="task",
        group=True,
        mentioned=True,
    )
    store.ingest(message)
    monkeypatch.setattr(progress.NativeProgress, "initial_delay", 0.001)
    monkeypatch.setattr(progress.NativeProgress, "interval", 0.01)
    try:
        yield transport, store, message, requests, controls
    finally:
        store.close()
        await transport.close()
        await client.aclose()


async def test_actual_gateway_agent_stream_uses_owned_preview_and_outbox_final_edit(
    telegram, tmp_path, monkeypatch
):
    transport, store, _, requests, _ = telegram

    class Adapter:
        name = "fake"

        async def capabilities(self):
            return Capabilities(streaming=True, tool_use=True)

        async def cancel(self, session_id):
            pass

        async def stream(self, **kwargs) -> AsyncIterator[Event]:
            # Both a private reasoning tag and an API token split across deltas
            # must stay out of all native progress requests.
            yield TextDelta(text="<think>private reasoning</think> sk-")
            await asyncio.sleep(0.035)
            yield TextDelta(text="ant-api03-" + "s" * 40)
            await asyncio.sleep(0.025)
            yield Done(final_message=Message(role="assistant", content="Completed safely."))

    def build(**kwargs):
        return Agent(
            adapters={"fake": Adapter()},
            tools=ToolRegistry(),
            storage=kwargs["storage"],
            failover=FailoverPolicy(chain=["fake"]),
            default_model="model",
            default_cwd=str(tmp_path),
        )

    monkeypatch.setattr(cli_main, "_build_agent", build)
    monkeypatch.setattr(runtime_helpers, "build_verifier", lambda *a, **k: None)
    monkeypatch.setattr(runtime_helpers, "build_critic", lambda *a, **k: None)

    async def receiver(**kwargs):
        text = await gateway_runtime._run_gateway_chat_turn(
            cwd=tmp_path,
            prompt="say hello",
            chain=["fake"],
            model="model",
            session_id="session",
            max_steps=4,
            config=HarnessConfig(),
            system_prompt="Reply",
            transport="telegram",
            user_id="u",
        )
        return {"reply": {"text": text}}

    assert await process_message(cwd=tmp_path, transport=transport, store=store, receiver=receiver)
    row = ledger.get(store, "reply:input")
    assert row is not None and row["native_id"] == "99" and row["status"] == "ready"
    assert sum(request.url.path.endswith("sendMessage") for request in requests) == 1
    assert any(request.url.path.endswith("sendChatAction") for request in requests)
    assert all(
        "private reasoning" not in request.content.decode()
        and "sk-" not in request.content.decode()
        for request in requests
    )
    assert await deliver_message(transport=transport, store=store)
    assert not await deliver_message(transport=transport, store=store)
    assert sum(request.url.path.endswith("sendMessage") for request in requests) == 1
    final = json.loads(requests[-1].content)
    assert (
        final["message_id"] == 99
        and final["chat_id"] == "7"
        and final["text"] == "Completed safely."
    )
    assert store.db.execute("SELECT status FROM outbox").fetchone()[0] == "sent"


async def fake_turn(**kwargs):
    progress.observe_gateway_event(StepStarted(step=0, description="private plan text"))
    await asyncio.sleep(0.025)
    progress.observe_gateway_event(
        ToolCallEvent(call=ToolCall(id="c", name="tool", arguments={"secret": "private arguments"}))
    )
    await asyncio.sleep(0.025)
    progress.observe_gateway_event(
        Done(final_message=Message(role="assistant", content="Finished."))
    )
    return {"reply": {"text": "Finished."}}


async def test_preview_failure_does_not_block_agent_and_final_edit_retries_same_receipt(
    telegram, tmp_path
):
    transport, store, _, requests, controls = telegram
    assert await process_message(cwd=tmp_path, transport=transport, store=store, receiver=fake_turn)
    controls["fail_edit"] = True
    assert await deliver_message(transport=transport, store=store)
    assert store.db.execute("SELECT status FROM outbox").fetchone()[0] == "pending"
    with store.db:
        store.db.execute("UPDATE outbox SET next_attempt=0")
    assert await deliver_message(transport=transport, store=store)
    assert sum(request.url.path.endswith("sendMessage") for request in requests) == 1
    assert store.db.execute("SELECT status FROM outbox").fetchone()[0] == "sent"
    assert "private arguments" not in b"".join(r.content for r in requests).decode()


async def test_unknown_preview_creation_is_not_retried_and_final_still_delivered(
    telegram, tmp_path
):
    transport, store, _, requests, controls = telegram
    controls["fail_create"] = True
    assert await process_message(cwd=tmp_path, transport=transport, store=store, receiver=fake_turn)
    row = ledger.get(store, "reply:input")
    assert row is not None and row["status"] == "uncertain" and not row["native_id"]
    assert sum(r.url.path.endswith("sendMessage") for r in requests) == 1
    assert await deliver_message(transport=transport, store=store)
    assert sum(r.url.path.endswith("sendMessage") for r in requests) == 2
    assert json.loads(requests[-1].content)["text"] == "Finished."


async def test_cancelled_dispatch_stops_typing_and_does_not_complete_outbox(telegram, tmp_path):
    transport, store, _, requests, _ = telegram
    started = asyncio.Event()

    async def receiver(**kwargs):
        progress.observe_gateway_event(StepStarted(step=0))
        await asyncio.sleep(0.03)
        started.set()
        await asyncio.Event().wait()
        return {}

    task = asyncio.create_task(
        process_message(cwd=tmp_path, transport=transport, store=store, receiver=receiver)
    )
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    before = len(requests)
    await asyncio.sleep(0.03)
    assert len(requests) == before
    assert "interrupted" in json.loads(requests[-1].content)["text"]
    assert store.db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0
    assert store.db.execute("SELECT status FROM inbox").fetchone()[0] == "uncertain"


async def test_allowlist_change_blocks_further_edits_and_final_delivery(telegram, tmp_path):
    transport, store, _, requests, _ = telegram
    assert await process_message(cwd=tmp_path, transport=transport, store=store, receiver=fake_turn)
    transport.config.allowed_users.clear()
    before = len(requests)
    assert await deliver_message(transport=transport, store=store)
    assert len(requests) == before
    assert store.db.execute("SELECT status FROM outbox").fetchone()[0] == "failed"


async def test_cancel_during_inflight_preview_create_keeps_receipt_for_cleanup(
    telegram, tmp_path, monkeypatch
):
    transport, store, _, requests, _ = telegram
    creating, release = asyncio.Event(), asyncio.Event()
    original = send_preview

    async def blocked(*args):
        creating.set()
        await release.wait()
        return await original(*args)

    async def receiver(**kwargs):
        progress.observe_gateway_event(StepStarted(step=0))
        await creating.wait()
        return {"reply": {"text": "Ready"}}

    monkeypatch.setattr(progress, "send_preview", blocked)
    task = asyncio.create_task(
        process_message(cwd=tmp_path, transport=transport, store=store, receiver=receiver)
    )
    await creating.wait()
    await asyncio.sleep(0.01)  # The context is exiting while create awaits its receipt.
    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    row = ledger.get(store, "reply:input")
    assert row is not None and row["status"] == "interrupted" and row["native_id"] == "99"
    assert store.db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0] == 0
    assert sum(r.url.path.endswith("sendMessage") for r in requests) == 1
    assert "interrupted" in json.loads(requests[-1].content)["text"]


async def test_restart_recovers_preview_and_final_uses_same_saved_message(telegram, tmp_path):
    transport, store, message, requests, _ = telegram
    assert await process_message(cwd=tmp_path, transport=transport, store=store, receiver=fake_turn)
    restarted = ChannelStore(cwd=tmp_path, transport="telegram")
    try:
        await progress.recover_progress(transport, restarted)
        assert await deliver_message(transport=transport, store=restarted)
        assert sum(r.url.path.endswith("sendMessage") for r in requests) == 1
        # A crash after status creation but before queuing any final answer is
        # visibly interrupted without replaying model/tool work.
        second = ChannelMessage(
            id="other",
            user_id=message.user_id,
            channel_id=message.channel_id,
            thread_id=message.thread_id,
            text="",
            group=True,
            mentioned=True,
        )
        ledger.reserve(restarted, second)
        ledger.update(restarted, "reply:other", "ready", "100")
        await progress.recover_progress(transport, restarted)
        assert json.loads(requests[-1].content)["message_id"] == 100
        assert "interrupted" in json.loads(requests[-1].content)["text"]
        row = ledger.get(restarted, "reply:other")
        assert row is not None and row["status"] == "interrupted"
    finally:
        restarted.close()


async def test_paused_approval_stops_background_progress_until_explicit_resume(telegram, tmp_path):
    transport, store, _, requests, _ = telegram

    async def receiver(**kwargs):
        progress.observe_gateway_event(StepStarted(step=0))
        await asyncio.sleep(0.03)
        progress.observe_gateway_event(Done(structured_result={"status": "waiting_for_approval"}))
        before = len(requests)
        await asyncio.sleep(0.03)
        assert len(requests) == before
        return {"reply": {"text": "Approval required."}}

    assert await process_message(cwd=tmp_path, transport=transport, store=store, receiver=receiver)
    assert await deliver_message(transport=transport, store=store)
    assert json.loads(requests[-1].content)["text"] == "Approval required."


@pytest.mark.parametrize(
    "kind,thread,receipt",
    [
        (TelegramTransport, "7:11", "99"),
        (DiscordTransport, "123", "456"),
        (SlackTransport, "TEAM:C1:11.22", "33.44"),
        (GoogleChatTransport, '["spaces/s","spaces/s/threads/t"]', "spaces/s/messages/m"),
        (FeishuTransport, '["oc_chat","om_root"]', "om_result"),
        (LarkTransport, '["oc_chat","om_root"]', "om_result"),
    ],
)
async def test_native_progress_transport_routes_and_edit_contracts(kind, thread, receipt):
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/typing"):
            return httpx.Response(204)
        if kind is TelegramTransport:
            return httpx.Response(
                200, json={"ok": True, "result": {"chat": {"id": 7}, "message_id": 99}}
            )
        if kind is DiscordTransport:
            return httpx.Response(200, json={"id": "456", "channel_id": "123"})
        if kind is SlackTransport:
            return httpx.Response(200, json={"ok": True, "ts": "33.44", "channel": "C1"})
        if kind is GoogleChatTransport:
            return httpx.Response(200, json={"name": receipt})
        return httpx.Response(200, json={"code": 0, "data": {"message_id": receipt}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    transport: Any = kind(config=ChannelConfig(allowed_users=["u"]), token="fixture", client=client)
    transport.team_id = "TEAM"

    async def access():
        return "access"

    transport._access = access
    try:
        assert await send_preview(transport, thread, "Working…", "delivery") == receipt
        await edit_preview(transport, thread, receipt, "Final")
        await send_typing(transport, thread)
        created, edited = requests[:2]
        body = json.loads(created.content)
        if kind is TelegramTransport:
            assert body["message_thread_id"] == 11
            assert requests[-1].url.path.endswith("sendChatAction")
        elif kind is DiscordTransport:
            assert edited.method == "PATCH" and edited.url.path.endswith(
                "/channels/123/messages/456"
            )
            assert json.loads(edited.content)["allowed_mentions"] == {"parse": []}
            assert requests[-1].url.path.endswith("/channels/123/typing")
        elif kind is SlackTransport:
            assert body["thread_ts"] == "11.22" and json.loads(edited.content)["ts"] == "33.44"
        elif kind is GoogleChatTransport:
            assert body["thread"] == {"name": "spaces/s/threads/t"}
            assert created.url.params["messageReplyOption"] == "REPLY_MESSAGE_OR_FAIL"
            assert edited.method == "PATCH" and edited.url.params["updateMask"] == "text"
        else:
            assert created.url.path.endswith("/om_root/reply") and body["reply_in_thread"] is True
            assert edited.method == "PUT" and edited.url.path.endswith("/om_result")
            assert json.loads(edited.content) == {
                "msg_type": "text",
                "content": '{"text": "Final"}',
            }
    finally:
        await transport.close()
        await client.aclose()
