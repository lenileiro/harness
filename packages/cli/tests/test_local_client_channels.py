import asyncio
import json
import sys

import httpx
import pytest
from aiohttp import web
from websockets.asyncio.server import serve

from harness.cli.channels import raft
from harness.cli.channels.buzz import BuzzTransport
from harness.cli.channels.process import client_environment, run_client
from harness.cli.channels.simplex import SimpleXTransport
from harness.cli.channels.transports import ChannelError
from harness.cli.gateway_runtime import GatewayApprovalPolicy
from harness.core import (
    Agent,
    Capabilities,
    Done,
    FailoverPolicy,
    InboxApprovalHandler,
    Message,
    RunRequest,
    ToolCall,
    ToolRegistry,
)
from harness.core.gateway_channels import ChannelConfig, ChannelStore
from harness.storage.sqlite import SQLiteStorage


async def test_simplex_actual_local_socket_identity_numeric_thread_ack_and_replay(tmp_path):
    store = ChannelStore(cwd=tmp_path, transport="simplex")
    frames = []
    received_send = asyncio.Event()
    event = {
        "resp": {
            "type": "newChatItems",
            "user": {"userId": 1},
            "chatItems": [
                {
                    "chatInfo": {"type": "direct", "contact": {"contactId": 23}},
                    "chatItem": {
                        "meta": {"itemId": 9},
                        "chatDir": {"type": "directRcv"},
                        "content": {
                            "type": "rcvMsgContent",
                            "msgContent": {"type": "text", "text": "approve pending"},
                        },
                    },
                }
            ],
        }
    }

    async def server(socket):
        frame = json.loads(await socket.recv())
        assert frame["cmd"] == "/user"
        await socket.send(json.dumps(event))
        await socket.send(
            json.dumps(
                {"corrId": frame["corrId"], "resp": {"type": "activeUser", "user": {"userId": 1}}}
            )
        )
        try:
            async for raw in socket:
                frame = json.loads(raw)
                frames.append(frame)
                if frame["cmd"] == "/user":
                    response = {"type": "activeUser", "user": {"userId": 1}}
                else:
                    assert frame["cmd"].startswith("/_send @23 json ")
                    response = {"type": "newChatItems", "chatItems": []}
                    received_send.set()
                await socket.send(json.dumps({"corrId": frame["corrId"], "resp": response}))
        except Exception:
            return

    async with serve(server, "127.0.0.1", 0) as local:
        url = f"ws://127.0.0.1:{local.sockets[0].getsockname()[1]}"
        transport = SimpleXTransport(
            config=ChannelConfig(homeserver=url, username="1", allowed_users=["contact:23"]),
            token="",
        )
        await transport.authenticate()
        task = asyncio.create_task(transport.receive(store))
        for _ in range(100):
            if transport.can_send():
                break
            await asyncio.sleep(0.01)
        transport.ingest(event, store)
        message = store.claim_message()
        assert (
            message
            and message.thread_id == '["1","direct","23"]'
            and message.user_id == "contact:23"
        )
        assert store.claim_message() is None
        await transport.send(message.thread_id, "first\n/accept 99", "delivery")
        assert received_send.is_set()
        sent = json.loads(frames[-1]["cmd"].split(" json ", 1)[1])
        assert sent[0]["msgContent"]["text"] == "first\n/accept 99"
        with pytest.raises(ChannelError, match="profile"):
            await transport.send('["2","direct","23"]', "private", "id")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await transport.close()
    store.close()


async def test_simplex_refuses_public_daemon_and_changed_profile(tmp_path):
    transport = SimpleXTransport(
        config=ChannelConfig(homeserver="ws://attacker.example", username="1"), token=""
    )
    with pytest.raises(ChannelError, match="loopback"):
        await transport.authenticate()
    await transport.close()

    async def server(socket):
        frame = json.loads(await socket.recv())
        await socket.send(
            json.dumps(
                {"corrId": frame["corrId"], "resp": {"type": "activeUser", "user": {"userId": 99}}}
            )
        )

    async with serve(server, "127.0.0.1", 0) as local:
        transport = SimpleXTransport(
            config=ChannelConfig(
                homeserver=f"ws://127.0.0.1:{local.sockets[0].getsockname()[1]}", username="1"
            ),
            token="",
        )
        with pytest.raises(ChannelError, match="differs"):
            await transport.authenticate()
        await transport.close()


async def test_buzz_authenticated_cli_cursor_mentions_and_receipt(tmp_path):
    calls = []
    event = {
        "id": "event",
        "pubkey": "owner",
        "created_at": 100,
        "kind": 9,
        "content": "@Harness approve pending",
        "tags": [["p", "bot"], ["e", "root", "", "root"]],
    }

    async def command(argv, **kwargs):
        calls.append((argv, kwargs))
        assert kwargs["env"] == {
            "BUZZ_RELAY_URL": "https://relay.example",
            "BUZZ_PRIVATE_KEY": "private-key",
        }
        if argv[1:] == ["users", "get"]:
            return 0, json.dumps([{"pubkey": "bot", "display_name": "Harness"}])
        if argv[1:] == ["channels", "list"]:
            return 0, json.dumps([{"id": "channel", "type": "group"}])
        if argv[1:3] == ["messages", "send"]:
            return 0, json.dumps({"accepted": True, "event_id": "sent"})
        assert "200" in argv
        return 0, json.dumps([event])

    transport = BuzzTransport(
        config=ChannelConfig(
            homeserver="https://relay.example",
            allowed_users=["owner"],
            allowed_channels=["channel"],
            allow_groups=True,
        ),
        token="private-key",
        command_runner=command,
    )
    await transport.authenticate()
    store = ChannelStore(cwd=tmp_path, transport="buzz")
    await transport.poll_once(store)
    assert store.claim_message() is None and store.get("since:channel") == 100
    event["id"] = "new"
    for _ in range(2):
        await transport.poll_once(store)
    message = store.claim_message()
    assert (
        message and message.text == "approve pending" and message.thread_id == '["channel","root"]'
    )
    assert store.claim_message() is None
    await transport.send(message.thread_id, "private result", "id")
    assert calls[-1][0][-2:] == ["--reply-to", "root"]
    assert calls[-1][1]["input_text"] == "private result" and "private-key" not in calls[-1][0]
    store.close()
    await transport.close()


async def test_bounded_client_real_subprocess_no_shell_secret_args_or_unrelated_env(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OTHER_API_KEY", "do-not-inherit")
    script = tmp_path / "client.py"
    script.write_text(
        "import json,os,sys\nprint(json.dumps({'args':sys.argv[1:], 'stdin':sys.stdin.read(),'private':os.getenv('PRIVATE'),'other':os.getenv('OTHER_API_KEY')}))"
    )
    code, output = await run_client(
        [sys.executable, str(script), "literal; exit 7"],
        env={"PRIVATE": "secret"},
        input_text="body",
    )
    assert code == 0
    assert json.loads(output) == {
        "args": ["literal; exit 7"],
        "stdin": "body",
        "private": "secret",
        "other": None,
    }
    assert "OTHER_API_KEY" not in client_environment({})
    with pytest.raises(ExceptionGroup):
        await run_client([sys.executable, "-c", "print('x'*10000)"], env={}, output_limit=100)
    with pytest.raises(TimeoutError):
        await run_client(
            [sys.executable, "-c", "import time;time.sleep(30)"], env={}, deadline_seconds=0.05
        )


def raft_config():
    return ChannelConfig(
        homeserver="https://raft.example",
        username="profile",
        app_id="agent",
        allowed_users=["raft:profile"],
        allowed_channels=["profile"],
        allowed_targets=["#project"],
        webhook_path="/wake",
    )


async def test_raft_real_local_wake_metadata_only_owner_dedup_and_local_results(
    tmp_path, monkeypatch
):
    async def identity(config):
        return {"agentId": "agent", "profileSlug": "profile"}

    monkeypatch.setattr(raft, "raft_identity", identity)
    transport = raft.RaftTransport(config=raft_config(), token="s" * 32)
    await transport.authenticate()
    store = ChannelStore(cwd=tmp_path, transport="raft")
    runner = web.AppRunner(transport.application(store), access_log=None)
    await runner.setup()
    await web.TCPSite(runner, "127.0.0.1", 0).start()
    url = f"http://127.0.0.1:{runner.addresses[0][1]}"
    event = {
        "schema": "raft-channel-wake.v1",
        "eventId": "one",
        "agentId": "agent",
        "profile": "profile",
    }
    headers = {"X-Raft-Bridge-Token": "s" * 32}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(url + "/wake", json=event)
            assert response.status_code == 401
            for invalid in (
                {**event, "content": "secret"},
                {**event, "eventId": {"text": "secret"}},
            ):
                response = await client.post(url + "/wake", json=invalid, headers=headers)
                assert response.status_code == 400
            response = await client.post(
                url + "/wake", json={**event, "profile": "other"}, headers=headers
            )
            assert response.status_code == 401
            for _ in range(2):
                response = await client.post(url + "/wake", json=event, headers=headers)
                assert (
                    response.status_code == 200 and response.json()["runtimeSession"] == "profile"
                )
            response = await client.get(url + "/activity/drain", headers=headers)
            assert response.json() == {
                "schema": "raft-activity-drain.v1",
                "events": [],
                "dropped": 0,
            }
        message = store.claim_message()
        assert message and message.user_id == "raft:profile" and store.claim_message() is None
        await transport.send("profile", "private approval", "delivery")
        assert "private approval" not in json.dumps(store.status())
        assert store.status(include_private=True)["local_responses"][0]["destination"] == "local"
    finally:
        await runner.cleanup()
        store.close()
        await transport.close()


async def test_raft_real_agent_queues_send_before_cli_and_replays_once(tmp_path, monkeypatch):
    calls = []

    async def identity(config):
        return {}

    async def command(argv, **kwargs):
        calls.append((argv, kwargs))
        return 0, "sent once"

    monkeypatch.setattr(raft, "raft_identity", identity)
    monkeypatch.setattr(raft, "run_client", command)
    config = raft_config()
    with pytest.raises(ChannelError, match="ownership"):
        raft.raft_tools(config, user_id="raft:other", cwd=tmp_path)
    registry = ToolRegistry()
    for tool in raft.raft_tools(config, user_id="raft:profile", cwd=tmp_path):
        registry.register(tool)

    class Adapter:
        name = "fake"

        async def capabilities(self):
            return Capabilities(streaming=True, tool_use=True)

        async def cancel(self, session_id):
            pass

        async def stream(self, **kwargs):
            if any(message.role == "tool" for message in kwargs["messages"]):
                yield Done(final_message=Message(role="assistant", content="Finished"))
            else:
                yield Done(
                    final_message=Message(
                        role="assistant",
                        tool_calls=[
                            ToolCall(
                                id="send",
                                name="raft_send",
                                arguments={"target": "#project:thread", "text": "Reviewed body"},
                            )
                        ],
                    )
                )

    storage = SQLiteStorage(path=tmp_path / "state.db")
    agent = Agent(
        adapters={"fake": Adapter()},
        tools=registry,
        storage=storage,
        approval_store=storage,
        approval_handler=InboxApprovalHandler(approval_store=storage),
        approval_policy=GatewayApprovalPolicy(),
        failover=FailoverPolicy(chain=["fake"], max_attempts=1),
        default_model="fake",
        default_cwd=str(tmp_path),
        pause_on_approval=True,
    )
    try:
        async for _ in agent.run(RunRequest(prompt="Reply", session_id="session", max_steps=3)):
            pass
        pending = await storage.list_approvals(status="pending", session_id="session")
        assert len(pending) == 1 and calls == []
        await storage.resolve_approval(pending[0].id, status="granted")
        async for _ in agent.run(RunRequest(prompt="Continue", session_id="session", max_steps=3)):
            pass
        assert len(calls) == 1
        assert calls[0][0] == [
            "raft",
            "--profile",
            "profile",
            "message",
            "send",
            "--target=#project:thread",
            "--json",
        ]
        assert calls[0][1]["input_text"] == "Reviewed body"
    finally:
        await storage.close()
